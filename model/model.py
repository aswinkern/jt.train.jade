import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import backbones as bb


class MemoryBottleneck(nn.Module):
    """
    Per-layer encoder-decoder bottleneck.
    Learns to reconstruct normal patch features for one backbone layer.
    """
    def __init__(self, feature_dim, bottleneck_size):
        super(MemoryBottleneck, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_size * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(bottleneck_size * 2, bottleneck_size),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, bottleneck_size * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(bottleneck_size * 2, feature_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class JADE_v3(nn.Module):
    def __init__(self, config_dict):
        super(JADE_v3, self).__init__()

        self.config_dict = config_dict
        fe_config  = config_dict["model"]["feature_extractor"]
        mem_config = config_dict["model"]["memory"]
        self.image_size = tuple(config_dict["data"]["image_size"])

        # Frozen backbone — DINOv2 or DINOv3, chosen by model.feature_extractor.type
        spec = bb.validate_config(fe_config, self.image_size)
        self.backbone_type = fe_config["type"]
        self.feature_extractor, _ = bb.load_backbone(fe_config)

        self.patch_size      = spec["patch_size"]
        self.feature_dim     = spec["embed_dim"]
        self.extract_layers  = fe_config["layers"]
        self.num_layers      = len(self.extract_layers)
        self.bottleneck_size = mem_config["bottleneck_size"]

        # One independent memory per layer
        self.memories = nn.ModuleList([
            MemoryBottleneck(self.feature_dim, self.bottleneck_size)
            for _ in range(self.num_layers)
        ])

        # Optional per-layer feature scale (model.feature_extractor.normalize_features: true).
        # Every layer is divided by its typical size (RMS over training patches, set in train.py and
        # saved in the checkpoint) so all layers share one scale. Stays 1.0 when the option is off.
        self.normalize_features = bool(fe_config.get("normalize_features", False))
        self.register_buffer("feat_scale", torch.ones(self.num_layers))

        # How much each layer counts in the final heat map (inference only, nothing to retrain).
        # model.layer_weights in the config: one value per layer in config order, e.g. [1.0, 0.7, 0.6, 1.0] for
        # layers [5, 11, 17, 23]. Normalised to sum 1. Key missing = equal weights.
        w = config_dict["model"].get("layer_weights", [1.0] * self.num_layers)
        if len(w) != self.num_layers:
            raise ValueError(f"[JADE_v3] layer_weights {w} must have {self.num_layers} values (layers {self.extract_layers}).")
        self.layer_weights = [x / sum(w) for x in w]

        # model.featurewise_threshold (inference only): a feature whose squared difference is at or below it counts as 0
        # before the features are averaged into one error per patch. 0 / missing = off.
        self.featurewise_threshold = float(config_dict["model"].get("featurewise_threshold") or 0.0)

    def patch_error(self, feat_flat, reconstructed):
        """Error per patch: mean over the features of the squared difference (differences <= featurewise_threshold -> 0)."""
        d = (feat_flat - reconstructed) ** 2
        if self.featurewise_threshold > 0:
            d = torch.where(d > self.featurewise_threshold, d, torch.zeros_like(d))
        return d.mean(dim=1)

    # ------------------------------------------------------------------
    # Feature extraction — no hooks, pure PyTorch, ONNX-traceable
    # ------------------------------------------------------------------

    def extract_features(self, x):
        """
        Extract patch features from each backbone layer independently.
        Uses get_intermediate_layers (no hooks) — works in training and ONNX export.
        norm=False matches the raw block output the memories were trained on.
        Class / register tokens are dropped, only patch tokens are returned.
        Returns: list of (B, N, C) — one tensor per layer.
        """
        features = self.feature_extractor.get_intermediate_layers(
            x, n=self.extract_layers, return_class_token=False, norm=False
        )
        return [f / self.feat_scale[i] for i, f in enumerate(features)]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, x):
        features   = self.extract_features(x)
        total_loss = 0.0

        for i, feat in enumerate(features):
            B, N, C  = feat.shape
            feat_flat     = feat.reshape(B * N, C)
            reconstructed = self.memories[i](feat_flat)
            total_loss   += F.mse_loss(reconstructed, feat_flat)

        return total_loss / self.num_layers

    # ------------------------------------------------------------------
    # Inference — pure PyTorch, ONNX-traceable
    # ------------------------------------------------------------------

    def forward_export(self, x):
        """
        Pure PyTorch forward pass for ONNX export. Any batch size B (nothing about B is fixed inside).
        Input:  (B, 3, H, W)
        Output: (B, H, W) float32 raw error map
        """
        H_img, W_img = self.image_size
        patch_h = H_img // self.patch_size
        patch_w = W_img // self.patch_size
        N = patch_h * patch_w

        features = self.extract_features(x)
        combined_error = None
        for i, feat in enumerate(features):
            feat_flat     = feat.reshape(-1, self.feature_dim)                 # (B*N, C)
            reconstructed = self.memories[i](feat_flat)
            error         = self.patch_error(feat_flat, reconstructed)         # (B*N,)
            weighted      = self.layer_weights[i] * error.reshape(-1, N)       # (B, N)
            combined_error = weighted if combined_error is None else combined_error + weighted

        heatmap = combined_error.reshape(-1, 1, patch_h, patch_w)
        heatmap = F.interpolate(heatmap, size=(H_img, W_img),
                                mode="bilinear", align_corners=False)
        return heatmap.reshape(-1, H_img, W_img)   # (B, H, W); reshape, not squeeze: squeeze is exported as an If node that TensorRT cannot parse

    def get_error_map(self, x):
        """
        Returns raw float32 error map as numpy (B, H, W).
        """
        with torch.no_grad():
            heatmap = self.forward_export(x)
        return heatmap.detach().cpu().numpy()

    def forward(self, x):
        """Per-image 99th percentile normalization — for production inference."""
        raw = self.get_error_map(x)
        out = np.zeros_like(raw)
        for i in range(raw.shape[0]):
            p99 = np.percentile(raw[i], 99)
            if p99 > 0:
                out[i] = np.clip(raw[i] / p99, 0, 1)
        return out

    def get_anomaly_heatmap(self, x):
        return (self.forward(x) * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Checkpoint — tagged with the backbone it was trained on, because
    # memories from one backbone are meaningless on another (same width,
    # so loading would not raise an error, just give garbage heatmaps).
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            "backbone": self.backbone_type,
            "layers":   list(self.extract_layers),
            "memories": self.memories.state_dict(),
            "feat_scale": self.feat_scale.detach().cpu(),
        }, save_path)
        print(f"[JADE_v3] Saved: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "memories" in ckpt:
            if ckpt["backbone"] != self.backbone_type:
                raise ValueError(f"[JADE_v3] Checkpoint was trained with '{ckpt['backbone']}' "
                                 f"but config backbone is '{self.backbone_type}'.")
            if list(ckpt["layers"]) != list(self.extract_layers):
                raise ValueError(f"[JADE_v3] Checkpoint layers {ckpt['layers']} != "
                                 f"config layers {list(self.extract_layers)}.")
            state = ckpt["memories"]
            if "feat_scale" in ckpt:
                self.feat_scale.copy_(ckpt["feat_scale"])
        else:
            print("[JADE_v3] WARNING: legacy checkpoint without backbone tag — "
                  "assuming it matches the config backbone.")
            state = ckpt
        self.memories.load_state_dict(state)
        print(f"[JADE_v3] Loaded: {checkpoint_path}")
