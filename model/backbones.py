import os
import torch


# Supported frozen backbones. Hub entry name == config "type".
#   patch_size : input H and W must be multiples of this
#   embed_dim  : per-patch feature width
#   depth      : number of transformer blocks (valid layer indices are 0..depth-1)
BACKBONES = {
    "dinov2_vits14": dict(family="dinov2", patch_size=14, embed_dim=384,  depth=12),
    "dinov2_vitb14": dict(family="dinov2", patch_size=14, embed_dim=768,  depth=12),
    "dinov2_vitl14": dict(family="dinov2", patch_size=14, embed_dim=1024, depth=24),
    "dinov2_vitl14_reg": dict(family="dinov2", patch_size=14, embed_dim=1024, depth=24),   # 4 register tokens, dropped in extract_features
    "dinov3_vits16": dict(family="dinov3", patch_size=16, embed_dim=384,  depth=12),
    "dinov3_vitb16": dict(family="dinov3", patch_size=16, embed_dim=768,  depth=12),
    "dinov3_vitl16": dict(family="dinov3", patch_size=16, embed_dim=1024, depth=24),
}

# Folder of this code (config.yaml ...). A relative path in the config (weights_path, adapted_weights_path) starts here.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(path):
    return path if (not path or os.path.isabs(path)) else os.path.join(_ROOT, path)


def get_spec(backbone_type):
    if backbone_type not in BACKBONES:
        raise ValueError(f"[Backbone] Unknown type '{backbone_type}'. "
                         f"Supported: {sorted(BACKBONES)}")
    return BACKBONES[backbone_type]


def validate_config(fe_config, image_size):
    """Fail early with a clear message instead of a shape error deep in the model."""
    spec = get_spec(fe_config["type"])
    ps = spec["patch_size"]
    H, W = image_size
    if H % ps or W % ps:
        raise ValueError(f"[Backbone] image_size {list(image_size)} must be multiples of "
                         f"{ps} for {fe_config['type']}. Try "
                         f"[{(H // ps) * ps}, {(W // ps) * ps}].")
    bad = [l for l in fe_config["layers"] if not 0 <= l < spec["depth"]]
    if bad:
        raise ValueError(f"[Backbone] layers {bad} out of range for {fe_config['type']} "
                         f"(depth={spec['depth']}, valid 0..{spec['depth'] - 1}).")
    return spec


def load_backbone(fe_config, use_adapted=True, freeze=True):
    """
    Load a DINOv2 / DINOv3 backbone from local weights (no downloads).
    DINOv2: the network code is model/dinov2/ (copied from facebookresearch/dinov2), no repo needed.
    DINOv3: built from the local repo (repo_path).

    fe_config keys:
        type                  : e.g. "dinov2_vitl14" or "dinov3_vitl16"
        repo_path             : local clone of the facebookresearch repo (hubconf.py), DINOv3 only
        weights_path          : base pretrained .pth
        adapted_weights_path  : optional domain-adapted .pth (from pretrain.py)
    """
    backbone_type = fe_config["type"]
    spec = get_spec(backbone_type)

    repo_path = fe_config.get("repo_path")
    weights_path = _abs(fe_config.get("weights_path"))
    if not weights_path or (spec["family"] != "dinov2" and not repo_path):
        raise ValueError(f"[Backbone] {backbone_type} needs 'weights_path' (and 'repo_path' for DINOv3) "
                         f"in model.feature_extractor.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[Backbone] weights not found: {weights_path}")

    if spec["family"] == "dinov2":
        from model.dinov2.hub import backbones as dinov2_hub   # model/dinov2/
        print(f"[Backbone] Loading {backbone_type}")
        model = getattr(dinov2_hub, backbone_type)(pretrained=False)
    else:
        print(f"[Backbone] Loading {backbone_type} from {repo_path}")
        model = torch.hub.load(repo_path, backbone_type, pretrained=False, source="local")

    adapted_path = _abs(fe_config.get("adapted_weights_path", ""))
    if use_adapted and adapted_path and os.path.exists(adapted_path):
        print(f"[Backbone] Loading adapted weights: {adapted_path}")
        model.load_state_dict(torch.load(adapted_path, map_location="cpu"))
    else:
        print(f"[Backbone] Loading pretrained weights: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model, spec
