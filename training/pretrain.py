import os
import sys
import copy
import yaml
import random
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import data_handler as dh
from model import backbones as bb

# Normalized black pixel value (RGB 0,0,0 after ImageNet normalization)
NORM_BLACK = [-2.1179, -2.0357, -1.8044]


# ------------------------------------------------------------------
# Pretrain augmentation — random window mask
# ------------------------------------------------------------------

def apply_pretrain_augmentation(batch, aug_config):
    """
    Apply random window mask to pretrain batch.
    Shows only a rectangular window of the chain, rest set to normalized black.
    Forces backbone to learn chain features at varied positions.
    """
    if not aug_config.get("random_window_mask", False):
        return batch

    B, C, H, W = batch.shape
    black = torch.tensor(NORM_BLACK, device=batch.device).view(3, 1, 1)
    result = batch.clone()

    for b in range(B):
        if random.random() < 0.5:
            win_w = random.randint(int(W * 0.4), int(W * 0.8))
            win_h = random.randint(int(H * 0.5), H)
            x = random.randint(0, W - win_w)
            y = random.randint(0, H - win_h)
            masked = black.expand(C, H, W).clone()
            masked[:, y:y + win_h, x:x + win_w] = batch[b, :, y:y + win_h, x:x + win_w]
            result[b] = masked

    # Random vertical shift
    if aug_config.get("random_vertical_shift", False):
        for b in range(B):
            if random.random() < 0.5:
                max_shift = int(H * 0.3)
                shift = random.randint(-max_shift, max_shift)
                if shift != 0:
                    shifted = black.expand(C, H, W).clone()
                    if shift > 0:
                        shifted[:, shift:, :] = result[b, :, :H - shift, :]
                    else:
                        shift = abs(shift)
                        shifted[:, :H - shift, :] = result[b, :, shift:, :]
                    result[b] = shifted

    # Random horizontal shift
    if aug_config.get("random_horizontal_shift", False):
        for b in range(B):
            if random.random() < 0.5:
                max_shift = int(W * 0.2)
                shift = random.randint(-max_shift, max_shift)
                if shift != 0:
                    shifted = black.expand(C, H, W).clone()
                    if shift > 0:
                        shifted[:, :, shift:] = result[b, :, :, :W - shift]
                    else:
                        shift = abs(shift)
                        shifted[:, :, :W - shift] = result[b, :, :, shift:]
                    result[b] = shifted

    return result

# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def extract_features(model, x, target_layers, no_grad=False):
    """
    Raw block outputs (norm=False) for the target layers, patch tokens only.
    get_intermediate_layers drops the class token and, for DINOv3, the register
    tokens, so this works for both families without hand-slicing tokens.
    """
    def run():
        return list(model.get_intermediate_layers(
            x, n=target_layers, return_class_token=False, norm=False))

    if no_grad:
        with torch.no_grad():
            return run()
    return run()


# ------------------------------------------------------------------
# Patch masking
# ------------------------------------------------------------------

def mask_patches(image, mask_ratio, patch_size):
    B, C, H, W = image.shape
    patch_h = H // patch_size
    patch_w = W // patch_size
    N = patch_h * patch_w
    num_masked = int(N * mask_ratio)

    masked_image = image.clone()
    mask_indices_batch = []

    for b in range(B):
        perm    = torch.randperm(N, device=image.device)
        mask_idx = perm[:num_masked]
        mask_indices_batch.append(mask_idx)

        rows = mask_idx // patch_w
        cols = mask_idx % patch_w
        for r, c in zip(rows.tolist(), cols.tolist()):
            masked_image[b, :,
                         r * patch_size:(r + 1) * patch_size,
                         c * patch_size:(c + 1) * patch_size] = 0.0

    return masked_image, mask_indices_batch


# ------------------------------------------------------------------
# Pretrain
# ------------------------------------------------------------------

def pretrain(config_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pretrain] Using device: {device}")

    pretrain_cfg  = config_dict["pretrain"]
    num_epochs    = pretrain_cfg["epochs"]
    learning_rate = pretrain_cfg["learning_rate"]
    mask_ratio    = pretrain_cfg["mask_ratio"]
    target_layers = pretrain_cfg["target_layers"]
    adapted_path  = pretrain_cfg["adapted_weights_path"]
    unfreeze_last = pretrain_cfg.get("unfreeze_last_blocks", 7)
    aug_config    = pretrain_cfg.get("augmentation", {})

    fe_config  = config_dict["model"]["feature_extractor"]
    image_size = tuple(config_dict["data"]["image_size"])
    spec       = bb.validate_config(fe_config, image_size)
    patch_size = spec["patch_size"]
    depth      = spec["depth"]

    os.makedirs(os.path.dirname(adapted_path), exist_ok=True)

    train_loader, train_dataset = dh.get_dataloader(config_dict, mode="train")

    # Always adapt from the base pretrained weights, never from a previous adaptation
    dinov2, _ = bb.load_backbone(fe_config, use_adapted=False, freeze=False)
    dinov2 = dinov2.to(device)

    # Teacher — fully frozen
    teacher = copy.deepcopy(dinov2)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Student — unfreeze the last N blocks only
    first_trainable = depth - unfreeze_last
    for param in dinov2.parameters():
        param.requires_grad = False
    for block_idx in range(first_trainable, depth):
        for param in dinov2.blocks[block_idx].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in dinov2.parameters() if p.requires_grad)
    print(f"[Pretrain] Backbone: {fe_config['type']} | patch={patch_size}")
    print(f"[Pretrain] Trainable params: {trainable:,}  (blocks {first_trainable}-{depth - 1})")
    print(f"[Pretrain] Target layers: {target_layers} | Mask ratio: {mask_ratio}")
    print(f"[Pretrain] Augmentations: {aug_config}")

    optimizer = torch.optim.Adam(
        [p for p in dinov2.parameters() if p.requires_grad],
        lr=learning_rate
    )

    log_dir = config_dict["training"]["tensorboard_dir"] + "_pretrain"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, num_epochs + 1):
        dinov2.train()
        for i in range(0, first_trainable):
            dinov2.blocks[i].eval()

        epoch_losses = []

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            B = images.shape[0]

            # Apply pretrain augmentation to the batch
            images = apply_pretrain_augmentation(images, aug_config)

            # Teacher: original augmented image → target features
            teacher_feats = extract_features(teacher, images, target_layers, no_grad=True)

            # Mask patches for student
            masked_images, mask_indices_batch = mask_patches(images, mask_ratio, patch_size)

            # Student: masked image → predicted features
            student_feats = extract_features(dinov2, masked_images, target_layers, no_grad=False)

            # Loss only at masked positions
            optimizer.zero_grad()
            total_loss = 0.0

            for t_feat, s_feat in zip(teacher_feats, student_feats):
                layer_loss = 0.0
                for b in range(B):
                    mask_idx = mask_indices_batch[b]
                    t = t_feat[b, mask_idx, :]
                    s = s_feat[b, mask_idx, :]
                    layer_loss += F.mse_loss(s, t)
                total_loss += layer_loss / B

            total_loss = total_loss / len(target_layers)
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())

            if batch_idx % 10 == 0:
                print(f"  Epoch [{epoch}/{num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {total_loss.item():.6f}")

        avg_loss = np.mean(epoch_losses)
        print(f"[Pretrain Epoch {epoch}] Avg Loss: {avg_loss:.6f}")
        writer.add_scalar("Pretrain/Loss", avg_loss, epoch)

        train_dataset.on_epoch_end()

    torch.save(dinov2.state_dict(), adapted_path)
    print(f"[Pretrain] Saved adapted weights: {adapted_path}")

    writer.close()
    print("[Pretrain] Done. Run './run.sh train' next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JADE_v3 Backbone Domain Adaptation")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    pretrain(config_dict)
