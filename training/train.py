import os
import sys
import yaml
import torch
import argparse
import numpy as np
import cv2
import glob
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import data_handler as dh
from model import model as md


def train(config_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    train_loader, train_dataset = dh.get_dataloader(config_dict, mode="train")

    jade = md.JADE_v3(config_dict).to(device)
    if jade.normalize_features:
        _set_feature_scale(jade, train_loader, device)
    optimizer = torch.optim.Adam(
        jade.memories.parameters(),
        lr=config_dict["model"]["memory"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config_dict["training"]["num_epochs"]
    )

    log_dir = config_dict["training"]["tensorboard_dir"]
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    num_epochs     = config_dict["training"]["num_epochs"]
    checkpoint_dir = config_dict["training"]["checkpoint_dir"]
    checkpoint_freq = config_dict["training"]["checkpoint_frequency_epochs"]
    eval_freq      = config_dict["training"]["eval_frequency_epochs"]

    print(f"[Train] Starting training for {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        jade.train()
        jade.feature_extractor.eval()

        epoch_losses = []

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            loss = jade.training_step(images)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            if batch_idx % 10 == 0:
                print(f"  Epoch [{epoch}/{num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}")

        scheduler.step()

        avg_loss   = np.mean(epoch_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.6f} | LR: {current_lr:.8f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        train_dataset.on_epoch_end()

        if epoch % checkpoint_freq == 0:
            jade.save_checkpoint(epoch, checkpoint_dir)

        if epoch % eval_freq == 0:
            _log_eval_images(jade, config_dict, device, writer, epoch)

    writer.close()
    print("[Train] Training complete.")


@torch.no_grad()
def _set_feature_scale(jade, loader, device, n_batches=8):
    """Per-layer scale = RMS of the raw backbone features over a few training batches."""
    jade.eval()
    jade.feat_scale.fill_(1.0)
    sq = [0.0] * jade.num_layers
    cnt = 0
    for b, (images, _) in enumerate(loader):
        if b >= n_batches:
            break
        for i, f in enumerate(jade.extract_features(images.to(device))):
            sq[i] += f.float().pow(2).mean().item()
        cnt += 1
    scale = torch.tensor([(s / cnt) ** 0.5 for s in sq], device=device)
    jade.feat_scale.copy_(scale)
    print("[Train] feature scale per layer " + ", ".join(f"L{l}={v:.2f}" for l, v in zip(jade.extract_layers, scale.tolist())))


def _log_eval_images(jade, config_dict, device, writer, epoch):
    """
    Log OK and NG heatmaps to TensorBoard.
    Global normalization — all images on the same scale.
    OK → cold blue, NG → hot red.
    """
    jade.eval()

    image_size = tuple(config_dict["data"]["image_size"])          # (height, width), like everywhere else
    size_wh = (image_size[1], image_size[0])                          # cv2.resize and DataHandler.preprocess_image want (width, height)
    extensions = config_dict["data"]["image_extensions"]
    max_images = 5

    def collect_paths(images_dir):
        paths = []
        for ext in extensions:
            paths += glob.glob(os.path.join(images_dir, ext))
        return paths[:max_images]

    ok_paths = collect_paths(dh.ok_dir(config_dict))          # the first OK pictures (they are also trained on)
    ng_dir   = dh.ng_dir(config_dict)
    ng_paths = []
    if os.path.exists(ng_dir):
        ng_paths = collect_paths(ng_dir)

    all_paths = ok_paths + ng_paths
    if not all_paths:
        return

    # Step 1 — collect raw error maps
    raw_imgs   = []
    error_maps = []

    for image_path in all_paths:
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            continue
        tensor    = dh.DataHandler.preprocess_image(raw_img, size_wh).to(device)
        error_map = jade.get_error_map(tensor)[0]
        raw_imgs.append(raw_img)
        error_maps.append(error_map)

    if not error_maps:
        return

    # Step 2 — global normalization
    all_errors = np.concatenate([e.flatten() for e in error_maps])
    global_p99 = np.percentile(all_errors, 99)
    print(f"[Eval epoch {epoch}] global p99 error = {global_p99:.6f}")

    # Step 3 — log
    tags = (["OK"] * len(ok_paths)) + (["NG"] * len(ng_paths))

    for i, (raw_img, error_map) in enumerate(zip(raw_imgs, error_maps)):
        tag = tags[i]

        normalized    = np.clip(error_map / global_p99, 0, 1) if global_p99 > 0 else np.zeros_like(error_map)
        heatmap_uint8 = (normalized * 255).astype(np.uint8)

        raw_resized   = cv2.resize(raw_img, size_wh)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlayed     = cv2.addWeighted(raw_resized, 0.6, heatmap_color, 0.4, 0)
        overlayed     = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
        raw_rgb       = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2RGB)

        comparison        = np.concatenate([raw_rgb, overlayed], axis=1)
        comparison_tensor = torch.from_numpy(comparison).permute(2, 0, 1).float() / 255.0

        idx = i - len(ok_paths) if tag == "NG" else i
        writer.add_image(f"{tag}/image_{idx}", comparison_tensor, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JADE_v3 Training")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    train(config_dict)
