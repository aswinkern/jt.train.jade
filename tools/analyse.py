import os
import sys
import yaml
import glob
import cv2
import torch
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import data_handler as dh
from model import model as md


def analyse(config_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Analyse] Using device: {device}")

    # Load model
    jade = md.JADE_v3(config_dict).to(device)
    checkpoint_path = config_dict["model"]["memory"].get("checkpoint_path", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        jade.load_checkpoint(checkpoint_path)
    else:
        print("[Analyse] Warning: No checkpoint loaded — using untrained memory.")
    jade.eval()

    # Load calibrated threshold
    threshold_path = config_dict["training"].get("threshold_path", None)
    if threshold_path and os.path.exists(threshold_path):
        threshold = float(np.load(threshold_path))
        print(f"[Analyse] Loaded threshold: {threshold:.6f}")
    else:
        print("[Analyse] WARNING: No threshold file. Run ./run.sh calibrate first.")
        threshold = None

    # Inference suppression config
    infer_cfg          = config_dict.get("inference", {})
    min_score          = infer_cfg.get("min_score", 0.0)
    black_pixel_thresh = infer_cfg.get("black_pixel_threshold", 2)

    test_dir   = config_dict["data"].get("test_images_dir") or dh.ok_dir(config_dict)      # --image_dir sets test_images_dir
    extensions = config_dict["data"]["image_extensions"]
    image_size = tuple(config_dict["data"]["image_size"])

    test_paths = []
    for ext in extensions:
        test_paths += glob.glob(os.path.join(test_dir, ext))
    print(f"[Analyse] Test images: {len(test_paths)}")

    output_dir = config_dict.get("analysis", {}).get("output_dir", "./output/analysis")
    os.makedirs(output_dir, exist_ok=True)

    for image_path in test_paths:
        image_name = os.path.basename(image_path)
        raw_img    = cv2.imread(image_path)
        if raw_img is None:
            continue

        tensor = dh.DataHandler.preprocess_image(raw_img, image_size).to(device)
        with torch.no_grad():
            error_map = jade.get_error_map(tensor)[0]

        t          = threshold if threshold else float(np.percentile(error_map, 99))
        normalized = np.clip(error_map / t, 0, 1) if t > 0 else np.zeros_like(error_map)

        # Suppression 1: score below min_score → 0
        if min_score > 0:
            normalized[normalized < min_score] = 0

        # Suppression 2: black pixels in original image → 0
        raw_resized = cv2.resize(raw_img, image_size)
        black_mask  = np.max(raw_resized, axis=2) <= black_pixel_thresh
        normalized[black_mask] = 0

        heatmap = (normalized * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlayed     = cv2.addWeighted(raw_resized, 0.6, heatmap_color, 0.4, 0)
        comparison    = np.concatenate([raw_resized, overlayed], axis=1)

        cv2.imwrite(os.path.join(output_dir, image_name), comparison)
        print(f"[Analyse] {image_name} p99={np.percentile(error_map,99):.4f} max={error_map.max():.4f}")

    print(f"[Analyse] Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JADE_v3 Analysis")
    parser.add_argument("-c", "--config",     type=str, default="./config.yaml")
    parser.add_argument("-i", "--image_dir",  type=str, default=None)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("-m", "--model",      type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    if args.image_dir:
        config_dict["data"]["test_images_dir"] = args.image_dir
    if args.output_dir:
        config_dict.setdefault("analysis", {})["output_dir"] = args.output_dir
    if args.model:
        config_dict["model"]["memory"]["checkpoint_path"] = args.model

    analyse(config_dict)
