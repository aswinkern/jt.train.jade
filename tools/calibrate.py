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


def calibrate(config_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Calibrate] Using device: {device}")

    jade = md.JADE_v3(config_dict).to(device)

    checkpoint_path = config_dict["model"]["memory"].get("checkpoint_path", None)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[Calibrate] ERROR: checkpoint not found: {checkpoint_path}")
        return

    jade.load_checkpoint(checkpoint_path)
    jade.eval()

    train_dir  = dh.ok_dir(config_dict)
    extensions = config_dict["data"]["image_extensions"]
    image_size = tuple(config_dict["data"]["image_size"])

    train_paths = []
    for ext in extensions:
        train_paths += glob.glob(os.path.join(train_dir, ext))
    print(f"[Calibrate] Running on {len(train_paths)} training images...")

    all_errors = []
    for image_path in train_paths:
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            continue
        tensor = dh.DataHandler.preprocess_image(raw_img, image_size).to(device)
        with torch.no_grad():
            error_map = jade.get_error_map(tensor)[0]
        all_errors.append(error_map.flatten())

    if not all_errors:
        print("[Calibrate] ERROR: No images processed.")
        return

    all_errors = np.concatenate(all_errors)
    threshold  = float(np.percentile(all_errors, 99))
    print(f"[Calibrate] p99 threshold = {threshold:.6f}")
    print(f"[Calibrate] p95           = {float(np.percentile(all_errors, 95)):.6f}")
    print(f"[Calibrate] mean          = {float(np.mean(all_errors)):.6f}")
    print(f"[Calibrate] max           = {float(np.max(all_errors)):.6f}")

    threshold_path = config_dict["training"].get("threshold_path", "./output/threshold.npy")
    os.makedirs(os.path.dirname(os.path.abspath(threshold_path)), exist_ok=True)
    np.save(threshold_path, np.array(threshold, dtype=np.float32))
    print(f"[Calibrate] Saved: {threshold_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JADE_v3 Threshold Calibration")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    parser.add_argument("-m", "--model",  type=str, default=None,
                        help="Override checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    if args.model:
        config_dict["model"]["memory"]["checkpoint_path"] = args.model
        config_dict["model"]["memory"]["load_checkpoint"] = True

    calibrate(config_dict)
