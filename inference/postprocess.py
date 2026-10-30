"""
Post-processing of the model's raw error map, shared by `run.sh --infer` (inference/infer_images.py) and the Triton client
(triton/triton_client.py), so both give the same result with the same numbers. Needs only numpy and opencv (no torch).

The model returns a raw error map. The three settings below are applied afterwards (they are NOT inside the exported model):
  heatmap_scaling_factor  error / this number, clipped to 0..1: 0 = dark blue, the factor or more = dark red
  pixelwise_threshold     (0-100) a pixel whose colour value, 100 x error / factor, is at or below it is drawn dark blue
  area_threshold          (pixels of the picture) a coloured region (pixels that touch each other) with fewer pixels is removed
"""
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)       # what the model was trained with (DINOv2)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare_picture(raw_bgr, image_size_hw):
    """A camera picture (BGR) -> float32 array (3, H, W) for the model: RGB, resized to (H, W) without cropping, /255, mean/std.
    The same as data.data_handler.DataHandler.preprocess_image."""
    H, W = image_size_hw
    rgb = cv2.resize(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), (W, H))
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1).astype(np.float32)


def apply_thresholds(d_px, colour_max, thr, min_area):
    """Colour value 0..1 of every pixel after pixelwise_threshold and area_threshold, and the areas (pixels) of the regions
    pixelwise_threshold left, before the area filter. A region = pixels above pixelwise_threshold that touch each other; a
    region with fewer than min_area pixels is removed (drawn dark blue)."""
    value = np.clip(d_px / colour_max, 0, 1)
    if thr is not None:
        value = np.where(value * 100.0 > thr, value, 0.0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((value > 0).astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if min_area > 0 and len(areas):
        value = np.where(np.concatenate([[False], areas >= min_area])[lab], value, 0.0)
    return value, areas


def scores(error_map):
    """p99 and max of the raw error map of one picture (the numbers of scores.csv)."""
    return float(np.percentile(error_map, 99)), float(error_map.max())
