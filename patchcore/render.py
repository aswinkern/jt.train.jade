"""
Colours, pictures and prediction boxes from a PatchCore error map. numpy + opencv only (NO torch), so the same code is used by
`run.sh --infer` and the fit (patchcore/common.py) and by the Triton client (triton/patchcore_client.py): same error map, same result.

  heat map colour  (error - floor) / (dark-red value - floor), clipped to 0..1 (0 = dark blue, 1 = dark red)
  pixelwise_threshold (0-100): a pixel at or below it is drawn dark blue
  min_box_score: a box (region above the threshold) whose largest error is below it is dropped, and its region is not coloured
  visualisation: "heatmap" = original | original + heat map, "boxes" = original | original with the boxes drawn
"""
import os
import csv

import cv2
import numpy as np


def colour_value(d_px, colour_max, thr, floor=0.0):
    value = np.clip((d_px - floor) / (colour_max - floor), 0, 1)
    if thr is not None:
        value = np.where(value * 100.0 > thr, value, 0.0)
    return value


def picture(raw, error_map, colour_max, thr, floor, size_hw, mode="heatmap", min_score=None):
    """The picture that is saved: original | original + heat map, or (mode "boxes") original | original with the prediction boxes.
    raw = the camera picture (BGR)."""
    from tools.boxes import boxes_from_value, draw_boxes, keep_mask
    H, W = size_hw
    base = cv2.resize(raw, (W, H))
    value = colour_value(error_map, colour_max, thr, floor)
    if mode == "boxes":
        return np.concatenate([base, draw_boxes(base.copy(), boxes_from_value(value, error_map, thr, (H, W), min_score))], axis=1)
    heat = cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
    if min_score is not None:                            # colour only the regions that become boxes (score >= min_box_score)
        over = np.where(keep_mask(value, error_map, thr, min_score)[..., None], over, base)
    return np.concatenate([base, over], axis=1)


def write_one(out_dir, rel, raw, error_map, colour_max, thr, floor, size_hw, mode="heatmap", min_score=None):
    """out_dir/pictures/<rel>.png and out_dir/boxes/<rel>.json of one picture. Returns ((rel, score, p99), number of boxes)."""
    from tools.boxes import boxes_from_value, write_boxes_json
    settings = {"heatmap_scaling_factor": colour_max, "heatmap_floor": floor, "pixelwise_threshold": thr}
    if min_score is not None:
        settings["min_box_score"] = min_score
    dst = os.path.join(out_dir, "pictures", os.path.splitext(rel)[0] + ".png")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, picture(raw, error_map, colour_max, thr, floor, size_hw, mode, min_score))
    boxes = boxes_from_value(colour_value(error_map, colour_max, thr, floor), error_map, thr, raw.shape[:2], min_score)
    write_boxes_json(out_dir, rel, raw.shape[:2], error_map.max(), boxes, settings)
    return (rel, float(error_map.max()), float(np.percentile(error_map, 99))), len(boxes)


def write_scores(out_dir, rows):
    """out_dir/scores.csv, highest score first."""
    rows = sorted(rows, key=lambda r: -r[1])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scores.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["image", "score_max", "score_p99"])
        w.writerows(rows)
    return rows
