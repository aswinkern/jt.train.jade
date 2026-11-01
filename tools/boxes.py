"""
Prediction boxes from a heat map, written by run.sh --infer of both methods (jade and patchcore) as one JSON file per picture.

A box is a region of the heat map that is not dark blue: the pixels whose colour value (0 = dark blue ... 100 = dark red) is above
model.pixelwise_threshold are joined (8-neighbours) into regions, and every region becomes one box. There is no other filter, so
tiny regions are boxes too: use "score" and "area" to filter them. With pixelwise_threshold 0 (or missing) every pixel above the
colour floor counts, which for JADE (no floor) is the whole picture: set a pixelwise_threshold to get useful boxes.

The coordinates are in the ORIGINAL picture (the size of the file), like the labels of the training data, not in the size the model saw.

model.visualisation chooses what the saved pictures show: "heatmap" (original | original + heat map, the default) or "boxes"
(original | original with these boxes drawn). The json files are written in both cases.

<out>/boxes/<same sub-folders as the input>/<picture>.json:
  {"image": ..., "width": W, "height": H, "score": largest error of the picture, "settings": {...colours used...},
   "boxes": [{"x1": .., "y1": .., "x2": .., "y2": .., "score": largest error inside the box, "area": pixels of the region (original size)}, ...]}
boxes are sorted by score, highest first.
"""
import os
import json

import cv2
import numpy as np


def boxes_from_value(value, error, thr, orig_hw):
    """value (h, w): colour value 0..1 of the picture as the model saw it. error (h, w): the error map. thr: pixelwise_threshold (0-100) or None.
    orig_hw: (height, width) of the original picture. Returns the boxes in original pixels."""
    h, w = value.shape
    H0, W0 = orig_hw
    sx, sy = W0 / float(w), H0 / float(h)
    mask = (value * 100.0 > (thr or 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        boxes.append({"x1": int(np.floor(x * sx)), "y1": int(np.floor(y * sy)),
                      "x2": int(min(W0, np.ceil((x + bw) * sx))), "y2": int(min(H0, np.ceil((y + bh) * sy))),
                      "score": round(float(error[lab == i].max()), 4), "area": int(round(area * sx * sy))})
    boxes.sort(key=lambda b: -b["score"])
    return boxes


def draw_boxes(img, boxes):
    """Draws the boxes (red, with the score) on img (BGR, in place) and returns it. The boxes must be in the pixels of img."""
    for bx in boxes:
        x1, y1, x2, y2 = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = "%.2f" % bx["score"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 6
        cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), (0, 0, 255), -1)
        cv2.putText(img, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_boxes_json(out_dir, rel, orig_hw, score, boxes, settings):
    path = os.path.join(out_dir, "boxes", os.path.splitext(rel)[0] + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"image": rel, "width": int(orig_hw[1]), "height": int(orig_hw[0]), "score": round(float(score), 4),
                   "settings": settings, "boxes": boxes}, f, indent=1)
    return path
