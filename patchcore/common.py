"""
Shared by fit.py and infer.py: finding and loading pictures, the colour scale and the picture that is saved.

The pictures are drawn like the TensorBoard pictures of JADE: [ original | 0.6 original + 0.4 heat map ].
Heat map colours: (error - floor) / (dark-red value - floor), clipped to 0..1 (0 = dark blue, 100 = dark red).
  dark-red value   model.heatmap_scaling_factor if it is in the config, else the 99th percentile of the error over all pixels of the pictures of that run.
  floor            model.heatmap_floor (default 0): an error at or below it is dark blue. PatchCore's error is a distance, it is never close to 0, even
                   for a normal patch. Set the floor to the normal level and the normal parts become dark blue.
model.pixelwise_threshold (0-100): a pixel at or below it (on that 0-100 scale) is drawn dark blue.
"""
import os
import csv
import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)       # ImageNet, what the backbone was trained with
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_images(root, exts):
    suffixes = tuple(e.replace("*", "").lower() for e in exts)
    out = []
    for d, _, files in os.walk(root):
        out += [os.path.join(d, f) for f in files if f.lower().endswith(suffixes)]
    return sorted(out)


def spread(paths, n):
    """n pictures evenly spread over the list (all of them when n is 0 or more than there are)."""
    if n <= 0 or len(paths) <= n:
        return list(paths)
    return [paths[int(i * len(paths) / n)] for i in range(n)]


class Pictures(torch.utils.data.Dataset):
    def __init__(self, paths, size_hw):
        self.paths = paths
        self.size = size_hw

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        raw = cv2.imread(self.paths[i])
        if raw is None:
            raise RuntimeError("cannot read " + self.paths[i])
        H, W = self.size
        rgb = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), (W, H))
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


def loader(paths, size_hw, batch, workers):
    return torch.utils.data.DataLoader(Pictures(paths, size_hw), batch_size=batch, shuffle=False, num_workers=workers)


def error_maps(model, paths, size_hw, batch, workers, device):
    """{path: float32 error map (H, W)} of every picture."""
    maps, done = {}, 0
    for x in loader(paths, size_hw, batch, workers):
        m = model.error_map(x.to(device), size_hw).cpu().numpy()
        for k in range(m.shape[0]):
            maps[paths[done + k]] = m[k]
        done += m.shape[0]
        if done % 200 < m.shape[0] and done < len(paths):
            print("[Score] %d / %d" % (done, len(paths)), flush=True)
    return maps


def global_p99(maps, q=0.99):
    """99th percentile of the error over all pixels of all the maps (the colour scale of the TensorBoard pictures of JADE).
    Counted in 100000 steps between 0 and the largest error, so the maps are not all in one array."""
    maps = list(maps)
    top = max(float(m.max()) for m in maps)
    if top <= 0:
        return 0.0
    steps = 100000
    count = np.zeros(steps + 1, dtype=np.int64)
    for m in maps:
        count += np.bincount(np.clip(m / top * steps, 0, steps).astype(np.int64).ravel(), minlength=steps + 1)
    return float(np.searchsorted(np.cumsum(count), q * count.sum()) / steps * top)


def colour_scale(maps, model_cfg, src):
    """(dark-red value, pixelwise_threshold, floor) from the config; prints what is used."""
    factor = model_cfg.get("heatmap_scaling_factor")
    if factor:
        colour_max = float(factor)
        print("[Result] heatmap_scaling_factor = %g from %s: dark red = error %g, dark blue = 0" % (colour_max, src, colour_max))
    else:
        colour_max = global_p99(maps.values()) or 1.0
        print("[Result] no heatmap_scaling_factor in %s: dark red = error %.4f (the 99th percentile of all pixels of these pictures), dark blue = 0" % (src, colour_max))
    floor = float(model_cfg.get("heatmap_floor") or 0)
    if floor >= colour_max:
        raise SystemExit("[Result] heatmap_floor %g must be smaller than the dark-red value %g" % (floor, colour_max))
    print("[Result] heatmap_floor = %g: errors at or below it are dark blue" % floor)
    thr = model_cfg.get("pixelwise_threshold")
    if thr is None:
        print("[Result] no pixelwise_threshold in %s: nothing is hidden" % src)
    else:
        print("[Result] pixelwise_threshold = %g from %s: pixels at or below %g (of 100) are drawn dark blue" % (thr, src, thr))
    return colour_max, thr, floor


def colour_value(d_px, colour_max, thr, floor=0.0):
    value = np.clip((d_px - floor) / (colour_max - floor), 0, 1)
    if thr is not None:
        value = np.where(value * 100.0 > thr, value, 0.0)
    return value


def visual_mode(model_cfg):
    """model.visualisation: "heatmap" (default) or "boxes"."""
    mode = model_cfg.get("visualisation") or "heatmap"
    if mode not in ("heatmap", "boxes"):
        raise SystemExit('[Result] model.visualisation must be "heatmap" or "boxes", got: %s' % mode)
    print("[Result] visualisation = %s" % mode)
    return mode


def picture(path, error_map, colour_max, thr, floor, size_hw, raw=None, mode="heatmap"):
    """The picture that is saved: original | original + heat map, or (mode "boxes") original | original with the prediction boxes.
    raw = the picture already read (BGR), else it is read from path."""
    from tools.boxes import boxes_from_value, draw_boxes
    H, W = size_hw
    base = cv2.resize(cv2.imread(path) if raw is None else raw, (W, H))
    value = colour_value(error_map, colour_max, thr, floor)
    if mode == "boxes":
        return np.concatenate([base, draw_boxes(base.copy(), boxes_from_value(value, error_map, thr, (H, W)))], axis=1)
    heat = cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return np.concatenate([base, cv2.addWeighted(base, 0.6, heat, 0.4, 0)], axis=1)


def save_results(maps, root, out_dir, colour_max, thr, floor, size_hw, mode="heatmap"):
    """out_dir/scores.csv, out_dir/pictures/ and out_dir/boxes/ (prediction boxes as json, see tools/boxes.py), same sub-folders as root,
    for these maps. Returns the rows (image, score, p99)."""
    from tools.boxes import boxes_from_value, write_boxes_json
    settings = {"heatmap_scaling_factor": colour_max, "heatmap_floor": floor, "pixelwise_threshold": thr}
    rows = []
    for p, m in maps.items():
        rel = os.path.relpath(p, root)
        raw = cv2.imread(p)
        rows.append((rel, float(m.max()), float(np.percentile(m, 99))))
        dst = os.path.join(out_dir, "pictures", os.path.splitext(rel)[0] + ".png")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, picture(p, m, colour_max, thr, floor, size_hw, raw, mode))
        boxes = boxes_from_value(colour_value(m, colour_max, thr, floor), m, thr, raw.shape[:2])
        write_boxes_json(out_dir, rel, raw.shape[:2], m.max(), boxes, settings)
    rows.sort(key=lambda r: -r[1])                                 # highest score first
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scores.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["image", "score_max", "score_p99"])
        w.writerows(rows)
    return rows
