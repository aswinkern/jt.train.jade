"""
Inference with a trained JADE model on ANY folder of pictures (sub-folders are searched too).

For every picture the combined heat map is scored (p99 and max of the raw error, combined and per layer) and
written to scores.csv. A picture [ original | original + heat map ] is saved for every image, with the same sub-folders as the input,
drawn exactly like the TensorBoard pictures of the training (0.6 picture + 0.4 colours, no text). How much each layer counts in the
combined map comes from model.layer_weights in the config (the --settings file, else the run's own config).

Colours: heat map = error / dark-red value, clipped to 0..1: 0 = dark blue, 100 = dark red. The dark-red value is
model.heatmap_scaling_factor if it is set in the config (--settings file, else the run's config). If the line is missing it is
what TensorBoard uses: the 99th percentile of the error over all pixels of the pictures scored in this run (the value is printed).
Two JSON files are saved per picture, in two folders of --out (same sub-folders as the input, file name + .json):
  raw/   the model's error map of every layer, before any post-processing (no layer_weights, factor or pixelwise_threshold;
         layer_maps_featurewise is added when featurewise_threshold is on). Made once by the GPU.
  post/  the result with the current settings: the settings used, scores, the combined error map and the 0-100 colour
         values. This is what the picture is drawn from.
The next run with the same --out, --data, checkpoint, image size and featurewise_threshold does not run the model:
layer_weights, heatmap_scaling_factor and pixelwise_threshold only redo post/, scores.csv and pictures/.
--redo runs the model again.
model.featurewise_threshold (optional, 0 = off): squared feature differences at or below it count as 0 before averaging.
model.pixelwise_threshold (0-100, optional): the only filter. A pixel whose colour value (0 = dark blue ... 100 = dark red) is at
or below it is drawn dark blue (as if nothing was found); the others keep their colour. Picture only: no OK/NG verdict.
"""
import os
import sys
import csv
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))        # JADE_v3, so that model/, data/ ... can be imported
from model import model as md                      # noqa: E402
from inference.evaluate import error_maps_batch, spread   # noqa: E402
from tools.boxes import boxes_from_value, draw_boxes, write_boxes_json   # noqa: E402
from inference.postprocess import colour_value   # noqa: E402  (shared with the Triton client)


def find_images(root, exts):
    suffixes = tuple(e.replace("*", "").lower() for e in exts)
    out = []
    for d, _, files in os.walk(root):
        out += [os.path.join(d, f) for f in files if f.lower().endswith(suffixes)]
    return sorted(out)


def make_meta(data, checkpoint, jade, layers, size):
    """What the saved layer maps depend on. the raw/ files are reused only when all of this is the same."""
    ck = os.path.abspath(checkpoint)
    return {"data": os.path.abspath(data), "checkpoint": ck, "ckpt_size": os.path.getsize(ck),
            "ckpt_mtime": int(os.path.getmtime(ck)), "featurewise_threshold": jade.featurewise_threshold,
            "layers": list(layers), "image_size": list(size)}


def raw_file(out, rel):
    return os.path.join(out, "raw", rel + ".json")


def write_raw(out, rel, meta, grid, raw_maps, fw_maps):
    """raw/<picture>.json: the model's error map of every layer. layer_maps has no threshold at all; layer_maps_featurewise
    is the same with featurewise_threshold applied (only when it is on)."""
    path = raw_file(out, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = lambda maps: [np.round(m, 4).ravel().tolist() for m in maps]
    d = {"image": rel, "meta": meta, "grid": list(grid), "layers": meta["layers"], "layer_maps": flat(raw_maps)}
    if fw_maps is not None:
        d["layer_maps_featurewise"] = flat(fw_maps)
    with open(path + ".tmp", "w") as f:
        json.dump(d, f, separators=(",", ":"))
    os.replace(path + ".tmp", path)


def read_raw(out, rel, meta):
    """The per-layer error maps to use (featurewise applied when it is on) from raw/<picture>.json, None if missing or made
    with another checkpoint / data folder / featurewise_threshold / image size."""
    path = raw_file(out, rel)
    if not os.path.isfile(path):
        return None
    try:
        d = json.load(open(path))
    except ValueError:
        return None
    if d.get("meta") != json.loads(json.dumps(meta)):
        return None
    key = "layer_maps_featurewise" if meta["featurewise_threshold"] > 0 else "layer_maps"
    if key not in d:
        return None
    return [np.array(m, dtype=np.float32).reshape(d["grid"]) for m in d[key]]


def timing(name, times):
    """times: [(seconds, number of pictures)] per step. Prints the time of one picture (min / mean / max) and pictures per minute.
    The first step is left out when there are more (GPU warm-up) and shown on its own."""
    if not times:
        return
    first = times[0]
    rest = times[1:] if len(times) > 1 else times
    per = [t / n for t, n in rest if n]
    if not per:
        return
    print("[Time] %s: one picture min %.0f ms | mean %.0f ms | max %.0f ms  ->  %.0f pictures per minute  (%d pictures%s)" % (
        name, 1000 * min(per), 1000 * sum(t for t, _ in rest) / sum(n for _, n in rest), 1000 * max(per),
        60 * sum(n for _, n in rest) / sum(t for t, _ in rest), sum(n for _, n in times),
        ", first step %.1f s left out as warm-up" % first[0] if len(times) > 1 else ""), flush=True)


def global_p99(maps, W, H):
    """The 99th percentile of the error over all pixels (at the size of the picture) of all the pictures: the colour scale of the
    TensorBoard pictures. Counted in 100000 steps between 0 and the largest error, so the pictures are not all in memory at once."""
    maps = list(maps)
    top = max(float(m.max()) for m in maps)
    if top <= 0:
        return 0.0
    steps = 100000
    count = np.zeros(steps + 1, dtype=np.int64)
    for m in maps:
        d = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        count += np.bincount(np.clip(d / top * steps, 0, steps).astype(np.int64).ravel(), minlength=steps + 1)
    return float(np.searchsorted(np.cumsum(count), 0.99 * count.sum()) / steps * top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--settings", default=None, help="config whose model.layer_weights replaces the run's own copy")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="folder of pictures (searched recursively)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="score only this many pictures, evenly spread over the folder (0 = all)")
    ap.add_argument("--batch", type=int, default=8, help="pictures sent to the GPU together (lower it if the GPU runs out of memory)")
    ap.add_argument("--redo", action="store_true", help="run the model on every picture again, ignoring the saved raw/ files")
    a = ap.parse_args()
    t_start = time.perf_counter()

    cfg = yaml.load(open(a.config), Loader=yaml.FullLoader)
    src = a.config
    if a.settings:
        st = yaml.load(open(a.settings), Loader=yaml.FullLoader)
        for key in ("layer_weights", "featurewise_threshold", "heatmap_scaling_factor", "pixelwise_threshold", "visualisation"):
            if key in (st.get("model") or {}):
                cfg["model"][key] = st["model"][key]
            else:
                cfg["model"].pop(key, None)
        src = a.settings
    H, W = cfg["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jade = md.JADE_v3(cfg).to(device)
    jade.load_checkpoint(a.checkpoint)
    jade.eval()
    grid = (H // jade.patch_size, W // jade.patch_size)
    layers = list(jade.extract_layers)
    print("[Infer] layer_weights (L%s) = %s from %s  (as used, normalised: %s)" % (
        ", L".join(str(l) for l in layers), cfg["model"].get("layer_weights", "not set = equal"), src,
        [round(w, 3) for w in jade.layer_weights]))
    if jade.featurewise_threshold > 0:
        print("[Infer] featurewise_threshold = %g from %s: feature differences (squared) at or below it count as 0" % (jade.featurewise_threshold, src))
    else:
        print("[Infer] featurewise_threshold = off (0 or not set in %s)" % src)

    paths = find_images(a.data, cfg["data"]["image_extensions"])
    if not paths:
        raise SystemExit("[Infer] no pictures found in %s" % a.data)
    if a.limit > 0 and len(paths) > a.limit:
        print("[Infer] %d pictures in %s, using %d of them (evenly spread, --limit)" % (len(paths), a.data, a.limit))
        paths = spread(paths, a.limit)
    else:
        print("[Infer] %d pictures in %s" % (len(paths), a.data), flush=True)

    os.makedirs(a.out, exist_ok=True)
    meta = make_meta(a.data, a.checkpoint, jade, layers, (H, W))
    results, todo = {}, []
    print("[Infer] looking for saved raw predictions in %s" % os.path.join(a.out, "raw"), flush=True)
    for p in paths:
        rel = os.path.relpath(p, a.data)
        got = None if a.redo else read_raw(a.out, rel, meta)
        if got is None:
            todo.append(p)
        else:
            results[rel] = got
    if todo:
        print("[Infer] model run on %d pictures, %d at a time (%d raw predictions taken from the saved files%s)"
              % (len(todo), a.batch, len(results), ", --redo" if a.redo else ""), flush=True)
    else:
        print("[Infer] all %d raw predictions are already saved: the model is not run" % len(results), flush=True)
    model_times = []
    t_model = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:                          # next pictures are read while the GPU works
        chunks = [todo[k:k + a.batch] for k in range(0, len(todo), a.batch)]
        loading = pool.submit(lambda c: [cv2.imread(p) for p in c], chunks[0]) if chunks else None
        for ci, chunk in enumerate(chunks):
            raws = loading.result()
            loading = pool.submit(lambda c: [cv2.imread(p) for p in c], chunks[ci + 1]) if ci + 1 < len(chunks) else None
            ok = [(p, r) for p, r in zip(chunk, raws) if r is not None]
            for p, r in zip(chunk, raws):
                if r is None:
                    print("[Infer] cannot read %s" % p)
            if ok:
                t0 = time.perf_counter()
                batch_maps = error_maps_batch(jade, [r for _, r in ok], (H, W), grid, device)      # pictures -> error maps (waits for the GPU)
                model_times.append((time.perf_counter() - t0, len(ok)))
                for (p, _), (raw_m, fw_m) in zip(ok, batch_maps):
                    rel = os.path.relpath(p, a.data)
                    write_raw(a.out, rel, meta, grid, raw_m, fw_m)
                    results[rel] = fw_m if fw_m is not None else raw_m
            if (ci + 1) % max(1, 200 // a.batch) == 0:
                print("[Infer] %d / %d" % (min((ci + 1) * a.batch, len(todo)), len(todo)), flush=True)

    if model_times:
        print("[Time] model stage: %d pictures in %.1f s (includes reading the pictures and saving raw/)" % (
            sum(n for _, n in model_times), time.perf_counter() - t_model), flush=True)
        timing("model (pictures to error map)", model_times)

    maps, rows = {}, []
    for i, p in enumerate(paths):
        rel = os.path.relpath(p, a.data)
        if rel not in results:
            continue
        em = results[rel] + [np.tensordot(np.array(jade.layer_weights), np.stack(results[rel]), axes=1)]   # layers..., combined
        maps[p] = em[-1]
        row = [rel]
        for m in [em[-1]] + em[:-1]:
            row += [float(np.percentile(m, 99)), float(m.max())]
        rows.append(row)

    os.makedirs(a.out, exist_ok=True)
    names = ["ALL"] + ["L%d" % l for l in layers]
    with open(os.path.join(a.out, "scores.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["image"] + ["%s_%s" % (n, k) for n in names for k in ("p99", "max")])
        w.writerows(rows)

    order = sorted(rows, key=lambda r: -r[2])                      # by combined max, highest first
    chosen = [r[0] for r in order]                                 # a picture for every image
    factor = cfg["model"].get("heatmap_scaling_factor")
    if factor:
        colour_max = float(factor)
        print("[Infer] heatmap_scaling_factor = %g from %s: dark red = raw error %g, dark blue = 0" % (colour_max, src, colour_max))
    else:
        colour_max = global_p99(maps.values(), W, H) or 1.0
        print("[Infer] no heatmap_scaling_factor in %s: colours like TensorBoard, dark red = raw error %.4f (the 99th percentile of all pixels of these pictures), dark blue = 0" % (src, colour_max))
    thr = cfg["model"].get("pixelwise_threshold")
    if thr is None:
        print("[Infer] no pixelwise_threshold in %s: nothing is hidden" % src)
    else:
        print("[Infer] pixelwise_threshold = %g from %s: pixels at or below %g (of 100) are drawn dark blue" % (thr, src, thr))
    mode = cfg["model"].get("visualisation") or "heatmap"            # "heatmap" (default) or "boxes": what the saved pictures show
    if mode not in ("heatmap", "boxes"):
        raise SystemExit('[Infer] model.visualisation must be "heatmap" or "boxes", got: %s' % mode)
    print("[Infer] visualisation = %s" % mode)
    pic_dir = os.path.join(a.out, "pictures")
    byname = {r[0]: r for r in rows}
    for rel in chosen:
        p = os.path.join(a.data, rel)
        raw = cv2.imread(p)
        base = cv2.resize(raw, (W, H))
        d = cv2.resize(maps[p], (W, H), interpolation=cv2.INTER_LINEAR)
        value = colour_value(d, colour_max, thr)                          # 0..1: 0 = dark blue, 1 = dark red
        write_boxes_json(a.out, rel, raw.shape[:2], byname[rel][2], boxes_from_value(value, d, thr, raw.shape[:2]),    # prediction boxes: tools/boxes.py
                         {"heatmap_scaling_factor": colour_max, "pixelwise_threshold": thr})
        heat = cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = np.concatenate([base, cv2.addWeighted(base, 0.6, heat, 0.4, 0)], axis=1)      # the TensorBoard picture: original | original + heat map
        if mode == "boxes":                                                                 # original | original with the prediction boxes
            img = np.concatenate([base, draw_boxes(base.copy(), boxes_from_value(value, d, thr, (H, W)))], axis=1)
        ph, pw = H // grid[0], W // grid[1]
        grid_value = value[:grid[0] * ph, :grid[1] * pw].reshape(grid[0], ph, grid[1], pw).mean(axis=(1, 3)) * 100.0   # the same colour value per 14 x 14 square, 0..100
        post = {"image": rel, "grid": list(grid), "layers": layers,
                "settings": {"layer_weights": [round(w, 6) for w in jade.layer_weights], "featurewise_threshold": jade.featurewise_threshold,
                             "heatmap_scaling_factor": colour_max, "pixelwise_threshold": thr},
                "scores": dict(zip(["%s_%s" % (n, k) for n in names for k in ("p99", "max")], [round(x, 5) for x in byname[rel][1:]])),
                "combined_map": np.round(maps[p], 4).ravel().tolist(), "colour_value_map": np.round(grid_value, 1).ravel().tolist()}
        post_path = os.path.join(a.out, "post", rel + ".json")
        os.makedirs(os.path.dirname(post_path), exist_ok=True)
        with open(post_path, "w") as f:
            json.dump(post, f, separators=(",", ":"))
        dst = os.path.join(pic_dir, os.path.splitext(rel)[0] + ".png")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, img)

    v = np.array([r[2] for r in rows])
    print("[Infer] combined max error over %d pictures: median %.3f | p95 %.3f | max %.3f" % (len(v), np.median(v), np.percentile(v, 95), v.max()))
    print("[Infer] 10 highest-scoring pictures:")
    for r in order[:10]:
        print("   %.3f  %s" % (r[2], r[0]))
    print("[Time] whole run: %.1f s for %d pictures (%.0f pictures per minute, everything included)" % (
        time.perf_counter() - t_start, len(rows), 60 * len(rows) / (time.perf_counter() - t_start)))
    print("[Infer] Done. %d pictures saved in %s, scores in %s" % (len(chosen), pic_dir, os.path.join(a.out, "scores.csv")))


if __name__ == "__main__":
    main()
