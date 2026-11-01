"""
PatchCore on ANY folder of pictures (sub-folders are searched too), called by run.sh --infer.

Writes to --out: scores.csv (score_max = the largest error of the picture, score_p99) and pictures/ [ original | original + heat map ]
with the same sub-folders as the input. The memory bank comes from --checkpoint, the model definition from the config of its run.
The colours (model.heatmap_scaling_factor, model.heatmap_floor, model.pixelwise_threshold) come from --settings (else the run's config); see common.py.
"""
import os
import sys
import time
import argparse

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                           # noqa: E402
from model import PatchCore             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="config of the run that made the checkpoint")
    ap.add_argument("--settings", default=None, help="config whose heatmap_scaling_factor and pixelwise_threshold are used")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="folder of pictures (searched recursively)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="score only this many pictures, evenly spread over the folder (0 = all)")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    t_start = time.time()

    cfg = yaml.safe_load(open(a.config))
    src = a.config
    if a.settings:
        st = yaml.safe_load(open(a.settings))
        for key in ("heatmap_scaling_factor", "heatmap_floor", "pixelwise_threshold"):
            if key in (st.get("model") or {}):
                cfg["model"][key] = st["model"][key]
            else:
                cfg["model"].pop(key, None)
        src = a.settings
    size = tuple(cfg["data"]["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.checkpoint, map_location="cpu")
    pc = PatchCore(cfg, device)
    for k, v in pc.meta(size).items():
        if ck["meta"].get(k) != v:
            raise SystemExit("[Infer] the memory bank was made with %s = %s, the config says %s" % (k, ck["meta"].get(k), v))
    pc.set_bank(ck["bank"])
    print("[Infer] memory bank: %d patches (%s, stages %s)" % (ck["bank"].shape[0], pc.type, pc.layers))

    paths = common.find_images(a.data, cfg["data"]["image_extensions"])
    if not paths:
        raise SystemExit("[Infer] no pictures found in %s" % a.data)
    if a.limit > 0 and len(paths) > a.limit:
        print("[Infer] %d pictures in %s, using %d of them (evenly spread, --limit)" % (len(paths), a.data, a.limit))
        paths = common.spread(paths, a.limit)
    else:
        print("[Infer] %d pictures in %s" % (len(paths), a.data), flush=True)

    maps = common.error_maps(pc, paths, size, a.batch, cfg["data"]["num_workers"], device)
    colour_max, thr, floor = common.colour_scale(maps, cfg["model"], src)
    print("[Result] error over all pixels of these pictures: half of them below %.3f | 90%% below %.3f | 99%% below %.3f" % tuple(common.global_p99(maps.values(), q) for q in (0.5, 0.9, 0.99)))
    rows = common.save_results(maps, a.data, a.out, colour_max, thr, floor, size)
    v = np.array([r[1] for r in rows])
    print("[Infer] score (largest error of a picture) over %d pictures: median %.3f | p95 %.3f | max %.3f" % (len(v), np.median(v), np.percentile(v, 95), v.max()))
    print("[Infer] 10 highest-scoring pictures:")
    for r in rows[:10]:
        print("   %.3f  %s" % (r[1], r[0]))
    print("[Time] whole run: %.1f s for %d pictures (%.0f pictures per minute)" % (time.time() - t_start, len(rows), 60 * len(rows) / (time.time() - t_start)))
    print("[Infer] Done. Pictures in %s, scores in %s" % (os.path.join(a.out, "pictures"), os.path.join(a.out, "scores.csv")))


if __name__ == "__main__":
    main()
