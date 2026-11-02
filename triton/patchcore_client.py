"""
PatchCore Triton client (gRPC). Gives the same result as `run.sh --infer`: the model returns the raw error map, and the colours, the boxes and the
saved pictures come from patchcore/render.py (the code of --infer) with the numbers below (the same numbers as config.yaml; `run.sh --export` prints
this block for the model it exported). The colour settings can be changed here any time, no new export needed.
  python patchcore_client.py -i <picture or folder> -o <output folder> [-u localhost:8001] [--batch 8]
Output, like `run.sh --infer`: <output folder>/scores.csv, pictures/ [original | original + heat map] and boxes/ (one json per picture), with the
same sub-folders as the input, so tools/analytics_html.py and tools/folder_report.py read it as it is.
A picture is flagged when it has at least one box.
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

for _p in ("/jidoka_training/code/jt.train.jade", "/jidoka/code/jt.train.jade"):      # the PatchCore code (numpy + opencv only): inference/postprocess.py, patchcore/render.py, tools/boxes.py
    if os.path.isdir(_p):
        sys.path.append(_p)
        sys.path.append(os.path.join(_p, "patchcore"))
        break
from inference.postprocess import prepare_picture      # noqa: E402
import render                                          # noqa: E402

# ── Settings: change them here ────────────────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "PatchCore_v1.0.0"     # model name in Triton (export.model_name)
INPUT_NAME = "image"
OUTPUT_NAME = "error_map"
IMAGE_SIZE = (518, 518)             # height, width the model was exported for
MAX_BATCH = 8                       # the largest batch (--max_batch of the export, default 8)
HEATMAP_SCALING_FACTOR = 1.2233     # (error - floor) / (this number - floor), clipped to 0..1: the number or more = dark red
HEATMAP_FLOOR = 0.9272              # an error at or below it is dark blue
PIXELWISE_THRESHOLD = 90            # (0-100) weaker pixels are drawn dark blue; None = nothing is hidden
MIN_BOX_SCORE = 1.45                # a box whose score (largest error inside it) is below it is dropped; None = every box is kept
VISUALISATION = "heatmap"           # "heatmap" or "boxes": what the saved pictures show
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def find_pictures(path):
    if os.path.isfile(path):
        return [path], os.path.dirname(os.path.abspath(path))
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs if f.lower().endswith((".png", ".jpg", ".jpeg"))), path


def main():
    ap = argparse.ArgumentParser(description="PatchCore Triton gRPC client")
    ap.add_argument("-i", "--image", required=True, help="picture or folder of pictures (searched recursively)")
    ap.add_argument("-o", "--output", required=True, help="output folder")
    ap.add_argument("-u", "--url", default="localhost:8001", help="Triton gRPC URL")
    ap.add_argument("--batch", type=int, default=1, help="pictures per request (at most MAX_BATCH)")
    a = ap.parse_args()

    H, W = IMAGE_SIZE
    if a.batch > MAX_BATCH:
        raise SystemExit("[Client] --batch %d is more than MAX_BATCH %d" % (a.batch, MAX_BATCH))
    print("[Client] HEATMAP_SCALING_FACTOR %g | HEATMAP_FLOOR %g | PIXELWISE_THRESHOLD %s | MIN_BOX_SCORE %s | VISUALISATION %s" % (
        HEATMAP_SCALING_FACTOR, HEATMAP_FLOOR, PIXELWISE_THRESHOLD, MIN_BOX_SCORE, VISUALISATION))

    client = grpcclient.InferenceServerClient(url=a.url, verbose=False)
    print("[Client] Triton: %s, model %s" % (a.url, MODEL_NAME))
    pics, root = find_pictures(a.image)
    if not pics:
        raise SystemExit("[Client] no pictures in %s" % a.image)
    os.makedirs(a.output, exist_ok=True)
    times, rows, flagged = [], [], 0                                 # times: (seconds, pictures) of every Triton request

    for k in range(0, len(pics), a.batch):
        chunk = pics[k:k + a.batch]
        raws = [cv2.imread(p) for p in chunk]
        good = [(p, r) for p, r in zip(chunk, raws) if r is not None]
        for p, r in zip(chunk, raws):
            if r is None:
                print("[Client] cannot read %s" % p)
        if not good:
            continue
        x = np.stack([prepare_picture(r, (H, W)) for _, r in good])                  # (B, 3, H, W)
        inp = grpcclient.InferInput(INPUT_NAME, list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        t0 = time.perf_counter()
        maps = client.infer(MODEL_NAME, inputs=[inp], outputs=[grpcclient.InferRequestedOutput(OUTPUT_NAME)]).as_numpy(OUTPUT_NAME)   # (B, H, W)
        dt = time.perf_counter() - t0                                # send + model + receive; per picture = dt / pictures in the request
        times.append((dt, len(good)))
        for (p, raw), err in zip(good, maps):
            row, nbox = render.write_one(a.output, os.path.relpath(p, root), raw, err, HEATMAP_SCALING_FACTOR, PIXELWISE_THRESHOLD, HEATMAP_FLOOR, IMAGE_SIZE, VISUALISATION, MIN_BOX_SCORE)
            rows.append(row)
            flagged += nbox > 0
            print("[Client] %s  max %.3f  p99 %.3f  boxes %d  inference %.1f ms" % (row[0], row[1], row[2], nbox, 1000 * dt / len(good)))
    render.write_scores(a.output, rows)
    if times:                                                        # the first request is left out of the average (warm-up) when there are more
        rest = times[1:] if len(times) > 1 else times
        per = [t / n for t, n in rest]
        print("[Time] inference per picture (Triton request): mean %.1f ms | min %.1f ms | max %.1f ms  ->  %.0f pictures per minute  (%d pictures%s)" % (
            1000 * sum(t for t, _ in rest) / sum(n for _, n in rest), 1000 * min(per), 1000 * max(per),
            60 * sum(n for _, n in rest) / sum(t for t, _ in rest), sum(n for _, n in times),
            ", first request %.1f ms per picture left out as warm-up" % (1000 * times[0][0] / times[0][1]) if len(times) > 1 else ""))
    print("[Client] %d of %d pictures flagged (at least one box). Done. Pictures in %s, scores in %s" % (flagged, len(rows), os.path.join(a.output, "pictures"), os.path.join(a.output, "scores.csv")))


if __name__ == "__main__":
    main()
