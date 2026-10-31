"""
JADE_v3 Triton client (gRPC). Gives the same result as `run.sh --infer`: the model returns the raw error map, and the post-processing
is the shared code of inference/postprocess.py with the numbers below (the same numbers as config.yaml; `run.sh --export` prints this
block for the model it exported). The thresholds can be changed here any time, no new export needed.
  python triton_client.py -i <picture or folder> -o <output folder> [-u localhost:8001] [--batch 8]
Output: for every picture <name>.png = [original | heat map] and one line with p99, max and the number of regions.
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/jidoka_training/code/jt.train.jade", "/jidoka_training/code/jt.train.jade"):      # the JADE code: inference/postprocess.py (numpy + opencv only)
    if os.path.isdir(_p):
        sys.path.append(_p)
        break
from inference.postprocess import prepare_picture, colour_value, scores

# ── Settings: change them here ────────────────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "Anomaly_v1.0.0"       # model name in Triton (export.model_name)
INPUT_NAME = "image"
OUTPUT_NAME = "error_map"
IMAGE_SIZE = (518, 518)             # height, width the model was exported for
MAX_BATCH = 16                      # the largest batch (--max_batch of the export, default 16)
HEATMAP_SCALING_FACTOR = 0.3        # error / this number, clipped to 0..1: the number or more = dark red
PIXELWISE_THRESHOLD = 70            # (0-100) weaker pixels are drawn dark blue; None = nothing is hidden
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def find_pictures(path):
    if os.path.isfile(path):
        return [path]
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs if f.lower().endswith((".png", ".jpg", ".jpeg")))


def main():
    ap = argparse.ArgumentParser(description="JADE_v3 Triton gRPC client")
    ap.add_argument("-i", "--image", required=True, help="picture or folder of pictures")
    ap.add_argument("-o", "--output", required=True, help="output folder")
    ap.add_argument("-u", "--url", default="localhost:8001", help="Triton gRPC URL")
    ap.add_argument("--batch", type=int, default=1, help="pictures per request (at most MAX_BATCH)")
    a = ap.parse_args()

    H, W = IMAGE_SIZE
    if a.batch > MAX_BATCH:
        raise SystemExit("[Client] --batch %d is more than MAX_BATCH %d" % (a.batch, MAX_BATCH))
    print("[Client] HEATMAP_SCALING_FACTOR %g | PIXELWISE_THRESHOLD %s" % (HEATMAP_SCALING_FACTOR, PIXELWISE_THRESHOLD))

    client = grpcclient.InferenceServerClient(url=a.url, verbose=False)
    print("[Client] Triton: %s, model %s" % (a.url, MODEL_NAME))
    pics = find_pictures(a.image)
    if not pics:
        raise SystemExit("[Client] no pictures in %s" % a.image)
    os.makedirs(a.output, exist_ok=True)
    times = []                                                       # (seconds, pictures) of every Triton request

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
            p99, mx = scores(err)
            value = colour_value(err, HEATMAP_SCALING_FACTOR, PIXELWISE_THRESHOLD)
            base = cv2.resize(raw, (W, H))
            heat = cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img = np.concatenate([base, cv2.addWeighted(base, 0.6, heat, 0.4, 0)], axis=1)      # original | original + heat map, like the TensorBoard pictures
            cv2.imwrite(os.path.join(a.output, os.path.splitext(os.path.basename(p))[0] + ".png"), img)
            print("[Client] %s  p99 %.3f  max %.3f  inference %.1f ms" % (os.path.basename(p), p99, mx, 1000 * dt / len(good)))
    if times:                                                        # the first request is left out of the average (warm-up) when there are more
        rest = times[1:] if len(times) > 1 else times
        per = [t / n for t, n in rest]
        print("[Time] inference per picture (Triton request): mean %.1f ms | min %.1f ms | max %.1f ms  ->  %.0f pictures per minute  (%d pictures%s)" % (
            1000 * sum(t for t, _ in rest) / sum(n for _, n in rest), 1000 * min(per), 1000 * max(per),
            60 * sum(n for _, n in rest) / sum(t for t, _ in rest), sum(n for _, n in times),
            ", first request %.1f ms per picture left out as warm-up" % (1000 * times[0][0] / times[0][1]) if len(times) > 1 else ""))
    print("[Client] Done. Pictures in %s" % a.output)


if __name__ == "__main__":
    main()
