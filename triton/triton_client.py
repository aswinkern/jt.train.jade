"""
JADE_v3 Triton client (gRPC). Gives the same result as `run.sh --infer`: the model returns the raw error map, and the post-processing
is the shared code of inference/postprocess.py with the numbers of settings.json (made by the export, next to the model; edit the
thresholds there any time, no new export needed).
  python triton_client.py -i <picture or folder> -o <output folder> -s <settings.json> [-u localhost:8001] [--batch 8]
Output: for every picture <name>.png = [original | heat map] and one line with p99, max and the number of regions.
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # the repo: inference/postprocess.py needs only numpy + opencv
from inference.postprocess import prepare_picture, apply_thresholds, scores


def find_pictures(path):
    if os.path.isfile(path):
        return [path]
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs if f.lower().endswith((".png", ".jpg", ".jpeg")))


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="JADE_v3 Triton gRPC client")
    ap.add_argument("-i", "--image", required=True, help="picture or folder of pictures")
    ap.add_argument("-o", "--output", required=True, help="output folder")
    ap.add_argument("-s", "--settings", required=True, help="settings.json written by the export")
    ap.add_argument("-u", "--url", default="localhost:8001", help="Triton gRPC URL")
    ap.add_argument("--batch", type=int, default=1, help="pictures per request (at most the max batch of the export)")
    a = ap.parse_args()

    S = json.load(open(a.settings))
    H, W = S["image_size"]
    colour_max, thr, area_thr = S["settings"]["heatmap_scaling_factor"], S["settings"]["pixelwise_threshold"], S["settings"]["area_threshold"]
    if not colour_max:
        raise SystemExit("[Client] heatmap_scaling_factor is not set in %s" % a.settings)
    if a.batch > S["batch"]["max"]:
        raise SystemExit("[Client] --batch %d is more than the max batch %d of the export" % (a.batch, S["batch"]["max"]))
    print("[Client] heatmap_scaling_factor %g | pixelwise_threshold %s | area_threshold %g  (from %s)" % (colour_max, thr, area_thr, a.settings))

    client = grpcclient.InferenceServerClient(url=a.url, verbose=False)
    print("[Client] Triton: %s, model %s" % (a.url, S["model_name"]))
    pics = find_pictures(a.image)
    if not pics:
        raise SystemExit("[Client] no pictures in %s" % a.image)
    os.makedirs(a.output, exist_ok=True)

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
        inp = grpcclient.InferInput(S["input"]["name"], list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        maps = client.infer(S["model_name"], inputs=[inp], outputs=[grpcclient.InferRequestedOutput(S["output"]["name"])]).as_numpy(S["output"]["name"])   # (B, H, W)
        for (p, raw), err in zip(good, maps):
            p99, mx = scores(err)
            value, areas = apply_thresholds(err, colour_max, thr, area_thr)
            base = cv2.resize(raw, (W, H))
            heat = cv2.applyColorMap((value * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img = np.concatenate([label(base.copy(), os.path.basename(p)[:60]),
                                  label(cv2.addWeighted(base, 0.5, heat, 0.5, 0), "p99 %.3f  max %.3f  (dark red = %.3f)" % (p99, mx, colour_max))], axis=1)
            cv2.imwrite(os.path.join(a.output, os.path.splitext(os.path.basename(p))[0] + ".png"), img)
            n_reg = int((areas >= area_thr).sum()) if area_thr > 0 else len(areas)
            print("[Client] %s  p99 %.3f  max %.3f  regions %d" % (os.path.basename(p), p99, mx, n_reg))
    print("[Client] Done. Pictures in %s" % a.output)


if __name__ == "__main__":
    main()
