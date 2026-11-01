"""
One HTML page of analytics that puts the labels and the prediction boxes of run.sh --infer side by side.

  python tools/analytics_html.py --labels <folder with okay/ and defect/> --pred <--out folder of --infer, with okay/ and defect/> --out analytics.html

--labels : for each group (okay, defect) a folder  <group>/image/<picture>.jpeg  and  <group>/label/<picture>.txt  (KITTI labels:
           class 0 0 0 x1 y1 x2 y2 ...), like od_train/<component>/ . The classes defect_1 to defect_4 are treated as marks of okay
           pictures, not as defects (they are drawn dashed and are not counted as labelled defects).
--pred   : for each group the folder  <group>/boxes/<picture>.json  written by --infer (see tools/boxes.py).

The page is one file: the data, the thumbnails (cropped to the part with the chain) and the code are inside it, no internet needed.
A cutoff slider changes every number: a picture is flagged when its score is at or above the cutoff; a labelled defect is found
when at least half of its box lies inside a predicted box that passes the cutoff.
"""
import os
import sys
import glob
import json
import base64
import argparse
import datetime

import cv2

OKAY_CLASSES = ["defect_1", "defect_2", "defect_3", "defect_4"]
GROUPS = ("okay", "defect")


def read_labels(path):
    out = []
    if not os.path.isfile(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) >= 8:
            x1, y1, x2, y2 = [int(round(float(v))) for v in p[4:8]]
            out.append({"c": p[0], "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def find_image(labels_root, group, stem):
    for ext in (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"):
        p = os.path.join(labels_root, group, "image", stem + ext)
        if os.path.isfile(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None, help="default: <pred>/analytics.html")
    ap.add_argument("--title", default=None)
    ap.add_argument("--thumb-width", type=int, default=640)
    ap.add_argument("--quality", type=int, default=62, help="JPEG quality of the thumbnails")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = a.out or os.path.join(a.pred, "analytics.html")

    pics, skipped = [], 0
    settings = None
    for g in GROUPS:
        for jp in sorted(glob.glob(os.path.join(a.pred, g, "boxes", "**", "*.json"), recursive=True)):
            j = json.load(open(jp))
            stem = os.path.splitext(os.path.basename(j["image"]))[0]
            img = find_image(a.labels, g, stem)
            if img is None:
                skipped += 1
                continue
            settings = settings or j.get("settings")
            pics.append({"name": stem, "group": g, "w": j["width"], "full_h": j["height"], "score": j["score"], "boxes": j["boxes"],
                         "labels": read_labels(os.path.join(a.labels, g, "label", stem + ".txt")), "_img": img})
    if not pics:
        raise SystemExit("[Analytics] no pictures: looked for %s/<group>/boxes/*.json with a matching picture in %s" % (a.pred, a.labels))
    if skipped:
        print("[Analytics] %d box files without a picture in --labels were left out" % skipped)

    # the part of the picture that holds anything (labels, boxes): the rest is black padding
    ymax = max([b["y2"] for p in pics for b in p["boxes"]] + [l["y2"] for p in pics for l in p["labels"]] + [1])
    full_h = max(p["full_h"] for p in pics)
    crop_h = min(full_h, int(-(-(ymax + 30) // 10) * 10))

    images = []
    for p in pics:
        im = cv2.imread(p.pop("_img"))
        im = im[:crop_h]
        tw = a.thumb_width
        th = max(1, int(round(im.shape[0] * tw / float(im.shape[1]))))
        ok, buf = cv2.imencode(".jpg", cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        images.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"))
        p["img"] = len(images) - 1
        p["h"] = crop_h
        p.pop("full_h")

    n = {g: sum(1 for p in pics if p["group"] == g) for g in GROUPS}
    s = settings or {}
    short = lambda p: "/".join(os.path.abspath(p).rstrip("/").split("/")[-3:])
    sub = "%d okay pictures and %d defect pictures | labels: .../%s | detections: .../%s" % (n["okay"], n["defect"], short(a.labels), short(a.pred))
    if s:
        sub += " | colours: " + ", ".join("%s %s" % (k.replace("heatmap_", "").replace("_threshold", " threshold").replace("scaling_factor", "scale"), v) for k, v in s.items())
    sub += " | made " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    data = {"meta": {"title": a.title or "Detection analytics: " + os.path.basename(os.path.normpath(a.labels)), "subtitle": sub,
                     "okay_classes": OKAY_CLASSES, "full_height": full_h, "crop_h": crop_h, "thumb_w": a.thumb_width},
            "pictures": pics, "images": images}
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    tpl = open(os.path.join(here, "analytics_template.html")).read()
    assert tpl.count("/*__DATA__*/") == 1
    with open(out_path, "w") as f:
        f.write(tpl.replace("/*__DATA__*/", text))
    print("[Analytics] %s  (%d pictures, %.1f MB, thumbnails cropped to the top %d of %d rows)" % (out_path, len(pics), os.path.getsize(out_path) / 1e6, crop_h, full_h))


if __name__ == "__main__":
    main()
