"""
One HTML report for a folder of pictures WITHOUT labels, from run.sh --infer.

  python tools/folder_report.py --data <the folder given to --infer> --pred <its --out folder> \\
      --reference ok=okay,ok_false_positive=okay,ng=defect \\
      --rates-from <--infer out folder of a labelled test, with okay/ and defect/> --out report.html

--data      : the folder that was given to --infer (the pictures, sub-folders are kept).
--pred      : the --out folder of --infer (boxes/ and pictures/).
--reference : optional, top-level folder name = okay | defect. These folders are used as the reference: a flagged okay picture is a
              false alarm, a not flagged defect picture is a miss. Folders not listed have no reference: their correct / wrong
              predictions are estimated from the sensitivity and specificity of the labelled test (--rates-from).
--rates-from: the --out folder of a labelled test with sub-folders okay/ and defect/ (boxes/ in each): sensitivity and specificity.
--test-data : the labelled test folder (okay/image, defect/image): pictures with the same content are marked and can be excluded.
--bank-data : the data folder the memory bank was built from (images/, NG/): pictures with the same content are marked and can be
              excluded on the page.

A picture is flagged when the model drew at least one box on it (see tools/boxes.py). The folder selector on the page filters
every number and every picture; "All" is the entire report.
The page is one file (data, thumbnails of the flagged pictures, code), no internet needed.
"""
import os
import re
import glob
import json
import base64
import hashlib
import argparse
import datetime

import cv2

EXT = (".jpeg", ".jpg", ".png")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None, help="default: <pred>/report.html")
    ap.add_argument("--title", default=None)
    ap.add_argument("--reference", default="", help="top-level folder = okay|defect, comma separated, e.g. ok=okay,ng=defect")
    ap.add_argument("--rates-from", required=True)
    ap.add_argument("--bank", default=None, help="name of the memory bank / run (shown on the page)")
    ap.add_argument("--bank-data", default=None)
    ap.add_argument("--test-data", default=None, help="the labelled test folder (okay/image, defect/image): pictures with the same content are marked and can be excluded")
    ap.add_argument("--thumb-width", type=int, default=360)
    ap.add_argument("--quality", type=int, default=55)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = a.out or os.path.join(a.pred, "report.html")
    reference = {}
    for kv in [x for x in a.reference.split(",") if x]:
        k, v = kv.split("=")
        if v not in ("okay", "defect"):
            raise SystemExit("[Report] --reference values must be okay or defect, got %s" % v)
        reference[k] = v

    rates = {}
    for key, g in (("spec", "okay"), ("sens", "defect")):
        js = glob.glob(os.path.join(a.rates_from, g, "boxes", "**", "*.json"), recursive=True)
        if not js:
            raise SystemExit("[Report] no box files in %s/%s/boxes" % (a.rates_from, g))
        flagged = sum(1 for p in js if json.load(open(p))["boxes"])
        rates[key] = {"k": (len(js) - flagged) if key == "spec" else flagged, "n": len(js)}

    bank_md5 = None
    if a.bank_data:
        bank_md5 = set()
        for d in ("images", "NG"):
            for p in glob.glob(os.path.join(a.bank_data, d, "*")):
                bank_md5.add(md5(p))

    test_md5 = None
    if a.test_data:
        test_md5 = set()
        for g in ("okay", "defect"):
            for q in glob.glob(os.path.join(a.test_data, g, "image", "*")):
                test_md5.add(md5(q))

    pics, settings = [], None
    for jp in sorted(glob.glob(os.path.join(a.pred, "boxes", "**", "*.json"), recursive=True)):
        j = json.load(open(jp))
        rel = j["image"]
        src = os.path.join(a.data, rel)
        if not os.path.isfile(src):
            continue
        settings = settings or j.get("settings")
        parts = rel.split(os.sep) if os.sep in rel else rel.split("/")
        pics.append({"name": os.path.splitext(os.path.basename(rel))[0], "group": parts[0] if len(parts) > 1 else "", "folder": "/".join(parts[:-1]) or ".",
                     "w": j["width"], "fullh": j["height"], "score": j["score"], "boxes": j["boxes"], "inBank": bool(bank_md5 and md5(src) in bank_md5), "inTest": bool(test_md5 and md5(src) in test_md5),
                     "_rel": rel})
    if not pics:
        raise SystemExit("[Report] no pictures: looked for %s/boxes/**/*.json with a picture in %s" % (a.pred, a.data))

    ymax = max([b["y2"] for p in pics for b in p["boxes"]] + [1])
    full_h = max(p["fullh"] for p in pics)
    crop_h = min(full_h, int(-(-(ymax + 30) // 10) * 10))
    images = []
    for p in pics:
        rel = p.pop("_rel")
        p["h"] = crop_h
        if not p["boxes"]:
            continue                                   # thumbnails only for flagged pictures (keeps the file small)
        pic = os.path.join(a.pred, "pictures", os.path.splitext(rel)[0] + ".png")
        im = cv2.imread(pic)
        if im is None:
            continue
        im = im[:, im.shape[1] // 2:]                  # saved picture = original | original + heat map
        im = im[:max(1, int(round(im.shape[0] * crop_h / float(p["fullh"]))))]
        tw = a.thumb_width
        th = max(1, int(round(im.shape[0] * tw / float(im.shape[1]))))
        ok, buf = cv2.imencode(".jpg", cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        images.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"))
        p["img"] = len(images) - 1
    for p in pics:
        if "img" not in p:
            p["img"] = -1
            p["boxes"] = p["boxes"]

    short = lambda p: "/".join(os.path.abspath(p).rstrip("/").split("/")[-3:])
    sub = "data: .../%s | detections: .../%s" % (short(a.data), short(a.pred))
    if a.bank:
        sub += " | memory bank: " + a.bank
    sub += " | made " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_bank = sum(1 for p in pics if p["inBank"]) if bank_md5 is not None else None
    if n_bank is not None:
        print("[Report] pictures with the same content as a picture of %s: %d of %d" % (a.bank_data, n_bank, len(pics)))
    n_test = sum(1 for p in pics if p["inTest"]) if test_md5 is not None else None
    if n_test is not None:
        print("[Report] pictures with the same content as a picture of the labelled test %s: %d of %d" % (a.test_data, n_test, len(pics)))
    data = {"meta": {"title": a.title or "Folder report: " + os.path.basename(os.path.normpath(os.path.dirname(a.data.rstrip("/")))) , "subtitle": sub, "reference": reference, "rates": rates,
                     "settings": settings or {}, "checks": {"data_folder": ".../" + short(a.data), "bank_data": os.path.basename(os.path.normpath(a.bank_data)) if a.bank_data else None, "in_bank": n_bank, "test_data": os.path.basename(os.path.normpath(a.test_data)) if a.test_data else None, "in_test": n_test}},
            "pictures": pics, "images": images}
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    tpl = open(os.path.join(here, "folder_report_template.html")).read()
    assert tpl.count("/*__DATA__*/") == 1
    with open(out_path, "w") as f:
        f.write(tpl.replace("/*__DATA__*/", text))
    print("[Report] %s  (%d pictures, %d thumbnails, %.1f MB)" % (out_path, len(pics), len(images), os.path.getsize(out_path) / 1e6))


if __name__ == "__main__":
    main()
