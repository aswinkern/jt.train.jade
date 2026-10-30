"""
Inference with a trained JADE on TRAIN, VAL (OK + NG) and TEST (OK + NG), with the labelled defect boxes drawn.

One picture per image:
  [ original | ALL (JADE's plain average) | EQ (each layer first divided by its own normal size) | L a | L b | L c | L d ]
White rectangles = labelled defects (from the XML files). Dark blue = rebuilt well, dark red = rebuilt badly.
Every column has ONE fixed scale: the 99.5th percentile of the error on the held-out OK validation pictures,
so red on a good picture is a false alarm.

Besides the OK-vs-NG separation (AUC) this prints a location check: on how many NG pictures is the single
hottest patch inside a labelled defect box? (Compare with 'by chance'.)
"""
import os
import sys
import csv
import argparse
import xml.etree.ElementTree as ET

import cv2
import yaml
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(HERE)
from model import model as md                                   # noqa: E402
from inference.evaluate import list_images, spread, auc, label, error_maps   # noqa: E402


def read_boxes(xml_dir, image_path):
    p = os.path.join(xml_dir, os.path.splitext(os.path.basename(image_path))[0] + ".xml")
    if not os.path.exists(p):
        return []
    out = []
    for o in ET.parse(p).getroot().findall("object"):
        name = (o.findtext("name") or "").strip()
        if name.lower() == "ok":
            continue
        b = o.find("bndbox")
        out.append((name, float(b.findtext("xmin")), float(b.findtext("ymin")),
                    float(b.findtext("xmax")), float(b.findtext("ymax"))))
    return out


def box_mask(boxes, raw_shape, grid):
    h, w = raw_shape[:2]
    m = np.zeros(grid, bool)
    for _, x0, y0, x1, y1 in boxes:
        m[max(int(y0 / h * grid[0]), 0):min(int(y1 / h * grid[0]) + 1, grid[0]),
          max(int(x0 / w * grid[1]), 0):min(int(x1 / w * grid[1]) + 1, grid[1])] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_ok_dir", required=True)
    ap.add_argument("--val_ng_dir", required=True)
    ap.add_argument("--test_ok_dir", required=True)
    ap.add_argument("--test_ng_dir", required=True)
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("--images_only", action="store_true",
                    help="inference pictures only: [original | final JADE output (ALL)], no defect boxes, no layer columns, no tables")
    ap.add_argument("--show", type=int, default=16, help="pictures written per group (NG groups: 2x this)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(cfg["data"]["image_size"])
    exts = cfg["data"]["image_extensions"]
    H, W = image_size
    jade = md.JADE_v3(cfg).to(device)
    jade.load_checkpoint(args.checkpoint)
    jade.eval()
    grid = (H // jade.patch_size, W // jade.patch_size)
    nl = len(jade.extract_layers)
    names = ["ALL", "EQ"] + [f"L{l}" for l in jade.extract_layers]

    groups = {
        "train": list_images(args.train_dir, exts),
        "val_ok": list_images(args.val_ok_dir, exts),
        "val_ng": list_images(args.val_ng_dir, exts),
        "test_ok": list_images(args.test_ok_dir, exts),
        "test_ng": list_images(args.test_ng_dir, exts) if os.path.isdir(args.test_ng_dir) else [],
    }
    n_show = {g: args.show * (2 if g.endswith("ng") else 1) for g in groups}
    n_show["train"] = 8
    shown = {g: set(spread(p, n_show[g])) for g, p in groups.items()}

    raw_maps, raws, boxes = {}, {}, {}
    for g, paths in groups.items():
        for p in paths:
            raw = cv2.imread(p)
            if raw is None:
                continue
            em = error_maps(jade, raw, image_size, grid, device)    # layers..., ALL
            raw_maps[(g, p)] = em
            if g.endswith("ng") and not args.images_only:
                boxes[(g, p)] = read_boxes(args.xml_dir, p)
            if p in shown[g]:
                raws[(g, p)] = raw
        print(f"[Eval] {g}: {len(paths)} pictures done")

    # scales: held-out OK validation
    ok_keys = [k for k in raw_maps if k[0] == "val_ok"]
    vmax_layer = [float(np.percentile(np.concatenate([raw_maps[k][i].ravel() for k in ok_keys]), 99.5)) for i in range(nl)]
    maps = {}
    for k, em in raw_maps.items():
        eq = np.mean([em[i] / vmax_layer[i] for i in range(nl)], axis=0)
        maps[k] = [em[nl], eq] + em[:nl]                              # ALL, EQ, layers
    vmax = [float(np.percentile(np.concatenate([maps[k][c].ravel() for k in ok_keys]), 99.5)) for c in range(len(names))]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for (g, p), ms in maps.items():
        row = [g, os.path.basename(p)]
        for m in ms:
            row += [float(np.percentile(m, 99)), float(m.max())]
        rows.append(row)
        if (g, p) not in raws:
            continue
        raw = raws[(g, p)]
        base = cv2.resize(raw, (W, H))
        bx = boxes.get((g, p), [])

        def draw(img, with_text=False):
            for name, x0, y0, x1, y1 in bx:
                a = (int(x0 / raw.shape[1] * W), int(y0 / raw.shape[0] * H))
                b = (int(x1 / raw.shape[1] * W), int(y1 / raw.shape[0] * H))
                cv2.rectangle(img, a, b, (255, 255, 255), 2)
                if with_text:
                    cv2.putText(img, name, (a[0], max(a[1] - 4, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return img

        cols = [label(draw(base.copy(), True), f"{g} | {os.path.basename(p)[:38]}")]
        for c, m in enumerate(ms):
            if args.images_only and c > 0:          # final JADE output (ALL) only, no per-layer columns
                break
            d = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            heat = cv2.applyColorMap((np.clip(d / vmax[c], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cols.append(label(draw(cv2.addWeighted(base, 0.5, heat, 0.5, 0)), f"{names[c]} (max {m.max():.2f}, scale {vmax[c]:.2f})"))
        os.makedirs(os.path.join(args.out_dir, g), exist_ok=True)
        cv2.imwrite(os.path.join(args.out_dir, g, os.path.splitext(os.path.basename(p))[0] + ".png"), np.concatenate(cols, axis=1))

    with open(os.path.join(args.out_dir, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "image"] + [f"{n}_{k}" for n in names for k in ("p99", "max")])
        w.writerows(rows)

    if args.images_only:
        print(f"[Eval] Done (images only). Pictures + scores.csv in: {args.out_dir}")
        return

    sel = lambda g, c, col: [r[2 + 2 * c + col] for r in rows if r[0] == g]
    print("\n[Eval] typical error size per layer on train pictures, and share of ALL:")
    means = [np.mean([em[i].mean() for (g, _), em in raw_maps.items() if g == "train"]) for i in range(nl)]
    for i in range(nl):
        print(f"   L{jade.extract_layers[i]:>2d}: {means[i]:9.4f}   share of ALL {100 * means[i] / sum(means):5.1f}%")

    print("\n[Eval] mean p99 error (scale = 99.5th pct of val OK):")
    for c, n in enumerate(names):
        print(f"   {n:>4s}: " + " | ".join(f"{g} {np.mean(sel(g, c, 0)):.3f}" for g in groups) + f"   (scale {vmax[c]:.3f})")

    print("\n[Eval] OK vs NG separation (0.5 = none, 1.0 = perfect)")
    print("        " + "  ".join(f"{s} p99 / max".rjust(24) for s in ("VAL", "TEST")))
    for c, n in enumerate(names):
        parts = []
        for s in ("val", "test"):
            parts.append(f"{auc(sel(s + '_ok', c, 0), sel(s + '_ng', c, 0)):.3f} / {auc(sel(s + '_ok', c, 1), sel(s + '_ng', c, 1)):.3f}".rjust(24))
        print(f"   {n:>4s}: " + "  ".join(parts))

    print("\n[Eval] location check: NG pictures where the hottest patch is INSIDE a labelled defect box")
    for g in ("val_ng", "test_ng"):
        hits = np.zeros(len(names))
        chance, cnt = 0.0, 0
        for (gg, p), ms in maps.items():
            if gg != g or not boxes.get((gg, p)):
                continue
            raw_shape = (1040, 1660)
            bm = box_mask(boxes[(gg, p)], raw_shape, grid)
            cnt += 1
            chance += bm.mean()
            for c, m in enumerate(ms):
                hits[c] += bm.flat[int(m.argmax())]
        if cnt:
            print(f"   {g} ({cnt} pictures, by chance about {100 * chance / cnt:.0f}%): " +
                  "  ".join(f"{n} {100 * h / cnt:.0f}%" for n, h in zip(names, hits)))
    print(f"[Eval] Done. Pictures + scores.csv in: {args.out_dir}")


if __name__ == "__main__":
    main()
