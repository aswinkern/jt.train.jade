"""
Look at a trained JADE model: per-layer and combined error maps on train / val OK / val NG pictures.

For every picture one image is written:
  [ original | ALL layers combined (what JADE uses) | layer a | layer b | layer c | layer d ]
Colours: dark blue = rebuilt well (normal) ... dark red = rebuilt badly (unusual).
Each column has ONE fixed scale for all pictures: the 99.5th percentile of the error on held-out
OK validation pictures. So red on a good picture is a false alarm, and every picture is comparable.

scores.csv: one row per picture. The end of the run prints the group means, how much each layer
contributes to the combined map, and how well NG separates from OK (AUC: 0.5 none, 1.0 perfect),
for each layer alone and for all layers combined.
"""
import os
import sys
import csv
import glob
import argparse

import cv2
import yaml
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(HERE)
from data import data_handler as dh   # noqa: E402
from model import model as md          # noqa: E402


def list_images(directory, exts):
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(directory, e))
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No images in {directory}")
    return paths


def spread(paths, n):
    if n >= len(paths):
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    return [paths[i] for i in idx]


def auc(ok, ng):
    ok = np.asarray(ok)[None, :]
    ng = np.asarray(ng)[:, None]
    return float(((ng > ok).sum() + 0.5 * (ng == ok).sum()) / (ok.size * ng.size))


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def error_maps(jade, raw, image_size, grid, device):
    """List of (gh, gw) error maps: one per layer, then the combined one (weighted by layer_weights, equal by default)."""
    t = dh.DataHandler.preprocess_image(raw, image_size).to(device)
    feats = jade.extract_features(t)
    errs = []
    for i, f in enumerate(feats):
        flat = f.reshape(-1, f.shape[-1])
        rec = jade.memories[i](flat)
        errs.append(jade.patch_error(flat, rec).reshape(grid).cpu().numpy())
    return errs + [np.tensordot(np.array(jade.layer_weights), np.stack(errs), axes=1)]


@torch.no_grad()
def error_maps_batch(jade, raws, image_size, grid, device):
    """Several pictures in one go. For each picture (raw, featurewise): raw = list of (gh, gw) error maps, one per layer,
    with no threshold at all; featurewise = the same maps with model.featurewise_threshold applied (None when it is off)."""
    t = torch.cat([dh.DataHandler.preprocess_image(r, image_size) for r in raws]).to(device)
    feats = jade.extract_features(t)                                   # per layer (B, N, C)
    thr = jade.featurewise_threshold
    raw_l, fw_l = [], []
    for i, f in enumerate(feats):
        B = f.shape[0]
        flat = f.reshape(-1, f.shape[-1])
        d = (flat - jade.memories[i](flat)) ** 2
        raw_l.append(d.mean(dim=1).reshape(B, *grid).cpu().numpy())
        if thr > 0:
            fw_l.append(torch.where(d > thr, d, torch.zeros_like(d)).mean(dim=1).reshape(B, *grid).cpu().numpy())
    return [([m[b] for m in raw_l], [m[b] for m in fw_l] if fw_l else None) for b in range(len(raws))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_ok_dir", required=True)
    ap.add_argument("--val_ng_dir", required=True)
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("--show_train", type=int, default=8)
    ap.add_argument("--show_ok", type=int, default=16)
    ap.add_argument("--show_ng", type=int, default=50)
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
    names = [f"L{l}" for l in jade.extract_layers] + ["ALL"]
    nm = len(names)

    groups = {
        "train": list_images(args.train_dir, exts),
        "val_ok": list_images(args.val_ok_dir, exts),
        "val_ng": list_images(args.val_ng_dir, exts),
    }
    show = {"train": args.show_train, "val_ok": args.show_ok, "val_ng": args.show_ng}
    shown = {g: set(spread(p, show[g])) for g, p in groups.items()}

    maps, raws = {}, {}
    for g, paths in groups.items():
        for p in paths:
            raw = cv2.imread(p)
            if raw is None:
                continue
            maps[(g, p)] = error_maps(jade, raw, image_size, grid, device)
            if p in shown[g]:
                raws[(g, p)] = raw
        print(f"[Eval] {g}: {len(paths)} pictures done")

    # one fixed scale per column: 99.5th percentile of held-out OK validation errors
    vmax = [float(np.percentile(np.concatenate([m[k].ravel() for (g, _), m in maps.items() if g == "val_ok"]), 99.5))
            for k in range(nm)]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for (g, p), ms in maps.items():
        row = [g, os.path.basename(p)]
        for m in ms:
            row += [float(np.percentile(m, 99)), float(m.max())]
        rows.append(row)
        if (g, p) in raws:
            raw = raws[(g, p)]
            shown_img = cv2.resize(raw, (W, H))
            cols = [label(shown_img.copy(), f"{g} | {os.path.basename(p)[:40]}")]
            for k, m in enumerate(ms):
                d = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                heat = cv2.applyColorMap((np.clip(d / vmax[k], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cols.append(label(cv2.addWeighted(shown_img, 0.5, heat, 0.5, 0),
                                  f"{names[k]} (max {m.max():.2f}, scale {vmax[k]:.2f})"))
            os.makedirs(os.path.join(args.out_dir, g), exist_ok=True)
            cv2.imwrite(os.path.join(args.out_dir, g, os.path.splitext(os.path.basename(p))[0] + ".png"),
                        np.concatenate(cols, axis=1))

    with open(os.path.join(args.out_dir, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "image"] + [f"{n}_{k}" for n in names for k in ("p99", "max")])
        w.writerows(rows)

    sel = lambda g, k, col: [r[2 + 2 * k + col] for r in rows if r[0] == g]
    print("\n[Eval] typical error size per layer on train pictures (mean over patches):")
    means = [np.mean([m[k].mean() for (g, _), m in maps.items() if g == "train"]) for k in range(nm - 1)]
    for k in range(nm - 1):
        print(f"   {names[k]:>4s}: {means[k]:9.4f}   share of the combined map: {100 * means[k] / sum(means):5.1f}%")

    print("\n[Eval] mean p99 error per group (same scale for every picture inside a column):")
    for k in range(nm):
        print(f"   {names[k]:>4s}: " + " | ".join(f"{g} {np.mean(sel(g, k, 0)):.3f}" for g in groups) + f"   (scale {vmax[k]:.3f})")

    print("\n[Eval] how well NG separates from OK on val (0.5 = none, 1.0 = perfect):")
    for k in range(nm):
        print(f"   {names[k]:>4s}: AUC(p99)={auc(sel('val_ok', k, 0), sel('val_ng', k, 0)):.3f}  "
              f"AUC(max)={auc(sel('val_ok', k, 1), sel('val_ng', k, 1)):.3f}")
    print(f"[Eval] Done. Pictures + scores.csv in: {args.out_dir}")


if __name__ == "__main__":
    main()
