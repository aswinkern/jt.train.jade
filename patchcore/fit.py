"""
PatchCore "training" (called by run.sh): builds the memory bank from the OK pictures (data_dir/images) and shows the result.

  1. the patch features of all OK pictures are collected
  2. a greedy k-center coreset of them is kept (model.memory.coreset_ratio) = the memory bank, saved in checkpoints/memory_bank.pt
  3. output.calibration_pictures OK pictures (spread over the folder) are kept OUT of the bank. With the NG pictures (data_dir/NG, never used
     for the bank) they are scored and drawn into results/ and into TensorBoard, and they set the colours:
       heatmap_floor            = the error below which 9 of 10 pixels of these normal pictures are: normal parts become dark blue
       heatmap_scaling_factor   = the error above which only 1 of 100 pixels of these pictures and the NG pictures is: dark red
     Both are written into the run's config.yaml (and run.sh copies them into the config.yaml next to it), so --infer draws the same way.
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                          # patchcore/common.py and patchcore/model.py
sys.path.append(os.path.dirname(HERE))                            # the repo root: tools/
import common                                                    # noqa: E402
from model import PatchCore, greedy_coreset, random_projection   # noqa: E402
from tools.set_config_value import set_model_value               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="PatchCore: build the memory bank")
    ap.add_argument("-c", "--config", default="./config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Fit] Using device: %s" % device)
    t_start = time.time()

    size = tuple(cfg["data"]["image_size"])                        # (height, width)
    exts = cfg["data"]["image_extensions"]
    batch, workers = cfg["data"]["batch_size"], cfg["data"]["num_workers"]
    root = cfg["data"]["data_dir"]
    ok_paths = common.find_images(os.path.join(root, "images"), exts)
    ng_paths = common.find_images(os.path.join(root, "NG"), exts)
    if not ok_paths:
        raise SystemExit("[Fit] no OK pictures in %s" % os.path.join(root, "images"))
    mem = cfg["model"]["memory"]
    out = cfg["output"]
    n_cal = int(out.get("calibration_pictures", 0))
    cal_paths = common.spread(ok_paths, n_cal) if n_cal > 0 else []        # OK pictures kept out of the bank: they show what unseen normal looks like
    if cal_paths and len(ok_paths) - len(cal_paths) < max(10, 2 * len(cal_paths)):
        raise SystemExit("[Fit] output.calibration_pictures %d is too many for %d OK pictures (the bank needs the rest)" % (n_cal, len(ok_paths)))
    held_out = set(cal_paths)
    bank_paths = [p for p in ok_paths if p not in held_out]

    pc = PatchCore(cfg, device)
    print("[Fit] backbone %s, stages %s, neighbourhood %d, embed_dim %d | %d OK pictures for the bank (+ %d held out to set the colours) of %s" % (
        pc.type, pc.layers, pc.neigh, pc.embed_dim, len(bank_paths), len(cal_paths), list(size)))

    # 1. the patch features of all OK pictures (float16, on the CPU)
    feats, filled, t0, nb = None, 0, time.time(), 0
    for x in common.loader(bank_paths, size, batch, workers):
        f = pc.embed(x.to(device))
        grid = tuple(f.shape[1:3])
        f = f.reshape(-1, f.shape[-1]).half().cpu()
        if feats is None:
            feats = torch.empty((f.shape[0] // x.shape[0] * len(bank_paths), f.shape[1]), dtype=torch.float16)
        feats[filled:filled + f.shape[0]] = f
        filled += f.shape[0]
        nb += 1
        if nb % 20 == 0:
            print("[Fit] features of %d / %d pictures" % (min(nb * batch, len(bank_paths)), len(bank_paths)), flush=True)
    feats = feats[:filled]
    N = feats.shape[0]
    print("[Fit] %d patches (grid %dx%d per picture, %d numbers each) in %.0f s" % (N, grid[0], grid[1], feats.shape[1], time.time() - t0), flush=True)

    # 2. the coreset = the memory bank
    n_select = max(1, int(N * mem["coreset_ratio"]))
    print("[Fit] coreset: %d of %d patches (coreset_ratio %g)" % (n_select, N, mem["coreset_ratio"]), flush=True)
    proj = random_projection(feats, mem["projection_dim"], mem["seed"], device)
    sel = greedy_coreset(proj, n_select, mem["seed"])
    del proj
    bank = feats[sel]
    del feats
    ck_dir = out["checkpoint_dir"]
    os.makedirs(ck_dir, exist_ok=True)
    ck = os.path.join(ck_dir, "memory_bank.pt")
    meta = pc.meta(size)
    meta.update({"pictures": len(bank_paths), "patches": N, "coreset_ratio": mem["coreset_ratio"], "bank_size": int(bank.shape[0])})
    torch.save({"bank": bank, "meta": meta}, ck)
    print("[Fit] Saved: %s  (%d patches, %.0f MB)" % (ck, bank.shape[0], bank.numel() * 2 / 1e6), flush=True)

    # 3. the held-out OK pictures and the NG pictures: scores, pictures, and the colours
    pc.set_bank(bank)
    ok_show = cal_paths or common.spread(ok_paths, out.get("ok_pictures", 20))          # without held-out pictures: some training pictures
    ng_show = common.spread(ng_paths, out["ng_pictures"])
    maps_ok = common.error_maps(pc, ok_show, size, batch, workers, device)
    maps_ng = common.error_maps(pc, ng_show, size, batch, workers, device) if ng_show else {}
    if cal_paths:
        floor_v = common.global_p99(maps_ok.values(), 0.90)
        scale_v = common.global_p99(list(maps_ok.values()) + list(maps_ng.values()), 0.99)
        if not maps_ng:
            print("[Result] no NG pictures: the dark-red value comes from the OK pictures alone")
        if scale_v <= floor_v * 1.05:
            scale_v = floor_v * 1.25
        cfg["model"]["heatmap_scaling_factor"], cfg["model"]["heatmap_floor"] = round(scale_v, 4), round(floor_v, 4)
        set_model_value(a.config, "heatmap_scaling_factor", "%.4f" % scale_v)
        set_model_value(a.config, "heatmap_floor", "%.4f" % floor_v)
        print("[Result] normal pictures (%d held out of the bank): 9 of 10 pixels have an error below %.4f -> heatmap_floor (below it = dark blue)" % (len(cal_paths), floor_v))
        print("[Result] pixels of the normal and NG pictures: only 1 of 100 has an error above %.4f -> heatmap_scaling_factor (dark red)" % scale_v)
        print("[Fit] heatmap_floor %.4f and heatmap_scaling_factor %.4f written to %s" % (floor_v, scale_v, a.config))
    else:
        print("[Result] output.calibration_pictures is 0: heatmap_floor and heatmap_scaling_factor are taken from the config as they are")
    colour_max, thr, floor = common.colour_scale({**maps_ok, **maps_ng}, cfg["model"], a.config)
    mode = common.visual_mode(cfg["model"])
    res = out["results_dir"]
    rows_ok = common.save_results(maps_ok, os.path.join(root, "images"), os.path.join(res, "OK"), colour_max, thr, floor, size, mode)
    rows_ng = common.save_results(maps_ng, os.path.join(root, "NG"), os.path.join(res, "NG"), colour_max, thr, floor, size, mode) if maps_ng else []

    def stats(rows):
        v = np.array([r[1] for r in rows])
        return "median %.3f | max %.3f" % (np.median(v), v.max()) if len(v) else "none"
    print("[Result] score (largest error of a picture) of %d OK pictures: %s" % (len(rows_ok), stats(rows_ok)))
    print("[Result] score of %d NG pictures: %s" % (len(rows_ng), stats(rows_ng)))
    print("[Result] pictures and scores.csv in %s" % res)

    try:                                                           # the same pictures in TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        w = SummaryWriter(out["tensorboard_dir"])
        for tag, maps, n in (("OK", maps_ok, 5), ("NG", maps_ng, 10)):
            for i, (p, m) in enumerate(list(maps.items())[:n]):
                img = cv2.cvtColor(common.picture(p, m, colour_max, thr, floor, size, None, mode), cv2.COLOR_BGR2RGB)
                w.add_image("%s/image_%d" % (tag, i), torch.from_numpy(img).permute(2, 0, 1).float() / 255.0, 0)
        w.close()
    except Exception as e:
        print("[Fit] TensorBoard pictures not written: %s" % e)
    print("[Fit] Done in %.0f s." % (time.time() - t_start))


if __name__ == "__main__":
    main()
