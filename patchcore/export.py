"""
ONNX export of a PatchCore memory bank for Triton / TensorRT (called by `run.sh --export`).

The exported model is RAW: picture (B,3,H,W) -> error map (B,H,W), the same numbers as `run.sh --infer` before the colours. Baked into it: the
backbone, the patch embedding, the memory bank (the nearest-neighbour search, in chunks of --chunk bank patches), the enlarging to the picture
size and the blur. NOT in it: heatmap_scaling_factor, heatmap_floor, pixelwise_threshold, min_box_score - they are applied by the client
(patchcore/render.py, the same code as `run.sh --infer`) with the variables at the top of triton/patchcore_client.py (the export prints them,
ready to paste), so they change any time without exporting again.

The memory bank is part of the model: a new fit is a new export (a new export.model_name). The bank is saved as float32 and centred on its mean (the distances do not change). The TensorRT engine must be
built WITHOUT --fp16 and with --noTF32: the distances are calculated as |q|^2 - 2 q.b + |b|^2, which loses its precision in fp16 and TF32.

Writes, in <export.models_dir>/<export.model_name>/Serving_model/ (--out replaces <models_dir>/<model_name>):
  1/model.onnx + model.onnx.data   the model (picture (B,3,H,W) -> error_map (B,H,W))
  config.pbtxt                     Triton model configuration (the engine model.plan goes into 1/)
An existing model of that name is never overwritten (give a new export.model_name, or --overwrite).
It checks the ONNX against the PyTorch model (the same code as --infer) on OK pictures and on NG pictures, batch of N and batch of 1.
Differences up to about 0.3 % of the largest error are normal and do not matter: the GPU convolutions of --infer run in TF32 (ONNX Runtime on the CPU
in full FP32 is within 0.02 % of the exact value), and what counts is that the same pictures get the same boxes (see convert.md).
"""
import os
import sys
import glob
import argparse

import cv2
import yaml
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                          # patchcore/common.py and patchcore/model.py
sys.path.append(os.path.dirname(HERE))                            # the repo root: tools/, inference/
import common                           # noqa: E402
from model import PatchCore, gaussian_blur   # noqa: E402

DEFAULT_MODEL_NAME = "patchcore"


def pool_matrix(c, d):
    """(c, d) matrix P with  x @ P == F.adaptive_avg_pool1d(x, d)  for x of c numbers (the pooling is linear). A matrix product exports to ONNX
    and TensorRT for any c and d, while the adaptive pooling does not when c and d do not divide."""
    return F.adaptive_avg_pool1d(torch.eye(c).unsqueeze(1), d).squeeze(1)


class ExportNet(torch.nn.Module):
    """PatchCore.error_map as pure tensor operations with a dynamic batch (nothing about the batch size is fixed inside)."""
    def __init__(self, pc, size_hw, chunk):
        super().__init__()
        self.net = pc.net
        self.layers, self.neigh, self.sigma, self.size = pc.layers, pc.neigh, pc.sigma, tuple(size_hw)
        D = pc.embed_dim
        with torch.no_grad():                                  # channels of the chosen stages
            n, y, ch = pc.net, torch.zeros(1, 3, 64, 64, device=pc.device), {}
            y = n.maxpool(n.relu(n.bn1(n.conv1(y))))
            for i, stage in enumerate([n.layer1, n.layer2, n.layer3, n.layer4], start=1):
                y = stage(y)
                ch[i] = y.shape[1]
        for i in self.layers:
            self.register_buffer("pool%d" % i, pool_matrix(ch[i], D).to(pc.device))                       # stage channels -> embed_dim
        self.register_buffer("pool_all", pool_matrix(len(self.layers) * D, D).to(pc.device))              # all stages together -> embed_dim
        bank = pc.bank.float()
        mu = bank.mean(0)                                      # the distances do not change when q and b are both shifted by mu, but |q - mu|^2 and |b - mu|^2 are
        self.register_buffer("mu", mu)                          # much smaller than |q|^2 and |b|^2: less cancellation in |q|^2 - 2 q.b + |b|^2 (matters in ONNX Runtime and TensorRT)
        bank = bank - mu
        self.nchunks = 0
        for s in range(0, bank.shape[0], chunk):
            b = bank[s:s + chunk]
            self.register_buffer("bt%d" % self.nchunks, b.t().contiguous())                               # (D, n)
            self.register_buffer("bb%d" % self.nchunks, (b * b).sum(1))                                   # |b|^2
            self.nchunks += 1

    def forward(self, x):
        n = self.net
        y = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        got = {}
        for i, stage in enumerate([n.layer1, n.layer2, n.layer3, n.layer4], start=1):
            y = stage(y)
            if i in self.layers:
                got[i] = y
            if i >= max(self.layers):
                break
        maps = [F.avg_pool2d(got[i], self.neigh, stride=1, padding=self.neigh // 2) for i in self.layers]
        h, w = max((m.shape[-2:] for m in maps), key=lambda s: s[0] * s[1])
        maps = [m if tuple(m.shape[-2:]) == (h, w) else F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False) for m in maps]
        parts = [m.permute(0, 2, 3, 1).reshape(-1, m.shape[1]) @ getattr(self, "pool%d" % i) for m, i in zip(maps, self.layers)]    # (B*h*w, embed_dim)
        f = torch.cat(parts, 1) @ self.pool_all - self.mu                                                # (B*h*w, embed_dim), centred on the bank mean
        best = None
        for k in range(self.nchunks):                                                                    # min over the bank of |b|^2 - 2 q.b
            v = torch.amin(getattr(self, "bb%d" % k).unsqueeze(0) - 2.0 * (f @ getattr(self, "bt%d" % k)), dim=1)
            best = v if best is None else torch.minimum(best, v)
        d = torch.sqrt(torch.clamp((f * f).sum(1) + best, min=0.0))                                      # distance to the nearest bank patch
        m = F.interpolate(d.reshape(-1, 1, h, w), size=self.size, mode="bilinear", align_corners=False)
        return gaussian_blur(m, self.sigma).reshape(-1, self.size[0], self.size[1])                      # reshape, not squeeze: squeeze exports as an If node TensorRT cannot parse


def main():
    ap = argparse.ArgumentParser(description="PatchCore ONNX export")
    ap.add_argument("-c", "--config", default="./config.yaml", help="settings: export.model_name / experiment and the colour settings (printed for patchcore_client.py)")
    ap.add_argument("--checkpoint", default=None, help="memory_bank.pt to export (default: export.experiment of the config, 'latest' = the newest patchcore_* run folder next to it)")
    ap.add_argument("--model_config", default=None, help="model definition: default the config.yaml of the run that owns the checkpoint (<checkpoint dir>/../config.yaml)")
    ap.add_argument("--out", default=None, help="model folder; default <export.models_dir>/<export.model_name>")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing model of the same name")
    ap.add_argument("--max_batch", type=int, default=1, help="largest batch: max_batch_size of config.pbtxt (default 1: TensorRT keeps the whole distance table of the search in memory, 4.2 GB per picture at 518 x 518, and one picture already keeps the GPU busy; ONNX Runtime can take more)")
    ap.add_argument("--chunk", type=int, default=4096, help="bank patches per matrix product (default 4096: with batch 8 about 0.5 GB per product)")
    ap.add_argument("--check", type=int, default=2, help="OK pictures and NG pictures each used to check the ONNX against PyTorch (0 = no check)")
    a = ap.parse_args()
    if a.max_batch < 1 or a.chunk < 1:
        raise SystemExit("[Export] --max_batch and --chunk must be 1 or more")

    st = yaml.safe_load(open(a.config))
    ex = st.get("export") or {}
    model_name = str(ex.get("model_name") or DEFAULT_MODEL_NAME)
    if a.out is None and not ex.get("models_dir"):
        raise SystemExit("[Export] set export.models_dir in %s (the models folder of the customer), or give --out" % a.config)
    model_root = a.out or os.path.join(str(ex["models_dir"]), model_name)
    model_dir = os.path.join(model_root, "Serving_model")
    ver_dir = os.path.join(model_dir, "1")
    final_onnx = os.path.join(ver_dir, "model.onnx")
    if os.path.exists(final_onnx) and not a.overwrite:
        raise SystemExit("[Export] %s already exists. A model is not overwritten: change export.model_name (a new version), or use --overwrite" % final_onnx)
    if a.checkpoint is None:
        experiment = str(ex.get("experiment") or "latest")
        base = os.path.dirname(os.path.abspath(a.config))
        if experiment == "latest":
            runs = [d for d in glob.glob(os.path.join(base, "patchcore_*")) if os.path.isfile(os.path.join(d, "checkpoints", "memory_bank.pt"))]
            if not runs:
                raise SystemExit("[Export] export.experiment is latest, but no patchcore_* run folder with checkpoints/memory_bank.pt is next to %s" % a.config)
            experiment = os.path.basename(max(runs, key=os.path.getmtime))
            print("[Export] latest run folder: %s" % experiment)
        a.checkpoint = os.path.join(base, experiment, "checkpoints", "memory_bank.pt")
        if not os.path.isfile(a.checkpoint):
            raise SystemExit("[Export] memory bank not found: %s (is export.experiment right?)" % a.checkpoint)
    model_cfg = a.model_config or os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), "..", "config.yaml")
    cfg = yaml.safe_load(open(model_cfg))
    size = tuple(cfg["data"]["image_size"])
    H, W = size
    print("[Export] model name: %s | memory bank: %s" % (model_name, a.checkpoint))
    print("[Export] model definition: %s | settings: %s" % (os.path.normpath(model_cfg), a.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu")
    pc = PatchCore(cfg, device)
    for k, v in pc.meta(size).items():
        if ck["meta"].get(k) != v:
            raise SystemExit("[Export] the memory bank was made with %s = %s, the config says %s" % (k, ck["meta"].get(k), v))
    pc.set_bank(ck["bank"])
    net = ExportNet(pc, size, a.chunk).to(device).eval()
    N, D = pc.bank.shape
    print("[Export] device %s | memory bank %d patches x %d (%.2f GB as float32) in %d chunks of %d | input (B, 3, %d, %d) -> error_map (B, %d, %d), B dynamic"
          % (device, N, D, N * D * 4 / 1e9, net.nchunks, a.chunk, H, W, H, W))

    os.makedirs(ver_dir, exist_ok=True)
    tmp_onnx = os.path.join(ver_dir, "_tmp.onnx")
    dummy = torch.zeros(2, 3, H, W, device=device)            # batch 2 (a batch of 1 can be fixed into the graph by the tracer)
    print("[Export] Exporting ONNX...", flush=True)
    with torch.no_grad():
        torch.onnx.export(net, dummy, tmp_onnx, opset_version=16, input_names=["image"], output_names=["error_map"],
                          dynamic_axes={"image": {0: "batch"}, "error_map": {0: "batch"}}, do_constant_folding=True, dynamo=False)
    import onnx
    m = onnx.load(tmp_onnx)
    onnx.checker.check_model(m)
    print("[Export] onnx.checker.check_model: PASSED")
    for f in (final_onnx, final_onnx + ".data"):
        if os.path.exists(f):
            os.remove(f)
    onnx.save_model(m, final_onnx, save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx.data", size_threshold=1024)
    os.remove(tmp_onnx)
    um = os.umask(0); os.umask(um)
    os.chmod(final_onnx + ".data", 0o666 & ~um)               # onnx writes the data file as 600 whatever the umask: same permissions as your own files
    print("[Export] saved %s and %s (%.0f MB)" % (final_onnx, final_onnx + ".data", os.path.getsize(final_onnx + ".data") / 1e6))

    # ── check against PyTorch (the code of --infer) on OK and on NG pictures, batch of N and batch of 1 ─────────────────────
    if a.check > 0:
        import onnxruntime as ort
        root = cfg["data"]["data_dir"]
        ok_pics = common.spread(common.find_images(os.path.join(root, "images"), cfg["data"]["image_extensions"]), a.check)
        ng_pics = common.spread(common.find_images(os.path.join(root, "NG"), cfg["data"]["image_extensions"]), a.check)
        pics = ok_pics + ng_pics
        if not pics:
            print("[Export] check skipped: no pictures in %s" % root)
        else:
            x = next(iter(common.loader(pics, size, len(pics), 0)))
            with torch.no_grad():
                ref = pc.error_map(x.to(device), size).cpu().numpy()                 # PyTorch, whole batch (run.sh --infer)
                own = net(x.to(device)).cpu().numpy()                               # the exported network in PyTorch
            sess = ort.InferenceSession(final_onnx, providers=[p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()])
            print("[Export] ONNX Runtime: %s" % sess.get_providers()[0])
            out_n = sess.run(["error_map"], {"image": x.numpy()})[0]                 # ONNX, batch of N
            out_1 = np.concatenate([sess.run(["error_map"], {"image": x[i:i + 1].numpy()})[0] for i in range(len(pics))])   # ONNX, one by one
            scale = float(np.abs(ref).max())
            d = {"export network (PyTorch)": np.abs(own - ref).max(), "ONNX batch of %d" % len(pics): np.abs(out_n - ref).max(), "ONNX batch of 1": np.abs(out_1 - ref).max()}
            print("[Export] check on %d pictures (%d OK, %d NG; error values up to %.4f), against PyTorch of --infer:" % (len(pics), len(ok_pics), len(ng_pics), scale))
            for k, v in d.items():
                print("[Export]   %-26s largest difference %.2e (%.4f %% of the largest value)" % (k, v, 100 * v / scale))
            ok = out_n.shape == (len(pics), H, W) and all(v / scale < 5e-3 for v in d.values())
            print("[Export] check: %s" % ("PASSED" if ok else "FAILED (do not use this export)"))

    # ── Triton config and TensorRT command ──────────────────────────────────────────────────────────────
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
        f.write('name: "%s"\nplatform: "tensorrt_plan"\nmax_batch_size: %d\n\n' % (model_name, a.max_batch))
        f.write('input [\n  {\n    name: "image"\n    data_type: TYPE_FP32\n    format: FORMAT_NCHW\n    dims: [3, %d, %d]\n  }\n]\n\n' % (H, W))
        f.write('output [\n  {\n    name: "error_map"\n    data_type: TYPE_FP32\n    dims: [%d, %d]\n  }\n]\n\n' % (H, W))
        f.write('instance_group [{ kind: KIND_GPU }]\n')
    sm = st.get("model") or {}
    print("[Export] Done. Model folder: %s" % model_root)
    print("[Export] Variables for the top of triton/patchcore_client.py (the colour settings can be changed there any time, no new export):")
    print('  MODEL_NAME = "%s"' % model_name)
    print("  IMAGE_SIZE = (%d, %d)        # height, width" % (H, W))
    print("  MAX_BATCH = %d" % a.max_batch)
    print("  HEATMAP_SCALING_FACTOR = %s" % sm.get("heatmap_scaling_factor"))
    print("  HEATMAP_FLOOR = %s" % sm.get("heatmap_floor"))
    print("  PIXELWISE_THRESHOLD = %s" % sm.get("pixelwise_threshold"))
    print("  MIN_BOX_SCORE = %s" % sm.get("min_box_score"))
    print("[Export] Build the TensorRT engine with convert.md (PatchCore: WITHOUT --fp16), Serving_model/1/ then holds model.plan.")


if __name__ == "__main__":
    main()
