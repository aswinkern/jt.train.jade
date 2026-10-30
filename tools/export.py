"""
ONNX export of a trained JADE_v3 for Triton / TensorRT.

The exported model is RAW: picture in, error map out. Baked into it: the backbone, the autoencoders, layer_weights and
featurewise_threshold (read from the config at export time). NOT in it: heatmap_scaling_factor, pixelwise_threshold and area_threshold -
they are applied by the client (inference/postprocess.py, the same code as `run.sh --infer`) with the variables at the top of
triton/triton_client.py (the export prints them, ready to paste), so they can be changed any time without exporting again.

The batch size is dynamic (the ONNX takes 1 to any number of pictures); --max_batch is the largest batch (max_batch_size of config.pbtxt).

Writes, in <export.models_dir>/<export.model_name>/Serving_model/ (the layout of the other models of the customer; --out replaces
<models_dir>/<model_name>):
  1/model.onnx + model.onnx.data   the model (picture (B,3,H,W) -> error_map (B,H,W))
  config.pbtxt                     Triton model configuration (the engine model.plan goes into 1/)
An existing model of that name is never overwritten (give a new export.model_name, or --overwrite).
and checks the ONNX against the PyTorch model on real OK pictures (batch of 1 and batch of N).
"""
import os
import re
import sys
import glob
import argparse

import cv2
import yaml
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import model as md
from data import data_handler as dh
from inference.postprocess import MEAN, STD

DEFAULT_MODEL_NAME = "jade_v3"        # used when export.model_name is not in the config


class ExportWrapper(torch.nn.Module):
    """Thin wrapper so torch.onnx.export uses forward_export (pure PyTorch, no numpy)."""
    def __init__(self, jade):
        super().__init__()
        self.jade = jade

    def forward(self, x):
        return self.jade.forward_export(x)


def main():
    ap = argparse.ArgumentParser(description="JADE_v3 ONNX export")
    ap.add_argument("-c", "--config", default="./config.yaml", help="settings: export.model_name / experiment / epoch, layer_weights, featurewise_threshold (baked in) and the three thresholds (printed for triton_client.py)")
    ap.add_argument("--checkpoint", default=None, help="epoch_N.pt to export (default: export.experiment and export.epoch of the config)")
    ap.add_argument("--model_config", default=None, help="model definition: default the config.yaml of the run that owns the checkpoint (<checkpoint dir>/../config.yaml)")
    ap.add_argument("--out", default=None, help="model folder; default <export.models_dir>/<export.model_name>")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing model of the same name")
    ap.add_argument("--max_batch", type=int, default=16, help="largest batch: max_batch_size of config.pbtxt (default 16)")
    ap.add_argument("--check", type=int, default=4, help="OK pictures used to check the ONNX against PyTorch (0 = no check)")
    a = ap.parse_args()

    bmax = a.max_batch
    if bmax < 1:
        raise SystemExit("[Export] --max_batch must be 1 or more, got %d" % bmax)
    st = yaml.load(open(a.config), Loader=yaml.FullLoader)
    ex = st.get("export") or {}
    MODEL_NAME = str(ex.get("model_name") or DEFAULT_MODEL_NAME)
    if a.out is None and not ex.get("models_dir"):
        raise SystemExit("[Export] set export.models_dir in %s (the models folder of the customer), or give --out" % a.config)
    model_root = a.out or os.path.join(str(ex["models_dir"]), MODEL_NAME)
    model_dir = os.path.join(model_root, "Serving_model")          # models/<model_name>/Serving_model/{config.pbtxt, 1/}
    ver_dir = os.path.join(model_dir, "1")
    final_onnx = os.path.join(ver_dir, "model.onnx")
    if os.path.exists(final_onnx) and not a.overwrite:
        raise SystemExit("[Export] %s already exists. A model is not overwritten: change export.model_name (a new version), or use --overwrite" % final_onnx)
    if a.checkpoint is None:                                   # export.experiment + export.epoch of the config, next to the config file
        if not ex.get("experiment") or ex.get("epoch") is None:
            raise SystemExit("[Export] give --checkpoint, or set export.experiment and export.epoch in %s" % a.config)
        ck_dir = os.path.join(os.path.dirname(os.path.abspath(a.config)), str(ex["experiment"]), "checkpoints")
        a.checkpoint = os.path.join(ck_dir, "epoch_%s.pt" % ex["epoch"])
        if not os.path.isfile(a.checkpoint):
            have = sorted((os.path.basename(f) for f in glob.glob(os.path.join(ck_dir, "epoch_*.pt"))), key=lambda n: int(re.findall(r"\d+", n)[0]))
            raise SystemExit("[Export] checkpoint not found: %s\n[Export] epochs saved there: %s" % (a.checkpoint, ", ".join(have) or "none (is export.experiment right?)"))
    model_cfg = a.model_config or os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), "..", "config.yaml")
    cfg = yaml.load(open(model_cfg), Loader=yaml.FullLoader)
    for key in ("layer_weights", "featurewise_threshold", "heatmap_scaling_factor", "pixelwise_threshold", "area_threshold"):
        if key in (st.get("model") or {}):
            cfg["model"][key] = st["model"][key]
        else:
            cfg["model"].pop(key, None)
    print("[Export] model name: %s | checkpoint: %s" % (MODEL_NAME, a.checkpoint))
    print("[Export] model definition: %s | settings: %s" % (os.path.normpath(model_cfg), a.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Export] device: %s" % device)
    jade = md.JADE_v3(cfg).to(device)
    jade.load_checkpoint(a.checkpoint)
    jade.eval()
    H, W = cfg["data"]["image_size"]
    print("[Export] layer_weights (baked in, normalised) = %s | featurewise_threshold (baked in) = %g" % (
        [round(w, 4) for w in jade.layer_weights], jade.featurewise_threshold))

    os.makedirs(ver_dir, exist_ok=True)
    tmp_onnx = os.path.join(ver_dir, "_tmp.onnx")

    wrapper = ExportWrapper(jade).to(device).eval()
    dummy = torch.zeros(2, 3, H, W, device=device)            # batch 2 (a batch of 1 can be fixed into the graph by the tracer)
    print("[Export] input (B, 3, %d, %d) -> output error_map (B, %d, %d), B dynamic. Exporting ONNX..." % (H, W, H, W))
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)               # trace first: decomposes LayerNorm into ops every TensorRT version supports
    torch.onnx.export(traced, dummy, tmp_onnx, opset_version=16, input_names=["image"], output_names=["error_map"],
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

    # ── check against PyTorch on real OK pictures, batch of N and batch of 1 ──────────────────────────────
    if a.check > 0:
        import onnxruntime as ort
        exts = tuple(e.replace("*", "").lower() for e in cfg["data"]["image_extensions"])
        pics = sorted(os.path.join(d, f) for d, _, fs in os.walk(os.path.join(cfg["data"]["data_dir"], "images")) for f in fs if f.lower().endswith(exts))
        pics = pics[::max(1, len(pics) // a.check)][:a.check]
        if not pics:
            print("[Export] check skipped: no OK pictures in %s" % os.path.join(cfg["data"]["data_dir"], "images"))
        else:
            x = torch.cat([dh.DataHandler.preprocess_image(cv2.imread(p), (W, H)) for p in pics])
            with torch.no_grad():
                ref = wrapper(x.to(device)).float().cpu().numpy()                    # PyTorch, whole batch
            sess = ort.InferenceSession(final_onnx, providers=["CPUExecutionProvider"])
            out_n = sess.run(["error_map"], {"image": x.numpy()})[0]                 # ONNX, batch of N
            out_1 = np.concatenate([sess.run(["error_map"], {"image": x[i:i + 1].numpy()})[0] for i in range(len(pics))])   # ONNX, one by one
            scale = float(np.abs(ref).max())
            d_n, d_1 = float(np.abs(out_n - ref).max()), float(np.abs(out_1 - ref).max())
            print("[Export] check on %d OK pictures (error values up to %.4f):" % (len(pics), scale))
            print("[Export]   ONNX batch of %d  vs PyTorch: largest difference %.2e (%.4f %% of the largest value)" % (len(pics), d_n, 100 * d_n / scale))
            print("[Export]   ONNX batch of 1   vs PyTorch: largest difference %.2e (%.4f %% of the largest value)" % (d_1, 100 * d_1 / scale))
            ok = out_n.shape == (len(pics), H, W) and d_n / scale < 1e-3 and d_1 / scale < 1e-3
            print("[Export] check: %s" % ("PASSED" if ok else "FAILED (do not use this export)"))

    # ── Triton config and TensorRT command ──────────────────────────────────────────────────────────────
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:               # same style as the customer's other models and yolonas
        f.write('name: "%s"\nplatform: "tensorrt_plan"\nmax_batch_size: %d\n\n' % (MODEL_NAME, bmax))
        f.write('input [\n  {\n    name: "image"\n    data_type: TYPE_FP32\n    format: FORMAT_NCHW\n    dims: [3, %d, %d]\n  }\n]\n\n' % (H, W))
        f.write('output [\n  {\n    name: "error_map"\n    data_type: TYPE_FP32\n    dims: [%d, %d]\n  }\n]\n\n' % (H, W))
        f.write('instance_group [{ kind: KIND_GPU }]\n')
    print("[Export] Done. Model folder: %s" % model_root)
    print("[Export] Variables for the top of triton/triton_client.py (the thresholds can be changed there any time, no new export):")
    print('  MODEL_NAME = "%s"' % MODEL_NAME)
    print("  IMAGE_SIZE = (%d, %d)        # height, width" % (H, W))
    print("  MAX_BATCH = %d" % bmax)
    print("  HEATMAP_SCALING_FACTOR = %s" % cfg["model"].get("heatmap_scaling_factor"))
    print("  PIXELWISE_THRESHOLD = %s" % cfg["model"].get("pixelwise_threshold"))
    print("  AREA_THRESHOLD = %g" % float(cfg["model"].get("area_threshold") or 0))
    print("[Export] The model folder is ready (largest batch %d). Build the TensorRT engine with your trtexec command: Serving_model/1/ then holds model.plan." % bmax)
    print("[Export] The .dvc file of the model folder is not touched: run dvc add yourself.")


if __name__ == "__main__":
    main()
