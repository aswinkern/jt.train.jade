import os
import sys
import yaml
import torch
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import model as md


class ExportWrapper(torch.nn.Module):
    """Thin wrapper so torch.onnx.export uses forward_export (pure PyTorch, no numpy)."""
    def __init__(self, jade):
        super().__init__()
        self.jade = jade

    def forward(self, x):
        return self.jade.forward_export(x)


def export(config_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Export] Using device: {device}")

    # Load model
    jade = md.JADE_v3(config_dict).to(device)
    checkpoint_path = config_dict["model"]["memory"].get("checkpoint_path", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        jade.load_checkpoint(checkpoint_path)
        print(f"[Export] Checkpoint loaded: {checkpoint_path}")
    else:
        print("[Export] WARNING: No checkpoint loaded.")

    jade.eval()

    image_size = tuple(config_dict["data"]["image_size"])
    H, W = image_size

    # Save directly to Triton model directory as model.onnx + model.onnx.data
    triton_dir = os.path.join(os.path.dirname(config_dict["export"]["onnx_path"]),
                              "..", "triton", "jade_v3", "1")
    triton_dir = os.path.normpath(triton_dir)
    os.makedirs(triton_dir, exist_ok=True)

    tmp_onnx  = config_dict["export"]["onnx_path"]
    os.makedirs(os.path.dirname(tmp_onnx), exist_ok=True)
    final_onnx = os.path.join(triton_dir, "model.onnx")

    wrapper     = ExportWrapper(jade).to(device)
    dummy_input = torch.zeros(1, 3, H, W, device=device)

    print(f"[Export] Input shape : (1, 3, {H}, {W})")
    print(f"[Export] Output shape: (1, {H}, {W}) — raw float32 error map")
    print(f"[Export] Exporting ONNX...")

    # Trace first — forces old TorchScript exporter which decomposes LayerNorm
    # into primitive ops (ReduceMean, Sub, Div...) supported by all TRT versions
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)

    torch.onnx.export(
        traced,
        dummy_input,
        tmp_onnx,
        opset_version=16,
        input_names=["image"],
        output_names=["error_map"],
        do_constant_folding=True,
        dynamo=False,
    )

    print("[Export] ONNX export complete. Re-saving with Triton-compatible naming...")

    import onnx
    m = onnx.load(tmp_onnx)

    # Verify
    onnx.checker.check_model(m)
    print("[Export] onnx.checker.check_model: PASSED")

    # Save as model.onnx + model.onnx.data so trtexec finds the data file correctly
    onnx.save_model(
        m,
        final_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024
    )
    print(f"[Export] Saved: {final_onnx}")
    print(f"[Export] Saved: {final_onnx}.data")

    # Quick onnxruntime sanity check
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(final_onnx, providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, 3, H, W), dtype=np.float32)
        out = sess.run(["error_map"], {"image": dummy_np})
        print(f"[Export] onnxruntime test: output shape {out[0].shape} — PASSED")
    except Exception as e:
        print(f"[Export] onnxruntime test FAILED: {e}")

    print(f"[Export] Done. Triton model dir: {triton_dir}")
    print("[Export] On target machine:")
    print(f"  cd {triton_dir}")
    print(f"  trtexec --onnx=model.onnx --saveEngine=model.plan --fp16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JADE_v3 ONNX Export")
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    parser.add_argument("-m", "--model",  type=str, default=None,
                        help="Override checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    if args.model:
        config_dict["model"]["memory"]["checkpoint_path"] = args.model

    export(config_dict)
