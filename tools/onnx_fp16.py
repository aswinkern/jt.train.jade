"""An FP16 copy of an exported ONNX, for TensorRT 11 and newer (RTX 50-series / Blackwell).

Why this exists
---------------
TensorRT 10 and older build **weakly typed** networks: the ONNX stays FP32 and
`trtexec --fp16` is permission for the builder to pick FP16 kernels layer by layer.

TensorRT 11 **removed `--fp16`**. It builds **strongly typed** networks, where the
builder honours the types declared in the ONNX and does no precision conversion of
its own (`--stronglyTyped` is listed as a deprecated no-op, "now enabled by default").
To get an FP16 engine there, the ONNX itself has to be FP16 - which is what this writes.

On a 50-series card this is not only about speed. The FP32 ONNX of PatchCore does
not build at all: TensorRT 11 fuses the whole network into one node whose only
tactic asks for about 7.4 GB, more than a 12 GB card has free, so it skips that
tactic, finds no other, and `trtexec` writes a 0-byte engine while still printing
`&&&& PASSED`. Halving the weights and the distance table brings the build in range.

The input and the output stay FP32 (`keep_io_types=True`), so `config.pbtxt`
(`TYPE_FP32`) and the Triton clients need no change - only the weights and the
arithmetic inside become FP16.

Measured on PatchCore_v1.0.0 (bank of 245,050 patches, 518 x 518), against the FP32
PyTorch result of `--infer` over 307 labelled pictures:

| engine                              | per picture | AUROC  | max per-picture deviation |
|-------------------------------------|-------------|--------|---------------------------|
| RTX 3080 Ti, TRT 10.3, FP32         | 100.6 ms    | 0.9411 | -                         |
| RTX 3080 Ti, TRT 10.3, `--fp16`     |  26.5 ms    | 0.9410 | 0.0061                    |
| RTX 5070,    TRT 11.3, FP16 ONNX    |  29.8 ms    | 0.9412 | 0.0067                    |

No picture changed its flagged / not-flagged decision at the 99 % or 95 % OK-pass
operating points. A deviation of about 0.006 is the normal FP16 signature; a
deviation of 0.1 or more means the engine is wrong, so always re-check an engine
with tools/analytics_html.py or the Triton client before using it.

Use
---
Normally through the export: `export.fp16: true` in config.yaml (or `--fp16`), which
writes `Serving_model/1/model_fp16.onnx` next to the FP32 one.

Standalone, on an ONNX that already exists:

    python tools/onnx_fp16.py -i <model.onnx> [-o <model_fp16.onnx>] [--block]

`--block` keeps the cancellation-sensitive tail of PatchCore's distance computation,
sqrt(clamp(|q|^2 + min(|b|^2 - 2 q.b))), in FP32 while the matrix products - which are
essentially all of the work - still run in FP16. Full FP16 passed our checks, so this
is only a fallback if a future bank or image size does not.
"""
import argparse
import os

# The ops that combine |q|^2, -2 q.b and |b|^2. Catastrophic cancellation lives here:
# the two squared norms are large and their difference is small. export.py already
# centres the bank on its mean, which shrinks both terms and is why full FP16 works.
SENSITIVE_OPS = ["ReduceSum", "Add", "Sqrt", "Clip", "ReduceMin", "Min", "Sub"]


def _converter():
    """convert_float_to_float16 from whichever package is installed.

    onnxruntime ships a copy of onnxconverter_common's converter; training:latest has
    onnxruntime but not onnxconverter_common, so try that first.
    """
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        return convert_float_to_float16, "onnxruntime.transformers.float16"
    except Exception:
        pass
    try:
        from onnxconverter_common.float16 import convert_float_to_float16
        return convert_float_to_float16, "onnxconverter_common.float16"
    except Exception:
        pass
    raise SystemExit(
        "[FP16] no float16 converter found. Needs onnxruntime (training:latest has it) "
        "or 'pip install onnxconverter-common'."
    )


def write_fp16(src_onnx, dst_onnx=None, block_sensitive=False, quiet=False):
    """Write an FP16 copy of src_onnx. Returns a dict describing what was written.

    src_onnx  an exported model.onnx (its external .data file must be beside it)
    dst_onnx  default: model_fp16.onnx next to the source
    """
    import onnx

    if dst_onnx is None:
        d, base = os.path.dirname(src_onnx), os.path.basename(src_onnx)
        dst_onnx = os.path.join(d, base[:-5] + "_fp16.onnx" if base.endswith(".onnx") else base + "_fp16.onnx")
    datafile = os.path.basename(dst_onnx) + ".data"
    say = (lambda *x: None) if quiet else (lambda *x: print(*x, flush=True))

    convert, where = _converter()
    say("[FP16] converter: %s" % where)
    say("[FP16] loading %s" % src_onnx)
    m = onnx.load(src_onnx)

    blocked = SENSITIVE_OPS if block_sensitive else None
    say("[FP16] converting | keep_io_types=True | op_block_list=%s" % (blocked or "none"))
    m16 = convert(m, keep_io_types=True, disable_shape_infer=True, op_block_list=blocked)

    for f in (dst_onnx, os.path.join(os.path.dirname(dst_onnx), datafile)):
        if os.path.exists(f):
            os.remove(f)
    onnx.save_model(m16, dst_onnx, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=datafile, size_threshold=1024, convert_attribute=False)
    um = os.umask(0); os.umask(um)
    data_path = os.path.join(os.path.dirname(dst_onnx), datafile)
    if os.path.exists(data_path):
        os.chmod(data_path, 0o666 & ~um)      # onnx writes the data file as 600 whatever the umask

    name_of = onnx.TensorProto.DataType.Name
    kinds = {}
    for init in m16.graph.initializer:
        kinds[name_of(init.data_type)] = kinds.get(name_of(init.data_type), 0) + 1
    rep = {
        "onnx": dst_onnx,
        "data": data_path,
        "mb": os.path.getsize(data_path) / 1e6 if os.path.exists(data_path) else 0.0,
        "initializers": kinds,
        "casts": sum(1 for n in m16.graph.node if n.op_type == "Cast"),
        "inputs": [(i.name, name_of(i.type.tensor_type.elem_type)) for i in m16.graph.input],
        "outputs": [(o.name, name_of(o.type.tensor_type.elem_type)) for o in m16.graph.output],
    }
    say("[FP16] initializers %s | Cast nodes %d" % (rep["initializers"], rep["casts"]))
    say("[FP16] inputs %s -> outputs %s  (kept FP32: config.pbtxt and the client need no change)"
        % (rep["inputs"], rep["outputs"]))
    say("[FP16] saved %s and %s (%.0f MB)" % (dst_onnx, data_path, rep["mb"]))
    return rep


def main():
    ap = argparse.ArgumentParser(description="FP16 copy of an exported ONNX (TensorRT 11 / RTX 50-series)")
    ap.add_argument("-i", "--inp", required=True, help="the exported model.onnx (external .data beside it)")
    ap.add_argument("-o", "--out", default=None, help="default: model_fp16.onnx next to the input")
    ap.add_argument("--block", action="store_true",
                    help="keep the cancellation-sensitive reduction ops in FP32 (%s)" % ", ".join(SENSITIVE_OPS))
    a = ap.parse_args()
    write_fp16(a.inp, a.out, block_sensitive=a.block)


if __name__ == "__main__":
    main()
