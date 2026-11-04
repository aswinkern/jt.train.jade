# Convert the ONNX to a TensorRT engine (trtexec)

The engine must be built with the **same Triton image that will serve it**, on the **same kind of GPU**.

## Quick reference: which command for which GPU

**The export is the same for both. The trtexec command is not.**

One export writes both ONNX files. In `config.yaml`:

```yaml
export:
    models_dir: "/jidoka/simulations/jdisk3/<customer>/models"
    model_name: "PatchCore_v1.1.0"      # a new fit is a new name
    experiment: "latest"
    fp16: true                          # also write 1/model_fp16.onnx (for the 50-series)
    query_tile: 512                     # PatchCore only: bounds the distance table
```

then, in the sim folder:

```bash
./run.sh --export
```

That writes `<models_dir>/<model_name>/Serving_model/`:

```
config.pbtxt
1/model.onnx          + .data     FP32   -> build the engine for a 30-series from this
1/model_fp16.onnx     + .data     FP16   -> build the engine for a 50-series from this
```

Now the engine, and here the two differ:

| | RTX 30-series (Ampere) | RTX 50-series (Blackwell) |
|---|---|---|
| Triton image | `24.08-py3` | `26.09-py3` (25.01 or newer) |
| TensorRT | 10.3 | 11.3 |
| ONNX to build from | `model.onnx` | **`model_fp16.onnx`** |
| precision flag | **`--fp16`** | **none** - `trtexec` has no `--fp16` |
| why | weakly typed: the flag tells the builder it may use FP16 | strongly typed: the precision comes from the ONNX |
| measured, PatchCore_v1.0.0 | 26.5 ms, 3080 Ti | 29.8 ms, 5070 |

**RTX 30-series** - FP16 from the FP32 ONNX, with the flag:

```bash
docker run --rm --gpus '"device=0"' --user $(id -u):$(id -g) -e HOME=/tmp \
  -v /mnt/jdisk3/jidoka_training_jdisk3/simulations/:/jidoka/simulations/jdisk3 \
  nvcr.io/nvidia/tritonserver:24.08-py3 bash -lc \
  'cd /jidoka/simulations/jdisk3/<customer>/models/<model_name>/Serving_model/1 && \
   /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 \
     --minShapes=image:1x3x518x518 --optShapes=image:1x3x518x518 \
     --maxShapes=image:1x3x518x518 --memPoolSize=workspace:6000'
```

**RTX 50-series** - FP16 from the FP16 ONNX, no flag:

```bash
docker run --rm --gpus 'device=0' --user $(id -u):$(id -g) -e HOME=/tmp \
  -v /mnt/hdd/jidoka/<wherever the model is>:/work \
  nvcr.io/nvidia/tritonserver:26.09-py3 bash -lc \
  'cd /work/<model_name>/Serving_model/1 && \
   /usr/src/tensorrt/bin/trtexec --onnx=model_fp16.onnx --saveEngine=model.plan \
     --minShapes=image:1x3x518x518 --optShapes=image:1x3x518x518 \
     --maxShapes=image:1x3x518x518 --memPoolSize=workspace:6000'
```

Both write `model.plan` next to the ONNX, which is where Triton wants it, and both need
`max_batch_size: 1` in `config.pbtxt` because the shapes above are min = opt = max = 1.
`--user` keeps the engine owned by you instead of root.

For **FP32** instead of FP16 on a 30-series, build from `model.onnx` with `--noTF32` and no
`--fp16` (PatchCore needs `--noTF32`: see the PatchCore section). There is no tested FP32 path
on a 50-series - the FP32 ONNX does not build there at all, see the last section.

Do not reuse an engine across machines, and re-check every engine you build: a `&&&& PASSED`
says nothing about the numbers. Both points are explained below.

---
The commands below use `nvcr.io/nvidia/tritonserver:24.08-py3` (TensorRT 10.3). On a 50-series GPU (RTX 5080) use a newer image
(Triton 25.01 or newer) and a recent driver, and build the engine on that machine.

## 0. Before you start

The ONNX must be the fixed export (the old one ends with an `If` operation that TensorRT cannot read).
Export it again if the model in `models/` was made before commit 256d810:

```bash
./run.sh --export --overwrite
```

## 1. Create a container and keep it running

```bash
docker run -d --name jade_convert --gpus '"device=0"' --user $(id -u):$(id -g) -e HOME=/tmp \
  -v /mnt/jdisk3/jidoka_training_jdisk3/simulations/:/jidoka/simulations/jdisk3 \
  nvcr.io/nvidia/tritonserver:24.08-py3 sleep infinity
```

Change `device=0` to `device=1` for the other GPU.

## 2. Go into the container

```bash
docker exec -it jade_convert bash
```

## 3. Inside the container: go to the model and convert

```bash
cd /jidoka/simulations/jdisk3/jt.cust.anomaly_jade_v3/models/Anomaly_v1.0.0/Serving_model/1

/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 \
  --minShapes=image:1x3x518x518 --optShapes=image:8x3x518x518 --maxShapes=image:16x3x518x518
```

- `model.onnx` and `model.onnx.data` must be in that folder. `model.plan` is written next to them.
- `--minShapes`, `--optShapes`, `--maxShapes` are the smallest, typical and largest batch (here 1, 8, 16). The largest must not be more than `max_batch_size` in `config.pbtxt`.
- For FP32 remove `--fp16`.
- `trtexec` is not on the PATH in this image, use the full path.

When it prints `&&&& PASSED`, type `exit`.

## 4. Remove the container

```bash
docker rm -f jade_convert
```

## 5. Start Triton on the model (to test the engine)

Put `model.plan` into a Triton model folder, for example `workspace/test_model/Anomaly_v1.0.0/1/`. The folder name must be the model name, and
`config.pbtxt` next to `1/` must say `platform: "tensorrt_plan"` (`"onnxruntime_onnx"` serves the `model.onnx` instead of the engine).

```bash
docker run -d --rm --name jade_triton --gpus '"device=0"' \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /mnt/jdisk3/jidoka_training_jdisk3/simulations/jt.cust.anomaly_jade_v3/workspace/test_model:/models:ro \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models
```

- Use the **same image** you built the engine with (here 24.08), or Triton cannot load it.
- Port 8001 is gRPC (the client's default), 8000 is HTTP. If a port is taken, change the left number of `-p` and give the client `-u localhost:<port>`.
- Change `device=0` to `device=1` for the other GPU.

Check that the model is ready (it should print `200`; if not within about a minute, run `docker logs jade_triton`):

```bash
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/v2/models/Anomaly_v1.0.0/ready
```

## 6. Run the client

```bash
cd /mnt/jdisk3/jidoka_training_jdisk3/simulations/jt.cust.anomaly_jade_v3/workspace/triton_client
./run_client.sh -i <picture or folder> -o ./out
```

`--batch 8` sends 8 pictures per request, `-u localhost:<port>` sets the gRPC port. For each picture it prints p99, max and the number of
regions, and writes `<name>.png` = [original | heat map] into the output folder. The settings (model name, picture size, thresholds) are the
variables at the top of `triton_client.py`.

Right numbers look like `--infer`: p99 around 0.1 to 0.3 and max around 0.2 to 0.45. Numbers around 5 to 6 mean the engine is wrong.

## 7. Stop Triton

```bash
docker stop jade_triton
```

## Check the engine before you use it

With TensorRT 10.3 and 10.4 an engine built from this model gave **wrong numbers** (about 20 times too large) in our tests, with FP16 and
with FP32. Compare a few pictures against `./run.sh --infer` before using an engine. The ONNX model served through the ONNX Runtime
backend of Triton (`platform: "onnxruntime_onnx"` in `config.pbtxt`) gave the same result as `--infer`.

---

# PatchCore

Same steps, with these differences. The model is made by `./run.sh --export` with `method: patchcore` (the `export:` block of `config.yaml`:
`models_dir`, `model_name`, `experiment`; `latest` = the newest `patchcore_*` run folder). Its output is the raw error map, like JADE's, and the **memory bank is part
of the model**: a new fit is a new export with a new `model_name`. Client: `triton/patchcore_client.py` (the colour settings are variables at its top).

- **Build the engine in full FP32: no `--fp16`, and `--noTF32`.** The search for the nearest bank patch calculates |q|^2 - 2 q.b + |b|^2, which loses its precision in FP16 and in TF32 (the default of an RTX 30-series GPU for FP32 matrix products).
- **The batch is 1.** TensorRT joins the chunks of the search back into one distance table and keeps it in memory: 4.2 GB per picture at 518 x 518 (4,225 patches x 245,050 bank patches x 4 bytes). A build with batch 8 asks for 16.6 GB and one with batch 2 for 8.4 GB, and fails on a 12 GB GPU. One picture already keeps the GPU busy, so a bigger batch would not be faster. `--max_batch` of the export is 1 for PatchCore; ONNX Runtime can take a bigger batch (change `max_batch_size` in `config.pbtxt`).
- A server holding the engine needs about 5 GB of GPU memory, so the GPU cannot be shared with another big job. At 960 x 960 the table is about 12 times bigger: use a smaller `coreset_ratio`.
- Measured on the RTX 3080 Ti (518 x 518, bank of 245,050 patches): **105 ms per picture through Triton, about 570 pictures per minute** (`--infer` is about 300 per minute).

```bash
cd /jidoka/simulations/jdisk3/jt.cust.anomaly_jade_v3/models/PatchCore_v1.0.0/Serving_model/1

/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32 \
  --minShapes=image:1x3x518x518 --optShapes=image:1x3x518x518 --maxShapes=image:1x3x518x518 --memPoolSize=workspace:6000
```

### `export.query_tile`: the 4.23 GB table

The second bullet above says a server needs about 5 GB. Most of that is one allocation. The
ONNX chunks the **bank** (60 chunks of 4096), but TensorRT fuses those back into a single
query x bank table and asks for it in one piece:

```
4225 query patches x 245050 bank patches x 4 bytes = 4.23 GB      (2.07 GB in FP16)
```

`export.query_tile` (default 512; `--qtile N`) tiles the **queries** instead, so the table is
`query_tile x bank` whatever the bank size. The tiling is over the patch grid, which the image
size fixes, so the batch axis stays dynamic, and every query is still compared against the same
bank chunks in the same order.

Measured on PatchCore_v1.0.0 (bank of 245,050 patches, 518 x 518, FP16 engine, RTX 3080 Ti),
over the 307 labelled test pictures:

| | untiled | `query_tile: 512` |
|---|---|---|
| server VRAM | 5160 MiB | **3210 MiB** |
| inference per picture | 26.48 ms | 26.23 ms |
| scores of all 307 pictures | - | **bit-identical** (max abs difference 0.000000) |
| flagged / not flagged | - | **0 of 307 changed**, at the 99 %, 95 % and 90 % OK-pass points |
| engine build | 183.6 s | 189.9 s |
| engine file | 561.8 MB | 562.7 MB |

The graph is about nine times larger (9 tiles x 60 bank chunks), which costs a few per cent of
build time and nothing at inference. Tracing that graph is heavy on the host: the export of a
245,050-patch bank made a 31 GB machine briefly unresponsive, so do not run it next to
something interactive.

Smaller tiles save more memory; 512 was enough to bring this model under half its old
footprint. `query_tile: 0` restores the old single-table behaviour.

Right numbers: `patchcore_client.py` on the labelled test gives the same flagged / not flagged as `./run.sh --infer` for every picture and nearly the same scores
(compare the `boxes/` folders; tools/analytics_html.py on the client output must show the same numbers as on the `--infer` output). Measured with the engine above on the 307 labelled
test pictures: 0 pictures flagged differently, the same number of boxes on every picture, picture scores differ by 0.0005 at most (the GPU convolutions of `--infer` run in TF32, so
exact equality is not expected). If the engine gives different
numbers (as it did for JADE with TensorRT 10.3 and 10.4), serve the ONNX through the ONNX Runtime backend instead: `platform: "onnxruntime_onnx"` in
`config.pbtxt`, and no `model.plan`.

---

# TensorRT 11 and the RTX 50-series (Blackwell)

Everything above assumes TensorRT 10 (Triton 24.08). A 50-series card needs Triton 25.01 or
newer, and from **TensorRT 11 the recipe changes**, so read this before converting for one.

## `--fp16` no longer exists

```
$ trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 ...
[E] Unknown option: --fp16
```

TensorRT 10 and older build **weakly typed** networks: the ONNX stays FP32 and `--fp16` is
permission for the builder to choose FP16 kernels layer by layer. TensorRT 11 builds
**strongly typed** networks and says so in its own help:

```
--stronglyTyped   [Deprecated] Strongly typed network is now enabled by default. This flag is a no-op.
```

In a strongly typed network the builder honours the types declared in the ONNX and does no
precision conversion of its own. There is no `--best`, no `--precisionConstraints` and no
`--layerPrecisions` either; `--noTF32` is the only precision knob left. **The precision has
to be in the ONNX.**

## So export an FP16 ONNX

Set `export.fp16: true` in `config.yaml` (or `./run.sh --export --fp16`). Next to the FP32
`1/model.onnx` the export then also writes `1/model_fp16.onnx` (+ `.data`), about half the
size. The input and the output stay FP32, so `config.pbtxt` (`TYPE_FP32`) and the Triton
clients need no change.

To convert an ONNX that already exists: `python tools/onnx_fp16.py -i <model.onnx>`.

## Build it with no precision flag at all

```bash
cd <models_dir>/<model_name>/Serving_model/1

/usr/src/tensorrt/bin/trtexec --onnx=model_fp16.onnx --saveEngine=model.plan \
  --minShapes=image:1x3x518x518 --optShapes=image:1x3x518x518 --maxShapes=image:1x3x518x518 \
  --memPoolSize=workspace:6000
```

The engine is FP16 because the model is. About 18 s to build on an RTX 5070.

## For PatchCore this is not optional

The FP32 ONNX **cannot be built** on a 12 GB 50-series card. TensorRT 11 fuses the whole
network into one node whose only tactic asks for 7.41 GB:

```
UNSUPPORTED_STATE: Skipping tactic 0 due to insufficient memory on requested size of 7410216432
Error Code 10: Could not find any implementation for node {ForeignNode[/conv1/Conv.../Reshape_5]}
Assertion failure: false && "Attempting to access an empty engine!"
```

It skips the tactic, finds no other, and writes a **0-byte engine while still printing
`&&&& PASSED`**. Halving the weights and the distance table brings the build into range.

## Measured, and checked

PatchCore_v1.0.0 (bank of 245,050 patches, 518 x 518), against the FP32 PyTorch result of
`--infer` over the 307 labelled test pictures:

| engine | per picture | throughput | AUROC | max per-picture deviation |
|---|---|---|---|---|
| RTX 3080 Ti, TRT 10.3, FP32 `--noTF32` | 100.6 ms | 479 /min | 0.9411 | - |
| RTX 3080 Ti, TRT 10.3, `--fp16` | 26.5 ms | 1379 /min | 0.9410 | 0.0061 |
| RTX 5070, TRT 11.3, FP16 ONNX | 29.8 ms | 1281 /min | 0.9412 | 0.0067 |

No picture changed its flagged / not-flagged decision at the 99 % or 95 % OK-pass operating
points. A deviation of about 0.006 is the normal FP16 signature.

**A passing build says nothing about the numbers.** Always compare an engine against
`--infer` before using it, as the warning at the end of the JADE section says - that failure
mode (numbers about 20 times too large, with TensorRT 10.3 and 10.4) was silent too.

## An engine is not portable

A `.plan` is tied to the GPU architecture **and** the TensorRT version. The 3080 Ti engine
will not load under TensorRT 11, and the 5070 engine will not load under 10.3. Keep the
ONNX - that is the portable artefact - and rebuild the engine on each machine, then re-check
it. Building takes well under a minute.
