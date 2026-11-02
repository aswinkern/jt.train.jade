# Convert the ONNX to a TensorRT engine (trtexec)

The engine must be built with the **same Triton image that will serve it**, on the **same kind of GPU**.
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

Right numbers: `patchcore_client.py` on the labelled test gives the same flagged / not flagged as `./run.sh --infer` for every picture and nearly the same scores
(compare the `boxes/` folders; tools/analytics_html.py on the client output must show the same numbers as on the `--infer` output). Measured with the engine above on the 307 labelled
test pictures: 0 pictures flagged differently, the same number of boxes on every picture, picture scores differ by 0.0005 at most (the GPU convolutions of `--infer` run in TF32, so
exact equality is not expected). If the engine gives different
numbers (as it did for JADE with TensorRT 10.3 and 10.4), serve the ONNX through the ONNX Runtime backend instead: `platform: "onnxruntime_onnx"` in
`config.pbtxt`, and no `model.plan`.
