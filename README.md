# jt.train.jade: anomaly detection training (JADE and PatchCore)

Two methods, one launcher, the same data layout. Both learn what **normal** looks like from OK pictures only.

| | JADE (`method: jade`) | PatchCore (`method: patchcore`) |
|---|---|---|
| Idea | A frozen DINOv2 gives patch features. Small autoencoders learn to rebuild normal ones. Bad rebuilds are errors. | A frozen ImageNet CNN gives patch features. A selection of normal patches is remembered. Far from all of them is an error. |
| "Training" | Epochs of gradient training | One pass and a coreset (minutes) |
| Config template | `config_jade.yaml` | `config_patchcore.yaml` |
| Code | `model/`, `training/`, `inference/`, `tools/` | `patchcore/` |
| Run saves | `jade_<data>_<date>_<time>/` with `checkpoints/epoch_N.pt` | `patchcore_<data>_<date>_<time>/` with `checkpoints/memory_bank.pt` |
| Export for Triton | `./run.sh --export` (ONNX) | not built |

## Use it

1. Make a sim folder and copy two files into it: `run.sh`, and one template renamed to `config.yaml`
   (`config_jade.yaml` or `config_patchcore.yaml`).
2. Check the paths in `config.yaml`. Paths are `/jidoka/...` paths (as seen inside the docker container).
   `data.data_dir` must hold `images/` (OK pictures) and `NG/` (NG pictures, never trained on).
3. `./run.sh` in that folder starts the run in docker. The `method:` line of `config.yaml` decides which method runs.
4. Other pictures: `./run.sh --infer <folder> --checkpoint <run folder>/checkpoints/<epoch_N.pt or memory_bank.pt> --out <folder>`
5. `./run.sh --help` lists every option.

The code is not copied into the sim folder: `code_dir` in `config.yaml` points to this repo (`/jidoka/code/jt.train.jade`).

## When a run has finished

The colours of the heat maps are written into `config.yaml` next to `run.sh`, so `--infer` draws like the run:
- jade: `heatmap_scaling_factor`, the TensorBoard colour scale of the last epoch.
- patchcore: `heatmap_scaling_factor` and `heatmap_floor`, worked out from OK pictures kept out of the memory bank and the NG pictures.

## More

- `patchcore/README.md`: PatchCore settings and how its colours are set.
- `convert.md`: from the ONNX export to a TensorRT engine and Triton (JADE).
- Weights are not in git. DINOv2: `/jidoka/models/jade/`. PatchCore: see `patchcore/weights/README.md`.
