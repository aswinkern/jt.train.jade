# PatchCore (`method: patchcore`)

PatchCore (Roth et al. 2022, [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)) with the same setup as JADE: `config.yaml` + `run.sh` in a sim folder, data folder with `images/` (OK) and `NG/`.
It is part of the `jt.train.jade` repo and uses the same `run.sh`; the template is `config_patchcore.yaml` in the repo root.
There is no training with gradient steps. The OK pictures are turned into patch features, a greedy k-center coreset of them is kept
(the memory bank), and the error of a patch is its distance to the nearest patch in the bank.

This code is our own implementation of the method in the paper. No code of the official repo (`amazon-science/patchcore-inspection`, Apache-2.0) is in this repo.

## What is in this folder

| File | What it is |
|---|---|
| `fit.py` | builds the memory bank, sets the colours, draws the result pictures |
| `infer.py` | scores any folder of pictures with a memory bank |
| `model.py` | the PatchCore model: backbone, patches, coreset, nearest-neighbour search |
| `common.py` | loading pictures, colours, saving results |
| `weights/` | the backbone weights (torchvision ImageNet, WideResNet50), not in git: see `weights/README.md`. No download at run time |

Outside this folder: `../run.sh` (the launcher), `../config_patchcore.yaml` (the settings) and `../tools/set_config_value.py` (writes one value into a `config.yaml`).

It needs only `torch`, `torchvision`, `opencv`, `numpy` and `pyyaml`, which `training:latest` already has.
The weights are an ImageNet-trained file. Check the license terms before commercial use.

## Run
1. Copy `run.sh` and `config_patchcore.yaml` (renamed `config.yaml`) from the repo root into a sim folder (for example `sim_object_detection/sim_4` next to the JADE sims).
2. Check `data.data_dir` (`images/` = OK pictures, `NG/` = NG pictures that are only scored).
3. `./run.sh` in that folder. The run folder `patchcore_<data>_<date>_<time>/` has `checkpoints/memory_bank.pt`, `results/` (scores.csv and the
   pictures of some OK and all NG pictures), `tensorboard/` and `run.log`.
4. Any folder of pictures: `./run.sh --infer <folder> --checkpoint <run folder>/checkpoints/memory_bank.pt --out <folder>`

## config.yaml
| Key | Meaning |
|---|---|
| `model.backbone.type` | torchvision ResNet: `wide_resnet50_2` (the PatchCore default), `resnet18`, `resnet50` ... |
| `model.backbone.weights_path` | its ImageNet weights (`patchcore/weights/wide_resnet50_2_imagenet.pth` in this repo, as `/jidoka/code/jt.train.jade/patchcore/weights/...`) |
| `model.backbone.layers` | stages 1 to 4 of the ResNet (stride 4, 8, 16, 32). `[2, 3]` is the PatchCore default |
| `model.patch.neighbourhood` | every patch is averaged with its neighbours in a square of this size |
| `model.patch.embed_dim` | numbers per patch after squeezing |
| `model.memory.coreset_ratio` | share of all OK patches that is kept in the bank. More = better and slower (0.01 is the paper's fast setting) |
| `model.memory.projection_dim`, `seed` | random projection used only to choose the coreset, and its seed |
| `model.blur_sigma` | blur of the error map (0 = none) |
| `model.heatmap_scaling_factor` | **worked out and written by the run** (when `calibration_pictures` is more than 0). Optional otherwise. Error that is drawn dark red. Missing = the 99th percentile of the pictures of that run (like JADE's TensorBoard) |
| `model.heatmap_floor` | **worked out and written by the run** (as above). Optional otherwise, default 0. An error at or below it is dark blue, and the colours spread from there up to the dark-red value. PatchCore's error is a distance and is never near 0, so a normal chain looks yellow without a floor. Use the normal level that the run prints (`[Result] normal level ...`) or about the OK pictures' typical error |
| `model.pixelwise_threshold` | 0-100. A pixel at or below it is drawn dark blue. 0 = nothing hidden |
| `output.calibration_pictures` | OK pictures (spread over the folder) kept **out of the memory bank**. They show what unseen normal looks like, set the colours (floor and dark red) and are the OK pictures drawn after the run. 0 = none held out, colours stay as in the config |
| `output.ng_pictures` | how many NG pictures are scored and drawn after the run |

## How the colours are set (automatic)
At the end of a run, the pictures kept out of the bank and the NG pictures are scored. Then:
- `heatmap_floor` = the error below which 9 of 10 pixels of the normal pictures are. Normal parts become dark blue.
- `heatmap_scaling_factor` = the error above which only 1 of 100 pixels of the normal and NG pictures is. That is dark red.
Both are written into the run's `config.yaml` and into the `config.yaml` next to `run.sh`, so `--infer` draws the same way.
They belong to that memory bank and that image size, and every finished run replaces them.

## Notes
- The score of a picture is the largest error in its map (`score_max` in scores.csv).
- The OK pictures shown after the run were used to build the bank, so they look better than pictures it has not seen. Use `--infer` on unseen OK pictures.
- Memory: the features of all OK pictures are held in RAM as float16 (about 5 MB per picture at 182 x 952 with `embed_dim` 1024).
