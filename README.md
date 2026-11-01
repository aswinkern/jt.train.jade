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

## Prediction boxes (--infer)

`--infer` writes, next to `scores.csv` and `pictures/`, a folder `boxes/` with one json file per picture (same sub-folders as the input):

```
{"image": "...jpeg", "width": 960, "height": 960, "score": 2.08, "settings": {...},
 "boxes": [{"x1": 16, "y1": 44, "x2": 714, "y2": 225, "score": 2.43, "area": 91060}, ...]}
```

- A box is a region of the heat map that is not dark blue: the pixels above `pixelwise_threshold` are joined into regions, and each region is one box.
  Set a `pixelwise_threshold` above 0 to get useful boxes. There is no other filter: use `score` (the largest error inside the box) and `area` to filter small ones.
- The coordinates are in the original picture (the size of the file), like the training labels, not in the size the model saw.
- The boxes are sorted by `score`, highest first. Code: `tools/boxes.py`. Both methods write them.

### What the saved pictures show: `model.visualisation`

| Value | The picture in `pictures/` |
|---|---|
| `"heatmap"` (default) | original \| original + heat map, as before |
| `"boxes"` | original \| original with the prediction boxes drawn (red, with the score) |

It is read like the other colour settings: from the `config.yaml` next to `run.sh`, or from `--settings`. The json files in `boxes/` are written in both cases.

## Analytics page (labels against prediction boxes)

```
python tools/analytics_html.py --labels <folder with okay/ and defect/> --pred <the --out folder of --infer> --out analytics.html
```

One HTML file you open in a browser (no internet needed, the thumbnails are inside, about 7 MB for 434 pictures).
`--labels` has `<group>/image/` and `<group>/label/` (KITTI labels) for the groups `okay` and `defect`, like `od_train/<component>/`;
`--pred` has `<group>/boxes/*.json` written by `--infer`. A cutoff slider changes every number on the page. A predicted box is **correct** when a labelled defect (not `defect_1` to `defect_4`)
lies in its region, and a labelled defect is **found** when a predicted box has it in its region; the "match rule" choice says what "in its region" means
(default: the label touches the box).
pictures caught and false alarms, a score histogram, the trade-off curve, labelled defects found per class, and a gallery and table of
all pictures with the labelled boxes (green) and the predicted boxes (orange).

## When a run has finished

The colours of the heat maps are written into `config.yaml` next to `run.sh`, so `--infer` draws like the run:
- jade: `heatmap_scaling_factor`, the TensorBoard colour scale of the last epoch.
- patchcore: `heatmap_scaling_factor` and `heatmap_floor`, worked out from OK pictures kept out of the memory bank and the NG pictures.

## More

- `patchcore/README.md`: PatchCore settings and how its colours are set.
- `convert.md`: from the ONNX export to a TensorRT engine and Triton (JADE).
- Weights are not in git. DINOv2: `/jidoka/models/jade/`. PatchCore: see `patchcore/weights/README.md`.
