#!/bin/bash
# JADE_v3: one entry point (docker image training:latest, GPU). Run it from anywhere.
#   ./run.sh train     [--config FILE] [--data FOLDER] [--out FOLDER] [--gpu N]
#   ./run.sh infer     --data FOLDER --checkpoint FILE --out FOLDER [--config FILE] [--settings FILE] [--gpu N] [--limit N] [--batch N] [--redo]
#   ./run.sh pretrain | calibrate | analyse | export     [--config FILE] [--gpu N]
#
# train       trains with --config (default: the config.yaml next to this file) on the data folder of data.data_dir (or --data).
#             The data folder holds images/ = OK pictures (training) and NG/ = NG pictures (only shown in TensorBoard, never trained on).
#             If a folder is missing it is created and you are told where to drop the pictures; with no OK picture it stops before training.
#             Every run gets its own folder <out>/exp_N (never
#             overwrites): config.yaml (@EXP@ in the config is replaced by that folder), checkpoints/, run.log. Default --out: <workspace_path of the config>/experiments
# infer       the trained model on any folder of pictures (sub-folders are searched too). Writes to --out: raw/ (model output, no
#             post-processing), post/ (result with the settings), regions/ (features of every predicted region, one json per picture),
#             scores.csv, pictures/ [original | heat map].
#   --config    model definition (backbone, layers, bottleneck): default the config.yaml of the run that owns the checkpoint (<checkpoint dir>/../config.yaml)
#   --settings  where model.layer_weights, featurewise_threshold, heatmap_scaling_factor, pixelwise_threshold, area_threshold are read from:
#               default the config.yaml next to this file (the one you edit). The run's own copy is NOT used for these.
#   --limit N   score only N pictures, evenly spread over the folder (default: all)
#   --batch N   pictures sent to the GPU together (default 8; lower it if the GPU runs out of memory, e.g. 4)
#   --redo      run the model on every picture again (default: use <out>/raw/ when the checkpoint, data folder and
#               featurewise_threshold are the same, so the other settings only redo post/, scores.csv and pictures/)
# pretrain ... export   the other steps of the original pipeline, each reads the paths it needs from --config (a run's config.yaml)
# The data, weights (model.feature_extractor.weights_path) and workspace_path of the config must exist on this machine. Files are written as you, not as root.
# Relative paths and symlinks are fine; they are resolved here.
CODE=$(dirname "$(readlink -f "$0")")
CMD=$1; shift
case "$CMD" in train|infer|pretrain|calibrate|analyse|export) ;; *) sed -n 2,6p "$0"; exit 1;; esac
GPU=0; CFG=""; OUT=""; DATA=""; CK=""; SET=""; LIMIT=0; BATCH=8; REDO=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CFG=$2; shift 2;; --out) OUT=$2; shift 2;; --gpu) GPU=$2; shift 2;; --data) DATA=$2; shift 2;;
    --checkpoint) CK=$2; shift 2;; --settings) SET=$2; shift 2;; --limit) LIMIT=$2; shift 2;; --batch) BATCH=$2; shift 2;; --redo) REDO="--redo"; shift;;
    *) echo "unknown option $1"; sed -n 2,6p "$0"; exit 1;;
  esac
done

if [ "$CMD" = infer ]; then
  if [ -z "$DATA" ] || [ -z "$CK" ] || [ -z "$OUT" ]; then sed -n 2,6p "$0"; echo "infer needs --data --checkpoint --out"; exit 1; fi
  [ -d "$DATA" ] || { echo "data folder not found: $DATA"; exit 1; }
  [ -f "$CK" ] || { echo "checkpoint not found: $CK"; exit 1; }
  DATA=$(readlink -f "$DATA"); CK=$(readlink -f "$CK")
  [ -n "$CFG" ] || CFG=$(dirname "$CK")/../config.yaml
  [ -n "$SET" ] || SET=$CODE/config.yaml
  [ -f "$CFG" ] || { echo "config not found: $CFG (give --config)"; exit 1; }
  [ -f "$SET" ] || { echo "settings file not found: $SET (give --settings)"; exit 1; }
  CFG=$(readlink -f "$CFG"); SET=$(readlink -f "$SET")
  WS_CFG=$CFG
else
  [ -n "$CFG" ] || CFG=$CODE/config.yaml
  [ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }
  CFG=$(readlink -f "$CFG"); WS_CFG=$CFG
fi

# workspace_path of the config: its data, weights and sim folders are mounted at that path (like the paths inside the config)
WS=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['customer']['workspace_path'])" "$WS_CFG") || { echo "no customer.workspace_path in $WS_CFG"; exit 1; }
[ -d "$WS" ] || { echo "workspace_path not found: $WS"; exit 1; }
REAL_WS=$(readlink -f "$WS")
WSMODE=rw; [ "$CMD" = infer ] && WSMODE=ro
MOUNTS=" "; MARGS=""
add_mount_as() { case "$MOUNTS" in *" $2 "*) ;; *) MOUNTS="$MOUNTS$2 "; MARGS="$MARGS -v $1:$2:$3";; esac; }      # add_mount_as <real folder> <path in the container> <ro|rw>
add_mount() { add_mount_as "$1" "$1" "$2"; }
add_mount "$CODE" ro; add_mount "$(dirname "$CFG")" ro
# the DINOv2 weights of the config (absolute path): their folder is mounted at that path
for KEY in weights_path adapted_weights_path; do
  WP=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))[\"model\"][\"feature_extractor\"].get(sys.argv[2]) or \"\")" "$CFG" $KEY 2>/dev/null)
  case "$WP" in /*) [ -e "$WP" ] && add_mount_as "$(readlink -f "$(dirname "$WP")")" "$(dirname "$WP")" ro;; esac
done

PY=/jidoka/install/conda/envs/torch-gpu/bin/python
FILTER="xFormers|warnings.warn|NVIDIA|TensorRT|Copyright|CUDA|Container|container|https://|Various|plugins|run /opt|To install|See https|Forward|WARNING|was built|with this|Using driver|====="
docker_run() {   # docker_run <script and arguments...>
  docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp --shm-size=8g --gpus "\"device=$GPU\"" -v $REAL_WS:$WS:$WSMODE $MARGS -w $CODE training:latest $PY -u "$@" 2>&1
}

case "$CMD" in
  train)
    # the data folder: images/ = OK pictures (training), NG/ = NG pictures (TensorBoard only)
    DD=${DATA:-$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['data']['data_dir'])" "$CFG")}
    mkdir -p "$DD/images" "$DD/NG"
    count() { find "$1" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l; }
    NOK=$(count "$DD/images"); NNG=$(count "$DD/NG")
    echo "[Run] data folder: $DD"
    echo "[Run]   images/  OK pictures, used for training:                     $NOK"
    echo "[Run]   NG/      NG pictures, only shown in TensorBoard (never trained on): $NNG"
    if [ "$NOK" = 0 ]; then
      echo "[Run] No OK pictures yet. Put the OK pictures in $DD/images and the NG pictures in $DD/NG, then run the same command again."
      exit 0
    fi
    [ "$NNG" = 0 ] && echo "[Run] Note: NG/ is empty, TensorBoard will show only OK pictures."
    [ -n "$OUT" ] || OUT=$REAL_WS/experiments
    mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
    n=1; while [ -e "$OUT/exp_$n" ]; do n=$((n+1)); done
    EXP=$OUT/exp_$n; mkdir -p "$EXP"
    sed "s#@EXP@#$EXP#g" "$CFG" > "$EXP/config.yaml"
    if [ -n "$DATA" ]; then      # --data: the run's config gets that folder (real path, mounted below)
      DD=$(readlink -f "$DATA"); sed -i "s#^\(    data_dir:\).*#\1 \"$DD\"#" "$EXP/config.yaml"
    fi
    case "$DD" in "$WS"/*|"$REAL_WS"/*) ;; *) add_mount_as "$(readlink -f "$DD")" "$DD" ro;; esac
    add_mount "$OUT" rw
    echo "[Run] output folder: $EXP  (gpu $GPU)"; echo "[Run] config: $CFG"
    docker_run $CODE/training/train.py -c "$EXP/config.yaml" | grep --line-buffered -v -E "$FILTER" | grep --line-buffered -v -E "^\s*$|Batch \[[1-9]" | tee "$EXP/run.log"
    ;;
  infer)
    mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
    echo "[Run] data: $DATA"; echo "[Run] checkpoint: $CK"; echo "[Run] model config: $CFG"; echo "[Run] settings read from: $SET"; echo "[Run] output: $OUT"
    add_mount "$OUT" rw; add_mount "$DATA" ro; add_mount "$(dirname "$CK")" ro; add_mount "$(dirname "$SET")" ro
    docker_run $CODE/inference/infer_images.py -c "$CFG" --settings "$SET" --checkpoint "$CK" --data "$DATA" -o "$OUT" --limit $LIMIT --batch $BATCH $REDO \
      | grep --line-buffered -v -E "$FILTER" | grep --line-buffered -v -E "^\s*$"
    ;;
  pretrain)  docker_run $CODE/training/pretrain.py -c "$CFG" | grep --line-buffered -v -E "$FILTER" ;;
  calibrate) docker_run $CODE/tools/calibrate.py -c "$CFG" | grep --line-buffered -v -E "$FILTER" ;;
  analyse)   docker_run $CODE/tools/analyse.py -c "$CFG" | grep --line-buffered -v -E "$FILTER" ;;
  export)    docker_run $CODE/tools/export.py -c "$CFG" | grep --line-buffered -v -E "$FILTER" ;;
esac
