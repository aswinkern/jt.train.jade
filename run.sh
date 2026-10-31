#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — JADE_v3 (Docker-enabled), same idea as jt.train.yolonas/run.sh
#
#   Copy run.sh and config.yaml into a sim folder, open a terminal there and run ./run.sh
#   It opens the docker container itself. The code is NOT copied: code_dir in config.yaml says where it is.
#   Everything it saves (jade_<data>_<date>_<time>/ ...) goes into the folder you run it from, as you (never root), with group-writable permissions.
#
#   ./run.sh                          train. Data folder = data_dir of config.yaml, it holds images/ (OK pictures, used for training)
#                                     and NG/ (NG pictures, only shown in TensorBoard, never trained on). A missing folder is created and
#                                     you are told where to drop the pictures; with no OK picture it stops before training.
#                                     Each run gets its own folder, named model + data folder + date and time, e.g. ./jade_data_2_20261031_114205
#                                     (never overwrites): config.yaml, checkpoints/, tensorboard/, run.log
#                                     TensorBoard starts with the training, its address is printed (the port is free-picked from 6001).
#   ./run.sh --infer <folder> --checkpoint <epoch_N.pt> --out <folder>   [--settings <config.yaml>] [--limit N] [--batch N] [--redo]
#                                     the trained model on any folder of pictures (sub-folders too). Writes to --out: raw/ (model output, no
#                                     post-processing), post/ (result with the settings), scores.csv, pictures/ [original | heat map].
#                                     The settings (layer_weights, featurewise_threshold, heatmap_scaling_factor, pixelwise_threshold)
#                                     come from config.yaml (or --settings). The pictures look like the TensorBoard ones. --limit N scores only N pictures spread over the
#                                     folder, --batch N pictures per GPU step (default 8, lower it if the GPU runs out of memory), --redo
#                                     runs the model again (default: raw/ is reused when checkpoint, pictures and featurewise_threshold are the same).
#   ./run.sh --tensorboard            only TensorBoard on this folder (Ctrl+C to stop)
#   ./run.sh --export [--checkpoint <epoch_N.pt>] [--out <folder>] [--max_batch N]
#                                     export.model_name, export.experiment and export.epoch of config.yaml say which model name and which
#                                     checkpoint (<experiment>/checkpoints/epoch_<epoch>.pt); --checkpoint overrides experiment and epoch.
#                                     ONNX for Triton / TensorRT into export.models_dir/export.model_name/Serving_model/ (the models folder of the customer,
#                                     --out replaces models_dir/model_name): 1/model.onnx (+ .data, dynamic batch), config.pbtxt.
#                                     export.model_name, export.experiment and export.epoch of config.yaml say which model name and which
#                                     checkpoint (<experiment>/checkpoints/epoch_<epoch>.pt); --checkpoint overrides experiment and epoch.
#                                     The model is RAW (picture -> error map): only layer_weights and featurewise_threshold of config.yaml are baked in;
#                                     heatmap_scaling_factor and pixelwise_threshold are variables at the top of triton/triton_client.py
#                                     (the export prints them, ready to paste), so they change without a new export. It checks the ONNX against PyTorch on OK
#                                     pictures. An existing model of that name is never overwritten: use a new export.model_name (version), or --overwrite.
#                                     --max_batch is the largest batch, max_batch_size of config.pbtxt (default 16).
#   ./run.sh --pretrain | --calibrate | --analyse [--config <run folder>/config.yaml]    the other steps of the original pipeline
#
#   --config <file>   default config.yaml in this folder          --gpu N   default gpu_id of the config (0 or 1)
#   Paths on the command line may be host paths (/mnt/jdisk3/...) or /jidoka/... paths. Paths inside config.yaml are /jidoka/... paths.
# ─────────────────────────────────────────────────────────────────────────────

USE_DOCKER=${USE_DOCKER:-true}
DOCKER_IMAGE="training:latest"
PY=/jidoka/install/conda/envs/torch-gpu/bin/python
TB=/jidoka/install/conda/envs/torch-gpu/bin/tensorboard
FILTER="xFormers|warnings.warn|NVIDIA|TensorRT|Copyright|CUDA|Container|container|https://|Various|plugins|run /opt|To install|See https|Forward|WARNING|was built|with this|Using driver|====="

# ── Arguments ─────────────────────────────────────────────────────────────────
ORIG_ARGS=("$@")
CMD=train; CONFIG="config.yaml"; GPU=""; INFER_DIR=""; CK=""; OUT=""; SET=""; LIMIT=0; BATCH=8; REDO=""; MAX_BATCH=16
while [ $# -gt 0 ]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --gpu)         GPU="$2"; shift 2 ;;
        --infer)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then echo "[ERROR] --infer needs a folder of pictures, e.g. ./run.sh --infer /path/to/pictures --checkpoint exp_1/checkpoints/epoch_50.pt --out ./infer_1"; exit 1; fi
            CMD=infer; INFER_DIR="$2"; shift 2 ;;
        --checkpoint)  CK="$2"; shift 2 ;;
        --out)         OUT="$2"; shift 2 ;;
        --settings)    SET="$2"; shift 2 ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        --batch)       BATCH="$2"; shift 2 ;;
        --max_batch)   MAX_BATCH="$2"; shift 2 ;;
        --redo)        REDO="--redo"; shift ;;
        --pretrain|--calibrate|--analyse|--export) CMD="${1#--}"; shift ;;
        --tensorboard) CMD=tensorboard; shift ;;
        -h|--help)     sed -n '2,/^# ───.*$/p' "$0" | sed '1d;$d' | cut -c3-; exit 0 ;;
        *)             echo "[ERROR] Unknown argument: $1 (./run.sh --help)"; exit 1 ;;
    esac
done

# ── Launch the Docker container (skipped when already inside it) ──────────────
if [ "$USE_DOCKER" = true ] && [ ! -f /.dockerenv ]; then
    [ -f "$CONFIG" ] || { echo "[ERROR] Config not found: $CONFIG (put run.sh and config.yaml in the same folder, or use --config)"; exit 1; }
    if [ -z "$GPU" ]; then
        GPU=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1])).get('gpu_id', 0))" "$CONFIG") || exit 1
    fi
    case "$GPU" in ""|*[!0-9]*) echo "[ERROR] gpu id must be a number (0, 1 ...), got: $GPU"; exit 1;; esac

    # Translate a host path to its Docker path (the mounts below)
    translate_path() {
        local p="$1"
        if   [[ "$p" == /mnt/jdisk3/jidoka_training/simulations/* ]]; then echo "/jidoka/simulations${p#/mnt/jdisk3/jidoka_training/simulations}"
        elif [[ "$p" == /mnt/jdisk3/jidoka_training_jdisk3/simulations/* ]]; then echo "/jidoka/simulations/jdisk3${p#/mnt/jdisk3/jidoka_training_jdisk3/simulations}"
        elif [[ "$p" == /mnt/jdisk4/jidoka_training_jdisk4/simulations/* ]]; then echo "/jidoka/simulations/jdisk4${p#/mnt/jdisk4/jidoka_training_jdisk4/simulations}"
        elif [[ "$p" == /jidoka_training/simulations/* ]]; then echo "/jidoka/simulations${p#/jidoka_training/simulations}"
        elif [[ "$p" == /mnt/jdisk3/jidoka_training_jdisk3/workspace/* ]]; then echo "/jidoka/workspace/jdisk3${p#/mnt/jdisk3/jidoka_training_jdisk3/workspace}"
        elif [[ "$p" == /mnt/jdisk4/jidoka_training_jdisk4/workspace/* ]]; then echo "/jidoka/workspace/jdisk4${p#/mnt/jdisk4/jidoka_training_jdisk4/workspace}"
        elif [[ "$p" == /jidoka_training/workspace/* ]]; then echo "/jidoka/workspace${p#/jidoka_training/workspace}"
        elif [[ "$p" == /mnt/jdisk3/jidoka_training/code/* ]]; then echo "/jidoka/code${p#/mnt/jdisk3/jidoka_training/code}"
        elif [[ "$p" == /jidoka_training/code/* ]]; then echo "/jidoka/code${p#/jidoka_training/code}"
        else echo "$p"
        fi
    }
    HOST_PWD="$(pwd)"
    DOCKER_PWD="$(translate_path "$HOST_PWD")"

    # One container per folder + mode, so runs on the two GPUs or train + infer side by side never collide.
    IFS='/' read -ra PATH_PARTS <<< "$(pwd -P | sed 's|^/||')"
    N_PARTS=${#PATH_PARTS[@]}; START=$(( N_PARTS > 3 ? N_PARTS - 3 : 0 ))
    CONTAINER_NAME="${JADE_NAME:-jade_$(IFS=_; echo "${PATH_PARTS[*]:$START}")_${CMD}}"
    CONTAINER_NAME="$(echo "$CONTAINER_NAME" | sed 's/[^a-zA-Z0-9_.-]/_/g')"
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "[ERROR] Already running: $CONTAINER_NAME (same folder, same mode). Stop it first:  docker stop $CONTAINER_NAME"; exit 1
    fi

    # TensorBoard port: the first free one from 6001
    port_in_use() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$" || docker ps --format '{{.Ports}}' | grep -qE ":$1->"; }
    PORT_ARGS=""; TB_HOST_PORT=""
    if [ "$CMD" = train ] || [ "$CMD" = tensorboard ]; then
        TB_HOST_PORT=6001; while port_in_use "$TB_HOST_PORT"; do TB_HOST_PORT=$((TB_HOST_PORT + 1)); done
        PORT_ARGS="-p ${TB_HOST_PORT}:6006"
    fi

    # Arguments for the run inside: host paths -> Docker paths
    PATH_FLAGS="--infer --checkpoint --out --config --settings"
    ESCAPED_ARGS=""; TRANSLATE_NEXT=false
    for arg in "${ORIG_ARGS[@]}"; do
        if [ "$TRANSLATE_NEXT" = true ]; then arg="$(translate_path "$arg")"; TRANSLATE_NEXT=false
        else for flag in $PATH_FLAGS; do [ "$arg" = "$flag" ] && TRANSLATE_NEXT=true; done; fi
        ESCAPED_ARGS="$ESCAPED_ARGS '$(echo "$arg" | sed "s/'/'\\\\''/g")'"
    done

    echo ""
    echo "  Mode           : $CMD"
    echo "  GPU            : $GPU"
    echo "  Folder         : $HOST_PWD"
    echo "  Container      : $CONTAINER_NAME"
    [ -n "$TB_HOST_PORT" ] && echo "  TensorBoard    : http://localhost:${TB_HOST_PORT}  (server port ${TB_HOST_PORT})"
    echo ""
    TTY=-i; [ -t 0 ] && [ -t 1 ] && TTY=-it

    # Run as you (files are never root-owned); umask 0002 = group-writable like the files you make yourself
    RUN_STAMP=$(date +%Y%m%d_%H%M%S)                  # the time on this machine, used in the name of the run folder
    docker run --gpus "\"device=$GPU\"" $TTY --rm -e RUN_STAMP="$RUN_STAMP" \
        --name "$CONTAINER_NAME" \
        --user "$(id -u):$(id -g)" \
        --shm-size=8g \
        -e HOME=/tmp \
        -e TB_HOST_PORT="$TB_HOST_PORT" \
        -v /mnt/jdisk3/jidoka_training/code:/jidoka/code \
        -v /mnt/jdisk3/jidoka_training/models:/jidoka/models \
        -v /mnt/jdisk3/jidoka_training_jdisk3/simulations/:/jidoka/simulations/jdisk3 \
        -v /mnt/jdisk4/jidoka_training_jdisk4/simulations/:/jidoka/simulations/jdisk4 \
        -v /mnt/jdisk3/jidoka_training_jdisk3/workspace/:/jidoka/workspace/jdisk3 \
        -v /mnt/jdisk4/jidoka_training_jdisk4/workspace/:/jidoka/workspace/jdisk4 \
        $PORT_ARGS \
        -w "$DOCKER_PWD" \
        $DOCKER_IMAGE \
        bash -c "umask 0002; USE_DOCKER=false ./run.sh $ESCAPED_ARGS"
    exit $?
fi

# ── From here on: inside Docker ───────────────────────────────────────────────
[ -f "$CONFIG" ] || { echo "[ERROR] Config not found: $CONFIG"; exit 1; }
cfg() { $PY -c "import yaml,sys;d=yaml.safe_load(open(sys.argv[1]));print(eval(sys.argv[2], {'d': d}) or '')" "$CONFIG" "$1"; }
CODE_DIR=$(cfg "d.get('code_dir')")
[ -f "$CODE_DIR/training/train.py" ] || { echo "[ERROR] The JADE code was not found. code_dir in $CONFIG is '$CODE_DIR' (it must be the folder with training/, model/ ...)"; exit 1; }
echo "[Run] code: $CODE_DIR   config: $CONFIG"

case "$CMD" in
  train)
    DD=$(cfg "d['data']['data_dir']")
    mkdir -p "$DD/images" "$DD/NG"
    count() { find "$1" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l; }
    NOK=$(count "$DD/images"); NNG=$(count "$DD/NG")
    echo "[Run] data folder: $DD"
    echo "[Run]   images/  OK pictures, used for training:                     $NOK"
    echo "[Run]   NG/      NG pictures, only shown in TensorBoard (never trained on): $NNG"
    if [ "$NOK" = 0 ]; then
        echo "[Run] No OK pictures yet. Put the OK pictures in $DD/images and the NG pictures in $DD/NG, then run ./run.sh again."
        exit 0
    fi
    [ "$NNG" = 0 ] && echo "[Run] Note: NG/ is empty, TensorBoard will show only OK pictures."
    OUTDIR=$(pwd)
    EXP=$OUTDIR/jade_$(basename "$DD")_${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}          # model + data folder + date and time: unique, and sorts by time
    mkdir -p "$EXP"
    sed "s#@EXP@#$EXP#g" "$CONFIG" > "$EXP/config.yaml"
    $TB --logdir "$OUTDIR" --host 0.0.0.0 --port 6006 &> /tmp/tensorboard.log &
    echo "[Run] TensorBoard: http://localhost:${TB_HOST_PORT:-6006}  (runs: $OUTDIR/jade_*/tensorboard)"
    echo "[Run] output folder: $EXP"
    $PY -u "$CODE_DIR/training/train.py" -c "$EXP/config.yaml" 2>&1 | grep --line-buffered -v -E "$FILTER" | grep --line-buffered -v -E "^\s*$|Batch \[[1-9]" | tee "$EXP/run.log"
    ;;
  infer)
    [ -n "$CK" ] && [ -n "$OUT" ] || { echo "[ERROR] --infer needs --checkpoint <epoch_N.pt> and --out <folder>"; exit 1; }
    [ -d "$INFER_DIR" ] || { echo "[ERROR] picture folder not found: $INFER_DIR"; exit 1; }
    [ -f "$CK" ] || { echo "[ERROR] checkpoint not found: $CK"; exit 1; }
    MODEL_CFG=$(dirname "$CK")/../config.yaml                    # the model definition of the run that owns the checkpoint
    [ -f "$MODEL_CFG" ] || { echo "[ERROR] the run's config was not found: $MODEL_CFG"; exit 1; }
    [ -n "$SET" ] || SET=$CONFIG
    [ -f "$SET" ] || { echo "[ERROR] settings file not found: $SET"; exit 1; }
    mkdir -p "$OUT"
    echo "[Run] pictures: $INFER_DIR"; echo "[Run] checkpoint: $CK"; echo "[Run] model config: $MODEL_CFG"; echo "[Run] settings read from: $SET"; echo "[Run] output: $OUT"
    $PY -u "$CODE_DIR/inference/infer_images.py" -c "$MODEL_CFG" --settings "$SET" --checkpoint "$CK" --data "$INFER_DIR" -o "$OUT" --limit "$LIMIT" --batch "$BATCH" $REDO 2>&1 \
        | grep --line-buffered -v -E "$FILTER" | grep --line-buffered -v -E "^\s*$"
    ;;
  tensorboard)
    echo "[Run] TensorBoard: http://localhost:${TB_HOST_PORT:-6006}  (Ctrl+C to stop)"
    $TB --logdir "$(pwd)" --host 0.0.0.0 --port 6006
    ;;
  pretrain)  $PY -u "$CODE_DIR/training/pretrain.py" -c "$CONFIG" 2>&1 | grep --line-buffered -v -E "$FILTER" ;;
  calibrate) $PY -u "$CODE_DIR/tools/calibrate.py" -c "$CONFIG" 2>&1 | grep --line-buffered -v -E "$FILTER" ;;
  analyse)   $PY -u "$CODE_DIR/tools/analyse.py" -c "$CONFIG" 2>&1 | grep --line-buffered -v -E "$FILTER" ;;
  export)
    CKARG=(); if [ -n "$CK" ]; then [ -f "$CK" ] || { echo "[ERROR] checkpoint not found: $CK"; exit 1; }; CKARG=(--checkpoint "$CK"); fi
    $PY -u "$CODE_DIR/tools/export.py" -c "$CONFIG" "${CKARG[@]}" ${OUT:+--out "$OUT"} --max_batch "$MAX_BATCH" 2>&1 | grep --line-buffered -v -E "$FILTER" ;;
esac
