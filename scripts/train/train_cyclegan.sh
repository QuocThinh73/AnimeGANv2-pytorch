#!/usr/bin/env bash
set -euo pipefail

STYLE="${1:?Usage: $0 <STYLE> <GPU_ID> <RESUME(0|1)> <START_EPOCH>}"
GPU_ID="${2:?}"
RESUME="${3:?}"
START_EPOCH="${4:?}"

IMG="cyclegan-trainer:cu118"
NAME="cyclegan-${STYLE,,}-gpu${GPU_ID}"
PHOTO_ROOT="/workspace/data/train_photo"
ANIME_STYLE_ROOT="/workspace/data/${STYLE}/style"
OUT_DIR="/workspace/output/${STYLE}"

NUM_EPOCHS=200
SAVE_EVERY=10
BATCH=8
NUM_WORKERS=4
IMAGE_SIZE=256
DECAY_EPOCH=100
LR=0.0002

HOST_OUT_ROOT="$(pwd)/output"
HOST_CKPT_DIR="${HOST_OUT_ROOT}/${STYLE}/cyclegan/checkpoints/epoch_$(printf "%03d" ${START_EPOCH})"
CONT_CKPT_DIR="/workspace/output/${STYLE}/cyclegan/checkpoints/epoch_$(printf "%03d" ${START_EPOCH})"

if docker ps -a --format '{{.Names}}' | grep -wq "$NAME" ; then
  echo "Container $NAME exists. Removing..."
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

CMD_ARGS=(
  --model cyclegan
  --photo_root "$PHOTO_ROOT"
  --anime_style_root "$ANIME_STYLE_ROOT"
  --batch_size "$BATCH"
  --num_workers "$NUM_WORKERS"
  --out_dir "$OUT_DIR"
  --num_epochs "$NUM_EPOCHS"
  --save_every "$SAVE_EVERY"
  --image_size "$IMAGE_SIZE"
  --decay_epoch "$DECAY_EPOCH"
  --lr "$LR"
)

if [[ "$RESUME" == "1" ]]; then
  if [[ ! -f "${HOST_CKPT_DIR}/ckpt.pth" ]]; then
    echo "[ERROR] Resume=1 nhưng không thấy checkpoint (host): ${HOST_CKPT_DIR}/ckpt.pth"
    echo "Gợi ý: find output -maxdepth 4 -name ckpt.pth"
    exit 1
  fi
  echo "[RESUME] Using checkpoint: ${CONT_CKPT_DIR}  (start_epoch=${START_EPOCH})"
  CMD_ARGS+=( --resume --start_epoch "$START_EPOCH" --ckpt_dir "$CONT_CKPT_DIR" )
fi

docker run -d \
  --name "$NAME" \
  --gpus "device=${GPU_ID}" \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/data":/workspace/data \
  -v "$(pwd)/output":/workspace/output \
  --ipc=host --shm-size=2g \
  --restart unless-stopped \
  "$IMG" \
  "${CMD_ARGS[@]}"