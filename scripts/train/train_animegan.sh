set -euo pipefail

CONFIG_ID="${1:?}"
GPU_ID="${2:?}"
RESUME="${3:?}"
START_EPOCH="${4:?}"
shift 4
EXTRAS=("$@")

IMG="cyclegan-trainer:cu118"
NAME="animegan${CONFIG_ID}"
PHOTO_ROOT="/workspace/data/train_photo"
ANIME_STYLE_ROOT="/workspace/data/Shinkai/style"
ANIME_SMOOTH_ROOT="/workspace/data/Shinkai/smooth"
ARGS_ROOT="/workspace/args"
OUT_DIR="/workspace/output/${CONFIG_ID}"
CONT_CKPT_DIR="/workspace/output/${CONFIG_ID}/checkpoints/epoch_$(printf "%03d" ${START_EPOCH})"

if docker ps -a --format '{{.Names}}' | grep -wq "$NAME" ; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

CMD_ARGS=(
  --model animegan
  --photo_root "$PHOTO_ROOT"
  --anime_style_root "$ANIME_STYLE_ROOT"
  --anime_smooth_root "$ANIME_SMOOTH_ROOT"
  --out_dir "$OUT_DIR"
  --args_root "$ARGS_ROOT"
)

if [[ "$RESUME" == "1" ]]; then
  CMD_ARGS+=( --resume 1 --start_epoch "$START_EPOCH" --ckpt_dir "$CONT_CKPT_DIR" )
fi

docker run -d \
  --name "$NAME" \
  --gpus "device=${GPU_ID}" \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/data":/workspace/data \
  -v "$(pwd)/output":/workspace/output \
  --ipc=host --shm-size=2g \
  "$IMG" \
  "${CMD_ARGS[@]}" "${EXTRAS[@]}"