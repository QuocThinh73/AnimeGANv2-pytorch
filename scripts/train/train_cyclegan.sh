set -euo pipefail

STYLE="${1:?}"
GPU_ID="${2:?}"
RESUME="${3:?}"
START_EPOCH="${4:?}"
shift 4
EXTRAS=("$@")

IMG="cyclegan-trainer:cu118"
NAME="cyclegan-${STYLE,,}-${GPU_ID}"
PHOTO_ROOT="/workspace/data/train_photo"
ANIME_STYLE_ROOT="/workspace/data/${STYLE}/style"
CONFIG_FILE="/workspace/args/cyclegan.yaml"
OUT_DIR="/workspace/output/${STYLE}/cyclegan"
CONT_CKPT_FILE="/workspace/output/${STYLE}/cyclegan/checkpoints/epoch_$(printf "%03d" ${START_EPOCH})/ckpt.pth"

if docker ps -a --format '{{.Names}}' | grep -wq "$NAME" ; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

CMD_ARGS=(
  --model cyclegan
  --photo_root "$PHOTO_ROOT"
  --anime_style_root "$ANIME_STYLE_ROOT"
  --out_dir "$OUT_DIR"
  --config_file "$CONFIG_FILE"
)

if [[ "$RESUME" == "1" ]]; then
  CMD_ARGS+=( --resume 1 --start_epoch "$START_EPOCH" --ckpt_file "$CONT_CKPT_FILE" )
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