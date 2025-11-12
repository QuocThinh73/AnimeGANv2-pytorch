set -euo pipefail

IMG="${1:?}"
STYLE="${2:?}"
GPU_ID="${3:?}"
RESUME="${4:?}"
START_EPOCH="${5:?}"
shift 5
EXTRAS=("$@")

NAME="cyclegan-${STYLE}"
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
  CMD_ARGS+=( --resume --start_epoch "$START_EPOCH" --ckpt_file "$CONT_CKPT_FILE" )
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