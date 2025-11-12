set -euo pipefail

IMG="${1:?}"
CONFIG_ID="${2:?}"
GPU_ID="${3:?}"
shift 3
EXTRAS=("$@")

NAME="animegan${CONFIG_ID}"
PHOTO_ROOT="/workspace/data/train_photo"
ANIME_STYLE_ROOT="/workspace/data/Shinkai/style"
ANIME_SMOOTH_ROOT="/workspace/data/Shinkai/smooth"
CONFIG_FILE="/workspace/args/animegan${CONFIG_ID}.yaml"
OUT_DIR="/workspace/output/Shinkai/animegan${CONFIG_ID}"

if docker ps -a --format '{{.Names}}' | grep -wq "$NAME" ; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

CMD_ARGS=(
  --model animegan
  --photo_root "$PHOTO_ROOT"
  --anime_style_root "$ANIME_STYLE_ROOT"
  --anime_smooth_root "$ANIME_SMOOTH_ROOT"
  --out_dir "$OUT_DIR"
  --config_file "$CONFIG_FILE"
)

docker run -d \
  --name "$NAME" \
  --gpus "device=${GPU_ID}" \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/data":/workspace/data \
  -v "$(pwd)/output":/workspace/output \
  --ipc=host --shm-size=2g \
  "$IMG" \
  "${CMD_ARGS[@]}" "${EXTRAS[@]}"