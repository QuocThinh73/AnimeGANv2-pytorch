set -euo pipefail

STYLE="${1:?}"
GPU_ID="${2:?}"
BATCH="${3:-8}"
NUM_WORKERS="${4:-4}"
OUT_ROOT="${5:-/workspace/output}"

IMG="cyclegan-trainer:cu118"
NAME="cyclegan-${STYLE,,}-gpu${GPU_ID}"
PHOTO_ROOT="/workspace/data/train_photo"
ANIME_STYLE_ROOT="/workspace/data/${STYLE}/style"
OUT_DIR="${OUT_ROOT}/${STYLE}"

if docker ps -a --format '{{.Names}}' | grep -wq "$NAME" ; then
  echo "Container $NAME exists. Removing..."
  docker rm -f "$NAME" >/dev/null 2>&1 || true
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
  --model cyclegan \
  --photo_root "$PHOTO_ROOT" \
  --anime_style_root "$ANIME_STYLE_ROOT" \
  --batch_size "$BATCH" \
  --num_workers "$NUM_WORKERS" \
  --out_dir "$OUT_DIR"

echo "Started: $NAME"
echo "Logs:    docker logs -f $NAME"
echo "Exec:    docker exec -it $NAME bash"
