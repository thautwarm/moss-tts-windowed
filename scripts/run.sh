#!/usr/bin/env bash
# 构建并运行 MOSS-TTS v1.5 (first-class, CUDA sm_120) —— CLI 模式
#   ./scripts/run.sh -m /models/moss_tts_v15_firstclass_q4_k_m.gguf \
#       --audio-decoder-model /models/moss_tts_audio_decoder_f16.gguf \
#       --text "你好，世界" --wav-out /out/hello.wav -ngl -1
# 常驻服务请用 docker compose up -d
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${MODELS:-./models}"
OUT="${OUT:-$ROOT/out}"
REFS="${REFS:-$ROOT/refs}"
IMAGE="${IMAGE:-stack-moss-tts-windowed:latest}"
mkdir -p "$OUT"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[build] image $IMAGE not found, building..."
  docker build -f "$ROOT/docker/Dockerfile" -t "$IMAGE" "$ROOT"
fi

exec docker run --rm --gpus all \
  -v "$MODELS":/models:ro \
  -v "$REFS":/refs:ro \
  -v "$OUT":/out \
  "$IMAGE" llama-moss-tts "$@"
