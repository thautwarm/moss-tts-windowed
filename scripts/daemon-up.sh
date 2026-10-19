#!/usr/bin/env bash
# 起/停常驻 daemon 容器（socket 落在 results/serve.sock，宿主可直接连）
#
#   ./scripts/daemon-up.sh     # 起（detached）
#   ./scripts/daemon-down.sh   # 停
#
# 注意：daemon 容器长时间占住整张卡（常驻 backbone + decoder ≈ 6.3 GB），
# 与任何别的 GPU 服务互斥。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${MODELS:-./models}"
IMAGE="${IMAGE:-moss-tts-runtime-base:cu128}"
MODEL_FILE="${MODEL_FILE:-moss_tts_v15_firstclass_q4_k_m_aq8.gguf}"
NAME="${NAME:-moss-tts-windowed-daemon}"

mkdir -p "$ROOT/results"
rm -f "$ROOT/results/serve.sock"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --user "$(id -u):$(id -g)" \
  -v "$ROOT/build/bin":/opt/llama/bin:ro \
  -v "$MODELS":/models:ro \
  -v "$ROOT/voices":/voices:ro \
  -v "$ROOT/results":/results \
  -e LD_LIBRARY_PATH=/opt/llama/bin \
  -e PATH=/opt/llama/bin:/opt/venv/bin:/usr/bin:/bin \
  "$IMAGE" bash -lc "
    exec llama-moss-tts --serve-unix /results/serve.sock \
      -m '/models/$MODEL_FILE' \
      --audio-decoder-model /models/moss_tts_audio_decoder_f16.gguf \
      -ngl -1
  " >/dev/null

for _ in $(seq 1 600); do
  [ -S "$ROOT/results/serve.sock" ] && { echo "daemon 就绪：unix://$ROOT/results/serve.sock"; exit 0; }
  sleep 0.2
done
echo "daemon 没起来，看日志：" >&2
docker logs --tail 30 "$NAME" >&2
exit 1
