#!/usr/bin/env bash
# M1 冒烟/压测：在一个容器里跑常驻 daemon + 客户端，量冷启与常驻的单请求耗时。
#
#   ./scripts/bench-m1.sh
#
# 用一次性容器（runtime-base 底包），只挂：build/bin（编译产物）、
# 权重（只读）、voices（只读）、results（写）。宿主不装任何东西。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${MODELS:-./models}"
IMAGE="${IMAGE:-moss-tts-runtime-base:cu128}"
MODEL_FILE="${MODEL_FILE:-moss_tts_v15_firstclass_q4_k_m_aq8.gguf}"
WARM="${WARM:-5}"

[ -x "$ROOT/build/bin/llama-moss-tts" ] || {
  echo "缺少 build/bin/llama-moss-tts，先跑 scripts/build-llama.sh" >&2; exit 1; }

mkdir -p "$ROOT/results"
rm -f "$ROOT/results/serve.sock" "$ROOT/results/m1-"*.wav

# 宿主侧采样显存（容器里没有 nvidia-smi）
gpu_log="$ROOT/results/m1-gpu.log"
: > "$gpu_log"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$gpu_log" 2>/dev/null || true
    echo "---" >> "$gpu_log"
    sleep 0.5
  done ) &
sampler=$!
trap 'kill $sampler 2>/dev/null || true' EXIT

t0=$(date +%s)
docker run --rm --gpus all \
  -v "$ROOT/build/bin":/opt/llama/bin:ro \
  -v "$MODELS":/models:ro \
  -v "$ROOT/voices":/voices:ro \
  -v "$ROOT/results":/results \
  -v "$ROOT/scripts/bench-daemon.py":/bench-daemon.py:ro \
  -e LD_LIBRARY_PATH=/opt/llama/bin \
  -e PATH=/opt/llama/bin:/opt/venv/bin:/usr/bin:/bin \
  -e MODEL_FILE="$MODEL_FILE" \
  -e WARM="$WARM" \
  "$IMAGE" bash -lc '
    set -e
    echo "[container] 启动常驻 daemon"
    llama-moss-tts --serve-unix /results/serve.sock \
      -m "/models/$MODEL_FILE" \
      --audio-encoder-model /models/moss_tts_audio_encoder_f16.gguf \
      --audio-decoder-model /models/moss_tts_audio_decoder_f16.gguf \
      -ngl -1 > /results/daemon.log 2>&1 &
    pid=$!
    for _ in $(seq 1 600); do [ -S /results/serve.sock ] && break; sleep 0.2; done
    [ -S /results/serve.sock ] || { echo "socket 没起来"; cat /results/daemon.log; exit 1; }
    echo "[container] socket 就绪，开始压测"
    python3 /bench-daemon.py --sock /results/serve.sock \
      --models /models --voices /voices --out /results \
      --model-file "$MODEL_FILE" --steady-runs "$WARM"
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
  '
echo "整体耗时 $(( $(date +%s) - t0 ))s"

kill $sampler 2>/dev/null || true
trap - EXIT
echo
echo "=== 显存采样（MiB，含启动过程）==="
awk 'BEGIN{max=0} /^[0-9]+$/{if($1>max)max=$1} END{printf "峰值 %d MiB\n", max}' "$gpu_log"
grep -v -- '---' "$gpu_log" | awk 'NF{print}' | tail -5 | while read v; do echo "  尾部采样 $v"; done
echo
echo "daemon 日志 -> results/daemon.log ；结果 -> results/m1.json"
