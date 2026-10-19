#!/usr/bin/env bash
# 容器入口：拉起**两个进程**
#   1) 常驻 C++ daemon（backbone+decoder 常驻，unix socket 收请求）
#   2) FastAPI 前端（HTTP 契约，把请求转给 daemon）
#
# daemon 崩了（例如极端显存边界）就重启它，前端不动 —— 前端会把 503/500 如实回报，
# 不会假装成功。
set -u

# 日志落盘：容器重建后 docker logs 会丢，这里同时写到挂载出来的 logs/（>32 MB 轮转一次）
LOG_DIR="${MOSS_LOG_DIR:-/var/log/moss}"
mkdir -p "$LOG_DIR" 2>/dev/null || true
if [ -w "$LOG_DIR" ]; then
  if [ -f "$LOG_DIR/app.log" ] && [ "$(stat -c %s "$LOG_DIR/app.log" 2>/dev/null || echo 0)" -gt 33554432 ]; then
    mv -f "$LOG_DIR/app.log" "$LOG_DIR/app.log.1" 2>/dev/null || true
  fi
  # 每行加 HH:MM:SS.mmm 时间戳（容器内没有 ts，用 while read 打）
  exec > >(while IFS= read -r l; do printf '%s %s\n' "$(date +%H:%M:%S.%3N)" "$l"; done | tee -a "$LOG_DIR/app.log") 2>&1
fi

MODELS_DIR="${MOSS_MODELS_DIR:-/models}"
MODEL_FILE="${MOSS_MODEL_FILE:-moss_tts_v15_firstclass_q4_k_m_aq8.gguf}"
DECODER_FILE="${MOSS_DECODER_FILE:-moss_tts_audio_decoder_f16.gguf}"
NGL="${MOSS_NGL:--1}"
SOCK="${MOSS_DAEMON_SOCK:-/run/moss/serve.sock}"
WINDOW="${MOSS_STREAM_WINDOW:-8}"
OVERLAP="${MOSS_STREAM_OVERLAP:-8}"
MAXMNT="${MOSS_SERVE_MAX_NEW_TOKENS:-448}"
UBATCH="${MOSS_PREFILL_UBATCH:-384}"   # patch 0006：prefill 单批 token 上限（0 = 不分块）
UBUDGET="${MOSS_PREFILL_BUDGET_MIB:-1400}"  # patch 0006：pp buffer 显存预算（MiB）
CTXROWS="${MOSS_CTX_ROWS:-6000}"   # patch 0009：backbone ctx 行数上限
CTXKEEP="${MOSS_CTX_KEEP:-2048}"   # patch 0009：滑窗保留的最近音频行数
PORT="${MOSS_LISTEN_PORT:-8080}"   # 容器内监听端口固定 8080；TTSPORT 只是宿主侧映射（compose 插值）

mkdir -p "$(dirname "$SOCK")"
rm -f "$SOCK"

start_daemon() {
  llama-moss-tts --serve-unix "$SOCK" \
    -m "$MODELS_DIR/$MODEL_FILE" \
    --audio-decoder-model "$MODELS_DIR/$DECODER_FILE" \
    -ngl "$NGL" \
    --serve-max-new-tokens "$MAXMNT" \
    --serve-stream-window "$WINDOW" \
    --serve-stream-overlap "$OVERLAP" \
    --serve-ubatch "$UBATCH" \
    --serve-ubatch-budget "$UBUDGET" \
    --serve-ctx-rows "$CTXROWS" \
    --serve-ctx-keep "$CTXKEEP" \
    --serve-vocab-model "$MODELS_DIR/$MODEL_FILE"
}

# 常驻 daemon：守护循环（崩了 2 秒后重来；模型会重新加载，这是有意的兜底）
(
  while true; do
    echo "[entrypoint] 启动常驻 daemon：$SOCK（window=$WINDOW overlap=$OVERLAP max_mnt=$MAXMNT ubatch=$UBATCH/${UBUDGET}MiB）"
    start_daemon
    rc=$?
    echo "[entrypoint] daemon 退出（rc=$rc），2s 后重启"
    sleep 2
  done
) &

echo "[entrypoint] 等 daemon socket 就绪..."
for _ in $(seq 1 900); do
  [ -S "$SOCK" ] && break
  sleep 0.2
done
if [ ! -S "$SOCK" ]; then
  echo "[entrypoint] ⚠️ daemon socket 没起来（前端会返回 503，但容器继续跑，方便看日志）"
fi

echo "[entrypoint] 启动 FastAPI 前端：0.0.0.0:$PORT"
exec uvicorn server:app --host 0.0.0.0 --port "$PORT"
