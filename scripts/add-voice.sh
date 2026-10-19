#!/usr/bin/env bash
# 注册一个克隆音色：任意音频 -> voices/<id>.wav (24 kHz mono)
#   ./scripts/add-voice.sh ./some/voice.mp3 alice
#
# 全程用**应用镜像里的静态 ffmpeg**，宿主不需要安装任何东西（只要有 Docker）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?usage: add-voice.sh <src-audio> <voice-id>}"
ID="${2:?usage: add-voice.sh <src-audio> <voice-id>}"
IMG="${IMAGE:-stack-moss-tts-windowed:latest}"

case "$ID" in
  *[!A-Za-z0-9_.-]*|default) echo "bad voice id: $ID" >&2; exit 1;;
esac

docker image inspect "$IMG" >/dev/null 2>&1 || {
  echo "镜像 $IMG 不存在 —— 先跑 ./scripts/build-image.sh" >&2; exit 1; }
[ -f "$SRC" ] || { echo "找不到输入音频: $SRC" >&2; exit 1; }

mkdir -p "$ROOT/voices"
SRC_ABS="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$(dirname "$SRC_ABS")":/in:ro -v "$ROOT/voices":/out \
  --entrypoint ffmpeg "$IMG" -y -v error -i "/in/$(basename "$SRC_ABS")" \
  -ar 24000 -ac 1 "/out/$ID.wav"

echo "ok -> $ROOT/voices/$ID.wav"
docker run --rm -v "$ROOT/voices":/out --entrypoint ffmpeg "$IMG" \
  -hide_banner -i "/out/$ID.wav" 2>&1 | grep -E "Duration|Audio:" || true
