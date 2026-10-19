#!/usr/bin/env bash
# 下载静态 ffmpeg（eugeneware/ffmpeg-static b6.1.1）到 docker/bin/，并校验 sha256。
# 用法： ./scripts/fetch-ffmpeg.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/docker/bin/ffmpeg"
SHA="e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
URL="https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-linux-x64"

mkdir -p "$(dirname "$DEST")"
if [ -f "$DEST" ] && echo "$SHA  $DEST" | sha256sum -c - >/dev/null 2>&1; then
  echo "已存在且校验通过：$DEST"
  exit 0
fi
echo "下载 $URL"
curl -fSL -o "$DEST.tmp" "$URL"
echo "$SHA  $DEST.tmp" | sha256sum -c -
mv "$DEST.tmp" "$DEST"
chmod +x "$DEST"
echo "OK: $DEST ($(du -h "$DEST" | cut -f1))"
