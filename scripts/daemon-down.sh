#!/usr/bin/env bash
# 停掉常驻 daemon 容器
set -euo pipefail
NAME="${NAME:-moss-tts-windowed-daemon}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "已停 $NAME"
