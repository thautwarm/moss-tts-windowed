#!/usr/bin/env bash
# 构建应用镜像（秒级）：底包 B + 编译产物 + 应用代码
#
# 这个镜像里没有任何"重活" —— apt / pip / 编译全都已经做进底包或 build/bin 了。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ -x "$ROOT/build/bin/llama-moss-tts" ] || {
  echo "缺少编译产物 build/bin/llama-moss-tts —— 先跑 scripts/build-llama.sh" >&2; exit 1; }
docker image inspect moss-tts-runtime-base:cu128 >/dev/null 2>&1 || {
  echo "缺少底包 moss-tts-runtime-base:cu128 —— 先跑 scripts/build-base.sh" >&2; exit 1; }

exec docker build -f "$ROOT/docker/Dockerfile" -t "${IMAGE:-stack-moss-tts-windowed:latest}" "$ROOT"
