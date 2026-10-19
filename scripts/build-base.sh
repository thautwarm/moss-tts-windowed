#!/usr/bin/env bash
# 构建两个底包镜像 —— 只在换 CUDA 版本 / 改工具链或运行时依赖时才需要跑
#
#   moss-tts-toolchain:cu128       ~1-2 分钟（编译工具链）
#   moss-tts-runtime-base:cu128    ~10 分钟（含 ffmpeg 的 apt + pip 依赖）
#
# 需要走代理时（daemon 的 systemd 代理对 docker build 不生效）：
#   BUILD_PROXY=http://127.0.0.1:7897 ./scripts/build-base.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARGS=()
if [ -n "${BUILD_PROXY:-}" ]; then
  NP="${BUILD_NO_PROXY:-localhost,127.0.0.1}"
  ARGS=(--build-arg "HTTP_PROXY=$BUILD_PROXY"  --build-arg "HTTPS_PROXY=$BUILD_PROXY"
        --build-arg "http_proxy=$BUILD_PROXY"  --build-arg "https_proxy=$BUILD_PROXY"
        --build-arg "NO_PROXY=$NP"             --build-arg "no_proxy=$NP")
  echo "(走代理 $BUILD_PROXY)"
else
  echo "(未设 BUILD_PROXY，直连；若 apt/pip 很慢就带上 BUILD_PROXY=...)"
fi
echo

echo "===== 底包 A：工具链 moss-tts-toolchain:cu128 ====="
docker build -f "$ROOT/docker/Dockerfile.toolchain" -t moss-tts-toolchain:cu128 \
  "${ARGS[@]}" "$ROOT/docker"
echo
echo "===== 底包 B：运行时 moss-tts-runtime-base:cu128 ====="
docker build -f "$ROOT/docker/Dockerfile.runtime-base" -t moss-tts-runtime-base:cu128 \
  "${ARGS[@]}" "$ROOT/docker"
echo
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}' \
  | grep -E "moss-tts-toolchain|moss-tts-runtime-base"
