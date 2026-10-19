#!/usr/bin/env bash
# 应用 patches/ + 增量编译 llama-moss-tts / llama-quantize -> build/bin
#
# 用一次性容器（docker run --rm），增量性完全靠宿主目录保持 —— 这些目录别删：
#
#   build/src   pristine 源码副本（每次从 upstream/llama.cpp 重同步，删了只多花 ~2s）
#   build/obj   cmake build 目录  ← 增量编译的关键，删了 = 全量重编
#   build/bin   最终产物（应用镜像 COPY 的就是它）
#   ccache/     ccache 缓存      ← 删了 = 全量重编（约 145s）
#
# 正常迭代（改一行 patch）：只有被 patch 的那个 .cpp 重编 + 链接，十几秒。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="$ROOT/upstream/llama.cpp"
WORK="$ROOT/build"
TOOLCHAIN="${TOOLCHAIN_IMAGE:-moss-tts-toolchain:cu128}"
CUDA_ARCH="${CUDA_ARCH:-120}"   # 12.0 = Blackwell(RTX 50xx)；4090=89 / 3090=86
JOBS="${JOBS:-$(nproc)}"

[ -d "$UP" ] || { echo "缺少 upstream/llama.cpp，先跑 scripts/fetch-llama-cpp.sh" >&2; exit 1; }
docker image inspect "$TOOLCHAIN" >/dev/null 2>&1 || {
  echo "缺少底包 $TOOLCHAIN —— 先跑 scripts/build-base.sh" >&2; exit 1; }

# 只重建源码副本；obj/ 与 ccache/ 一律不动
rm -rf "$WORK/src"
mkdir -p "$WORK/src" "$WORK/obj" "$WORK/bin" "$ROOT/ccache"
cp -a "$UP/." "$WORK/src/"
cp "$ROOT/docker/compile-in-toolchain.sh" "$WORK/compile.sh"

t0=$(date +%s)
docker run --rm \
  -e JOBS="$JOBS" \
  -e CUDA_ARCH="$CUDA_ARCH" \
  -v "$WORK/src":/src \
  -v "$WORK/obj":/build \
  -v "$ROOT/patches":/patches:ro \
  -v "$ROOT/ccache":/ccache \
  -v "$WORK/compile.sh":/c.sh:ro \
  -w /src "$TOOLCHAIN" bash /c.sh
echo "编译耗时 $(( $(date +%s) - t0 ))s"

# 产物收拢到 build/bin（应用镜像只 COPY 这个目录，obj/ 不进构建上下文）
rm -rf "$WORK/bin"; mkdir -p "$WORK/bin"
cp -a "$WORK/obj/bin/." "$WORK/bin/"
echo
echo "产物 -> build/bin:"
ls -lh "$WORK/bin" | tail -n +2 | awk '{printf "  %8s  %s\n", $5, $9}'
