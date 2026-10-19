#!/usr/bin/env bash
# 把 patches/ 下的补丁应用到本地 upstream/llama.cpp。
#
# 平时不需要跑：docker build 会在镜像内自动应用（见 docker/Dockerfile）。
# 只有在宿主机上直接编译/调试 fork 源码时才需要。
#
#   ./scripts/apply-patches.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/upstream/llama.cpp"
[ -d "$SRC" ] || { echo "缺少 $SRC，先跑 scripts/fetch-llama-cpp.sh" >&2; exit 1; }

cd "$SRC"
for p in "$ROOT"/patches/*.patch; do
  [ -e "$p" ] || { echo "没有补丁文件"; exit 0; }
  echo ">>> $(basename "$p")"
  patch -p1 --forward --batch < "$p"
done
echo
echo "提示：upstream/llama.cpp 现在已不是 pristine 状态。"
echo "     要还原：scripts/fetch-llama-cpp.sh（重新拉取）"
