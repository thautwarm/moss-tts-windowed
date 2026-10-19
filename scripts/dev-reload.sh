#!/usr/bin/env bash
# 迭代闭环：增量编译 -> 重建薄应用镜像 -> 重启服务 -> 健康检查
#
# 改完 patches/ 或 server.py 后跑这一条即可。
#   BUILD=0 跳过编译（只改了 server.py 时用）：BUILD=0 ./scripts/dev-reload.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${BUILD:-1}" = "1" ]; then
  "$ROOT/scripts/build-llama.sh"
  echo
fi
"$ROOT/scripts/build-image.sh"

echo
cd "$ROOT"
docker compose up -d
sleep 5
echo
if curl -sf -m 10 localhost:6677/health; then
  echo
else
  echo "⚠️  健康检查失败，看日志： docker logs --tail 50 stack-moss-tts-windowed" >&2
  exit 1
fi
