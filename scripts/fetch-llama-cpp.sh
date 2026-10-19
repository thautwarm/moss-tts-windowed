#!/usr/bin/env bash
# 拉取 MOSS-TTS 所需的 llama.cpp fork（OpenMOSS/llama.cpp @ moss-tts-firstclass）
#
# 用 tarball 而不是 git clone：只需要一个快照，且更快。
# 默认走镜像域名；网络正常时也可以直接用 codeload.github.com。
#
# 用 commit SHA 而不是 branch 名，保证可复现。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/upstream/llama.cpp}"

REPO="OpenMOSS/llama.cpp"
# 2026-04-08, "Remove bundled MOSS audio tokenizer fallback"（baseline: upstream b8369）
SHA="b785003ba497794ecfa337c3e47f01af79489888"
TARBALL_SHA256="86ba74788b05ea1b26728cb31fb48aef4cb5d063170257ab4f85e350fb11e637"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 按优先级尝试：镜像 -> 官方 codeload -> 官方 github
URLS=(
  "https://gh.llkk.cc/https://codeload.github.com/${REPO}/tar.gz/${SHA}"
  "https://codeload.github.com/${REPO}/tar.gz/${SHA}"
  "https://github.com/${REPO}/archive/${SHA}.tar.gz"
)

ok=0
for u in "${URLS[@]}"; do
  echo "[fetch] $u"
  rm -f "$TMP/fork.tar.gz"
  if timeout 600 curl -s -L --retry 3 --retry-all-errors -o "$TMP/fork.tar.gz" "$u" \
     && tar -tzf "$TMP/fork.tar.gz" >/dev/null 2>&1; then
    ok=1; break
  fi
done
[ "$ok" = 1 ] || { echo "全部源都失败" >&2; exit 1; }

got="$(sha256sum "$TMP/fork.tar.gz" | cut -d' ' -f1)"
if [ "$got" = "$TARBALL_SHA256" ]; then
  echo "[verify] tarball sha256 匹配 ✓"
else
  echo "[verify] tarball sha256 = $got"
  echo "         与记录的 $TARBALL_SHA256 不同（mirror 可能重新打包），继续，但请留意。" >&2
fi

rm -rf "$DEST"
mkdir -p "$DEST"
tar -xzf "$TMP/fork.tar.gz" -C "$DEST" --strip-components=1

# 功能性校验：MOSS 关键文件在不在
for f in src/models/moss-tts-delay.cpp tools/tts/run-moss-tts-delay.cpp \
         convert_hf_to_gguf.py convert_moss_audio_tokenizer_split_to_gguf.py; do
  [ -e "$DEST/$f" ] || { echo "缺少 $f —— 取到的不是 moss-tts-firstclass" >&2; exit 1; }
done

echo "[done] $DEST  ($(find "$DEST" -type f | wc -l) files)"
echo "[note] 上游 commit: $REPO @ $SHA"
