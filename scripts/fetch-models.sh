#!/usr/bin/env bash
# 下载转换好的 MOSS-TTS v1.5 first-class GGUF（默认从 HuggingFace 拉）。
#
#   ./scripts/fetch-models.sh                 # 下到 ./models
#   MOSS_MODELS_HOST=/data/models ./scripts/fetch-models.sh
#   HF_ENDPOINT=https://hf-mirror.com ./scripts/fetch-models.sh   # 国内镜像
#
# 三个文件都会校验 sha256；已存在且正确的会跳过。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${MOSS_MODELS_HOST:-$ROOT/models}"
REPO="${MOSS_GGUF_REPO:-thaut/moss-tts-windowed-gguf}"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
BASE="$ENDPOINT/$REPO/resolve/main"

mkdir -p "$DEST"

# 文件:sha256
FILES="
moss_tts_v15_firstclass_q4_k_m_aq8.gguf:ae09b5f4b4603d6e61edd2ff0024bb521239680819a956a0d16ee18c47fe7a3e
moss_tts_audio_decoder_f16.gguf:84034fa745ee38bc0bacfde5904e19c2db3fb402e5fe079c655550b1f9b189c4
moss_tts_audio_encoder_f16.gguf:8ca93e721ac95a56a64661b1fddfd990a8054abec58ac522a0103b8923902b46
"

fail=0
for entry in $FILES; do
  name="${entry%%:*}"
  want="${entry##*:}"
  out="$DEST/$name"

  if [ -f "$out" ] && echo "$want  $out" | sha256sum -c - >/dev/null 2>&1; then
    echo "跳过（已存在且校验通过）：$name"
    continue
  fi

  echo "下载 $name ..."
  if ! curl -fL --retry 3 --retry-all-errors -o "$out.part" "$BASE/$name"; then
    echo "  下载失败：$BASE/$name" >&2
    rm -f "$out.part"; fail=1; continue
  fi
  if ! echo "$want  $out.part" | sha256sum -c - >/dev/null 2>&1; then
    echo "  sha256 不匹配，已丢弃：$name" >&2
    rm -f "$out.part"; fail=1; continue
  fi
  mv "$out.part" "$out"
  echo "  ok -> $out  ($(du -h "$out" | cut -f1))"
done

echo
ls -lh "$DEST"/*.gguf 2>/dev/null | awk '{printf "  %6s  %s\n", $5, $9}'
[ "$fail" = 0 ] || { echo; echo "有文件未取到（见上面错误）" >&2; exit 1; }
echo
echo "下一步：cp .env.example .env（设 MOSS_MODELS_HOST），然后 docker compose up -d"
