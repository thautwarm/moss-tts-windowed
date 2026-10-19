#!/usr/bin/env bash
# 为已注册音色预计算参考 codes -> voices/<id>.<参考音频sha8>.codes.bin
#
# 之后合成时可以直接 --reference-codes，跳过 1.7GB 的 audio encoder 加载
# 和参考音频编码（实测省 0.4~0.5s/请求）。
#
# 文件名里带的是**参考音频内容的 sha256 前 8 位**，所以：
#   - 换了参考音频 -> 哈希变了 -> 旧 codes 自动不会被采用（服务端按名查找）
#   - 导入/拷错别人的 codes -> 名称对不上 -> 同样不会被采用
# 也就是说「codes 与音色张冠李戴」在结构上不可能发生，不依赖任何校验或人工纪律。
#
#   ./scripts/precompute-codes.sh              # 给所有需要的音色生成
#   ./scripts/precompute-codes.sh voice-a voice-b # 只处理指定的
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${MOSS_MODELS_DIR_HOST:-./models}"
MODEL="${MOSS_MODEL_FILE:-moss_tts_v15_firstclass_q4_k_m_aq8.gguf}"
ENC="${MOSS_ENCODER_FILE:-moss_tts_audio_encoder_f16.gguf}"
DEC="${MOSS_DECODER_FILE:-moss_tts_audio_decoder_f16.gguf}"
IMG="${IMAGE:-stack-moss-tts-windowed:latest}"

[ -d "$ROOT/voices" ] || { echo "没有 voices/ 目录" >&2; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "缺少镜像 $IMG" >&2; exit 1; }

sha8() { sha256sum "$1" | cut -c1-8; }

# 决定要处理哪些音色
if [ "$#" -gt 0 ]; then
  ids=("$@")
else
  ids=()
  for w in "$ROOT"/voices/*.wav; do
    [ -e "$w" ] || continue
    id="$(basename "$w" .wav)"
    [ -f "$ROOT/voices/$id.$(sha8 "$w").codes.bin" ] || ids+=("$id")
  done
fi

if [ "${#ids[@]}" -eq 0 ]; then
  echo "所有音色都已有匹配当前参考音频的 codes，无需处理。"
  echo "（强制刷新：./scripts/precompute-codes.sh voice-a）"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for id in "${ids[@]}"; do
  WAV="$ROOT/voices/$id.wav"
  [ -f "$WAV" ] || { echo "  跳过 $id：没有 $id.wav" >&2; continue; }
  H="$(sha8 "$WAV")"
  DST="$ROOT/voices/$id.$H.codes.bin"
  printf '  %-12s sha8=%s  ' "$id" "$H"

  if ! docker run --rm --gpus all \
    -v "$MODELS":/models:ro \
    -v "$WAV":/ref.wav:ro \
    -v "$TMP":/out \
    --entrypoint llama-moss-tts "$IMG" \
    -m "/models/$MODEL" \
    --audio-encoder-model "/models/$ENC" \
    --audio-decoder-model "/models/$DEC" \
    --reference-audio /ref.wav \
    --dump-reference-codes "/out/new.codes.bin" \
    --text "。" --max-new-tokens 1 --seed 1 \
    --wav-out /out/_discard.wav -ngl -1 > "$TMP/run.log" 2>&1; then
    echo "失败"
    sed -n '$p' "$TMP/run.log" | sed 's/^/                /'
    continue
  fi

  if [ -s "$TMP/new.codes.bin" ]; then
    cp "$TMP/new.codes.bin" "$DST"
    printf -- '-> %s (%d B)\n' "$(basename "$DST")" "$(stat -c%s "$DST")"
    # 清理这个 id 下对不上的旧 codes（换了参考音频后留下的）
    for old in "$ROOT/voices/$id".*.codes.bin; do
      [ -e "$old" ] || continue
      [ "$old" = "$DST" ] || { rm -f "$old"; echo "               清理旧文件 $(basename "$old")"; }
    done
  else
    echo "失败（没有产出）"
  fi
done

echo
ls -l "$ROOT/voices"/*.codes.bin 2>/dev/null | awk '{printf "  %8s  %s\n", $5, $9}'
