#!/usr/bin/env bash
# MOSS-TTS v1.5 自转：HF 权重 -> first-class 融合 GGUF(f16) -> Q4_K_M
#                              + MOSS-Audio-Tokenizer -> encoder/decoder GGUF
#
# 依赖（全部自建，不依赖任何私有基础镜像）：
#   - 转换镜像 moss-tts-convert:latest  <- docker/Dockerfile.convert (python:3.12-slim + CPU torch)
#   - 量化镜像 stack-moss-tts-windowed:latest <- docker/Dockerfile        (CUDA 12.8.2, 含 llama-quantize)
#   - fork 源码 upstream/llama.cpp      <- ./scripts/fetch-llama-cpp.sh
#   - 权重 work/hf/{MOSS-TTS-v1.5,MOSS-Audio-Tokenizer}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/work"
FORK="$ROOT/upstream/llama.cpp"
GGUF="$WORK/gguf"
CONV_IMAGE="${CONV_IMAGE:-moss-tts-convert:latest}"
QUANT_IMAGE="${QUANT_IMAGE:-stack-moss-tts-windowed:latest}"
QUANT="${QUANT:-Q4_K_M}"
mkdir -p "$GGUF"

F16="$GGUF/moss_tts_v15_firstclass_f16.gguf"
Q4="$GGUF/moss_tts_v15_firstclass_${QUANT,,}_aq8.gguf"   # aq8 = 音频表提到 Q8_0

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

ensure() { docker image inspect "$1" >/dev/null 2>&1 || { step "build $1"; shift; "$@"; }; }

ensure "$CONV_IMAGE"  docker build -f "$ROOT/docker/Dockerfile.convert" -t "$CONV_IMAGE" "$ROOT/docker"
ensure "$QUANT_IMAGE" docker build -f "$ROOT/docker/Dockerfile"        -t "$QUANT_IMAGE" "$ROOT"

[ -d "$FORK" ] || { echo "缺少 $FORK，先跑 ./scripts/fetch-llama-cpp.sh" >&2; exit 1; }
for m in MOSS-TTS-v1.5 MOSS-Audio-Tokenizer; do
  [ -d "$WORK/hf/$m" ] || { echo "缺少 work/hf/$m（见 README 的下载命令）" >&2; exit 1; }
done

# ── 1. 融合 backbone GGUF (f16) + codec GGUF ───────────────────────────────
if [ ! -s "$F16" ]; then
  step "convert HF -> first-class f16 GGUF (+ codec)"
  # 注意：convert_hf_to_gguf.py 自带 gguf-py（sys.path 注入），无需 pip install gguf。
  # 注意：fork docs 里写的 `--outdir` 已过时，实际是 --encoder-outfile/--decoder-outfile。
  docker run --rm -v "$WORK":/work -v "$FORK":/fork:ro -w /fork "$CONV_IMAGE" bash -lc '
    set -e
    python3 convert_hf_to_gguf.py /work/hf/MOSS-TTS-v1.5 \
        --outfile /work/gguf/moss_tts_v15_firstclass_f16.gguf --outtype f16
    python3 convert_moss_audio_tokenizer_split_to_gguf.py /work/hf/MOSS-Audio-Tokenizer \
        --encoder-outfile /work/gguf/moss_tts_audio_encoder_f16.gguf \
        --decoder-outfile /work/gguf/moss_tts_audio_decoder_f16.gguf \
        --outtype f16
  '
else
  echo "skip (exists): $F16"
fi

# ── 2. 列出 GGUF tensor（决定量化排除项）───────────────────────────────────
step "GGUF tensors"
docker run --rm --entrypoint python3 -v "$GGUF":/gguf:ro -v "$FORK":/fork:ro "$CONV_IMAGE" -c '
import sys; sys.path.insert(0, "/fork/gguf-py")
import gguf, glob
for f in sorted(glob.glob("/gguf/*firstclass_f16.gguf")):
    r = gguf.GGUFReader(f)
    print(f, len(r.tensors), "tensors")
    for t in r.tensors:
        if any(k in t.name for k in ("embd", "output", "head")):
            print("   *", t.name, tuple(t.shape))
'

# ── 3. 量化 ────────────────────────────────────────────────────────────────
# 官方 first_class Q4_K_M 也是把全部 tensor 一起量化（17GB -> 5.2GB）；
# 表（token_embd/output/32 组 audio 表，~1.54B 参数）若保 f16 会多 ~2.5GB，8GB 装不下。
#
# ⚠️ 但通用的 Q4_K_M 配方是给文本 LLM 设计的：它保护 output.weight（已 Q6_K），
#    却不知道 output_audio.* / token_embd_audio.* 这 64 张表同样是「输出头/嵌入」——
#    而它们直接决定音频 token 的 logits。所以手工把这两组提到 Q8_0（仅 +128 MiB）。
#    tensor-type 的名字是 std::regex_search 匹配，且 -- 选项必须写在位置参数**之前**。
if [ ! -s "$Q4" ]; then
  step "quantize -> $QUANT + 音频表 Q8_0"
  docker run --rm --gpus all -v "$GGUF":/gguf "$QUANT_IMAGE" \
    llama-quantize \
      --tensor-type 'output_audio=q8_0' \
      --tensor-type 'token_embd_audio=q8_0' \
      "/gguf/$(basename "$F16")" "/gguf/$(basename "$Q4")" "$QUANT"
else
  echo "skip (exists): $Q4"
fi

step "done"
ls -lh "$GGUF"
