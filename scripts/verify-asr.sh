#!/usr/bin/env bash
# 用外部 ASR 服务做 round-trip 校验：合成音频 -> 转写 -> 对比原文
#   ./scripts/verify-asr.sh out/hello.wav "你好，世界"
set -euo pipefail
WAV="${1:?usage: verify-asr.sh <wav> [expected text]}"
EXPECT="${2:-}"
ASR="${ASR:-http://127.0.0.1:8082/transcribe}"

echo "== wav info =="
ffprobe -v error -show_entries stream=sample_rate,channels,duration -of default=nw=1 "$WAV" 2>/dev/null || ls -l "$WAV"

echo "== ASR ($ASR) =="
TXT=$(curl -s -F "file=@${WAV}" "$ASR" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("text",""))
except Exception: print("")')
echo "transcript: $TXT"
[ -n "$EXPECT" ] && echo "expected  : $EXPECT"
