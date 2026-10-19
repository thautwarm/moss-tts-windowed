# 长上下文 + decoder 窗口改造（2026-10-18）

## 做了什么

| 补丁 | 内容 |
|---|---|
| `0009-backbone-sliding` | backbone 单 ctx；instruction+参考 codes+全文文本作为**固定前缀永不淘汰**；ctx 满时 `memory_seq_rm` 删最老**音频行** + `memory_seq_add` 平移位置；切点在目标 ±16 帧内取**能量最低帧**（能量在流式解码时逐帧记录） |
| `0011-abort-flag-reset` | 每个请求都复位 `g_moss_stream_aborted`（原来只在流式请求里复位，会把后续整段请求毒死成 0 帧/500） |
| `0010-tokens-op` | daemon 新增 `TOKENS\t<text>`，用**常驻词表**精确分词（vocab_only，不占显存）；前端据此决定解码路径与生成预算 |
| 前端 | 不再按句切段；`FRAMES_PER_TOKEN=3.6`；**契约上限 4000→2048 字**（用户确认） |

输入：`MOSS_MAX_INPUT_CHARS=2048`，超出**截断**（不 422），响应头 `X-Input-Truncated` 写明丢弃字数。

配置：`MOSS_CTX_ROWS=6000`、`MOSS_CTX_KEEP=2048`、`MOSS_SYNTH_WINDOW=32`、`MOSS_SYNTH_OVERLAP=40`（窗口 128 帧 = decoder 训练上下文 125 帧）。

## 验收

| AC | 结果 | 证据 |
|---|---|---|
| AC1 短文本 md5 | ✅ 不变（与改造前同一请求逐字节一致） | 改造前后对比 |
| AC2 长文本无丢句 | ⚠️ **284 字不重复文本 ASR 覆盖率 0.996**；620 字可用；原定 4000 字场景因显存改为 **2048 字上限**，**超限截断而非报错**（响应头 `X-Input-Truncated=<丢弃字数>`，并在句末切） | `/tmp/uniq.wav`、`/tmp/p2-mid.wav` |
| AC3 流式首字节 | ✅ 短句 **0.218 s**；284 字 **0.348 s** | curl time_starttransfer |
| AC4 帧相关 ≥0.90 | ⏳ **口径不可测**：单段 whole 参考最多 448 帧（35.8 s），而窗口路径只在 >448 帧时启用。历史 O=64 配置测得 0.95（14 s 片段，非同一路径） | — |
| AC5 边界 |Δ| ≤1.2× | ⏳ 未在 128 帧窗口下复测（历史 O=64/L=0 为 1.33×） | — |
| AC6 契约 | ✅ **25/25** | `scripts/contract-test.py` |
| AC7 显存 | ✅ 峰值 7731 MiB（620 字）/ 7623 MiB（4000 字），< 7.8 GB | nvidia-smi 采样 |
| AC8 RTF | ✅ 284 字 0.198；620 字 0.201；4000 字 0.234；短句 ≈0.38 | 上述请求 |

## TOKENS op 实测（说明为什么必须精确分词）

| 文本 | 字数 | token | 比例 |
|---|---|---|---|
| 你好世界 | 4 | 2 | 0.50 |
| hello world this is a test | 26 | 6 | 0.23 |
| 31 字中文 | 31 | 23 | 0.74 |

按“字数”估，英文会高估 3–4 倍 → 预算撑爆；改用词表后方差消失。

## 踩过的坑（都已修）

1. `emit_range` 的 O 用在**左右两侧**，实际窗口是 `W+2O`；patch 0008 按 `W+O` 建 ctx →
   `GGML_ASSERT(encoder requires n_ubatch >= n_tokens)` 崩溃。
2. ctx 9500/7500 行在 8 GB 上 KV 分配或 decode 阶段 OOM，**6000 行才稳**。
3. ASR 覆盖率早先偏低是测试方法错：`-f segment -c copy` 切 wav 不重写头 → 后续分段转写全空；
   改 `-c:a pcm_s16le` 后 0.996。

## 本轮新增验证

- 超限截断：3000 字 → **200**，`x-input-truncated: 952`，首字节 0.616 s
- 断线取消：`curl -m 3` 中断后 GPU 立刻 0%，下一条请求 200 / 1.06 s（patch 0011 修复）
- 契约 **25/25**（最终镜像）

## 仍未完成

- 4000 字路径：已按用户决定降为 2048 上限；4000 的端到端只在旧上限下测过（撞 20000 帧硬上限，未自然结束）。
- AC4/AC5 没有在最终窗口配置下测（AC4 口径本身需要重新定义）。
- `/v1/audio/speech/stream`（msgpack 分块）未在 0009/0010 之后回归。
