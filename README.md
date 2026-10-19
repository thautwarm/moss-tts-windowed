# moss-tts-windowed

**MOSS-TTS 8B 的 ≤8 GB 显存流式方案。** 模型常驻内存，边生成边出音频，
每条请求都不再重新加载权重——在一张 8 GB 显卡上就能跑 8B 的 TTS。

| | |
|---|---|
| 模型 | MOSS-TTS v1.5（8B Delay） |
| 显存 | 单卡 **≤8 GB**（实测稳态 6.5 GB，峰值 7.8 GB） |
| 流式 | Ogg/Opus、MP3、WAV、PCM，裸音频流 |
| 延迟 | 首字节 0.6 s（短句）/ 0.8 s（长句） |
| 吞吐 | 长文本 RTF **0.19–0.23** |

- 常驻 backbone（4.6 GB）+ audio decoder（1.7 GB），**每请求零权重重载**
- 长文本走**单 context**（全文一次喂入）+ 音频行滑窗，没有按句切分的拼接感
- 流式响应是**裸音频流**，不是分块协议；客户端边收边解
- 输入上限 2048 字（超出**截断**并在响应头标注），单卡串行、显式排队

---

## 1. 需要什么

| 项 | 要求 |
|---|---|
| GPU | 单卡 ≥8 GB（本项目在 8 GB 消费级显卡上验证） |
| 驱动 / 运行时 | NVIDIA 驱动 + `nvidia-container-toolkit` |
| 磁盘 | 模型 ~6.5 GB + 镜像 ~10 GB |
| 宿主 | 只需要 Docker；不需要 ffmpeg / CUDA / PyTorch / Python 依赖 |

### 模型文件

```bash
./scripts/fetch-models.sh        # 从 HuggingFace 下载（推荐）
```

三个文件放在宿主某个目录（默认 `./models`），挂载到容器 `/models`：

```
moss_tts_v15_firstclass_q4_k_m_aq8.gguf   # backbone（约 4.6 GB）
moss_tts_audio_decoder_f16.gguf           # 音频解码器（约 1.7 GB）
moss_tts_audio_encoder_f16.gguf           # 只在注册新音色时用到
```

> ⚠️ **必须是本仓库 `convert.sh` 转出的 GGUF**（`general.architecture = moss-tts-delay`）。
> 官方发布的 `MOSS_TTS_Q4_K_M.gguf` 是 **`qwen3` 架构**、把 embedding/heads 拆成 `.npy`，
> 本 fork **不能直接用**：能加载，但生成的 codes 全是 pad，输出只有零点几秒。

#### 直接下载（推荐）

```bash
./scripts/fetch-models.sh
```

从 **https://huggingface.co/thaut/moss-tts-windowed-gguf** 拉三个文件（带 sha256 校验，已存在会跳过）。
国内网络可以走镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com ./scripts/fetch-models.sh
```

#### 自己转换

也可以在本地从 HuggingFace 权重生成（CPU 转换，不需要 GPU）：

```bash
mkdir -p work/hf
# 用 huggingface-cli / git-lfs 下载到：
#   work/hf/MOSS-TTS-v1.5            ← OpenMOSS-Team/MOSS-TTS-v1.5
#   work/hf/MOSS-Audio-Tokenizer     ← OpenMOSS-Team/MOSS-Audio-Tokenizer

./scripts/convert.sh
```

产物在 `work/gguf/`：

```
moss_tts_v15_firstclass_q4_k_m_aq8.gguf   # backbone（aq8 = 音频表 Q8_0）
moss_tts_audio_encoder_f16.gguf           # 注册音色时用
moss_tts_audio_decoder_f16.gguf           # 推理时用
```

把它们放进 `MOSS_MODELS_HOST` 指向的目录即可。

> `convert.sh` 会自动构建它需要的两个镜像（CPU 转换镜像 + 应用镜像），
> 所以请**先完成第 2 节的构建**再跑它。
>
> 如果已经拿到别人的 backbone GGUF（例如官方发布的 `MOSS_TTS_Q4_K_M.gguf`），
> 只需要转编解码器：`./scripts/convert.sh --tokenizer-only`。

---

## 2. 构建

```bash
git clone <repo> moss-tts-windowed && cd moss-tts-windowed

./scripts/fetch-ffmpeg.sh        # 静态 ffmpeg（77 MB，带 libopus/libmp3lame），校验 sha256
./scripts/fetch-llama-cpp.sh     # upstream llama.cpp 源码（patches 基于它）
./scripts/build-base.sh          # 运行时底包（CUDA + python + 编译工具链）——首次约 10 分钟
./scripts/build-llama.sh         # 应用补丁并编译 daemon（有 ccache 时几秒）
./scripts/build-image.sh         # 拼装应用镜像（秒级）
```

只需要改 `server.py` / 补丁时，重复后两步即可。

**显卡架构**：`build-llama.sh` 会用 `nvidia-smi --query-gpu=compute_cap` **自动探测**。
探测不到时可以显式指定，否则默认 120：

```bash
CUDA_ARCH=86 ./scripts/build-llama.sh    # A6000 / RTX 30 系
CUDA_ARCH=89 ./scripts/build-llama.sh    # RTX 40 系
```

⚠️ 架构选错的典型表现：服务能起、但第一次合成报
`CUDA error: no kernel image is available for execution on the device`。

---

## 3. 运行

```bash
cp .env.example .env
# 改 MOSS_MODELS_HOST 指向模型目录
docker compose up -d
```

默认监听 **6677**（`.env` 里的 `TTSPORT`）。容器内是**两个进程**：uvicorn 前端 + 常驻 C++ daemon，
两者用 `/run/moss/serve.sock` 通信。

```bash
curl -s http://127.0.0.1:6677/health | jq
```

`/health` 会告诉我们：daemon 是否在线、模型是否齐全、注册了多少音色、当前队列状态、解码参数。

---

## 4. 使用

### 4.1 注册音色

零样本克隆：给一段参考音频（任意格式），脚本会转成 24 kHz 单声道并预算 codes。

```bash
./scripts/add-voice.sh ./ref/alice.mp3 alice
```

脚本会把它转成 24 kHz 单声道。转换用的是**应用镜像里的静态 ffmpeg**——
宿主不需要安装 ffmpeg，只需要 Docker（镜像需先构建好）。

也可以直接调 API：

```bash
curl -X POST http://127.0.0.1:6677/api/v1/voices \
  -F "voice=alice" -F "file=@ref/alice.wav"
```

约定（见 `voices/README.md`）：`voices/<id>.wav` + `voices/<id>.<wav sha8>.codes.bin`，**不入库**。

```bash
curl -s http://127.0.0.1:6677/api/v1/voices | jq
curl -X DELETE http://127.0.0.1:6677/api/v1/voices/alice
```

### 4.2 流式合成（推荐）

`POST /v1/audio/speech/stream` 使用 4 个字段；**多给的字段会被忽略**（不报错），方便复用别家客户端的请求体：

```jsonc
{
  "prompt": "要合成的文本（≤2048 字）",
  "voice": "alice",            // 默认 "default"（无参考音色）
  "seed": 42,                  // -1 = 随机
  "audio_format": "opus",      // wav | opus | mp3 | pcm
  "speed": 1.0                 // 0.5–2.0，默认 1.0（atempo 变速，保持音高）
}
```

响应体是**裸音频流**，`Content-Type` 随格式变化：

| `audio_format` | Content-Type | 说明 |
|---|---|---|
| `opus` | `audio/ogg` | 80 kbps，**48 kHz**（Opus 固定） |
| `mp3` | `audio/mpeg` | 128 kbps，24 kHz |
| `wav` | `audio/wav` | 24 kHz / 单声道 / 16-bit，长度字段填 `0xFFFFFFFF` |
| `pcm` | `audio/pcm` | 24 kHz / 单声道 / s16le 裸流 |

`speed` 用 ffmpeg 的 `atempo` 实现（保持音高），0.5–2.0；`1.0` 时不走任何滤镜。

```bash
# 存成文件
curl -sN -X POST http://127.0.0.1:6677/v1/audio/speech/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"你好，世界。","voice":"alice","audio_format":"opus"}' \
  -o hello.ogg

# 直接播放（边下边放）
curl -sN -X POST http://127.0.0.1:6677/v1/audio/speech/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"今天天气不错。","voice":"alice","audio_format":"opus"}' | ffplay -nodisp -
```

响应头：

```
X-Voice: alice
X-Format: opus
X-Streaming: 1
X-Sample-Rate: 48000              # opus；wav/pcm/mp3 是 24000
X-Input-Truncated: 952            # 仅当输入超 2048 字被截断时出现，值是丢弃的字数
```

### 4.3 OpenAI 兼容端点

`POST /v1/audio/speech` 保留同样的请求结构，但**只支持 `stream=true`**：

```bash
curl -sN -X POST http://127.0.0.1:6677/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好。","voice":"alice","response_format":"opus","stream":true}'
```

> 非流式（不带 `stream`）默认返回 **400**，因为 whole 模式已从对外接口下线：
> 本部署只保证流式路径。需要恢复可设 `MOSS_DISABLE_NONSTREAM=0`。

其它端点：`GET /v1/models`、`POST /api/v1/synthesize`（内部/调试用，支持非流式）。

### 4.4 语言与采样参数

这些**不在请求里**，由服务端 `.env` 统一决定（Delay-8B 的官方推荐值）：

```
MOSS_LANGUAGE=zh
MOSS_AUDIO_TEMPERATURE=1.7
MOSS_AUDIO_TOP_P=0.8
MOSS_AUDIO_TOP_K=25
MOSS_AUDIO_REPETITION_PENALTY=1.0
```

---

## 5. 配置（`.env`）

| 变量 | 默认 | 作用 |
|---|---|---|
| `TTSPORT` | `6677` | 宿主端口 |
| `MOSS_UID` / `MOSS_GID` | `1000` | 容器以宿主的 uid/gid 运行（`id -u` / `id -g`） |
| `MOSS_MODELS_HOST` | `./models` | 模型目录（宿主路径） |
| `MOSS_MODEL_FILE` | `moss_tts_v15_firstclass_q4_k_m_aq8.gguf` | backbone |
| `MOSS_DECODER_FILE` | `moss_tts_audio_decoder_f16.gguf` | 解码器 |
| `MOSS_STREAM_WINDOW` | `32` | 解码窗口推进帧数（1 帧 = 0.08 s） |
| `MOSS_STREAM_OVERLAP` | `40` | 左侧上下文帧数 |
| `MOSS_STREAM_LOOKAHEAD` | `0` | 右侧前瞻（0 = 不等待） |
| `MOSS_CTX_ROWS` | `6000` | backbone KV 行数上限（滑窗触发点） |
| `MOSS_CTX_KEEP` | `2048` | 滑窗保留的最近音频行 |
| `MOSS_MAX_INPUT_CHARS` | `2048` | 输入上限，超出截断 |
| `MOSS_QUEUE_MAX` | `8` | 队列上限（含正在跑的） |
| `MOSS_QUEUE_WAIT` | `60` | 排队超时（秒），超过返回 503 |
| `MOSS_LOG_TEXT` | `0` | 1 = 日志里记录请求文本 |
| `MOSS_DISABLE_NONSTREAM` | `1` | 1 = 非流式 `/v1/audio/speech` 返回 400 |

改完 `docker compose up -d` 生效（`.env` 通过 `env_file` 注入）。

---

## 6. 并发与限流

单卡只有一个 daemon slot（同一时刻只合成一条），前端做显式队列：

| 情况 | 行为 |
|---|---|
| 并发 ≤ `MOSS_QUEUE_MAX` | 排队，按 FIFO 依次执行 |
| 超过队列上限 | 立即 **503** + `Retry-After: 2` |
| 排队超过 `MOSS_QUEUE_WAIT` | **503** + `Retry-After: 5` |
| 客户端读得慢 | 不占 slot：音频进内存队列，生成完立刻释放计算位 |
| 客户端断开 | 立即停止生成（逐帧检测），释放 slot |

实时状态在 `/health.queue`：`inflight` / `waiting` / `rejected{queue_full,queue_timeout}`。

---

## 7. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `503 daemon 忙` | 队列满或排队超时；看 `/health.queue`，客户端应尊重 `Retry-After` |
| `health.daemon.online=false` | daemon 没起来：`docker logs` 看模型路径、显存 |
| `400 未知音色` | 音色 id 不存在；`GET /api/v1/voices` 看现有列表 |
| 声音变调 / 像男声 | 客户端按 `X-Sample-Rate` 解码；Opus 必须按 **48000** 解（或读 `OpusHead`） |
| 文本被截断 | `X-Input-Truncated` 非空；分段请求或调大 `MOSS_MAX_INPUT_CHARS`（受显存限制） |
| 首字节慢 | 首字节 = prefill + 攒够 `MOSS_STREAM_WINDOW` 帧；调小 window 会更快但接缝变多 |
| 显存不足 | 降 `MOSS_CTX_ROWS`（如 5000）或 `MOSS_STREAM_OVERLAP` |

---

## 8. 设计要点（为什么是现在这个形态）

```
HTTP :6677 ──▶ FastAPI（契约/音色/ffmpeg/队列）
                  │ unix socket
                  ▼
             C++ daemon（常驻 backbone + decoder，单 slot）
```

- **为什么常驻**：上游形态是"每请求起一次 CLI"，每次要重新反序列化 ~8.4 GB 权重，固定开销约 2.1 s。
  常驻后这部分归零，长文本 RTF 从 0.258 降到 0.216。
- **为什么必须改 fork**：上游 fork 没有 server 模式，也没有流式钩子。`patches/0003–0013` 补上了
  常驻模型、增量流式、prefill 分块、KV 滑窗、断线检测等。
- **为什么长文本用单 context**：backbone 的 `n_ctx_train = 40960`，文本根本不需要切句；
  受限的是显存（KV ≈ 0.14 MiB/行）。所以全文进前缀，只对**音频行**滑窗。
- **为什么解码要窗口**：音频 decoder 是**非因果**的（整段一次 encode），输入 `n_ctx_train` 只有 125 帧。
  窗口（默认 32+2×40=112 帧）是对它的近似，越小越省显存、越大越接近整段。
- **为什么流式不等于整段**：窗口化解码与一次性整段解码不是逐字节相同；对音质最敏感的场景
  可以把窗口调大（`MOSS_STREAM_WINDOW/OVERLAP`）用延迟换质量。

---

## 9. 目录

```
server.py              HTTP 契约 + 队列 + 流式 + ffmpeg 管道
docker/                Dockerfile / entrypoint / 工具链
patches/               0001–0013：对 upstream llama.cpp 的全部改动
scripts/               构建、契约测试、压测、音色注册、ASR 校验
docs/                  DESIGN.md（设计与实测日志）+ daemon 内部 API 笔记
voices/                音色目录（不入库，见 voices/README.md）
results/               验收报告与实测数据
```

常用：

```bash
python3 scripts/contract-test.py      # 契约自检（25 项）
./scripts/add-voice.sh <音频> <id>     # 注册音色
```

---

## License

MIT — 见 [LICENSE](LICENSE)。

依赖说明：ffmpeg 以**独立进程**方式调用（`exec`，不链接），构建时由 `scripts/fetch-ffmpeg.sh`
从上游下载，不随本仓库源码分发；上游 llama.cpp / MOSS-TTS 各自的许可证见其仓库。
