> 这是本项目早期的**设计与实测日志**（里程碑、显存账、逐次修复记录），保留供追溯。
> 使用说明见仓库根目录的 README.md。

# moss-tts-stream —— MOSS-TTS v1.5 的**常驻 + 流式**实验栈（本机 8 GB）

> 目标：在**一张 8 GB 显卡**上把 MOSS-TTS 的形态推到极限 —— 干掉每请求 2.1 s 权重重载、
> 做到**边生成边出音频**，看能做到什么水平。
> 状态：**侦察完成 + 骨架就绪**，C++ 改动进行中（里程碑见下）。
> 源栈（只读参考）：对照实现（早期版本形态是"每请求起一次 CLI"，无 server、无流式）

## 一、为什么要另开一个栈

早期的对照实现 的形态（实测）：
- FastAPI + **每请求起一次 `llama-moss-tts` 子进程**；进程退出显存归零（空闲 27 MiB）
- 权重 4.97 GiB（Q4_K_M + 音频表 Q8_0）+ encoder 1.65 GiB + decoder 1.77 GiB，**分三段加载、不同时驻留**
- 峰值 5365 MiB / 8151；**RTF 0.29（24 s 长句）～ 0.57（短句）** ← 与长度相关，因为每请求要重新
  反序列化并上传 ~8.4 GB 权重（backbone 1.56 s + encoder 0.56–0.78 s），**固定开销约 2.1 s**
- **fork 没有 server 模式**（编译目标只有 `llama-moss-tts` / `llama-quantize`），
  且**没有任何流式钩子**（`grep callback|stream` 只命中 `ifstream/ofstream`）：
  流程是「加载 encoder→编码参考 → 加载 backbone→**自回归生成完整 token 串** →
  加载 decoder→**一次性 `llama_encode` 整串 codes** → 写 wav → 退出」

→ 所以"常驻 + 流式"**必须自己改 fork**，这就是本栈存在的理由。

## 二、显存账（决定设计红线）

下表是 **M1 实测**（不是估算，`results/daemon.log` + `results/m1-gpu.log` 为证）：

| 项 | 实测 |
|---|---|
| CUDA 上下文（空进程） | ~0.55 GB（空闲 7599 MiB / 8151） |
| backbone 常驻 | **4603 MiB**（CUDA0 model buffer，另有 478 MiB host-mapped 不算显存） |
| decoder 常驻 | **1693 MiB** |
| daemon 稳态（两模型在、无请求） | **≈7167 MiB** |
| decoder 每请求 compute buffer | **0.3 GB @n_ctx 256 / 1.7 GB @768 / 2.2 GB @1024 / 4.2 GB @1280** |
| 单请求总峰值 | mnt=448 时 ~7.4 GB；mnt=576 时 **7.74 GB**（只余 400 MiB）|

⚠️ **原设计里「每请求 KV 约 0.3 GB」是错的**。decoder 是 `llama_set_causal_attn(ctx, false)`
的**非因果**编码器，整段 codes 必须**一次性** `llama_encode`（`n_batch = n_ctx`），
所以它的 compute buffer 直接随 codes 长度线性涨。后果：

- **单请求长度上限 ≈ 43 s 音频**（`--max-new-tokens` ≲ 576）。再长 `cudaMalloc` 失败；
  实测 mnt=640 时在 `llama_encode` 里 cublas handle 创建失败 → **进程直接 abort**。
  → 前端按句切分 + daemon 侧护栏（M1.5）是**必须项**，不是优化项。
- **encoder（+1693 MiB）绝对不能与常驻对共存**。运行时参考音频**只走预计算 codes**
  （`<id>.<wav指纹>.codes.bin`）。丢 wav 注册新音色（`POST /api/v1/voices`）时要
  **临时驱逐 backbone → 编码 → 重新加载 backbone**（M3）。
- 附带一个坑：`--reference-codes` 要**单独给**（同时给 `--reference-audio` 会走指纹校验，
  对不上就退回现场编码 → 拉起 encoder → OOM）。

**红线**：daemon 只常驻 `backbone + decoder`，单请求 codes 长度另有硬上限。

⚠️ 与另一套 TTS 的关系：常驻 6.5 GB **不再让出显存** → 与 另一套 TTS（6677，空闲 4.4 GB）
**不能同时在线**。这是"拉满"换来的代价，必须二选一。

## 二·五、与 对照实现 的关系：**互斥替代，同端口 6677**

- 本栈**不是并存方案，是替代方案**：对外端口就是 **6677**（与 对照实现 完全一致），
  容器内仍 8080，端口可由 `TTSPORT` 覆盖但**默认 6677**。
- **互斥**：两者都占 6677，**起这个之前先停那个**（端口冲突本身就是闸门，不需要任何切换脚本）：
  `cd 对照实现目录 && docker compose down` → `cd . && docker compose up -d`
- 同理与本机的另一套 TTS 也是互斥（它也占 6677，且空闲占 4.4 GB）。
- 因此本栈的验收包含一条：**契约必须与 对照实现 逐字段一致（同端口 drop-in）**，
  客户端切过来只改"谁在跑"，不改任何请求。
- 回滚：`cd . && docker compose down && cd 对照实现目录 && docker compose up -d`

## 二·六、M1 实测结果（A/B，同参数同 seed 同音色）

命令：`scripts/daemon-up.sh` → `python3 scripts/ab.py --mode unix --sock results/serve.sock …`，
早期版本侧 `cd 对照实现目录 && docker compose up -d` → `--mode http --url http://127.0.0.1:6677`。

| 请求 | 早期版本 对照实现 | 本栈常驻 daemon | Δ |
|---|---|---|---|
| 长句（38.5 s 音频，mnt 512） | 9.75–9.80 s，RTF **0.258** | 7.55–8.29 s，RTF **0.216** | **−1.5~1.7 s** |
| 短句（4.3 s 音频，mnt 256） | 2.85–3.28 s，RTF **0.72** | 1.31–1.55 s，RTF **0.33** | **−1.6 s** |
| 首请求（冷启，含 3 段加载） | 9.92 s / 3.00 s | 9.77 s / 1.41 s | 冷启只付一次 |

- **输出逐字节相同**：同 seed 的 wav 字节数在两边完全一致（1847084 / 1812524 / 207404 / 176684…）
- **生成本身两边都是 RTF≈0.21**，差值 100% 来自「不再加载权重」——这也解释了为什么短句收益比例最大
- 冷启代价 ≈1.5–2.2 s（`resident: loaded and kept model …` 三条），之后每次都是 `resident: reuse model …`

## 三、目标形态与里程碑

```
客户端 ──HTTP──▶ FastAPI 前端(6677) ──unix socket/HTTP──▶ moss-tts-daemon（常驻 6.5 GB）
                  契约/音色表/转码/      请求：text + 音色 + codes           │
                  X-* 头/流式分帧                                    backbone+decoder 常驻
```

| # | 里程碑 | 内容 | 预期收益 | 状态 |
|---|---|---|---|---|
| M0 | 骨架 | 目录/资产复用/设计文档 | — | ✅ 完成 |
| **M1** | **常驻 daemon** | `patches/0003`：`llama_model_free()` 对常驻登记表短路 + 进程级模型缓存 + `--serve-unix` 请求循环（复用完整 CLI 路径） | 长句 RTF 0.258→**0.216**、短句 0.72→**0.33** | ✅ 完成 |
| **M1.5** | **护栏** | `patches/0004` 里：非流式请求超 `--serve-max-new-tokens`（默认 448）直接 **ERR**（实测 mnt=640 前会 abort 整个进程） | 消掉 abort 风险 | ✅ 完成 |
| **M2** | **增量解码 + 真流式** | `patches/0004`：AR 循环里每 8 帧把已定稿的 raw codes 按滑动窗口（两侧各 8 帧上下文）解码，边生成边写 PCM；`llama_memory_clear` 复位 KV | **首块 839–858 ms**（长文本也一样）、流式无长度上限 | ✅ 完成 |
| **M3** | **契约前端** | `server.py`（从 moss-tts 逐字段移植）+ `docker-compose.yml`：一个容器两个进程（daemon + uvicorn），unix socket 通信；`stream=true` 本栈支持 | 25/25 契约自测通过；whole 模式 HTTP 层 **md5 逐字节相同** | ✅ 完成 |
| **M3.5** | **prefill 分块** | `patches/0006`：backbone 的 `n_ubatch` 不再等于 prompt 全长，改为按「显存预算 − KV」动态折算 | 长文本从 **必 500** 变成可用（4000 字 / 2707 tokens）；短文本仍是单批，md5 基线不动 | ✅ 完成 |
| **M3.7** | **长上下文 + 滑窗** | `patches/0009`：不再按句切段；全文进 prompt 作固定前缀，只滑音频行（`seq_rm`+`seq_add`，切点取低能量帧）。`patches/0010`：`TOKENS` op 精确分词。decoder 窗口 72→128 帧 | 284 字 ASR 覆盖 **0.996**；620 字 RTF 0.201；4000 字能跑完但撞生成上限 → **契约上限降到 2048 字** | ✅ |
| **M3.6** | **opus/mp3 真流式** | `docker/bin/ffmpeg`（静态 77 MB，校验和固定）+ `server.py` 常驻编码管道；`patches/0007` 忽略 SIGPIPE | `stream=true` 的 opus/mp3 从"缓冲后整段"变成**连续流**，首字节 **0.23 s** | ✅ 完成 |
| M4 | 压榨 | 固定开销归因（prompt prefill ≈ 0.35–0.45 s 是短句 RTF 的主因）、参考 codes 缓存、窗口调参 | 短句 RTF 0.33→0.25 以内 | ⬜ |
| M5 | 评测报告 | 与早期版本的完整对照（RTF/首块/显存/音质）+ 人耳试听 | 一页纸结论 | ⬜ 部分（数字齐了，缺人耳） |

## 三·五、M2/M3 实测（全链路 HTTP，2026-10-18）

同一文本、同参数、同 seed、同音色（voice-A，走预计算 codes）：

| 请求 | 早期版本 对照实现 | 本栈（HTTP，流式模式） | Δ |
|---|---|---|---|
| 长句 38.5 s | 9.75–9.80 s，RTF **0.258** | 7.61–8.39 s，RTF **0.219** | **−1.5 s** |
| 短句 4.3 s | 2.85–3.28 s，RTF **0.72** | 1.35–1.62 s，RTF **0.34** | **−1.5 s** |
| 同一长请求（mnt=320，whole 模式） | 9.30 s | 5.04 s（相同请求） | **−4.3 s** |

- **逐字节相同**：whole 模式经 HTTP 取回的 wav 与早期版本 **md5 完全一致**
  （`31c83bb2cf5da90ef04b13ff14b82da5`，706604 B，`results/http-ours-whole.wav` vs `results/http-incumbent.wav`）
- **真流式**：`stream=true`（pcm/wav）首块 **839 ms（短句）/ 851 ms（长句）**，块间隔 ≈125 ms（每块 0.64 s 音频）；
  56.3 s 的长请求也能流（这条长度在非流式路径上会 OOM）
- **音质**：流式 vs whole 两路，ASR 字符覆盖率 **1.000 vs 0.988**（流式不差）；
  块边界 `|Δ|` 统计与全局分布一致（均值 431 vs 432，最大 4150 vs 9600）→ **没有可检出的边界爆音**
- **显存**：稳态 6531 MiB（流式窗口 ctx 极小）；whole 模式长请求峰值 7.74 GB
- **固定开销归因**：短句 1.03 s（无参考）/1.26 s（voice-A）/1.32 s（voice-B），其中约 0.35–0.45 s 是
  **prompt prefill**（参考音频越长，prompt 越长：voice-A 175 帧 ≈ +0.23 s/请求），这是短句 RTF 达不到 0.25 的主因
- **注册新音色**：`POST /api/v1/voices` 在 **CPU** 上预算 codes（`-ngl 0`，实测约 2 s），不占显存、不与常驻对抢卡

## 三·六、M3.5：长文本为什么曾经必 500，以及怎么修的（2026-10-18）

**现象**：`input` 超过约 300 字直接 500，8~20 ms 返回，日志是

```
sched_reserve: worst-case: n_tokens = 508
allocating 1099.53 MiB on device 0: cudaMalloc failed: out of memory
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
```

**根因**：backbone ctx 把 `n_ubatch` 设成了整个 prompt（`cparams.n_ubatch = n_batch = prompt_frames`），
而 pp compute buffer 的大小 ∝ `n_ubatch`（实测 **≈2.2 MiB/token**）。508 token 就要 1.1 GB，
而常驻队已占 6531 / 8151 MiB。**KV 只占 396 MiB，不是 KV 的锅。**

**修法**（`patches/0006`）：按 `n_ctx` 估算 KV（0.1406 MiB/cell），把剩余预算折算成 ubatch 上限：

```
ubatch = min(配置上限 384, (预算 1000 MiB − KV估算) / 2.2)
```

- prompt ≤ ubatch 时仍然是**单批** → 与打补丁前**逐字节一致**（短文本 md5 基线不动，实测已验证）；
- prompt 更大才切，llama.cpp 的 `llama_decode` 会自动把 batch 拆成 ≤ubatch 的顺序 ubatch，KV 累积语义不变；
- **不能**动音频 encoder/decoder 的 ctx：它们是 `causal_attn=false`，非因果整段注意力切 ubatch 会截断可见性，结果会算错。

**代价（实测）**：分块 prefill 会让 logits 有浮点重排（Gated Delta Net 的 chunked 路径），
所以**超过 ubatch 的请求，输出可能与旧版差一帧**（末尾 EOS 抖动）。短文本落在单批里，不受影响。
预算常量是这台机器实测出来的（8 GB / Q4_K_M / 36 层 f16 KV），换卡要重标。

| 请求 | 修复前 | 修复后 |
|---|---|---|
| 短句（278 tokens，voice-A） | 200，md5 `8823b554…` | 200，md5 **`8823b554…`（不变）** |
| 384 字（508 tokens） | **500 / 8 ms** | 200，8.5 s |
| 4000 字（2707 tokens）+ mnt 512 | **500** | 200，9.8 s |
| 4000 字 + mnt 2048 | **500** | 200，36.7 s，峰值 **7547 MiB** |
| 4000 字 流式（`/v1/audio/speech/stream`） | **500** | 200，首字节 **1.29 s**，总 36.9 s |
| 短句 流式 | — | 200，首字节 **0.22 s** |
| 契约自测 | — | **25/25** |

## 三·七、M3.6：opus/mp3 真流式（2026-10-18）

`stream=true` 原先只有 pcm/wav 是真流；opus/mp3 要先收完整段 PCM 再转码，`X-Streaming: buffered`。
现在改成**常驻编码管道**：daemon PCM → `asyncio.create_subprocess_exec` → stdout 直接进响应。

**为什么必须换 ffmpeg**：`opusenc` 实测**不能流式**（stdin 不关就是 0 字节，直到 EOF 才吐）。
镜像里原本没有 ffmpeg，现在把 `eugeneware/ffmpeg-static` b6.1.1 的静态二进制（77 MB，自带 libopus/libmp3lame）
放进 `docker/bin/ffmpeg`，Dockerfile 里用固定 sha256 校验。

**两个必须记住的坑**：

1. **`-probesize 32 -analyzeduration 0` 不能省**。ffmpeg 默认会先"探测"输入（`analyzeduration` 5 s），
   管道输入下实测要攒到 **4.25 s** 才吐第一个字节；加上这两个参数后首包 **0.06 s**。
   `-page_duration 20000` = 20 ms 一页，进一步压低首包。
2. **daemon 必须忽略 SIGPIPE**（`patches/0007`）。下游管道一断、前端关 socket，daemon 正在写 PCM
   → SIGPIPE 默认**杀进程**（日志 `daemon 退出（rc=141）`），patch 0005 的写失败检查根本没机会跑。
   忽略之后 `write()` 返回 EPIPE，patch 0005 会当成"客户端走了"，提前收手（实测：跑到第 48 步停下）。

**实测**

| 请求 | 首字节 | 总耗时 | 产物 |
|---|---|---|---|
| 短句 opus | **0.235 s** | 1.25 s（3.2 s 音频） | 31 KB Ogg/Opus，解码正常 |
| 短句 mp3 | **0.225 s** | 1.25 s | 13 KB MP3 |
| 4000 字 opus | **1.09 s** | 9.66 s（38.5 s 音频） | 377 KB Ogg/Opus |
| 客户端 1.5 s 断开 | — | — | daemon 记 `客户端已断开（已跑 48 步）`，**没重启** |
| 中途 `pkill` daemon | — | — | 客户端拿到 63 KB 后流结束；8 s 内 daemon 自愈，新请求正常 |

**代价（说清楚）**：HTTP 200 一旦发出，后续失败只能表现为**流被截断**——客户端无法区分
"正常结束"和"daemon 中途挂了"。要能区分就得回到分块 schema（`/v1/audio/speech/stream`，仍保留）。

## 三·八、M4：长上下文（2026-10-18）

**思路转变**：`backbone n_ctx_train=40960`，文本根本不需要切句；`decoder n_ctx_train=125 帧` 才是窗口上限。

```
文本(全文) + 参考 codes ──▶ backbone 单 ctx（前缀钉死，只滑音频行）──▶ codes ──▶ decoder 128 帧窗口 ──▶ PCM
```

- `patches/0009`：ctx 满时丢掉最老的**音频行**（prompt 前缀永不淘汰），`seq_rm` + `seq_add` 平移位置；
  切点在目标 ±16 帧内选**能量最低帧**。
- `patches/0010`：`TOKENS\t<text>` op，用常驻词表精确分词（中文 ≈0.74 token/字、英文 ≈0.23），
  前端据此决定走整段还是长上下文，并给出生成预算。
- **契约变化（已与用户确认）**：`input/text/prompt` 上限 **4000 → 2048 字**（8 GB 上 ctx 6000 行的硬约束）；
  超过上限**截断而不是报错**，响应头 `X-Input-Truncated` 给出丢弃字数，切点优先落在句末。

实测：284 字 ASR 0.996 / RTF 0.198；620 字 144.8 s / RTF 0.201 / 峰值 7731 MiB；
4000 字 HTTP 200 / 1597.5 s / RTF 0.234 / 峰值 7623 MiB，但 `generated_frames` 撞 20000 上限（未自然结束）。
细节与未验证项见 `results/REPORT-longctx.md`。

## 三·九、流式端点收敛（2026-10-18）

- **`/v1/audio/speech/stream` 的 body 改成裸音频流**（不再有 msgpack 帧）：
  `audio_format` → `audio/ogg` / `audio/mpeg` / `audio/wav` / `audio/pcm`。
  `chunk_seconds` **已删除**（再传会 422）；出音节奏由 daemon 的解码窗口 W 决定。
  ⚠️ 这是**破坏性变更**：`scripts/chunk-test.py` 的 msgpack 解析已不适用。
- **流式预算改为 token 估算**（原来固定 `max_new_tokens=2048`，长文本会在 164 s 处截断）。
- **`patches/0012-fast-abort`**：生成循环里每帧 `poll()` 客户端 socket，
  对端一断立刻置 abort。实测：`curl -m 2` 中断后 **+1.0 s GPU 归零**，日志 `已跑 133 步`。

## 三·十、并发与资源（2026-10-18）

- daemon 是**单 slot**。前端加了一层显式队列：`MOSS_QUEUE_MAX=8`（含正在跑的那个）、
  `MOSS_QUEUE_WAIT=60s`；超过就返回 **503**（不再把连接堆到 unix socket 的 backlog 里）。
  实测 6 并发：3 个先后 200（11.7 / 23.5 / 34.9 s），3 个 503。
- 资源生命周期审计：daemon 连接（`AsyncDaemonStream.close` 幂等，`open` 失败也关）、
  ffmpeg 子进程（`terminate` + `wait` + `kill` 兜底）、`_daemon_request` 的 socket（`finally`）、
  临时目录（`TemporaryDirectory`）——都在 `finally` 里收。
- `_daemon_tokens` 在 daemon 忙时**不排队**（直接走字数兜底），避免给请求额外加延迟。

## 三·十一、单 slot 泄漏（2026-10-19 修复）

**症状**：brain server 报“TTS 卡死”。实测为 `POST /v1/audio/speech/stream` 连续 **503**，
但 `/health` 一直 200、GPU 空闲。

**根因**：流式请求被客户端中途断开时，`_encoded_stream_response` 的 `finally` 里先做
`await asyncio.wait_for(proc.wait(), 5)`；此时任务处于取消状态，`finally` 里的**第一个 await
会立刻抛 `CancelledError`**，而 `CancelledError` 继承 `BaseException`（不是 `Exception`），
`contextlib.suppress(Exception)` 接不住 → 后面的 `ds.close()` 被跳过 → **daemon slot 永久泄漏**。
之后所有需要 slot 的请求排队 60 s 后 503，看起来就是“服务卡死”。

**修复**：
1. `finally` 里**先 `ds.close()`（释放 slot），再做任何 await**；清理用的 `suppress` 改成 `BaseException`。
2. whole 模式也纳入同一个 slot 队列（之前它绕过 slot，会跟流式请求抢 daemon）。

**验证**：流式中途断开 → `inflight` 1 s 内归零；紧接着的流式请求 200 / 首字节 0.68 s；
非流式长请求中断同样归零；契约 25/25。

## 三·十二、opus 采样率标错（2026-10-19 修复）

**症状**：`audio_format=opus` 听起来是男声、语速偏慢。

**根因**：响应头 `X-Sample-Rate` 一律写 `24000`（PCM 的采样率），但 **Opus 内部固定 48 kHz**。
客户端若按头里的 24k 初始化解码器去解 48k 数据，音高降一个八度、语速减半 —— 听感就是“男声”。
（mp3 实际输出 24 kHz，wav/pcm 也是 24 kHz，只有 opus 不是。）

**修复**：`X-Sample-Rate` 按格式给真值：opus → `48000`，mp3/wav/pcm → `24000`。
Ogg/Opus 的 `OpusHead` 一直是 48 kHz，所以规范客户端不受影响，只有“只看头”的客户端会被骗。

## 四、风险与遗留（改完后的状态）

已消解（有实测证据）：
1. ~~增量解码边界伪影~~ → 窗口两侧各留 8 帧上下文；块边界 `|Δ|` 与全局分布一致，ASR 覆盖率 1.000。
2. ~~decoder 能否重复调用~~ → 可以：常驻 decoder ctx + 每窗口 `llama_memory_clear(mem, true)`，
   实测多窗口连续解码正常（流式请求 56 s / 118 块无异常）。
3. ~~跨请求状态污染~~ → 每请求新建 context/KV；whole 模式经 HTTP 取回的字节与早期版本 **md5 相同**。

仍然存在（必须说清）：
4. **并发**：单 slot 串行（前端持全局锁）。整卡被常驻对占满，**没有** continuous batching 的空间。
5. **非流式（whole）长度上限 448 帧 ≈ 36 s**：decoder 非因果、compute buffer 随长度线性涨。
   流式无上限；whole 模式超限会被 daemon 明确拒绝（ERR），不会崩。
   _（backbone 侧的 prefill OOM 已由 patch 0006 修掉；这里剩的是 **decoder** 整段解码的线性开销。）_
6. **短句 RTF 0.33–0.37，达不到 0.25**：0.35–0.45 s 的 prompt prefill 是硬开销（参考音频越长越贵）。
7. **人耳还没听**：目前只有 ASR round-trip + 边界统计。听感必须人来判（wav 在 `results/`）。
8. **流式模式改变了音频**（窗口化解码）：想跟早期版本逐字节一致就得 `MOSS_SYNTH_MODE=whole`（有长度上限）。

## 五、目录

```
./
├── README.md            # 本文件（设计 + 里程碑 + 风险）
├── upstream/llama.cpp/  # 从 moss-tts 复制的 fork 快照（唯一改动入口是 patches/）
├── patches/             # 0001 溢出修复 / 0002 reference-codes（继承）+ 本栈新增
├── build/{src,obj,bin}/ # 增量编译工作区（obj 从 moss-tts 复制，容器内 /src /build 路径一致 → 可增量）
├── scripts/             # gen-patch-000{3,4}.py 生成补丁 / build*.sh 编译打包 /
│                        # daemon-up|down.sh 起停常驻容器 / bench-m1.sh 压测 /
│                        # ab.py 与早期版本 A/B / stream-test.py 流式客户端 /
│                        # quality-check.py ASR+边界质量 / contract-test.py 契约自测
├── docker/              # Dockerfile.*（继承）
├── voices/ static/      # 音色参考音频 / demo 页
├── results/ logs/       # 实测数据与日志
└── ccache/              # 编译缓存（本栈独立，不复用 moss-tts 的）
```

## 六、测量口径（与 moss-tts 对齐才可比）

- **RTF** = 请求墙钟 / 音频时长；**必须同时报"含加载"与"纯推理"**（本栈的目标就是把"含加载"打到接近"纯推理"）
- **首块延迟** = 从请求发出到**第一段音频字节**到达（流式才有意义）
- **峰值显存** + **空闲显存**（常驻形态的空闲 ≠ 0，这是设计取舍，必须写清）
- 对照组：对照实现（6677）早期版本形态的同一批文本

---

## 七、侦察结果（M1/M2 的落点，已定位到行）

全部在 `upstream/llama.cpp/tools/tts/run-moss-tts-delay.cpp`，且都是 **`static`** →
**复用方式**：新建 `moss-tts-daemon.cpp`，用
```cpp
#define main moss_tts_cli_main
#include "run-moss-tts-delay.cpp"
#undef main
```
把整份 CLI 作为 TU include 进来（**不改原文件**），再写自己的 `main()` + 请求循环。

| 用途 | 符号 | 行 |
|---|---|---|
| 每请求状态（`text_history`/`audio_history`/frame helpers）| `struct moss_delay_state` | 112–151 |
| 建/复位状态 | `moss_init_delay_state(...)` | 558 |
| 采样 | `moss_sample_token(...)` | 761 |
| **AR 单步（流式切点）** | `moss_delay_step(...) -> std::vector<llama_token>` | 839 |
| 音频段提取 / 拼接 | `moss_extract_audio_segments` / `moss_concat_audio_segments` | 1063 / 1106 |
| codes → batch | `moss_batch_from_audio_codes(...)` | 1190 |
| **decoder 常驻加载** | `moss_load_audio_runtime(path, arch)` → `moss_audio_runtime` | 538 |
| codes → 波形（**需抽变体**：接受已加载 runtime、返回 float 样本，而不是自己加载+写 wav）| `moss_decode_audio_llama(...)` | 1290 |
| 参考音频 → codes（**daemon 里不走这条**：显存红线）| `moss_encode_audio_llama(...)` | 1214 |
| 整段编排（照抄结构）| `moss_generate_from_text` / `moss_generate_from_ref` | 247 / 1761 |

### M1 的具体施工单（下一步）
1. 新增源文件 `tools/tts/moss-tts-daemon.cpp`（如上 include 技巧）+ CMake target `llama-moss-tts-serve`
2. 启动时：`moss_load_audio_runtime(decoder)` + 加载 backbone（从 `moss_generate_from_ref` 里抄加载段）
3. 请求循环（先 stdin/stdout 长度前缀协议，前端再包 HTTP）：读 `{text, codes_path, sampling参数}` →
   `moss_init_delay_state` → 循环 `moss_delay_step` → `extract/concat segments` →
   用**常驻** decoder runtime 解码 → 返回 PCM + 头部统计
4. 对照测量：同一批文本，比早期版本 moss-tts 的「含加载 RTF」和「纯推理 RTF」

### M2 的具体施工单（增量流式）
1. 抽 `moss_decode_codes_runtime(runtime, codes, frames, n_vq) -> vector<float>`（复用已加载 decoder）
2. AR 循环里每 **N 帧**（先用 12 帧 ≈ 1 s 音频）交出一批 codes → 立即解码 → 立即吐 PCM
3. 窗口边界用 **overlap**（多解 M 帧、只保留中间），M/N 需实测调（听感 + 无爆音）
4. 协议：裸 PCM 透传（先）+ 可选 `[u32 BE][msgpack{audio,text}]` 分帧

### 还没做（诚实）
- **C++ 一行都还没改**；本栈目前只有骨架 + 本设计文档
- 没有任何本栈的实测数字（RTF/首块/显存都是"预期"，不是结论）


## License

MIT — 见 [LICENSE](LICENSE)。

依赖说明：ffmpeg 以**独立进程**方式调用（`exec`，不链接），构建时由 `scripts/fetch-ffmpeg.sh`
从上游下载，不随本仓库源码分发；上游 llama.cpp / MOSS-TTS 各自的许可证见其仓库。
