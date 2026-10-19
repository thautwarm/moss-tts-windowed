# moss-tts-windowed

English | [中文](README.zh.md)

**Streaming MOSS-TTS 8B on a single ≤8 GB GPU.** The model stays resident and audio is
emitted as it is generated — no weight reload per request, so an 8B TTS fits on one
consumer card.

| | |
|---|---|
| Model | MOSS-TTS v1.5 (8B Delay) |
| VRAM | Single card **≤8 GB** (measured: 6.5 GB steady, 7.8 GB peak) |
| Streaming | Ogg/Opus, MP3, WAV, PCM — raw audio stream |
| Latency | First byte 0.6 s (short) / 0.8 s (long) |
| Throughput | Long-text RTF **0.19–0.23** |

- Resident backbone (4.6 GB) + audio decoder (1.7 GB): **zero weight reload per request**
- Long text runs as a **single context** (the whole text goes in once) with a sliding
  window over audio rows — no sentence-by-sentence stitching artifacts
- Streaming responses are **raw audio streams**, not a chunked protocol; clients decode
  as bytes arrive
- Input limit 2048 characters (excess is **truncated** and flagged in a response header);
  one GPU, one synthesis slot, explicit FIFO queue

---

## Quick start

You need: one NVIDIA GPU with ≥8 GB VRAM, Docker, and `nvidia-container-toolkit`.
Nothing else on the host — no ffmpeg, CUDA, PyTorch, or Python dependencies.

```bash
git clone https://github.com/thautwarm/moss-tts-windowed.git && cd moss-tts-windowed

./scripts/fetch-models.sh       # 3 GGUF files from HuggingFace (~6.5 GB, sha256-checked)
./scripts/fetch-ffmpeg.sh       # static ffmpeg (77 MB, sha256-checked)
./scripts/fetch-llama-cpp.sh    # upstream llama.cpp fork source (patches apply onto it)
./scripts/build-base.sh         # runtime base image — ~10 min the first time
./scripts/build-llama.sh        # apply patches + compile the daemon (seconds with ccache)
./scripts/build-image.sh        # assemble the app image (seconds)

cp .env.example .env            # point MOSS_MODELS_HOST at your model dir (default ./models)
docker compose up -d

curl -s http://127.0.0.1:6677/health | jq
```

On a restricted network, fetch models via a mirror:
`HF_ENDPOINT=https://hf-mirror.com ./scripts/fetch-models.sh`

> ⚠️ **The backbone GGUF must be produced by this repo's `convert.sh`**
> (`general.architecture = moss-tts-delay`). The officially released
> `MOSS_TTS_Q4_K_M.gguf` is **`qwen3` architecture** with embeddings/heads split into
> `.npy` files. It **loads** under this fork but generates all-pad codes — you get a
> fraction of a second of silence and no error. `fetch-models.sh` pulls the correct
> first-class GGUFs from <https://huggingface.co/thaut/moss-tts-windowed-gguf>; if you
> convert your own, see [Build](#build).

---

## Requirements

| Item | Requirement |
|---|---|
| GPU | One NVIDIA card, ≥8 GB (verified on an 8 GB consumer card) |
| Driver / runtime | NVIDIA driver + `nvidia-container-toolkit` |
| Disk | ~6.5 GB models + ~10 GB images |
| Host | Docker only (plus `curl`/`sha256sum` for the fetch scripts); no ffmpeg / CUDA / PyTorch / Python |

### Model files

Three files in a host directory (default `./models`), mounted read-only at `/models`:

```
moss_tts_v15_firstclass_q4_k_m_aq8.gguf   # backbone (~4.6 GB; aq8 = audio tables at Q8_0)
moss_tts_audio_decoder_f16.gguf           # audio decoder (~1.7 GB)
moss_tts_audio_encoder_f16.gguf           # only used when registering a new voice
```

`./scripts/fetch-models.sh` downloads all three (sha256-verified, existing files are
skipped). To convert from the HuggingFace weights yourself instead (CPU-only conversion,
no GPU needed), see [Build](#build).

---

## Build

```bash
./scripts/fetch-ffmpeg.sh        # static ffmpeg (77 MB, with libopus/libmp3lame), sha256-checked
./scripts/fetch-llama-cpp.sh     # OpenMOSS/llama.cpp fork, pinned by commit SHA + tarball sha256
./scripts/build-base.sh          # two base images (toolchain + CUDA runtime) — ~10 min first time
./scripts/build-llama.sh         # apply patches/, compile llama-moss-tts (incremental; seconds with ccache)
./scripts/build-image.sh         # assemble the app image (seconds)
```

Incremental rebuilds: after changing `server.py` or a patch, re-run only the last two
steps. Keep `build/obj/` and `ccache/` — deleting either forces a full recompile (~145 s).

**GPU architecture**: `build-llama.sh` auto-detects via `nvidia-smi --query-gpu=compute_cap`.
If detection fails it falls back to `120`; set it explicitly when needed:

```bash
CUDA_ARCH=86 ./scripts/build-llama.sh    # A6000 / RTX 30 series
CUDA_ARCH=89 ./scripts/build-llama.sh    # RTX 40 series
```

⚠️ A wrong architecture typically shows up as: the service starts, but the first
synthesis fails with `CUDA error: no kernel image is available for execution on the device`.

**Converting your own GGUF** (alternative to `fetch-models.sh`): download the HF weights
to `work/hf/MOSS-TTS-v1.5` and `work/hf/MOSS-Audio-Tokenizer`, then run
`./scripts/convert.sh` (it builds its own CPU conversion image, so finish the build above
first). Output lands in `work/gguf/`; point `MOSS_MODELS_HOST` at it. If you already have
a backbone GGUF, `./scripts/convert.sh --tokenizer-only` converts just the encoder/decoder.

---

## Run

```bash
cp .env.example .env
# edit MOSS_MODELS_HOST to point at the model directory
docker compose up -d
```

The host port is **6677** by default (`TTSPORT` in `.env`; the container listens on 8080).
One container runs **two processes**: the uvicorn/FastAPI frontend and the resident C++
daemon, connected over the unix socket `/run/moss/serve.sock` (mounted to `./run` on the
host for debugging). If the daemon crashes it is restarted automatically; the frontend
reports 503/500 truthfully in the meantime.

```bash
curl -s http://127.0.0.1:6677/health | jq
```

`/health` reports: whether the daemon is online (`daemon.online`), missing model files,
registered voice count, queue state (`queue.inflight` / `waiting` / `rejected`), and the
effective decode parameters. The compose healthcheck allows a 90 s `start_period` —
resident model loading takes a while.

---

## Usage

### Registering a voice

Zero-shot cloning: give it a reference audio in any format and `add-voice.sh` converts it
to 24 kHz mono using the **static ffmpeg inside the app image** — the host needs nothing
but Docker (the image must be built first).

```bash
./scripts/add-voice.sh ./ref/alice.mp3 alice
```

Optionally precompute reference codes so the first synthesis is fast
(otherwise the server computes them on CPU once, a few seconds, on first use):

```bash
./scripts/precompute-codes.sh            # all voices; or pass specific ids
```

You can also register via the API — the WAV must already be 24 kHz mono, and codes are
precomputed on CPU at registration time (~2 s, no VRAM involved):

```bash
curl -X POST http://127.0.0.1:6677/api/v1/voices \
  -F "id=alice" -F "file=@ref/alice.wav"
```

Convention (see `voices/README.md`): `voices/<id>.wav` +
`voices/<id>.<wav sha8>.codes.bin`; the directory is **not committed** to git.
`<id>` matches `[A-Za-z0-9_.-]{1,64}`; `default` is reserved (no reference audio).

```bash
curl -s http://127.0.0.1:6677/api/v1/voices | jq
curl -X DELETE http://127.0.0.1:6677/api/v1/voices/alice
```

### Streaming synthesis (recommended)

`POST /v1/audio/speech/stream` takes these fields; **extra fields are silently ignored**
(not an error), so request bodies written for other TTS services can be reused:

```jsonc
{
  "prompt": "text to synthesize (≤2048 chars)",
  "voice": "alice",            // default "default" (no reference voice)
  "seed": 42,                  // -1 = random
  "audio_format": "opus",      // wav | opus | mp3 | pcm
  "speed": 1.0                 // 0.5–2.0, default 1.0 (atempo: pitch-preserving)
}
```

The response body is a **raw audio stream**; `Content-Type` follows the format:

| `audio_format` | Content-Type | Notes |
|---|---|---|
| `opus` | `audio/ogg` | 64 kbps default (`MOSS_OPUS_BITRATE`), **48 kHz** (Opus is always 48 kHz) |
| `mp3` | `audio/mpeg` | 128 kbps, 24 kHz |
| `wav` | `audio/wav` | 24 kHz / mono / 16-bit; length fields are `0xFFFFFFFF` (unknown length) |
| `pcm` | `audio/pcm` | 24 kHz / mono / s16le raw stream |

`speed` is implemented with ffmpeg's `atempo` (pitch-preserving), range 0.5–2.0; `1.0`
bypasses the filter entirely.

```bash
# save to a file
curl -sN -X POST http://127.0.0.1:6677/v1/audio/speech/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello, world.","voice":"alice","audio_format":"opus"}' \
  -o hello.ogg

# play while downloading
curl -sN -X POST http://127.0.0.1:6677/v1/audio/speech/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The weather is nice today.","voice":"alice","audio_format":"opus"}' | ffplay -nodisp -
```

Response headers:

```
X-Voice: alice
X-Format: opus
X-Streaming: 1
X-Sample-Rate: 48000              # opus; wav/pcm/mp3 report 24000
X-Input-Truncated: 952            # only present when input exceeded 2048 chars; value = dropped chars
```

(plus `Cache-Control: no-store` and `X-Accel-Buffering: no` on all streaming responses).
⚠️ Decode Opus as **48000 Hz** (or honor `OpusHead`) — decoding it as 24 kHz drops the
pitch an octave and sounds like a different speaker.

### OpenAI-compatible endpoint

`POST /v1/audio/speech` keeps the OpenAI request shape (`input`, `voice`,
`response_format`, `stream`, …) but **only supports `stream=true`**:

```bash
curl -sN -X POST http://127.0.0.1:6677/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello.","voice":"alice","response_format":"opus","stream":true}'
```

> Non-streaming requests (no `stream=true`) return **400** by default: the whole-audio
> path has been retired from the public contract, and this deployment only guarantees the
> streaming path. Set `MOSS_DISABLE_NONSTREAM=0` to re-enable it.
> Per-request reference audio (`ref_audio`/`ref_text`) and duration control
> (`token_count`/`duration_tokens`) are rejected — register a voice first.

Other endpoints: `GET /v1/models`, `POST /api/v1/synthesize` (internal/debug; supports
non-streaming), `POST /api/v1/synthesize/upload` (one-shot cloning with an uploaded
reference, not persisted).

### Language and sampling parameters

These are **not request fields**; the server fixes them via `.env` (the official
recommended values for Delay-8B):

```
MOSS_LANGUAGE=zh
MOSS_AUDIO_TEMPERATURE=1.7
MOSS_AUDIO_TOP_P=0.8
MOSS_AUDIO_TOP_K=25
MOSS_AUDIO_REPETITION_PENALTY=1.0
```

---

## Configuration (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `TTSPORT` | `6677` | Host port |
| `MOSS_UID` / `MOSS_GID` | `1000` | Container runs as this uid/gid (`id -u` / `id -g`), so `voices/` `run/` `logs/` ownership stays correct |
| `MOSS_MODELS_HOST` | `./models` | Model directory (host path), mounted at `/models` |
| `MOSS_MODEL_FILE` | `moss_tts_v15_firstclass_q4_k_m_aq8.gguf` | Backbone |
| `MOSS_DECODER_FILE` | `moss_tts_audio_decoder_f16.gguf` | Decoder |
| `MOSS_LANGUAGE` | `zh` | Language tag for synthesis |
| `MOSS_AUDIO_TEMPERATURE` / `TOP_P` / `TOP_K` / `REPETITION_PENALTY` | `1.7` / `0.8` / `25` / `1.0` | Sampling parameters (server-side, official recommended) |
| `MOSS_STREAM_WINDOW` | `32` | Decode-window advance, in frames (1 frame = 0.08 s) |
| `MOSS_STREAM_OVERLAP` | `40` | Left-context frames (window = WINDOW + 2×OVERLAP) |
| `MOSS_STREAM_LOOKAHEAD` | `0` | Right lookahead frames (0 = don't wait) |
| `MOSS_CTX_ROWS` | `6000` | Backbone KV row cap (sliding-window trigger) |
| `MOSS_CTX_KEEP` | `2048` | Most-recent audio rows kept across a slide |
| `MOSS_PREFILL_UBATCH` | `384` | Max tokens per prefill batch (pp buffer ∝ ubatch; a whole-prompt prefill OOMs on 8 GB) |
| `MOSS_PREFILL_BUDGET_MIB` | `1000` | VRAM budget (MiB) for the prefill compute buffer |
| `MOSS_MAX_INPUT_CHARS` | `2048` | Input cap; excess is truncated |
| `MOSS_QUEUE_MAX` | `8` | Queue cap (including the running request) |
| `MOSS_QUEUE_WAIT` | `60` | Queue timeout (seconds); past it, 503 |
| `MOSS_LOG_TEXT` | `0` | 1 = log request text |
| `MOSS_DISABLE_NONSTREAM` | `1` | 1 = non-streaming `/v1/audio/speech` returns 400 |

Apply changes with `docker compose up -d` (`.env` is injected via `env_file`).

The defaults above mirror the shipped `.env.example`. A few tuning knobs have lower code
fallbacks when omitted entirely (`MOSS_STREAM_WINDOW`/`MOSS_STREAM_OVERLAP` → 8/8,
`MOSS_PREFILL_BUDGET_MIB` → 1400), so copy the template rather than picking variables.

---

## Concurrency & limits

A single GPU means a single daemon slot (one synthesis at a time); the frontend runs an
explicit FIFO queue:

| Situation | Behavior |
|---|---|
| Concurrency ≤ `MOSS_QUEUE_MAX` | Queued, served FIFO |
| Queue full | Immediate **503** + `Retry-After: 2` |
| Waiting longer than `MOSS_QUEUE_WAIT` | **503** + `Retry-After: 5` |
| Slow-reading client | Doesn't hold the slot: audio spools to an in-memory queue and the compute slot is released as soon as generation finishes |
| Client disconnects | Generation stops immediately (per-frame check) and the slot is freed |

Live state is at `/health.queue`: `inflight` / `waiting` / `rejected{queue_full,queue_timeout}`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `503 daemon busy` | Queue full or wait timed out; check `/health.queue`, and honor `Retry-After` on the client |
| `health.daemon.online=false` | Daemon didn't come up: `docker logs` for model paths and VRAM |
| `400 unknown voice` | Voice id doesn't exist; `GET /api/v1/voices` lists what is registered |
| Wrong pitch / sounds male | Decode at `X-Sample-Rate`; Opus must be decoded as **48000** (or read `OpusHead`) |
| Text truncated | `X-Input-Truncated` is set; split the request or raise `MOSS_MAX_INPUT_CHARS` (VRAM-bound) |
| Slow first byte | First byte = prefill + filling `MOSS_STREAM_WINDOW` frames; a smaller window is faster but adds seams |
| `CUDA error: no kernel image...` on first synthesis | Wrong `CUDA_ARCH` at build time — rebuild with the right value |
| Out of VRAM | Lower `MOSS_CTX_ROWS` (e.g. 5000) or `MOSS_STREAM_OVERLAP` |

---

## Design notes (why it looks like this)

```
HTTP :6677 ──▶ FastAPI (contract / voices / ffmpeg pipes / queue)
                  │ unix socket
                  ▼
             C++ daemon (resident backbone + decoder, single slot)
```

- **Why resident**: the upstream shape was "spawn a CLI per request", which re-deserializes
  ~8.4 GB of weights every time — a fixed ~2.1 s overhead. With residency that drops to
  zero, and long-text RTF went from 0.258 to 0.216.
- **Why a patched fork**: the upstream fork has no server mode and no streaming hooks.
  `patches/0001–0015` add resident models, incremental streaming, chunked prefill, the KV
  sliding window, disconnect detection, and more.
- **Why long text uses a single context**: the backbone's `n_ctx_train = 40960`, so text
  never needs sentence-splitting; the binding constraint is VRAM (KV ≈ 0.14 MiB/row). The
  full text goes into the prefix once, and only the **audio rows** slide.
- **Why the decode window**: the audio decoder is **non-causal** (it encodes a whole
  segment at once) and its `n_ctx_train` is only 125 frames. The window (default
  32+2×40 = 112 frames) approximates it — smaller saves VRAM, larger is closer to
  whole-segment decoding.
- **Why streaming ≠ whole-segment**: windowed decoding is not byte-identical to one-shot
  whole-segment decoding. For quality-critical use, enlarge the window
  (`MOSS_STREAM_WINDOW`/`OVERLAP`) to trade latency for quality, or use
  `MOSS_SYNTH_MODE=whole` (byte-identical to the reference implementation, capped at
  448 frames ≈ 36 s).

Full design log and measurements: [docs/DESIGN.md](docs/DESIGN.md); daemon internals:
[docs/daemon-api-notes.md](docs/daemon-api-notes.md).

---

## Directory layout

```
server.py              HTTP contract + queue + streaming + ffmpeg pipes
docker/                Dockerfile / entrypoint / toolchain
patches/               0001–0015: all changes on top of upstream llama.cpp
scripts/               build, contract tests, benchmarks, voice registration, ASR checks
docs/                  DESIGN.md (design + measurement log) + daemon API notes
voices/                voice directory (not committed; see voices/README.md)
results/               acceptance reports and measurements
```

## Development & self-tests

```bash
python3 scripts/contract-test.py      # contract self-check against a running service
./scripts/add-voice.sh <audio> <id>   # register a voice
./scripts/precompute-codes.sh         # precompute reference codes for registered voices
```

## License

MIT — see [LICENSE](LICENSE).

Dependency note: ffmpeg is invoked as a **separate process** (`exec`, not linked) and is
downloaded from upstream at build time by `scripts/fetch-ffmpeg.sh` — it is not
distributed with this repository's source. Upstream llama.cpp / MOSS-TTS carry their own
licenses; see their repositories.
