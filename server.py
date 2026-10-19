"""MOSS-TTS v1.5 HTTP API —— 常驻 daemon 版（`moss-tts-stream`）

与上游 moss-tts 的 `server.py` **逐字段一致**的契约：那个 563 行的服务的
端点/字段/响应头/错误语义都保留，唯一改变的是**谁在生成**：

    之前：每请求 subprocess.run(llama-moss-tts)  → 每请求重新反序列化 ~8.4 GB 权重（+2.1 s）
    现在：unix socket 打到常驻 daemon（backbone 4.6 GB + decoder 1.7 GB 常驻）
          → 权重只加载一次；流式请求还能**边生成边吐 PCM**

端到端契约（与前服务一致）：
  GET    /health
  GET    /api/v1/voices
  POST   /api/v1/voices         (multipart) 注册克隆参考音频（顺带在 CPU 上预算 codes）
  DELETE /api/v1/voices/{id}
  POST   /api/v1/synthesize     (JSON)      -> audio/wav|mp3|opus|pcm
  POST   /api/v1/synthesize/upload          -> 一次性克隆，不落库
  POST   /v1/audio/speech       (OpenAI 兼容；**本栈额外支持 stream=true**)
  GET    /v1/models

与前服务的差别（都是"放宽"，不放宽契约里的拒绝集）：
  * `stream=true` 现在能用（前服务直接 400）：pcm/wav 真流式；opus 走 ffmpeg 管道
    （ogg/opus 连续流，首包 <100 ms）、mp3 走 lame 管道，都是边生成边发
  * 单请求长度上限：流式无上限；非流式受 daemon 的 --serve-max-new-tokens 约束（默认 448 帧 ≈ 36 s）
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

# ── config ────────────────────────────────────────────────────────────────

MODELS = Path(os.environ.get("MOSS_MODELS_DIR", "/models"))
MODEL_FILE = os.environ.get("MOSS_MODEL_FILE", "moss_tts_v15_firstclass_q4_k_m_aq8.gguf")
ENCODER_FILE = os.environ.get("MOSS_ENCODER_FILE", "moss_tts_audio_encoder_f16.gguf")
DECODER_FILE = os.environ.get("MOSS_DECODER_FILE", "moss_tts_audio_decoder_f16.gguf")
VOICES = Path(os.environ.get("MOSS_VOICES_DIR", "/voices"))
BIN = os.environ.get("MOSS_BIN", "llama-moss-tts")
NGL = os.environ.get("MOSS_NGL", "-1")
REQ_TIMEOUT = int(os.environ.get("MOSS_TIMEOUT", "600"))
MAX_NEW_TOKENS_CAP = int(os.environ.get("MOSS_MAX_NEW_TOKENS_CAP", "2048"))
TRANSCODE_TIMEOUT = int(os.environ.get("MOSS_TRANSCODE_TIMEOUT", "120"))

# 流式端点的请求只有 4 个字段（prompt/voice/seed/audio_format）；
# 语言 / 采样参数 / token 预算 / 分帧方式都是服务端默认 —— 要改改 env，不动契约。
STREAM_LANGUAGE = os.environ.get("MOSS_LANGUAGE", "zh")
STREAM_MNT = 2048   # 固定：不暴露给请求，也不放 env
STREAM_TEMP = float(os.environ.get("MOSS_AUDIO_TEMPERATURE", "1.7"))
STREAM_TOP_P = float(os.environ.get("MOSS_AUDIO_TOP_P", "0.8"))
STREAM_TOP_K = int(os.environ.get("MOSS_AUDIO_TOP_K", "25"))
STREAM_REP = float(os.environ.get("MOSS_AUDIO_REPETITION_PENALTY", "1.0"))

# 常驻 daemon 的 unix socket
DAEMON_SOCK = os.environ.get("MOSS_DAEMON_SOCK", "/run/moss/serve.sock")
# stream = 走 daemon 的流式 op（无长度上限、首块 <1 s）；whole = 走整段解码（与老栈逐字节一致）
SYNTH_MODE = os.environ.get("MOSS_SYNTH_MODE", "stream").strip().lower()
# 流式窗口/上下文（帧；1 帧 ≈ 0.08 s 音频）。daemon 启动时也可固定，这里按请求覆盖。
STREAM_WINDOW = int(os.environ.get("MOSS_STREAM_WINDOW", "8"))
STREAM_OVERLAP = int(os.environ.get("MOSS_STREAM_OVERLAP", "8"))
# 右侧前瞻帧数：0 = 有多少用多少（零等待）；N = 只等 N 帧再吐
STREAM_LOOKAHEAD = int(os.environ.get("MOSS_STREAM_LOOKAHEAD", "0"))
# 非流式请求的窗口上下文：不急着出音，用大上下文换质量（O=64 时与整段解码帧相关 p50 0.95）
# 注意 daemon 的 decoder ctx ≈ W + O + L + 16 帧，O 太大（≥96）会 OOM
SYNTH_OVERLAP = int(os.environ.get("MOSS_SYNTH_OVERLAP", "64"))
SYNTH_LOOKAHEAD = int(os.environ.get("MOSS_SYNTH_LOOKAHEAD", "0"))

# whole 模式（整段解码：质量最好，与参考实现逐字节一致）的 daemon 上限是 448 帧 ≈ 36 s。
# 长文本按句切分、逐句整段解码，段间补自然停顿 —— 每句都是完整 decode，不产生窗口失真。
WHOLE_MAX_FRAMES = int(os.environ.get("MOSS_WHOLE_MAX_FRAMES", "448"))
WHOLE_CHUNK_CHARS = int(os.environ.get("MOSS_WHOLE_CHUNK_CHARS", "110"))  # ≈28 s，留足余量
WHOLE_GAP_SEC = float(os.environ.get("MOSS_WHOLE_GAP_SEC", "0.18"))
# P1：长文本不再按句切，直接单 ctx 全文生成 + 大窗口解码。
# 帧数估算：实测 56 字 → 177 帧 ≈ 3.16 帧/字；取 3.2 留余量。
# 每 token 生成多少帧：实测 284 字(≈213 tok)→690 帧=3.2；620 字(≈465 tok)→1843 帧=3.96
FRAMES_PER_TOKEN = float(os.environ.get("MOSS_FRAMES_PER_TOKEN", "3.6"))
# 长文本解码窗口（帧）：64 ≈ 5.12 s，对齐 decoder 训练上下文 125 帧的一半
SYNTH_WINDOW = int(os.environ.get("MOSS_SYNTH_WINDOW", "64"))
# backbone ctx 预算（行）：可用显存约 1.4 GB / 0.1406 MiB ≈ 10k 行（由 daemon 滑窗维持）
CTX_ROWS_BUDGET = int(os.environ.get("MOSS_CTX_ROWS", os.environ.get("MOSS_CTX_ROWS_BUDGET", "6000")))
# 单次生成的帧数硬上限（防跑飞）：20000 帧 ≈ 26 min 音频
MAX_GEN_FRAMES = int(os.environ.get("MOSS_MAX_GEN_FRAMES", "20000"))
# 输入字数上限：超过就**截断**（不报错），并在响应头 X-Input-Truncated 写明丢了多少字
MAX_INPUT_CHARS = int(os.environ.get("MOSS_MAX_INPUT_CHARS", "2048"))
# 本部署只对外提供流式合成：非流式的 /v1/audio/speech 直接 400（/api/v1/* 管理端点不受影响）
DISABLE_NONSTREAM = os.environ.get("MOSS_DISABLE_NONSTREAM", "1") == "1"
# 调试开关：把请求文本也写进日志（默认关，避免日志里落用户说的话）
LOG_TEXT = os.environ.get("MOSS_LOG_TEXT", "0") == "1"


def _text_echo(text: str) -> str:
    if not LOG_TEXT:
        return ""
    t = text if len(text) <= 200 else text[:200] + "…"
    return f" text={t!r}"

# 上传参考音频的 codes 缓存（LRU），按内容哈希寻址
CODES_CACHE = Path(os.environ.get("MOSS_CODES_CACHE", "/tmp/moss-codes-cache"))
CODES_CACHE_MAX = int(os.environ.get("MOSS_CODES_CACHE_MAX", "64"))

VOICE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

FORMATS: dict[str, tuple[Optional[list[str]], str, str]] = {
    "wav":  (None, "audio/wav", "wav"),
    "pcm":  (None, "audio/pcm", "pcm"),
    "mp3":  (["lame", "--quiet", "-b", "128", "-", "-"],         "audio/mpeg", "mp3"),
    "opus": (["opusenc", "--quiet", "--bitrate", "64", "-", "-"], "audio/ogg",  "opus"),
}
OPENAI_ONLY_FORMATS = ("aac", "flac")

# 真流式编码（`stream=true` + opus/mp3）：边收 PCM 边出容器。
# opus 走 ffmpeg；`-probesize 32 -analyzeduration 0` 必须给 —— 否则 ffmpeg 会先"探测"
# 整个输入，管道下实测要攒 ~4.25 s 才吐第一个字节。`-page_duration 20000` = 20 ms 一页。
OPUS_BITRATE = os.environ.get("MOSS_OPUS_BITRATE", "64k")
MP3_BITRATE = os.environ.get("MOSS_MP3_BITRATE", "128k")

# ── state ────────────────────────────────────────────────────────────────

app = FastAPI(title="MOSS-TTS v1.5", version="0.3.0")
_codes_lock = threading.Lock()   # 只保护「缺 codes 时在 CPU 上补算」这一段
started_at = time.time()

# ── request model ────────────────────────────────────────────────────────


class SpeechRequest(BaseModel):
    """OpenAI Create speech 协议（POST /v1/audio/speech）。字段与前服务完全一致。"""

    model: str = Field(default="", description="OpenAI 协议要求；本部署忽略（可用于推断 voice）")
    input: str = Field(..., min_length=1, max_length=100000, description="要合成的文本（上限 2048 字，8 GB 卡上的显存约束）")
    voice: str = Field(default="default", description="已注册的音色 id，见 /api/v1/voices")
    response_format: str = Field(default="wav", description="wav / mp3 / opus / pcm，见 FORMATS")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速倍率（原生时长控制）")
    language: str = Field(default="zh", description="语言标签，如 zh / en / Chinese")
    max_new_tokens: int = Field(default=512, ge=16, le=MAX_NEW_TOKENS_CAP)
    seed: int = Field(default=-1)
    temperature: float = Field(default=1.7, ge=0.0, le=3.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=25, ge=0, le=200)
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    stream: bool = False
    instruct: Optional[str] = None
    instructions: Optional[str] = None
    token_count: Optional[int] = None
    duration_tokens: Optional[int] = None
    tokens: Optional[int] = None


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000, description="要合成的文本（上限 2048 字）")
    voice: str = Field(default="default", description="音色ID（= 参考音频），见 /api/v1/voices")
    language: str = Field(default="zh", description="语言标签，直接写进 prompt，如 zh / en / Chinese / French")
    max_new_tokens: int = Field(default=512, ge=16, le=MAX_NEW_TOKENS_CAP,
                                description="音频 token 预算；1 秒 ≈ 12.5 token")
    seed: int = Field(default=-1, description="随机种子，-1 = 随机")
    temperature: float = Field(default=1.7, ge=0.0, le=3.0, description="audio_temperature")
    top_p: float = Field(default=0.8, ge=0.0, le=1.0, description="audio_top_p")
    top_k: int = Field(default=25, ge=0, le=200, description="audio_top_k")
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
    response_format: str = Field(default="wav", description="wav / mp3 / opus / pcm，见 FORMATS")


class ChunkStreamRequest(BaseModel):
    """POST /v1/audio/speech/stream 的请求：**只有这 5 个字段**。

    语言、采样参数、token 预算、分帧方式都是服务端默认（env 可改，不动契约）。
    多给的字段**静默忽略**（extra="ignore"），方便客户端复用别的 TTS 的请求体。
    """

    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(..., min_length=1, max_length=100000, description="要合成的文本（上限 2048 字）")
    voice: str = Field(default="default", description="已注册音色 id；default = 无参考")
    seed: int = Field(default=-1, description="-1 = 随机")
    audio_format: str = Field(default="wav", description="wav / opus / mp3 / pcm")
    speed: float = Field(default=1.0, ge=0.5, le=2.0,
                         description="语速倍率（原生时长控制：1.0 = 模型自然语速）")


# ── helpers ──────────────────────────────────────────────────────────────


def _wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        ch = w.getnchannels()
    return {"sample_rate": rate, "channels": ch, "frames": frames,
            "duration": round(frames / rate, 3) if rate else 0.0}


def _pcm_wav(pcm: bytes, rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """把裸 PCM 包成 wav（与 CLI 的 save_wav16 同口径：16-bit LE）"""
    return struct.pack("<4sI4s4sIHHIIHH4sI",
                       b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, channels, rate,
                       rate * channels * bits // 8, channels * bits // 8, bits,
                       b"data", len(pcm)) + pcm


def _valid_ref(p: Path) -> Optional[str]:
    try:
        info = _wav_info(p)
    except Exception as e:
        return f"不是合法 WAV ({e})"
    if info["sample_rate"] != 24000 or info["channels"] != 1:
        return f"必须是 24 kHz mono，实际 {info['sample_rate']} Hz / {info['channels']}ch"
    return None


_SHA_CACHE: dict[str, tuple[int, int, str]] = {}


def _wav_sha8(p: Path) -> Optional[str]:
    try:
        st = p.stat()
    except OSError:
        return None
    ent = _SHA_CACHE.get(str(p))
    if ent and ent[0] == st.st_mtime_ns and ent[1] == st.st_size:
        return ent[2]
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    _SHA_CACHE[str(p)] = (st.st_mtime_ns, st.st_size, h)
    return h


def _codes_for(wav: Path) -> Optional[Path]:
    """按内容寻址取 codes：`voices/<id>.<参考音频 sha8>.codes.bin`（与前服务同约定）"""
    h = _wav_sha8(wav)
    if not h:
        return None
    c = VOICES / f"{wav.stem}.{h}.codes.bin"
    return c if c.exists() else None


def _daemon_request(line: str, timeout: int = REQ_TIMEOUT) -> str:
    """一行进、一行出的一次性 daemon 请求。短生命周期：任何出口都关连接。"""
    if not os.path.exists(DAEMON_SOCK):
        raise HTTPException(503, f"常驻 daemon 不在线（socket {DAEMON_SOCK} 不存在）")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(DAEMON_SOCK)
        s.sendall((line.rstrip("\n") + "\n").encode("utf-8"))
        buf = bytearray()
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                raise HTTPException(500, "常驻 daemon 关闭了连接")
            buf += chunk
        return bytes(buf).split(b"\n", 1)[0].decode("utf-8", "replace")
    except socket.timeout:
        raise HTTPException(504, f"常驻 daemon 超时（>{timeout}s）")
    except OSError as e:
        raise HTTPException(503, f"连接常驻 daemon 失败：{e}")
    finally:
        s.close()


def _precompute_codes(wav: Path, out: Path) -> bool:
    """算参考 codes：走 daemon 的 **CODES** op（daemon 内 `-ngl 0` 纯 CPU）。

    不起子进程、不重复加载权重；也不占显存 —— 常驻对已经吃掉 ~7 GB，
    encoder 再上卡（+1.7 GB）必 OOM。注意：daemon 是单 slot，这几秒会占用它。
    """
    line = "\t".join(["CODES", str(MODELS / MODEL_FILE), str(MODELS / ENCODER_FILE),
                      str(wav), str(out)])
    try:
        resp = _daemon_request(line)
    except HTTPException as e:
        print(f"[voices] CODES 请求失败：{e.detail}", flush=True)
        return False
    if not resp.startswith("OK"):
        print(f"[voices] CODES 失败：{resp}", flush=True)
    return resp.startswith("OK") and out.exists()


def _ensure_codes(wav: Path) -> Optional[Path]:
    """拿到（必要时在 CPU 上算出来）某个参考音频的 codes；失败返回 None。"""
    c = _codes_for(wav)
    if c is not None:
        return c
    out = VOICES / f"{wav.stem}.{_wav_sha8(wav)}.codes.bin"
    if not _precompute_codes(wav, out):
        return None
    return out if out.exists() else None


def _ref_or_400(voice_id: str) -> tuple[Optional[Path], Optional[Path]]:
    """解析音色 -> (参考 WAV, 预计算 codes)。与前服务同语义。"""
    if voice_id == "default":
        return None, None
    p = VOICES / f"{voice_id}.wav"
    if not p.exists():
        avail = [v["id"] for v in _list_voices()]
        raise HTTPException(400, f"未知音色: {voice_id}（现有: {', '.join(avail) or '无'}）")
    err = _valid_ref(p)
    if err:
        raise HTTPException(
            400,
            f"音色 {voice_id} 的 {p.name} 不合格：{err}。"
            f"用 `./scripts/add-voice.sh <源音频> {voice_id}` 生成 24 kHz mono 版本。",
        )
    codes = _codes_for(p)
    if codes is None:
        # 有 wav 但没有预计算 codes：绝不能什么都不给 daemon —— 那样它就没有参考，
        # 会**静默用默认音色**合成（音色错了还不报错）。这里在 CPU 上补算一次
        # （-ngl 0，一次性、内容寻址落盘，之后免算）。
        with _codes_lock:
            codes = _codes_for(p)  # 双检：并发请求时只算一次
            if codes is None:
                print(f"[voices] {p.name} 缺预计算 codes，在 CPU 上补算（约数秒）…", flush=True)
                codes = _ensure_codes(p)
                print(f"[voices] codes -> {codes}", flush=True)
    if codes is None:
        raise HTTPException(
            500,
            f"音色 {voice_id} 的参考 codes 既不存在、也算不出来（看容器日志："
            f"llama-moss-tts 的 CPU 预计算失败）",
        )
    return p, codes


# ── 上传参考音频的 codes LRU 缓存 ─────────────────────────────────────────

def _lru_touch(p: Path) -> None:
    try:
        os.utime(p, None)
    except OSError:
        pass


def _lru_evict() -> None:
    try:
        files = sorted(CODES_CACHE.glob("*.codes.bin"), key=lambda f: f.stat().st_mtime)
    except OSError:
        return
    for f in files[: max(0, len(files) - CODES_CACHE_MAX)]:
        f.unlink(missing_ok=True)


def _list_voices() -> list[dict]:
    out = [{"id": "default", "name": "默认音色（无参考）", "language": "auto",
            "description": "不提供参考音频，由模型自行决定音色", "valid": True}]
    if VOICES.is_dir():
        for p in sorted(VOICES.glob("*.wav")):
            err = _valid_ref(p)
            if err:
                out.append({"id": p.stem, "name": p.stem, "language": "auto",
                            "description": f"⚠️ 不可用：{err}", "valid": False})
                continue
            try:
                info = _wav_info(p)
            except Exception:
                continue
            out.append({"id": p.stem, "name": p.stem, "language": "auto",
                        "description": f"克隆参考音频 · {info['duration']}s", "valid": True,
                        "codes": _codes_for(p) is not None})
    return out


def _resolve_voice(voice: str, model: str = "") -> str:
    if voice and voice != "default":
        return voice
    ids = sorted((v["id"] for v in _list_voices() if v["id"] != "default" and v.get("valid")),
                 key=len, reverse=True)
    for vid in ids:
        if vid in (model or ""):
            return vid
    return "default"


def _wav_to_pcm(wav: bytes) -> bytes:
    i = wav.find(b"data")
    return wav[i + 8:] if i != -1 else wav


def _resolve_format(fmt: str) -> str:
    fmt = (fmt or "wav").lower().strip()
    if fmt in FORMATS:
        return fmt
    supported = ", ".join(FORMATS)
    if fmt in OPENAI_ONLY_FORMATS:
        raise HTTPException(
            400,
            f"response_format={fmt!r} 是 OpenAI 的合法值，但本部署未启用"
            f"（要额外装编码器，如 flac / ffmpeg；当前只用 lame + opusenc）。支持: {supported}",
        )
    raise HTTPException(400, f"不支持的 response_format: {fmt!r}。支持: {supported}")


def _check_model_files() -> list[str]:
    missing = []
    for f in (MODEL_FILE, ENCODER_FILE, DECODER_FILE):
        if not (MODELS / f).exists():
            missing.append(str(MODELS / f))
    return missing


# ── msgpack（够用就好：map / str / bin / int / float / bool / nil / array）───
# 只用标准编码，客户端拿任意 msgpack 解码器都能读；不引依赖 = 不动镜像构建。

def _daemon_args(text: str, ref_codes: Optional[Path], language: str, max_new_tokens: int,
                 seed: int, temperature: float, top_p: float, top_k: float,
                 repetition_penalty: float, overlap: Optional[int] = None,
                 lookahead: Optional[int] = None, window: Optional[int] = None,
                 duration_tokens: Optional[int] = None) -> list[str]:
    # patch 0015：文本走 base64 —— 内部协议是按行的，换行/制表符会截断请求
    args = ["--text-b64", base64.b64encode(text.encode("utf-8")).decode("ascii"),
            "--language", language,
            "--max-new-tokens", str(max_new_tokens),
            "--audio-temperature", str(temperature),
            "--audio-top-p", str(top_p),
            "--audio-top-k", str(top_k),
            "--audio-repetition-penalty", str(repetition_penalty),
            "--stream-window", str(STREAM_WINDOW if window is None else window),
            "--stream-overlap", str(STREAM_OVERLAP if overlap is None else overlap),
            "--stream-lookahead", str(STREAM_LOOKAHEAD if lookahead is None else lookahead)]
    if seed >= 0:
        args += ["--seed", str(seed)]
    if duration_tokens is not None:
        # patch 0014：写进 prompt 的 - Tokens 字段（原生时长控制）
        args += ["--duration-tokens", str(duration_tokens)]
    if ref_codes is not None:
        # 只给 codes：daemon 里 encoder 不常驻，现场编码会 OOM（见 README 二）
        args += ["--reference-codes", str(ref_codes)]
    return args


async def _audio_response(pcm: bytes, info: dict, fmt: str = "wav",
                          extra: Optional[dict] = None) -> Response:
    """按 response_format 转码并返回（入参是裸 PCM）。"""
    wav = _pcm_wav(pcm, info.get("sample_rate", 24000), info.get("channels", 1))
    body = await _async_transcode(wav, fmt)
    _, mime, ext = FORMATS[fmt]
    return Response(
        content=body,
        media_type=mime,
        headers={
            "Content-Disposition": f"inline; filename=speech.{ext}",
            "X-Elapsed": str(info.get("elapsed")),
            "X-Audio-Duration": str(info.get("duration")),
            "X-RTF": str(info.get("rtf")),
            "X-Format": fmt,
            "X-Mode": str(info.get("mode")),
            **(extra or {}),
        },
    )


async def _async_transcode(wav: bytes, fmt: str) -> bytes:
    """异步转码：wav/pcm 零拷贝；mp3/opus 起一次外部编码器，不阻塞事件循环。"""
    cmd = FORMATS[fmt][0]
    if cmd is None:
        return wav
    tool = cmd[0]
    if shutil.which(tool) is None:
        raise HTTPException(501, f"response_format={fmt!r} 需要 {tool}，但容器里没有")
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(wav), TRANSCODE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(504, f"转码超时（{fmt}）")
    finally:
        # 无论超时还是被取消，都收尸：kill + wait（不然会留僵尸进程）
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(BaseException):
                await proc.wait()
    if proc.returncode != 0 or not out:
        tail = (err or b"").decode("utf-8", "replace")[-500:]
        raise HTTPException(500, f"{tool} 转码失败（{fmt}, rc={proc.returncode}）: {tail}")
    return out


class DaemonError(Exception):
    """daemon 这条线出问题（协议错、连接断、ERR 回复）—— 与 HTTP 错误分开，方便流里区分。"""


# ── daemon 单 slot：并发请求自动排队 ─────────────────────────────────────
# daemon 是单线程、一次只服务一个请求。前端这里做一层显式队列：
#   · 最多同时排队 MOSS_QUEUE_MAX 个（含正在跑的那个）；
#   · 排队超过 MOSS_QUEUE_WAIT 秒直接 503，而不是把连接堆到 daemon 的 backlog 里。
N_DAEMONS = max(1, int(os.environ.get("MOSS_DAEMONS", "1")))


def _daemon_sock(i: int) -> str:
    """池里第 i 个 daemon 的 socket（0 号沿用 MOSS_DAEMON_SOCK，其余 serve-<i>.sock）。"""
    if i == 0:
        return DAEMON_SOCK
    return f"{os.path.dirname(DAEMON_SOCK)}/serve-{i}.sock"


_QUEUE_MAX = int(os.environ.get("MOSS_QUEUE_MAX", "8"))
_QUEUE_WAIT = float(os.environ.get("MOSS_QUEUE_WAIT", "60"))
_FREE: "asyncio.Queue[int]" = asyncio.Queue()
for _i in range(N_DAEMONS):
    _FREE.put_nowait(_i)
_QUEUE = 0
_INFLIGHT = 0          # 正在跑（持有 slot）的请求数
_STAT = {"queue_full": 0, "queue_timeout": 0, "ok": 0}


class _DaemonSlot:
    """持有/释放 daemon 单 slot。close() 幂等（open 失败路径会重复调）。"""

    def __init__(self) -> None:
        self.held = False
        self.index = -1

    async def acquire(self) -> None:
        global _QUEUE, _INFLIGHT
        if _QUEUE >= _QUEUE_MAX:
            _STAT["queue_full"] += 1
            print(f"[queue] 拒绝：队列已满 depth={_QUEUE}/{_QUEUE_MAX} inflight={_INFLIGHT}", flush=True)
            raise HTTPException(503, "daemon 忙：队列已满，请稍后重试",
                                headers={"Retry-After": "2"})
        _QUEUE += 1
        try:
            self.index = await asyncio.wait_for(_FREE.get(), _QUEUE_WAIT)
            self.held = True
            _INFLIGHT += 1
        except asyncio.TimeoutError:
            _STAT["queue_timeout"] += 1
            print(f"[queue] 拒绝：排队超时 >{_QUEUE_WAIT:g}s depth={_QUEUE}/{_QUEUE_MAX} inflight={_INFLIGHT}", flush=True)
            raise HTTPException(503, f"daemon 忙：排队超过 {_QUEUE_WAIT:g}s，请稍后重试",
                                headers={"Retry-After": "5"})
        finally:
            _QUEUE -= 1

    def release(self) -> None:
        global _INFLIGHT
        if self.held:
            self.held = False
            _INFLIGHT -= 1
            _STAT["ok"] += 1
            _FREE.put_nowait(self.index)


class AsyncDaemonStream:
    """daemon 流式 op 的连接（唯一的一种客户端：异步）。

    为什么必须 async：客户端断线时框架会取消这个 task，`finally` 里的 `close()`
    必然执行。阻塞 socket + 生成器的组合下，清理只会在 happy path 上跑 ——
    实测因此泄漏 fd，把单 slot 的 daemon 写阻塞死，整个服务卡住。
    """

    def __init__(self, reader, writer, slot: "_DaemonSlot"):
        self.r, self.w = reader, writer
        self.slot = slot
        self.rate, self.channels, self.bits = 24000, 1, 16
        self.bytes_per_sec = 48000

    @classmethod
    async def open(cls, args: list[str], timeout: int = REQ_TIMEOUT) -> "AsyncDaemonStream":
        """连接 + 发请求 + 读协议头：参数/daemon 级问题在这里就变成正常 HTTP 错误码。"""
        slot = _DaemonSlot()
        await slot.acquire()
        try:
            reader, writer = await asyncio.open_unix_connection(_daemon_sock(slot.index))
        except OSError as e:
            slot.release()
            raise HTTPException(503, f"连接常驻 daemon 失败：{e}")
        self = cls(reader, writer, slot)
        try:
            writer.write(("STREAM\t" + "\t".join(args) + "\n").encode("utf-8"))
            await writer.drain()
            head = (await asyncio.wait_for(reader.readline(), timeout)).decode("utf-8", "replace").strip()
        except asyncio.TimeoutError:
            self.close()
            raise HTTPException(504, f"常驻 daemon 超时（>{timeout}s）")
        except BaseException:
            self.close()
            raise
        if head.startswith("S "):
            _, r, c, b = head.split()
            self.rate, self.channels, self.bits = int(r), int(c), int(b)
            self.bytes_per_sec = self.rate * self.channels * self.bits // 8
            return self
        self.close()
        if head.startswith("ERR"):
            raise HTTPException(500, f"常驻 daemon 拒绝：{head[4:].strip()}")
        raise HTTPException(500, f"常驻 daemon 协议意外：{head!r}")

    async def read_pcm(self, timeout: int = REQ_TIMEOUT):
        """下一块 PCM；None = 正常结束（E）。"""
        try:
            head = (await asyncio.wait_for(self.r.readline(), timeout)).decode("utf-8", "replace").strip()
        except asyncio.TimeoutError:
            raise HTTPException(504, f"常驻 daemon 超时（>{timeout}s）")
        if head.startswith("A "):
            n = int(head.split()[1])
            try:
                return await self.r.readexactly(n)
            except asyncio.IncompleteReadError:
                raise DaemonError("daemon 数据被截断（可能已崩溃）")
        if head.startswith("E"):
            return None
        if head.startswith("ERR"):
            raise DaemonError(head[4:].strip() if head.startswith("ERR ") else head)
        if head == "":
            raise DaemonError("daemon 关闭了连接（可能已崩溃）")
        raise DaemonError(f"协议意外: {head!r}")

    def close(self) -> None:
        # 同步 close：finally 里不能 await（取消时 await 会再抛 CancelledError，吃掉清理）
        try:
            self.w.close()
        except Exception:  # noqa: BLE001
            pass
        self.slot.release()


def _info_from(ds: "AsyncDaemonStream", pcm: bytes, elapsed: float) -> dict:
    bps = ds.bytes_per_sec
    dur = len(pcm) / bps if bps else 0.0
    return {"sample_rate": ds.rate, "channels": ds.channels, "bits": ds.bits,
            "frames": len(pcm) // (ds.channels * ds.bits // 8),
            "duration": round(dur, 3), "elapsed": round(elapsed, 3),
            "rtf": round(elapsed / dur, 3) if dur else None,
            "realtime": bool(dur and elapsed / dur < 1.0), "mode": "stream"}


async def _async_daemon_synth(args: list[str],
                              on_pcm: Optional[Callable[[bytes], None]] = None):
    """收完整段流（非流式端点用）。on_pcm 非空则边收边回调。"""
    ds = await AsyncDaemonStream.open(args)
    t0 = time.time()
    chunks: list[bytes] = []
    try:
        while True:
            data = await ds.read_pcm()
            if data is None:
                break
            chunks.append(data)
            if on_pcm is not None:
                on_pcm(data)
    finally:
        ds.close()
    pcm = b"".join(chunks)
    return pcm, _info_from(ds, pcm, time.time() - t0)


def _daemon_whole_sync(args: list[str]) -> bytes:
    """整段模式（MOSS_SYNTH_MODE=whole，与参考实现逐字节一致的那条路）：同步、短生命周期。"""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.wav"
        resp = _daemon_request("\t".join(args + ["--wav-out", str(out)]))
        if not resp.startswith("OK") or not out.exists():
            raise HTTPException(500, f"常驻 daemon 失败：{resp}")
        return _wav_to_pcm(out.read_bytes())


def _punct_sentences(text: str) -> list[str]:
    """按标点切句：逐字符扫描（无正则）。保证不丢字符、标点留在句尾。"""
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch in "。！？!?；;…\n":
            if cur.strip():
                out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur)
    return out


def _split_for_whole(text: str, limit: int) -> list[str]:
    """把长文本切成 ≤limit 字的段：优先句末标点，单句超长再按逗号、最后硬切。"""
    text = text.strip()
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    cur = ""
    for s in _punct_sentences(text):
        s = s.strip()
        if not s:
            continue
        while len(s) > limit:
            cut = max(s.rfind(c, 0, limit) for c in "，,、：:")
            cut = cut if cut >= limit // 2 else limit - 1
            chunks.append(s[:cut + 1])
            s = s[cut + 1:]
        if len(cur) + len(s) <= limit:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


async def _run_with_disconnect(request: "Request", coro):
    """跑协程，同时盯客户端是否断开 —— 断开就取消（否则非流式请求会白烧几分钟 GPU）。"""
    task = asyncio.create_task(coro)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.5)
            if done:
                return task.result()
            if await request.is_disconnected():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
                raise HTTPException(499, "客户端已断开，已停止生成")
    except asyncio.CancelledError:
        task.cancel()
        raise


def _truncate_text(text: str) -> tuple[str, int]:
    """超过 MAX_INPUT_CHARS 就截断：优先切在句末，否则硬切。返回 (文本, 丢弃字数)。"""
    if len(text) <= MAX_INPUT_CHARS:
        return text, 0
    head = text[:MAX_INPUT_CHARS]
    cut = max(head.rfind(ch) for ch in "。！？!?；;…\n，,")
    if cut < MAX_INPUT_CHARS // 2:
        cut = MAX_INPUT_CHARS - 1
    kept = head[:cut + 1]
    return kept, len(text) - len(kept)


def _speed_tokens(speed: float, ntok: int) -> Optional[int]:
    """speed -> 目标音频帧数（1 帧 = 0.08 s）。speed=1.0 返回 None（不干预，保持模型自然长度）。"""
    if speed is None or abs(speed - 1.0) < 1e-3:
        return None
    base = int(ntok * FRAMES_PER_TOKEN) + 32          # 估算的自然帧数
    lo = max(16, int(ntok * 2.0))                     # 别压到说不出话
    hi = MAX_GEN_FRAMES
    return max(lo, min(hi, int(round(base / speed))))


def _codes_frames(p: Optional[Path]) -> int:
    """参考音色的 codes 帧数（文件头第 3 个 u32）。"""
    if p is None:
        return 0
    try:
        with open(p, "rb") as f:
            _magic, _ver, n = struct.unpack("<III", f.read(12))
        return int(n)
    except Exception:
        return 0


def _daemon_tokens_sync(text: str) -> int:
    """用常驻模型的词表精确分词（daemon 的 TOKENS op）。"""
    resp = _daemon_request("TOKENS\t" + text)
    if not resp.startswith("OK "):
        raise RuntimeError(resp.strip())
    return int(resp[3:])


async def _daemon_tokens(text: str) -> int:
    # daemon 正忙（有请求在跑）时不排队等分词——直接走字数兜底，别给请求加延迟
    if _FREE.empty():
        raise RuntimeError("daemon busy")
    return await asyncio.to_thread(_daemon_tokens_sync, text)


async def synth(text: str, ref_codes: Optional[Path], language: str, max_new_tokens: int,
                seed: int, temperature: float, top_p: float, top_k: int,
                repetition_penalty: float, mnt_explicit: bool = False):
    """合成一段，返回裸 PCM + info。

    两条路：
      · 短文本（估算 ≤ WHOLE_MAX_FRAMES）：整段解码 —— 与参考实现逐字节一致，无接缝；
      · 长文本：单 ctx 全文生成（不切句）+ 大窗口解码 —— 无接缝（decoder 侧），无长度上限。
    """
    seed = seed if seed >= 0 else int.from_bytes(os.urandom(4), "little") % (2 ** 31)
    if LOG_TEXT:
        print(f"[req] synth chars={len(text)}{_text_echo(text)}", flush=True)
    t0 = time.time()
    try:
        ntok = await _daemon_tokens(text)
    except Exception as e:                      # 词表不可用就退回字数估算
        print(f"[tokens] 精确分词失败（{e}），退回字数估算", flush=True)
        ntok = int(len(text) * 0.75)
    ref_rows = _codes_frames(ref_codes) + 32
    prompt_rows = ntok + ref_rows + 48
    est = int(ntok * FRAMES_PER_TOKEN) + 32
    if prompt_rows + 512 > CTX_ROWS_BUDGET:
        raise HTTPException(400, f"文本过长：需要 {prompt_rows} 行上下文，"
                                 f"本部署上限 {CTX_ROWS_BUDGET} 行（约 {CTX_ROWS_BUDGET} 个 token）")

    if SYNTH_MODE == "whole" or est <= WHOLE_MAX_FRAMES:
        cap = max(16, min(max_new_tokens, WHOLE_MAX_FRAMES))
        args = _daemon_args(text, ref_codes, language, cap, seed,
                            temperature, top_p, top_k, repetition_penalty,
                            overlap=SYNTH_OVERLAP, lookahead=SYNTH_LOOKAHEAD)
        _slot = _DaemonSlot()
        await _slot.acquire()          # whole 模式也走同一队列，避免绕过 slot
        try:
            pcm = await asyncio.to_thread(_daemon_whole_sync, args)
        finally:
            _slot.release()
        if not pcm:
            raise HTTPException(500, "常驻 daemon 没有返回音频")
        dur = len(pcm) / 48000.0
        elapsed = time.time() - t0
        return pcm, {"sample_rate": 24000, "channels": 1, "bits": 16,
                     "frames": len(pcm) // 2, "duration": round(dur, 3),
                     "elapsed": round(elapsed, 2),
                     "rtf": round(elapsed / dur, 3) if dur else None,
                     "mode": "whole"}

    # 长文本：ctx 超预算靠 daemon 滑窗滚动，所以这里给的预算可以放宽。
    # 没显式指定 max_new_tokens 就按估算的 2 倍 + 512 给足，避免静默截断（EOS 会提前收手）。
    if mnt_explicit:
        mnt = max_new_tokens
    else:
        mnt = max(max_new_tokens, est * 2 + 512)
    mnt = min(mnt, MAX_GEN_FRAMES)
    args = _daemon_args(text, ref_codes, language, mnt, seed,
                        temperature, top_p, top_k, repetition_penalty,
                        overlap=SYNTH_OVERLAP, lookahead=SYNTH_LOOKAHEAD,
                        window=SYNTH_WINDOW)
    pcm, info = await _async_daemon_synth(args)
    info["mode"] = "longctx" if mnt > est else "longctx-capped"
    info["mnt"] = mnt
    return pcm, info


# 解耦：daemon 生成完就释放 slot，音频留在内存队列里等客户端按自己的节奏取。
# 上限防止极慢客户端把内存吃满；超过上限才退化成"阻塞上游"（回到旧行为）。
_SPOOL_MAX = int(os.environ.get("MOSS_STREAM_BUFFER_MB", "256")) * 1024 * 1024
_SPOOL_SENTINEL = object()


async def _pump_into_queue(ds: "AsyncDaemonStream", q: "asyncio.Queue",
                            tag: str = "") -> None:
    """把 daemon 的 PCM 全部搬进内存队列；结束时**立刻**关闭 daemon 连接（释放 slot）。"""
    pending = 0
    try:
        while True:
            data = await ds.read_pcm()
            if data is None:
                break
            while pending > _SPOOL_MAX:          # 内存兜底：真放不下才阻塞上游
                pending = sum(len(x) for x in list(q._queue)) if hasattr(q, "_queue") else 0
                await asyncio.sleep(0.05)
            q.put_nowait(data)
            pending += len(data)
    except DaemonError as e:
        print(f"[stream] 常驻 daemon 失败: {e}", flush=True)
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        ds.close()                                # slot 在这里释放，与客户端读取无关
        q.put_nowait(_SPOOL_SENTINEL)


async def _pcm_stream_response(ds: "AsyncDaemonStream", fmt: str, speed: float = 1.0):
    """`stream=true` 的裸 PCM 流：pcm 直接透传；wav 先发 44 字节头（长度未知）。

    speed != 1.0 时先过一个 ffmpeg 的 atempo（保持音高）。
    """
    mime = FORMATS[fmt][1]

    if speed is None or abs(speed - 1.0) < 1e-3:
        async def gen():
            q: asyncio.Queue = asyncio.Queue()
            pump = asyncio.create_task(_pump_into_queue(ds, q))
            try:
                if fmt == "wav":
                    head = bytearray(_pcm_wav(b"", ds.rate, ds.channels))
                    head[4:8] = struct.pack("<I", 0xFFFFFFFF)
                    head[40:44] = struct.pack("<I", 0xFFFFFFFF)
                    yield bytes(head)
                while True:
                    item = await q.get()
                    if item is _SPOOL_SENTINEL:
                        break
                    yield item
            finally:
                ds.close()
                if not pump.done():
                    pump.cancel()
                with contextlib.suppress(BaseException):
                    await pump
        return StreamingResponse(gen(), media_type=mime, headers={
            "Cache-Control": "no-store", "X-Accel-Buffering": "no",
            "X-Format": fmt, "X-Streaming": "1", "X-Sample-Rate": str(ds.rate),
        })

    # speed != 1：PCM -> ffmpeg atempo -> PCM
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error",
            "-probesize", "32", "-analyzeduration", "0",
            "-f", "s16le", "-ar", str(ds.rate), "-ac", str(ds.channels), "-i", "-",
            "-filter:a", f"atempo={speed:.3f}", "-f", "s16le", "-"]

    async def gen_speed():
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, limit=1 << 16)
        except BaseException:
            ds.close()          # 启动失败也必须释放 slot
            raise
        q: asyncio.Queue = asyncio.Queue()

        async def pump_pcm():
            try:
                while True:
                    data = await ds.read_pcm()
                    if data is None:
                        break
                    proc.stdin.write(data)
                    await proc.stdin.drain()
            except (DaemonError, BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with contextlib.suppress(Exception):
                    proc.stdin.close()
                ds.close()

        async def drain_out():
            try:
                while True:
                    chunk = await proc.stdout.read(8192)
                    if not chunk:
                        break
                    q.put_nowait(chunk)
            finally:
                q.put_nowait(_SPOOL_SENTINEL)

        async def drain_err():
            with contextlib.suppress(Exception):
                err = await proc.stderr.read()
                if err:
                    print(f"[stream:{fmt}] {err.decode('utf-8','replace').strip()[:300]}", flush=True)

        t1 = asyncio.create_task(pump_pcm())
        t2 = asyncio.create_task(drain_out())
        t3 = asyncio.create_task(drain_err())
        try:
            if fmt == "wav":
                head = bytearray(_pcm_wav(b"", ds.rate, ds.channels))
                head[4:8] = struct.pack("<I", 0xFFFFFFFF)
                head[40:44] = struct.pack("<I", 0xFFFFFFFF)
                yield bytes(head)
            while True:
                item = await q.get()
                if item is _SPOOL_SENTINEL:
                    break
                yield item
        finally:
            ds.close()
            for t in (t1, t2, t3):
                if not t.done():
                    t.cancel()
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(proc.wait(), 5)
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            for t in (t1, t2, t3):
                with contextlib.suppress(BaseException):
                    await t

    return StreamingResponse(gen_speed(), media_type=mime, headers={
        "Cache-Control": "no-store", "X-Accel-Buffering": "no",
        "X-Format": fmt, "X-Streaming": "1", "X-Sample-Rate": str(ds.rate),
    })


def _atempo_filter(speed: float) -> list[str]:
    """语速：atempo 保持音高。1.0 不加滤镜。"""
    if speed is None or abs(speed - 1.0) < 1e-3:
        return []
    return ["-filter:a", f"atempo={speed:.3f}"]


def _encoder_argv(fmt: str, rate: int, channels: int,
                  speed: float = 1.0) -> list[str]:
    """流式编码器的命令行。stdin = 裸 s16le PCM，stdout = 容器字节流。"""
    if fmt == "opus":
        return ["ffmpeg", "-hide_banner", "-loglevel", "error",
                "-probesize", "32", "-analyzeduration", "0",
                "-f", "s16le", "-ar", str(rate), "-ac", str(channels), "-i", "-"] \
               + _atempo_filter(speed) + [
                "-c:a", "libopus", "-b:a", OPUS_BITRATE,
                "-f", "ogg", "-page_duration", "20000", "-"]
    if fmt == "mp3":
        return ["ffmpeg", "-hide_banner", "-loglevel", "error",
                "-probesize", "32", "-analyzeduration", "0",
                "-f", "s16le", "-ar", str(rate), "-ac", str(channels), "-i", "-"] \
               + _atempo_filter(speed) + [
                "-c:a", "libmp3lame", "-b:a", MP3_BITRATE, "-f", "mp3", "-"]
    raise HTTPException(400, f"response_format={fmt!r} 不支持流式编码")


def _encoded_stream_response(ds: "AsyncDaemonStream", fmt: str, speed: float = 1.0):
    """`stream=true` 的压缩流（opus/mp3）：ffmpeg/lame 常驻管道，边收边编边发。

    生命周期只有三条出口，全在 `finally` 里收：正常结束 / 客户端断线（框架取消 task）/
    编码器或 daemon 出错。任何一条都会关掉 daemon socket、结束编码器进程。
    """
    mime = FORMATS[fmt][1]

    async def gen():
        try:
            argv = _encoder_argv(fmt, ds.rate, ds.channels, speed)
            if shutil.which(argv[0]) is None:
                raise HTTPException(501, f"response_format={fmt!r} 需要 {argv[0]}，但容器里没有")
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1 << 16,
            )
        except BaseException:
            ds.close()          # 同理：任何失败都先还 slot
            raise

        q: asyncio.Queue = asyncio.Queue()

        async def pump_pcm():
            """daemon -> 编码器 stdin；生成结束就关 daemon 连接（释放 slot）。"""
            try:
                while True:
                    data = await ds.read_pcm()
                    if data is None:
                        break
                    proc.stdin.write(data)
                    await proc.stdin.drain()
            except (DaemonError, BrokenPipeError, ConnectionResetError) as e:
                print(f"[stream:{fmt}] 上游结束: {e}", flush=True)
            finally:
                with contextlib.suppress(Exception):
                    proc.stdin.close()
                ds.close()                     # ← slot 在这里释放，不等客户端读完

        async def drain_out():
            """编码器 stdout -> 内存队列（客户端读得慢不会反压到 GPU）。"""
            try:
                while True:
                    chunk = await proc.stdout.read(8192)
                    if not chunk:
                        break
                    q.put_nowait(chunk)
            finally:
                q.put_nowait(_SPOOL_SENTINEL)

        async def drain_stderr():
            with contextlib.suppress(Exception):
                err = await proc.stderr.read()
                if err:
                    print(f"[stream:{fmt}] {err.decode('utf-8', 'replace').strip()[:500]}", flush=True)

        producer = asyncio.create_task(pump_pcm())
        out = asyncio.create_task(drain_out())
        logs = asyncio.create_task(drain_stderr())
        sent = 0
        try:
            while True:
                item = await q.get()
                if item is _SPOOL_SENTINEL:
                    break
                sent += len(item)
                yield item
        finally:
            print(f"[req] resp done fmt={fmt} sent={sent}B", flush=True)
            # 关键顺序：先释放 daemon slot **再**做任何 await。
            # 客户端断开时这里处于取消状态，finally 里第一个 await 就会抛 CancelledError
            # （它继承 BaseException，不是 Exception），后面的清理全被跳过 —— 之前 slot 就是这么泄漏的。
            ds.close()
            for t in (producer, out, logs):
                if not t.done():
                    t.cancel()
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(proc.wait(), 5)
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            for t in (producer, out, logs):
                with contextlib.suppress(BaseException):
                    await t

    # Opus 内部固定 48 kHz（无论输入多少），mp3 保持源采样率 —— 头必须说真话，
    # 否则客户端按 24k 解码 48k 数据 → 音高低一个八度，听起来像男声。
    out_rate = 48000 if fmt == "opus" else ds.rate
    return StreamingResponse(gen(), media_type=mime, headers={
        "Cache-Control": "no-store", "X-Accel-Buffering": "no",
        "X-Format": fmt, "X-Streaming": "1", "X-Sample-Rate": str(out_rate),
    })


@app.get("/health")
async def health():
    missing = _check_model_files()
    daemon_ok = os.path.exists(DAEMON_SOCK)
    return {
        "status": "ok" if (not missing and daemon_ok) else "degraded",
        "backend": "llama-moss-tts (常驻 daemon, first-class GGUF)",
        "model": MODEL_FILE,
        "ngl": NGL,
        "missing_files": missing,
        "voices": len(_list_voices()),
        "response_formats": list(FORMATS),
        "codec_tools": {t: shutil.which(t) for t in ("ffmpeg", "lame", "opusenc")},
        "codes_cache": {
            "dir": str(CODES_CACHE),
            "files": len(list(CODES_CACHE.glob("*.codes.bin"))) if CODES_CACHE.is_dir() else 0,
            "max": CODES_CACHE_MAX,
        },
        "voices_with_codes": [v["id"] for v in _list_voices() if v.get("codes")],
        "daemon": {"socket": DAEMON_SOCK, "online": daemon_ok},
        "synth_mode": SYNTH_MODE,
        "daemons": {"count": N_DAEMONS, "free": _FREE.qsize(),
                    "sockets": [_daemon_sock(i) for i in range(N_DAEMONS)]},
        "queue": {"max": _QUEUE_MAX, "wait_sec": _QUEUE_WAIT,
                  "inflight": _INFLIGHT, "waiting": _QUEUE,
                  "rejected": dict(_STAT)},
        "nonstream_disabled": DISABLE_NONSTREAM,
        "decode_paths": {"whole_max_frames": WHOLE_MAX_FRAMES,
                         "ctx_rows": CTX_ROWS_BUDGET,
                         "frames_per_token": FRAMES_PER_TOKEN,
                         "max_gen_frames": MAX_GEN_FRAMES,
                         "split_by_sentence": False,
                         "max_input_chars": MAX_INPUT_CHARS,
                         "on_overflow": "truncate"},
        "stream": {
            "path": "/v1/audio/speech/stream",
            "request": ["prompt", "voice", "seed", "audio_format", "speed"],
            "extra_fields": "ignored",
            "default": {"voice": "default", "seed": -1, "audio_format": "wav"},
            "server_defaults": {"language": STREAM_LANGUAGE, "max_new_tokens": STREAM_MNT,
                                "temperature": STREAM_TEMP, "top_p": STREAM_TOP_P,
                                "top_k": STREAM_TOP_K, "repetition_penalty": STREAM_REP},
            "body": "裸音频流（不再有 msgpack 帧）：wav / opus(Ogg) / mp3 / pcm",
            "content_type": {"wav": "audio/wav", "opus": "audio/ogg",
                             "mp3": "audio/mpeg", "pcm": "audio/pcm"},
            "note": "出音节奏由 daemon 的解码窗口 W 决定",
        },
        "stream_window": STREAM_WINDOW,
        "stream_overlap": STREAM_OVERLAP,
        "synth_overlap": SYNTH_OVERLAP,
        "max_new_tokens_cap": MAX_NEW_TOKENS_CAP,
        "uptime_seconds": round(time.time() - started_at, 1),
    }


@app.get("/api/v1/voices")
async def list_voices():
    return _list_voices()


@app.post("/api/v1/voices")
async def add_voice(id: str = Form(...), file: UploadFile = File(...)):
    if not VOICE_ID_RE.match(id):
        raise HTTPException(400, "id 只允许字母数字和 . _ -（≤64）")
    if id == "default":
        raise HTTPException(400, "'default' 是保留 id")
    VOICES.mkdir(parents=True, exist_ok=True)
    dst = VOICES / f"{id}.wav"
    tmp = VOICES / f".{id}.upload"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        info = _wav_info(tmp)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"不是合法 WAV: {e}")
    if info["sample_rate"] != 24000 or info["channels"] != 1:
        tmp.unlink(missing_ok=True)
        raise HTTPException(
            400,
            "参考音频必须是 24 kHz mono WAV；用 "
            f"`scripts/add-voice.sh <src> {id}` 先转换 "
            f"(收到 {info['sample_rate']} Hz / {info['channels']}ch)",
        )
    tmp.replace(dst)
    # 注册时就把 codes 预算好（CPU 上做）：之后每次合成都免掉 encoder
    codes_ok = False
    h = _wav_sha8(dst)
    if h:
        # CPU 上算 codes，几秒级 —— 必须丢线程池，不能阻塞事件循环
        codes_ok = await asyncio.to_thread(_precompute_codes, dst, VOICES / f"{id}.{h}.codes.bin")
    return {"ok": True, "id": id, **info, "codes": codes_ok}


@app.delete("/api/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    p = VOICES / f"{voice_id}.wav"
    if not p.exists():
        raise HTTPException(404, "音色不存在")
    p.unlink()
    # 顺手清掉它的 codes（按内容寻址，可能有多份）
    for c in VOICES.glob(f"{voice_id}.*.codes.bin"):
        c.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/v1/synthesize")
async def synthesize(req: SynthesizeRequest, request: Request):
    fmt = _resolve_format(req.response_format)
    missing = _check_model_files()
    if missing:
        raise HTTPException(503, f"模型文件缺失: {missing}")
    ref_wav, ref_codes = await asyncio.to_thread(_ref_or_400, req.voice)
    text, dropped = _truncate_text(req.text)
    pcm, info = await _run_with_disconnect(
        request, synth(text, ref_codes, req.language, req.max_new_tokens, req.seed,
                       req.temperature, req.top_p, req.top_k, req.repetition_penalty,
                       mnt_explicit="max_new_tokens" in req.model_fields_set))
    return await _audio_response(pcm, info, fmt,
                                 extra={"X-Input-Truncated": str(dropped)} if dropped else None)


@app.post("/api/v1/synthesize/upload")
async def synthesize_with_upload(
    text: str = Form(...),
    file: UploadFile = File(...),
    language: str = Form("zh"),
    max_new_tokens: int = Form(512),
    seed: int = Form(-1),
    temperature: float = Form(1.7),
    top_p: float = Form(0.8),
    top_k: int = Form(25),
    repetition_penalty: float = Form(1.0),
    response_format: str = Form("wav"),
):
    """一次性克隆：直接上传参考音频，不落库（codes 按内容哈希进 LRU 缓存）。"""
    fmt = _resolve_format(response_format)
    missing = _check_model_files()
    if missing:
        raise HTTPException(503, f"模型文件缺失: {missing}")
    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "ref.wav"
        with open(ref, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            info = _wav_info(ref)
        except Exception as e:
            raise HTTPException(400, f"不是合法 WAV: {e}")
        if info["sample_rate"] != 24000 or info["channels"] != 1:
            raise HTTPException(400, "参考音频必须是 24 kHz mono WAV（见 scripts/add-voice.sh）")

        h = hashlib.sha256(ref.read_bytes()).hexdigest()[:32]
        cached = CODES_CACHE / f"{h}.codes.bin"
        if cached.exists():
            _lru_touch(cached)
            ref_codes = cached
        else:
            CODES_CACHE.mkdir(parents=True, exist_ok=True)
            ref_codes = cached if await asyncio.to_thread(_precompute_codes, ref, cached) else None

        pcm, out = await synth(text, ref_codes, language, max_new_tokens, seed,
                              temperature, top_p, top_k, repetition_penalty)
        if cached.exists():
            _lru_evict()
    return await _audio_response(pcm, out, fmt)


# ── OpenAI 兼容：POST /v1/audio/speech ────────────────────────────────────


@app.post("/v1/audio/speech")
async def openai_speech(req: SpeechRequest, request: Request):
    """OpenAI Create speech。`stream=true` 本栈支持（参考实现直接 400）。"""
    if req.ref_audio or req.ref_text:
        raise HTTPException(
            400,
            "本部署不支持按请求传参考音频。请先注册为音色："
            "POST /api/v1/voices (id=<voice>, file=@ref.wav)，再用 voice=<id>；"
            "现有音色见 GET /api/v1/voices",
        )
    if req.token_count or req.duration_tokens or req.tokens:
        raise HTTPException(400, "本部署不支持时长控制（token_count / ${token:N}）")
    fmt = _resolve_format(req.response_format)
    missing = _check_model_files()
    if missing:
        raise HTTPException(503, f"模型文件缺失: {missing}")

    if not req.stream and DISABLE_NONSTREAM:
        raise HTTPException(400, "本部署只提供流式合成：请用 stream=true，"
                                 "或 POST /v1/audio/speech/stream")
    voice_id = _resolve_voice(req.voice, req.model)
    ref_wav, ref_codes = await asyncio.to_thread(_ref_or_400, voice_id)
    text, dropped = _truncate_text(req.input)
    seed = req.seed if req.seed >= 0 else int.from_bytes(os.urandom(4), "little") % (2 ** 31)
    # 流式分支也要用 token 估算给足预算，否则默认 512 帧会把长文本截断在 41 s
    try:
        ntok = await _daemon_tokens(text)
    except Exception:
        ntok = int(len(text) * 0.75)
    est = int(ntok * FRAMES_PER_TOKEN) + 32
    if "max_new_tokens" in req.model_fields_set:
        mnt_stream = req.max_new_tokens
    else:
        mnt_stream = min(max(req.max_new_tokens, est * 2 + 512), MAX_GEN_FRAMES)
    args = _daemon_args(text, ref_codes, req.language, mnt_stream, seed,
                        req.temperature, req.top_p, req.top_k, req.repetition_penalty)

    if req.stream:
        ds = await AsyncDaemonStream.open(args)
        if fmt in ("mp3", "opus"):
            resp = _encoded_stream_response(ds, fmt, req.speed)      # 常驻管道：边收边编边发
        else:
            resp = await _pcm_stream_response(ds, fmt, req.speed)    # pcm 透传 / wav 未知长度头
        resp.headers["X-Voice"] = voice_id
        if dropped:
            resp.headers["X-Input-Truncated"] = str(dropped)
        return resp

    pcm, info = await _run_with_disconnect(
        request, synth(text, ref_codes, req.language, req.max_new_tokens, seed,
                       req.temperature, req.top_p, req.top_k, req.repetition_penalty,
                       mnt_explicit="max_new_tokens" in req.model_fields_set))
    extra = {"X-Voice": voice_id}
    if dropped:
        extra["X-Input-Truncated"] = str(dropped)
    return await _audio_response(pcm, info, fmt, extra=extra)


@app.post("/v1/audio/speech/stream")
async def openai_speech_stream(req: ChunkStreamRequest):
    """真流式接口：响应体是**裸音频流**（`audio/ogg` / `audio/mpeg` / `audio/wav` / `audio/pcm`）。

    2026-10-18 起不再返回 msgpack 分块。`audio_format` 决定编码；
    出音节奏由 daemon 的解码窗口 W 决定。
    """
    fmt = _resolve_format(req.audio_format)
    text = req.prompt.strip()
    if not text:
        raise HTTPException(400, "prompt 不能为空")
    missing = _check_model_files()
    if missing:
        raise HTTPException(503, f"模型文件缺失: {missing}")

    voice_id = _resolve_voice(req.voice)
    ref_wav, ref_codes = await asyncio.to_thread(_ref_or_400, voice_id)   # 可能含 CPU 预算 codes
    text, dropped = _truncate_text(text)
    seed = req.seed if req.seed >= 0 else int.from_bytes(os.urandom(4), "little") % (2 ** 31)
    try:
        ntok = await _daemon_tokens(text)
    except Exception:
        ntok = int(len(text) * 0.75)
    est = int(ntok * FRAMES_PER_TOKEN) + 32
    mnt = min(max(STREAM_MNT, est * 2 + 512), MAX_GEN_FRAMES)   # 不再固定 2048 帧（164 s）
    args = _daemon_args(text, ref_codes, STREAM_LANGUAGE, mnt, seed,
                        STREAM_TEMP, STREAM_TOP_P, STREAM_TOP_K, STREAM_REP)
    ds = await AsyncDaemonStream.open(args)      # 开流：参数/daemon 问题 -> 正常 HTTP 错误
    print(f"[req] stream fmt={fmt} chars={len(text)} mnt={mnt} voice={voice_id}"
          f"{_text_echo(text)}", flush=True)
    # 2026-10-18：body 改成**裸音频流**（ogg/opus、mp3、wav、pcm），不再套 msgpack 帧。
    if fmt in ("mp3", "opus"):
        resp = _encoded_stream_response(ds, fmt, req.speed)
    else:
        resp = await _pcm_stream_response(ds, fmt, req.speed)
    resp.headers["X-Voice"] = voice_id
    if dropped:
        resp.headers["X-Input-Truncated"] = str(dropped)
    return resp


@app.get("/v1/models")
async def openai_models():
    now = int(time.time())
    data = [{"id": "moss-tts-v1.5", "object": "model", "created": now, "owned_by": "local"}]
    for v in _list_voices():
        if v["id"] == "default":
            continue
        data.append({"id": f"moss-tts-v1.5-{v['id']}-reference",
                     "object": "model", "created": now, "owned_by": "local"})
    return {"object": "list", "data": data}
