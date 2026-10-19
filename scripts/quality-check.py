#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""质量对照：同一文本/种子下，「整段解码」与「流式窗口解码」两路音频互相对照。

  * 两路都送外部 ASR 服务转写，跟原文比字符覆盖率
  * 再算两路 PCM 的逐样本相关/差异（长度对齐后），看窗口边界有没有明显劣化
  * 生成的 wav 留在 results/ 供试听

用法：
  python3 scripts/quality-check.py                 # 默认文本
  python3 scripts/quality-check.py --text "..." --max-new-tokens 320 --window 8 --overlap 8
"""
import argparse
import difflib
import json
import os
import socket
import struct
import subprocess
import urllib.request

DEFAULT_TEXT = (
    "今天的天气比昨天好一些，早上出门的时候还有一点薄雾，到了中午就散开了。"
    "我沿着河边走了大概半小时，看见几只白鹭停在浅水里，应该是等着抓鱼。"
    "回家以后泡了一杯茶，把上周没看完的那本书翻了几页。"
)


def wav_read(path):
    with open(path, "rb") as fh:
        data = fh.read()
    rate = ch = bits = None
    pos = 12
    pcm = b""
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            ch = struct.unpack_from("<H", body, 2)[0]
            rate = struct.unpack_from("<I", body, 4)[0]
            bits = struct.unpack_from("<H", body, 14)[0]
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    samples = struct.unpack("<%dh" % (len(pcm) // 2), pcm)
    return rate, ch, bits, samples


def whole_request(sock_path, toks):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(900)
    s.connect(sock_path)
    s.sendall(("\t".join(toks) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            return "连接关闭"
        buf += chunk
    return buf.split(b"\n")[0].decode()


def stream_request(sock_path, toks, wav_out):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(900)
    s.connect(sock_path)
    s.sendall(("STREAM\t" + "\t".join(toks) + "\n").encode())
    buf = b""

    def fill():
        nonlocal buf
        chunk = s.recv(65536)
        if not chunk:
            raise RuntimeError("连接关闭")
        buf += chunk

    def read_line():
        nonlocal buf
        while b"\n" not in buf:
            fill()
        line, buf = buf.split(b"\n", 1)
        return line.decode()

    def read_exact(n):
        nonlocal buf
        while len(buf) < n:
            fill()
        out, buf = buf[:n], buf[n:]
        return out

    rate, pcm = None, b""
    while True:
        line = read_line()
        if line.startswith("S "):
            rate = int(line.split()[1])
        elif line.startswith("A "):
            pcm += read_exact(int(line.split()[1]))
        elif line.startswith("ERR"):
            return "ERR: " + line
        elif line.startswith("E"):
            break
    with open(wav_out, "wb") as fh:
        fh.write(struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE",
                             b"fmt ", 16, 1, 1, rate, rate * 2, 2, 16, b"data", len(pcm)))
        fh.write(pcm)
    return "OK"


def asr(path, endpoint=os.environ.get("ASR", "http://127.0.0.1:8082/transcribe")):
    with open(path, "rb") as fh:
        body = fh.read()
    boundary = "----mossstream"
    head = ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n" % boundary).encode()
    tail = ("\r\n--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(endpoint, data=head + body + tail,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode()).get("text", "")
    except Exception as exc:  # noqa: BLE001
        return "<ASR 失败: %s>" % exc


def coverage(expected, got):
    """字符级覆盖率：转写里能对上原文的比例（去标点空白）"""
    strip = lambda s: "".join(ch for ch in s if ch.isalnum())
    exp, g = strip(expected), strip(got)
    if not exp:
        return 0.0
    sm = difflib.SequenceMatcher(None, exp, g)
    return sum(b.size for b in sm.get_matching_blocks()) / len(exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="results/serve.sock")
    ap.add_argument("--voices", default="voices")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--ref-voice", default="", help="参考音色 id（codes 必须已预计算）")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--skip-asr", action="store_true")
    args = ap.parse_args()

    common = ["--text", args.text, "--language", "zh",
              "--max-new-tokens", str(args.max_new_tokens),
              "--audio-temperature", "1.7", "--audio-top-p", "0.8", "--audio-top-k", "25",
              "--audio-repetition-penalty", "1.0", "--seed", str(args.seed),
              "--reference-codes", os.path.join(args.voices, args.ref_voice + ".codes.bin")]

    wav_whole = os.path.join(args.outdir, "qc-whole.wav")
    wav_stream = os.path.join(args.outdir, "qc-stream.wav")

    r1 = whole_request(args.sock, common + ["--wav-out", "/results/qc-whole.wav"])
    print("整段解码:", r1)
    r2 = stream_request(args.sock, common + ["--stream-window", str(args.window),
                                             "--stream-overlap", str(args.overlap)], wav_stream)
    print("流式解码:", r2)

    rate_a, _, _, a = wav_read(wav_whole)
    rate_b, _, _, b = wav_read(wav_stream)
    dur_a, dur_b = len(a) / rate_a, len(b) / rate_b
    print("音频: 整段 %.2f s / 流式 %.2f s（差 %.2f s）" % (dur_a, dur_b, dur_b - dur_a))

    n = min(len(a), len(b))
    if n:
        import math
        dot = sum(a[i] * b[i] for i in range(0, n, 7))
        na = math.sqrt(sum(a[i] * a[i] for i in range(0, n, 7)))
        nb = math.sqrt(sum(b[i] * b[i] for i in range(0, n, 7)))
        print("两路 PCM 相关系数（抽 1/7 样本，等长前缀）: %.4f" % (dot / (na * nb + 1e-9)))

    if not args.skip_asr:
        ta = asr(wav_whole)
        tb = asr(wav_stream)
        print("\nASR 整段: %s" % ta)
        print("ASR 流式: %s" % tb)
        print("原文     : %s" % args.text)
        print("\n字符覆盖率  整段 %.3f   流式 %.3f" % (coverage(args.text, ta), coverage(args.text, tb)))


if __name__ == "__main__":
    main()
