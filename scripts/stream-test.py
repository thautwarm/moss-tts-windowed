#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式请求测试：发 `STREAM\t` 请求，读 `S` 头 + `A` 音频块，记录**首块延迟**，
并把 PCM 拼成 wav 落盘（供 ASR / 试听）。

协议（patch 0004）：
    S <rate> <channels> <bits>\n      ← 流开头
    A <nbytes>\n<raw PCM16LE>         ← 一块音频（可多块）
    E\n                               ← 正常结束
    ERR <ms> <msg>\n                  ← 出错（可能出现在任意位置）
"""
import argparse
import os
import socket
import struct
import time

LONG_TEXT = (
    "今天的天气比昨天好一些，早上出门的时候还有一点薄雾，到了中午就散开了。"
    "我沿着河边走了大概半小时，看见几只白鹭停在浅水里，应该是等着抓鱼。"
    "回家以后泡了一杯茶，把上周没看完的那本书翻了几页，讲的是城市里的老建筑怎么被保留下来的。"
    "作者说他小时候住的那条街，房子还在，但住的人已经换了两三茬。"
    "我觉得这种变化挺正常的，只是需要有人把它记下来，不然过二十年就没人说得清了。"
    "后来我又想到，其实很多手艺也是这样，会的人少了，但总会剩下几个。"
    "他们把东西做得很慢，也不着急，做完了就放在那里，等需要的人来取。"
)
SHORT_TEXT = "你好，世界。这会儿方便说两句话吗？"


def write_wav(path, pcm, rate, channels=1, bits=16):
    data = struct.pack("<4sI4s4sIHHIIHH4sI",
                       b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, channels,
                       rate, rate * channels * bits // 8, channels * bits // 8, bits,
                       b"data", len(pcm))
    with open(path, "wb") as fh:
        fh.write(data)
        fh.write(pcm)


class StreamClient:
    def __init__(self, sock_path, timeout=900):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(sock_path)
        self.buf = b""

    def _fill(self):
        chunk = self.s.recv(65536)
        if not chunk:
            raise RuntimeError("连接被关闭")
        self.buf += chunk

    def read_line(self):
        while b"\n" not in self.buf:
            self._fill()
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode("utf-8", "replace")

    def read_exact(self, n):
        while len(self.buf) < n:
            self._fill()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def stream(self, tokens, wav_out):
        self.s.sendall(("STREAM\t" + "\t".join(tokens) + "\n").encode("utf-8"))
        t0 = time.time()
        rate, pcm, chunks = None, b"", []
        err = None
        while True:
            line = self.read_line()
            if line.startswith("S "):
                _, rate, ch, bits = line.split()
                rate, ch, bits = int(rate), int(ch), int(bits)
            elif line.startswith("A "):
                n = int(line.split()[1])
                data = self.read_exact(n)
                chunks.append(round((time.time() - t0) * 1000.0, 1))
                pcm += data
            elif line.startswith("ERR"):
                err = line
                break
            elif line.startswith("E"):
                break
            else:
                err = "协议意外：" + line
                break
        total = (time.time() - t0) * 1000.0
        if rate:
            write_wav(wav_out, pcm, rate)
            audio_s = len(pcm) / (rate * 2)
        else:
            audio_s = 0.0
        return {"rate": rate, "chunks": len(chunks), "first_chunk_ms": chunks[0] if chunks else None,
                "chunk_arrivals_ms": chunks[:6], "total_ms": round(total, 1),
                "audio_s": round(audio_s, 3), "bytes": len(pcm), "err": err,
                "rtf": round(total / 1000.0 / audio_s, 3) if audio_s else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="results/serve.sock")
    ap.add_argument("--voices", default="voices")
    ap.add_argument("--ref-voice", default="")
    ap.add_argument("--ref-sha8", default="3b004866")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tag", default="stream")
    ap.add_argument("--text", default="")
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=0)
    args = ap.parse_args()

    text = SHORT_TEXT if args.short else (args.text or LONG_TEXT)
    toks = ["--text", text, "--language", "zh",
            "--max-new-tokens", str(args.max_new_tokens),
            "--audio-temperature", "1.7", "--audio-top-p", "0.8", "--audio-top-k", "25",
            "--audio-repetition-penalty", "1.0", "--seed", str(args.seed),
            "--reference-codes", os.path.join(args.voices, "%s.%s.codes.bin" % (args.ref_voice, args.ref_sha8))]
    if args.window:
        toks += ["--stream-window", str(args.window)]
    if args.overlap:
        toks += ["--stream-overlap", str(args.overlap)]
    cli = StreamClient(args.sock)
    wav = os.path.join(args.outdir, "stream-%s.wav" % args.tag)
    info = cli.stream(toks, wav)
    print(info)
    print("wav ->", wav)


if __name__ == "__main__":
    main()
