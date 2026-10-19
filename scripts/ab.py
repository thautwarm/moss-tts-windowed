#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B 压测：同一组请求分别在「对照组服务（HTTP）」和「本栈常驻 daemon（unix socket）」
上跑，参数逐字段一致，输出可比对的表。

  # 对照组（先 cd "$BASELINE_DIR" && docker compose up -d）
  python3 scripts/ab.py --mode http --url http://127.0.0.1:6677 --out results/ab-incumbent.json

  # 本栈（先 scripts/daemon-up.sh）
  python3 scripts/ab.py --mode unix --sock results/serve.sock --out results/ab-ours.json

两边的采样参数固定为 对照组服务 /v1/audio/speech 的默认值：
  temperature=1.7  top_p=0.8  top_k=25  repetition_penalty=1.0
"""
import argparse
import json
import os
import socket
import struct
import time
import urllib.request

LONG_TEXT = (
    "今天的天气比昨天好一些，早上出门的时候还有一点薄雾，到了中午就散开了。"
    "我沿着河边走了大概半小时，看见几只白鹭停在浅水里，应该是等着抓鱼。"
    "回家以后泡了一杯茶，把上周没看完的那本书翻了几页，讲的是城市里的老建筑怎么被保留下来的。"
    "作者说他小时候住的那条街，房子还在，但住的人已经换了两三茬。"
    "我觉得这种变化挺正常的，只是需要有人把它记下来，不然过二十年就没人说得清了。"
)
SHORT_TEXT = "你好，世界。这会儿方便说两句话吗？"

TEMP, TOP_P, TOP_K, REP = 1.7, 0.8, 25, 1.0


def wav_info_bytes(data: bytes):
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return {}
    pos, rate, ch, bits = 12, None, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            ch = struct.unpack_from("<H", body, 2)[0]
            rate = struct.unpack_from("<I", body, 4)[0]
            bits = struct.unpack_from("<H", body, 14)[0]
        elif cid == b"data":
            dur = size / (rate * ch * bits // 8) if rate else 0.0
            return {"rate": rate, "channels": ch, "bits": bits,
                    "bytes": len(data), "duration": round(dur, 3)}
        pos += 8 + size + (size & 1)
    return {}


class HttpBackend:
    kind = "http"

    def __init__(self, url, voice):
        self.url, self.voice = url.rstrip("/"), voice

    def synth(self, text, mnt, seed):
        body = json.dumps({
            "input": text, "voice": self.voice, "language": "zh",
            "max_new_tokens": mnt, "seed": seed,
            "temperature": TEMP, "top_p": TOP_P, "top_k": TOP_K,
            "repetition_penalty": REP, "response_format": "wav",
        }).encode()
        req = urllib.request.Request(self.url + "/v1/audio/speech", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, r.read()


class UnixBackend:
    kind = "unix"

    def __init__(self, sock_path, voices, ref_voice, ref_sha8, out_dir):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(900)
        self.s.connect(sock_path)
        self.buf = b""
        self.codes = os.path.join(voices, "%s.%s.codes.bin" % (ref_voice, ref_sha8))
        self.out = out_dir

    def synth(self, text, mnt, seed, tag="x"):
        toks = ["--text", text, "--language", "zh", "--max-new-tokens", str(mnt),
                "--audio-temperature", str(TEMP), "--audio-top-p", str(TOP_P),
                "--audio-top-k", str(TOP_K), "--audio-repetition-penalty", str(REP),
                "--seed", str(seed), "--reference-codes", self.codes,
                "--wav-out", os.path.join(self.out, "ab-%s.wav" % tag)]
        self.s.sendall(("\t".join(toks) + "\n").encode())
        while b"\n" not in self.buf:
            chunk = self.s.recv(65536)
            if not chunk:
                return -1, b""
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        line = line.decode("utf-8", "replace").strip()
        rc = 0 if line.startswith("OK") else 1
        try:
            with open(os.path.join(self.out, "ab-%s.wav" % tag), "rb") as fh:
                return rc, fh.read()
        except OSError:
            return rc, b""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["http", "unix"], required=True)
    ap.add_argument("--url", default="http://127.0.0.1:6677")
    ap.add_argument("--sock", default="results/serve.sock")
    ap.add_argument("--voices", default="voices")
    ap.add_argument("--ref-voice", default="")
    ap.add_argument("--ref-sha8", default="3b004866")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.mode == "http":
        be = HttpBackend(args.url, args.ref_voice)
        cases = [("long-cold", LONG_TEXT, 512), ("short-cold", SHORT_TEXT, 256)]
        cases += [("long-warm-%d" % i, LONG_TEXT, 512) for i in range(args.warm)]
        cases += [("short-warm-%d" % i, SHORT_TEXT, 256) for i in range(args.warm)]
    else:
        be = UnixBackend(args.sock, args.voices, args.ref_voice, args.ref_sha8, args.outdir)
        cases = [("long-cold", LONG_TEXT, 512), ("short-cold", SHORT_TEXT, 256)]
        cases += [("long-warm-%d" % i, LONG_TEXT, 512) for i in range(args.warm)]
        cases += [("short-warm-%d" % i, SHORT_TEXT, 256) for i in range(args.warm)]

    runs = []
    for i, (tag, text, mnt) in enumerate(cases):
        t0 = time.time()
        rc, data = be.synth(text, mnt, 400 + i, tag=tag) if args.mode == "unix" else be.synth(text, mnt, 400 + i)
        wall = (time.time() - t0) * 1000.0
        info = wav_info_bytes(data)
        dur = info.get("duration") or 0.0
        rec = {"tag": tag, "rc": rc, "wall_ms": round(wall, 1), "audio_s": dur,
               "rtf": round(wall / 1000.0 / dur, 3) if dur else None, "bytes": len(data)}
        runs.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"mode": args.mode, "runs": runs}, fh, ensure_ascii=False, indent=2)

    print("\n== %s ==" % args.mode)
    print("%-16s %10s %9s %7s %8s" % ("tag", "wall_ms", "audio_s", "rtf", "bytes"))
    for r in runs:
        print("%-16s %10.0f %9.2f %7s %8d" % (r["tag"], r["wall_ms"], r["audio_s"] or 0.0, r["rtf"], r["bytes"]))
    warm = [r for r in runs if "warm" in r["tag"]]
    if warm:
        long_w = [r["rtf"] for r in warm if r["tag"].startswith("long") and r["rtf"]]
        short_w = [r["rtf"] for r in warm if r["tag"].startswith("short") and r["rtf"]]
        if long_w:
            print("长句稳态 RTF：中位 %.3f（%s）" % (sorted(long_w)[len(long_w) // 2], long_w))
        if short_w:
            print("短句稳态 RTF：中位 %.3f（%s）" % (sorted(short_w)[len(short_w) // 2], short_w))


if __name__ == "__main__":
    main()
