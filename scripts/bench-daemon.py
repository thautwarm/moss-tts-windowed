#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1 压测客户端 v2：常驻 daemon 的冷启 / 稳态 / 长度上限 / 参考 codes 四条线。

协议（见 patches/0003）：unix socket 上每条请求 = 一行制表符分隔的 CLI 参数；
daemon 回一行 `OK <内部毫秒>` 或 `ERR <内部毫秒> <消息>`。
"""
import argparse
import json
import os
import socket
import struct
import time

SHORT_TEXT = "你好，世界。这会儿方便说两句话吗？"
LONG_TEXT = (
    "今天的天气比昨天好一些，早上出门的时候还有一点薄雾，到了中午就散开了。"
    "我沿着河边走了大概半小时，看见几只白鹭停在浅水里，应该是等着抓鱼。"
    "回家以后泡了一杯茶，把上周没看完的那本书翻了几页，讲的是城市里的老建筑怎么被保留下来的。"
    "作者说他小时候住的那条街，房子还在，但住的人已经换了两三茬。"
    "我觉得这种变化挺正常的，只是需要有人把它记下来，不然过二十年就没人说得清了。"
)


def wav_info(path):
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return {}
            rate = channels = bits = None
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    return {}
                cid = hdr[:4]
                size = struct.unpack("<I", hdr[4:])[0]
                if cid == b"fmt ":
                    fmt = f.read(size)
                    channels = struct.unpack_from("<H", fmt, 2)[0]
                    rate = struct.unpack_from("<I", fmt, 4)[0]
                    bits = struct.unpack_from("<H", fmt, 14)[0]
                elif cid == b"data":
                    dur = size / (rate * channels * bits // 8) if rate else 0.0
                    return {"rate": rate, "channels": channels, "bits": bits,
                            "bytes": size + 44, "duration": round(dur, 3)}
                else:
                    f.seek(size + (size & 1), 1)
    except OSError:
        return {}


class Client:
    def __init__(self, sock_path, timeout=900):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(sock_path)
        self.buf = b""

    def request(self, tokens):
        self.s.sendall(("\t".join(tokens) + "\n").encode("utf-8"))
        while b"\n" not in self.buf:
            chunk = self.s.recv(65536)
            if not chunk:
                raise RuntimeError("daemon 关闭了连接")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode("utf-8", "replace").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-file", default="moss_tts_v15_firstclass_q4_k_m_aq8.gguf")
    ap.add_argument("--steady-runs", type=int, default=5)
    ap.add_argument("--sweep", default="320,384,448,512,576,640")
    ap.add_argument("--ref-voice", default="")
    ap.add_argument("--ref-sha8", default="3b004866")
    args = ap.parse_args()

    cli = Client(args.sock)
    runs = []

    def req(tag, text, mnt, seed, ref_codes=None):
        toks = ["--text", text, "--language", "zh",
                "--max-new-tokens", str(mnt),
                "--audio-temperature", "0.8", "--audio-top-p", "0.9", "--audio-top-k", "50",
                "--audio-repetition-penalty", "1.0",
                "--seed", str(seed)]
        if ref_codes:
            toks += ["--reference-codes", ref_codes]
        wav = os.path.join(args.out, "m1-%s.wav" % tag)
        if os.path.exists(wav):
            os.unlink(wav)
        toks += ["--wav-out", wav]
        t0 = time.time()
        resp = cli.request(toks)
        wall = (time.time() - t0) * 1000.0
        info = wav_info(wav)
        dur = info.get("duration") or 0.0
        rec = {"tag": tag, "resp": resp, "wall_ms": round(wall, 1),
               "daemon_ms": int(resp.split()[1]) if resp.split()[1:2] and resp.split()[1].isdigit() else None,
               "chars": len(text), "mnt": mnt, "audio_s": dur,
               "rtf_wall": round(wall / 1000.0 / dur, 3) if dur else None,
               "rtf_daemon": round(rec_daemon(resp) / 1000.0 / dur, 3) if dur else None,
               "ok": resp.startswith("OK")}
        runs.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        return rec

    def rec_daemon(resp):
        parts = resp.split()
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    ref_codes = os.path.join(args.voices, "%s.%s.codes.bin" % (args.ref_voice, args.ref_sha8))

    # 1) 冷启：第一个请求 = 三段模型加载 + 生成
    req("cold", "你好。", 96, 1)

    # 2) 长度上限扫描：同样的文本，调 --max-new-tokens 找解码阶段能塞进显存的上限
    for mnt in [int(x) for x in args.sweep.split(",") if x.strip()]:
        r = req("sweep-%d" % mnt, LONG_TEXT, mnt, 11)
        req("sweep-%d-health" % mnt, "好。", 64, 12)

    # 3) 稳态：安全长度下连续请求
    safe = 448
    for i in range(args.steady_runs):
        req("steady-long-%d" % i, LONG_TEXT, safe, 100 + i)

    for i in range(3):
        req("steady-short-%d" % i, SHORT_TEXT, 256, 200 + i)

    # 4) 参考 codes（只给 codes，不碰 encoder）—— 前端就是这么用的
    for i in range(3):
        req("ref-codes-%d" % i, SHORT_TEXT, 256, 300 + i, ref_codes=ref_codes)

    with open(os.path.join(args.out, "m1.json"), "w", encoding="utf-8") as fh:
        json.dump(runs, fh, ensure_ascii=False, indent=2)

    print("\n== 汇总 ==")
    print("%-22s %10s %8s %8s %7s %7s %s" % ("tag", "wall_ms", "dm_ms", "audio_s", "rtf_w", "rtf_d", "ok"))
    for r in runs:
        print("%-22s %10.0f %8s %8.2f %7s %7s %s" % (
            r["tag"], r["wall_ms"], r["daemon_ms"], r["audio_s"] or 0.0,
            r["rtf_wall"], r["rtf_daemon"], r["ok"]))


if __name__ == "__main__":
    main()
