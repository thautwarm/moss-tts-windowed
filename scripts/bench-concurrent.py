#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并发压测：K 个请求**同时**打 /v1/audio/speech/stream，看总墙钟与单请求耗时。

用来验证 MOSS_DAEMONS 的收益（单 slot vs 多实例）：
    python3 scripts/bench-concurrent.py --k 2 --chars 280
    python3 scripts/bench-concurrent.py --k 4 --repeat 3
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

BASE_TEXT = "这是一段用来做并发压测的文本，长度可以通过 --chars 控制，内容本身不重要，重要的是它要生成足够长的音频。"


def one_request(url: str, payload: dict) -> tuple[float, float, int]:
    """返回 (首字节, 总耗时, 字节数)"""
    req = urllib.request.Request(
        url + "/v1/audio/speech/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    first = 0.0
    total_bytes = 0
    with urllib.request.urlopen(req, timeout=900) as resp:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            if first == 0.0:
                first = time.time() - t0
            total_bytes += len(chunk)
    return first, time.time() - t0, total_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:6677")
    ap.add_argument("--k", type=int, default=2, help="并发请求数")
    ap.add_argument("--chars", type=int, default=280, help="每个请求的文本长度")
    ap.add_argument("--voice", default="", help="默认取注册表里第一个可用音色")
    ap.add_argument("--repeat", type=int, default=1, help="重复几轮（取中位）")
    args = ap.parse_args()

    url = args.url.rstrip("/")
    voice = args.voice
    if not voice:
        with urllib.request.urlopen(url + "/api/v1/voices", timeout=30) as r:
            vs = json.load(r)
        ids = [v["id"] for v in vs if v.get("code") or v.get("codes")]
        voice = ids[0] if ids else "default"

    text = (BASE_TEXT * (args.chars // len(BASE_TEXT) + 1))[: args.chars]

    totals, firsts = [], []
    for rnd in range(args.repeat):
        results: list[tuple[float, float, int]] = []
        lock = threading.Lock()
        barrier = threading.Barrier(args.k)

        def worker(i: int) -> None:
            payload = {"prompt": text, "voice": voice, "seed": 1000 + i, "audio_format": "opus"}
            barrier.wait()                       # 尽量同时发车
            f, t, n = one_request(url, payload)
            with lock:
                results.append((f, t, n))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.k)]
        t0 = time.time()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        total = time.time() - t0

        totals.append(total)
        firsts.append(min(f for f, _, _ in results))
        # opus 80 kbps → 音频秒数 ≈ 字节 / 10000
        # opus 80 kbps → 音频秒数 ≈ 字节 / 10000；RTF = 生成耗时 / 音频时长（<1 = 比实时快）
        audio = [n / 10000.0 for _, _, n in results]
        rtfs = [t / a for (_, t, _), a in zip(results, audio)]
        print("round %d: 总墙钟 %.2fs | 单请求 %.2fs | RTF 中位 %.3f（最差 %.3f，即 %.1f× 实时）"
              " | 首字节 %.2fs | 音频 %.1fs/请求"
              % (rnd + 1, total,
                 statistics.median([t for _, t, _ in results]),
                 statistics.median(rtfs), max(rtfs), 1.0 / max(rtfs),
                 firsts[-1], statistics.median(audio)))

    print("\nK=%d chars=%d voice=%s → 总墙钟中位 %.2fs，首字节中位 %.2fs"
          % (args.k, args.chars, voice, statistics.median(totals), statistics.median(firsts)))


if __name__ == "__main__":
    main()
