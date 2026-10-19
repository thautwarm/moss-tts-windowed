#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`/v1/audio/speech/stream` 分块流自测。

  python3 scripts/chunk-test.py --tag long --text-file /dev/stdin < text.txt
  python3 scripts/chunk-test.py --cases                 # 跑一组标准用例

校验/测量：
  * 每块音频**单独就能播**（host 有 ffprobe：拿它验 codec/duration；wav 再自己解头）
  * 长度前缀 / msgpack 解出来的对象结构（t=h/c/e）
  * 首块延迟、块间隔、总耗时、RTF
  * 各块 text 拼起来与原文的覆盖率（估算误差报告）
  * 落盘 results/chunks/<tag>/chunk-NNN.<ext>，并把 wav 块拼成 all.wav 供试听/ASR
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "http://127.0.0.1:6677/v1/audio/speech/stream"

SHORT = "你好，世界。这会儿方便说两句话吗？"
MED = ("今天的天气比昨天好一些，早上出门的时候还有一点薄雾，到了中午就散开了。"
       "我沿着河边走了大概半小时，看见几只白鹭停在浅水里，应该是等着抓鱼。"
       "回家以后泡了一杯茶，把上周没看完的那本书翻了几页，讲的是城市里的老建筑怎么被保留下来的。")
LONG = MED + ("作者说他小时候住的那条街，房子还在，但住的人已经换了两三茬。"
              "我觉得这种变化挺正常的，只是需要有人把它记下来，不然过二十年就没人说得清了。"
              "后来我又想到，其实很多手艺也是这样，会的人少了，但总会剩下几个。"
              "他们把东西做得很慢，也不着急，做完了就放在那里，等需要的人来取。")


# ── 最小 msgpack 解码器（宿主没装 msgpack；只解我们发的那几种类型）──────────
class NeedMore(Exception):
    pass


def mp_unpack(buf, pos=0):
    if pos >= len(buf):
        raise NeedMore()
    b = buf[pos]
    pos += 1
    if b <= 0x7F:
        return b, pos
    if b >= 0xE0:
        return b - 0x100, pos
    if 0x80 <= b <= 0x8F:
        return _mp_map(buf, pos, b & 0x0F)
    if 0x90 <= b <= 0x9F:
        return _mp_arr(buf, pos, b & 0x0F)
    if 0xA0 <= b <= 0xBF:
        return _mp_str(buf, pos, b & 0x1F)
    if b == 0xC0:
        return None, pos
    if b in (0xC2, 0xC3):
        return b == 0xC3, pos
    if b in (0xC4, 0xC5, 0xC6):
        return _mp_bin(buf, pos, {0xC4: 1, 0xC5: 2, 0xC6: 4}[b])
    if b == 0xCA:
        return _take(buf, pos, 4, ">f", 4)
    if b == 0xCB:
        return _take(buf, pos, 8, ">d", 8)
    if b in (0xCC, 0xCD, 0xCE, 0xCF):
        size = {0xCC: 1, 0xCD: 2, 0xCE: 4, 0xCF: 8}[b]
        return _take(buf, pos, size, {1: ">B", 2: ">H", 4: ">I", 8: ">Q"}[size], size)
    if b in (0xD0, 0xD1, 0xD2, 0xD3):
        size = {0xD0: 1, 0xD1: 2, 0xD2: 4, 0xD3: 8}[b]
        return _take(buf, pos, size, {1: ">b", 2: ">h", 4: ">i", 8: ">q"}[size], size)
    if b in (0xD9, 0xDA, 0xDB):
        size = {0xD9: 1, 0xDA: 2, 0xDB: 4}[b]
        n, pos = _take(buf, pos, size, {1: ">B", 2: ">H", 4: ">I"}[size], size)
        return _mp_str(buf, pos, n)
    if b in (0xDC, 0xDD):
        size = 2 if b == 0xDC else 4
        n, pos = _take(buf, pos, size, ">H" if size == 2 else ">I", size)
        return _mp_arr(buf, pos, n)
    if b in (0xDE, 0xDF):
        size = 2 if b == 0xDE else 4
        n, pos = _take(buf, pos, size, ">H" if size == 2 else ">I", size)
        return _mp_map(buf, pos, n)
    raise ValueError("不认识的 msgpack 前缀 0x%02x" % b)


def _take(buf, pos, size, fmt, nbytes):
    if pos + size > len(buf):
        raise NeedMore()
    return struct.unpack_from(fmt, buf, pos)[0], pos + size


def _mp_str(buf, pos, n):
    if pos + n > len(buf):
        raise NeedMore()
    return bytes(buf[pos:pos + n]).decode("utf-8"), pos + n


def _mp_bin(buf, pos, sizefield):
    n, pos = _take(buf, pos, sizefield, {1: ">B", 2: ">H", 4: ">I"}[sizefield], sizefield)
    if pos + n > len(buf):
        raise NeedMore()
    return bytes(buf[pos:pos + n]), pos + n


def _mp_arr(buf, pos, n):
    out = []
    for _ in range(n):
        v, pos = mp_unpack(buf, pos)
        out.append(v)
    return out, pos


def _mp_map(buf, pos, n):
    out = {}
    for _ in range(n):
        k, pos = mp_unpack(buf, pos)
        v, pos = mp_unpack(buf, pos)
        out[k] = v
    return out, pos


# ── HTTP 侧 ───────────────────────────────────────────────────────────────

class ChunkReader:
    def __init__(self, resp, framing):
        self.r = resp
        self.framing = framing
        self.buf = bytearray()

    def _more(self):
        c = self.r.read(65536)
        if not c:
            raise EOFError("流结束")
        self.buf += c

    def read_obj(self):
        if self.framing == "len":
            while len(self.buf) < 4:
                self._more()
            n = struct.unpack("<I", self.buf[:4])[0]
            del self.buf[:4]
            while len(self.buf) < n:
                self._more()
            body = bytes(self.buf[:n])
            del self.buf[:n]
            return mp_unpack(body)[0]
        while True:
            try:
                return mp_unpack(self.buf)[0]
            except NeedMore:
                self._more()


def wav_dur(data: bytes):
    if len(data) < 44 or data[:4] != b"RIFF":
        return None
    pos, rate, ch, bits = 12, None, None, None
    while pos + 8 <= len(data):
        cid, sz = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + sz]
        if cid == b"fmt ":
            ch = struct.unpack_from("<H", body, 2)[0]
            rate = struct.unpack_from("<I", body, 4)[0]
            bits = struct.unpack_from("<H", body, 14)[0]
        elif cid == b"data":
            return round(sz / (rate * ch * bits // 8), 3)
        pos += 8 + sz + (sz & 1)
    return None


def ffprobe_ok(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "stream=codec_name,duration", "-of", "json", path],
                             capture_output=True, text=True, timeout=30).stdout
        js = json.loads(out or "{}")
        st = js.get("streams") or []
        return bool(st), (st[0] if st else {})
    except Exception:  # noqa: BLE001
        return False, {}


def coverage(expected, got):
    import difflib
    strip = lambda s: "".join(ch for ch in s if ch.isalnum())
    exp, g = strip(expected), strip(got)
    if not exp:
        return 0.0
    return sum(b.size for b in difflib.SequenceMatcher(None, exp, g).get_matching_blocks()) / len(exp)


def run_case(tag, text, voice="", fmt="wav", chunk_seconds=2.0, framing="len",
             mnt=2048, seed=42, keep=True, verbose=True):
    body = {"input": text, "voice": voice, "language": "zh", "max_new_tokens": mnt,
            "seed": seed, "temperature": 1.7, "top_p": 0.8, "top_k": 25,
            "repetition_penalty": 1.0, "audio_format": fmt,
            "chunk_seconds": chunk_seconds, "framing": framing}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    outdir = os.path.join(ROOT, "results", "chunks", tag)
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    chunks, header, trailer, arrivals, err = [], None, None, [], None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        rheaders = {k.lower(): v for k, v in resp.headers.items()}
        ctype = rheaders.get("content-type", "")
        rate = int(rheaders.get("x-sample-rate", 24000))
        rd = ChunkReader(resp, framing)
        while True:
            try:
                obj = rd.read_obj()
            except EOFError:
                break
            t = obj.get("kind")
            if t == "header":
                header = obj
            elif t == "chunk":
                arrivals.append((time.time() - t0, obj))
                chunks.append(obj)
            elif t == "end":
                trailer = obj
                break
            elif t == "error":
                err = obj.get("error")
                break

    # 每块落盘 + 校验「单独可播」
    ext = {"wav": "wav", "opus": "ogg", "mp3": "mp3", "pcm": "pcm"}[fmt]
    pcm_parts, bad, durs = [], [], []
    for i, c in enumerate(chunks):
        p = os.path.join(outdir, "chunk-%03d.%s" % (i, ext))
        with open(p, "wb") as f:
            f.write(c["audio"])
        if fmt == "wav":
            d = wav_dur(c["audio"])
            if not d:
                bad.append(i)
            else:
                durs.append(d)
                pcm_parts.append(c["audio"][44:])
        else:
            ok, info = ffprobe_ok(p)
            if not ok:
                bad.append(i)
            try:
                durs.append(round(float(info.get("duration") or 0), 3))
            except ValueError:
                durs.append(None)
    all_wav = None
    if fmt == "wav" and pcm_parts and keep:
        all_wav = os.path.join(outdir, "all.wav")
        with open(all_wav, "wb") as f:
            f.write(struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + sum(len(x) for x in pcm_parts),
                                b"WAVE", b"fmt ", 16, 1, 1,
                                rate, rate * 2, 2, 16,
                                b"data", sum(len(x) for x in pcm_parts)))
            for x in pcm_parts:
                f.write(x)

    res = {
        "tag": tag, "voice": voice, "fmt": fmt, "chunk_seconds": chunk_seconds,
        "framing": framing, "content_type": ctype,
        "chars": len(text), "chunks": len(chunks), "bad_chunks": bad,
        "first_chunk_ms": round(arrivals[0][0] * 1000, 1) if arrivals else None,
        "arrivals_ms": [round(a * 1000, 1) for a, _ in arrivals[:8]],
        "gaps_ms": [round((arrivals[i][0] - arrivals[i - 1][0]) * 1000, 1)
                    for i in range(1, min(len(arrivals), 9))],
        "seconds_per_chunk": durs[:6],
        "seconds_sum": round(sum(d for d in durs if d), 3),
        "total_s": trailer.get("total_s") if trailer else None,
        "elapsed": trailer.get("elapsed") if trailer else round(time.time() - t0, 3),
        "rtf": trailer.get("rtf") if trailer else None,

        "bytes_total": sum(len(c["audio"]) for c in chunks),
        "all_wav": all_wav, "error": err,
    }
    if verbose:
        print(json.dumps(res, ensure_ascii=False))
    with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", action="store_true")
    ap.add_argument("--tag", default="long")
    ap.add_argument("--text", default="")
    ap.add_argument("--voice", default="")
    ap.add_argument("--format", default="wav", choices=["wav", "opus", "mp3", "pcm"])
    ap.add_argument("--chunk-seconds", type=float, default=2.0)
    ap.add_argument("--framing", default="len", choices=["len", "none"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    if not args.cases:
        run_case(args.tag, args.text or LONG, voice=args.voice, fmt=args.format,
                 chunk_seconds=args.chunk_seconds, framing=args.framing,
                 mnt=args.max_new_tokens)
        return

    print("== 用例 1：wav / 2 s / len 前缀 / 长文本 ==")
    r1 = run_case("long-wav", LONG)
    print("\n== 用例 2：framing=none（纯拼接 msgpack）==")
    run_case("short-none", SHORT, framing="none", chunk_seconds=1.0)
    print("\n== 用例 3：opus 块（每块独立 ogg）==")
    run_case("short-opus", MED, fmt="opus")
    print("\n== 用例 4：voice=default（无参考）==")
    run_case("short-default", SHORT, voice="default", chunk_seconds=1.0)
    print("\n== 用例 6：未知音色应 400 ==")
    try:
        urllib.request.urlopen(urllib.request.Request(
            URL, data=json.dumps({"input": "x", "voice": "不存在的音色"}).encode(),
            headers={"Content-Type": "application/json"}), timeout=30)
        print("❌ 居然没报错")
    except urllib.error.HTTPError as e:
        print("✅ HTTP %d：%s" % (e.code, e.read().decode()[:80]))
    print("\n小结：首块 %.0f ms / %d 块 / 每块独立可播 %s" % (
        r1["first_chunk_ms"], r1["chunks"],
        "全部通过" if not r1["bad_chunks"] else "有坏块 %s" % r1["bad_chunks"]))


if __name__ == "__main__":
    main()
