#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""契约自测：把前端每个端点都打一遍，跟原服务的语义对照。

用法： python3 scripts/contract-test.py [--url http://127.0.0.1:6677]
"""
import argparse
import io
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("%s %-52s %s" % ("✅" if ok else "❌", name, detail))


def req(url, method="GET", data=None, headers=None, raw=None, timeout=600):
    r = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            # uvicorn 会把头名规范化成小写，这里统一小写比较（HTTP 头本来就大小写不敏感）
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def wav_dur(data):
    if data[:4] != b"RIFF":
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


def jpost(url, obj, timeout=600):
    return req(url, "POST", json.dumps(obj).encode(), {"Content-Type": "application/json"}, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:6677")
    ap.add_argument("--voice", default="", help="默认取注册表里第一个可用音色")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    if not args.voice:
        try:
            st, _, body = req(base + "/api/v1/voices")
            vs = json.loads(body) if st == 200 else []
            ids = [v["id"] for v in vs if v.get("codes") and v["id"] != "default"]
            args.voice = ids[0] if ids else "default"
        except Exception:
            args.voice = "default"
        print("使用音色:", args.voice)
    text = "契约自测：你好，世界。这会儿方便说两句话吗？"

    # 1) health
    st, _, body = req(base + "/health")
    h = json.loads(body)
    check("GET /health 200", st == 200, h.get("backend", ""))
    check("health.daemon.online", h.get("daemon", {}).get("online") is True, str(h.get("daemon")))

    # 2) voices / models
    st, _, body = req(base + "/api/v1/voices")
    voices = json.loads(body)
    check("GET /api/v1/voices 200 + 列表", st == 200 and isinstance(voices, list), "%d 个" % len(voices))
    check("voices 含 default", any(v["id"] == "default" for v in voices))
    st, _, body = req(base + "/v1/models")
    check("GET /v1/models OpenAI 形状", st == 200 and json.loads(body).get("object") == "list")

    # 3) POST /api/v1/synthesize（wav）
    t0 = time.time()
    st, hdr, body = jpost(base + "/api/v1/synthesize",
                          {"text": text, "voice": args.voice, "language": "zh",
                           "max_new_tokens": 256, "seed": 42, "temperature": 1.7,
                           "top_p": 0.8, "top_k": 25, "response_format": "wav"})
    dt = time.time() - t0
    check("/api/v1/synthesize wav 200", st == 200 and body[:4] == b"RIFF",
          "%.2fs %.2fs 音频" % (dt, wav_dur(body) or 0))
    for k in ("X-Elapsed", "X-Audio-Duration", "X-RTF", "X-Format"):
        check("响应头 %s" % k, k.lower() in hdr, str(hdr.get(k.lower())))
    check("Content-Type audio/wav", hdr.get("content-type", "").startswith("audio/wav"), hdr.get("content-type", ""))

    # 4) OpenAI 非流式
    st, hdr, body = jpost(base + "/v1/audio/speech",
                          {"input": text, "voice": args.voice, "language": "zh",
                           "max_new_tokens": 256, "seed": 42, "response_format": "wav"})
    if st == 400 and "只提供流式".encode() in body:
        check("/v1/audio/speech 非流式", True, "已按配置禁用（400）")
    else:
        check("/v1/audio/speech 非流式 200", st == 200 and body[:4] == b"RIFF", "X-Voice=%s" % hdr.get("x-voice"))

    # 5) OpenAI 流式（pcm）：量首块延迟
    t0 = time.time()
    r = urllib.request.Request(base + "/v1/audio/speech",
                               data=json.dumps({"input": text, "voice": args.voice, "language": "zh",
                                                "max_new_tokens": 256, "seed": 42,
                                                "response_format": "pcm", "stream": True}).encode(),
                               headers={"Content-Type": "application/json"})
    first = None
    n = 0
    with urllib.request.urlopen(r, timeout=600) as resp:
        st = resp.status
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            if first is None:
                first = time.time() - t0
            n += len(chunk)
    check("/v1/audio/speech stream=true (pcm)", st == 200 and n > 0,
          "首块 %.0f ms，共 %.1f KB" % ((first or 0) * 1000, n / 1024))
    check("流式首块 < 1.5 s", (first or 9) < 1.5, "%.0f ms" % ((first or 0) * 1000))

    # 6) wav 流式（带 44 字节头）
    r = urllib.request.Request(base + "/v1/audio/speech",
                               data=json.dumps({"input": text, "voice": args.voice,
                                                "max_new_tokens": 256, "seed": 42,
                                                "response_format": "wav", "stream": True}).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        head = resp.read(44)
        rest = resp.read()
    check("stream=true (wav) 带头 + PCM", head[:4] == b"RIFF" and len(rest) > 1000,
          "%d B 头 + %d B PCM" % (len(head), len(rest)))

    # 7) 转码
    for fmt, magic in (("mp3", b"ID3"), ("opus", b"OggS"), ("pcm", None)):
        st, hdr, body = jpost(base + "/api/v1/synthesize",
                              {"text": "转码测试。", "voice": args.voice, "max_new_tokens": 96,
                               "seed": 7, "response_format": fmt})
        ok = st == 200 and len(body) > 500
        if magic:
            ok = ok and (body[:len(magic)] == magic or fmt == "mp3")
        check("response_format=%s" % fmt, ok, "%d B %s" % (len(body), hdr.get("content-type", "")))

    # 8) 错误路径（拒绝集必须是参考实现的子集）
    st, _, body = jpost(base + "/api/v1/synthesize", {"text": "x", "voice": "不存在的音色"})
    check("未知音色 -> 400", st == 400, body.decode()[:60])
    st, _, body = jpost(base + "/v1/audio/speech", {"input": "x", "voice": args.voice, "response_format": "aac"})
    check("aac -> 400（未实现）", st == 400, body.decode()[:50])
    st, _, body = jpost(base + "/v1/audio/speech", {"input": "x", "voice": args.voice, "token_count": 100})
    check("token_count -> 400（不支持时长控制）", st == 400, body.decode()[:50])
    st, _, body = jpost(base + "/v1/audio/speech", {"input": "x", "voice": args.voice, "ref_audio": "http://x"})
    check("ref_audio -> 400（不支持按请求传参考）", st == 400, body.decode()[:40])

    # 9) 音色注册 / 删除（含 CPU 预算 codes）
    vid = "ctest%s" % uuid.uuid4().hex[:6]
    src = os.path.join(os.path.dirname(__file__), "..", "voices", args.voice + ".wav")
    with open(src, "rb") as fh:
        payload = fh.read()
    boundary = "----ct"
    body = (("--%s\r\nContent-Disposition: form-data; name=\"id\"\r\n\r\n%s\r\n" % (boundary, vid)) +
            ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"r.wav\"\r\n"
             "Content-Type: audio/wav\r\n\r\n" % boundary)).encode() + payload + \
           ("\r\n--%s--\r\n" % boundary).encode()
    st, _, resp = req(base + "/api/v1/voices", "POST", body,
                      {"Content-Type": "multipart/form-data; boundary=" + boundary}, timeout=900)
    j = json.loads(resp) if st == 200 else {}
    check("POST /api/v1/voices 注册（附带 codes 预算）", st == 200 and j.get("codes") is True,
          "id=%s codes=%s" % (vid, j.get("codes")))
    st, _, body = jpost(base + "/api/v1/synthesize",
                        {"text": "新音色测试。", "voice": vid, "max_new_tokens": 96, "seed": 3})
    check("新注册音色可合成", st == 200 and body[:4] == b"RIFF", "%.2fs" % (wav_dur(body) or 0))
    st, _, _ = req(base + "/api/v1/voices/" + vid, "DELETE")
    check("DELETE /api/v1/voices/{id}", st == 200)

    print("\n== %d 通过 / %d 失败 ==" % (len(PASS), len(FAIL)))
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
