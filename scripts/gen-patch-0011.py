#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0011-abort-flag-reset.patch：
把 `g_moss_stream_aborted` 的复位从"只有流式请求"改成"每个请求都复位"。

背景（实测踩出来的）：
  patch 0005 在 AR 循环里无条件检查
      if (g_moss_stream_aborted) break;
  但复位写在
      if (streaming) { ... g_moss_stream_aborted = false; ... }
  里。于是一个被客户端中断的**流式**请求会把标志留在 true，
  下一个**非流式**（整段解码）请求在第 0 步就 break → 生成 0 帧 →
  前端报 `常驻 daemon 没有返回音频`（500），而且会一直错下去。

做法：把复位挪到每次请求开始、无条件执行。

用法： python3 scripts/gen-patch-0011.py
"""
import difflib
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = os.path.join(ROOT, "upstream", "llama.cpp")
PATCH_DIR = os.path.join(ROOT, "patches")
OUT = os.path.join(PATCH_DIR, "0011-abort-flag-reset.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1 次）：\n%s" % (tag, n, anchor[:400]))
    return text.replace(anchor, replacement, 1)


A_ANCHOR = """        if (streaming) {
            g_moss_stream_fd          = cfd;
            g_moss_stream_emitted     = 0;
            g_moss_stream_header_sent = false;
            g_moss_stream_aborted     = false;   // patch 0005
        }
"""
A_NEW = """        // patch 0011：标志必须每个请求都复位 —— AR 循环是无条件检查它的，
        // 否则一个被中断的流式请求会把后续（整段）请求全部毒死（0 帧 → 500）。
        g_moss_stream_aborted = false;
        if (streaming) {
            g_moss_stream_fd          = cfd;
            g_moss_stream_emitted     = 0;
            g_moss_stream_header_sent = false;
        }
"""

EDITS = [(F_TOOL, [(A_ANCHOR, A_NEW, "abort/reset-every-request")])]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)
    tmp = tempfile.mkdtemp(prefix="moss-patch11-")
    base = os.path.join(tmp, "src")
    shutil.copytree(UP, base)
    mine = os.path.basename(OUT)
    for p in sorted(x for x in os.listdir(PATCH_DIR) if x.endswith(".patch") and x < mine):
        r = apply_patch(base, os.path.join(PATCH_DIR, p))
        if r.returncode != 0:
            die("基线补丁 %s 应用失败：%s%s" % (p, r.stdout, r.stderr))
        print("基线: 已应用 %s" % p)

    changed = {}
    for rel, edits in EDITS:
        path = os.path.join(base, rel)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        for anchor, repl, tag in edits:
            text = sub_once(text, anchor, repl, tag)
            print("编辑: %-24s %s" % (tag, rel))
        changed[rel] = text

    hunks = []
    for rel, new_text in changed.items():
        with open(os.path.join(base, rel), "r", encoding="utf-8") as fh:
            old_lines = fh.readlines()
        hunks.extend(difflib.unified_diff(old_lines, new_text.splitlines(keepends=True),
                                          fromfile="a/" + rel, tofile="b/" + rel, n=3))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.writelines(hunks)
    print("\n写出 %s（%d 行）" % (OUT, len(hunks)))

    chk = tempfile.mkdtemp(prefix="moss-check11-")
    chk_src = os.path.join(chk, "src")
    shutil.copytree(UP, chk_src)
    for p in sorted(x for x in os.listdir(PATCH_DIR) if x.endswith(".patch")):
        r = apply_patch(chk_src, os.path.join(PATCH_DIR, p))
        if r.returncode != 0:
            die("自检失败：%s 不干净：%s%s" % (p, r.stdout, r.stderr))
        print("自检: %s 干净" % p)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(chk, ignore_errors=True)
    print("OK")


if __name__ == "__main__":
    main()
