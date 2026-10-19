#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0012-fast-abort.patch：
生成循环里**每帧**检查客户端 socket 是否已断开，而不是等到下一次写失败。

背景：
  patch 0005 只在"往 socket 写 PCM 失败"时置 abort 标志。但写是按窗口 W 帧触发的，
  所以客户端断开后，最多还要白算 W 帧（W=32 时 2.56 s 音频 ≈ 0.5 s 墙钟）+ 一次解码。
  更糟的是：如果客户端断开时正好在 pump 解码窗口，还会把整个窗口算完。

做法：
  每步用 `poll(fd, 0)` 查 POLLHUP/POLLERR/POLLNVAL/POLLRDHUP（以及 POLLIN —— 我们的协议
  是"客户端发完请求就不再发"，所以可读 = EOF），命中就置 `g_moss_stream_aborted`。

用法： python3 scripts/gen-patch-0012.py
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
OUT = os.path.join(PATCH_DIR, "0012-fast-abort.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1 次）：\n%s" % (tag, n, anchor[:400]))
    return text.replace(anchor, replacement, 1)


# 1) include <poll.h>
A_ANCHOR = """#include <csignal>
"""
A_NEW = """#include <csignal>
#include <poll.h>
"""

# 2) 工具函数：客户端是否已走
B_ANCHOR = """static uint32_t g_moss_stream_lookahead      = 0;    // patch 0008：右侧前瞻帧数（0 = 有多少用多少，零等待）
"""
B_NEW = B_ANCHOR + """
// patch 0012：逐帧探测客户端是否已断开（poll 不读数据，只是查 socket 事件）
static bool moss_stream_client_gone() {
    if (g_moss_stream_fd < 0) {
        return false;
    }
    struct pollfd pfd;
    pfd.fd      = g_moss_stream_fd;
    pfd.events  = POLLIN;
    pfd.revents = 0;
    if (::poll(&pfd, 1, 0) <= 0) {
        return false;
    }
#ifdef POLLRDHUP
    if (pfd.revents & POLLRDHUP) {
        return true;
    }
#endif
    if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) {
        return true;
    }
    // 协议是"请求发完就不再发"，所以对端可读 = 已经 EOF
    return (pfd.revents & POLLIN) != 0;
}
"""

# 3) AR 循环里每帧检查
C_ANCHOR = """            if (g_moss_stream_aborted) {
                LOG("stream: 客户端已断开，提前结束生成（已跑 %d 步）\\n", step + 1);
                break;
            }
"""
C_NEW = """            if (!g_moss_stream_aborted && moss_stream_client_gone()) {   // patch 0012
                g_moss_stream_aborted = true;
            }
            if (g_moss_stream_aborted) {
                LOG("stream: 客户端已断开，提前结束生成（已跑 %d 步）\\n", step + 1);
                break;
            }
"""

EDITS = [(F_TOOL, [
    (A_ANCHOR, A_NEW, "abort/include-poll"),
    (B_ANCHOR, B_NEW, "abort/client-gone"),
    (C_ANCHOR, C_NEW, "abort/per-step"),
])]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)
    tmp = tempfile.mkdtemp(prefix="moss-patch12-")
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

    chk = tempfile.mkdtemp(prefix="moss-check12-")
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
