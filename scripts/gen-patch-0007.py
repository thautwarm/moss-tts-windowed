#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0007-ignore-sigpipe.patch：
daemon 忽略 SIGPIPE —— 客户端/下游管道一断，写 socket 不能把整个进程打死。

背景（实测踩出来的）：
  流式请求的下游（ffmpeg/lame 管道）一旦提前退出，前端会关掉 daemon socket。
  daemon 此时正往 socket 写 PCM，`write()` 触发 SIGPIPE，**默认动作是终止进程** ——
  patch 0005 里对 write 返回值的检查根本没机会跑。日志里就是
      [entrypoint] daemon 退出（rc=141）       # 141 = 128 + 13(SIGPIPE)
  然后守护循环重启、模型重新加载，正在服务的其它请求全断。

做法：
  serve 入口里 `signal(SIGPIPE, SIG_IGN)`，让 write 返回 -1/EPIPE。
  patch 0005 已经把这个返回值当成"客户端走了"，会置 abort 标志、提前收手。

用法： python3 scripts/gen-patch-0007.py
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
OUT = os.path.join(PATCH_DIR, "0007-ignore-sigpipe.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1 次）：\n%s" % (tag, n, anchor[:400]))
    return text.replace(anchor, replacement, 1)


# 1) include <csignal>
A_ANCHOR = """#include <cstring>
"""
A_NEW = """#include <cstring>
#include <csignal>
"""

# 2) serve 入口：忽略 SIGPIPE（放在 socket 建好之前，整个服务期都生效）
B_ANCHOR = """static int moss_serve_unix(
        const std::string & sock_path,
        const std::vector<std::string> & base_args,
        const std::string & argv0) {
    ::unlink(sock_path.c_str());
"""
B_NEW = """static int moss_serve_unix(
        const std::string & sock_path,
        const std::vector<std::string> & base_args,
        const std::string & argv0) {
    // patch 0007：客户端/下游管道一断，写 socket 会触发 SIGPIPE（默认=杀进程）。
    // 忽略它，让 write 返回 EPIPE —— patch 0005 会把这种情况当成"客户端走了"。
    ::signal(SIGPIPE, SIG_IGN);

    ::unlink(sock_path.c_str());
"""

EDITS = [(F_TOOL, [
    (A_ANCHOR, A_NEW, "sigpipe/include"),
    (B_ANCHOR, B_NEW, "sigpipe/serve-entry"),
])]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)
    tmp = tempfile.mkdtemp(prefix="moss-patch7-")
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

    chk = tempfile.mkdtemp(prefix="moss-check7-")
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
