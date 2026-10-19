#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0015-text-b64.patch：文本改成 base64 传输。

背景（踩了两次的坑）：
  前端和 daemon 之间的协议是**按行的**（读到 \\n 算一条命令，字段用 \\t 分列）。
  文本里只要出现换行，请求就被截断：
    · 前半段少了 --reference-codes → 静默用默认音色（听感上"音色不对/男声"）；
    · 后半段被当成新请求 → "unknown argument" + 打一堆 usage。
  制表符同理（会把字段切错位）。

做法：
  新增 `--text-b64 <base64>`，daemon 解回原文。base64 里只有 A-Za-z0-9+/=，
  既不含 \\n 也不含 \\t，协议重新变得安全，而**文本可以随便含换行/制表符**。
  原来的 `--text` 保留（CODES 那条内部路径还在用）。

用法： python3 scripts/gen-patch-0015.py
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
OUT = os.path.join(PATCH_DIR, "0015-text-b64.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def once(text, anchor, repl, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1）：\n%s" % (tag, n, anchor[:400]))
    return text.replace(anchor, repl, 1)


# 1) base64 解码器（放在 CLI 参数解析附近）
A_ANCHOR = """static int moss_cli_main(int argc, char ** argv) {
"""
A_NEW = """// patch 0015：请求文本用 base64 传，避免换行/制表符截断"按行"的内部协议
static std::string moss_b64_decode(const std::string & in) {
    static const std::string tbl =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static const std::vector<int> map = [&] {
        std::vector<int> m(256, -1);
        for (size_t i = 0; i < tbl.size(); ++i) {
            m[(unsigned char) tbl[i]] = (int) i;
        }
        return m;
    }();

    std::string out;
    int val = 0;
    int bits = -8;
    for (const unsigned char c : in) {
        if (c == '=') {
            break;
        }
        const int d = map[c];
        if (d < 0) {
            continue;               // 跳过空白（base64 里本来不会有换行）
        }
        val = (val << 6) + d;
        bits += 6;
        if (bits >= 0) {
            out.push_back((char) ((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

static int moss_cli_main(int argc, char ** argv) {
"""

# 2) 参数
B_ANCHOR = """        if (arg == "--text" && i + 1 < argc) {
            text = argv[++i];
            continue;
        }
"""
B_NEW = B_ANCHOR + """        if (arg == "--text-b64" && i + 1 < argc) {
            // patch 0015：base64 文本（可含换行/制表符而不会破坏行协议）
            text = moss_b64_decode(argv[++i]);
            continue;
        }
"""

EDITS = [(F_TOOL, [
    (A_ANCHOR, A_NEW, "b64/decoder"),
    (B_ANCHOR, B_NEW, "b64/cli-arg"),
])]


def ap(root, p):
    with open(p, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    tmp = tempfile.mkdtemp(prefix="p15-")
    base = os.path.join(tmp, "src")
    shutil.copytree(UP, base)
    mine = os.path.basename(OUT)
    for p in sorted(x for x in os.listdir(PATCH_DIR) if x.endswith(".patch") and x < mine):
        if ap(base, os.path.join(PATCH_DIR, p)).returncode != 0:
            die("基线失败 " + p)
    rel, ed = EDITS[0]
    path = os.path.join(base, rel)
    text = open(path, encoding="utf-8").read()
    for a, r, tag in ed:
        text = once(text, a, r, tag)
        print("编辑:", tag)
    old = open(path, encoding="utf-8").readlines()
    open(OUT, "w", encoding="utf-8").writelines(
        difflib.unified_diff(old, text.splitlines(keepends=True),
                             fromfile="a/" + rel, tofile="b/" + rel, n=3))
    print("写出", OUT)

    chk = tempfile.mkdtemp(prefix="c15-")
    src = os.path.join(chk, "src")
    shutil.copytree(UP, src)
    for p in sorted(x for x in os.listdir(PATCH_DIR) if x.endswith(".patch")):
        if ap(src, os.path.join(PATCH_DIR, p)).returncode != 0:
            die("自检失败 " + p)
    print("OK")


main()
