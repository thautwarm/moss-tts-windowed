#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0008-stream-lookahead.patch：
流式出音的"前瞻"（右侧上下文）从固定 O 帧改成可配，**默认 0 = 有多少用多少**。

背景：
  原实现 `moss_stream_pump` 里有一句
      if (now.raw_frames <= O) return;            // O = 左侧上下文帧数
      const size_t stable_end = now.raw_frames - O;
  它的意思是"后面还得有 O 帧，才敢把当前这 W 帧吐出去"。
  于是 O 同时承担了两件事：
    - 左侧上下文（质量）
    - **右侧前瞻 = 直接变成出音延迟**（O=64 → 先攒 5.1 s；O=120 → 9.6 s）

  实测 O=96/120 还会直接 OOM：decoder ctx 大小是 `W + 3*O + 16` 帧（O=120 → 384 帧），
  非因果 decode 的 pp compute buffer 要 831 MiB，常驻 6.5 GB 之外放不下。

做法：
  拆成两个量 —— 左侧上下文仍是 O（`--stream-overlap`），右侧前瞻新增
  `g_moss_stream_lookahead`（默认 0）：
    - lookahead = 0：生成到 W 帧就立刻吐，右侧上下文用 `min(可用, O)` → **零等待**；
    - lookahead = N：只等 N 帧（0.08 s/帧），比原来"等 O 帧"小得多。
  `moss_stream_emit_range` 本来就把 win_end 夹在 `raw_frames` 以内，所以"有多少用多少"天然成立。
  decoder ctx 相应改成 `W + 2*O + 16`（窗口最大就是 W+2O；原来按 3*O 多留了 O 帧）。

用法： python3 scripts/gen-patch-0008.py
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
OUT = os.path.join(PATCH_DIR, "0008-stream-lookahead.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1 次）：\n%s" % (tag, n, anchor[:400]))
    return text.replace(anchor, replacement, 1)


# 1) 新增 lookahead 全局量
A_ANCHOR = """static uint32_t g_moss_stream_overlap        = 12;   // 每侧上下文帧数（≈0.96 s）
"""
A_NEW = """static uint32_t g_moss_stream_overlap        = 12;   // 左侧上下文帧数（≈0.96 s）
static uint32_t g_moss_stream_lookahead      = 0;    // patch 0008：右侧前瞻帧数（0 = 有多少用多少，零等待）
"""

# 2) pump：等待量从 O 改成 lookahead
B_ANCHOR = """    const size_t   O = g_moss_stream_overlap;
    const uint32_t W = g_moss_stream_window;
    moss_stream_decoder_init(decoder_path, n_gpu_layers, cfg, (size_t) W + 3 * O + 16);
    moss_stream_send_header();

    const moss_generation_audio now = moss_decode_generation_audio(state, prompt_frames, cfg);
    // 右侧留 O 帧：只有「后面还有 O 帧」的部分才是真正定稿的，这样每帧吐出去时左右都有上下文
    if (now.raw_frames <= O) {
        return;
    }
    const size_t stable_end = now.raw_frames - O;
"""
B_NEW = """    const size_t   O = g_moss_stream_overlap;
    const size_t   L = g_moss_stream_lookahead;
    const uint32_t W = g_moss_stream_window;
    // patch 0008：emit_range 里 O 用在**左右两侧**，窗口最大 W + 2*O；ctx 必须装得下它。
    moss_stream_decoder_init(decoder_path, n_gpu_layers, cfg, (size_t) W + 2 * O + 16);
    moss_stream_send_header();

    const moss_generation_audio now = moss_decode_generation_audio(state, prompt_frames, cfg);
    // patch 0008：只等 lookahead 帧（默认 0 = 不等）。
    // 右侧上下文由 emit_range 夹在 raw_frames 以内 —— 有多少用多少。
    if (now.raw_frames <= L) {
        return;
    }
    const size_t stable_end = now.raw_frames - L;
"""

# 3) flush 里的 decoder ctx 同样收窄（flush 的 keep_right=false，右侧本来就 0）
C_ANCHOR = """    const size_t   O = g_moss_stream_overlap;
    const uint32_t W = g_moss_stream_window;
    moss_stream_decoder_init(decoder_path, n_gpu_layers, cfg, (size_t) W + 3 * O + 16);
    moss_stream_send_header();
"""
C_NEW = """    const size_t   O = g_moss_stream_overlap;
    const size_t   L = g_moss_stream_lookahead;
    const uint32_t W = g_moss_stream_window;
    moss_stream_decoder_init(decoder_path, n_gpu_layers, cfg, (size_t) W + 2 * O + 16);
    moss_stream_send_header();
"""

# 4) serve 参数：--serve-stream-lookahead
D_ANCHOR = """        if (arg == "--serve-stream-overlap" && i + 1 < argc) {
            g_moss_stream_overlap = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""
D_NEW = D_ANCHOR + """        if (arg == "--serve-stream-lookahead" && i + 1 < argc) {
            // patch 0008：右侧前瞻帧数（0 = 有多少用多少）
            g_moss_stream_lookahead = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""

# 5) CLI 参数：--stream-lookahead（前端按请求透传）
E_ANCHOR = """        if (arg == "--stream-overlap" && i + 1 < argc) {
            // patch 0004：流式窗口每侧的上下文帧数
            g_moss_stream_overlap = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""
E_NEW = E_ANCHOR + """        if (arg == "--stream-lookahead" && i + 1 < argc) {
            // patch 0008：右侧前瞻帧数（0 = 有多少用多少）
            g_moss_stream_lookahead = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""

EDITS = [(F_TOOL, [
    (A_ANCHOR, A_NEW, "lookahead/global"),
    (B_ANCHOR, B_NEW, "lookahead/pump"),
    (C_ANCHOR, C_NEW, "lookahead/flush-ctx"),
    (D_ANCHOR, D_NEW, "lookahead/serve-arg"),
    (E_ANCHOR, E_NEW, "lookahead/cli-arg"),
])]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)
    tmp = tempfile.mkdtemp(prefix="moss-patch8-")
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

    chk = tempfile.mkdtemp(prefix="moss-check8-")
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
