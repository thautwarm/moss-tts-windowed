#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0006-prefill-ubatch.patch：
backbone 的 prefill 按固定 ubatch 分块，长 prompt 不再 OOM。

背景（实测踩出来的）：
  384 字（508 tokens）请求直接 500，日志是
      sched_reserve: worst-case: n_tokens = 508
      allocating 1099.53 MiB on device 0: cudaMalloc failed: out of memory
      failed to allocate compute pp buffers
  根因是 backbone ctx 把 n_ubatch 设成了 prompt 全长（cparams.n_ubatch = n_batch = prompt_frames），
  而 pp compute buffer 的大小 ∝ n_ubatch（约 2.2 MiB/token）。8 GB 卡上 ~508 token 就吃掉 1.1 GB，
  扣掉常驻的 6.5 GB 直接分配失败。KV 只占 396 MiB，不是 KV 的锅。

做法：
  按 n_ctx 估算 KV 占用，把剩下的显存预算（默认 1400 MiB）折成 ubatch 上限，再与
  配置上限（默认 384）取小。n_batch 不动 —— llama.cpp 的 llama_decode 本来就会把一个
  batch 自动切成 ≤n_ubatch 的若干 ubatch 顺序执行，KV 累积与因果注意力语义完全不变。

  为什么要"算"而不是固定一个数：
    - 固定 128：常规请求（278 tokens）也会被切，输出与旧版差 1 帧（末尾 EOS 时抖动），
      md5 基线就保不住了。
    - 固定 384：短文本仍是单批（md5 不变），但 4000 字 + max_new_tokens=2048 时
      KV 更大，pp 845 MiB + KV 670 MiB 直接把 8 GB 撑爆（实测 500）。
    - 动态：prompt ≤ 上限时**单批**（逐字节一致），超了才切，且切多大取决于 KV 占了多少。

  注意：**不能**动 moss_init_audio_context —— 音频 encoder/decoder 是 causal_attn=false，
  那种非因果整段注意力切 ubatch 会截断彼此可见性，结果会算错。只有 backbone 是因果的。

用法： python3 scripts/gen-patch-0006.py
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
OUT = os.path.join(PATCH_DIR, "0006-prefill-ubatch.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1 次）：\n%s" % (tag, n, anchor[:400]))
    return text.replace(anchor, replacement, 1)


# 1) 新增全局上限 + 截断函数（跟在 0004 的全局量后面）
A_ANCHOR = """static long long g_moss_serve_max_new_tokens = 448;  // 非流式整段解码上限（0 = 不限）
"""
A_NEW = A_ANCHOR + """static uint32_t g_moss_prefill_ubatch       = 384;  // patch 0006：prefill 每批 token 上限（0 = 不分块）
static uint32_t g_moss_prefill_budget_mib  = 1000;  // patch 0006：pp buffer 显存预算（MiB）

// patch 0006：按 n_ctx 估算 KV，剩下多少显存就切多大一批 prefill。
// 常数是这台机器实测出来的（36 层 f16 KV、Q4_K_M 骨干、8 GB 显卡）：
//   KV    ≈ 0.1406 MiB / cell（实测 396 MiB / 2816 cells）
//   pp    ≈ 2.2 MiB / token（实测 1099.53 MiB / 508 tokens）
//   预算  ≈ 1400 MiB（总 8151 - 常驻 6531 = 1620，留 ~220 余量）
// prompt ≤ ubatch 时仍是**单批**，与打补丁前逐字节一致（保住 md5 基线）。
static uint32_t moss_auto_ubatch(uint32_t n_batch, uint32_t n_ctx) {
    if (g_moss_prefill_ubatch == 0) {
        return n_batch;
    }
    uint32_t cap = g_moss_prefill_ubatch;

    const double kv_mib = (double) n_ctx * 0.1406;
    const double room   = (double) g_moss_prefill_budget_mib - kv_mib;
    const uint32_t by_mem = room > 0.0 ? (uint32_t) (room / 2.2) : 0u;
    if (by_mem < cap) {
        cap = by_mem;
    }
    if (cap < 64u) {
        cap = 64u;   // 再小也没意义：llama.cpp 单批至少要能跑起来
    }
    return std::max<uint32_t>(std::min<uint32_t>(n_batch, cap), 1u);
}
"""

# 2) backbone ctx：n_ubatch 不再等于 prompt 全长
B_ANCHOR = """        cparams.n_ctx = std::max<uint32_t>((uint32_t) prompt_frames + (uint32_t) max_new_tokens + 8u, 64u);
        cparams.n_batch = std::max<uint32_t>((uint32_t) prompt_frames, 1u);
        cparams.n_ubatch = cparams.n_batch;
"""
B_NEW = """        cparams.n_ctx = std::max<uint32_t>((uint32_t) prompt_frames + (uint32_t) max_new_tokens + 8u, 64u);
        cparams.n_batch = std::max<uint32_t>((uint32_t) prompt_frames, 1u);
        // patch 0006：pp compute buffer 的大小 ∝ n_ubatch（实测 508 token ≈ 1099 MiB）。
        // 一次性 prefill 整个 prompt 会在 8 GB 卡上 cudaMalloc 失败，所以按显存预算分块。
        cparams.n_ubatch = moss_auto_ubatch(cparams.n_batch, cparams.n_ctx);
"""

# 3) serve 模式参数透传：--serve-ubatch / --serve-ubatch-budget
C_ANCHOR = """        if (arg == "--serve-stream-overlap" && i + 1 < argc) {
            g_moss_stream_overlap = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""
C_NEW = C_ANCHOR + """        if (arg == "--serve-ubatch" && i + 1 < argc) {
            // patch 0006：prefill 单批 token 上限（0 = 不分块，退回旧行为）
            g_moss_prefill_ubatch = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--serve-ubatch-budget" && i + 1 < argc) {
            // patch 0006：pp buffer 可用的显存预算（MiB）
            g_moss_prefill_budget_mib = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""

EDITS = [(F_TOOL, [
    (A_ANCHOR, A_NEW, "ubatch/global"),
    (B_ANCHOR, B_NEW, "ubatch/backbone-ctx"),
    (C_ANCHOR, C_NEW, "ubatch/serve-arg"),
])]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)
    tmp = tempfile.mkdtemp(prefix="moss-patch6-")
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

    chk = tempfile.mkdtemp(prefix="moss-check6-")
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
