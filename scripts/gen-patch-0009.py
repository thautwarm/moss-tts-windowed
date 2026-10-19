#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0009-backbone-sliding.patch：
backbone 的 context 满了就滑窗 —— 钉住 prompt 前缀（instruction + 参考 codes + 全文文本），
只丢最老的**音频行**，并把剩余行的位置平移回去。

背景：
  backbone 的 n_ctx_train 是 40960，理论上全文都能喂进去；真正的限制是显存：
  KV ≈ 0.1406 MiB/行，可用预算约 1.4 GB → 约 10k 行。
  4000 字文本：prompt ≈ 3k 行 + 音频 ≈ 12.6k 行 → 16k 行，超预算。
  现在按句切段是"文本级切分"，会在句间丢上下文；这里改成"只滑音频行"，文本前缀永不淘汰。

做法：
  · 行数上限 `--serve-ctx-rows`（默认 9500），保留最近 `--serve-ctx-keep`（默认 2048）音频行；
  · 位置 `pos` 独立跟踪，不再用 `prompt_frames + step`；
  · 快满时：
      seq_rm([pinned, cut))                    删掉最老的音频行
      seq_add(cut, pos, -(cut - pinned))       把尾巴位置平移回去
      pos -= (cut - pinned)
  · 切点 `cut` 在目标点 ±16 帧内选**能量最低**的帧（能量由流式解码逐帧记录；
    丢掉的是"模型的历史"，不会产生播放接缝，但切在低能量处更不容易引起语调抖动）。

用法： python3 scripts/gen-patch-0009.py
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
OUT = os.path.join(PATCH_DIR, "0009-backbone-sliding.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1 次）：\n%s" % (tag, n, anchor[:500]))
    return text.replace(anchor, replacement, 1)


# 1) 全局量：ctx 行数上限 / 保留最近多少行 / 逐帧能量
A_ANCHOR = """static uint32_t g_moss_prefill_budget_mib  = 1000;  // patch 0006：pp buffer 显存预算（MiB）
"""
A_NEW = A_ANCHOR + """
// patch 0009：backbone 滑窗（只丢音频行，prompt 前缀钉死）
static uint32_t g_moss_ctx_rows        = 6000;   // ctx 行数上限（≈1.34 GB KV）
static uint32_t g_moss_ctx_keep_recent = 2048;   // 滑窗时保留的最近音频行数
static std::vector<float> g_moss_frame_rms;      // 每个 raw 帧的能量（流式解码时记录）

// patch 0009：在目标点附近挑能量最低的帧作为切点
static size_t moss_pick_cut(size_t target, size_t pinned, size_t pos) {
    const size_t lo = target > 16 ? target - 16 : pinned;
    const size_t hi = std::min<size_t>(target + 16, pos);
    size_t best = target;
    float best_rms = 3.4e38f;
    for (size_t i = lo; i < hi; ++i) {
        if (i >= g_moss_frame_rms.size()) {
            break;
        }
        const float v = g_moss_frame_rms[i];
        if (v > 0.0f && v < best_rms) {
            best_rms = v;
            best = i;
        }
    }
    return best;
}

// patch 0009：ctx 快满时丢掉最老的音频行并把尾巴位置平移回去。
static void moss_maybe_slide(
        llama_context * ctx,
        size_t pinned,
        size_t & pos,
        size_t n_ctx) {
    if (pos + 1 < n_ctx) {
        return;
    }
    const size_t keep = g_moss_ctx_keep_recent;
    const size_t target = pos > keep ? pos - keep : pinned;
    if (target <= pinned) {
        return;
    }
    const size_t cut = moss_pick_cut(target, pinned, pos);
    if (cut <= pinned || cut >= pos) {
        return;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, (llama_pos) pinned, (llama_pos) cut);
    llama_memory_seq_add(mem, 0, (llama_pos) cut, (llama_pos) pos, -(llama_pos) (cut - pinned));
    LOG("ctx: 滑窗 dropped %zu rows（切点 %zu，能量 %.5f），pos %zu -> %zu\\n",
        cut - pinned, cut, cut < g_moss_frame_rms.size() ? g_moss_frame_rms[cut] : 0.0f,
        pos, pos - (cut - pinned));
    pos -= (cut - pinned);
}
"""

# 2) n_ctx 按预算封顶
B_ANCHOR = """        cparams.n_ctx = std::max<uint32_t>((uint32_t) prompt_frames + (uint32_t) max_new_tokens + 8u, 64u);
        cparams.n_batch = std::max<uint32_t>((uint32_t) prompt_frames, 1u);
"""
B_NEW = """        const uint32_t want_ctx = std::max<uint32_t>(
                (uint32_t) prompt_frames + (uint32_t) max_new_tokens + 8u, 64u);
        // patch 0009：ctx 按行数预算封顶；超出的部分靠滑窗滚动（prompt 前缀必须装得下）
        const uint32_t min_ctx = (uint32_t) prompt_frames + 256u;
        cparams.n_ctx = std::max<uint32_t>(
                std::min<uint32_t>(want_ctx, std::max<uint32_t>(g_moss_ctx_rows, min_ctx)),
                min_ctx);
        cparams.n_batch = std::max<uint32_t>((uint32_t) prompt_frames, 1u);
"""

# 3) 位置独立跟踪 + 每请求清空能量表
C_ANCHOR = """        moss_delay_state state = moss_init_delay_state(prompt_packed, cfg);
"""
C_NEW = """        moss_delay_state state = moss_init_delay_state(prompt_packed, cfg);
        size_t pos = prompt_frames;               // patch 0009：滑窗后位置会变，独立跟踪
        g_moss_frame_rms.clear();
"""

# 4) 生成循环：滑窗 + 用 pos 作为位置
D_ANCHOR = """            moss_owned_batch batch = moss_batch_from_packed_rows(
                    generated_packed, generated_packed.size() / cfg.packed_stride() - 1, 1, cfg,
                    prompt_frames + (size_t) step, true);
            const int ret = llama_decode(ctx.get(), batch.batch);
            if (ret != 0) {
                throw std::runtime_error("generation llama_decode failed: " + std::to_string(ret));
            }
"""
D_NEW = """            moss_maybe_slide(ctx.get(), prompt_frames, pos, cparams.n_ctx);   // patch 0009
            moss_owned_batch batch = moss_batch_from_packed_rows(
                    generated_packed, generated_packed.size() / cfg.packed_stride() - 1, 1, cfg,
                    pos, true);
            const int ret = llama_decode(ctx.get(), batch.batch);
            if (ret != 0) {
                throw std::runtime_error("generation llama_decode failed: " + std::to_string(ret));
            }
            ++pos;
"""

# 5) 流式解码时记录每帧能量（滑窗选切点用）
E_ANCHOR = """    std::vector<int16_t> ints(take);
    for (size_t i = 0; i < take; ++i) {
        ints[i] = (int16_t) std::clamp(pcm[drop + i] * 32767.0f, -32768.0f, 32767.0f);
    }
"""
E_NEW = """    // patch 0009：记录每帧 RMS，供滑窗挑低能量切点
    {
        const size_t dsr = (size_t) g_moss_stream_dec.downsample_rate;
        if (g_moss_frame_rms.size() < emit_to) {
            g_moss_frame_rms.resize(emit_to, 0.0f);
        }
        for (size_t f = emit_from; f < emit_to; ++f) {
            const size_t base = drop + (f - emit_from) * dsr;
            double acc = 0.0;
            for (size_t i = 0; i < dsr; ++i) {
                const double v = pcm[base + i];
                acc += v * v;
            }
            g_moss_frame_rms[f] = (float) std::sqrt(acc / (double) std::max<size_t>(dsr, 1));
        }
    }
    std::vector<int16_t> ints(take);
    for (size_t i = 0; i < take; ++i) {
        ints[i] = (int16_t) std::clamp(pcm[drop + i] * 32767.0f, -32768.0f, 32767.0f);
    }
"""

# 6) serve 参数
F_ANCHOR = """        if (arg == "--serve-ubatch-budget" && i + 1 < argc) {
            // patch 0006：pp buffer 可用的显存预算（MiB）
            g_moss_prefill_budget_mib = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""
F_NEW = F_ANCHOR + """        if (arg == "--serve-ctx-rows" && i + 1 < argc) {
            // patch 0009：backbone ctx 行数上限（滑窗触发点）
            g_moss_ctx_rows = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--serve-ctx-keep" && i + 1 < argc) {
            // patch 0009：滑窗时保留的最近音频行数
            g_moss_ctx_keep_recent = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""

EDITS = [(F_TOOL, [
    (A_ANCHOR, A_NEW, "slide/globals"),
    (B_ANCHOR, B_NEW, "slide/ctx-budget"),
    (C_ANCHOR, C_NEW, "slide/pos"),
    (D_ANCHOR, D_NEW, "slide/loop"),
    (E_ANCHOR, E_NEW, "slide/frame-rms"),
    (F_ANCHOR, F_NEW, "slide/serve-args"),
])]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)
    tmp = tempfile.mkdtemp(prefix="moss-patch9-")
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

    chk = tempfile.mkdtemp(prefix="moss-check9-")
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
