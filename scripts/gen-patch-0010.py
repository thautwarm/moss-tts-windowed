#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0010-tokens-op.patch：
给常驻 daemon 加一个 `TOKENS\\t<text>` 操作，用**常驻模型的词表**精确分词，返回 token 数。

为什么需要：
  前端要预估"这段文本会生成多少帧"，才能决定走整段解码还是长上下文、以及给多少 token 预算。
  之前按"字数 × 3.2"估：中文 ≈0.7 token/字、英文 ≈0.25 token/字——同一套系数在英文上会高估 4 倍，
  直接把预算撑爆。用真词表分词后，"字数→token"的方差就没了。

做法：
  · 新增 `--serve-vocab-model <gguf>`（entrypoint 直接传 `-m` 的同一个文件）；
  · 首次 TOKENS 请求时以 `vocab_only=1` 加载（只读词表，几十 MB 内存、不占显存），走常驻登记表复用；
  · 回 `OK <n>\\n`，失败回 `ERR <ms> <msg>\\n`（与其他 op 一致）。

用法： python3 scripts/gen-patch-0010.py
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
OUT = os.path.join(PATCH_DIR, "0010-tokens-op.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1 次）：\n%s" % (tag, n, anchor[:400]))
    return text.replace(anchor, replacement, 1)


# 1) 全局量：词表模型路径 + 缓存的 vocab
A_ANCHOR = """static std::vector<float> g_moss_frame_rms;      // 每个 raw 帧的能量（流式解码时记录）
"""
A_NEW = A_ANCHOR + """
// patch 0010：精确分词（常驻词表，vocab_only，不占显存）
static std::string        g_moss_vocab_model_path;
static const llama_vocab * g_moss_vocab = nullptr;
"""

# 2) TOKENS op：放在 CODES op 之前
B_ANCHOR = """        // patch 0005：CODES\\t<backbone.gguf>\\t<encoder.gguf>\\t<ref.wav>\\t<out.codes.bin>
"""
B_NEW = """        // patch 0010：TOKENS\\t<text> -> OK <token 数>
        if (body.rfind("TOKENS\\t", 0) == 0) {
            const std::string text = body.substr(7);
            const auto t0 = std::chrono::steady_clock::now();
            std::string resp;
            try {
                if (g_moss_vocab == nullptr) {
                    if (g_moss_vocab_model_path.empty()) {
                        throw std::runtime_error("TOKENS: 没有 --serve-vocab-model");
                    }
                    llama_model_params mp = llama_model_default_params();
                    mp.vocab_only = true;
                    mp.use_mmap   = true;
                    llama_model * m = moss_get_or_load_resident_model(g_moss_vocab_model_path, mp);
                    if (m == nullptr) {
                        throw std::runtime_error("TOKENS: 词表模型加载失败");
                    }
                    g_moss_vocab = llama_model_get_vocab(m);
                }
                const std::vector<llama_token> ids = common_tokenize(g_moss_vocab, text, false, true);
                const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                (void) ms;
                resp = "OK " + std::to_string(ids.size()) + "\\n";
            } catch (const std::exception & e) {
                resp = std::string("ERR 0 ") + e.what() + "\\n";
            }
            if (!moss_write_all(cfd, resp)) {
                break;
            }
            continue;
        }

""" + B_ANCHOR

# 3) serve 参数：--serve-vocab-model
C_ANCHOR = """        if (arg == "--serve-ctx-keep" && i + 1 < argc) {
            // patch 0009：滑窗时保留的最近音频行数
            g_moss_ctx_keep_recent = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""
C_NEW = C_ANCHOR + """        if (arg == "--serve-vocab-model" && i + 1 < argc) {
            // patch 0010：TOKENS op 用的词表模型（通常就是 -m 那个文件）
            g_moss_vocab_model_path = argv[++i];
            continue;
        }
"""

EDITS = [(F_TOOL, [
    (A_ANCHOR, A_NEW, "tokens/globals"),
    (B_ANCHOR, B_NEW, "tokens/op"),
    (C_ANCHOR, C_NEW, "tokens/serve-arg"),
])]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)
    tmp = tempfile.mkdtemp(prefix="moss-patch10-")
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

    chk = tempfile.mkdtemp(prefix="moss-check10-")
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
