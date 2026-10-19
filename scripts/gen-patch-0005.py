#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0005-stream-abort.patch：
流式请求的客户端断开后，daemon **停止生成**（而不是照算到 token 预算结束）。

背景（实测踩出来的）：流式 PCM 是 daemon 主动往 socket 写的。客户端一走，
写立刻失败，但 AR 循环原先不管这事 —— 它会把 max_new_tokens 整个预算算完，
白烧 GPU 十几秒~几十秒，还占着单 slot 挡住后面所有请求。

做法：写失败时置 `g_moss_stream_aborted`，AR 循环每 W 帧检查一次就 break。
（不用 llama_set_abort_callback 是因为那要改 llama_context 的注入点，
 而 AR 循环本来每步都在我们手里，改动更小、语义更清楚。）

用法： python3 scripts/gen-patch-0005.py
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
OUT = os.path.join(PATCH_DIR, "0005-stream-abort.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1 次）：\n%s" % (tag, n, anchor[:400]))
    return text.replace(anchor, replacement, 1)


# 1) 新增 abort 标志（跟在 0004 那批全局量后面）
A_ANCHOR = """static bool     g_moss_stream_header_sent    = false;
"""
A_NEW = A_ANCHOR + """static bool     g_moss_stream_aborted        = false;  // 客户端断开 -> 让生成循环尽早收手
"""

# 2) 写失败 = 客户端没了：置标志（所有流式写都走这个 choke point）
B_ANCHOR = """        const ssize_t n = ::write(fd, p + off, bytes - off);
        if (n <= 0) {
            return false;
        }
"""
B_NEW = """        const ssize_t n = ::write(fd, p + off, bytes - off);
        if (n <= 0) {
            g_moss_stream_aborted = true;   // patch 0005：客户端走了，别再白算
            return false;
        }
"""

# 3) AR 循环里检查标志（紧跟 0004 的 pump 调用）
C_ANCHOR = """            if (g_moss_stream_fd >= 0 && (step + 1) % (int32_t) g_moss_stream_window == 0) {
                moss_stream_pump(state, prompt_frames, cfg, audio_decoder_model_path, n_gpu_layers);
            }
"""
C_NEW = C_ANCHOR + """            if (g_moss_stream_aborted) {
                LOG("stream: 客户端已断开，提前结束生成（已跑 %d 步）\\n", step + 1);
                break;
            }
"""

# 4) 每个流式请求开始时复位标志
D_ANCHOR = """        if (streaming) {
            g_moss_stream_fd          = cfd;
            g_moss_stream_emitted     = 0;
            g_moss_stream_header_sent = false;
        }
"""
D_NEW = """        if (streaming) {
            g_moss_stream_fd          = cfd;
            g_moss_stream_emitted     = 0;
            g_moss_stream_header_sent = false;
            g_moss_stream_aborted     = false;   // patch 0005
        }
"""

# 5) accepted fd 设发送/接收超时：坏客户端不能无限期占住单 slot
E_ANCHOR = """        const int cfd = ::accept(lfd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) {
                continue;
            }
            LOG_ERR("serve: accept() failed: %s\\n", std::strerror(errno));
            break;
        }
"""
E_NEW = """        const int cfd = ::accept(lfd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) {
                continue;
            }
            LOG_ERR("serve: accept() failed: %s\\n", std::strerror(errno));
            break;
        }
        {
            // patch 0005：给这条连接装上限，坏客户端（不读/不发）不能把单 slot 占死
            struct timeval tv_snd;
            tv_snd.tv_sec  = 30;   // PCM 写不出去 30s -> 认为客户端没了
            tv_snd.tv_usec = 0;
            ::setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, &tv_snd, sizeof(tv_snd));
            struct timeval tv_rcv;
            tv_rcv.tv_sec  = 60;   // 一条连接最多空闲 60s
            tv_rcv.tv_usec = 0;
            ::setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv_rcv, sizeof(tv_rcv));
        }
"""

# 6) CODES op：注册音色算参考 codes，在常驻进程内跑（CPU），不再起第二个进程
B = chr(92)   # 反斜杠：显式拼出 C++ 源码里需要的 \t / \n 字面量

F_ANCHOR = """        // 非流式：整段一次性解码，decoder 非因果 → compute buffer 随长度线性涨，
"""
F_NEW = ("""        // patch 0005：CODES%(t)s<backbone.gguf>%(t)s<encoder.gguf>%(t)s<ref.wav>%(t)s<out.codes.bin>
        // 参考音频 -> codes（注册音色用）。复用现成 CLI 路径但强制 -ngl 0（纯 CPU）：
        // 不起新进程、不重复加载权重、不占显存（常驻对已经吃掉 6.5~7.7 GB）。
        if (body.rfind("CODES%(t)s", 0) == 0) {
            const std::vector<std::string> t = moss_split_tab(body.substr(6));
            const auto tc0 = std::chrono::steady_clock::now();
            int crc = EXIT_FAILURE;
            std::string cerr;
            if (t.size() < 4) {
                cerr = "CODES 需要 4 个参数：<backbone.gguf> <encoder.gguf> <ref.wav> <out.codes.bin>";
            } else {
                std::string a[] = {
                    "-m", t[0],
                    "--audio-encoder-model", t[1],
                    "--reference-audio", t[2],
                    "--dump-reference-codes", t[3],
                    "--text", "。", "--max-new-tokens", "1", "--seed", "1",
                    "-ngl", "0",
                };
                std::vector<char *> ca;
                ca.reserve(15);
                ca.push_back(const_cast<char *>("llama-moss-tts"));
                for (std::string & v : a) {
                    ca.push_back(const_cast<char *>(v.c_str()));
                }
                try {
                    crc = moss_cli_main(static_cast<int>(ca.size()), ca.data());
                } catch (const std::exception & e) {
                    cerr = e.what();
                }
            }
            const long long cms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - tc0).count();
            const std::string resp = (crc == EXIT_SUCCESS && cerr.empty())
                    ? ("OK " + std::to_string(cms) + "%(n)s")
                    : ("ERR " + std::to_string(cms) + " " + cerr + "%(n)s");
            if (!moss_write_all(cfd, resp)) {
                break;
            }
            continue;
        }

""" % {"t": B + "t", "n": B + "n"}) + F_ANCHOR

EDITS = [(F_TOOL, [
    (A_ANCHOR, A_NEW, "abort/flag"),
    (B_ANCHOR, B_NEW, "abort/on-write-fail"),
    (C_ANCHOR, C_NEW, "abort/ar-loop-check"),
    (D_ANCHOR, D_NEW, "abort/reset-per-request"),
    (E_ANCHOR, E_NEW, "serve/sock-timeouts"),
    (F_ANCHOR, F_NEW, "serve/codes-op"),
])]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)
    tmp = tempfile.mkdtemp(prefix="moss-patch5-")
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

    chk = tempfile.mkdtemp(prefix="moss-check5-")
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
