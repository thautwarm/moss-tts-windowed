#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0004-streaming.patch：
  (1) 真流式：AR 循环里每 W 帧把「已定稿的 raw codes」用滑动窗口增量解码，边生成边写 PCM
  (2) 非流式护栏：整段解码的 --max-new-tokens 上限（超了直接 ERR，不冒 CUDA abort 的险）

同上一个生成器：锚点替换 + 断言，基线 = pristine + 之前所有补丁。
用法： python3 scripts/gen-patch-0004.py
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
OUT = os.path.join(PATCH_DIR, "0004-streaming.patch")
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望 1 次）：\n%s" % (tag, n, anchor[:400]))
    return text.replace(anchor, replacement, 1)


# ---------------------------------------------------------------------------
# 1) 全局状态 + 前向声明（放在已有前向声明区之后、moss_generate_from_ref 之前）
# ---------------------------------------------------------------------------
A_ANCHOR = """static bool save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate);
static std::vector<llama_token> moss_encode_audio_llama(
"""
A_NEW = r"""// ============================================================================
// ---- moss-tts-windowed patch 0004：真流式（增量解码）----
//
// 生成是逐帧出的（1 帧 ≈ 0.08 s 音频），但非流式路径要等整段 codes 生成完再**一次**
// llama_encode —— 这既是「首块要等 5~8 s」的原因，也是单请求长度上限（decoder 非因果、
// compute buffer 随长度线性涨）的来源。
//
// 流式路径：每 W 帧，把已经**定稿**的 raw codes 按滑动窗口（左右各留 O 帧上下文）交给
// decoder 解一次，只把中间那 W 帧的 PCM 写回 socket。窗口 ≤ W+2O 帧 → decoder ctx 极小
// （几十 MiB），所以流式请求**没有长度上限**，也不会碰到非流式的 OOM/abort 边界。
// ============================================================================

static int      g_moss_stream_fd             = -1;   // >=0：本次请求边生成边吐 PCM
static size_t   g_moss_stream_emitted        = 0;    // 已经吐出去的 raw 帧数
static bool     g_moss_stream_header_sent    = false;
static uint32_t g_moss_stream_window         = 12;   // 每块推进的帧数（≈0.96 s）
static uint32_t g_moss_stream_overlap        = 12;   // 每侧上下文帧数（≈0.96 s）
static long long g_moss_serve_max_new_tokens = 448;  // 非流式整段解码上限（0 = 不限）

static void moss_stream_pump(
        const moss_delay_state & state,
        size_t prompt_frames,
        const moss_delay_config & cfg,
        const std::string & decoder_path,
        int32_t n_gpu_layers);

static void moss_stream_pump_flush(
        const moss_delay_state & state,
        size_t prompt_frames,
        const moss_delay_config & cfg,
        const std::string & decoder_path,
        int32_t n_gpu_layers);

""" + A_ANCHOR

# ---------------------------------------------------------------------------
# 2) 流式实现（放在 decoder 非流式解码函数之前；依赖都已定义）
# ---------------------------------------------------------------------------
B_ANCHOR = """static void moss_decode_audio_llama(
        const std::string & audio_decoder_model_path,
        const std::vector<llama_token> & raw_codes,
        size_t raw_frames,
        const moss_delay_config & cfg,
        int32_t n_gpu_layers,
        const std::string & wav_out_path) {
    moss_audio_runtime runtime = moss_load_audio_runtime(
"""
B_NEW = r"""// ---- patch 0004：流式解码器（常驻 ctx，窗口固定）----
struct moss_stream_decoder_state {
    bool              ready = false;
    llama_model_ptr   model;
    llama_context_ptr ctx;
    size_t            n_ctx           = 0;
    int               sample_rate     = 24000;
    uint32_t          downsample_rate = 0;
};

static moss_stream_decoder_state g_moss_stream_dec;

static bool moss_stream_write_all(int fd, const void * data, size_t bytes) {
    const char * p = static_cast<const char *>(data);
    size_t off = 0;
    while (off < bytes) {
        const ssize_t n = ::write(fd, p + off, bytes - off);
        if (n <= 0) {
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

static void moss_stream_send_header() {
    if (g_moss_stream_header_sent || g_moss_stream_fd < 0) {
        return;
    }
    const std::string hdr = "S " + std::to_string(g_moss_stream_dec.sample_rate) + " 1 16\n";
    moss_stream_write_all(g_moss_stream_fd, hdr.data(), hdr.size());
    g_moss_stream_header_sent = true;
}

static void moss_stream_decoder_init(
        const std::string & decoder_path,
        int32_t n_gpu_layers,
        const moss_delay_config & cfg,
        size_t max_frames) {
    const size_t n_ctx = std::max<size_t>(max_frames + 8, 32);
    if (g_moss_stream_dec.ready && g_moss_stream_dec.n_ctx >= n_ctx) {
        return;
    }

    if (!g_moss_stream_dec.ready) {
        // 模型只加载一次（并且是常驻的：登记表里不会被释放）
        g_moss_stream_dec.model = moss_load_audio_model(
                decoder_path, "moss-tts-audio-decoder", n_gpu_layers);

        const uint32_t quantizers = moss_audio_model_num_quantizers(
                g_moss_stream_dec.model.get(), "moss-tts-audio-decoder");
        if (quantizers != cfg.n_vq) {
            throw std::runtime_error(
                    "streaming: decoder quantizer count mismatch: model has " +
                    std::to_string(quantizers) + ", cfg has " + std::to_string(cfg.n_vq));
        }
        g_moss_stream_dec.downsample_rate = moss_audio_model_downsample_rate(
                g_moss_stream_dec.model.get(), "moss-tts-audio-decoder");
        g_moss_stream_dec.sample_rate = (int) moss_audio_model_sampling_rate(
                g_moss_stream_dec.model.get(), "moss-tts-audio-decoder");
    }

    // 窗口变大时只重建 ctx（模型留着）；ctx 很小，重建代价可忽略
    g_moss_stream_dec.ctx.reset();
    g_moss_stream_dec.ctx = moss_init_audio_context(
            g_moss_stream_dec.model.get(), (uint32_t) n_ctx);
    g_moss_stream_dec.n_ctx = n_ctx;
    g_moss_stream_dec.ready = true;

    LOG("stream: decoder ctx 就绪 n_ctx=%zu downsample=%u rate=%d\n",
        n_ctx, g_moss_stream_dec.downsample_rate, g_moss_stream_dec.sample_rate);
}

// 解一段 codes（非因果整段 encode，窗口自洽：每次先清 KV）
static void moss_stream_decode(
        const moss_delay_config & cfg,
        const std::vector<llama_token> & codes,
        size_t frames,
        std::vector<float> & out) {
    moss_owned_batch batch = moss_batch_from_audio_codes(codes, frames, cfg.n_vq);
    llama_memory_clear(llama_get_memory(g_moss_stream_dec.ctx.get()), true);
    const int ret = llama_encode(g_moss_stream_dec.ctx.get(), batch.batch);
    if (ret != 0) {
        throw std::runtime_error("streaming decode failed: " + std::to_string(ret));
    }
    const float * embd = llama_get_embeddings(g_moss_stream_dec.ctx.get());
    if (embd == nullptr) {
        throw std::runtime_error("streaming decode returned null embeddings");
    }
    out.assign(embd, embd + frames * (size_t) g_moss_stream_dec.downsample_rate);
}

// [emit_from, emit_to) 这 W 帧的 PCM 吐出去；窗口左右各带 ctx 帧（丢弃=不吐）
static bool moss_stream_emit_range(
        const moss_delay_config & cfg,
        const std::vector<llama_token> & raw_codes,
        size_t raw_frames,
        size_t emit_from,
        size_t emit_to,
        size_t ctx,
        bool   keep_right_context) {
    const size_t win_begin = emit_from > ctx ? emit_from - ctx : 0;
    const size_t win_end   = keep_right_context
            ? std::min(raw_frames, emit_to + ctx)
            : emit_to;

    std::vector<llama_token> codes;
    codes.reserve((win_end - win_begin) * cfg.n_vq);
    codes.insert(
            codes.end(),
            raw_codes.begin() + (long) (win_begin * cfg.n_vq),
            raw_codes.begin() + (long) (win_end * cfg.n_vq));

    std::vector<float> pcm;
    moss_stream_decode(cfg, codes, win_end - win_begin, pcm);

    const size_t drop = (emit_from - win_begin) * (size_t) g_moss_stream_dec.downsample_rate;
    const size_t take = (emit_to - emit_from) * (size_t) g_moss_stream_dec.downsample_rate;
    if (drop + take > pcm.size()) {
        throw std::runtime_error("streaming emit range out of bounds");
    }

    const std::string hdr = "A " + std::to_string(take * 2) + "\n";
    if (!moss_stream_write_all(g_moss_stream_fd, hdr.data(), hdr.size())) {
        return false;
    }
    std::vector<int16_t> ints(take);
    for (size_t i = 0; i < take; ++i) {
        ints[i] = (int16_t) std::clamp(pcm[drop + i] * 32767.0f, -32768.0f, 32767.0f);
    }
    return moss_stream_write_all(g_moss_stream_fd, ints.data(), ints.size() * sizeof(int16_t));
}

static void moss_stream_pump(
        const moss_delay_state & state,
        size_t prompt_frames,
        const moss_delay_config & cfg,
        const std::string & decoder_path,
        int32_t n_gpu_layers) {
    if (g_moss_stream_fd < 0) {
        return;
    }
    const size_t   O = g_moss_stream_overlap;
    const uint32_t W = g_moss_stream_window;
    moss_stream_decoder_init(decoder_path, n_gpu_layers, cfg, (size_t) W + 3 * O + 16);
    moss_stream_send_header();

    const moss_generation_audio now = moss_decode_generation_audio(state, prompt_frames, cfg);
    // 右侧留 O 帧：只有「后面还有 O 帧」的部分才是真正定稿的，这样每帧吐出去时左右都有上下文
    if (now.raw_frames <= O) {
        return;
    }
    const size_t stable_end = now.raw_frames - O;

    while (g_moss_stream_emitted + W <= stable_end) {
        const size_t emit_from = g_moss_stream_emitted;
        const size_t emit_to   = emit_from + W;
        if (!moss_stream_emit_range(cfg, now.raw_codes, now.raw_frames,
                                    emit_from, emit_to, O, true)) {
            g_moss_stream_fd = -1;
            return;
        }
        g_moss_stream_emitted = emit_to;
    }
}

static void moss_stream_pump_flush(
        const moss_delay_state & state,
        size_t prompt_frames,
        const moss_delay_config & cfg,
        const std::string & decoder_path,
        int32_t n_gpu_layers) {
    if (g_moss_stream_fd < 0) {
        return;
    }
    const size_t   O = g_moss_stream_overlap;
    const uint32_t W = g_moss_stream_window;
    moss_stream_decoder_init(decoder_path, n_gpu_layers, cfg, (size_t) W + 3 * O + 16);
    moss_stream_send_header();

    const moss_generation_audio now = moss_decode_generation_audio(state, prompt_frames, cfg);
    // 尾巴没有「未来的上下文」了，只能不带右侧上下文解（与整段路径的结尾同理）
    while (g_moss_stream_emitted < now.raw_frames) {
        const size_t emit_from = g_moss_stream_emitted;
        const size_t emit_to   = std::min(now.raw_frames, emit_from + (size_t) W);
        if (!moss_stream_emit_range(cfg, now.raw_codes, now.raw_frames,
                                    emit_from, emit_to, O, false)) {
            g_moss_stream_fd = -1;
            return;
        }
        g_moss_stream_emitted = emit_to;
    }
}

""" + B_ANCHOR

# ---------------------------------------------------------------------------
# 3) AR 循环里的钩子 + 收尾 flush
# ---------------------------------------------------------------------------
C_ANCHOR = """            if (state.is_stopping) {
                break;
            }
        }

        generated_frames = generated_packed.size() / cfg.packed_stride();
        decoded = moss_decode_generation_audio(state, prompt_frames, cfg);
"""
C_NEW = """            if (g_moss_stream_fd >= 0 && (step + 1) % (int32_t) g_moss_stream_window == 0) {
                moss_stream_pump(state, prompt_frames, cfg, audio_decoder_model_path, n_gpu_layers);
            }

            if (state.is_stopping) {
                break;
            }
        }

        if (g_moss_stream_fd >= 0) {
            moss_stream_pump_flush(state, prompt_frames, cfg, audio_decoder_model_path, n_gpu_layers);
        }

        generated_frames = generated_packed.size() / cfg.packed_stride();
        decoded = moss_decode_generation_audio(state, prompt_frames, cfg);
"""

# ---------------------------------------------------------------------------
# 4) 服务端：流式请求分派 + 非流式护栏（替换 0003 插入的那段 handler 主体）
# ---------------------------------------------------------------------------
D_ANCHOR = """        std::vector<std::string> args;
        args.push_back(argv0);
        args.insert(args.end(), base_args.begin(), base_args.end());
        for (const std::string & tok : moss_split_tab(line)) {
            if (!tok.empty()) {
                args.push_back(tok);
            }
        }

        std::vector<char *> cargs;
        cargs.reserve(args.size());
        for (std::string & s : args) {
            cargs.push_back(const_cast<char *>(s.c_str()));
        }

        const auto t0 = std::chrono::steady_clock::now();
        int rc = EXIT_FAILURE;
        std::string err;
        try {
            rc = moss_cli_main(static_cast<int>(cargs.size()), cargs.data());
        } catch (const std::exception & e) {
            err = e.what();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const long long ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::string resp;
        if (rc == EXIT_SUCCESS && err.empty()) {
            resp = "OK " + std::to_string(ms) + "\\n";
        } else {
            resp = "ERR " + std::to_string(ms) + " " + err + "\\n";
        }
        if (!moss_write_all(cfd, resp)) {
            break;
        }
"""
D_NEW = r"""        // ---- patch 0004：请求分派 ----
        // "STREAM\t" 前缀 = 流式请求（边生成边吐 PCM），不带前缀 = 老的非流式请求。
        bool streaming = false;
        std::string body = line;
        if (body.rfind("STREAM\t", 0) == 0) {
            streaming = true;
            body = body.substr(7);
        }

        std::vector<std::string> args;
        args.push_back(argv0);
        args.insert(args.end(), base_args.begin(), base_args.end());
        for (const std::string & tok : moss_split_tab(body)) {
            if (!tok.empty()) {
                // 流式请求不需要 --wav-out（那会触发一次整段解码，白白吃显存）
                if (streaming && tok == "--wav-out") {
                    args.push_back(tok);
                    args.push_back("/dev/null");
                    continue;
                }
                args.push_back(tok);
            }
        }

        // 非流式：整段一次性解码，decoder 非因果 → compute buffer 随长度线性涨，
        // 超过上限会 CUDA OOM、极端情况 libggml 直接 abort 掉整个进程。这里先拦下来。
        if (!streaming && g_moss_serve_max_new_tokens > 0) {
            long long want = -1;
            for (size_t i = 0; i + 1 < args.size(); ++i) {
                if (args[i] == "--max-new-tokens") {
                    want = std::strtoll(args[i + 1].c_str(), nullptr, 10);
                }
            }
            if (want > g_moss_serve_max_new_tokens) {
                const std::string resp =
                        "ERR 0 --max-new-tokens " + std::to_string(want) +
                        " 超过本栈非流式上限 " + std::to_string(g_moss_serve_max_new_tokens) +
                        "（decoder 非因果整段解码的显存上限）；"
                        "要么调小，要么用 STREAM\\t 前缀走流式（流式无长度上限）\n";
                LOG_ERR("serve: 拒绝非流式超长请求（--max-new-tokens %lld > %lld）\n",
                        want, g_moss_serve_max_new_tokens);
                if (!moss_write_all(cfd, resp)) {
                    break;
                }
                continue;
            }
        }

        std::vector<char *> cargs;
        cargs.reserve(args.size());
        for (std::string & s : args) {
            cargs.push_back(const_cast<char *>(s.c_str()));
        }

        const auto t0 = std::chrono::steady_clock::now();
        int rc = EXIT_FAILURE;
        std::string err;
        if (streaming) {
            g_moss_stream_fd          = cfd;
            g_moss_stream_emitted     = 0;
            g_moss_stream_header_sent = false;
        }
        try {
            rc = moss_cli_main(static_cast<int>(cargs.size()), cargs.data());
        } catch (const std::exception & e) {
            err = e.what();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const long long ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        if (streaming) {
            g_moss_stream_fd = -1;
            std::string tail;
            if (rc == EXIT_SUCCESS && err.empty()) {
                if (!g_moss_stream_header_sent) {
                    // 一帧音频都没有（例如极短文本）：还是要给前端一个合法的头
                    moss_stream_send_header();
                }
                tail = "E\n";
            } else {
                tail = "ERR " + std::to_string(ms) + " " + err + "\n";
            }
            if (!moss_write_all(cfd, tail)) {
                break;
            }
            continue;
        }

        std::string resp;
        if (rc == EXIT_SUCCESS && err.empty()) {
            resp = "OK " + std::to_string(ms) + "\n";
        } else {
            resp = "ERR " + std::to_string(ms) + " " + err + "\n";
        }
        if (!moss_write_all(cfd, resp)) {
            break;
        }
"""

# ---------------------------------------------------------------------------
# 5) 服务入口：新增 --serve-* 开关
# ---------------------------------------------------------------------------
E_ANCHOR = """        if (arg == "--serve-unix" && i + 1 < argc) {
            serve_unix_path = argv[++i];
            continue;
        }
        user_args.push_back(arg);
"""
E_NEW = """        if (arg == "--serve-unix" && i + 1 < argc) {
            serve_unix_path = argv[++i];
            continue;
        }
        if (arg == "--serve-max-new-tokens" && i + 1 < argc) {
            // patch 0004：非流式整段解码的上限（0 = 不限；流式不受它约束）
            g_moss_serve_max_new_tokens = std::strtoll(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--serve-stream-window" && i + 1 < argc) {
            g_moss_stream_window = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--serve-stream-overlap" && i + 1 < argc) {
            g_moss_stream_overlap = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
        user_args.push_back(arg);
"""

# ---------------------------------------------------------------------------
# 6) 每请求可覆盖窗口参数（便于 A/B；前端也能按请求调）
# ---------------------------------------------------------------------------
F_ANCHOR = """        if (arg == "--max-new-tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
            continue;
        }
"""
F_NEW = """        if (arg == "--max-new-tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--stream-window" && i + 1 < argc) {
            // patch 0004：流式每块推进的帧数（1 帧 ≈ 0.08 s 音频）
            g_moss_stream_window = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--stream-overlap" && i + 1 < argc) {
            // patch 0004：流式窗口每侧的上下文帧数
            g_moss_stream_overlap = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
            continue;
        }
"""

EDITS = [(F_TOOL, [
    (A_ANCHOR, A_NEW, "stream/globals"),
    (B_ANCHOR, B_NEW, "stream/impl"),
    (C_ANCHOR, C_NEW, "stream/ar-hook"),
    (D_ANCHOR, D_NEW, "serve/dispatch"),
    (E_ANCHOR, E_NEW, "serve/flags"),
    (F_ANCHOR, F_NEW, "stream/per-request-knobs"),
])]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        return subprocess.run(["patch", "-p1", "--forward", "--batch"],
                              cwd=root, stdin=fh, capture_output=True, text=True)


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)

    tmp = tempfile.mkdtemp(prefix="moss-patch4-")
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
            print("编辑: %-20s %s" % (tag, rel))
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

    chk = tempfile.mkdtemp(prefix="moss-check4-")
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
