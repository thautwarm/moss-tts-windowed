#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 patches/0003-resident-models.patch（常驻模型 + unix socket 服务模式）。

为什么不手写补丁：本补丁必须叠在 0001/0002 之后，行号会随它们漂移。
所以这里做的是「锚点替换」：
  1. 取 pristine 源码（upstream/llama.cpp）
  2. 先打 0001 + 0002 得到基线（补丁实际应用时的那个状态）
  3. 在基线上按**唯一子串**做替换（重复 0 次或 >1 次直接断言失败）
  4. diff 基线 vs 改后 -> unified diff 写入 patches/0003-resident-models.patch
  5. 自检：把 0001+0002+0003 依次打进一份全新副本，必须干净通过

用法： python3 scripts/gen-patch-0003.py
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
OUT = os.path.join(PATCH_DIR, "0003-resident-models.patch")

F_LLAMA_H = "include/llama.h"
F_MODEL_CPP = "src/llama-model.cpp"
F_TOOL = "tools/tts/run-moss-tts-delay.cpp"


def die(msg):
    print("!! " + msg, file=sys.stderr)
    sys.exit(1)


def sub_once(text, anchor, replacement, tag):
    n = text.count(anchor)
    if n != 1:
        die("[%s] 锚点命中 %d 次（期望恰好 1 次）：%r" % (tag, n, anchor[:120]))
    return text.replace(anchor, replacement, 1)


# --------------------------------------------------------------------------
# 1) include/llama.h —— 导出常驻登记 API
# --------------------------------------------------------------------------
LLAMA_H_ANCHOR = "    LLAMA_API void llama_model_free(struct llama_model * model);\n"
LLAMA_H_NEW = LLAMA_H_ANCHOR + """
    // ---- moss-tts-windowed patch 0003：常驻模型登记 ----
    // 登记为常驻（keep=true）后，llama_model_free() 不再真正释放该模型；
    // 由进程出口（或显式 keep=false）回收。用途：常驻服务只反序列化一次权重、
    // 跨请求复用，消灭每请求的权重加载开销。
    LLAMA_API void llama_model_keep_resident(struct llama_model * model, bool keep);
    LLAMA_API bool llama_model_is_resident(const struct llama_model * model);
"""

# --------------------------------------------------------------------------
# 2) src/llama-model.cpp —— 登记表实现 + 在 free 处短路
# --------------------------------------------------------------------------
MODEL_CPP_INC_ANCHOR = "#include <functional>\n#include <map>\n"
MODEL_CPP_INC_NEW = "#include <functional>\n#include <map>\n#include <mutex>\n#include <unordered_set>\n"

MODEL_CPP_FREE_ANCHOR = "void llama_model_free(llama_model * model) {\n    delete model;\n}\n"
MODEL_CPP_FREE_NEW = """// ---- moss-tts-windowed patch 0003：常驻模型登记表 ----
// 只记录指针，不持有别的资源；进程退出时由操作系统回收。
static std::mutex                              g_moss_resident_mutex;
static std::unordered_set<const llama_model *> g_moss_resident_models;

void llama_model_keep_resident(llama_model * model, bool keep) {
    if (model == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_moss_resident_mutex);
    if (keep) {
        g_moss_resident_models.insert(model);
    } else {
        g_moss_resident_models.erase(model);
    }
}

bool llama_model_is_resident(const struct llama_model * model) {
    if (model == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_moss_resident_mutex);
    return g_moss_resident_models.find(model) != g_moss_resident_models.end();
}

void llama_model_free(llama_model * model) {
    // patch 0003：常驻模型不释放（保活到进程退出）
    if (llama_model_is_resident(model)) {
        return;
    }
    delete model;
}
"""

# --------------------------------------------------------------------------
# 3) tools/tts/run-moss-tts-delay.cpp —— 缓存助手 / 4 个加载点 / 服务模式
# --------------------------------------------------------------------------
TOOL_INC_ANCHOR = "#include <utility>\n#include <vector>\n"
TOOL_INC_NEW = """#include <utility>
#include <vector>
// ---- moss-tts-windowed patch 0003 ----
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
"""

# (3a) 常驻缓存助手，插在 moss_load_audio_model 之前
TOOL_CACHE_ANCHOR = """static llama_model_ptr moss_load_audio_model(
        const std::string & model_path,
        const char * expected_arch,
        int32_t n_gpu_layers) {
"""
TOOL_CACHE_NEW = """// ---- moss-tts-windowed patch 0003：常驻模型缓存 ----
// 进程级缓存：同一 (路径 + 关键加载参数) 只反序列化一次，之后跨请求复用。
// 命中时返回同一个裸指针；llama_model_ptr 析构会调 llama_model_free()，
// 但被登记为常驻的模型不会被真正释放（见 src/llama-model.cpp）。
//
// key 必须包含 vocab_only 与 n_gpu_layers：
//   backbone 在「文本 -> prompt」路径下是 vocab_only=true 加载，在生成路径下是全量加载，
//   两者是不同的对象，混用会直接出错。
static llama_model * moss_get_or_load_resident_model(
        const std::string & model_path,
        const llama_model_params & mparams) {
    static std::mutex mu;
    static std::unordered_map<std::string, llama_model *> cache;

    const std::string key = model_path +
            "|vocab_only=" + (mparams.vocab_only ? "1" : "0") +
            "|ngl=" + std::to_string(mparams.n_gpu_layers);

    std::lock_guard<std::mutex> lock(mu);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        LOG("resident: reuse model %s\\n", model_path.c_str());
        return it->second;
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        return nullptr;
    }
    llama_model_keep_resident(model, true);
    cache.emplace(key, model);
    LOG("resident: loaded and kept model %s (vocab_only=%d ngl=%d)\\n",
        model_path.c_str(), mparams.vocab_only ? 1 : 0, mparams.n_gpu_layers);
    return model;
}

""" + TOOL_CACHE_ANCHOR

# (3b) moss_load_audio_model 里：只缓存 audio-decoder，encoder 不缓存
TOOL_AUDIO_ANCHOR = """    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), mparams));
    if (!model) {
        throw std::runtime_error("failed to load audio model: " + model_path);
"""
TOOL_AUDIO_NEW = """    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    mparams.n_gpu_layers = n_gpu_layers;

    // patch 0003：只让 audio-decoder 进常驻缓存。
    // encoder（参考音频编码用）常驻会让显存从 7.0 GB 涨到 7.9 GB —— 8.15 GB 的卡上
    // 与每请求 KV 叠加必 OOM，所以 encoder 仍按需加载、用完即走；
    // 参考音频改走预计算 codes（voices/<id>.<指纹>.codes.bin），根本不碰 encoder。
    const bool keep_resident =
            expected_arch != nullptr && std::strcmp(expected_arch, "moss-tts-audio-decoder") == 0;

    llama_model_ptr model(keep_resident
            ? moss_get_or_load_resident_model(model_path, mparams)
            : llama_model_load_from_file(model_path.c_str(), mparams));
    if (!model) {
        throw std::runtime_error("failed to load audio model: " + model_path);
"""

# (3c) 生成路径上的全量 backbone 加载（8 空格缩进，错误信息 "failed to load model: "）
TOOL_GEN_ANCHOR = """        llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), mparams));
        if (!model) {
            throw std::runtime_error("failed to load model: " + model_path);
"""
TOOL_GEN_NEW = """        llama_model_ptr model(moss_get_or_load_resident_model(model_path, mparams));
        if (!model) {
            throw std::runtime_error("failed to load model: " + model_path);
"""

# (3d) 文本->prompt 路径上的 vocab_only 加载
TOOL_VOCAB_ANCHOR = """        llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), mparams));
        if (!model) {
            throw std::runtime_error("failed to load vocab-only model: " + model_path);
"""
TOOL_VOCAB_NEW = """        llama_model_ptr model(moss_get_or_load_resident_model(model_path, mparams));
        if (!model) {
            throw std::runtime_error("failed to load vocab-only model: " + model_path);
"""

# (3e) main 里 --print-delay-config 的全量加载
TOOL_MAIN_LOAD_ANCHOR = """    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), mparams));
    if (!model) {
        LOG_ERR("failed to load model: %s\\n", model_path.c_str());
        return EXIT_FAILURE;
    }
"""
TOOL_MAIN_LOAD_NEW = """    llama_model_ptr model(moss_get_or_load_resident_model(model_path, mparams));
    if (!model) {
        LOG_ERR("failed to load model: %s\\n", model_path.c_str());
        return EXIT_FAILURE;
    }
"""

# (3f) 服务模式：把 main 改名，前面插入 serve 入口
TOOL_MAIN_ANCHOR = "int main(int argc, char ** argv) {\n"
TOOL_MAIN_NEW = r"""// ============================================================================
// ---- moss-tts-windowed patch 0003：常驻服务模式 ----
//
// --serve-unix <path>：进程常驻，在 unix socket 上收请求。每条请求 = 一行
// 制表符分隔的 CLI 参数（与命令行完全同一套参数），并发回应一行结果：
//
//   OK <毫秒>                 生成成功（wav 已写到请求里 --wav-out 指定处）
//   ERR <毫秒> <消息>         失败
//
// 设计要点：
//   * 复用完整 CLI 代码路径（moss_cli_main），不复制生成逻辑 —— 行为与命令行逐位一致
//   * 全程持有一个 llama_backend_scope（fork 里已有的引用计数 RAII），
//     使每请求内部的 scope 不再 init/free 后端，常驻模型得以存活
//   * 串行处理：单卡单槽，排队交给前端
//   * 日志照旧走 stdout/stderr，socket 上只有协议，互不污染
// ============================================================================

static int moss_cli_main(int argc, char ** argv);

struct moss_line_reader {
    int fd;
    std::string buf;

    // 读一行（不含 '\n'）；对端关闭且无残留返回 false
    bool read_line(std::string & out) {
        while (true) {
            const size_t pos = buf.find('\n');
            if (pos != std::string::npos) {
                out.assign(buf, 0, pos);
                buf.erase(0, pos + 1);
                return true;
            }
            char tmp[8192];
            const ssize_t n = ::read(fd, tmp, sizeof(tmp));
            if (n <= 0) {
                if (!buf.empty()) {
                    out = buf;
                    buf.clear();
                    return true;
                }
                return false;
            }
            buf.append(tmp, static_cast<size_t>(n));
            if (buf.size() > (8u << 20)) {
                buf.clear();
                throw std::runtime_error("serve: request line too long");
            }
        }
    }
};

static bool moss_write_all(int fd, const std::string & s) {
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::write(fd, s.data() + off, s.size() - off);
        if (n <= 0) {
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

static std::vector<std::string> moss_split_tab(const std::string & line) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        const size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            out.push_back(line.substr(start));
            break;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

static void moss_handle_connection(
        int cfd,
        const std::vector<std::string> & base_args,
        const std::string & argv0) {
    moss_line_reader reader{cfd, std::string()};
    std::string line;
    while (reader.read_line(line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> args;
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
            resp = "OK " + std::to_string(ms) + "\n";
        } else {
            resp = "ERR " + std::to_string(ms) + " " + err + "\n";
        }
        if (!moss_write_all(cfd, resp)) {
            break;
        }
    }
}

static int moss_serve_unix(
        const std::string & sock_path,
        const std::vector<std::string> & base_args,
        const std::string & argv0) {
    ::unlink(sock_path.c_str());

    const int lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (lfd < 0) {
        LOG_ERR("serve: socket() failed: %s\n", std::strerror(errno));
        return EXIT_FAILURE;
    }

    sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (sock_path.size() + 1 > sizeof(addr.sun_path)) {
        LOG_ERR("serve: socket path too long: %s\n", sock_path.c_str());
        return EXIT_FAILURE;
    }
    std::strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(lfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        LOG_ERR("serve: bind(%s) failed: %s\n", sock_path.c_str(), std::strerror(errno));
        return EXIT_FAILURE;
    }
    if (::listen(lfd, 16) != 0) {
        LOG_ERR("serve: listen() failed: %s\n", std::strerror(errno));
        return EXIT_FAILURE;
    }
    LOG("serve: listening on %s (base args: %zu)\n", sock_path.c_str(), base_args.size());

    // 常驻后端：整个服务期只 init 一次，常驻模型得以存活
    llama_backend_scope serve_scope;

    while (true) {
        const int cfd = ::accept(lfd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR) {
                continue;
            }
            LOG_ERR("serve: accept() failed: %s\n", std::strerror(errno));
            break;
        }
        try {
            moss_handle_connection(cfd, base_args, argv0);
        } catch (const std::exception & e) {
            LOG_ERR("serve: connection error: %s\n", e.what());
        }
        ::close(cfd);
    }

    ::close(lfd);
    return EXIT_SUCCESS;
}

int main(int argc, char ** argv) {
    // patch 0003：先在入口摘掉 --serve-unix，其余参数原样透传给正常 CLI 路径
    std::string serve_unix_path;
    std::vector<std::string> user_args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--serve-unix" && i + 1 < argc) {
            serve_unix_path = argv[++i];
            continue;
        }
        user_args.push_back(arg);
    }

    if (!serve_unix_path.empty()) {
        return moss_serve_unix(serve_unix_path, user_args, argv[0]);
    }

    std::vector<char *> cargs;
    cargs.reserve(user_args.size() + 1);
    cargs.push_back(argv[0]);
    for (std::string & s : user_args) {
        cargs.push_back(const_cast<char *>(s.c_str()));
    }
    return moss_cli_main(static_cast<int>(cargs.size()), cargs.data());
}

static int moss_cli_main(int argc, char ** argv) {
"""


EDITS = [
    (F_LLAMA_H, [(LLAMA_H_ANCHOR, LLAMA_H_NEW, "llama.h/declare")]),
    (F_MODEL_CPP, [
        (MODEL_CPP_INC_ANCHOR, MODEL_CPP_INC_NEW, "model.cpp/includes"),
        (MODEL_CPP_FREE_ANCHOR, MODEL_CPP_FREE_NEW, "model.cpp/free-hook"),
    ]),
    (F_TOOL, [
        (TOOL_INC_ANCHOR, TOOL_INC_NEW, "tool/includes"),
        (TOOL_CACHE_ANCHOR, TOOL_CACHE_NEW, "tool/cache-helper"),
        (TOOL_AUDIO_ANCHOR, TOOL_AUDIO_NEW, "tool/audio-decoder-only"),
        (TOOL_GEN_ANCHOR, TOOL_GEN_NEW, "tool/backbone-full"),
        (TOOL_VOCAB_ANCHOR, TOOL_VOCAB_NEW, "tool/backbone-vocabonly"),
        (TOOL_MAIN_LOAD_ANCHOR, TOOL_MAIN_LOAD_NEW, "tool/main-debug-load"),
        (TOOL_MAIN_ANCHOR, TOOL_MAIN_NEW, "tool/serve-entry"),
    ]),
]


def apply_patch(root, patch_path):
    with open(patch_path, "rb") as fh:
        r = subprocess.run(["patch", "-p1", "--forward", "--batch"],
                           cwd=root, stdin=fh, capture_output=True, text=True)
    return r


def main():
    if not os.path.isdir(UP):
        die("缺少 %s" % UP)

    # 基线 = pristine + 0001 + 0002（补丁真实应用时的状态）
    tmp = tempfile.mkdtemp(prefix="moss-patch-")
    base = os.path.join(tmp, "src")
    shutil.copytree(UP, base)

    prev = sorted(p for p in os.listdir(PATCH_DIR)
                  if p.endswith(".patch") and p < "0003")
    for p in prev:
        r = apply_patch(base, os.path.join(PATCH_DIR, p))
        if r.returncode != 0:
            die("基线补丁 %s 应用失败：%s%s" % (p, r.stdout, r.stderr))
        print("基线: 已应用 %s" % p)

    # 在基线上做锚点替换
    changed = {}
    for rel, edits in EDITS:
        path = os.path.join(base, rel)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        for anchor, repl, tag in edits:
            text = sub_once(text, anchor, repl, tag)
            print("编辑: %-28s %s" % (tag, rel))
        changed[rel] = text

    # 生成 unified diff
    hunks = []
    for rel, new_text in changed.items():
        with open(os.path.join(base, rel), "r", encoding="utf-8") as fh:
            old_lines = fh.readlines()
        # 此时 base 还是基线（未改），需要与改后比较
        new_lines = new_text.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines,
                                    fromfile="a/" + rel, tofile="b/" + rel, n=3)
        hunks.extend(diff)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.writelines(hunks)
    print("\n写出 %s（%d 行）" % (OUT, len(hunks)))

    # 自检：全新副本上依次应用 0001 0002 0003
    chk = tempfile.mkdtemp(prefix="moss-check-")
    chk_src = os.path.join(chk, "src")
    shutil.copytree(UP, chk_src)
    for p in sorted(x for x in os.listdir(PATCH_DIR) if x.endswith(".patch")):
        r = apply_patch(chk_src, os.path.join(PATCH_DIR, p))
        if r.returncode != 0:
            die("自检失败：%s 应用不干净：%s%s" % (p, r.stdout, r.stderr))
        print("自检: %s 干净" % p)

    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(chk, ignore_errors=True)
    print("OK")


if __name__ == "__main__":
    main()
