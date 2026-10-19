#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 patches/0014-duration-tokens.patch：把 prompt 里写死的 `- Tokens: None` 变成可控。

上游参考实现（moss_tts_processor.py）该字段写的是 `- Tokens:\\n{tokens}`，
是模型的**原生时长控制**入口；本 fork 里写死成 None，所以一直没得用。
"""
import difflib, os, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = os.path.join(ROOT, "upstream", "llama.cpp"); PATCH_DIR = os.path.join(ROOT, "patches")
OUT = os.path.join(PATCH_DIR, "0014-duration-tokens.patch"); F = "tools/tts/run-moss-tts-delay.cpp"
def die(m): print("!! "+m, file=sys.stderr); sys.exit(1)
def once(t,a,r,tag):
    n=t.count(a)
    if n!=1: die("[%s] 锚点 %d 次：\n%s"%(tag,n,a[:300]))
    return t.replace(a,r,1)

A = '''static std::string        g_moss_vocab_model_path;
'''
B = '''static long long g_moss_duration_tokens = -1;   // patch 0014：-1 = 不指定（写 None）
'''
C_OLD = '''            "- Instruction:\\nNone\\n"
            "- Tokens:\\nNone\\n"
'''
C_NEW = '''            "- Instruction:\\nNone\\n"
            "- Tokens:\\n" + (g_moss_duration_tokens >= 0
                    ? std::to_string(g_moss_duration_tokens)
                    : std::string("None")) + "\\n"
'''
D_OLD = '''        if (arg == "--serve-vocab-model" && i + 1 < argc) {
            // patch 0010：TOKENS op 用的词表模型（通常就是 -m 那个文件）
            g_moss_vocab_model_path = argv[++i];
            continue;
        }
'''
D_NEW = D_OLD + '''        if (arg == "--serve-duration-tokens" && i + 1 < argc) {
            g_moss_duration_tokens = std::strtoll(argv[++i], nullptr, 10);
            continue;
        }
'''
E_OLD = '''        if (arg == "--max-new-tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
            continue;
        }
'''
E_NEW = E_OLD + '''        if (arg == "--duration-tokens" && i + 1 < argc) {
            // patch 0014：写进 prompt 的 - Tokens 字段（原生时长控制）
            g_moss_duration_tokens = std::strtoll(argv[++i], nullptr, 10);
            continue;
        }
'''
EDITS=[(F,[(A,A+B,"dur/global"),(C_OLD,C_NEW,"dur/prompt"),(D_OLD,D_NEW,"dur/serve-arg"),(E_OLD,E_NEW,"dur/cli-arg")])]
def ap(root,p):
    with open(p,"rb") as fh: return subprocess.run(["patch","-p1","--forward","--batch"],cwd=root,stdin=fh,capture_output=True,text=True)
def main():
    tmp=tempfile.mkdtemp(prefix="p14-"); base=os.path.join(tmp,"src"); shutil.copytree(UP,base)
    mine=os.path.basename(OUT)
    for p in sorted(x for x in os.listdir(PATCH_DIR) if x.endswith(".patch") and x<mine):
        if ap(base,os.path.join(PATCH_DIR,p)).returncode!=0: die("基线失败 "+p)
    rel,ed=EDITS[0]; path=os.path.join(base,rel); text=open(path,encoding="utf-8").read()
    for a,r,tag in ed: text=once(text,a,r,tag); print("编辑:",tag)
    old=open(path,encoding="utf-8").readlines()
    open(OUT,"w",encoding="utf-8").writelines(difflib.unified_diff(old,text.splitlines(keepends=True),fromfile="a/"+rel,tofile="b/"+rel,n=3))
    print("写出",OUT)
    chk=tempfile.mkdtemp(prefix="c14-"); src=os.path.join(chk,"src"); shutil.copytree(UP,src)
    for p in sorted(x for x in os.listdir(PATCH_DIR) if x.endswith(".patch")):
        if ap(src,os.path.join(PATCH_DIR,p)).returncode!=0: die("自检失败 "+p)
    print("OK")
main()
