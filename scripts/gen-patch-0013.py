#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 patches/0013-reject-unknown-op.patch：非协议行直接回 ERR，不再当 CLI 参数解析（会把 usage 打满日志）。"""
import difflib, os, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = os.path.join(ROOT, "upstream", "llama.cpp"); PATCH_DIR = os.path.join(ROOT, "patches")
OUT = os.path.join(PATCH_DIR, "0013-reject-unknown-op.patch"); F_TOOL = "tools/tts/run-moss-tts-delay.cpp"
def die(m): print("!! "+m, file=sys.stderr); sys.exit(1)
def once(t,a,r,tag):
    n=t.count(a)
    if n!=1: die("[%s] 锚点命中 %d 次：\n%s"%(tag,n,a[:300]))
    return t.replace(a,r,1)
A = '''        // patch 0010：TOKENS\\t<text> -> OK <token 数>
'''
B = '''        // patch 0013：只接受协议行（STREAM/CODES/TOKENS/以 '-' 开头的 CLI 参数）。
        // 其他内容直接 ERR —— 否则会被当参数解析，把 usage 打满日志。
        if (!body.empty() && body[0] != '-' &&
                body.rfind("STREAM\\t", 0) != 0 &&
                body.rfind("CODES\\t", 0) != 0 &&
                body.rfind("TOKENS\\t", 0) != 0) {
            if (!moss_write_all(cfd, "ERR 0 unknown request op\\n")) {
                break;
            }
            continue;
        }

''' + A
EDITS=[(F_TOOL,[(A,B,"serve/reject-unknown")])]
def ap(root,p):
    with open(p,"rb") as fh: return subprocess.run(["patch","-p1","--forward","--batch"],cwd=root,stdin=fh,capture_output=True,text=True)
def main():
    tmp=tempfile.mkdtemp(prefix="p13-"); base=os.path.join(tmp,"src"); shutil.copytree(UP,base)
    mine=os.path.basename(OUT)
    for p in sorted(x for x in os.listdir(PATCH_DIR) if x.endswith(".patch") and x<mine):
        if ap(base,os.path.join(PATCH_DIR,p)).returncode!=0: die("基线失败 "+p)
    rel,ed=EDITS[0]; path=os.path.join(base,rel); text=open(path,encoding="utf-8").read()
    for a,r,tag in ed: text=once(text,a,r,tag); print("编辑:",tag)
    old=open(path,encoding="utf-8").readlines()
    hunks=list(difflib.unified_diff(old,text.splitlines(keepends=True),fromfile="a/"+rel,tofile="b/"+rel,n=3))
    open(OUT,"w",encoding="utf-8").writelines(hunks); print("写出",OUT,len(hunks),"行")
    chk=tempfile.mkdtemp(prefix="c13-"); src=os.path.join(chk,"src"); shutil.copytree(UP,src)
    for p in sorted(x for x in os.listdir(PATCH_DIR) if x.endswith(".patch")):
        if ap(src,os.path.join(PATCH_DIR,p)).returncode!=0: die("自检失败 "+p)
        print("自检干净:",p)
main()
