#!/usr/bin/env python3
"""build_summary.py — bang TONG HOP theo PHUONG PHAP x BACKBONE, so voi baseline.

Khac tools/build_table.py: cai do mot dong = mot phep so sanh (1 165 dong, de xem chi
tiet). Cai nay gom lai theo (phuong phap, backbone) de nhin duoc "phuong phap nao an
tren backbone nao" trong mot man hinh.

GOM the nao cho dung: gop cac Delta ghep-cap-theo-fold lai, KHONG lay hieu cua hai trung
binh. Moi Delta da duoc ghep cap trong cung may cung fold tu truoc, nen gop chung lai
chi la mo rong mau, khong bac cau qua may.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest

BACKBONES = ("t5pe", "t5p", "codebert", "unixcoder", "roberta", "t5")
MET = [("f1", "test_macro_f1_at_0.5"), ("f1v", "test_macro_f1_at_valcal"),
       ("roc", "test_roc_auc"), ("pr", "test_pr_auc")]
SRCS = ("4cwe", "com", "full")

def bb_of(hay):
    for b in BACKBONES:
        if b in hay: return b
    return "?"

def parse_arm(arm):
    m = re.match(r"transfer_(none|cwe|latent_bottleneck|latent_proto)_?(.*)$", arm)
    if not m: return None
    mode, rest = m.group(1), m.group(2)
    opt = "recadam"
    for o in ("adamw", "recadam", "spd"):
        if rest.endswith("_" + o): rest, opt = rest[: -len(o) - 1], o; break
    src = "-"
    for s in SRCS:
        if rest == s or rest.startswith(s + "_"): src, rest = s, rest[len(s):].lstrip("_"); break
    else:
        mp = re.match(r"((?:lm|pur)\d+_n\d+)_?(.*)$", rest)
        if mp: src, rest = mp.group(1), mp.group(2)
    lam = "0.05"
    ml = re.match(r"l(\d+)p(\d+)_?(.*)$", rest)
    if ml: lam, rest = f"{ml.group(1)}.{ml.group(2)}", ml.group(3)
    elif rest.startswith("l02"): lam, rest = "0.02", rest[3:].lstrip("_")
    return mode, src, (rest or "(gốc)"), opt, lam

# ---- doc, khu trung theo (may, experiment, seed, fold) ----
MACHINE = [("_ntat2","ntat2"),("_ntat","ntat"),("_158","158"),("_161","161")]
def may_of(root):
    for suf, n in MACHINE:
        if root.endswith(suf): return n
    return "161"

seen=set(); cells=defaultdict(dict); ndup=0
for root in sys.argv[1:]:
    may = may_of(root)
    for dp,_,fns in os.walk(root):
        for fn in fns:
            m = re.fullmatch(r"fold(\d+)\.json", fn)
            if not m: continue
            parts = os.path.relpath(os.path.join(dp,fn), root).split(os.sep)
            if len(parts) < 3: continue
            try: d = json.load(open(os.path.join(dp,fn)))
            except Exception: continue
            exp = d.get("experiment_name", os.sep.join(parts[:-1]))
            k = (may, exp, parts[-2], int(m.group(1)))
            if k in seen: ndup += 1; continue
            seen.add(k)
            cells[(may, root, parts[:-3] and os.sep.join(parts[:-3]) or "", parts[-2], int(m.group(1)))][parts[-3]] = (d, exp)

# ---- gom theo (phuong phap, backbone) ----
agg = defaultdict(lambda: defaultdict(list))
srcs = defaultdict(set); mays = defaultdict(set)
for key, arms in cells.items():
    if "baseline" not in arms: continue
    b, _ = arms["baseline"]
    for arm, (d, exp) in arms.items():
        if arm == "baseline": continue
        pa = parse_arm(arm)
        if pa is None: continue
        mode, src, tag, opt, lam = pa
        bb = bb_of(f"{exp} {key[1]} {key[2]}")
        gk = (mode, tag, opt, lam, bb)
        for mk, f in MET:
            if d.get(f) is not None and b.get(f) is not None:
                agg[gk][mk].append(d[f] - b[f])
        srcs[gk].add(src); mays[gk].add(key[0])

def stat(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return None
    nz = v[v != 0]; pos = int((nz > 0).sum())
    p = float(binomtest(pos, len(nz), .5).pvalue) if len(nz) else None
    return {"d": round(float(v.mean()),4), "pos": pos, "n": int(len(v)),
            "p": round(p,4) if p is not None else None}

rows=[]
for (mode, tag, opt, lam, bb), v in agg.items():
    r = {"nhanh": mode, "tag": tag, "opt": opt, "lam": lam, "backbone": bb,
         "nguon": len(srcs[(mode,tag,opt,lam,bb)]), "may": sorted(mays[(mode,tag,opt,lam,bb)])}
    for mk,_ in MET: r[mk] = stat(v.get(mk, []))
    r["n"] = r["roc"]["n"] if r["roc"] else 0
    if r["n"] >= 3: rows.append(r)
rows.sort(key=lambda r: (r["nhanh"], r["tag"], r["backbone"]))
sys.stderr.write(f"# {len(cells)} khoi, bo {ndup} o trung, {len(rows)} dong tong hop\n")
print(json.dumps(rows, ensure_ascii=False))
