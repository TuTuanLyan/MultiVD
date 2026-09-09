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

# DOT CHAY. Nguoi dung 09/09: "tong hop may cai chay tu hoi optimize-v1 thoi chu?"
# Ranh gioi doc duoc tu moc thoi gian cua chinh cac o:
#   truoc optimize-v1  24-31/08 : m1 (465 o), n1 (258), r1 (89), s42_* (723), l1_t5pe (61)
#   optimize-v1        03/09 -> : sw, n48, opt1, ret1, spd1, int1, pool1, poolcb, e60,
#                                 asam1, asamcb, asamaw, wb, wblend, confirm47, vast47, night48
# Gop hai dot vao mot bang la tron hai giao thuc khac nhau — cac the la trong bang
# (uw, sam1r01, l50, ctl) deu la cua dot cu. Van GIU lai du lieu cu vi cau hinh chot cua
# §7 duoc quyet dinh tren luoi s42; chi tach ra de doc rieng.
PRE = ("m1", "n1", "r1", "l1", "s42", "s42tw")
def dot_of(root, run, exp):
    hay = f"{root}/{run}/{exp}"
    for k in PRE:
        if f"/{k}_" in hay or f"_{k}_" in hay or hay.startswith(f"results_{k}") or f"/{k}/" in hay:
            return "trước optimize-v1"
    return "optimize-v1"
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
        dot = dot_of(key[1], key[2], exp)
        # "TB" = gop MOI backbone. Gop cac Delta DA ghep cap theo fold lai voi nhau —
        # KHONG lay trung binh cua cac trung binh (do la loi CLAUDE.md muc 2 cam).
        # NGUON phai nam trong DINH DANH phuong phap khi no la bien thi nghiem.
        # Bay da mac: gop luoi PHA LOANG (lm12/pur12... co y lam hong, §B.2e am 0/10 fold)
        # chung voi nguon nguyen chat vao mot dong "(goc) adamw" -> dong do cho ROC -0.0052
        # va doc thang se ra ket luan SAI "AdamW am". Tach ra: nguon chuan (4cwe/com/full)
        # gop lam mot; moi pool la mot dong rieng vi pool CHINH LA bien cua thi nghiem do.
        sgrp = "3 nguồn chuẩn" if src in SRCS else (src if src != "-" else "—")
        for gk in ((mode, tag, opt, lam, bb, dot, sgrp), (mode, tag, opt, lam, "TB", dot, sgrp)):
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
for (mode, tag, opt, lam, bb, dot, sgrp), v in agg.items():
    gk = (mode,tag,opt,lam,bb,dot,sgrp)
    r = {"nhanh": mode, "tag": tag, "opt": opt, "lam": lam, "backbone": bb, "dot": dot,
         "sgrp": sgrp, "nguon": len(srcs[gk]), "may": sorted(mays[gk])}
    for mk,_ in MET: r[mk] = stat(v.get(mk, []))
    r["n"] = r["roc"]["n"] if r["roc"] else 0
    if r["n"] >= 3: rows.append(r)
rows.sort(key=lambda r: (r["dot"], r["nhanh"], r["sgrp"], r["tag"], r["backbone"]))
sys.stderr.write(f"# {len(cells)} khoi, bo {ndup} o trung, {len(rows)} dong tong hop\n")
print(json.dumps(rows, ensure_ascii=False))
