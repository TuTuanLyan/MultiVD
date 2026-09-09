#!/usr/bin/env python3
"""build_table.py — trich TOAN BO ket qua thanh bang JSON cho trang tuong tac.

MOT DONG = mot phep so sanh (nhanh vs doi chung), ghep cap theo fold trong CUNG cay,
CUNG seed (CLAUDE.md muc 2 + muc 4). Khong bao gio lay hieu cua hai trung binh.

BA cho da sua sau ban dau:
 1. Dung CHI SO DA LUU DANG SO (test_macro_f1_at_0.5 / _at_valcal / roc / pr) thay vi
    tinh lai tu `test_probabilities`. Truong xac suat chi co tu 07/09 nen ban dau bo
    het luoi `s42` — chinh la luoi DA CHOT cua §7, 700+ o.
 2. KHU TRUNG theo (may, experiment_name, seed, fold). Vai cay chua CA HAI bo cuc
    (results_asam1_ntat co 95 o phang VA 95 o long y het) nen khong khu thi mot o bi
    dem thanh hai. Hai MAY khac nhau chay cung cau hinh thi KHONG phai trung — do la
    ban lap doc lap, phai giu ca hai.
 3. Doc backbone tu `experiment_name` cua chinh o, khong doan tu duong dan.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest

# THU TU QUAN TRONG: t5pe truoc t5p truoc t5 — chuoi ngan la HAU TO cua chuoi dai,
# xep sai thi "l1_t5pe" bi doc thanh t5p. `t5` la CodeT5 goc (khoi m1), khac codet5p.
BACKBONES = ("t5pe", "t5p", "codebert", "unixcoder", "roberta", "t5")
MACHINE = [("_ntat2", "ntat2"), ("_ntat", "ntat"), ("_158", "158"), ("_161", "161")]
SRCS = ("4cwe", "com", "full")
MET = [("f1", "test_macro_f1_at_0.5"), ("f1v", "test_macro_f1_at_valcal"),
       ("roc", "test_roc_auc"), ("pr", "test_pr_auc")]

def machine_of(root):
    for suf, name in MACHINE:
        if root.endswith(suf): return name
    return "161"

def backbone_of(exp, fallback=""):
    hay = f"{exp} {fallback}"
    for b in BACKBONES:
        if b in hay: return b
    return "?"

def parse_arm(arm):
    if arm == "baseline": return ("baseline", "-", "-", "-")
    m = re.match(r"transfer_(none|cwe|latent_bottleneck|latent_proto)_?(.*)$", arm)
    if not m: return (arm, "-", "-", "-")
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
    return (mode, src, rest or "(gốc)", opt, lam)

def stat(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return None
    nz = v[v != 0]; pos = int((nz > 0).sum())
    p = float(binomtest(pos, len(nz), .5).pvalue) if len(nz) else None
    return {"d": round(float(v.mean()), 4), "pos": pos, "n": int(len(v)),
            "p": round(p, 4) if p is not None else None}

# ---------- doc + khu trung ----------
seen = set(); cells = defaultdict(dict); n_dup = 0; n_read = 0
for root in sys.argv[1:]:
    may = machine_of(root)
    for dp, _, fns in os.walk(root):
        for fn in fns:
            m = re.fullmatch(r"fold(\d+)\.json", fn)
            if not m: continue
            parts = os.path.relpath(os.path.join(dp, fn), root).split(os.sep)
            if len(parts) < 3: continue
            try:
                with open(os.path.join(dp, fn)) as f: d = json.load(f)
            except Exception: continue
            n_read += 1
            exp = d.get("experiment_name", os.sep.join(parts[:-1]))
            k = (may, exp, parts[-2], int(m.group(1)))
            if k in seen: n_dup += 1; continue
            seen.add(k)
            d["_may"] = may; d["_bb"] = backbone_of(exp, root)
            cells[(may, exp.rsplit("/", 1)[0] if "/" in exp else root,
                   parts[-2], int(m.group(1)))][parts[-3]] = d

# ---------- ghep cap ----------
groups = defaultdict(lambda: defaultdict(list))
for (may, run, seed, fold), arms in cells.items():
    for a in arms:
        if a == "baseline": continue
        pa = parse_arm(a)
        pairs = [("baseline", "baseline")] if "baseline" in arms else []
        for b in arms:
            if b == a or b == "baseline": continue
            pb = parse_arm(b)
            if pb[0] != pa[0] or pb[3] != pa[3] or pb[4] != pa[4]: continue
            if pa[1] == pb[1] and pb[2] in ("r0", "aw_r0", "plain", "(gốc)") and pa[2] != pb[2]:
                pairs.append((b, pb[2]))                       # cung nguon, doi chung la muc goc
            elif pa[2] == pb[2] and pb[1] in ("lm100_n930", "pur100_n930") and pa[1] != pb[1]:
                pairs.append((b, pb[1]))                       # cung tag, doi chung la pool nguyen chat
        for bn, blabel in pairs:
            A, B = arms[a], arms[bn]
            key = (may, run, seed, a, bn, blabel)
            for k, f in MET:
                if A.get(f) is not None and B.get(f) is not None:
                    groups[key][k].append(A[f] - B[f])
            if A.get("phase1_val_macro_f1"): groups[key]["p1"].append(A["phase1_val_macro_f1"])
            groups[key]["xs"].append(1 if "test_probabilities" in A else 0)

# ---------- GOP: bo seed va nguon khoi khoa de co dong n=15 (bac 3) ----------
# Bang chi co dong n=5 thi khong bao gio thay duoc phat bieu bac 3 — ma do moi la con so
# dua vao bai. Gop TRONG CUNG may + cung khoi + cung nhanh/cau hinh + cung doi chung;
# KHONG bao gio gop qua may (CLAUDE.md muc 4).
# CHI gop qua NGUON khi cac nguon la BA NGUON CHUAN (4cwe/com/full) — do dung la cach
# §21/§24 bao n=15. Gop ca 12 pool `pur*`/`lm*` vao mot so la VO NGHIA: chung la cac muc
# pha loang KHAC NHAU co chu dich, khong phai lat cat cua cung mot thu.
agg = defaultdict(lambda: defaultdict(list))
agg_src = defaultdict(set)
for (may, run, seed, a, bn, blabel), v in groups.items():
    mode, src, tag, opt, lam = parse_arm(a)
    if src not in SRCS and src != "-":
        k = (may, run, mode, tag, opt, lam, blabel, src)     # pool: gop qua SEED thoi
    else:
        k = (may, run, mode, tag, opt, lam, blabel, "")      # nguon chuan: gop ca nguon
    agg_src[k].add(src)
    for kk, vv in v.items(): agg[k][kk].extend(vv)

rows = []
for (may, run, mode, tag, opt, lam, blabel, fixed_src), v in agg.items():
    r = {"khoi": re.sub(r"^results_?|_(ntat2|ntat|158|161)$", "", run).strip("/_") or "s42",
         "may": may, "backbone": backbone_of(run, tag), "nhanh": mode, "tag": tag,
         "nguon": fixed_src or "(3 nguồn)", "opt": opt, "lam": lam, "seed": "(gộp)",
         "doi_chung": blabel,
         "pham_vi": "gộp seed" + ("" if fixed_src else "+nguồn"),
         "p1val": round(float(np.mean(v["p1"])), 4) if v.get("p1") else None,
         "xs": int(sum(v.get("xs", [0])))}
    for k, _ in MET: r[k] = stat(v.get(k, []))
    r["n"] = r["roc"]["n"] if r["roc"] else 0
    if r["n"] > 5: rows.append(r)      # chi giu dong GOP khi no thuc su rong hon mot lat

for (may, run, seed, a, bn, blabel), v in groups.items():
    mode, src, tag, opt, lam = parse_arm(a)
    bb = backbone_of(run, a)
    r = {"khoi": re.sub(r"^results_?|_(ntat2|ntat|158|161)$", "", run).strip("/_") or "s42",
         "may": may, "backbone": bb, "nhanh": mode, "tag": tag, "nguon": src,
         "opt": opt, "lam": lam, "seed": seed.replace("seed_", ""), "doi_chung": blabel,
         "pham_vi": "1 seed × 1 nguồn",
         "p1val": round(float(np.mean(v["p1"])), 4) if v.get("p1") else None,
         "xs": int(sum(v.get("xs", [0]))) }
    for k, _ in MET: r[k] = stat(v.get(k, []))
    r["n"] = r["roc"]["n"] if r["roc"] else 0
    if r["n"]: rows.append(r)

rows.sort(key=lambda r: (-r["n"], r["khoi"], r["nhanh"], r["tag"]))
sys.stderr.write(f"# doc {n_read} o, bo {n_dup} o TRUNG (cung may+experiment+seed+fold), "
                 f"{len(cells)} khoi, {len(rows)} dong so sanh\n")
print(json.dumps(rows, ensure_ascii=False))
