#!/usr/bin/env python3
"""ens_strat.py — phep tron chi CUU ban chuyen giao yeu, hay no bo tro that?

Neu loi cua phep tron chi la keo mot mo hinh kem ve phia baseline thi no khong phai
"bo tro", chi la chinh quy hoa. Cach tam thuong de kiem — chia khoi theo Delta cua
chinh nhanh chuyen giao — BI LECH: chon nhom theo x roi do y=f(x) tren CUNG du lieu
test la hoi quy ve trung binh, nhom x<0 se tu dong cho y>0.

Nen chia theo mot dai luong DOC LAP HOAN TOAN voi tap test dich: **val cua Pha 1**
(`phase1_val_macro_f1`, do tren tap val cua chinh Pha 1). Neu phep tron van duong o
CA HAI nua tren/duoi trung vi thi no khong phai chi cuu ban yeu.

Chia them theo NGUON — cung la tieu chi doc lap voi ket qua test.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ONLY = re.compile("latent_bottleneck")
EXCL = re.compile("(lm12|lm25|pur12|pur25|pur50|r4p0|r8p0|_e60)")

def met(y, p):
    o = {"f1@0.5": f1_score(y, (p >= .5).astype(int), average="macro", zero_division=0)}
    two = len(set(y.tolist())) > 1
    o["roc"] = roc_auc_score(y, p) if two else np.nan
    o["pr"] = average_precision_score(y, p) if two else np.nan
    return o

def fmt(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return f"{'-':>24}"
    nz = v[v != 0]; pos = int((nz > 0).sum())
    p = binomtest(pos, len(nz), .5).pvalue if len(nz) else np.nan
    return f"{v.mean():>+9.4f} {pos:>3d}/{len(v):<3d} p={p:<6.4f}"

blocks = defaultdict(dict)
for root in sys.argv[1:]:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            m = re.fullmatch(r"fold(\d+)\.json", fn)
            if not m: continue
            parts = os.path.relpath(os.path.join(dp, fn), root).split(os.sep)
            if len(parts) < 3: continue
            with open(os.path.join(dp, fn)) as f: d = json.load(f)
            if "test_probabilities" in d:
                blocks[(root, os.sep.join(parts[:-3]), parts[-2], int(m.group(1)))][parts[-3]] = d

# thu thap tung NHANH kem phase1_val va nguon, roi moi gop ve khoi
recs = []
for key, arms in sorted(blocks.items()):
    base = arms.get("baseline")
    if base is None: continue
    y = np.asarray(base["test_labels"]); pb = np.asarray(base["test_probabilities"], float)
    mb = met(y, pb)
    for arm, d in sorted(arms.items()):
        if arm == "baseline" or d["test_labels"] != base["test_labels"]: continue
        if not ONLY.search(arm) or EXCL.search(arm): continue
        p1 = d.get("phase1_val_macro_f1")
        if p1 is None or p1 < 0.40: continue
        pt = np.asarray(d["test_probabilities"], float)
        m5 = met(y, .5 * pb + .5 * pt); m1 = met(y, pt)
        src = next((s for s in ("4cwe", "com", "full") if f"_{s}_" in arm), "?")
        recs.append({"key": key, "p1": p1, "src": src,
                     **{f"blend_{k}": m5[k] - mb[k] for k in mb},
                     **{f"pure_{k}": m1[k] - mb[k] for k in mb}})

p1s = np.array([r["p1"] for r in recs]); med = float(np.median(p1s))
print(f"# {len(recs)} nhanh | trung vi phase1_val = {med:.4f}")
print("# 'tron - thuan' = phep tron THEM duoc gi so voi chuyen giao thuan (ghep cap trong khoi)\n")

def show(title, sel):
    sub = [r for r in recs if sel(r)]
    if not sub: return
    byblk = defaultdict(list)
    for r in sub: byblk[r["key"]].append(r)
    print(f"===== {title}  ({len(sub)} nhanh / {len(byblk)} khoi) =====")
    print(f"{'':<16}" + "".join(f"{h:>24}" for h in ("F1@0.5", "ROC-AUC", "PR-AUC")))
    for lab, pre in (("tron vs base", "blend_"), ("thuan vs base", "pure_")):
        print(f"{lab:<16}" + "".join(
            fmt([np.mean([r[pre + k] for r in v]) for v in byblk.values()])
            for k in ("f1@0.5", "roc", "pr")))
    print(f"{'tron - thuan':<16}" + "".join(
        fmt([np.mean([r["blend_" + k] - r["pure_" + k] for r in v]) for v in byblk.values()])
        for k in ("f1@0.5", "roc", "pr")))
    print()

show("Pha 1 val TREN trung vi (chuyen giao manh)", lambda r: r["p1"] >= med)
show("Pha 1 val DUOI trung vi (chuyen giao yeu)", lambda r: r["p1"] < med)
for s in ("4cwe", "com", "full"):
    show(f"nguon {s}", lambda r, s=s: r["src"] == s)
