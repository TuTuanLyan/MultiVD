#!/usr/bin/env python3
"""ens_leak.py — §25 co song sot phep kiem RO RI khong?

VI SAO BAT BUOC: bo `sven_python_folds_norm` chia theo TUNG DONG nen ~16% hang test co
ban doi nghich gan trung trong TRAIN. Phep kiem nay da tung DANH SAP mot ket luan cua
chinh du an (§21.1: loi cua ASAM rho=2.0 hoa ra chu yeu nam o nhom `train`, con nhom
`none` chiem 73% du lieu thi duoi san nhieu). Neu loi cua phep TRON cung don vao nhom
`train` thi §25 khong phai la khai quat hoa, ma la hoc vet.

Nhom `none` (~73% hang, khong co ban doi nghich nao) la nhom PHAI duong thi ket luan
moi dung. Don vi doc lap van la KHOI, giong tools/ensemble2.py.
"""
import argparse, json, os, re
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

GR = json.load(open("data/leak_groups.json"))
BACKBONES = ("t5pe", "t5p", "codebert", "unixcoder", "roberta")

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

ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--only", default="latent_bottleneck")
ap.add_argument("--exclude", default="(lm12|lm25|pur12|pur25|pur50|r4p0|r8p0|_e60)")
ap.add_argument("--min-p1val", type=float, default=0.40)
a = ap.parse_args()
only, excl = re.compile(a.only), re.compile(a.exclude)

blocks = defaultdict(dict)
for root in a.roots:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            m = re.fullmatch(r"fold(\d+)\.json", fn)
            if not m: continue
            parts = os.path.relpath(os.path.join(dp, fn), root).split(os.sep)
            if len(parts) < 3: continue
            with open(os.path.join(dp, fn)) as f: d = json.load(f)
            if "test_probabilities" in d:
                blocks[(root, os.sep.join(parts[:-3]), parts[-2], int(m.group(1)))][parts[-3]] = d

# store[(nhom, mo hinh, chi so)][khoi] = [gia tri tung nhanh]
st = defaultdict(lambda: defaultdict(list))
nrow = defaultdict(list)
nskip_len = 0
for key, arms in sorted(blocks.items()):
    base = arms.get("baseline")
    if base is None: continue
    fold = str(key[3])
    lab = GR.get(fold)
    if lab is None: continue
    y = np.asarray(base["test_labels"]); pb = np.asarray(base["test_probabilities"], float)
    # BAY DA MAC: nhan `val` KHONG phai "hang nay thuoc val" ma la "ban doi nghich cua
    # hang test nay nam trong VAL". Danh sach co DUNG mot muc cho moi hang TEST. Loc bo
    # `val` lam lech do dai -> 100/100 khoi bi bo im lang. Dung nguyen danh sach.
    g = np.asarray(lab)
    if len(g) != len(y):
        nskip_len += 1; continue
    for arm, d in sorted(arms.items()):
        if arm == "baseline" or d["test_labels"] != base["test_labels"]: continue
        if not only.search(arm) or excl.search(arm): continue
        if (d.get("phase1_val_macro_f1") or 0) < a.min_p1val: continue
        pt = np.asarray(d["test_probabilities"], float)
        pe = (1 - a.alpha) * pb + a.alpha * pt
        for grp in ("train", "val", "test", "none", "TAT CA"):
            m = np.ones(len(y), bool) if grp == "TAT CA" else (g == grp)
            if m.sum() < 8 or len(set(y[m].tolist())) < 2: continue
            mb = met(y[m], pb[m])
            for name, pp in (("tron", pe), ("chuyen giao", pt)):
                mm = met(y[m], pp[m])
                for k in mb: st[(grp, name, k)][key].append(mm[k] - mb[k])
            nrow[grp].append(int(m.sum()))

def blk(k): return [float(np.nanmean(v)) for v in st[k].values() if len(v)]

print(f"# alpha={a.alpha} | o bo vi so hang khong khop nhan nhom: {nskip_len}")
print(f"# Delta so voi BASELINE tren cung nhom hang. Moi KHOI mot so.\n")
for grp in ("TAT CA", "train", "val", "test", "none"):
    if (grp, "tron", "roc") not in st: continue
    n = np.mean(nrow[grp]) if nrow[grp] else 0
    print(f"===== nhom {grp}  (~{n:.0f} hang/fold, {len(st[(grp,'tron','roc')])} khoi) =====")
    print(f"{'':<14}" + "".join(f"{h:>24}" for h in ("F1@0.5", "ROC-AUC", "PR-AUC")))
    for name in ("tron", "chuyen giao"):
        print(f"{name:<14}" + "".join(fmt(blk((grp, name, k))) for k in ("f1@0.5", "roc", "pr")))
    print()
