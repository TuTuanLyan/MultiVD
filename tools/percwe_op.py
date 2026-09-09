#!/usr/bin/env python3
"""percwe_op.py — hai CWE hiem duoc gi o DIEM VAN HANH, khong chi o AUC.

VI SAO: §23/§25 phat bieu bang ROC-AUC. Nguoi doc bai muon biet "bat them duoc bao
nhieu lo hong CWE-022/079" — do la RECALL o mot nguong cu the, khong phai dien tich
duoi duong cong. Cau nay moi la "loi the cua phuong phap".

HAI diem van hanh, bao ca hai vi moi cai co the doc sai theo mot kieu:
  (a) nguong 0.5 co dinh — khong hieu chinh gi, khong the ro ri, nhung co the lech
      neu mo hinh khong hieu chinh tot;
  (b) nguong hieu chinh tren VAL cua CHINH BASELINE, dung chung cho ca ba mo hinh —
      "giu nguyen diem van hanh dang trien khai, doi mo hinh thoi". Nguong nay da luu
      san (`val_calibrated_threshold`) nen KHONG phai chon tren test.
Ban tron KHONG co nguong val cua rieng no (o cu khong luu xac suat val) nen (b) la
cach so sanh cong bang duy nhat lam duoc ngoai tuyen.

Don vi doc lap van la KHOI (cay, run, seed, fold) — trung binh cac nhanh trong khoi
truoc roi moi dem dau, giong tools/ensemble2.py.
"""
import argparse, json, os, re
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest

CWEN = {0: "CWE-022", 1: "CWE-078", 2: "CWE-079", 3: "CWE-089"}

def rec_prec(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return rec, prec, tp, tp + fn

ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--only", default="latent_bottleneck")
ap.add_argument("--exclude", default="(lm12|lm25|pur12|pur25|pur50|r4p0|r8p0|_e60)")
ap.add_argument("--min-p1val", type=float, default=0.40)
a = ap.parse_args()
only = re.compile(a.only) if a.only else None
excl = re.compile(a.exclude) if a.exclude else None

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

# store[(thr_kind, cwe, model, metric)][block] = [gia tri tung nhanh]
st = defaultdict(lambda: defaultdict(list))
npos = defaultdict(list)   # so hang DUONG cua moi CWE, dem 1 lan/khoi tu baseline
for key, arms in sorted(blocks.items()):
    base = arms.get("baseline")
    if base is None: continue
    y = np.asarray(base["test_labels"]); pb = np.asarray(base["test_probabilities"], float)
    cw = np.asarray(base.get("test_cwe_classes") or [])
    if not len(cw): continue
    thr_val = base.get("val_calibrated_threshold", 0.5)
    for c in sorted(set(cw.tolist())):
        npos[c].append(int((y[cw == c] == 1).sum()))
    for arm, d in sorted(arms.items()):
        if arm == "baseline" or d["test_labels"] != base["test_labels"]: continue
        if only and not only.search(arm): continue
        if excl and excl.search(arm): continue
        if (d.get("phase1_val_macro_f1") or 0) < a.min_p1val: continue
        pt = np.asarray(d["test_probabilities"], float)
        pe = (1 - a.alpha) * pb + a.alpha * pt
        for tk, thr in (("0.5", 0.5), ("val-baseline", thr_val)):
            for c in sorted(set(cw.tolist())):
                msk = cw == c
                for mdl, pp in (("baseline", pb), ("chuyen giao", pt), ("tron", pe)):
                    r, pr, tp, pos = rec_prec(y[msk], pp[msk], thr)
                    st[(tk, c, mdl, "rec")][key].append(r)
                    st[(tk, c, mdl, "prec")][key].append(pr)
                    st[(tk, c, mdl, "tp")][key].append(tp)

def blk(k):
    v = [np.nanmean(x) for x in st[k].values() if len(x)]
    return np.array([x for x in v if not np.isnan(x)])

def dsign(k1, k0):
    v1, v0 = blk(k1), blk(k0)
    n = min(len(v1), len(v0)); d = v1[:n] - v0[:n]
    nz = d[d != 0]
    p = binomtest(int((nz > 0).sum()), len(nz), .5).pvalue if len(nz) else np.nan
    return d.mean(), int((nz > 0).sum()), len(nz), p

print(f"# alpha={a.alpha} | only={a.only} | min-p1val={a.min_p1val}")
print(f"# so KHOI: {len(set().union(*[set(st[k]) for k in st]) ) if st else 0}")
for tk in ("0.5", "val-baseline"):
    print(f"\n===== nguong = {tk} =====")
    print(f"{'CWE':<10}{'hang duong/fold':>16}{'recall base':>13}{'recall tron':>13}"
          f"{'D recall (tron-base)':>30}{'D prec':>26}")
    for c in sorted(CWEN):
        if (tk, c, "baseline", "rec") not in st: continue
        rb, rt = blk((tk, c, "baseline", "rec")).mean(), blk((tk, c, "tron", "rec")).mean()
        dr = dsign((tk, c, "tron", "rec"), (tk, c, "baseline", "rec"))
        dp = dsign((tk, c, "tron", "prec"), (tk, c, "baseline", "prec"))
        print(f"{CWEN[c]:<10}{np.mean(npos[c]):>16.1f}"
              f"{rb:>13.3f}{rt:>13.3f}"
              f"{dr[0]:>+16.3f} {dr[1]:>3d}/{dr[2]:<3d} p={dr[3]:<5.3f}"
              f"{dp[0]:>+12.3f} {dp[1]:>3d}/{dp[2]:<3d}")
