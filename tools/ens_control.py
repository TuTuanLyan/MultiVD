#!/usr/bin/env python3
"""ens_control.py — DOI CHUNG cho phat hien tron: "tron hai mo hinh nao cung loi"?

Phan bien hien nhien: ban tron vuot ca hai dau mut chi vi TRUNG BINH HOA lam giam
phuong sai, khong lien quan gi den chuyen giao. Neu dung vay thi tron BASELINE(seed A)
voi BASELINE(seed B) — hai mo hinh cung kien truc, cung du lieu, chi khac seed —
phai cho DUNG cai loi do.

Ghep cap: cung (cay, run, fold), hai seed khac nhau. Delta do so voi seed A, va
doi xung hoa bang cach chay ca hai chieu.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def met(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return {"f1@0.5": f1_score(y, (p >= .5).astype(int), average="macro", zero_division=0),
            "roc": roc_auc_score(y, p), "pr": average_precision_score(y, p)}

groups = defaultdict(dict)
for root in sys.argv[1:]:
    for dp, _, fns in os.walk(root):
        if os.sep + "baseline" + os.sep not in dp + os.sep: continue
        for fn in fns:
            m = re.fullmatch(r"fold(\d+)\.json", fn)
            if not m: continue
            fp = os.path.join(dp, fn)
            sm = re.search(r"seed_(\d+)", dp)
            run = dp[:dp.index(os.sep + "baseline")]
            with open(fp) as f: d = json.load(f)
            if "test_probabilities" in d:
                groups[(run, int(m.group(1)))][sm.group(1)] = d

res = defaultdict(list)
npair = 0
for key, seeds in sorted(groups.items()):
    ss = sorted(seeds)
    if len(ss) < 2: continue
    for i in range(len(ss)):
        for j in range(len(ss)):
            if i == j: continue
            A, B = seeds[ss[i]], seeds[ss[j]]
            if A["test_labels"] != B["test_labels"]: continue
            y = np.asarray(A["test_labels"])
            pa = np.asarray(A["test_probabilities"], float)
            pb = np.asarray(B["test_probabilities"], float)
            ma = met(y, pa); npair += 1
            for al in (0.25, 0.5, 0.75, 1.0):
                me = met(y, (1 - al) * pa + al * pb)
                for k in ma: res[(al, k)].append(me[k] - ma[k])

print(f"# cap baseline<->baseline (ca hai chieu): {npair}")
print(f"{'alpha':<7}" + "".join(f"{h:>24}" for h in ("F1@0.5", "ROC-AUC", "PR-AUC")))
for al in (0.25, 0.5, 0.75, 1.0):
    line = f"{al:<7.2f}"
    for k in ("f1@0.5", "roc", "pr"):
        v = np.asarray(res[(al, k)]); v = v[~np.isnan(v)]
        nz = v[v != 0]; pos = int((nz > 0).sum())
        p = binomtest(pos, len(nz), .5).pvalue if len(nz) else float("nan")
        line += f"{v.mean():>+9.4f} {pos:>4d}/{len(nz):<4d} p={p:<6.4f}"
    print(line)
