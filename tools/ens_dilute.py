#!/usr/bin/env python3
"""ens_dilute.py — DOI CHUNG AM cho §25, chay ngoai tuyen, 0 GPU.

Phan bien: "tron hai mo hinh nao cung loi, do la trung binh hoa phuong sai."
Neu dung vay thi tron voi mot mo hinh chuyen giao tu nguon PHA LOANG (lm12/pur12 —
§B.2e do duoc la am, 0/10 fold) phai cho DUNG cai loi nhu tron voi nguon nguyen chat.
Ca hai deu la "mot mo hinh thu hai" da finetune tren cung du lieu dich.

Cac cay pool co CA HAI loai nhanh trong CUNG mot khoi, so voi CUNG mot baseline,
nen phep so sanh ghep cap trong khoi — khong lech may, khong lech fold.

In kem Delta cua chinh nhanh do o alpha=1.0 de doc duoc: loi cua ban tron chi la
bam theo chat luong nhanh, hay vuot len tren no.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ROOTS = sys.argv[1:] or ["results/pool1_t5p", "results/poolcb_codebert",
                         "results_pool1_ntat2", "results_poolcb_158"]
NGUYEN = ("pur100_n930", "lm100_n930")            # 100% CWE dich / 100% khop ngon ngu
LOANG  = ("pur12_n930", "pur25_n930", "lm12_n930", "lm25_n930")

def met(y, p):
    return {"f1@0.5": f1_score(y, (p >= .5).astype(int), average="macro", zero_division=0),
            "roc": roc_auc_score(y, p), "pr": average_precision_score(y, p)}

blocks = defaultdict(dict)
for root in ROOTS:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            m = re.fullmatch(r"fold(\d+)\.json", fn)
            if not m: continue
            parts = os.path.relpath(os.path.join(dp, fn), root).split(os.sep)
            if len(parts) < 3: continue
            with open(os.path.join(dp, fn)) as f: d = json.load(f)
            if "test_probabilities" in d:
                blocks[(root, os.sep.join(parts[:-3]), parts[-2], int(m.group(1)))][parts[-3]] = d

# grp -> block -> {"blend":[..], "pure":[..]}
acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for key, arms in sorted(blocks.items()):
    base = arms.get("baseline")
    if base is None: continue
    y = np.asarray(base["test_labels"]); pb = np.asarray(base["test_probabilities"], float)
    mb = met(y, pb)
    for arm, d in arms.items():
        if arm == "baseline" or d["test_labels"] != base["test_labels"]: continue
        grp = "NGUYEN CHAT" if any(s in arm for s in NGUYEN) else \
              "PHA LOANG"   if any(s in arm for s in LOANG)  else None
        if grp is None: continue
        pt = np.asarray(d["test_probabilities"], float)
        m5 = met(y, 0.5 * pb + 0.5 * pt); m1 = met(y, pt)
        for k in mb:
            acc[grp][key]["blend_" + k].append(m5[k] - mb[k])
            acc[grp][key]["pure_"  + k].append(m1[k] - mb[k])

def line(grp, pre):
    out = []
    for k in ("f1@0.5", "roc", "pr"):
        v = np.array([np.mean(b[pre + k]) for b in acc[grp].values() if b[pre + k]])
        nz = v[v != 0]; pos = int((nz > 0).sum())
        p = binomtest(pos, len(nz), .5).pvalue if len(nz) else float("nan")
        out.append(f"{v.mean():>+9.4f} {pos:>3d}/{len(v):<3d} p={p:<6.4f}")
    return "".join(out)

print(f"# cay: {', '.join(ROOTS)}")
print(f"# NGUYEN CHAT = {NGUYEN} | PHA LOANG = {LOANG}")
print(f"{'':<26}" + "".join(f"{h:>24}" for h in ("F1@0.5", "ROC-AUC", "PR-AUC")))
for grp in ("NGUYEN CHAT", "PHA LOANG"):
    print(f"{grp+' | tron a=0.5':<26}" + line(grp, "blend_"))
    print(f"{grp+' | thuan a=1.0':<26}" + line(grp, "pure_"))
    print(f"{'':<26}" + f"({len(acc[grp])} khoi)")

# hieu ghep cap TRONG khoi: khoi nao co ca hai loai nhanh
common = set(acc["NGUYEN CHAT"]) & set(acc["PHA LOANG"])
print(f"\n# GHEP CAP TRONG KHOI ({len(common)} khoi co ca hai loai nhanh)")
print(f"{'':<26}" + "".join(f"{h:>24}" for h in ("F1@0.5", "ROC-AUC", "PR-AUC")))
for pre, lab in (("blend_", "tron a=0.5"), ("pure_", "thuan a=1.0")):
    out = []
    for k in ("f1@0.5", "roc", "pr"):
        v = np.array([np.mean(acc["NGUYEN CHAT"][b][pre+k]) - np.mean(acc["PHA LOANG"][b][pre+k])
                      for b in sorted(common)])
        nz = v[v != 0]; pos = int((nz > 0).sum())
        p = binomtest(pos, len(nz), .5).pvalue if len(nz) else float("nan")
        out.append(f"{v.mean():>+9.4f} {pos:>3d}/{len(v):<3d} p={p:<6.4f}")
    print(f"{'nguyen - loang | '+lab:<26}" + "".join(out))
