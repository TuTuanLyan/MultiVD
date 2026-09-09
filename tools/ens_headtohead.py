#!/usr/bin/env python3
"""ens_headtohead.py — phep so SAC NHAT cho doi chung cua §25.

§25.8 so hai thu tren hai tap khac nhau:
   tron(base42, chuyen_giao) vs base42      — tren 87 khoi
   tron(base42, base7)       vs base42      — tren 32 cap
Hai con so do ghep cap voi CUNG mot moc nhung KHONG ghep cap voi NHAU.

Day la ban ghep cap truc tiep: TRONG CUNG mot (cay, fold), lay
   A = tron(base42, chuyen_giao_42)
   B = tron(base42, base_khac_seed)
roi do A - B. Cung baseline, cung fold, cung may => moi thu triet tieu tru DUY NHAT
cau hoi "mo hinh thu hai la ban chuyen giao hay la mot ban chay lai".
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

CWEN = {0: "CWE-022", 1: "CWE-078", 2: "CWE-079", 3: "CWE-089"}

def met(y, p):
    two = len(set(y.tolist())) > 1
    return {"f1": f1_score(y, (p >= .5).astype(int), average="macro", zero_division=0),
            "roc": roc_auc_score(y, p) if two else np.nan,
            "pr": average_precision_score(y, p) if two else np.nan}

def fmt(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return f"{'-':>24}"
    nz = v[v != 0]; pos = int((nz > 0).sum())
    p = binomtest(pos, len(nz), .5).pvalue if len(nz) else np.nan
    return f"{v.mean():>+9.4f} {pos:>3d}/{len(v):<3d} p={p:<6.4f}"

# gom: (cay, run, fold) -> {"base42":o, "base_khac":[o], "transfer":[o]}
blk = defaultdict(lambda: {"b42": None, "bo": [], "tr": []})
for root in sys.argv[1:]:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            m = re.fullmatch(r"fold(\d+)\.json", fn)
            if not m: continue
            fp = os.path.join(dp, fn)
            parts = os.path.relpath(fp, root).split(os.sep)
            if len(parts) < 3: continue
            arm, seed = parts[-3], parts[-2]
            with open(fp) as f: d = json.load(f)
            if "test_probabilities" not in d: continue
            k = (root, os.sep.join(parts[:-3]), int(m.group(1)))
            if arm == "baseline":
                (blk[k].__setitem__("b42", d) if seed == "seed_42" else blk[k]["bo"].append(d))
            elif seed == "seed_42" and "latent_bottleneck" in arm:
                blk[k]["tr"].append((arm, d))

agg = defaultdict(list); cwe = defaultdict(list); npair = 0; nblk = 0
for k, v in sorted(blk.items()):
    b, others, trs = v["b42"], v["bo"], v["tr"]
    if b is None or not others or not trs: continue
    y = np.asarray(b["test_labels"]); pb = np.asarray(b["test_probabilities"], float)
    cw = np.asarray(b.get("test_cwe_classes") or [])
    ok = lambda d: d["test_labels"] == b["test_labels"]
    others = [o for o in others if ok(o)]; trs = [(n, t) for n, t in trs if ok(t)]
    if not others or not trs: continue
    nblk += 1
    # B: trung binh cua CAC ban tron voi baseline khac seed (co the co 2 seed)
    Bs = [0.5 * pb + 0.5 * np.asarray(o["test_probabilities"], float) for o in others]
    mB = [met(y, x) for x in Bs]
    for _, t in trs:
        A = 0.5 * pb + 0.5 * np.asarray(t["test_probabilities"], float)
        mA = met(y, A); npair += 1
        for kk in ("f1", "roc", "pr"):
            agg[kk].append(mA[kk] - float(np.nanmean([m[kk] for m in mB])))
        if len(cw) == len(y):
            for c in sorted(set(cw.tolist())):
                msk = cw == c
                if len(set(y[msk].tolist())) < 2: continue
                ra = roc_auc_score(y[msk], A[msk])
                rb = float(np.nanmean([roc_auc_score(y[msk], x[msk]) for x in Bs]))
                cwe[c].append(ra - rb)

print(f"# {nblk} khoi co DU ca ba thanh phan (base seed42 + base khac seed + nhanh chuyen giao)")
print(f"# {npair} cap ghep truc tiep\n")
print("A = tron(base42, chuyen giao)   B = tron(base42, base khac seed)   -> in A - B")
print(f"{'':<10}" + "".join(f"{h:>24}" for h in ("F1@0.5", "ROC-AUC", "PR-AUC")))
print(f"{'A - B':<10}" + "".join(fmt(agg[k]) for k in ("f1", "roc", "pr")))
if cwe:
    print(f"\n{'A - B':<10}" + "".join(f"{CWEN.get(c, c)+' dROC':>24}" for c in sorted(cwe)))
    print(f"{'':<10}" + "".join(fmt(cwe[c]) for c in sorted(cwe)))
