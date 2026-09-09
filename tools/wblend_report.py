#!/usr/bin/env python3
"""wblend_report.py — doc khoi noi suy TRONG SO (run/wblend.sh -> tools/wblend.py).

Moi file JSON la MOT o (nguon, seed, fold) va chua ca duong alpha. Ghep cap theo fold,
khong bao gio lay hieu cua hai trung binh (CLAUDE.md muc 2).

BON cot so sanh, tat ca so voi CUNG mot baseline (alpha=0) trong CUNG file:
  thuan       chuyen giao thuan (alpha=1)
  tron TS@val noi suy TRONG SO, alpha chon tren VAL  <- phep do chinh
  tron TS@0.5 noi suy TRONG SO, alpha=0.5 khai bao truoc
  tron XS@0.5 tron XAC SUAT, alpha=0.5 (phep cua §25, de so truc tiep)

Cot 'tron TS@val - thuan' moi tra loi cau cua khoi: noi suy trong so co them duoc gi
so voi chi dung mo hinh chuyen giao khong. Neu CO thi chi phi suy luan ve mot mo hinh.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

roots = [Path(x) for x in (sys.argv[1:] or ["results/wblend_t5p"])]
files = sorted(f for r in roots for f in r.glob("*.json"))
if not files:
    print(f"khong co file nao trong {', '.join(map(str, roots))}"); sys.exit(1)

rows = defaultdict(list)
alphas_chosen = []
missing = 0
for f in files:
    d = json.loads(f.read_text())
    g = {r["alpha"]: r for r in d["grid"]}
    if 0.0 not in g or 1.0 not in g:
        missing += 1; continue
    base, pure = g[0.0], g[1.0]
    a_val = d.get("alpha_on_val_roc")
    if a_val is None or a_val not in g:
        missing += 1; continue
    alphas_chosen.append(a_val)
    wv, wh = g[a_val], g.get(0.5)
    pb = d.get("prob_blend_0.5", {})
    for met, key in (("F1@0.5", "test_macro_f1"), ("ROC-AUC", "test_roc_auc"),
                     ("PR-AUC", "test_pr_auc")):
        b = base[key]
        rows[("thuan", met)].append(pure[key] - b)
        rows[("tron TS@val", met)].append(wv[key] - b)
        if wh: rows[("tron TS@0.5", met)].append(wh[key] - b)
        pk = {"F1@0.5": "macro_f1", "ROC-AUC": "roc_auc", "PR-AUC": "pr_auc"}[met]
        if pk in pb: rows[("tron XS@0.5", met)].append(pb[pk] - b)
        rows[("TS@val - thuan", met)].append(wv[key] - pure[key])
        if pk in pb: rows[("TS@val - XS@0.5", met)].append(wv[key] - pb[pk])

def fmt(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return f"{'-':>24}"
    nz = v[v != 0]; pos = int((nz > 0).sum())
    p = binomtest(pos, len(nz), .5).pvalue if len(nz) else np.nan
    return f"{v.mean():>+9.4f} {pos:>3d}/{len(v):<3d} p={p:<6.4f}"

print(f"# {len(files)} o, bo {missing} o thieu duong alpha | cay: {', '.join(map(str, roots))}")
if alphas_chosen:
    u, c = np.unique(alphas_chosen, return_counts=True)
    print(f"# alpha chon tren VAL: " + "  ".join(f"{a:.2f}x{n}" for a, n in zip(u, c))
          + f"  | trung binh {np.mean(alphas_chosen):.3f}")
    # goi y chi in khi DUNG voi so lieu — in vo dieu kien thi no thanh cau sai.
    ac = np.asarray(alphas_chosen)
    if np.all(ac == 1.0):
        print("# alpha=1.0 o MOI o => VAL luon chon mo hinh chuyen giao nguyen ban:"
              " noi suy trong so khong dong gop gi.")
    elif np.all(ac == 0.0):
        print("# alpha=0.0 o MOI o => VAL luon chon BASELINE: mo hinh chuyen giao"
              " khong dong gop gi trong khong gian trong so.")
    elif np.all((ac > 0) & (ac < 1)):
        print("# alpha nam HAN trong (0,1) o moi o => ca hai mo hinh cung dong gop.")
print()
print(f"{'so voi baseline':<18}" + "".join(f"{h:>24}" for h in ("F1@0.5", "ROC-AUC", "PR-AUC")))
for lab in ("thuan", "tron TS@val", "tron TS@0.5", "tron XS@0.5"):
    if (lab, "ROC-AUC") in rows:
        print(f"{lab:<18}" + "".join(fmt(rows[(lab, m)]) for m in ("F1@0.5", "ROC-AUC", "PR-AUC")))
print()
print(f"{'ghep cap truc tiep':<18}" + "".join(f"{h:>24}" for h in ("F1@0.5", "ROC-AUC", "PR-AUC")))
for lab in ("TS@val - thuan", "TS@val - XS@0.5"):
    if (lab, "ROC-AUC") in rows:
        print(f"{lab:<18}" + "".join(fmt(rows[(lab, m)]) for m in ("F1@0.5", "ROC-AUC", "PR-AUC")))
