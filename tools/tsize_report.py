#!/usr/bin/env python3
"""tsize_report.py — duong cong Δ theo CO TAP TRAIN dich (khoi `tsize`).

    python3 tools/tsize_report.py

Doc `results/sz<N>_<bb>` cho N = 228/152/76 va `results/bridge3_<bb>` cho N = 456 (da co san).
Ghep cap TRONG cung o (cung cay, cung N, cung seed, cung fold): nhanh chuyen giao tru baseline.
Ca hai nhanh nhan CUNG tap con vi cung seed, nen phep so chi doi DUNG MOT bien la co tap train.

In CA BON chi so kem so fold cung dau (CLAUDE.md muc 2b), roi per-CWE tinh lai tu xac suat
tung mau. Nguong loai o khai bao truoc: baseline F1@0.5 < 0.40 => o do KHONG DOC DUOC, bo ra
va bao rieng (ca hai nhanh sap thi Δ lon khong con nghia gi).

Du doan khai bao truoc o records/prediction_2026-09-11_duong_cong_co_dich.md.
"""
import glob, json, os, re, sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

EPS = 1e-12
MIN_BASELINE_F1 = 0.40          # nguong loai o, KHAI BAO TRUOC
MET = [("F1@0.5", "test_macro_f1_at_0.5"), ("F1@val", "test_macro_f1_at_valcal"),
       ("ROC-AUC", "test_roc_auc"), ("PR-AUC", "test_pr_auc")]
CWE = {0: "022", 1: "078", 2: "079", 3: "089"}
# so hang train ky vong theo CWE o tung N (do that tu limit_records, fold 1)
TRAIN_N = {456: {"022": 40, "078": 124, "079": 47, "089": 245},
           228: {"022": 19, "078": 60, "079": 23, "089": 126},
           152: {"022": 9, "078": 37, "079": 15, "089": 91},
           76:  {"022": 7, "078": 18, "079": 8, "089": 43}}


def stat(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return None
    nz = v[np.abs(v) >= EPS]; pos = int((nz > 0).sum()); ties = len(v) - len(nz)
    p = binomtest(pos, len(nz), .5).pvalue if len(nz) else float("nan")
    return v.mean(), pos, len(v), p, ties


def fmt(s):
    if not s: return f"{'-':>22}"
    d, pos, n, p, t = s
    return f"{d:>+8.4f} {pos:>2d}/{n:<2d}{'~%d'%t if t else '  '} p={p:<5.3f}"


def per_cwe(d):
    p = np.asarray(d.get("test_probabilities") or [], float)
    y = np.asarray(d.get("test_labels") or [], int)
    c = np.asarray(d.get("test_cwe_classes") or [], int)
    out = {}
    if not len(p) or len(p) != len(y) or len(y) != len(c): return out
    for k in sorted(set(c.tolist())):
        m = c == k
        if m.sum() < 2: continue
        yk, pk = y[m], p[m]
        out[CWE.get(k, str(k))] = (
            f1_score(yk, (pk >= 0.5).astype(int), average="macro", zero_division=0),
            roc_auc_score(yk, pk) if len(set(yk.tolist())) == 2 else float("nan"))
    return out


def load():
    """{(bb, N, seed, fold): (transfer_json, baseline_json)}

    SEED PHAI nam trong khoa. Ban truoc khoa la (bb, N, fold) nen ba seed de len nhau va
    chi mot seed song sot — bang van in ra binh thuong, khong bao gi ca. Bay "o le bi BO va
    bao ro so o bo" cua CLAUDE.md muc 2b chi chan duoc o THIEU doi chung, khong chan duoc
    o bi GHI DE."""
    cells = {}
    roots = [(456, r) for r in glob.glob("results/bridge3_*")]
    roots += [(int(re.search(r"sz(\d+)_", r).group(1)), r) for r in glob.glob("results/sz*_*")]
    for N, root in roots:
        bb = root.split("_", 1)[1]
        for f in glob.glob(os.path.join(root, "transfer_*_plain_adamw", "seed_*", "fold*.json")):
            fold = int(re.search(r"fold(\d+)", f).group(1))
            seed = int(os.path.basename(os.path.dirname(f)).split("_")[1])
            b = os.path.join(root, "baseline", f"seed_{seed}", f"fold{fold}.json")
            if not os.path.exists(b): continue
            try:
                cells[(bb, N, seed, fold)] = (json.load(open(f)), json.load(open(b)))
            except Exception as e:
                print(f"# hong: {f}: {e}", file=sys.stderr)
    return cells


def main():
    cells = load()
    if not cells:
        print("khong co o nao"); return
    bbs = sorted({k[0] for k in cells})
    Ns = sorted({k[1] for k in cells}, reverse=True)
    print(f"{len(cells)} o ghep cap | backbone: {bbs} | N: {Ns}")

    seeds = sorted({k[2] for k in cells})
    print(f"seed co mat: {seeds}")
    bad = [k for k, (t, b) in cells.items()
           if (b.get("test_macro_f1_at_0.5") or 0) < MIN_BASELINE_F1]
    if bad:
        print(f"\n!! {len(bad)} o BI LOAI (baseline F1@0.5 < {MIN_BASELINE_F1}, nguong khai bao truoc): {bad}")
    keep = {k: v for k, v in cells.items() if k not in set(bad)}

    print("\n=== Δ (chuyen giao − baseline), ghep cap TRONG cung o ===")
    print(f"{'backbone':<10}{'N':>5}{'n':>3}  " + "".join(f"{m:>24}" for m, _ in MET))
    for bb in bbs:
        for N in Ns:
            pr = [(t, b) for (b2, N2, _, _), (t, b) in keep.items() if b2 == bb and N2 == N]
            if not pr: continue
            cols = [stat([(t.get(k) or np.nan) - (b.get(k) or np.nan) for t, b in pr]) for _, k in MET]
            print(f"{bb:<10}{N:>5}{len(pr):>3}  " + "".join(f"  {fmt(s)}" for s in cols))
        print()

    print("=== Δ ROC-AUC theo CWE — cot ky vong la SO HANG TRAIN o N do ===")
    for bb in bbs:
        print(f"\n-- {bb}")
        print(f"{'N':>5}{'n':>3}   " + "".join(f"{'CWE-'+c:>22}" for c in ("022", "078", "079", "089")))
        for N in Ns:
            pr = [(t, b) for (b2, N2, _, _), (t, b) in keep.items() if b2 == bb and N2 == N]
            if not pr: continue
            dd = defaultdict(list)
            for t, b in pr:
                pt, pb = per_cwe(t), per_cwe(b)
                for c in pt:
                    if c in pb and not (np.isnan(pt[c][1]) or np.isnan(pb[c][1])):
                        dd[c].append(pt[c][1] - pb[c][1])
            row = f"{N:>5}{len(pr):>3}   "
            for c in ("022", "078", "079", "089"):
                s = stat(dd.get(c, []))
                tr = TRAIN_N.get(N, {}).get(c, "?")
                row += f"{(f'{s[0]:+.3f} {s[1]}/{s[2]}' if s else '-'):>13}{f'[{tr}]':>9}"
            print(row)

    print("\n=== TRI TUYET DOI (trung binh qua fold) ===")
    print(f"{'backbone':<10}{'N':>5}{'nhanh':>12}{'F1@0.5':>10}{'ROC-AUC':>10}{'PR-AUC':>10}{'best_ep':>9}")
    for bb in bbs:
        for N in Ns:
            pr = [(t, b) for (b2, N2, _, _), (t, b) in keep.items() if b2 == bb and N2 == N]
            if not pr: continue
            for name, idx in (("chuyen giao", 0), ("baseline", 1)):
                ds = [x[idx] for x in pr]
                print(f"{bb:<10}{N:>5}{name:>12}"
                      f"{np.mean([d['test_macro_f1_at_0.5'] for d in ds]):>10.4f}"
                      f"{np.mean([d['test_roc_auc'] for d in ds]):>10.4f}"
                      f"{np.mean([d['test_pr_auc'] for d in ds]):>10.4f}"
                      f"{np.mean([d.get('best_epoch') or np.nan for d in ds]):>9.1f}")
    print("\n=== TACH THEO SEED (ROC-AUC) — hieu ung phai giu dau o TUNG seed ===")
    print(f"{'backbone':<10}{'N':>5}" + "".join(f"{'seed '+str(sd):>18}" for sd in seeds))
    for bb in bbs:
        for N in Ns:
            row = f"{bb:<10}{N:>5}"
            any_cell = False
            for sd in seeds:
                pr = [(t, b) for (b2, N2, s2, _), (t, b) in keep.items()
                      if b2 == bb and N2 == N and s2 == sd]
                if not pr:
                    row += f"{'-':>18}"; continue
                any_cell = True
                d = [(t.get("test_roc_auc") or np.nan) - (b.get("test_roc_auc") or np.nan)
                     for t, b in pr]
                st = stat(d)
                row += f"{f'{st[0]:+.4f} {st[1]}/{st[2]}':>18}"
            if any_cell: print(row)
        print()

    print("\nBac 1 (n=3 fold, seed 42): chi SANG LOC. Phan chac la SO FOLD CUNG DAU, khong phai p.")


if __name__ == "__main__":
    main()
