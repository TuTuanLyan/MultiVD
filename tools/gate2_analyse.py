#!/usr/bin/env python3
"""GATE2: tach BU TRU khoi GIAM PHUONG SAI.

    X = ghep(transfer_42, baseline_42)  - transfer_42      (model 2 KHONG co tri thuc nguon)
    Y = ghep(transfer_42, transfer_7)   - transfer_42      (model 2 CO CUNG tri thuc nguon)
    X - Y  = gia tri rieng cua "biet thu khac nhau"

Ghep cap theo (cay, fold). Nguong o records/prediction_2026-09-16_gate2_bu_tru.md.
"""
import json, sys, statistics as st
from pathlib import Path
from collections import defaultdict
from math import comb
sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion_gate import fuse, four, calib
from leak_groups_pair import arm_tag

M4 = ("F1@0.5", "F1@val", "ROC", "PR")
TR = "transfer_none_com_real"


def sp(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)


def main(roots, mode="logreg"):
    allp = defaultdict(list)
    for root in roots:
        cells = defaultdict(dict)
        for d in sorted(Path(root).glob("*/seed_*/fold*.json")):
            js = json.loads(d.read_text())
            arm = d.parent.parent.name
            tag = "baseline" if arm == "baseline" else arm_tag(arm)
            cells[js["fold"]][f"{tag}@{js['seed']}"] = js
        per = defaultdict(list)
        for f in sorted(cells):
            c = cells[f]
            need = [f"{TR}@42", f"{TR}@7", "baseline@42"]
            if any(k not in c for k in need):
                print(f"  bo {Path(root).name} fold {f}: thieu {[k for k in need if k not in c]}")
                continue
            T, T7, B = c[f"{TR}@42"], c[f"{TR}@7"], c["baseline@42"]
            if not (T["val_labels"] == T7["val_labels"] == B["val_labels"]
                    and T["test_labels"] == T7["test_labels"] == B["test_labels"]):
                print(f"  !! BO fold {f}: thu tu hang khong khop"); continue
            y = T["test_labels"]
            mT = four(y, T["test_probabilities"], calib(T["val_labels"], T["val_probabilities"]))
            pX, tX = fuse(T, B, mode)     # ghep voi baseline  -> tri thuc KHAC
            pY, tY = fuse(T, T7, mode)    # ghep voi transfer  -> tri thuc GIONG
            mX, mY = four(y, pX, tX), four(y, pY, tY)
            for m in M4:
                per[("X", m)].append(mX[m] - mT[m])
                per[("Y", m)].append(mY[m] - mT[m])
                per[("X-Y", m)].append(mX[m] - mY[m])
        if not per:
            continue
        print(f"\n--- {Path(root).name} ({len(per[('X','ROC')])} fold) ---")
        for k, lbl in (("X", "X = ghep(transfer, baseline) - transfer"),
                       ("Y", "Y = ghep(transfer, transfer7) - transfer"),
                       ("X-Y", "X - Y  [gia tri cua 'biet thu khac']")):
            row = []
            for m in M4:
                d = per[(k, m)]
                row.append(f"{st.mean(d):+8.4f} {sum(1 for x in d if x>1e-9)}/{len(d)}")
            print(f"  {lbl:42} " + " | ".join(row))
            for m in M4:
                allp[(k, m)] += per[(k, m)]

    n = len(allp[("X", "ROC")])
    if n == 0:
        print("KHONG CO DU LIEU"); return 2
    print(f"\n=== GOP {n} diem ===")
    print(f"  {'':42} " + " | ".join(f"{m:^20}" for m in M4))
    for k, lbl in (("X", "X = ghep(transfer, baseline) - transfer"),
                   ("Y", "Y = ghep(transfer, transfer7) - transfer"),
                   ("X-Y", "X - Y  [gia tri cua 'biet thu khac']")):
        row = []
        for m in M4:
            d = allp[(k, m)]
            pos = sum(1 for x in d if x > 1e-9)
            row.append(f"{st.mean(d):+8.4f} {pos:>2}/{len(d)} p={sp(pos,len(d)):.2f}")
        print(f"  {lbl:42} " + " | ".join(row))

    d = allp[("X-Y", "ROC")]
    mean, pos = st.mean(d), sum(1 for x in d if x > 1e-9)
    print(f"\nBIEN QUYET DINH  (X-Y)@ROC = {mean:+.4f}  {pos}/{len(d)}")
    if n != 6:
        print(f"  CHUA DU DIEM ({n}/6) — KHONG ap nguong."); return 2
    if mean >= 0.010 and pos >= 4:
        print("  => BU TRU LA THAT (model thu hai mang tri thuc khac thi dang gia hon mot ban sao)"); return 0
    if abs(mean) < 0.005 or pos <= 3:
        print("  => CHI LA GIAM PHUONG SAI — De xuat 1 dong"); return 1
    print("  => KHONG KET LUAN"); return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["results/gate1_codebert", "results/gate1_t5p"]))
