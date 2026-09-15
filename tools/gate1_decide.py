#!/usr/bin/env python3
"""Quyet dinh LEO BAC cho khoi gate1, theo DUNG nguong da ghi truoc.

    python3 tools/gate1_decide.py results/gate1_codebert results/gate1_t5p

Bien quyet dinh (records/prediction_2026-09-15_gate1_late_fusion.md):
    D_nguon = [ghep(base42, transfer) - base42] - [ghep(base42, base7) - base42]
            =  ghep(base42, transfer) - ghep(base42, base7)          (tren ROC-AUC)
ghep cap theo (cay, fold). Cong `logreg`, hoc tren val cua chinh fold do.

Bac 1 (n=3, 6 diem): LEO len n=5 neu D_nguon >= +0.010 VA >= 4/6.
Bac 2 (n=5, 10 diem): cong bo theo bang trong file khai bao.

Ma thoat: 0 = dat (leo bac / cong bo), 1 = khong dat (dung), 2 = thieu du lieu.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import statistics as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion_gate import fuse, four, calib        # noqa: E402
from leak_groups_pair import arm_tag                  # noqa: E402

M4 = ("F1@0.5", "F1@val", "ROC", "PR")
TRANSFER = "transfer_none_com_real"


def main(roots):
    pts = defaultdict(list)
    nfold = 0
    for root in roots:
        cells = defaultdict(dict)
        for d in sorted(Path(root).glob("*/seed_*/fold*.json")):
            js = json.loads(d.read_text())
            arm = d.parent.parent.name
            tag = "baseline" if arm == "baseline" else arm_tag(arm)
            cells[js["fold"]][f"{tag}@{js['seed']}"] = js
        for f in sorted(cells):
            c = cells[f]
            need = ["baseline@42", "baseline@7", f"{TRANSFER}@42"]
            if any(k not in c for k in need):
                print(f"  bo {Path(root).name} fold {f}: thieu {[k for k in need if k not in c]}")
                continue
            A, S, B = c["baseline@42"], c["baseline@7"], c[f"{TRANSFER}@42"]
            if not (A["val_labels"] == S["val_labels"] == B["val_labels"]
                    and A["test_labels"] == S["test_labels"] == B["test_labels"]):
                print(f"  !! BO {Path(root).name} fold {f}: thu tu hang khong khop")
                continue
            nfold += 1
            y = A["test_labels"]
            mA = four(y, A["test_probabilities"], calib(A["val_labels"], A["val_probabilities"]))
            pS, tS = fuse(A, S, "logreg")          # ghep ENSEMBLE THUAN
            pB, tB = fuse(A, B, "logreg")          # ghep NGUON
            mS, mB = four(y, pS, tS), four(y, pB, tB)
            for m in M4:
                pts[("src_vs_base", m)].append(mB[m] - mA[m])
                pts[("ens_vs_base", m)].append(mS[m] - mA[m])
                pts[("D_nguon", m)].append(mB[m] - mS[m])

    if nfold == 0:
        print("KHONG CO DU LIEU GHEP DUOC"); return 2
    print(f"\n=== gate1 · {nfold} diem ghep cap · cong logreg ===")
    print(f"{'phep so':34} " + " ".join(f"{m:>18}" for m in M4))
    for k, lbl in (("src_vs_base", "ghep(base,transfer) − base"),
                   ("ens_vs_base", "ghep(base,base7)    − base  [ĐC]"),
                   ("D_nguon",     "D_NGUON = ghep nguon − ĐC")):
        row = []
        for m in M4:
            d = pts[(k, m)]
            pos = sum(1 for x in d if x > 1e-9)
            row.append(f"{st.mean(d):+8.4f} {pos:>2}/{len(d)}")
        print(f"{lbl:34} " + " ".join(f"{c:>18}" for c in row))

    d = pts[("D_nguon", "ROC")]
    mean = st.mean(d); pos = sum(1 for x in d if x > 1e-9); n = len(d)
    print(f"\nBIEN QUYET DINH  D_nguon@ROC = {mean:+.4f}  {pos}/{n}")
    if n <= 6:
        ok = mean >= 0.010 and pos >= 4
        print(f"  Bac 1 (n=3): nguong >= +0.010 VA >= 4/6  ->  {'DAT, LEO LEN n=5' if ok else 'KHONG DAT, DUNG'}")
    else:
        if mean >= 0.010 and pos >= 8:
            print("  Bac 2 (n=5): DAT — De xuat 1 dung, du dieu kien len bac 3"); ok = True
        elif mean <= 0.003 or pos <= 5:
            print("  Bac 2 (n=5): HONG — ensemble thuan cho ngan ay, khong phai dong gop transfer"); ok = False
        else:
            print("  Bac 2 (n=5): KHONG KET LUAN — phai them seed"); ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["results/gate1_codebert", "results/gate1_t5p"]))
