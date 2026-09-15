#!/usr/bin/env python3
"""A.3 cua DE_XUAT_1: doc PHAN BO g THEO CWE + he so cong theo fold — CHI de kiem co che.

    python3 tools/gate_readout.py --a baseline --b <tag> <cay ket qua...>

Ba thu in ra:
  1. g cua cong HANG theo tung fold   — cong nghieng ve SR hay ve TR?
  2. he so cong LOGREG theo tung fold — w_B/(w_A+w_B) la muc dua vao TR
  3. g khop RIENG tren tung CWE       — day la "phan bo g theo CWE" ma A.3 yeu cau

QUAN TRONG: muc 3 khop g tren VAL cua rieng tung CWE. Day la doc CO CHE, **khong** dung
luc huan luyen — neu dung thi la mot cong khac (va la ro ri nhan CWE vao luc suy luan).
Val moi fold 152 hang, chia 4 CWE nen ~38 hang/CWE: rat nhieu, doc xu huong chu khong doc so.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import statistics as st
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion_gate import logit, feats                 # noqa: E402
from leak_groups_pair import arm_tag                      # noqa: E402
from sklearn.linear_model import LogisticRegression       # noqa: E402
from sklearn.metrics import roc_auc_score                 # noqa: E402

GRID = np.linspace(0.0, 1.0, 21)


def best_g(y, pa, pb):
    """g toi uu ROC tren tap dang xet. Tra None neu tap chi co mot lop."""
    if len(set(y)) < 2:
        return None
    sc = [roc_auc_score(y, (1 - g) * pa + g * pb) for g in GRID]
    return float(GRID[int(np.argmax(sc))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()

    CWEV = {k: [json.loads(l)["cwe"] for l in open(f"data/sven_python_folds_norm/fold{k}/val.jsonl")]
            for k in range(1, 6)}
    for root in a.roots:
        cells = defaultdict(dict)
        for d in sorted(Path(root).glob("*/seed_*/fold*.json")):
            js = json.loads(d.read_text())
            arm = d.parent.parent.name
            cells[js["fold"]]["baseline" if arm == "baseline" else arm_tag(arm)] = js
        folds = [f for f in sorted(cells) if a.a in cells[f] and a.b in cells[f]]
        if not folds:
            print(f"\n=== {Path(root).name}: khong ghep duoc cap nao ({a.a} / {a.b})")
            continue
        print(f"\n=== {Path(root).name} · A={a.a} · B={a.b} · {len(folds)} fold ===")
        print(f"{'fold':>5} {'g hang':>8} {'w_A':>8} {'w_B':>8} {'w_|d|':>8} {'dua vao B':>10}")
        per_cwe = defaultdict(list)
        for f in folds:
            A, B = cells[f][a.a], cells[f][a.b]
            yv = A["val_labels"]
            va = np.asarray(A["val_probabilities"]); vb = np.asarray(B["val_probabilities"])
            g = best_g(yv, va, vb)
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(feats(va, vb), yv)
            wA, wB, wD = clf.coef_[0]
            rel = wB / (wA + wB) if abs(wA + wB) > 1e-9 else float("nan")
            print(f"{f:>5} {g:8.2f} {wA:8.3f} {wB:8.3f} {wD:8.3f} {rel:10.2f}")
            cw = CWEV[f]
            for c in sorted(set(cw)):
                idx = [i for i, x in enumerate(cw) if x == c]
                gg = best_g([yv[i] for i in idx], va[idx], vb[idx])
                if gg is not None:
                    per_cwe[c].append((gg, len(idx)))
        print(f"\n  g khop RIENG tren tung CWE (tren VAL, doc co che):")
        print(f"  {'CWE':>9} {'g trung binh':>13} {'cac fold':>28} {'hang/fold':>10}")
        for c, v in sorted(per_cwe.items()):
            gs = [x for x, _ in v]
            print(f"  {c:>9} {st.mean(gs):13.2f} {str([round(x,2) for x in gs]):>28} "
                  f"{st.mean(n for _, n in v):10.1f}")
        print("  g→1 = cong dua han vao TR (transfer) · g→0 = dua han vao SR (baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
