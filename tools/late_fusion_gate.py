#!/usr/bin/env python3
"""DE XUAT 1 — ghep muon (late fusion) hai model da dong cung bang mot cong hoc tren VAL.

    python3 tools/late_fusion_gate.py --a baseline --b <tag> <cay ket qua...>

    p = (1 - g) * p_A + g * p_B

Ba muc cong, dung DUNG thu tu de xuat neu:
  g05     : g = 0.5 co dinh          -> 0 tham so, KHONG doc val, khong the qua khop
  const   : quet g tren val (17 diem) -> 1 tham so
  logreg  : hoi quy logistic tren val, dac trung [logit_A, logit_B, |pA - pB|] -> 4 tham so

VI SAO CHAY DUOC O DAY MA KHONG TON GPU. Moi o ket qua da luu `val_probabilities`,
`val_labels`, `test_probabilities`, `test_labels` (152 hang moi ben). Cong chi can bay nhieu.

BA CHO DE SAI, da chan:
  1. Hai nhanh phai cung THU TU HANG o ca val lan test — neu khong la cong hoc tren nhan
     cua model kia. Kiem bang cach so nguyen list nhan, khong chi so do dai.
  2. Nguong cho mo hinh GHEP phai calib lai tren val cua chinh no, khong dung lai
     `val_calibrated_threshold` cua nhanh A.
  3. Cong hoc tren val roi do tren test; val KHONG duoc dinh vao test. Moi fold hoc lai.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import statistics as st
from math import comb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leak_groups import macro_f1                                   # noqa: E402
from leak_groups_pair import arm_tag                               # noqa: E402
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

EPS = 1e-9


def sp(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def feats(pa, pb):
    return np.stack([logit(pa), logit(pb), np.abs(np.asarray(pa) - np.asarray(pb))], 1)


def calib(y, p):
    """Nguong toi uu macro-F1 tren val — quet chinh cac gia tri xac suat quan sat duoc."""
    best, bt = -1.0, 0.5
    for t in sorted(set(np.round(np.asarray(p), 2).tolist())) or [0.5]:
        f = macro_f1(y, p, t)
        if f > best:
            best, bt = f, t
    return bt


def four(y, p, thr):
    return {"F1@0.5": macro_f1(y, p, 0.5), "F1@val": macro_f1(y, p, thr),
            "ROC": roc_auc_score(y, p), "PR": average_precision_score(y, p)}


def fuse(a, b, mode):
    """Tra ve (p_test_ghep, thr_calib). Cong hoc CHI tren val."""
    va, vb = np.asarray(a["val_probabilities"]), np.asarray(b["val_probabilities"])
    ta, tb = np.asarray(a["test_probabilities"]), np.asarray(b["test_probabilities"])
    yv = a["val_labels"]
    if mode == "g05":
        pv, pt = 0.5 * va + 0.5 * vb, 0.5 * ta + 0.5 * tb
    elif mode == "const":
        gs = np.linspace(0.1, 0.9, 17)
        g = gs[int(np.argmax([roc_auc_score(yv, (1 - g) * va + g * vb) for g in gs]))]
        pv, pt = (1 - g) * va + g * vb, (1 - g) * ta + g * tb
    elif mode == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(feats(va, vb), yv)
        pv = clf.predict_proba(feats(va, vb))[:, 1]
        pt = clf.predict_proba(feats(ta, tb))[:, 1]
    else:
        raise SystemExit("mode la g05 | const | logreg")
    return pt, calib(yv, pv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--a", required=True, help="model A (thuong la baseline)")
    ap.add_argument("--b", required=True, help="model B")
    ap.add_argument("--modes", default="g05,const,logreg")
    a = ap.parse_args()

    cells = defaultdict(dict)
    for root in a.roots:
        root = Path(root)
        for d in sorted(root.glob("*/seed_*/fold*.json")):
            arm = d.parent.parent.name
            tag = "baseline" if arm == "baseline" else arm_tag(arm)
            js = json.loads(d.read_text())
            if "val_probabilities" not in js or "test_probabilities" not in js:
                continue
            cells[(str(root), js["seed"], js["fold"])][tag] = js

    keys = [k for k, v in sorted(cells.items()) if a.a in v and a.b in v]
    if not keys:
        print("khong ghep duoc cap nao — kiem lai --a/--b")
        return 1

    res = {m: defaultdict(list) for m in a.modes.split(",")}
    base = {"A": defaultdict(list), "B": defaultdict(list)}
    for k in keys:
        A, B = cells[k][a.a], cells[k][a.b]
        if A["val_labels"] != B["val_labels"] or A["test_labels"] != B["test_labels"]:
            print(f"  !! BO {k}: thu tu hang val/test khong khop giua hai nhanh")
            continue
        y = A["test_labels"]
        for nm, M in (("A", A), ("B", B)):
            thr = calib(M["val_labels"], M["val_probabilities"])
            for kk, vv in four(y, M["test_probabilities"], thr).items():
                base[nm][kk].append(vv)
        for m in res:
            pt, thr = fuse(A, B, m)
            for kk, vv in four(y, pt, thr).items():
                res[m][kk].append(vv)

    n = len(base["A"]["ROC"])
    M4 = ("F1@0.5", "F1@val", "ROC", "PR")
    print(f"\n=== GHEP  A={a.a}   B={a.b}   ({n} fold) ===")
    print(f"{'nhanh':22} " + " ".join(f"{m:>9}" for m in M4))
    for nm, lbl in (("A", f"A · {a.a}"), ("B", f"B · {a.b}")):
        print(f"{lbl:22} " + " ".join(f"{st.mean(base[nm][m]):9.4f}" for m in M4))
    for m in res:
        print(f"{'ghep · ' + m:22} " + " ".join(f"{st.mean(res[m][m4]):9.4f}" for m4 in M4))

    for m in res:
        print(f"\n--- Δ ghep cap theo fold | cong = {m} ---")
        for nm, lbl in (("A", a.a), ("B", a.b)):
            cells_ = []
            for m4 in M4:
                d = [x - y_ for x, y_ in zip(res[m][m4], base[nm][m4])]
                pos = sum(1 for x in d if x > 1e-9)
                cells_.append(f"{st.mean(d):+8.4f} {pos}/{len(d)} p={sp(pos, len(d)):.3f}")
            print(f"  ghep − {lbl:28} " + " | ".join(cells_))
    return 0


if __name__ == "__main__":
    sys.exit(main())
