#!/usr/bin/env python3
"""Nhu tools/leak_groups_pair.py nhung in ROC-AUC va PR-AUC theo TUNG NHOM RO RI.

    python3 tools/leak_groups_auc.py --a <tag A> --b <tag B> <cay ket qua...>

VI SAO CAN. `leak_groups_pair.py` chi in macro-F1@0.5 — mot chi so, trai voi luat bon chi so
(CLAUDE.md §2b). Da mot lan giu ket luan sai ba tuan vi doc mot chi so ("ASAM null"). Nhom
`train` chi ~23 hang nen F1@0.5 o do rat nhay voi diem cat; chi so THU HANG khong phu thuoc
diem cat, nen no la phep kiem doc lap cho cung mot cau hoi.

Nhom qua nho hoac chi co MOT lop thi AUC khong dinh nghia duoc -> bo o do va BAO RO so o bo.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import statistics as st
from math import comb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leak_groups import CACHE                      # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

GRP = ("train", "test", "none")


def sp(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)


def arm_tag(name):
    b = name.replace("transfer_latent_bottleneck_", "")
    if "_l0p05_" in b:
        return b.partition("_l0p05_")[2].removesuffix("_adamw") or "goc"
    if b.endswith("_l0p05_adamw") or b.endswith("_l0p05"):
        return "goc"
    return b.removesuffix("_adamw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()
    groups = json.loads(CACHE.read_text())

    cells, skipped = defaultdict(dict), 0
    for root in a.roots:
        root = Path(root)
        for d in sorted(root.glob("*/seed_*/fold*.json")):
            arm = d.parent.parent.name
            tag = "baseline" if arm == "baseline" else arm_tag(arm)
            js = json.loads(d.read_text())
            if "test_probabilities" not in js:
                skipped += 1
                continue
            cells[(str(root), js["seed"], js["fold"])][tag] = js
    print(f"(bo qua {skipped} o khong co test_probabilities)")

    acc = {g: {"roc": [], "pr": []} for g in list(GRP) + ["TAT CA"]}
    bo_mot_lop = defaultdict(int)
    npair = 0
    for k, arms in sorted(cells.items()):
        if a.a not in arms or a.b not in arms:
            continue
        lab = groups.get(str(k[2]))
        if lab is None:
            continue
        A, B = arms[a.a], arms[a.b]
        yA, pA = A["test_labels"], A["test_probabilities"]
        yB, pB = B["test_labels"], B["test_probabilities"]
        if len(lab) != len(yA) or yA != yB:
            print(f"  !! BO {k}: nhan/thu tu khong khop")
            continue
        npair += 1
        for g in list(GRP) + ["TAT CA"]:
            idx = range(len(yA)) if g == "TAT CA" else [i for i, x in enumerate(lab) if x == g]
            idx = list(idx)
            ys = [yA[i] for i in idx]
            if len(idx) < 6 or len(set(ys)) < 2:
                bo_mot_lop[g] += 1
                continue
            acc[g]["roc"].append((roc_auc_score(ys, [pA[i] for i in idx])
                                  - roc_auc_score(ys, [pB[i] for i in idx]), len(idx)))
            acc[g]["pr"].append((average_precision_score(ys, [pA[i] for i in idx])
                                 - average_precision_score(ys, [pB[i] for i in idx]), len(idx)))

    print(f"\n=== {a.a}  −  {a.b} ===   {npair} cap o")
    print(f"{'nhom':8} {'hang TB':>8} {'n o':>4} | {'Δ ROC-AUC':>22} | {'Δ PR-AUC':>22}")
    for g in list(GRP) + ["TAT CA"]:
        v = acc[g]["roc"]
        w = acc[g]["pr"]
        if not v:
            print(f"{g:8} {'-':>8} {0:>4} |  (moi o mot lop / qua nho)")
            continue
        dr = [x for x, _ in v]
        dp = [x for x, _ in w]
        pr_, pp = sum(1 for x in dr if x > 0), sum(1 for x in dp if x > 0)
        print(f"{g:8} {st.mean(n for _, n in v):8.1f} {len(v):>4} | "
              f"{st.mean(dr):+8.4f} {pr_:>2}/{len(dr):<2} p={sp(pr_, len(dr)):.3f} | "
              f"{st.mean(dp):+8.4f} {pp:>2}/{len(dp):<2} p={sp(pp, len(dp)):.3f}")
    if bo_mot_lop:
        print("bo vi chi co mot lop hoac < 6 hang:", dict(bo_mot_lop))
    return 0


if __name__ == "__main__":
    sys.exit(main())
