#!/usr/bin/env python3
"""Gop ba cay ket qua cua khoi NIGHT48 va bao cao Delta ghep cap.

    python3 tools/n48_report.py

Ba cay, vi khoi bi chia ba may:
    results_night48/       seed 42        (vast cu, da huy)
    results/n48_t5p/       seed 7 fold 1-2 (161)
    results_night48b/      seed 7 fold 3-5 (vast moi)
    results_night48_158/   seed 1234      (158)

KIEM CHONG LAN truoc khi tinh: moi o (nguon, rho, seed, fold) phai xuat hien DUNG MOT
LAN. Neu mot o co o hai cay thi hoac ta dem trung, hoac hai lan chay khac may bi tron —
ca hai deu lam hong Delta ghep cap. Ngay 05/09 fold 3 va 5 cua seed 7 tung co o le tren
161 truoc khi OOM; chung da duoc chuyen sang results_n48_161_partial/ va cay do KHONG
duoc doc o day.
"""
import json, glob, math, re, sys
from collections import defaultdict

F1 = "test_macro_f1_at_valcal"
AUC = "test_roc_auc"
TREES = ["results_night48", "results", "results_night48b", "results_night48_158"]
SOURCES = ["4cwe", "com", "full"]
SEEDS = ["42", "7", "1234"]
FOLDS = [str(f) for f in range(1, 6)]


def sign_test(vals):
    n = len(vals)
    if n == 0:
        return 1.0
    k = sum(v > 0 for v in vals)
    tail = lambda j: sum(math.comb(n, i) for i in range(j, n + 1)) / 2 ** n
    return min(1.0, 2 * min(tail(k), tail(n - k)))


def fmt(vals, floor=0.010):
    n = len(vals)
    if n == 0:
        return "  (khong co o nao)"
    m = sum(vals) / n
    k = sum(v > 0 for v in vals)
    p = sign_test(vals)
    mark = "" if abs(m) >= floor else "  [duoi san nhieu]"
    return "%+.4f (%2d/%d, p=%.4f)%s" % (m, k, n, p, mark)


def load():
    base, cell, seen = {}, {}, defaultdict(list)
    for tree in TREES:
        for p in glob.glob(f"{tree}/n48_t5p/baseline/seed_*/fold*.json"):
            sd, fd = re.search(r"seed_(\d+)/fold(\d+)", p).groups()
            seen[("baseline", "-", sd, fd)].append(p)
            d = json.load(open(p)); base[(sd, fd)] = (d[F1], d[AUC])
        for p in glob.glob(f"{tree}/n48_t5p/transfer_latent_bottleneck_*_l0p05_r*/seed_*/fold*.json"):
            src, rho, sd, fd = re.search(
                r"latent_bottleneck_(\w+?)_l0p05_(r[0-9p]+)/seed_(\d+)/fold(\d+)", p).groups()
            seen[(src, rho, sd, fd)].append(p)
            d = json.load(open(p)); cell[(src, rho, sd, fd)] = (d[F1], d[AUC])
    return base, cell, seen


def main():
    base, cell, seen = load()
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    print("=" * 78)
    print("KIEM TOAN O — phai du va khong chong lan")
    print("=" * 78)
    if dup:
        print("!! %d o xuat hien nhieu hon mot lan — PHAI XU LY TRUOC KHI TIN SO:" % len(dup))
        for k, v in sorted(dup.items())[:10]:
            print("   %s:" % (k,))
            for p in v:
                print("      %s" % p)
        return 2
    print("  khong o nao chong lan.")
    miss_b = [(s, f) for s in SEEDS for f in FOLDS if (s, f) not in base]
    miss_c = [(src, r, s, f) for src in SOURCES for r in ("r0", "r0p1")
              for s in SEEDS for f in FOLDS if (src, r, s, f) not in cell]
    print("  baseline: %d/15%s" % (len(base), "" if not miss_b else "  THIEU: %s" % miss_b))
    print("  o Pha 2 : %d/90%s" % (len(cell), "" if not miss_c else "  THIEU %d: %s%s" % (
        len(miss_c), miss_c[:6], " ..." if len(miss_c) > 6 else "")))
    if miss_c or miss_b:
        print("\n  (bao cao duoi day chi tinh tren nhung o CO THAT)")

    print("\nBaseline theo seed (trung binh 5 fold):")
    for s in SEEDS:
        b = [base[(s, f)] for f in FOLDS if (s, f) in base]
        if b:
            print("  seed %-5s F1 %.4f (%.4f–%.4f) | ROC %.4f  [n=%d]" % (
                s, sum(x[0] for x in b) / len(b), min(x[0] for x in b),
                max(x[0] for x in b), sum(x[1] for x in b) / len(b), len(b)))

    print("\n" + "=" * 78)
    print("A) CHUYEN GIAO vs BASELINE — ghep theo (nguon, seed, fold)")
    print("=" * 78)
    print("   %-6s %-4s %-30s %s" % ("nguon", "rho", "Δ macro-F1", "Δ ROC-AUC"))
    for src in SOURCES:
        for rho, rl in (("r0", "0"), ("r0p1", "0.1")):
            d = [(cell[(src, rho, s, f)][0] - base[(s, f)][0],
                  cell[(src, rho, s, f)][1] - base[(s, f)][1])
                 for s in SEEDS for f in FOLDS
                 if (src, rho, s, f) in cell and (s, f) in base]
            print("   %-6s %-4s %-30s %s" % (
                src, rl, fmt([x[0] for x in d]), fmt([x[1] for x in d])))

    print("\n" + "=" * 78)
    print("B) HIEU RIENG CUA ASAM — rho 0.1 vs rho 0, CHUNG checkpoint Pha 1")
    print("=" * 78)
    allf, alla = [], []
    for src in SOURCES:
        d = [(cell[(src, "r0p1", s, f)][0] - cell[(src, "r0", s, f)][0],
              cell[(src, "r0p1", s, f)][1] - cell[(src, "r0", s, f)][1])
             for s in SEEDS for f in FOLDS
             if (src, "r0p1", s, f) in cell and (src, "r0", s, f) in cell]
        f_ = [x[0] for x in d]; a = [x[1] for x in d]
        allf += f_; alla += a
        rng = "  [%+.4f…%+.4f]" % (min(f_), max(f_)) if f_ else ""
        print("   %-6s ΔF1 %-30s | ΔROC %s%s" % (src, fmt(f_), fmt(a), rng))
    print("   %-6s ΔF1 %-30s | ΔROC %s" % ("GOP", fmt(allf), fmt(alla)))

    print("\n   Tach theo seed (gop ba nguon):")
    for s in SEEDS:
        d = [cell[(src, "r0p1", s, f)][0] - cell[(src, "r0", s, f)][0]
             for src in SOURCES for f in FOLDS
             if (src, "r0p1", s, f) in cell and (src, "r0", s, f) in cell]
        print("     seed %-5s ΔF1 %s" % (s, fmt(d)))
    print("\n   San nhieu 0.010 (chay lai cung seed cung cau hinh, khac may).")
    print("   San kiem dau: p=0.0625 o n=5, p=0.0020 o n=10, p=0.0001 o n=15 neu cung dau het.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
