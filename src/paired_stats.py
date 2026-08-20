#!/usr/bin/env python3
"""Paired significance for fold-level comparisons, with the caveats stated.

Fold-level standard deviation here reaches 0.09 while the effects being measured
are 0.02-0.04, so a bare mean difference says very little. This reports the
tests the methodology literature actually recommends for this situation, and
labels what each one can and cannot support at n=5.

- Wilcoxon signed-rank: Demsar, JMLR 7:1-30, 2006, the standard non-parametric
  paired comparison for classifiers.
- Vargha-Delaney A12: DOI 10.3102/10769986025002101; Kitchenham & Madeyski,
  EMSE 29(6):137, 2024 find it less biased than Cohen's d at small n.
- Nadeau & Bengio correction, Machine Learning 52, 2003: folds share training
  data, so the naive paired t-test underestimates variance. The correction
  inflates it by n_test/n_train.

At n=5 the Wilcoxon test cannot go below p=0.0625 whatever the data, so it can
never reach 0.05. That is a property of the sample size, not of the effect, and
is reported rather than hidden.
"""

import argparse
import glob
import json
import math
import statistics
from pathlib import Path


def load(directory, metric):
    scores = {}
    for path in sorted(glob.glob(str(Path(directory) / "fold*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get(metric) is not None:
            scores[int(record["fold"])] = float(record[metric])
    return scores


def wilcoxon_signed_rank(differences):
    """Exact two-sided p for small n, ignoring zero differences."""
    values = [d for d in differences if d != 0]
    n = len(values)
    if n == 0:
        return 1.0, None
    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        shared = (i + j + 2) / 2  # average rank over the tie block, 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, values) if v > 0)
    w_minus = sum(r for r, v in zip(ranks, values) if v < 0)
    statistic = min(w_plus, w_minus)

    # Enumerate every sign assignment; n is at most a handful of folds here.
    total = 0
    count = 0
    for mask in range(1 << n):
        plus = sum(ranks[k] for k in range(n) if mask >> k & 1)
        minus = sum(ranks) - plus
        if min(plus, minus) <= statistic:
            count += 1
        total += 1
    return count / total, statistic


def vargha_delaney(a, b):
    """P(a > b) with ties counted as half."""
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return wins / (len(a) * len(b))


def corrected_t(differences, n_train, n_test):
    """Nadeau-Bengio corrected paired t statistic."""
    n = len(differences)
    if n < 2:
        return None, None
    mean = statistics.mean(differences)
    variance = statistics.variance(differences)
    if variance == 0:
        return None, None
    corrected = variance * (1.0 / n + n_test / n_train)
    return mean / math.sqrt(corrected), n - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True,
                        help="directory holding <method>/seed_<seed>/foldN.json")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--metric", default="test_macro_f1_at_0.5")
    parser.add_argument("--n_train", type=int, default=456)
    parser.add_argument("--n_test", type=int, default=152)
    args = parser.parse_args()

    root = Path(args.results_root)
    base = load(root / args.baseline / f"seed_{args.seed}", args.metric)
    if not base:
        print(f"no baseline results under {root / args.baseline}")
        return

    methods = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != args.baseline)
    print(f"[{args.metric}] baseline n={len(base)}  "
          f"mean={statistics.mean(base.values()):.4f}\n")
    header = f"{'method':<30}{'n':>3}{'Δ mean':>10}{'Δ std':>9}{'A12':>7}{'Wilcoxon p':>12}{'t_corr':>9}"
    print(header)
    print("-" * len(header))
    for method in methods:
        scores = load(root / method / f"seed_{args.seed}", args.metric)
        folds = sorted(set(scores) & set(base))
        if len(folds) < 2:
            print(f"{method:<30}{len(folds):>3}  (need at least two paired folds)")
            continue
        differences = [scores[f] - base[f] for f in folds]
        p_value, _ = wilcoxon_signed_rank(differences)
        a12 = vargha_delaney([scores[f] for f in folds], [base[f] for f in folds])
        t_stat, _ = corrected_t(differences, args.n_train, args.n_test)
        print(
            f"{method:<30}{len(folds):>3}"
            f"{statistics.mean(differences):>+10.4f}"
            f"{statistics.stdev(differences):>9.4f}"
            f"{a12:>7.2f}"
            f"{p_value:>12.4f}"
            f"{(f'{t_stat:+.2f}' if t_stat is not None else '   n/a'):>9}"
        )

    n = len(base)
    print(f"\nSmallest reachable Wilcoxon p at n={n} is {2 / (2 ** n):.4f}; "
          f"significance at 0.05 is unreachable below n=6.")
    print("A12 reads as P(method > baseline) on a randomly drawn fold: "
          "0.50 is no difference, 0.71 and above is a large effect.")


if __name__ == "__main__":
    main()
