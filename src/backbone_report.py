#!/usr/bin/env python3
"""Compare the transfer method across pretrained backbones.

Each backbone gets its own baseline, trained on the same folds and seed, because
a baseline from a different backbone is not a valid reference.
"""

import glob
import json
import statistics

ROWS = [
    ("CodeBERT", "results_vast/auxmatrix_ccppjs/baseline/seed_36",
     "results_vast/auxmatrix_ccppjs/transfer_cwe/seed_36"),
    ("CodeT5", "results_vast/backbone_codet5/baseline/seed_36",
     "results_vast/backbone_codet5/transfer/seed_36"),
    ("CodeT5+", "results_vast/backbone_codet5p/baseline/seed_36",
     "results_vast/backbone_codet5p/transfer/seed_36"),
]
METRICS = [("test_macro_f1_at_0.5", "Macro-F1"), ("test_roc_auc", "ROC-AUC"),
           ("test_pr_auc", "PR-AUC")]


def load(directory, key):
    scores = {}
    for path in sorted(glob.glob(directory + "/fold*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get(key) is not None:
            scores[record["fold"]] = record[key]
    return scores


def main():
    print(f"{'backbone':<10} {'metric':<10} {'baseline':>17} {'transfer':>17} {'delta':>9}")
    print("-" * 68)
    for name, baseline_dir, transfer_dir in ROWS:
        for key, label in METRICS:
            base = load(baseline_dir, key)
            transfer = load(transfer_dir, key)
            folds = sorted(set(base) & set(transfer))
            if not folds:
                print(f"{name:<10} {label:<10} no paired folds")
                continue
            bv = [base[f] for f in folds]
            tv = [transfer[f] for f in folds]
            delta = statistics.mean(t - b for b, t in zip(bv, tv))
            print(
                f"{name:<10} {label:<10} "
                f"{statistics.mean(bv):.4f} ± {statistics.stdev(bv):.4f}  "
                f"{statistics.mean(tv):.4f} ± {statistics.stdev(tv):.4f}  "
                f"{delta:+.4f}"
            )
        print()


if __name__ == "__main__":
    main()
