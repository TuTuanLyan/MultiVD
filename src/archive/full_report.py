#!/usr/bin/env python3
"""Full per-config report: absolute per-fold scores, not just deltas.

Deltas answer "did it beat the baseline". Absolute numbers are what goes in a
paper table and what tells you whether a config is in a sane range at all.
"""

import argparse
import json
import statistics
from pathlib import Path

METRICS = [
    ("test_macro_f1_at_0.5", "Macro-F1 @0.5"),
    ("test_macro_f1_at_valcal", "Macro-F1 @valcal"),
    ("test_positive_f1_at_0.5", "Positive-F1 @0.5"),
    ("test_precision_at_0.5", "Precision @0.5"),
    ("test_recall_at_0.5", "Recall @0.5"),
    ("test_accuracy_at_0.5", "Accuracy @0.5"),
    ("test_roc_auc", "ROC-AUC"),
    ("test_pr_auc", "PR-AUC"),
]

LABELS = {
    "baseline": "baseline (CodeBERT, Python only)",
    "transfer_cwe": "cwe (explicit 4-class head)",
    "transfer_latent_bottleneck": "latent_bottleneck (K=8 -> 4)",
    "transfer_latent_proto": "latent_proto (K=8, no labels)",
    "transfer_none": "none (lambda=0, no aux task)",
}


def load(root, method, seed):
    runs = {}
    for path in sorted((root / method / f"seed_{seed}").glob("fold*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        runs[int(record["fold"])] = record
    return runs


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser(description="Full absolute-number report")
    parser.add_argument("--results_root", default="results_vast/auxmatrix_ccppjs")
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.results_root)
    methods = ["baseline"] + sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("transfer")
    )
    data = {m: load(root, m, args.seed) for m in methods}
    folds = list(range(1, args.folds + 1))

    print(f"Source: ccpp+js | Target: 5 Python folds | Seed: {args.seed}")
    print(f"Backbone: microsoft/codebert-base | Results: {root}\n")

    for key, title in METRICS:
        print(f"### {title}")
        header = f"{'config':<34}" + "".join(f"fold{f}   " for f in folds) + " mean ± std"
        print(header)
        print("-" * len(header))
        base = {f: data["baseline"][f][key] for f in folds if f in data["baseline"]}
        for method in methods:
            runs = data[method]
            cells = ""
            values = []
            for f in folds:
                if f in runs and runs[f].get(key) is not None:
                    value = runs[f][key]
                    values.append(value)
                    cells += f"{value:.4f}  "
                else:
                    cells += "  --    "
            mean, std = mean_std(values)
            line = f"{LABELS.get(method, method):<34}{cells}{mean:.4f} ± {std:.4f}"
            if method != "baseline":
                paired = [runs[f][key] - base[f] for f in folds if f in runs and f in base]
                if paired:
                    dmean, dstd = mean_std(paired)
                    line += f"   Δ {dmean:+.4f} ± {dstd:.4f}"
            print(line)
        print()

    print("### Per-CWE Macro-F1 at the validation-calibrated threshold")
    cwes = ["CWE-022", "CWE-078", "CWE-079", "CWE-089"]
    header = f"{'config':<34}" + "".join(f"{c:<12}" for c in cwes)
    print(header)
    print("-" * len(header))
    for method in methods:
        runs = data[method]
        cells = ""
        for cwe in cwes:
            values = [
                runs[f]["per_cwe"][cwe]["macro_f1"]
                for f in folds
                if f in runs and runs[f].get("per_cwe", {}).get(cwe, {}).get("macro_f1") is not None
            ]
            cells += f"{statistics.mean(values):.4f}      " if values else "  --        "
        print(f"{LABELS.get(method, method):<34}{cells}")
    counts = []
    for cwe in cwes:
        n = [
            data["baseline"][f]["per_cwe"][cwe]["number_of_samples"]
            for f in folds
            if f in data["baseline"]
        ]
        counts.append(f"{cwe}: {sum(n)} test samples total")
    print("\n" + " | ".join(counts))


if __name__ == "__main__":
    main()
