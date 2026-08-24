#!/usr/bin/env python3
"""Aggregate fold/seed JSON results from the transfer experiment."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from logging_utils import configure_logging, get_logger


logger = get_logger()


METRICS = {
    "macro_f1_at_0.5": "test_macro_f1_at_0.5",
    "macro_f1_at_valcal": "test_macro_f1_at_valcal",
    "positive_f1_at_0.5": "test_positive_f1_at_0.5",
    "positive_f1_at_valcal": "test_positive_f1_at_valcal",
    "roc_auc": "test_roc_auc",
    "pr_auc": "test_pr_auc",
    "precision_at_0.5": "test_precision_at_0.5",
    "recall_at_0.5": "test_recall_at_0.5",
    "accuracy_at_0.5": "test_accuracy_at_0.5",
    "validation_and_threshold_seconds": "validation_and_threshold_seconds",
    "test_inference_seconds": "test_inference_seconds",
}
PER_CWE_METRICS = (
    "number_of_samples",
    "positive_ratio",
    "macro_f1",
    "positive_f1",
    "precision",
    "recall",
    "accuracy",
)


def mean_std(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "mean_std": "null"}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "std": std, "mean_std": f"{mean:.4f} ± {std:.4f}"}


def aggregate(runs):
    output = {
        "n_runs": len(runs),
        "metrics": {
            display_name: mean_std(run.get(json_key) for run in runs)
            for display_name, json_key in METRICS.items()
        },
        "per_cwe": {},
        "per_cwe_at_0.5": {},
        "per_cwe_at_valcal": {},
    }
    for cwe in ("CWE-022", "CWE-078", "CWE-079", "CWE-089"):
        for output_key, run_key in (
            ("per_cwe_at_0.5", "per_cwe_at_0.5"),
            ("per_cwe_at_valcal", "per_cwe_at_valcal"),
        ):
            output[output_key][cwe] = {
                metric: mean_std(
                    run.get(run_key, run.get("per_cwe", {})).get(cwe, {}).get(metric)
                    for run in runs
                )
                for metric in PER_CWE_METRICS
            }
        output["per_cwe"][cwe] = output["per_cwe_at_valcal"][cwe]
    return output


def csv_row(run):
    row = {
        "experiment_name": run.get("experiment_name"),
        "fold": run["fold"],
        "seed": run["seed"],
        "source_checkpoint": run.get("source_checkpoint"),
        "target_checkpoint": run.get("target_checkpoint"),
        "best_epoch": run.get("best_epoch"),
        "val_calibrated_threshold": run.get("val_calibrated_threshold"),
    }
    for display_name, json_key in METRICS.items():
        row[display_name] = run.get(json_key)
    for threshold_name, run_key in (
        ("at_0.5", "per_cwe_at_0.5"),
        ("at_valcal", "per_cwe_at_valcal"),
    ):
        for cwe, metrics in run.get(run_key, run.get("per_cwe", {})).items():
            for metric in PER_CWE_METRICS:
                row[f"{cwe}_{threshold_name}_{metric}"] = metrics.get(metric)
    return row


def main():
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="results/default_run/transfer/seed_42")
    parser.add_argument("--output_dir", default="results/default_run/transfer/seed_42")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("fold*.json"))
    if not paths:
        raise FileNotFoundError(f"no fold*.json results found in {input_dir}")
    runs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            run = json.load(handle)
        if "fold" not in run or "seed" not in run:
            raise ValueError(f"{path}: result lacks fold or seed")
        runs.append(run)
    seeds = {run["seed"] for run in runs}
    if len(seeds) != 1:
        raise ValueError(
            f"input directory mixes seeds {sorted(seeds)}; summarize one seed directory at a time"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [csv_row(run) for run in sorted(runs, key=lambda item: (item["fold"], item["seed"]))]
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = output_dir / "all_runs.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_seed, by_fold = defaultdict(list), defaultdict(list)
    for run in runs:
        by_seed[str(run["seed"])].append(run)
        by_fold[f"fold{run['fold']}"] .append(run)
    summary = {
        "input_files": [str(path) for path in paths],
        "by_seed": {seed: aggregate(items) for seed, items in sorted(by_seed.items())},
        "by_fold": {fold: aggregate(items) for fold, items in sorted(by_fold.items())},
        "ALL": aggregate(runs),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    logger.info("Runs CSV saved: %s", csv_path)
    logger.info("Summary JSON saved: %s", summary_path)


if __name__ == "__main__":
    main()
