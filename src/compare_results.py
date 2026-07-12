#!/usr/bin/env python3
"""Create a paired five-fold transfer-vs-baseline comparison for one seed."""

import argparse
import csv
import json
from pathlib import Path

from summarize_results import METRICS, aggregate, mean_std
from logging_utils import configure_logging, get_logger


logger = get_logger()


def load_runs(directory, seed):
    runs = {}
    for path in sorted(Path(directory).glob("fold*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            run = json.load(handle)
        if run.get("seed") != seed:
            raise ValueError(f"{path}: seed={run.get('seed')} does not match requested seed={seed}")
        fold = int(run["fold"])
        if fold in runs:
            raise ValueError(f"duplicate fold {fold} in {directory}")
        runs[fold] = run
    if set(runs) != {1, 2, 3, 4, 5}:
        raise ValueError(f"{directory}: expected folds 1..5, found {sorted(runs)}")
    return runs


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Paired comparison of transfer and baseline over five folds",
    )
    parser.add_argument("--transfer_dir", required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--transfer_pipeline_seconds", type=float)
    parser.add_argument("--baseline_pipeline_seconds", type=float)
    args = parser.parse_args()

    transfer = load_runs(args.transfer_dir, args.seed)
    baseline = load_runs(args.baseline_dir, args.seed)
    rows = []
    deltas = {display_name: [] for display_name in METRICS}
    for fold in range(1, 6):
        row = {"seed": args.seed, "fold": fold}
        for display_name, json_key in METRICS.items():
            transfer_value = transfer[fold].get(json_key)
            baseline_value = baseline[fold].get(json_key)
            delta = (
                float(transfer_value) - float(baseline_value)
                if transfer_value is not None and baseline_value is not None
                else None
            )
            row[f"transfer_{display_name}"] = transfer_value
            row[f"baseline_{display_name}"] = baseline_value
            row[f"delta_{display_name}"] = delta
            if delta is not None:
                deltas[display_name].append(delta)
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_folds.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    transfer_runs = [transfer[fold] for fold in range(1, 6)]
    baseline_runs = [baseline[fold] for fold in range(1, 6)]
    comparison = {
        "seed": args.seed,
        "delta_definition": "transfer_minus_baseline",
        "pipeline_seconds": {
            "transfer": args.transfer_pipeline_seconds,
            "baseline": args.baseline_pipeline_seconds,
        },
        "transfer": aggregate(transfer_runs),
        "baseline": aggregate(baseline_runs),
        "paired_delta": {
            display_name: mean_std(values) for display_name, values in deltas.items()
        },
    }
    json_path = output_dir / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    logger.info("Paired-fold CSV saved: %s", csv_path)
    logger.info("Comparison JSON saved: %s", json_path)


if __name__ == "__main__":
    main()
