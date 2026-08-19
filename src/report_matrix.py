#!/usr/bin/env python3
"""Print whatever the auxiliary-mode sweep has produced so far.

Written for a running sweep: it reads only the fold results that exist, so it
can be called after every fold to show the table filling in. Deltas are against
the baseline on the same fold, which is the only comparison that means anything
here since results shift across machines.
"""

import json
import os
from pathlib import Path

RUN_NAME = os.environ.get("RUN_NAME", "auxmatrix_ccppjs")
SEED = os.environ.get("SEED", "36")
METRIC = os.environ.get("METRIC", "test_macro_f1_at_0.5")
ROOT = Path("results") / RUN_NAME


def load(method):
    directory = ROOT / method / f"seed_{SEED}"
    scores = {}
    for path in sorted(directory.glob("fold*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if record.get(METRIC) is not None:
            scores[int(record["fold"])] = float(record[METRIC])
    return scores


def main():
    if not ROOT.is_dir():
        print(f"no results yet under {ROOT}")
        return
    methods = ["baseline"] + sorted(
        p.name for p in ROOT.iterdir() if p.name.startswith("transfer_")
    )
    table = {m: load(m) for m in methods}
    folds = sorted({f for scores in table.values() for f in scores})
    if not folds:
        print("no fold results yet")
        return

    baseline = table.get("baseline", {})
    width = max(len(m) for m in methods)
    header = f"{'config'.ljust(width)} " + " ".join(f"fold{f}  " for f in folds)
    print(f"[{METRIC}]")
    print(header + "  mean    vs baseline")
    print("-" * len(header + "  mean    vs baseline"))
    for method in methods:
        scores = table[method]
        if not scores:
            continue
        cells = " ".join(
            (f"{scores[f]:.4f} " if f in scores else "  --   ") for f in folds
        )
        values = [scores[f] for f in folds if f in scores]
        mean = sum(values) / len(values)
        paired = [
            scores[f] - baseline[f] for f in folds if f in scores and f in baseline
        ]
        delta = (
            f"{sum(paired) / len(paired):+.4f} (n={len(paired)})" if paired else "  --"
        )
        if method == "baseline":
            delta = "  --"
        print(f"{method.ljust(width)} {cells}  {mean:.4f}  {delta}")


if __name__ == "__main__":
    main()
