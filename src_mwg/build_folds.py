#!/usr/bin/env python3
"""Rebuild the Python folds so near-duplicate families never cross a split.

The shipped folds in data/sven_python_folds_norm were split per row. Because the
corpus is built from vulnerable/fixed pairs, that puts a patched copy of a test
function into the training set for roughly 40% of test rows. This script groups
rows into near-duplicate clusters first and assigns whole clusters to folds.

Clusters are unioned within a CWE only, and only between rows whose normalized
length is within LENGTH_BAND of each other -- a real vulnerable/fixed pair edits
a few lines, so a wider comparison costs time without finding pairs.

Writes a new directory; the original folds are left untouched.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

LENGTH_BAND = 0.5


def normalize(code):
    return re.sub(r"\s+", " ", code).strip()


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("code", "label", "cwe"):
                if field not in record:
                    raise ValueError(f"{path}:{line_number}: missing field {field!r}")
            record["_index"] = len(records)
            record["_normalized"] = normalize(record["code"])
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def cluster_records(records, threshold):
    """Union-find over near-duplicate pairs, returning clusters of row indices."""
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    by_cwe = defaultdict(list)
    for record in records:
        by_cwe[record["cwe"]].append(record)

    for group in by_cwe.values():
        for i in range(len(group)):
            first = group[i]
            for j in range(i + 1, len(group)):
                second = group[j]
                len_a, len_b = len(first["_normalized"]), len(second["_normalized"])
                if abs(len_a - len_b) / max(len_a, len_b, 1) > LENGTH_BAND:
                    continue
                matcher = SequenceMatcher(None, first["_normalized"], second["_normalized"])
                if matcher.quick_ratio() < threshold:
                    continue
                if matcher.ratio() >= threshold:
                    union(first["_index"], second["_index"])

    clusters = defaultdict(list)
    for record in records:
        clusters[find(record["_index"])].append(record["_index"])
    return [sorted(members) for members in clusters.values()]


def assign_folds(clusters, records, n_folds, seed):
    """Round-robin the clusters into folds, largest first, balancing CWE strata.

    Greedy balancing keeps each fold's CWE mix close to the corpus mix, which
    matters here because CWE-089 is over half the data and CWE-022 under 10%.
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for cluster in clusters:
        key = records[cluster[0]]["cwe"]
        strata[key].append(cluster)

    fold_of_cluster = {}
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        group.sort(key=len, reverse=True)
        load = [0] * n_folds
        for cluster in group:
            target = min(range(n_folds), key=lambda f: (load[f], f))
            fold_of_cluster[id(cluster)] = target
            load[target] += len(cluster)

    folds = [[] for _ in range(n_folds)]
    for key in sorted(strata):
        for cluster in strata[key]:
            folds[fold_of_cluster[id(cluster)]].extend(cluster)
    return [sorted(fold) for fold in folds]


def write_split(path, records, indices, pair_of):
    """Write rows in the original column order plus pair_id.

    pair_id is the first entry in the loader's GROUP_FIELDS, so its presence is
    enough to make source splitting group-aware without any other change.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for index in indices:
            record = {
                key: value
                for key, value in records[index].items()
                if not key.startswith("_")
            }
            record["pair_id"] = pair_of[index]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def describe(name, records, indices):
    labels = Counter(records[i]["label"] for i in indices)
    cwes = Counter(records[i]["cwe"] for i in indices)
    return (
        f"{name}: n={len(indices)} labels={dict(sorted(labels.items()))} "
        f"cwes={dict(sorted(cwes.items()))}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild Python folds with group-aware (near-duplicate) splitting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="data/sven_python_folds_norm/data.jsonl")
    parser.add_argument("--output_dir", default="data/sven_python_folds_grouped")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="SequenceMatcher ratio above which two rows are one group. "
                             "0.75 recovers 384 clusters from SVEN Python, matching its "
                             "documented 380 vulnerable/fixed pairs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    clusters = cluster_records(records, args.threshold)
    sizes = Counter(len(c) for c in clusters)
    print(f"Clusters: {len(clusters)} (threshold={args.threshold})")
    print(f"Cluster size distribution: {dict(sorted(sizes.items()))}")
    print(f"Effective independent units: {len(clusters)} vs {len(records)} rows")

    pair_of = {}
    for number, cluster in enumerate(clusters):
        for index in cluster:
            pair_of[index] = f"pair_{number:04d}"

    folds = assign_folds(clusters, records, args.n_folds, args.seed)
    output_dir = Path(args.output_dir)

    manifest = {
        "source": args.input,
        "threshold": args.threshold,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "n_records": len(records),
        "n_clusters": len(clusters),
        "cluster_size_distribution": {str(k): v for k, v in sorted(sizes.items())},
        "folds": {},
    }

    # Fold i is the test block; the next fold is validation; the rest train.
    # Splitting by whole folds keeps every cluster inside exactly one split.
    for i in range(args.n_folds):
        test_indices = folds[i]
        val_indices = folds[(i + 1) % args.n_folds]
        train_indices = sorted(
            index
            for j in range(args.n_folds)
            if j != i and j != (i + 1) % args.n_folds
            for index in folds[j]
        )
        fold_dir = output_dir / f"fold{i + 1}"
        write_split(fold_dir / "train.jsonl", records, train_indices, pair_of)
        write_split(fold_dir / "val.jsonl", records, val_indices, pair_of)
        write_split(fold_dir / "test.jsonl", records, test_indices, pair_of)
        print(f"fold{i + 1}:")
        for name, indices in (
            ("  train", train_indices), ("  val", val_indices), ("  test", test_indices)
        ):
            print("  " + describe(name.strip(), records, indices))
        manifest["folds"][f"fold{i + 1}"] = {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
