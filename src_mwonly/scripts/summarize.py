#!/usr/bin/env python3
"""Bảng từng fold + trung bình từ các file kết quả fold<k>.json:  python scripts/summarize.py <thư mục chứa fold*.json> ..."""
import glob
import json
import os
import sys

KEYS = [("test_roc_auc", "ROC"), ("test_macro_f1_at_0.5", "mF1@0,5"), ("test_macro_f1_at_valcal", "mF1@cal"), ("test_pr_auc", "PR")]
for d in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(d, "fold*.json")), key=lambda p: int(os.path.basename(p)[4:-5]))
    rows = [json.load(open(p)) for p in files]
    if not rows:
        print(f"{d}: không có kết quả")
        continue
    print(f"\n{d}\n| fold | best epoch | " + " | ".join(n for _, n in KEYS) + " |\n|---|---:|" + "---:|" * len(KEYS))
    for r in rows:
        print(f"| {r['fold']} | {r['best_epoch']} | " + " | ".join(f"{r[k]:.4f}" for k, _ in KEYS) + " |")
    print(f"| **TB ({len(rows)} fold)** | | " + " | ".join(f"**{sum(r[k] for r in rows) / len(rows):.4f}**" for k, _ in KEYS) + " |")
