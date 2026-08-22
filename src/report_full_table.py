#!/usr/bin/env python3
"""Bảng đầy đủ: từng fold, mean, Δ so với baseline, cho mọi run target Python.

Mỗi dòng ghi rõ backbone, pooling, λ_cwe, tín hiệu head phụ, và **dữ liệu source
dùng cho Phase 1** — thứ không được ghi vào file kết quả nên phải khai ở đây.

Baseline **không đọc dữ liệu source**, nên nó là mốc chung cho mọi nhánh trong
cùng một run. Δ luôn tính theo TỪNG fold rồi mới trung bình; không bao giờ trừ
hai trung bình tính trên số fold khác nhau (lỗi đã buộc rút §21).

Fold nào có nhánh sập về đoán một lớp (Macro-F1 < 0.55) thì đánh dấu `*` và loại
khỏi trung bình — giữ lại thì nó đo "một nhánh tình cờ hỏng" chứ không đo phương pháp.
"""

import argparse
import glob
import json
import os
import statistics

COLLAPSE = 0.55

# run_dir -> (backbone, pooling, lambda_cwe, source Phase 1, ghi chú)
RUNS = [
    ("results_vast2/unix_unixcoder",      "UniXcoder",   "cls",  "0.2",  "train_ccpp_js", ""),
    ("results_vast2/cb_lam_codebert",     "CodeBERT",    "cls",  "0.05", "train_ccpp_js", ""),
    ("results_vast/lam_ref_t5p",          "CodeT5+ 220m", "cls",  "0.2",  "train_ccpp_js", ""),
    ("results_vast/t5p_mean_t5pm",        "CodeT5+ 220m", "mean", "0.2",  "train_ccpp_js", ""),
    ("results_vast/t5p_lam05_t5pm",       "CodeT5+ 220m", "mean", "0.05", "train_ccpp_js",
     "baseline+none lấy từ t5p_mean"),
    ("results_vast2/t5base_mean_codet5m", "CodeT5-base", "mean", "0.2",  "train_ccpp_js", ""),
    ("results_vast2/t5base_lam05_codet5m", "CodeT5-base", "mean", "0.05", "train_ccpp_js",
     "baseline+none lấy từ t5base_mean"),
    ("results_vast/t5base_s7_codet5m",    "CodeT5-base", "mean", "0.05", "train_ccpp_js", "seed 7"),
    ("results_vast2/t5base_s12_codet5m",  "CodeT5-base", "mean", "0.05", "train_ccpp_js", "seed 12"),
    ("results_vast2/parent_ref_t5pm",     "CodeT5+ 220m", "mean", "0.2",
     "ccpp_primevul_paired_common", "head phụ 73 lớp CWE"),
    ("results_vast2/parent_new_t5pm",     "CodeT5+ 220m", "mean", "0.2",
     "ccpp_common_parent", "head phụ 9 lớp PILLAR"),
    ("results_vast/auxmatrix_ccppjs",     "CodeBERT",    "cls",  "0.2",  "train_ccpp_js", ""),
    ("results/seed42_ccppjs_py_compare_v1", "CodeBERT", "cls", "0.2", "train_ccpp_js", "run doi dau"),
    ("results/seed12_ccppjs_py_compare_v1", "CodeBERT", "cls", "0.2", "train_ccpp_js", "run doi dau"),
    ("results/seed18_ccppjs_py_compare_v1", "CodeBERT", "cls", "0.2", "train_ccpp_js", "run doi dau"),
    ("results_vast/seed7_ccppjs_py_compare_v1",  "CodeBERT", "cls", "0.2", "train_ccpp_js", ""),
    ("results_vast/backbone_codet5",      "CodeT5-base", "mean", "0.2", "train_ccpp_js",
     "trước khi sửa pooling"),
    ("results_vast/backbone_codet5p",     "CodeT5+ 220m", "mean", "0.2", "train_ccpp_js",
     "trước khi sửa pooling"),
]

# Nhánh nào mượn baseline/none từ run khác (cùng máy, cùng seed, cùng fold).
BORROW = {
    "results_vast/t5p_lam05_t5pm": "results_vast/t5p_mean_t5pm",
    "results_vast2/t5base_lam05_codet5m": "results_vast2/t5base_mean_codet5m",
    "results_vast2/parent_new_t5pm": "results_vast2/parent_ref_t5pm",
}


def load(directory, metric):
    out = {}
    for path in glob.glob(f"{directory}/fold*.json"):
        record = json.load(open(path))
        if record.get(metric) is not None:
            out[int(record["fold"])] = float(record[metric])
    return out


def seeds_in(run_dir):
    return sorted({os.path.basename(p) for p in glob.glob(f"{run_dir}/*/seed_*")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="test_macro_f1_at_0.5")
    args = parser.parse_args()

    metric_label = "Macro-F1" if "macro" in args.metric else "ROC-AUC"
    print(f"\nTARGET PYTHON — bộ fold gốc sven_python_folds_norm — {metric_label}")
    print("Δ = nhánh − baseline, tính từng fold rồi trung bình.  * = fold bị loại (sập về một lớp)\n")

    header = (f"{'backbone':<13}{'pool':<6}{'λ':<6}{'source Phase 1':<30}{'seed':>5}"
              f"{'nhánh':<20}" + "".join(f"{'f'+str(i):>9}" for i in range(1, 6))
              + f"{'MEAN':>10}{'Δ vs base':>11}{'+/n':>6}")
    print(header)
    print("=" * len(header))

    for run_dir, backbone, pooling, lam, source, note in RUNS:
        if not os.path.isdir(run_dir):
            continue
        for seed in seeds_in(run_dir):
            borrow = BORROW.get(run_dir, run_dir)
            base = load(f"{borrow}/baseline/{seed}", args.metric)
            if not base:
                base = load(f"{run_dir}/baseline/{seed}", args.metric)
            if not base:
                continue

            rows = [("baseline", base)]
            for method_dir in sorted(set(glob.glob(f"{run_dir}/transfer_*"))
                                     | set(glob.glob(f"{run_dir}/transfer"))):
                name = os.path.basename(method_dir).replace("transfer_", "")
                if name == "transfer":
                    name = "cwe (transfer)"
                scores = load(f"{method_dir}/{seed}", args.metric)
                if scores:
                    rows.append((name, scores))
            if len(rows) == 1:
                continue

            first = True
            for name, scores in rows:
                cells, deltas = [], []
                for fold in range(1, 6):
                    value = scores.get(fold)
                    if value is None:
                        cells.append(f"{'—':>9}")
                        continue
                    bad = value < COLLAPSE or base.get(fold, 1.0) < COLLAPSE
                    cells.append(f"{value:>8.4f}" + ("*" if bad else " "))
                    if not bad and fold in base:
                        deltas.append(value - base[fold])
                clean = [scores[f] for f in scores
                         if scores[f] >= COLLAPSE and base.get(f, 1.0) >= COLLAPSE]
                mean = f"{statistics.mean(clean):>10.4f}" if clean else f"{'—':>10}"
                if name == "baseline":
                    delta, ratio = f"{'—':>11}", f"{'—':>6}"
                else:
                    delta = f"{statistics.mean(deltas):>+11.4f}" if deltas else f"{'—':>11}"
                    ratio = (f"{sum(d > 0 for d in deltas):>3}/{len(deltas):<2}"
                             if deltas else f"{'—':>6}")
                lead = (f"{backbone:<13}{pooling:<6}{lam:<6}{source:<30}"
                        if first else f"{'':<13}{'':<6}{'':<6}{'':<30}")
                seed_cell = f"{seed.replace('seed_', ''):>5}" if first else f"{'':>5}"
                print(lead + seed_cell + f"{name:<20}" + "".join(cells) + mean + delta + ratio)
                first = False
            if note:
                print(f"{'':<13}{'':<6}{'':<6}→ {note}")
            print("-" * len(header))


if __name__ == "__main__":
    main()
