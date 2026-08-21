#!/usr/bin/env python3
"""Gộp mọi kết quả đã có, tách theo BỘ FOLD, rồi so T5 / CodeT5+ với CodeBERT.

Bộ fold không được ghi vào file kết quả, nên suy ra từ cỡ tập test: lấy
`test_accuracy_at_0.5` của từng fold và tìm mẫu số nguyên duy nhất khớp với mọi
fold của run đó. Ba bộ tách nhau rõ ở fold 1 — 152 (gốc), 154 (twin), 146
(random) — nên phép suy này xác định được, không phải đoán.

Việc tách theo bộ fold là bắt buộc chứ không phải cho gọn: §33 gộp CodeT5+ qua ba
seed mà không nói rõ chúng chạy trên bộ twin, và điều đó khiến bảng bị đọc như thể
áp cho bộ dùng để báo cáo.
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

METRICS = ("test_macro_f1_at_0.5", "test_roc_auc")

# Cỡ tập test theo fold của từng bộ dữ liệu, đọc trực tiếp từ data/.
FOLD_SIZES = {
    "goc":    {1: 152, 2: 152, 3: 152, 4: 152, 5: 152},
    "twin":   {1: 154, 2: 153, 3: 152, 4: 151, 5: 150},
    "random": {1: 146, 2: 153, 3: 157, 4: 153, 5: 151},
    "js":     {1: 230, 2: 229, 3: 227, 4: 226, 5: 226},
}

BACKBONES = [("codebert", "CodeBERT"), ("t5p", "CodeT5+"),
             ("codet5", "CodeT5"), ("t5", "CodeT5+")]


COLLAPSE_F1 = 0.55   # dưới mức này thì mô hình đang đoán gần như một lớp


def backbone_of(label):
    for key, name in BACKBONES:
        if key in label:
            return name
    # Các run đời đầu không ghi backbone vào tên vì chỉ có một backbone; mặc định
    # của run/config.sh là microsoft/codebert-base. Đánh dấu * để phân biệt với
    # những run có backbone ghi rõ trong tên.
    return "CodeBERT*"


def collapsed_folds(scores):
    """Fold mà mô hình sập về đoán một lớp.

    Bắt buộc phải tách riêng, không được gộp vào trung bình. Baseline CodeT5-base
    fold 1 ra Macro-F1 0.3184 với precision và recall đều bằng 0 — mô hình đoán
    toàn lớp âm. Δ của nhánh transfer trên fold đó là +0.5171, và nếu để nguyên
    trong bảng thì nó đo "baseline tình cờ sập" chứ không đo tác dụng của phương
    pháp; một fold như vậy đủ sức lật kết luận của cả run 5 fold.
    """
    return sorted(f for f, value in scores.items() if value < COLLAPSE_F1)


def load(directory, metric):
    scores = {}
    for path in glob.glob(f"{directory}/fold*.json"):
        record = json.load(open(path))
        if record.get(metric) is not None:
            scores[int(record["fold"])] = float(record[metric])
    return scores


def observed_accuracies(directory):
    out = {}
    for path in glob.glob(f"{directory}/fold*.json"):
        record = json.load(open(path))
        accuracy = record.get("test_accuracy_at_0.5")
        if accuracy is not None:
            out[int(record["fold"])] = accuracy
    return out


def dataset_of(directory):
    """Bộ fold nào giải thích được accuracy của MỌI fold trong run.

    Bản đầu tìm mẫu số nguyên nhỏ nhất cho từng fold rồi so với bảng cỡ. Cách đó
    sai vì phân số accuracy rút gọn được: 114/152 cho ra mẫu số 4, và với dải tìm
    bắt đầu từ 100 nó trả về 114 thay vì 152. Hệ quả là gần như mọi run bị xếp
    vào "khác", kể cả những run đã biết chắc chạy trên bộ twin.

    Cách đúng là đi ngược lại: với từng bộ ứng viên, kiểm accuracy nhân cỡ THẬT
    của fold đó có ra số nguyên không. Bộ nào khớp nhiều fold nhất thì chọn.
    """
    accuracies = observed_accuracies(directory)
    if not accuracies:
        return "?", {}
    best, best_hits = "khac", 0
    for name, expected in FOLD_SIZES.items():
        hits = sum(
            1 for fold, accuracy in accuracies.items()
            if fold in expected
            and abs(accuracy * expected[fold] - round(accuracy * expected[fold])) < 1e-6
        )
        if hits > best_hits:
            best, best_hits = name, hits
    if best_hits < len(accuracies):
        return ("khac" if best_hits < len(accuracies) / 2 else best), accuracies
    return best, accuracies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+",
                        default=["results", "results_vast", "results_vast2"])
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        for run_dir in sorted(glob.glob(f"{root}/*")):
            label = os.path.basename(run_dir)
            for baseline_dir in sorted(glob.glob(f"{run_dir}/baseline/seed_*")):
                seed = os.path.basename(baseline_dir).replace("seed_", "")
                dataset, _ = dataset_of(baseline_dir)
                base = {m: load(baseline_dir, m) for m in METRICS}
                if not base[METRICS[0]]:
                    continue
                for method_dir in sorted(glob.glob(f"{run_dir}/*/seed_{seed}")):
                    method = os.path.basename(os.path.dirname(method_dir))
                    if method == "baseline":
                        continue
                    scores = {m: load(method_dir, m) for m in METRICS}
                    folds = sorted(set(scores[METRICS[0]]) & set(base[METRICS[0]]))
                    if not folds:
                        continue
                    bad = set(collapsed_folds(base[METRICS[0]])) | \
                          set(collapsed_folds(scores[METRICS[0]]))
                    clean = [f for f in folds if f not in bad]
                    entry = {"run": label, "backbone": backbone_of(label),
                             "dataset": dataset, "seed": seed,
                             "method": method.replace("transfer_", ""),
                             "n": len(clean), "dropped": sorted(bad & set(folds))}
                    folds = clean
                    if not folds:
                        rows.append({**entry, **{m: None for m in METRICS},
                                     **{m + "_pos": 0 for m in METRICS},
                                     **{m + "_base": float("nan") for m in METRICS}})
                        continue
                    for metric in METRICS:
                        deltas = [scores[metric][f] - base[metric][f]
                                  for f in folds if f in scores[metric] and f in base[metric]]
                        entry[metric] = statistics.mean(deltas) if deltas else None
                        entry[metric + "_pos"] = sum(d > 0 for d in deltas)
                        entry[metric + "_base"] = statistics.mean(
                            base[metric][f] for f in folds if f in base[metric])
                    rows.append(entry)

    order = {"goc": 0, "random": 1, "twin": 2, "js": 3, "khac": 4, "?": 5}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    for dataset in sorted(grouped, key=lambda d: order.get(d, 9)):
        title = {"goc": "BỘ FOLD GỐC (sven_python_folds_norm) — bộ dùng để báo cáo",
                 "random": "BỘ FOLD RANDOM (sven_python_random)",
                 "twin": "BỘ FOLD TWIN (sven_python_twin) — thí nghiệm phụ",
                 "js": "TARGET JAVASCRIPT (js_twin) — target khác, không phải Python",
                 }.get(dataset, f"BỘ KHÁC ({dataset}) — target không phải Python")
        print(f"\n{'=' * 104}\n{title}\n{'=' * 104}")
        header = (f"{'backbone':<10}{'run':<22}{'nhánh':<20}{'seed':>5}{'n':>3}"
                  f"{'baseline F1':>13}{'ΔF1':>10}{'+/n':>6}{'ΔAUC':>10}{'+/n':>6}")
        print(header)
        print("-" * len(header))
        block = sorted(grouped[dataset],
                       key=lambda r: (r["backbone"], r["run"], r["method"], r["seed"]))
        for row in block:
            f1, auc = row[METRICS[0]], row[METRICS[1]]
            note = f"  [bo fold {','.join(map(str, row['dropped']))}: sap]" if row.get("dropped") else ""
            if f1 is None:
                print(f"{row['backbone']:<10}{row['run']:<22}{row['method']:<20}"
                      f"{row['seed']:>5}{0:>3}{'—':>13}{'—':>10}{'—':>6}{'—':>10}{'—':>6}{note}")
                continue
            print(f"{row['backbone']:<10}{row['run']:<22}{row['method']:<20}"
                  f"{row['seed']:>5}{row['n']:>3}{row[METRICS[0] + '_base']:>13.4f}"
                  f"{f1:>+10.4f}{row[METRICS[0] + '_pos']:>4}/{row['n']:<2}"
                  f"{(auc if auc is not None else float('nan')):>+10.4f}"
                  f"{row[METRICS[1] + '_pos']:>4}/{row['n']:<2}{note}")


if __name__ == "__main__":
    main()
