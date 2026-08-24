#!/usr/bin/env python3
"""Gộp một cấu hình qua nhiều seed, và phán quyết theo đúng quy trình cổng chặn.

Đại lượng là **giá trị gia tăng của head phụ**, Δ(cwe) − Δ(none), tính RIÊNG
từng fold rồi mới trung bình. Không bao giờ trừ hai trung bình cho nhau: §21 có
một khẳng định phải rút lại vì làm đúng như vậy trên hai số n khác nhau.

Trục seed quan trọng hơn trục fold ở đây. §33 đo được `cwe` trên CodeT5+ dao động
từ −0.0132 (seed 18) tới +0.0279 (seed 42), sd giữa seed 0.0206 — **lớn hơn hiệu
ứng trung bình +0.0062**. Một cấu hình dương ở một seed nói rất ít; điều cần biết
là dấu có giữ qua các seed không.

Phán quyết chỉ có ba khả năng, và "chưa đủ dữ liệu" là một trong ba.
"""

import argparse
import glob
import json
import os
import statistics

NOISE = 0.005
COLLAPSE = 0.55


def load(directory, metric):
    scores = {}
    for path in glob.glob(f"{directory}/fold*.json"):
        record = json.load(open(path))
        if record.get(metric) is not None:
            scores[int(record["fold"])] = float(record[metric])
    return scores


def added_per_fold(base, none, method, metric):
    """Giá trị gia tăng của head phụ trên từng fold, bỏ fold có nhánh nào sập."""
    b, n, m = load(base, metric), load(none, metric), load(method, metric)
    out = {}
    for fold in sorted(set(b) & set(n) & set(m)):
        if min(b[fold], n[fold], m[fold]) < COLLAPSE:
            continue
        out[fold] = (m[fold] - b[fold]) - (n[fold] - b[fold])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", nargs="+", required=True,
                        help="seed:baseline_dir:none_dir:cwe_dir")
    parser.add_argument("--metric", default="test_macro_f1_at_0.5")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    print(f"\n{args.title or 'Gộp qua seed'}  [{args.metric}]")
    header = f"{'seed':>6}{'n fold':>8}{'head phụ':>12}{'dương':>8}   từng fold"
    print(header)
    print("-" * 78)

    means = {}
    for spec in args.specs:
        seed, base, none, method = spec.split(":", 3)
        if not os.path.isdir(method):
            print(f"{seed:>6}{'—':>8}{'chưa chạy':>12}")
            continue
        folds = added_per_fold(base, none, method, args.metric)
        if not folds:
            print(f"{seed:>6}{0:>8}{'chưa có fold':>12}")
            continue
        values = list(folds.values())
        means[seed] = statistics.mean(values)
        print(f"{seed:>6}{len(values):>8}{means[seed]:>+12.4f}"
              f"{sum(v > 0 for v in values):>5}/{len(values):<2}   "
              + ", ".join(f"f{f}{v:+.4f}" for f, v in sorted(folds.items())))

    print("-" * 78)
    if len(means) < 2:
        print(f"\nPHÁN QUYẾT: CHƯA ĐỦ — mới {len(means)} seed. "
              f"Một seed không phân biệt được phương pháp với xổ số seed.")
        return

    values = list(means.values())
    pooled = statistics.mean(values)
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    positive = sum(v > 0 for v in values)
    print(f"\ngộp {len(values)} seed: trung bình {pooled:+.4f}   "
          f"sd giữa seed {spread:.4f}   dương {positive}/{len(values)} seed")

    if positive == len(values) and pooled > NOISE:
        verdict = (f"ĐỨNG VỮNG — dương ở mọi seed và trung bình {pooled:+.4f} "
                   f"vượt ngưỡng nhiễu {NOISE}.")
    elif positive == 0:
        verdict = "BỊ BÁC — âm ở mọi seed."
    elif spread > abs(pooled):
        verdict = (f"KHÔNG KẾT LUẬN ĐƯỢC — sd giữa seed {spread:.4f} lớn hơn hiệu ứng "
                   f"{abs(pooled):.4f}. Đây là xổ số seed, không phải phương pháp. "
                   f"Thêm seed KHÔNG gỡ được: nó thu hẹp sai số của trung bình chứ "
                   f"không thu hẹp dải mà một lần chạy đơn lẻ có thể rơi vào.")
    else:
        verdict = (f"HỖN HỢP — dương {positive}/{len(values)} seed. "
                   f"Chưa đủ để tuyên bố, chưa đủ để bỏ.")
    print(f"\nPHÁN QUYẾT: {verdict}")


if __name__ == "__main__":
    main()
