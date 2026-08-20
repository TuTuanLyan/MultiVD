#!/usr/bin/env python3
"""Bảng theo dõi xu hướng cho run/gated.sh: mọi backbone và phương pháp, từng fold.

Mỗi Δ quy về baseline của **chính backbone đó**, ghép cặp theo từng fold. So Δ của
backbone này với baseline của backbone kia là so nhầm biến, nên bảng không bao giờ
gộp hai backbone vào một baseline chung.

Với `--gate`, in thêm phần đọc xu hướng: mỗi cấu hình được xếp vào một trong ba
nhóm để quyết định có chạy tiếp hay không.

Ngưỡng lấy từ nhiễu đã đo trong dự án này, không phải quy ước chung:
độ lệch chuẩn giữa các fold lên tới 0.09 trên tập test 152 mẫu, nên một Δ cỡ
0.00x nằm gọn trong nhiễu và không phải tín hiệu.
"""

import argparse
import glob
import json
import os
import statistics
from pathlib import Path

METRICS = ("test_macro_f1_at_0.5", "test_roc_auc")
NOISE = 0.005  # Δ dưới mức này coi như ngang baseline


def load(directory, metric):
    scores = {}
    for path in glob.glob(f"{directory}/fold*.json"):
        record = json.load(open(path))
        if record.get(metric) is not None:
            scores[int(record["fold"])] = float(record[metric])
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true",
                        help="in thêm phần đọc xu hướng để quyết định dừng hay chạy tiếp")
    parser.add_argument("--run_prefix", default=os.environ.get("RUN_PREFIX", "gate"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    args = parser.parse_args()

    runs = sorted(p for p in glob.glob(f"results/{args.run_prefix}_*") if os.path.isdir(p))
    if not runs:
        print(f"chưa có kết quả nào dưới results/{args.run_prefix}_*")
        return

    verdicts = []
    for metric in METRICS:
        print(f"\n[{metric}]  seed {args.seed}")
        header = f"{'backbone':<12}{'nhánh':<22}" + "".join(f"{'f%d' % f:>9}" for f in range(1, 6)) \
                 + f"{'mean':>9}{'Δ':>10}"
        print(header)
        print("-" * len(header))
        for run in runs:
            label = Path(run).name[len(args.run_prefix) + 1:]
            base = load(f"{run}/baseline/seed_{args.seed}", metric)
            if not base:
                continue
            cells = "".join(f"{base[f]:>9.4f}" if f in base else f"{'--':>9}" for f in range(1, 6))
            print(f"{label:<12}{'baseline':<22}{cells}{statistics.mean(base.values()):>9.4f}{'—':>10}")
            for method_dir in sorted(glob.glob(f"{run}/transfer_*")):
                method = Path(method_dir).name.replace("transfer_", "")
                scores = load(f"{method_dir}/seed_{args.seed}", metric)
                folds = sorted(set(scores) & set(base))
                if not folds:
                    continue
                deltas = [scores[f] - base[f] for f in folds]
                delta = statistics.mean(deltas)
                cells = "".join(f"{scores[f]:>9.4f}" if f in scores else f"{'--':>9}"
                                for f in range(1, 6))
                print(f"{'':<12}{method:<22}{cells}"
                      f"{statistics.mean(scores[f] for f in folds):>9.4f}{delta:>+10.4f}")
                if metric == METRICS[0]:
                    verdicts.append((label, method, delta, deltas))

    if not args.gate:
        return

    print("\n" + "=" * 72)
    print("ĐỌC XU HƯỚNG (theo Macro-F1)")
    print("=" * 72)
    for label, method, delta, deltas in verdicts:
        positive = sum(1 for d in deltas if d > 0)
        n = len(deltas)
        if abs(delta) < NOISE:
            verdict = f"NGANG BASELINE — |Δ| < {NOISE}, nằm trong nhiễu"
        elif delta < 0:
            verdict = "ÂM — bỏ, tìm phương án khác"
        elif positive == n and n >= 3:
            verdict = f"MẠNH — dương {positive}/{n} fold, chạy tiếp"
        elif positive >= (n + 1) // 2:
            verdict = f"YẾU — chỉ {positive}/{n} fold dương, cân nhắc bỏ"
        else:
            verdict = f"KHÔNG NHẤT QUÁN — {positive}/{n} fold dương"
        print(f"  {label:<10} {method:<20} Δ {delta:>+8.4f}   {verdict}")

    print(f"\n  Ngưỡng nhiễu {NOISE} lấy từ dự án này: sd giữa các fold tới 0.09 trên")
    print("  tập test 152 mẫu, nên Δ cỡ 0.00x không phân biệt được với nhiễu.")
    print("  Ba fold chỉ đủ để QUYẾT ĐỊNH DỪNG. Muốn KẾT LUẬN thì phải đủ 5 fold —")
    print("  tài liệu này có bốn lần tín hiệu n=3 đảo dấu ở n=5 (RESULT.md §30.1).")


if __name__ == "__main__":
    main()
