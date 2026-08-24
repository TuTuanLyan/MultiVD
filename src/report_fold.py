#!/usr/bin/env python3
"""Bảng đọc-ngay trong lúc chạy: sau mỗi fold, mọi backbone × nhánh đứng ở đâu.

Chạy trực tiếp trên cây `results/` đang được ghi, không cần dựng records trước —
mục đích là để theo dõi xu hướng sau 2–4 fold và dừng sớm khi nhánh nào rõ ràng
không đi đến đâu.

Mọi Δ ở đây đều GHÉP CẶP THEO FOLD với baseline của chính backbone đó, chỉ trên
những fold đã có ở cả hai. Không có phép trung bình nào trên tập fold khác nhau.

Ba fold đủ để QUYẾT ĐỊNH DỪNG, không đủ để KẾT LUẬN: tài liệu dự án có bốn lần
một tín hiệu ở n=3 co lại hoặc đổi dấu ở n=5.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COLLAPSE = 0.55
CHANCE_AUC = 0.65
NOISE = 0.005          # Δ nhỏ hơn mức này không phân biệt được với nhiễu fold


def load_run(root, prefix, seed, metric):
    """{(run, arm): {fold: giá trị}} cho mọi run bắt đầu bằng prefix."""
    table = defaultdict(dict)
    aucs = defaultdict(dict)
    for path in sorted(glob.glob(f"{root}/{prefix}*/*/seed_{seed}/fold*.json")):
        try:
            raw = json.load(open(path))
        except Exception:
            continue
        if raw.get(metric) is None:
            continue
        parts = path.split(os.sep)
        run, arm = parts[-4], parts[-3]
        table[(run, arm)][int(raw["fold"])] = float(raw[metric])
        if raw.get("test_roc_auc") is not None:
            aucs[(run, arm)][int(raw["fold"])] = float(raw["test_roc_auc"])
    return table, aucs


def paired_delta(base, arm_scores):
    folds = sorted(set(base) & set(arm_scores))
    return [(f, arm_scores[f] - base[f]) for f in folds]


def verdict(deltas):
    """Đọc xu hướng — CHỈ để quyết định chạy tiếp hay dừng, không để kết luận."""
    if not deltas:
        return "chưa có fold chung"
    values = [v for _, v in deltas]
    n, mean = len(values), sum(values) / len(values)
    positive = sum(1 for v in values if v > 0)
    if abs(mean) < NOISE:
        return f"NGANG baseline (|Δ| < {NOISE})"
    if mean > 0 and positive == n:
        return "MẠNH — dương mọi fold, chạy tiếp"
    if mean > 0:
        return f"YẾU — chỉ {positive}/{n} fold dương"
    if positive == 0:
        return "ÂM mọi fold — cân nhắc dừng"
    return f"ÂM — chỉ {positive}/{n} fold dương"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.environ.get("RESULTS_ROOT", "results"))
    parser.add_argument("--prefix", default=os.environ.get("RUN_PREFIX", ""))
    parser.add_argument("--seed", default=os.environ.get("SEED", "42"))
    parser.add_argument("--metric", default="test_macro_f1_at_0.5")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    table, aucs = load_run(args.root, args.prefix, args.seed, args.metric)
    if not table:
        print(f"chua co ket qua nao khop {args.root}/{args.prefix}*  seed {args.seed}")
        return

    folds = list(range(1, args.folds + 1))
    runs = sorted({run for run, _ in table})

    header = f"\n[{args.metric}]  seed {args.seed}"
    print(header)
    print(f"{'backbone':<12}{'nhánh':<22}" + "".join(f"{'f'+str(f):>9}" for f in folds)
          + f"{'mean':>9}{'Δbase':>10}{'+/n':>7}")
    print("-" * (34 + 9 * len(folds) + 26))

    trends = []
    for run in runs:
        label = run.split("_", 1)[-1] if "_" in run else run
        base = table.get((run, "baseline"), {})
        shown = False
        for arm in ["baseline"] + sorted(a for r, a in table if r == run and a != "baseline"):
            scores = table.get((run, arm))
            if not scores:
                continue
            cells = ""
            for fold in folds:
                if fold not in scores:
                    cells += f"{'--':>9}"
                else:
                    value = scores[fold]
                    mark = "*" if value < COLLAPSE else (
                        "~" if aucs.get((run, arm), {}).get(fold, 1) < CHANCE_AUC else " ")
                    cells += f"{value:>8.4f}{mark}"
            mean = sum(scores.values()) / len(scores)
            name = label if not shown else ""
            shown = True
            if arm == "baseline":
                print(f"{name:<12}{'baseline':<22}{cells}{mean:>9.4f}{'—':>10}{'':>7}")
                continue
            deltas = paired_delta(base, scores)
            if deltas:
                values = [v for _, v in deltas]
                delta_mean = sum(values) / len(values)
                positive = sum(1 for v in values if v > 0)
                extra = f"{delta_mean:>+10.4f}{f'{positive}/{len(values)}':>7}"
                trends.append((label, arm.replace("transfer_", ""), delta_mean, deltas))
            else:
                extra = f"{'—':>10}{'':>7}"
            print(f"{name:<12}{arm.replace('transfer_',''):<22}{cells}{mean:>9.4f}{extra}")
        print()

    # Giá trị gia tăng của head phụ: Δ(nhánh) − Δ(none), ghép cặp theo fold.
    print("giá trị gia tăng của head phụ — Δ(nhánh) − Δ(none), ghép cặp theo fold")
    print("-" * 78)
    for run in runs:
        label = run.split("_", 1)[-1] if "_" in run else run
        none = table.get((run, "transfer_none"))
        if not none:
            print(f"  {label:<12} thiếu nhánh none → không tách được giá trị gia tăng")
            continue
        for arm in sorted(a for r, a in table if r == run and a.startswith("transfer_")
                          and a != "transfer_none"):
            deltas = paired_delta(none, table[(run, arm)])
            if not deltas:
                continue
            values = [v for _, v in deltas]
            mean = sum(values) / len(values)
            positive = sum(1 for v in values if v > 0)
            print(f"  {label:<12}{arm.replace('transfer_',''):<22}"
                  f"n={len(values)}  {mean:+.4f}  +{positive}/{len(values)}")
    print()

    print("ĐỌC XU HƯỚNG (chỉ để quyết định chạy tiếp hay dừng)")
    print("-" * 78)
    for label, arm, mean, deltas in trends:
        print(f"  {label:<12}{arm:<22}Δ {mean:+.4f}   {verdict(deltas)}")
    print()
    print("  * = Macro-F1 < 0.55 (nhánh hỏng)   ~ = ROC-AUC < 0.65 (ngang ngẫu nhiên)")
    print("  Ba fold đủ để DỪNG, không đủ để KẾT LUẬN — n=3 đã bốn lần đổi dấu ở n=5.")
    print()


if __name__ == "__main__":
    main()
