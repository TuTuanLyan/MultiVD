#!/usr/bin/env python3
"""Kiểm kê: cấu hình nào ĐÃ có dữ liệu, cấu hình nào THIẾU, và thiếu ở đâu.

Viết vì một run "đã chạy" chưa chắc dùng được. Ba kiểu hỏng đã gặp thật:

* **Thiếu nhánh `none`.** Không có nó thì không tách được "pretrain có tác dụng"
  khỏi "head phụ có tác dụng", mà đó mới là đại lượng cần báo cáo (§23.2).
* **Thiếu fold.** §30.1 ghi bốn lần tín hiệu n=3 đảo dấu ở n=5.
* **Fold sập** về đoán một lớp. Giữ lại thì Δ đo "một nhánh tình cờ hỏng".

Bộ fold suy từ cỡ tập test hàm ý bởi accuracy, vì file kết quả không ghi nó.
"""

import glob
import json
import os
from collections import defaultdict

COLLAPSE = 0.55        # Macro-F1 dưới mức này: gần như chắc chắn hỏng
CHANCE_AUC = 0.65      # ROC-AUC dưới mức này: mô hình gần như không phân biệt được
METRIC = "test_macro_f1_at_0.5"

FOLD_SIZES = {
    "gốc":    {1: 152, 2: 152, 3: 152, 4: 152, 5: 152},
    "twin":   {1: 154, 2: 153, 3: 152, 4: 151, 5: 150},
    "random": {1: 146, 2: 153, 3: 157, 4: 153, 5: 151},
    "JS":     {1: 230, 2: 229, 3: 227, 4: 226, 5: 226},
}


def load(directory):
    out = {}
    for path in glob.glob(f"{directory}/fold*.json"):
        r = json.load(open(path))
        if r.get(METRIC) is not None:
            out[int(r["fold"])] = (float(r[METRIC]), r.get("test_accuracy_at_0.5"),
                                   r.get("test_roc_auc"), r.get("test_precision_at_0.5"),
                                   r.get("test_recall_at_0.5"))
    return out


def fold_set_of(scores):
    if not scores:
        return "?"
    best, hits = "?", 0
    for name, sizes in FOLD_SIZES.items():
        h = sum(1 for f, (_, acc, *_r) in scores.items()
                if acc is not None and f in sizes
                and abs(acc * sizes[f] - round(acc * sizes[f])) < 1e-6)
        if h > hits:
            best, hits = name, h
    return best if hits >= max(1, len(scores) / 2) else "?"


def main():
    roots = ["results", "results_vast", "results_vast2"]
    runs = defaultdict(dict)
    for root in roots:
        for run_dir in sorted(glob.glob(f"{root}/*")):
            if not os.path.isdir(run_dir):
                continue
            for arm_dir in sorted(glob.glob(f"{run_dir}/*/seed_*")):
                arm = os.path.basename(os.path.dirname(arm_dir))
                seed = os.path.basename(arm_dir).replace("seed_", "")
                scores = load(arm_dir)
                if scores:
                    runs[(run_dir, seed)][arm] = scores

    print(f"\n{'run':<44}{'seed':>5}{'bộ fold':>9}  nhánh có / số fold / cảnh báo")
    print("=" * 118)

    usable, partial, unusable = [], [], []
    for (run_dir, seed), arms in sorted(runs.items()):
        base = arms.get("baseline")
        any_scores = base or next(iter(arms.values()))
        fold_set = fold_set_of(any_scores)
        notes, arm_bits = [], []
        for arm in sorted(arms):
            s = arms[arm]
            # Hai kiểu hỏng KHÁC NHAU, và gộp chung là sai:
            #   "đoán một lớp"  -> precision hoặc recall bằng 0, huấn luyện hỏng hẳn
            #   "ngang ngẫu nhiên" -> vẫn đoán cả hai lớp nhưng AUC ~0.5, tức bài toán
            #                         không học được ở cấu hình đó
            one_class = [f for f, (v, _a, _auc, p_, r_) in s.items()
                         if (p_ == 0 or r_ == 0) or v < COLLAPSE and (p_ == 0 or r_ == 0)]
            chance = [f for f, (v, _a, auc, p_, r_) in s.items()
                      if f not in one_class and auc is not None and auc < CHANCE_AUC]
            bit = f"{arm.replace('transfer_', '')}:{len(s)}"
            if one_class:
                bit += f"(đoán-1-lớp f{','.join(map(str, sorted(one_class)))})"
            if chance:
                bit += f"(≈ngẫu-nhiên f{','.join(map(str, sorted(chance)))})"
            arm_bits.append(bit)
            if len(chance) == len(s) and s:
                notes.append(f"nhánh {arm.replace('transfer_', '')} NGANG NGẪU NHIÊN ở MỌI fold")
        methods = [a for a in arms if a.startswith("transfer")]
        if not base:
            notes.append("KHÔNG CÓ BASELINE")
        if not methods:
            notes.append("không có nhánh transfer")
        if base and methods and "transfer_none" not in arms and arms.keys() != {"baseline", "transfer"}:
            notes.append("thiếu nhánh none → không tính được giá trị gia tăng")
        counts = {len(s) for s in arms.values()}
        if counts and max(counts) < 5:
            notes.append(f"chỉ {max(counts)}/5 fold")

        line = (f"{run_dir:<44}{seed:>5}{fold_set:>9}  " + " ".join(arm_bits))
        if notes:
            line += "   ⚠ " + "; ".join(notes)
        print(line)

        if not base or not methods:
            unusable.append((run_dir, seed))
        elif notes:
            partial.append((run_dir, seed))
        else:
            usable.append((run_dir, seed))

    print("\n" + "=" * 118)
    print(f"đầy đủ (baseline + ≥1 nhánh + none + đủ 5 fold): {len(usable)}")
    print(f"thiếu một phần: {len(partial)}")
    print(f"không dùng được (thiếu baseline hoặc thiếu nhánh transfer): {len(unusable)}")


if __name__ == "__main__":
    main()
