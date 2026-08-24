#!/usr/bin/env python3
"""So ba nhánh cùng một baseline: none, nhãn CWE, và nhóm kích thước sửa đổi.

Ba nhánh dùng chung baseline vì baseline không hề thấy dữ liệu source, nên mọi Δ
quy về đúng một mốc trên đúng một máy.

Đại lượng quyết định **không phải** Δ tuyệt đối mà là **Δ(nhánh) − Δ(none)** — giá
trị gia tăng thật của head phụ. §34.3 đo được đại lượng này với nhãn CWE là +0.076
và +0.080 trên hai lớp hiếm của CodeBERT nhưng chỉ −0.014 và +0.012 trên CodeT5+.
Câu hỏi của thí nghiệm này là tín hiệu mới có bù được phần thiếu đó không.

Δ tuyệt đối gây hiểu nhầm ở đây vì bản thân việc pretrain trên source rồi RecAdam
đã cho một phần lợi ích mà không cần head phụ nào (§23.2). Không trừ `none` đi thì
không phân biệt được "head phụ có tác dụng" với "pretrain có tác dụng".
"""

import argparse
import glob
import json
import os
import statistics

METRICS = ("test_macro_f1_at_0.5", "test_roc_auc")
NOISE = 0.005


def load(directory, metric):
    scores = {}
    for path in glob.glob(f"{directory}/fold*.json"):
        record = json.load(open(path))
        if record.get(metric) is not None:
            scores[int(record["fold"])] = float(record[metric])
    return scores


def delta(method_dir, base, metric):
    scores = load(method_dir, metric)
    folds = sorted(set(scores) & set(base))
    if not folds:
        return None, 0
    return statistics.mean(scores[f] - base[f] for f in folds), len(folds)


def per_fold_added(method_dir, none_dir, base, metric):
    """Giá trị gia tăng của head phụ, tính riêng TỪNG fold.

    Trung bình một mình che mất trường hợp một fold gánh cả kết quả. Chuyện đó đã
    xảy ra: khẳng định 'common hơn full gấp mười lần' ở RESULT.md §21 sống nhờ
    đúng một fold và phải rút lại. Bản đầu của hàm dưới đây cũng chỉ nhìn trung
    bình, và nó lập tức phán 'đáng chạy tiếp' cho một cấu hình chỉ dương 1/3 fold.
    """
    m = load(method_dir, metric)
    n = load(none_dir, metric)
    folds = sorted(set(m) & set(n) & set(base))
    return {f: (m[f] - base[f]) - (n[f] - base[f]) for f in folds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = sorted({
        os.path.basename(p)[len("edit_ref_"):]
        for p in glob.glob("results/edit_ref_*") if os.path.isdir(p)
    })
    if not labels:
        print("chưa có kết quả nào dưới results/edit_ref_*")
        return

    for metric in METRICS:
        print(f"\n[{metric}]  seed {args.seed}")
        header = (f"{'backbone':<12}{'nhánh':<26}{'n':>3}{'Δ vs baseline':>16}"
                  f"{'Δ − Δ(none)':>16}")
        print(header)
        print("-" * len(header))
        for label in labels:
            base = load(f"results/edit_ref_{label}/baseline/seed_{args.seed}", metric)
            if not base:
                continue
            d_none, n_none = delta(
                f"results/edit_ref_{label}/transfer_none/seed_{args.seed}", base, metric)
            rows = [
                ("none (pretrain trần)", f"results/edit_ref_{label}/transfer_none/seed_{args.seed}"),
                ("cwe  (nhãn CWE)", f"results/edit_ref_{label}/transfer_cwe/seed_{args.seed}"),
                ("edit (kích thước sửa)", f"results/edit_new_{label}/transfer_cwe/seed_{args.seed}"),
            ]
            first = True
            for name, directory in rows:
                value, n = delta(directory, base, metric)
                if value is None:
                    continue
                added = "—" if name.startswith("none") or d_none is None \
                    else f"{value - d_none:+.4f}"
                print(f"{(label if first else ''):<12}{name:<26}{n:>3}"
                      f"{value:>+16.4f}{added:>16}")
                first = False

    print("\n" + "=" * 68)
    print("ĐỌC KẾT QUẢ — cột cuối mới là cột quyết định")
    print("=" * 68)
    for label in labels:
        base = load(f"results/edit_ref_{label}/baseline/seed_{args.seed}", METRICS[0])
        if not base:
            continue
        d_none, _ = delta(f"results/edit_ref_{label}/transfer_none/seed_{args.seed}",
                          base, METRICS[0])
        d_cwe, _ = delta(f"results/edit_ref_{label}/transfer_cwe/seed_{args.seed}",
                         base, METRICS[0])
        d_edit, n_edit = delta(f"results/edit_new_{label}/transfer_cwe/seed_{args.seed}",
                               base, METRICS[0])
        if None in (d_none, d_cwe, d_edit):
            print(f"  {label:<10} chưa đủ ba nhánh để so")
            continue
        del d_cwe, d_edit   # không dùng: hai trung bình này có thể khác số fold
        # Trên fold ĐÃ GHÉP CẶP, không trừ hai trung bình tính trên số fold khác
        # nhau. Bản trước làm đúng như vậy và cho `edit` một con số so với `none`
        # ở n=5 trong khi `edit` mới có n=3 — cùng lỗi đã khiến khẳng định
        # "gấp mười lần" ở RESULT.md §21 phải rút lại.
        cwe_folds = per_fold_added(
            f"results/edit_ref_{label}/transfer_cwe/seed_{args.seed}",
            f"results/edit_ref_{label}/transfer_none/seed_{args.seed}",
            base, METRICS[0])
        edit_folds = per_fold_added(
            f"results/edit_new_{label}/transfer_cwe/seed_{args.seed}",
            f"results/edit_ref_{label}/transfer_none/seed_{args.seed}",
            base, METRICS[0])
        shared = sorted(set(cwe_folds) & set(edit_folds))
        if not shared:
            print(f"  {label:<10} chưa có fold nào đủ cả ba nhánh")
            continue
        add_cwe = statistics.mean(cwe_folds[f] for f in shared)
        add_edit = statistics.mean(edit_folds[f] for f in shared)
        n_edit = len(shared)
        per_fold = per_fold_added(
            f"results/edit_new_{label}/transfer_cwe/seed_{args.seed}",
            f"results/edit_ref_{label}/transfer_none/seed_{args.seed}",
            base, METRICS[0])
        positive = sum(1 for v in per_fold.values() if v > 0)
        total = len(per_fold)
        drop_best = None
        if total > 1:
            worst_without = sorted(per_fold.values())[:-1]   # bỏ fold tốt nhất
            drop_best = statistics.mean(worst_without)

        if add_edit <= add_cwe + NOISE:
            verdict = "KHÔNG hơn cwe — dừng"
        elif total >= 3 and positive <= total // 2:
            verdict = (f"trung bình dương nhưng chỉ {positive}/{total} fold — "
                       f"KHÔNG nhất quán, chưa đủ để chạy tiếp")
        elif drop_best is not None and drop_best <= 0:
            verdict = (f"trung bình dương nhưng bỏ fold tốt nhất thì còn "
                       f"{drop_best:+.4f} — một fold đang gánh cả kết quả")
        else:
            verdict = f"hơn cwe và dương {positive}/{total} fold — đáng chạy tiếp"

        print(f"  {label:<10} head phụ cộng thêm (trên {n_edit} fold ghép cặp):  "
              f"cwe {add_cwe:+.4f}   edit {add_edit:+.4f}   (dương {positive}/{total})")
        print(f"  {'':<10}   từng fold: "
              + ", ".join(f"f{f} {v:+.4f}" for f, v in sorted(per_fold.items())))
        if drop_best is not None:
            print(f"  {'':<10}   bỏ fold tốt nhất → {drop_best:+.4f}")
        print(f"  {'':<10}   → {verdict}")

    print(f"\n  Ngưỡng nhiễu {NOISE} lấy từ sd giữa các fold của dự án (tới 0.09 trên 152 mẫu).")
    print("  Ba fold chỉ đủ để quyết định DỪNG; muốn kết luận phải đủ 5 fold.")


if __name__ == "__main__":
    main()
