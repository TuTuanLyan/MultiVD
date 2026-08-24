#!/usr/bin/env python3
"""So hai nhánh bằng cách GHÉP CẶP THEO FOLD, không bao giờ bằng hai trung bình.

Vì sao công cụ này tồn tại: bốn lỗi trong tài liệu dự án đều cùng một dạng —
một quy ước gộp hoặc loại trừ được chọn SAU KHI nhìn số.

* `common` "hơn `full` gấp mười lần" — hai trung bình ở n=3 và n=5. Ghép cặp:
  thắng 2/4 fold, và một fold duy nhất mang toàn bộ hiệu ứng.
* §48 "bỏ RecAdam tốt hơn ở 7/8 nhánh" — ghép cặp thì 6/8 nằm trong nhiễu fold,
  và toàn bộ biên độ đến từ một nhánh bị sập.
* Quy ước `*` bỏ fold có Macro-F1 < 0.55 biến −0.2345 (n=5) thành −0.1156 (n=2),
  tức nó che đúng hiện tượng đang cần thấy.

Nên công cụ này bắt buộc ba thứ và không cho tắt:

1. chỉ dùng fold có mặt ở CẢ HAI nhánh, và in ra n;
2. in cả bản ĐỦ FOLD lẫn bản đã loại fold sập, cạnh nhau, để quy ước loại trừ
   không thể chọn sau khi nhìn kết quả;
3. in số fold dương, sd, và giá trị sau khi bỏ fold có |Δ| lớn nhất — với sd
   giữa các fold tới 0.09 trên tập test 152 mẫu, một fold mang được cả kết luận.

Đọc từ records/results_all.jsonl (dựng bằng src/build_records.py).
"""

import argparse
import json
import os
from collections import defaultdict

COLLAPSE = 0.55       # Macro-F1 dưới mức này: nhánh đó gần như chắc chắn hỏng
RECORDS = "records/results_all.jsonl"


def load(path=RECORDS, metric="test_macro_f1_at_0.5"):
    if not os.path.exists(path):
        raise SystemExit(f"chua co {path} — chay: python src/build_records.py")
    table = defaultdict(dict)
    meta = {}
    for line in open(path):
        row = json.loads(line)
        if row.get(metric) is None:
            continue
        key = (row["run"], row["arm"], row["seed"])
        table[key][row["fold"]] = float(row[metric])
        meta.setdefault(key, {"fold_set": row.get("fold_set"),
                              "model": row.get("hp_model_name"),
                              "pooling": row.get("hp_pooling"),
                              "lambda": row.get("hp_lambda_cwe"),
                              "optimizer": row.get("hp_phase2_optimizer"),
                              "data_root": row.get("hp_data_root")})
    return table, meta


def describe(deltas):
    """n, trung bình, số fold dương, sd, và giá trị sau khi bỏ fold cực đoan."""
    values = [v for _, v in deltas]
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    positive = sum(1 for v in values if v > 0)
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else float("nan")
    if n > 1:
        extreme = max(range(n), key=lambda i: abs(values[i] - mean))
        rest = [v for i, v in enumerate(values) if i != extreme]
        without = sum(rest) / len(rest)
    else:
        without = float("nan")
    return {"n": n, "mean": mean, "pos": positive, "sd": sd, "without_extreme": without,
            "flips": n > 1 and (mean > 0) != (without > 0)}


def wilcoxon(values):
    """Wilcoxon signed-rank hai phía.

    Lưu ý về giới hạn cứng: với n=5 fold, p nhỏ nhất mà kiểm định này CÓ THỂ trả
    về là 2/2^5 = 0.0625. Nên một dòng 5 fold không bao giờ đạt p<0.05, dù hiệu
    ứng lớn đến đâu — p=0.0625 ở n=5 nghĩa là "dương ở cả 5 fold", không phải
    "gần có ý nghĩa". Muốn xuống dưới 0.05 phải có ít nhất 6 quan sát ghép cặp,
    tức thêm seed hoặc thêm fold.
    """
    try:
        from scipy.stats import wilcoxon as w
    except ImportError:
        return None
    if len(values) < 5 or all(v == 0 for v in values):
        return None
    try:
        return float(w(values).pvalue)
    except Exception:
        return None


def pair(a_scores, b_scores, drop_collapsed):
    """Vector hiệu theo fold, chỉ trên fold có mặt ở CẢ HAI nhánh."""
    folds = sorted(set(a_scores) & set(b_scores))
    out = []
    for fold in folds:
        if drop_collapsed and (a_scores[fold] < COLLAPSE or b_scores[fold] < COLLAPSE):
            continue
        out.append((fold, b_scores[fold] - a_scores[fold]))
    return out


def line(label, deltas, width=34):
    stats = describe(deltas)
    if stats is None:
        return f"  {label:<{width}}  (không có fold chung)"
    p = wilcoxon([v for _, v in deltas])
    flag = "  ← ĐỔI DẤU khi bỏ 1 fold" if stats["flips"] else ""
    p_text = f"  p={p:.4f}" if p is not None else ""
    return (f"  {label:<{width}}  n={stats['n']}  Δ {stats['mean']:+.4f}  "
            f"+{stats['pos']}/{stats['n']}  sd {stats['sd']:.4f}  "
            f"bỏ 1 fold {stats['without_extreme']:+.4f}{p_text}{flag}")


def check_comparable(meta, key_a, key_b):
    """Cảnh báo khi hai nhánh khác nhau ở thứ KHÔNG phải biến đang so."""
    warnings = []
    a, b = meta.get(key_a, {}), meta.get(key_b, {})
    for field, name in [("fold_set", "bộ fold"), ("data_root", "thư mục target"),
                        ("model", "backbone"), ("pooling", "pooling")]:
        if a.get(field) and b.get(field) and a[field] != b[field]:
            warnings.append(f"{name}: {a[field]} vs {b[field]}")
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="in mọi nhánh của run này, Δ so với baseline của chính nó")
    parser.add_argument("--vs", nargs=2, metavar=("A", "B"),
                        help="ghép cặp hai nhánh, dạng run/arm — hiệu là B trừ A")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="test_macro_f1_at_0.5")
    parser.add_argument("--records", default=RECORDS)
    args = parser.parse_args()

    table, meta = load(args.records, args.metric)

    if args.vs:
        (run_a, arm_a), (run_b, arm_b) = (s.split("/", 1) for s in args.vs)
        key_a, key_b = (run_a, arm_a, args.seed), (run_b, arm_b, args.seed)
        for key in (key_a, key_b):
            if key not in table:
                raise SystemExit(f"khong co {key[0]}/{key[1]} seed {key[2]}")
        print(f"\n[{args.metric}]  seed {args.seed}")
        print(f"  A = {args.vs[0]}   B = {args.vs[1]}   (hiệu = B − A)\n")
        for warning in check_comparable(meta, key_a, key_b):
            print(f"  ⚠ hai nhánh khác nhau ở {warning} — không chỉ một biến")
        print(line("đủ fold", pair(table[key_a], table[key_b], False)))
        print(line("bỏ fold sập (<0.55)", pair(table[key_a], table[key_b], True)))
        a, b = table[key_a], table[key_b]
        print("\n  theo từng fold:")
        for fold in sorted(set(a) & set(b)):
            mark = "  (sập)" if min(a[fold], b[fold]) < COLLAPSE else ""
            print(f"    fold {fold}   A {a[fold]:.4f}   B {b[fold]:.4f}   "
                  f"hiệu {b[fold]-a[fold]:+.4f}{mark}")
        print()
        return

    if not args.run:
        runs = sorted({f"{run}  (seed {seed})" for run, arm, seed in table if arm == "baseline"})
        print("\ncần --run hoặc --vs. Các run có baseline:\n")
        for name in runs:
            print("   ", name)
        print()
        return

    base_key = (args.run, "baseline", args.seed)
    if base_key not in table:
        raise SystemExit(f"khong co baseline cho {args.run} seed {args.seed}")
    base = table[base_key]
    arms = sorted(arm for run, arm, seed in table
                  if run == args.run and seed == args.seed and arm != "baseline")

    # baseline không có λ hay optimizer của Phase 2 (nó không chạy Phase 2), nên
    # lấy hai trường đó từ một nhánh transfer bất kỳ của cùng run.
    info = dict(meta[base_key])
    for arm in arms:
        other = meta.get((args.run, arm, args.seed), {})
        for field in ("lambda", "optimizer"):
            if info.get(field) is None:
                info[field] = other.get(field)
    print(f"\n[{args.metric}]  {args.run}  seed {args.seed}")
    print(f"  backbone {info['model']} · pooling {info['pooling']} · "
          f"bộ fold {info['fold_set']} · λ {info['lambda']} · phase2 {info['optimizer']}")
    print(f"  baseline trung bình {sum(base.values())/len(base):.4f} trên {len(base)} fold\n")
    none_key = (args.run, "transfer_none", args.seed)
    for arm in arms:
        key = (args.run, arm, args.seed)
        print(line(arm.replace("transfer_", "") + "  vs baseline",
                   pair(base, table[key], False)))
    if none_key in table:
        print("\n  giá trị gia tăng của head phụ — Δ(nhánh) − Δ(none), ghép cặp theo fold:")
        for arm in arms:
            if arm == "transfer_none":
                continue
            key = (args.run, arm, args.seed)
            print(line(arm.replace("transfer_", "") + "  vs none",
                       pair(table[none_key], table[key], False)))
    else:
        print("\n  ⚠ thiếu nhánh `none` → không tách được 'pretrain có tác dụng'"
              " khỏi 'head phụ có tác dụng'")
    if len(base) == 5:
        print("\n  n=5: p nhỏ nhất Wilcoxon có thể trả về là 0.0625. p=0.0625 ở đây"
              " nghĩa là 'cùng dấu ở cả 5 fold', không phải 'gần có ý nghĩa'.")
    print()


if __name__ == "__main__":
    main()
