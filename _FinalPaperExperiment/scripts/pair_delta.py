#!/usr/bin/env python3
"""Δ ghép cặp theo fold của một run so với một hoặc nhiều run đối chứng, đủ BỐN chỉ số (CLAUDE.md §2b).

    python3 scripts/pair_delta.py <run_a> <ref> [<ref> ...]

Chỉ ghép những fold mà CẢ HAI run đều có kết quả (n in rõ). Hoà khi |Δ| <= 1e-3 (không đếm là cùng dấu).
In kèm các khoá hyperparameters khác nhau giữa hai ô ghép cặp của fold chung đầu tiên (bỏ khoá đường dẫn / tên chạy),
để thấy ngay phép so đổi đúng một biến hay không.
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
METRICS = [("F1@0.5", "test_macro_f1_at_0.5"), ("F1@val", "test_macro_f1_at_valcal"),
           ("ROC", "test_roc_auc"), ("PR", "test_pr_auc")]
TIE = 1e-3
# khoá chỉ là đường dẫn / tên, khác nhau giữa hai run là đương nhiên
SKIP_HP = {"checkpoint_path", "output_dir", "results_dir", "experiment_name", "run_name", "log_file", "save_dir", "result_path"}
# tiền tố riêng từng máy (161 / 158): bỏ đi trước khi so, để cùng một checkpoint ở hai máy không bị đọc thành "khác"
HOST_PREFIXES = ("/drive1/cuongtm/ntat/MultiVD/", "/data/ntat/MultiVD/", "/drive1/cuongtm/models/", "/data/ntat/models/")


def norm(v):
    if isinstance(v, str):
        for p in HOST_PREFIXES:
            if v.startswith(p):
                return v[len(p):]
    return v


def load(run):
    out = {}
    for f in range(1, 6):
        p = os.path.join(ROOT, run, f"fold{f}.json")
        if os.path.exists(p):
            out[f] = json.load(open(p))
    return out


def hp_diff(a, b):
    ha, hb = a.get("hyperparameters", {}), b.get("hyperparameters", {})
    keys = sorted((set(ha) | set(hb)) - SKIP_HP)
    return [(k, ha.get(k), hb.get(k)) for k in keys if norm(ha.get(k)) != norm(hb.get(k))]


def main(run_a, refs):
    A = load(run_a)
    print(f"{run_a}: fold có kết quả {sorted(A)}")
    for f in sorted(A):
        print(f"  f{f}  " + "  ".join(f"{n} {A[f][k]:.4f}" for n, k in METRICS))
    for ref in refs:
        B = load(ref)
        folds = sorted(set(A) & set(B))
        print(f"\n{run_a} − {ref}   (n={len(folds)}, fold {folds})")
        if not folds:
            print("  không ghép được fold nào")
            continue
        diff = hp_diff(A[folds[0]], B[folds[0]])
        if len(diff) > 6:                                   # khác trainer (vd. baseline): chỉ in số khoá, không liệt kê
            print(f"  hp khác {len(diff)} khoá (khác trainer / kiến trúc)")
        else:
            for k, va, vb in diff:
                print(f"  hp khác: {k}: {va!r} vs {vb!r}")
        for name, key in METRICS:
            d = [A[f][key] - B[f][key] for f in folds]
            pos = sum(x > TIE for x in d)
            neg = sum(x < -TIE for x in d)
            per = " ".join(f"{x:+.4f}" for x in d)
            print(f"  {name:7s} TB {sum(d) / len(d):+.4f}  +{pos}/−{neg}/n{len(d)}  "
                  f"min {min(d):+.4f} max {max(d):+.4f}  [{per}]")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
