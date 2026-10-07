#!/usr/bin/env python3
"""Tổng hợp khối tab "Kết quả paper" (bậc 3, n = 15 = 5 fold × seed 42 / 1234 / 7): baseline + Pha 1 trộn JS:Py 4:1 ngẫu nhiên,
val Pha 1 chỉ JS (common / 4CWE / full), Pha 2 cột chính RecAdam + ASAM.

In: (1) đối chiếu hyperparameters giữa các ô sẽ ghép cặp; (2) TB ± SD bốn chỉ số; (3) Δ ghép cặp (seed, fold) với baseline, kèm +k/-k
(hoà |Δ| < 1e-3) và p Wilcoxon theo FOLD (TB 3 seed mỗi fold -> 5 cặp, sàn 0,0625); (4) độ nhạy khi bỏ hai ô sập; (5) chấm dự đoán Y1-Y4.
"""
import json
import os
import statistics as st

from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "..", "results")
SEEDS = (42, 1234, 7)
FOLDS = (1, 2, 3, 4, 5)
CFG = {
    "baseline": {42: "baseline", 1234: "baseline_s1234", 7: "baseline_s7"},
    "common": {42: "rasam_jspy41jsval_common", 1234: "rasam_jspy41jsval_common_s1234", 7: "rasam_jspy41jsval_common_s7"},
    "4cwe": {42: "rasam_jspy41jsval_4cwe", 1234: "rasam_jspy41jsval_4cwe_s1234", 7: "rasam_jspy41jsval_4cwe_s7"},
    "full": {42: "rasam_jspy41jsval_full", 1234: "rasam_jspy41jsval_full_s1234", 7: "rasam_jspy41jsval_full_s7"},
}
MET = (("test_roc_auc", "ROC"), ("test_pr_auc", "PR"), ("test_macro_f1_at_0.5", "F1@0,5"), ("test_macro_f1_at_valcal", "F1@val"))
TIE = 1e-3
COLLAPSE = 0.75
# khoá được phép khác giữa các ô cùng cấu hình (đường dẫn, tên, seed, fold)
PATHLIKE = {"seed", "fold", "checkpoint_path", "init_ckpt", "result_path", "run_name", "model_name", "data_root"}


def load(run, k):
    return json.load(open(os.path.join(R, run, "fold%d.json" % k), encoding="utf-8"))


cells = {c: {(s, k): load(m[s], k) for s in SEEDS for k in FOLDS} for c, m in CFG.items()}

print("== 1. Đối chiếu hyperparameters trong từng cấu hình (khác ngoài đường dẫn / seed / fold) ==")
for c, cc in cells.items():
    ref = cc[(42, 1)]["hyperparameters"]
    diff = {}
    for key, d in cc.items():
        hp = d["hyperparameters"]
        assert hp["seed"] == key[0] and hp["fold"] == key[1], (c, key)
        for x in set(ref) | set(hp):
            if x not in PATHLIKE and ref.get(x) != hp.get(x):
                diff.setdefault(x, set()).add((str(ref.get(x)), str(hp.get(x))))
        ini = hp.get("init_ckpt")
        if ini:
            assert "seed_%d/fold%d/" % key in ini, (c, key, ini)
    print("  %-8s %s" % (c, "TRÙNG (15/15)" if not diff else diff))
base_hp, arm_hp = cells["baseline"][(42, 1)]["hyperparameters"], cells["common"][(42, 1)]["hyperparameters"]
print("  baseline vs nhánh (khác):", {x: (base_hp.get(x), arm_hp.get(x)) for x in sorted(set(base_hp) | set(arm_hp))
                                     if x not in PATHLIKE and base_hp.get(x) != arm_hp.get(x)})


def fmt(x):
    return ("%+.4f" % x).replace(".", ",")


def mean_sd(v):
    return "%.4f ± %.4f" % (st.mean(v), st.stdev(v))


print("\n== 2. TB ± SD (n = 15) ==")
print("  %-8s " % "" + "  ".join("%-17s" % n for _, n in MET))
for c, cc in cells.items():
    print("  %-8s " % c + "  ".join("%-17s" % mean_sd([cc[x][m] for x in cc]) for m, _ in MET))


def paired(c, m, drop=()):
    keys = [(s, k) for s in SEEDS for k in FOLDS if (s, k) not in drop]
    d = [cells[c][x][m] - cells["baseline"][x][m] for x in keys]
    pos, neg = sum(v > TIE for v in d), sum(v < -TIE for v in d)
    per_fold = [st.mean(cells[c][(s, k)][m] - cells["baseline"][(s, k)][m] for s in SEEDS if (s, k) not in drop) for k in FOLDS]
    p = wilcoxon(per_fold).pvalue if any(per_fold) else 1.0
    return st.mean(d), pos, neg, len(d), sum(v > 0 for v in per_fold), p


print("\n== 3. Δ ghép cặp (seed, fold) với baseline: TB Δ (+k/-k trên n) | fold dương (TB 3 seed) p ==")
for c in ("common", "4cwe", "full"):
    parts = []
    for m, n in MET:
        mu, pos, neg, cnt, fpos, p = paired(c, m)
        parts.append("%s %s (+%d/-%d n%d) f%d/5 p%.3f" % (n, fmt(mu), pos, neg, cnt, fpos, p))
    print("  %-8s " % c + " | ".join(parts))

collapsed = [(c, s, k, cc[(s, k)]["test_roc_auc"]) for c, cc in cells.items() for (s, k) in cc if cc[(s, k)]["test_roc_auc"] < COLLAPSE]
print("\n== 4. Ô sập (ROC < %.2f): %s" % (COLLAPSE, collapsed))
drop = {(s, k) for _, s, k, _ in collapsed}
print("  Δ ROC khi BỎ mọi cặp chứa ô sập (%s):" % sorted(drop))
for c in ("common", "4cwe", "full"):
    mu, pos, neg, n, fpos, p = paired(c, "test_roc_auc", drop)
    print("  %-8s %s (+%d/-%d n%d) f%d/5 p%.3f" % (c, fmt(mu), pos, neg, n, fpos, p))

print("\n== 5. TB ROC theo seed ==")
seed_mean = {c: {s: st.mean(cells[c][(s, k)]["test_roc_auc"] for k in FOLDS) for s in SEEDS} for c in cells}
for c in cells:
    v = seed_mean[c]
    print("  %-8s " % c + "  ".join("s%d %.4f" % (s, v[s]) for s in SEEDS) + "  | chênh max-min %.4f" % (max(v.values()) - min(v.values())))

print("\n== 6. Chấm dự đoán (ghi TRƯỚC) ==")
new = [(c, s, k) for c in cells for s in (1234, 7) for k in FOLDS]
new_col = [x for x in new if cells[x[0]][(x[1], x[2])]["test_roc_auc"] < COLLAPSE]
print("  Y1 (0/40 ô mới sập): %d/40 sập %s -> %s" % (len(new_col), new_col, "ĐÚNG" if not new_col else "SAI"))
for c in ("common", "4cwe", "full"):
    mu, pos, neg, n, _, _ = paired(c, "test_roc_auc")
    ok = mu >= 0.030 and pos >= 14
    print("  Y2 %-7s TB Δ ROC %s (cần ≥ +0,030), cặp dương %d/15 (cần ≥ 14) -> %s" % (c, fmt(mu), pos, "ĐÚNG" if ok else "SAI"))
for c in ("common", "4cwe", "full"):
    v = seed_mean[c].values()
    r = max(v) - min(v)
    print("  Y3 %-7s chênh TB ROC 3 seed %.4f (cần ≤ 0,015) -> %s" % (c, r, "ĐÚNG" if r <= 0.015 else "SAI"))
b = seed_mean["baseline"]
for s in (1234, 7):
    print("  Y4 baseline s%d %.4f vs s42 %.4f: lệch %s (cần trong ±0,015) -> %s" % (s, b[s], b[42], fmt(b[s] - b[42]),
                                                                                "ĐÚNG" if abs(b[s] - b[42]) <= 0.015 else "SAI"))
