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
    # 07/10 18:4x người dùng: "common MỞ RỘNG chạy đủ n=15" (seed 42 = khối tab Kết quả gốc)
    "common_ext": {42: "rasam_jspy41jsval_common_ext", 1234: "rasam_jspy41jsval_common_ext_s1234", 7: "rasam_jspy41jsval_common_ext_s7"},
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
for c in ("common", "4cwe", "full", "common_ext"):
    parts = []
    for m, n in MET:
        mu, pos, neg, cnt, fpos, p = paired(c, m)
        parts.append("%s %s (+%d/-%d n%d) f%d/5 p%.3f" % (n, fmt(mu), pos, neg, cnt, fpos, p))
    print("  %-8s " % c + " | ".join(parts))

collapsed = [(c, s, k, cc[(s, k)]["test_roc_auc"]) for c, cc in cells.items() for (s, k) in cc if cc[(s, k)]["test_roc_auc"] < COLLAPSE]
print("\n== 4. Ô sập (ROC < %.2f): %s" % (COLLAPSE, collapsed))
drop = {(s, k) for _, s, k, _ in collapsed}
print("  Δ ROC khi BỎ mọi cặp chứa ô sập (%s):" % sorted(drop))
for c in ("common", "4cwe", "full", "common_ext"):
    mu, pos, neg, n, fpos, p = paired(c, "test_roc_auc", drop)
    print("  %-8s %s (+%d/-%d n%d) f%d/5 p%.3f" % (c, fmt(mu), pos, neg, n, fpos, p))

print("\n== 5. TB ROC theo seed ==")
seed_mean = {c: {s: st.mean(cells[c][(s, k)]["test_roc_auc"] for k in FOLDS) for s in SEEDS} for c in cells}
for c in cells:
    v = seed_mean[c]
    print("  %-8s " % c + "  ".join("s%d %.4f" % (s, v[s]) for s in SEEDS) + "  | chênh max-min %.4f" % (max(v.values()) - min(v.values())))

print("\n== 6. Chấm dự đoán (ghi TRƯỚC) ==")
new = [(c, s, k) for c in ("baseline", "common", "4cwe", "full") for s in (1234, 7) for k in FOLDS]
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


# ---------- trang Nhận định (collection insights, doc paper_n15) ----------
num = lambda x, d=4: ("%.*f" % (d, x)).replace(".", ",")
sg = lambda x, d=4: ("%+.*f" % (d, x)).replace(".", ",")
LAB = {"common": "common", "4cwe": "4CWE", "full": "full", "common_ext": "common mở rộng"}
t_mean = {"head": ["cấu hình (n = 15)"] + [n for _, n in MET], "rows": [
    [c] + ["%s ± %s" % (num(st.mean([cells[c][x][m] for x in cells[c]])), num(st.stdev([cells[c][x][m] for x in cells[c]]))) for m, _ in MET] for c in cells]}
t_pair = {"head": ["- baseline, ghép (seed, fold)"] + [n for _, n in MET], "rows": []}
for c in ("common", "4cwe", "full", "common_ext"):
    row = [LAB[c] + " (n = 15)"]
    for m, _ in MET:
        mu, pos, neg, cnt, fpos, p = paired(c, m)
        row.append("%s (+%d/-%d; %d/5 fold, p %s)" % (sg(mu), pos, neg, fpos, num(p, 3)))
    t_pair["rows"].append(row)
for c in ("common", "4cwe", "full", "common_ext"):
    mu, pos, neg, cnt, fpos, p = paired(c, "test_roc_auc", drop)
    t_pair["rows"].append([LAB[c] + " bỏ 2 cặp sập (n = 13), chỉ ROC", "%s (+%d/-%d; %d/5 fold, p %s)" % (sg(mu), pos, neg, fpos, num(p, 3)), "", "", ""])
t_seed = {"head": ["TB ROC theo seed", "42", "1234", "7", "chênh"], "rows": [
    [c] + [num(seed_mean[c][s]) for s in SEEDS] + [num(max(seed_mean[c].values()) - min(seed_mean[c].values()))] for c in cells]}
g_y2 = {c: paired(c, "test_roc_auc") for c in ("common", "4cwe", "full")}
doc = {
    "title": "Kết quả paper n = 15: trộn JS:Py 4:1, val Pha 1 chỉ JS",
    "updated": "2026-10-07 23:4x",
    "question": "Người dùng 07/10 01:4x: \"tạo thêm 1 tab kết quả nữa, gồm baseline + common/4cwe/full · JS:Py 4:1 ..., val Pha 1 chỉ JS chạy lên n = 15\" (seed 42 / 1234 / 7, rút Python ngẫu nhiên theo seed của run).",
    "verdict": "Cả ba nguồn hơn baseline: common +0,060 / 4CWE +0,062 ROC (15/15 cặp, 5/5 fold), full +0,038 (13/15). Hai ô sập (baseline s7 f1 0,502; full s1234 f5 0,570) gánh phần lớn trung bình; bỏ hai cặp đó cả ba còn ~+0,033-0,034 ROC, 5/5 fold. Chưa quyết định giữ / chạy lại hai ô sập.",
    "confidence": "bậc 3 (5 fold × 3 seed, cùng phần cứng A4000 trùng bit); p Wilcoxon tính theo FOLD (TB 3 seed) nên sàn 0,0625",
    "tier": "Bậc 3 (n = 15 = 5 fold × seed 42 / 1234 / 7, TF32, tất định); Δ ghép cặp theo (seed, fold) với baseline cùng seed, cùng fold.",
    "sections": [
        {"heading": "1. TB ± SD (n = 15)", "table": t_mean},
        {"heading": "2. Δ ghép cặp với baseline (+k/-k: số cặp dương/âm, hoà 0,001; fold dương: TB 3 seed mỗi fold)", "table": t_pair,
         "paras": ["Trung vị Δ ROC: common +0,037, 4CWE +0,037, full +0,035. Cặp (seed 7, f1) có baseline sập nên Δ ~+0,43-0,45; cặp (seed 1234, f5) có full sập nên Δ -0,31."]},
        {"heading": "3. Ô sập (ROC < 0,75)", "bullets": [
            "baseline seed 7 fold 1: train loss ở ln2 9 epoch, dừng ep10 với checkpoint ep1, ROC 0,502 (seed 42 / 1234 cùng fold 0,860 / 0,891).",
            "full seed 1234 fold 5: Pha 1 kẹt cả 16 epoch (val JS 0,49-0,54), Pha 2 cũng kẹt, dừng ep17 với checkpoint ep1, ROC 0,570 (seed 42 cùng fold 0,933).",
            "Chạy lại full s1234 f5 với Pha 1 trần 30 epoch (run *_p1e30, lịch LR khác: warmup 25 % của 30 epoch): Pha 1 thoát ln2 ep7-8, chọn ep13 (val JS 0,613); "
            "Pha 2 test ROC 0,908 / PR 0,913 / F1@0,5 0,842. Nếu THAY ô cũ: full n = 15 ROC TB 0,916 -> 0,938, Δ với baseline +0,038 (+13/-2) -> +0,061 (+14/-1). "
            "Ô cũ VẪN GIỮ trong bảng n = 15 tới khi người dùng chọn cách xử lý (luật chạy lại chung cho mọi nhánh / thay + ghi chú / đổi cấu hình cả nhóm / giữ).",
            "Kiểm TẤT ĐỊNH (paper_night2, run *_repro, cấu hình y hệt, 40/40 cờ): TRÙNG TUYỆT ĐỐI - Pha 1 16/16 dòng epoch trùng từng chữ số, Pha 2 13/13, "
            "test ROC 0,5698784722222223 = bản gốc, mảng xác suất trùng từng phần tử. Ô sập là tất định (hai máy A4000 khác nhau cùng kết quả); chỉ đổi cấu hình mới đổi được.",
            "Đang chạy trên paper_night2: Pha 1 RAdam (Liu et al. 2019) thay AdamW, mọi thứ khác như ô cũ (run *_radam), Pha 2 RecAdam + ASAM như cũ."]},
        {"heading": "4. TB ROC theo seed", "table": t_seed,
         "paras": ["Bỏ fold 1, baseline ba seed: 0,906 / 0,909 / 0,913 (chênh 0,007) - độ lệch seed 7 là do ô sập."]},
        {"heading": "5. Chấm dự đoán ghi trước", "bullets": [
            "Y1 SAI: 2/40 ô mới sập (baseline s7 f1, full s1234 f5).",
            "Y2 (TB Δ ROC ≥ +0,030 và ≥ 14/15 cặp dương): common ĐÚNG (%s, %d/15), 4CWE ĐÚNG (%s, %d/15), full SAI (%s, %d/15)." % (
                sg(g_y2["common"][0], 3), g_y2["common"][1], sg(g_y2["4cwe"][0], 3), g_y2["4cwe"][1], sg(g_y2["full"][0], 3), g_y2["full"][1]),
            "Y3 (chênh TB ROC 3 seed ≤ 0,015): common ĐÚNG (0,010), 4CWE ĐÚNG (0,004), full SAI (0,092 - do ô sập).",
            "Y4 (baseline s1234 / s7 trong ±0,015 của s42): s1234 ĐÚNG (+0,008), s7 SAI (-0,066 - do ô sập).",
            "Common MỞ RỘNG (thêm 07/10 tối): N1 ĐÚNG - 0/10 ô Pha 2 mới sập (ô s1234 f4 Pha 1 kẹt cả 16 epoch nhưng Pha 2 thoát muộn, 0,938); "
            "N2 ĐÚNG - TB ROC 0,9405 so với common gốc 0,9378 (Δ ghép cặp +0,003, +10/-5, 3/5 fold); N3 ĐÚNG - so với baseline +0,063, 14/15 cặp dương."]},
    ],
    "next": ["(cần người dùng quyết định) Hai ô sập: giữ nguyên / báo cáo cả hai bản / chạy lại với luật chọn checkpoint khác (chặn chọn khi train loss còn ở ln2)."],
    "sources": ["_FinalPaperExperiment/results/{baseline,rasam_jspy41jsval_<nguồn>}{,_s1234,_s7}/fold<k>.json",
                "_FinalPaperExperiment/meta/insights/paper_n15/build_paper_n15.py (dựng lại trang này)"],
}
json.dump(doc, open(os.path.join(HERE, "paper_n15.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nghi", os.path.join(HERE, "paper_n15.json"))
