#!/usr/bin/env python3
"""Nhận định NGẮN: Pha 1 trộn JS:Py 4:1 theo fold (người dùng 06/10 11:5x). Đọc results/ + dữ liệu Pha 1, ghi jspy41.json (tab Nhận định)
và jspy41.md. Chỉ đọc; chạy được khi chưa đủ 10 ô (chấm dự đoán ghi "chưa đủ")."""
import json, os, statistics as st, sys
import numpy as np
from sklearn.metrics import roc_auc_score
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(BASE, "..", "data", "mwonly5_sources")
K = [("test_roc_auc", "ROC-AUC"), ("test_pr_auc", "PR-AUC"), ("test_macro_f1_at_0.5", "F1@0,5"), ("test_macro_f1_at_valcal", "F1@ngưỡng val")]
TIE = 1e-3
num = lambda x, d=3: "-" if x is None else ("%.*f" % (d, x)).replace(".", ",")
sgn = lambda x, d=3: "±0,000" if abs(x) < 0.5 * 10 ** -d else ("%+.*f" % (d, x)).replace(".", ",")


def res(run, k):
    p = os.path.join(BASE, "results", run, "fold%d.json" % k)
    return json.load(open(p)) if os.path.exists(p) else None


def p1_lang(lvl, k):
    p = os.path.join(BASE, "results", "p1_jspy41_%s" % lvl, "fold%d.probs.npz" % k)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    rows = [json.loads(l) for l in open(os.path.join(DATA, "js_py41_%s" % lvl, "fold%d" % k, "test.jsonl"), encoding="utf-8")]
    y = np.array([r["label"] for r in rows]); lang = np.array([r["lang"] for r in rows]); pr = z["probabilities"]
    assert (z["labels"] == y).all()
    return roc_auc_score(y, pr), roc_auc_score(y[lang == "js"], pr[lang == "js"]), roc_auc_score(y[lang == "python"], pr[lang == "python"])


rows, per_fold, n_done, grades_j3, grades_j4 = [], [], 0, {}, []
t_p1 = {"head": ["Pha 1", "f1", "f2", "f3", "f4", "f5"], "rows": []}
p1_all = []
for lvl in ("common", "full"):
    F = [k for k in range(1, 6) if res("rasam_jspy41_%s" % lvl, k)]
    n_done += len(F)
    A = {k: res("rasam_jspy41_%s" % lvl, k) for k in F}; B = {k: res("rasam_%s_jsonly" % lvl, k) for k in F}
    cells = []
    for key, _ in K:
        d = [A[k][key] - B[k][key] for k in F]
        cells.append("%s (+%d/-%d)" % (sgn(st.mean(d)), sum(v > TIE for v in d), sum(v < -TIE for v in d)) if d else "-")
        if key == "test_roc_auc":
            grades_j3[lvl] = d
    rows.append([lvl, str(len(F))] + cells)
    per_fold.append([lvl] + [sgn(A[k]["test_roc_auc"] - B[k]["test_roc_auc"]) if k in A else "-" for k in range(1, 6)])
    grades_j4 += [A[k]["test_roc_auc"] for k in F]
    L = [p1_lang(lvl, k) for k in range(1, 6)]
    p1_all += [x for x in L if x]
    t_p1["rows"].append(["%s: val trộn / JS / Python" % lvl] + ["%s / %s / %s" % tuple(num(v) for v in x) if x else "-" for x in L])
t_pair = {"head": ["Pha 1 trộn - chỉ JS (cột chính, ghép theo fold)", "n"] + [lab for _, lab in K], "rows": rows}
t_fold = {"head": ["ROC theo fold", "f1", "f2", "f3", "f4", "f5"], "rows": per_fold}

complete = n_done == 10
def g(ok, txt, need):
    return ("ĐÚNG" if ok else "SAI") + ": " + txt if need else "chưa đủ: " + txt
same = lambda d: all(v > TIE for v in d) or all(v < -TIE for v in d)
j3txt = "; ".join("%s TB %s, %s" % (l, sgn(st.mean(d)) if d else "-", "cùng dấu %d/%d" % (max(sum(v > TIE for v in d), sum(v < -TIE for v in d)), len(d)) if d else "-")
                  for l, d in grades_j3.items())
grades = [
    "J1 " + g(all(x[0] >= 0.70 for x in p1_all), "val ROC Pha 1 (trộn) ≥ 0,70 ở %d/%d checkpoint" % (sum(x[0] >= 0.70 for x in p1_all), len(p1_all)), len(p1_all) == 10),
    "J2 " + g(sum(x[2] >= 0.75 for x in p1_all) >= 8, "phần Python của val Pha 1 ROC ≥ 0,75 ở %d/%d checkpoint" % (sum(x[2] >= 0.75 for x in p1_all), len(p1_all)), len(p1_all) == 10),
    "J3 " + g(all(abs(st.mean(d)) <= 0.015 and not same(d) for d in grades_j3.values()),
              "Pha 2 so với chỉ JS: ROC TB trong ±0,015 và không 5/5 cùng dấu ở cả hai mức (%s)" % j3txt, complete),
    "J4 " + g(all(x >= 0.75 for x in grades_j4), "0/10 ô Pha 2 sập (min ROC %s)" % num(min(grades_j4) if grades_j4 else None), complete),
]
gj3 = {l: (st.mean(d), sum(v > TIE for v in d), sum(v < -TIE for v in d), len(d)) for l, d in grades_j3.items() if d}
verdict = ("Đã xong %d/10 ô Pha 2. " % n_done) + ("Trộn 211-267 hàm SVEN train của fold vào Pha 1 cho ROC cột chính so với Pha 1 chỉ JS: common %s (+%d/-%d), full %s (+%d/-%d); "
            "không fold nào âm nhưng chỉ vài fold vượt sàn nhiễu 0,010. Pha 1 học phần Python rất nhanh (ROC trên SVEN val 0,89-0,95) còn phần JS gần ngẫu nhiên "
            "(0,54-0,63), và checkpoint Pha 1 được chọn trên val có SVEN val - chính là val Pha 2. Tách ở mục 9: chọn checkpoint trên val CHỈ JS với cùng train "
            "không đổi ROC TB (+0,001), nên val đích KHÔNG phải nguồn của lợi TB. Còn mở: lợi đến từ chuyển giao hay chỉ từ việc Pha 1 thấy dữ liệu TRAIN của đích "
            "(phép tách Python-only ở mục Đề xuất); chưa dùng làm bằng chứng cho transfer." % (
                sgn(gj3["common"][0]), gj3["common"][1], gj3["common"][2], sgn(gj3["full"][0]), gj3["full"][1], gj3["full"][2])
            if "common" in gj3 and "full" in gj3 else "")
# 06/10 16:0x tỷ lệ 3:1 chỉ common (người dùng: "thử với common, train tỷ lệ 3:1 để kiểm tra tác động tỷ lệ nguồn và đích cho hướng này")
ratio_sections = []
if all(res("rasam_jspy31_common", k) for k in range(1, 6)):
    A3 = [res("rasam_jspy31_common", k) for k in range(1, 6)]
    t5 = {"head": ["common, ghép theo fold", "n"] + [lab for _, lab in K], "rows": []}
    dd = {}
    for ref, lab in (("rasam_jspy41_common", "3:1 - 4:1"), ("rasam_common_jsonly", "3:1 - chỉ JS")):
        B3 = [res(ref, k) for k in range(1, 6)]
        cells = []
        for key, _ in K:
            d = [a[key] - b[key] for a, b in zip(A3, B3)]
            cells.append("%s (+%d/-%d)" % (sgn(st.mean(d)), sum(v > TIE for v in d), sum(v < -TIE for v in d)))
            if key == "test_roc_auc": dd[ref] = d
        t5["rows"].append([lab, "5"] + cells)
    def lang31(k):
        import numpy as _np
        z = _np.load(os.path.join(BASE, "results", "p1_jspy31_common", "fold%d.probs.npz" % k))
        rows_ = [json.loads(l) for l in open(os.path.join(DATA, "js_py31_common", "fold%d" % k, "test.jsonl"), encoding="utf-8")]
        y = _np.array([r["label"] for r in rows_]); lg = _np.array([r["lang"] for r in rows_]); pr = z["probabilities"]
        assert (z["labels"] == y).all()
        return roc_auc_score(y[lg == "python"], pr[lg == "python"])
    py31 = [lang31(k) for k in range(1, 6)]; py41 = [p1_lang("common", k)[2] for k in range(1, 6)]
    d41, dj = dd["rasam_jspy41_common"], dd["rasam_common_jsonly"]
    nonneg = sum(v >= -TIE for v in dj)
    ratio_sections = [{"heading": "5. Tỷ lệ 3:1 (common, n = 5): 842 JS + 281 Python",
        "paras": ["ROC cột chính 3:1: %s. Đổi 4:1 → 3:1 (thêm 70 hàm Python) KHÔNG đổi thứ hạng theo hướng nào: 3:1 - 4:1 ROC %s (+%d/-%d), theo fold %s. So với chỉ JS: %s (+%d/-%d). Phần Python của val Pha 1: 3:1 - 4:1 theo fold %s." % (
            " / ".join(num(a["test_roc_auc"]) for a in A3), sgn(st.mean(d41)), sum(v > TIE for v in d41), sum(v < -TIE for v in d41), " / ".join(sgn(v) for v in d41),
            sgn(st.mean(dj)), sum(v > TIE for v in dj), sum(v < -TIE for v in dj), " / ".join(sgn(a - b) for a, b in zip(py31, py41)))],
        "table": t5,
        "bullets": ["K1 %s: 3:1 - 4:1 ROC TB %s trong ±0,010, không 5/5 cùng dấu." % ("ĐÚNG" if abs(st.mean(d41)) <= 0.010 and not same(d41) else "SAI", sgn(st.mean(d41))),
                    "K2 %s: 3:1 - chỉ JS ROC TB %s ≥ 0, %d/5 fold không âm (hoà 0,001)." % ("ĐÚNG" if st.mean(dj) >= 0 and nonneg >= 4 else "SAI", sgn(st.mean(dj)), nonneg),
                    "K3 %s: phần Python của val Pha 1 cao hơn 4:1 ở %d/5 fold (ngưỡng ≥ 3/5)." % ("ĐÚNG" if sum(a > b for a, b in zip(py31, py41)) >= 3 else "SAI", sum(a > b for a, b in zip(py31, py41))),
                    "K4 %s: 0/5 ô sập (min ROC %s)." % ("ĐÚNG" if min(a["test_roc_auc"] for a in A3) >= 0.75 else "SAI", num(min(a["test_roc_auc"] for a in A3)))]}]
# 06/10 17:5x chỉ ASAM trên Pha 1 trộn 4:1 (người dùng: "khi 2 máy dưới xong thì chạy thêm 4:1 chỉ ASAM")
asam_sections = []
if all(res("asamonly_jspy41_%s" % l, k) for l in ("common", "full") for k in range(1, 6)):
    t6 = {"head": ["chỉ ASAM trộn 4:1, ghép theo fold", "n"] + [lab for _, lab in K], "rows": []}
    rocd = {}
    for lvl in ("common", "full"):
        A6 = [res("asamonly_jspy41_%s" % lvl, k) for k in range(1, 6)]
        for ref, lab in (("asamonly_%s_jsonly" % lvl, "%s: - chỉ ASAM chỉ JS" % lvl), ("rasam_jspy41_%s" % lvl, "%s: - cột chính trộn" % lvl)):
            B6 = [res(ref, k) for k in range(1, 6)]
            cells = []
            for key, _ in K:
                d = [a[key] - b[key] for a, b in zip(A6, B6)]
                cells.append("%s (+%d/-%d)" % (sgn(st.mean(d)), sum(v > TIE for v in d), sum(v < -TIE for v in d)))
                if key == "test_roc_auc": rocd[(lvl, ref.split("_")[0])] = d
            t6["rows"].append([lab, "5"] + cells)
    mins = min(res("asamonly_jspy41_%s" % l, k)["test_roc_auc"] for l in ("common", "full") for k in range(1, 6))
    b_ok = {l: st.mean(rocd[(l, "asamonly")]) >= 0 and sum(v >= -TIE for v in rocd[(l, "asamonly")]) >= 4 for l in ("common", "full")}
    c_ok = {l: abs(st.mean(rocd[(l, "rasam")])) <= 0.010 and not same(rocd[(l, "rasam")]) for l in ("common", "full")}
    asam_sections = [{"heading": "6. Chỉ ASAM trên Pha 1 trộn 4:1 (common + full, n = 5 mỗi mức)",
        "paras": ["Cùng checkpoint Pha 1 trộn 4:1, Pha 2 chỉ ASAM thay cho RecAdam + ASAM. ROC theo fold so với chỉ ASAM Pha 1 chỉ JS - common: %s; full: %s. Với chỉ ASAM, lợi của trộn chỉ thấy ở full; common đổi dấu. So với cột chính cùng Pha 1 trộn, chỉ ASAM thấp hơn nhẹ ở cả hai mức (dưới sàn nhiễu)." % (
            " / ".join(sgn(v) for v in rocd[("common", "asamonly")]), " / ".join(sgn(v) for v in rocd[("full", "asamonly")]))],
        "table": t6,
        "bullets": ["A41a %s: 0/10 ô sập (min ROC %s)." % ("ĐÚNG" if mins >= 0.75 else "SAI", num(mins)),
                    "A41b %s: chỉ ASAM trộn - chỉ ASAM chỉ JS ROC TB ≥ 0 và ≥ 4/5 fold không âm ở MỖI mức (common %s, %d/5; full %s, %d/5)." % (
                        "ĐÚNG" if all(b_ok.values()) else "SAI", sgn(st.mean(rocd[("common", "asamonly")])), sum(v >= -TIE for v in rocd[("common", "asamonly")]),
                        sgn(st.mean(rocd[("full", "asamonly")])), sum(v >= -TIE for v in rocd[("full", "asamonly")])),
                    "A41c %s: chỉ ASAM trộn - cột chính trộn |ROC TB| ≤ 0,010, không 5/5 cùng dấu ở mỗi mức (common %s, full %s)." % (
                        "ĐÚNG" if all(c_ok.values()) else "SAI", sgn(st.mean(rocd[("common", "rasam")])), sgn(st.mean(rocd[("full", "rasam")])))]}]
def lang_roc(p1run, ddir, k):
    """(val trộn, JS, Python) ROC của checkpoint Pha 1 p1run fold k trên val Pha 1 của ddir."""
    z = np.load(os.path.join(BASE, "results", p1run, "fold%d.probs.npz" % k))
    rows_ = [json.loads(l) for l in open(os.path.join(DATA, ddir, "fold%d" % k, "test.jsonl"), encoding="utf-8")]
    y = np.array([r["label"] for r in rows_]); lg = np.array([r["lang"] for r in rows_]); pr = z["probabilities"]
    assert (z["labels"] == y).all()
    return roc_auc_score(y, pr), roc_auc_score(y[lg == "js"], pr[lg == "js"]), roc_auc_score(y[lg == "python"], pr[lg == "python"])


def pair_table(run, refs):
    """Bảng ghép cặp 4 chỉ số của run với từng đối chứng (ref, nhãn); trả (bảng, {ref: list Δ ROC})."""
    A_ = [res(run, k) for k in range(1, 6)]
    t = {"head": ["ghép theo fold", "n"] + [lab for _, lab in K], "rows": []}
    roc = {}
    for ref, lab in refs:
        B_ = [res(ref, k) for k in range(1, 6)]
        cells = []
        for key, _ in K:
            d = [a[key] - b[key] for a, b in zip(A_, B_)]
            cells.append("%s (+%d/-%d)" % (sgn(st.mean(d)), sum(v > TIE for v in d), sum(v < -TIE for v in d)))
            if key == "test_roc_auc": roc[ref] = d
        t["rows"].append([lab, "5"] + cells)
    return A_, t, roc


# 06/10 17:5x tỷ lệ 3:1 trên full (người dùng: "tôi đã thuê thêm paper_mw4 trên vast chạy 3:1 trên full nhé")
full31_sections = []
if all(res("rasam_jspy31_full", k) and res("p1_jspy31_full", k) for k in range(1, 6)):
    A7, t7, r7 = pair_table("rasam_jspy31_full", (("rasam_jspy41_full", "3:1 full - 4:1 full"), ("rasam_full_jsonly", "3:1 full - chỉ JS full")))
    L7 = [lang_roc("p1_jspy31_full", "js_py31_full", k) for k in range(1, 6)]
    d41, dj = r7["rasam_jspy41_full"], r7["rasam_full_jsonly"]
    nonneg = sum(v >= -TIE for v in dj)
    full31_sections = [{"heading": "7. Tỷ lệ 3:1 trên full (n = 5): 1 066 JS + 355 Python",
        "paras": ["ROC cột chính 3:1 full: %s (f1-f3 vast paper_mw4, f4 161, f5 158; kiểm máy trước: 3 epoch đầu trùng từng chữ số log 161). "
                  "So với 4:1 full theo fold %s; so với chỉ JS full theo fold %s. Pha 1 f5 KẸT suốt 16 epoch (val ROC 0,540; JS %s, Python %s) "
                  "nhưng Pha 2 vẫn đạt %s - thấp hơn chỉ JS cùng fold %s. Như 3:1 common (mục 5): đổi tỉ lệ không cho lợi thêm." % (
            " / ".join(num(a["test_roc_auc"]) for a in A7), " / ".join(sgn(v) for v in d41), " / ".join(sgn(v) for v in dj),
            num(L7[4][1]), num(L7[4][2]), num(A7[4]["test_roc_auc"]), sgn(dj[4])),
                  "Pha 1 (val trộn / JS / Python) theo fold: " + " ; ".join("f%d %s / %s / %s" % ((k + 1,) + tuple(num(v) for v in L7[k])) for k in range(5)) + "."],
        "table": t7,
        "bullets": ["F1 ĐÚNG: máy paper_mw4 trùng 161 ở mọi dòng Epoch đã kiểm (3 epoch Pha 1 4:1 full f1).",
                    "F2 %s: 3:1 full - 4:1 full ROC TB %s trong ±0,010, không 5/5 cùng dấu." % ("ĐÚNG" if abs(st.mean(d41)) <= 0.010 and not same(d41) else "SAI", sgn(st.mean(d41))),
                    "F3 %s: 3:1 full - chỉ JS full ROC TB %s ≥ 0, %d/5 fold không âm (cần ≥ 4/5)." % ("ĐÚNG" if st.mean(dj) >= 0 and nonneg >= 4 else "SAI", sgn(st.mean(dj)), nonneg),
                    "F4 %s: 0/5 ô sập (min ROC %s)." % ("ĐÚNG" if min(a["test_roc_auc"] for a in A7) >= 0.75 else "SAI", num(min(a["test_roc_auc"] for a in A7)))]}]
# 06/10 18:2x 4CWE trộn 4:1 (người dùng: "158 chạy 4CWE pha 1 trộn đi")
cwe_sections = []
if all(res("rasam_jspy41_4cwe", k) and res("p1_jspy41_4cwe", k) for k in range(1, 6)):
    A8, t8, r8 = pair_table("rasam_jspy41_4cwe", (("rasam_4cwe_jsonly", "4CWE trộn 4:1 - 4CWE chỉ JS"),))
    L8 = [lang_roc("p1_jspy41_4cwe", "js_py41_4cwe", k) for k in range(1, 6)]
    d8 = r8["rasam_4cwe_jsonly"]
    nonneg = sum(v >= -TIE for v in d8)
    cwe_sections = [{"heading": "8. 4CWE trộn 4:1 (n = 5): 500 JS + 125 Python, val = 88 JS + 152 SVEN val",
        "paras": ["ROC cột chính: %s; so với 4CWE chỉ JS theo fold %s. Lợi TB lớn hơn common (mục 1) và gần full, nhưng f5 âm. Pha 1: phần JS (88 hàm) %s, phần Python %s. "
                  "Giả thuyết CHƯA kiểm: JS val 4CWE chỉ 88 hàm nên checkpoint Pha 1 gần như được chọn theo phần Python của đích - cùng nghi vấn ở mục Đề xuất." % (
            " / ".join(num(a["test_roc_auc"]) for a in A8), " / ".join(sgn(v) for v in d8),
            "-".join(num(v) for v in (min(x[1] for x in L8), max(x[1] for x in L8))), "-".join(num(v) for v in (min(x[2] for x in L8), max(x[2] for x in L8))))],
        "table": t8,
        "bullets": ["G1 %s: 0/5 ô sập (min ROC %s)." % ("ĐÚNG" if min(a["test_roc_auc"] for a in A8) >= 0.75 else "SAI", num(min(a["test_roc_auc"] for a in A8))),
                    "G2 %s: trộn - chỉ JS ROC TB %s ≥ 0, %d/5 fold không âm (cần ≥ 4/5)." % ("ĐÚNG" if st.mean(d8) >= 0 and nonneg >= 4 else "SAI", sgn(st.mean(d8)), nonneg),
                    "G3 %s: phần Python của val Pha 1 ROC ≥ 0,75 ở %d/5 checkpoint (cần ≥ 4/5)." % ("ĐÚNG" if sum(x[2] >= 0.75 for x in L8) >= 4 else "SAI", sum(x[2] >= 0.75 for x in L8)),
                    "G4 %s: phần JS của val Pha 1 ROC < 0,70 ở %d/5 checkpoint (cần ≥ 4/5)." % ("ĐÚNG" if sum(x[1] < 0.70 for x in L8) >= 4 else "SAI", sum(x[1] < 0.70 for x in L8))]}]

# 06/10 21:0x + 22:0x tách nguồn lợi: val Pha 1 CHỈ JS (người dùng: "chạy thêm 1 bản JS common + 1/4 số lượng của JS hàm SVEN train_k (rút theo cặp).
# pha 1 dùng best roc js" và "thêm 1 nhánh pha 1 trộn ngẫu nhiên 4:1 lấy val js nữa")
sep_sections = []
if all(res(r, k) for r in ("rasam_jspy41pair_common", "rasam_jspy41jsval_common", "p1_jspy41pair_common", "p1_jspy41jsval_common") for k in range(1, 6)):
    A9p, t9p, r9p = pair_table("rasam_jspy41pair_common", (("rasam_common_jsonly", "theo cặp, val JS - chỉ JS"), ("rasam_jspy41_common", "theo cặp, val JS - 4:1 cũ")))
    A9v, t9v, r9v = pair_table("rasam_jspy41jsval_common", (("rasam_common_jsonly", "ngẫu nhiên, val JS - chỉ JS"), ("rasam_jspy41_common", "ngẫu nhiên, val JS - 4:1 cũ (CHỈ khác val)"),
                                                           ("rasam_jspy41pair_common", "ngẫu nhiên - theo cặp (cùng val JS, CHỈ khác cách rút)")))
    t9 = {"head": t9p["head"], "rows": t9p["rows"] + t9v["rows"]}
    vp = [res("p1_jspy41pair_common", k)["test_roc_auc"] for k in range(1, 6)]
    vv = [res("p1_jspy41jsval_common", k)["test_roc_auc"] for k in range(1, 6)]
    nn = lambda d: sum(v >= -TIE for v in d)
    inr = lambda L: sum(0.60 <= v <= 0.70 for v in L)
    pj, p4 = r9p["rasam_common_jsonly"], r9p["rasam_jspy41_common"]
    vj, v4, vpair = r9v["rasam_common_jsonly"], r9v["rasam_jspy41_common"], r9v["rasam_jspy41pair_common"]
    mn = lambda A_: min(a["test_roc_auc"] for a in A_)
    full_rows, full_bul, full_para = [], [], []
    if all(res(r, k) for r in ("rasam_jspy41jsval_full", "p1_jspy41jsval_full") for k in range(1, 6)):
        A9f, t9f, r9f = pair_table("rasam_jspy41jsval_full", (("rasam_full_jsonly", "(c) full, ngẫu nhiên, val JS - chỉ JS full"),
                                                               ("rasam_jspy41_full", "(c) full - 4:1 full cũ (CHỈ khác val)")))
        full_rows = t9f["rows"]
        vf = [res("p1_jspy41jsval_full", k)["test_roc_auc"] for k in range(1, 6)]
        epn = [res("p1_jspy41jsval_full", k)["best_epoch"] for k in range(1, 6)]; epo = [res("p1_jspy41_full", k)["best_epoch"] for k in range(1, 6)]
        fj, f4 = r9f["rasam_full_jsonly"], r9f["rasam_jspy41_full"]
        full_para = ["(c) full (người dùng 07/10 01:2x): train TRÙNG BYTE 4:1 full cũ, val chỉ 188 JS full. Vì train trùng và huấn luyện tất định, quỹ đạo Pha 1 "
                     "giống hệt bản cũ - val chỉ đổi EPOCH được chọn: val chỉ JS chọn ep %s, bản cũ ep %s; fold chọn cùng epoch cho Pha 2 trùng 16 chữ số (Δ = 0). "
                     "ROC (c) %s; - 4:1 full cũ theo fold %s; - chỉ JS full theo fold %s." % (
                         " / ".join(str(e) for e in epn), " / ".join(str(e) for e in epo), " / ".join(num(a["test_roc_auc"]) for a in A9f),
                         " / ".join(sgn(v) for v in f4), " / ".join(sgn(v) for v in fj))]
        full_bul = ["VF1 %s: (c) 0/5 ô sập (min ROC %s)." % ("ĐÚNG" if mn(A9f) >= 0.75 else "SAI", num(mn(A9f))),
                    "VF2 %s: (c) val ROC Pha 1 (chỉ JS full) trong [0,55; 0,70] ở %d/5 fold (cần ≥ 4/5)." % ("ĐÚNG" if sum(0.55 <= v <= 0.70 for v in vf) >= 4 else "SAI", sum(0.55 <= v <= 0.70 for v in vf)),
                    "VF3 %s: (c) - 4:1 full cũ |ROC TB| %s ≤ 0,010, không 5/5 cùng dấu." % ("ĐÚNG" if abs(st.mean(f4)) <= 0.010 and not same(f4) else "SAI", sgn(st.mean(f4))),
                    "VF4 %s: (c) - chỉ JS full ROC TB %s ≥ 0, %d/5 fold không âm (cần ≥ 4/5)." % ("ĐÚNG" if st.mean(fj) >= 0 and nn(fj) >= 4 else "SAI", sgn(st.mean(fj)), nn(fj))]
    sep_sections = [{"heading": "9. Tách nguồn lợi: chọn checkpoint Pha 1 trên val CHỈ JS (common + full, n = 5 mỗi bản)",
        "paras": ["Hai bản mới, Pha 1 đều chọn checkpoint trên 148 JS val (đúng val của Pha 1 chỉ JS), không nhìn val SVEN: (a) theo cặp - 842 JS + 105 cặp SVEN "
                  "(hàm lỗi + bản vá, cả hai nửa ở SVEN train fold k); (b) ngẫu nhiên - train TRÙNG BYTE 4:1 cũ (842 JS + 211 Python cân nhãn), chỉ đổi val. "
                  "ROC (a) %s; (b) %s. Val ROC Pha 1 (chỉ JS): (a) %s; (b) %s." % (
                      " / ".join(num(a["test_roc_auc"]) for a in A9p), " / ".join(num(a["test_roc_auc"]) for a in A9v),
                      " / ".join(num(v) for v in vp), " / ".join(num(v) for v in vv)),
                  "ROC theo fold - (b) - 4:1 cũ (chỉ khác val): %s; (b) - chỉ JS: %s; (a) - chỉ JS: %s. Đổi val chọn checkpoint (JS + SVEN → chỉ JS) với cùng train "
                  "không đổi ROC TB, nên val SVEN không phải nguồn của lợi TB (~ +0,01 ở cả ba bản). Nhưng mọi chênh lệch giữa ba bản đều dưới / sát sàn nhiễu 0,010, "
                  "và F1@ngưỡng val không theo (âm nhẹ ở cả hai bản mới). Câu hỏi còn mở: lợi đến từ chuyển giao hay chỉ từ việc Pha 1 đã thấy dữ liệu TRAIN của đích." % (
                      " / ".join(sgn(v) for v in v4), " / ".join(sgn(v) for v in vj), " / ".join(sgn(v) for v in pj))],
        "table": {"head": t9["head"], "rows": t9["rows"] + full_rows},
        "bullets": ["P1 %s: (a) 0/5 ô sập (min ROC %s)." % ("ĐÚNG" if mn(A9p) >= 0.75 else "SAI", num(mn(A9p))),
                    "P2 %s: (a) val ROC Pha 1 (chỉ JS) trong [0,60; 0,70] ở %d/5 fold (cần ≥ 4/5)." % ("ĐÚNG" if inr(vp) >= 4 else "SAI", inr(vp)),
                    "P3 %s: (a) - chỉ JS ROC TB %s ≥ 0, %d/5 fold không âm (cần ≥ 4/5)." % ("ĐÚNG" if st.mean(pj) >= 0 and nn(pj) >= 4 else "SAI", sgn(st.mean(pj)), nn(pj)),
                    "P4 %s: (a) - 4:1 cũ |ROC TB| %s ≤ 0,010, không 5/5 cùng dấu." % ("ĐÚNG" if abs(st.mean(p4)) <= 0.010 and not same(p4) else "SAI", sgn(st.mean(p4))),
                    "V1 %s: (b) 0/5 ô sập (min ROC %s)." % ("ĐÚNG" if mn(A9v) >= 0.75 else "SAI", num(mn(A9v))),
                    "V2 %s: (b) val ROC Pha 1 (chỉ JS) trong [0,60; 0,70] ở %d/5 fold (cần ≥ 4/5)." % ("ĐÚNG" if inr(vv) >= 4 else "SAI", inr(vv)),
                    "V3 %s: (b) - 4:1 cũ |ROC TB| %s ≤ 0,010, không 5/5 cùng dấu." % ("ĐÚNG" if abs(st.mean(v4)) <= 0.010 and not same(v4) else "SAI", sgn(st.mean(v4))),
                    "V4 %s: (b) - chỉ JS ROC TB %s ≥ 0, %d/5 fold không âm (cần ≥ 4/5)." % ("ĐÚNG" if st.mean(vj) >= 0 and nn(vj) >= 4 else "SAI", sgn(st.mean(vj)), nn(vj)),
                    "V5 %s: (b) - (a) |ROC TB| %s ≤ 0,010, không 5/5 cùng dấu." % ("ĐÚNG" if abs(st.mean(vpair)) <= 0.010 and not same(vpair) else "SAI", sgn(st.mean(vpair)))] + full_bul}]
    sep_sections[0]["paras"] += full_para
# 07/10 00:3x chỉ RecAdam trên Pha 1 trộn 4:1 (người dùng 06/10: "thử thêm các nhánh ASAM only và recadam only của nhánh trộn pha 1"; phần chung của mọi phạm vi)
ra_sections = []
if all(res("raonly_jspy41_%s" % l, k) for l in ("common", "full") for k in range(1, 6)):
    t10 = {"head": ["chỉ RecAdam trộn 4:1, ghép theo fold", "n"] + [lab for _, lab in K], "rows": []}
    rr = {}
    for lvl in ("common", "full"):
        A10 = [res("raonly_jspy41_%s" % lvl, k) for k in range(1, 6)]
        for ref, lab in (("raonly_%s_jsonly" % lvl, "%s: - chỉ RecAdam chỉ JS" % lvl), ("rasam_jspy41_%s" % lvl, "%s: - cột chính trộn" % lvl),
                         ("asamonly_jspy41_%s" % lvl, "%s: - chỉ ASAM trộn" % lvl)):
            B10 = [res(ref, k) for k in range(1, 6)]
            cells = []
            for key, _ in K:
                d = [a[key] - b[key] for a, b in zip(A10, B10)]
                cells.append("%s (+%d/-%d)" % (sgn(st.mean(d)), sum(v > TIE for v in d), sum(v < -TIE for v in d)))
                if key == "test_roc_auc": rr[(lvl, ref.split("_")[0])] = d
            t10["rows"].append([lab, "5"] + cells)
    mn10 = min(res("raonly_jspy41_%s" % l, k)["test_roc_auc"] for l in ("common", "full") for k in range(1, 6))
    b10 = {l: st.mean(rr[(l, "raonly")]) >= 0 and sum(v >= -TIE for v in rr[(l, "raonly")]) >= 4 for l in ("common", "full")}
    c10 = {l: abs(st.mean(rr[(l, "rasam")])) <= 0.010 and not same(rr[(l, "rasam")]) for l in ("common", "full")}
    base_full = st.mean(res("raonly_full_jsonly", k)["test_roc_auc"] for k in range(1, 6))
    ra_sections = [{"heading": "10. Chỉ RecAdam trên Pha 1 trộn 4:1 (common + full, n = 5 mỗi mức)",
        "paras": ["Cùng checkpoint Pha 1 trộn 4:1, Pha 2 chỉ RecAdam. ROC theo fold so với chỉ RecAdam Pha 1 chỉ JS - common: %s; full: %s. Pha 1 trộn nâng chỉ RecAdam "
                  "rõ nhất trong ba optimizer (full 5/5 trên cả 4 chỉ số), một phần vì đối chứng chỉ RecAdam chỉ JS full yếu (ROC TB %s). Nhưng so với cột chính "
                  "cùng Pha 1 trộn vẫn thấp hơn ~0,01 ở cả hai mức; thứ tự trên Pha 1 trộn: cột chính > chỉ ASAM > chỉ RecAdam." % (
            " / ".join(sgn(v) for v in rr[("common", "raonly")]), " / ".join(sgn(v) for v in rr[("full", "raonly")]), num(base_full))],
        "table": t10,
        "bullets": ["R41a %s: 0/10 ô sập (min ROC %s)." % ("ĐÚNG" if mn10 >= 0.75 else "SAI", num(mn10)),
                    "R41b %s: chỉ RecAdam trộn - chỉ RecAdam chỉ JS ROC TB ≥ 0 và ≥ 4/5 fold không âm ở MỖI mức (common %s, %d/5; full %s, %d/5)." % (
                        "ĐÚNG" if all(b10.values()) else "SAI", sgn(st.mean(rr[("common", "raonly")])), sum(v >= -TIE for v in rr[("common", "raonly")]),
                        sgn(st.mean(rr[("full", "raonly")])), sum(v >= -TIE for v in rr[("full", "raonly")])),
                    "R41c %s: chỉ RecAdam trộn - cột chính trộn |ROC TB| ≤ 0,010, không 5/5 cùng dấu ở mỗi mức (common %s, full %s)." % (
                        "ĐÚNG" if all(c10.values()) else "SAI", sgn(st.mean(rr[("common", "rasam")])), sgn(st.mean(rr[("full", "rasam")])))]}]
# 07/10 15:0x người dùng: "tab chung, chạy thêm uncommon 4:1, val Pha 1 chỉ JS" (bản CHẶT)
unc_sections = []
UNC = "rasam_jspy41jsval_uncommon_strict"
if all(res(UNC, k) for k in range(1, 6)):
    A11, t11, r11 = pair_table(UNC, (("baseline", "- baseline"), ("rasam_jspy41jsval_common", "- 4:1 val JS · common"),
                                     ("rasam_jspy41jsval_4cwe", "- 4:1 val JS · 4CWE"), ("rasam_jspy41jsval_full", "- 4:1 val JS · full"),
                                     ("rasam_common_jsonly", "- chỉ JS common (không trộn)")))
    p11 = [res("p1_jspy41jsval_uncommon_strict", k) for k in range(1, 6)]
    unc_sections = [{"heading": "11. Uncommon CHẶT trộn 4:1, val Pha 1 chỉ JS (n = 5): 102 JS + 26 Python, val = 18 JS",
        "paras": ["Nguồn: 120 hàm JS = 60 cặp (CWE-1321 56, CWE-843 4) - JS full bỏ JS common và bỏ mọi hàm có nhãn common hoặc nhãn unknown "
                  "(luật chặt 01/10). ROC theo fold %s (TB %s). Pha 1 gần như không học: val JS ROC %s (dưới 0,5 cả 5 fold), checkpoint ep %s; "
                  "Pha 2 thoát bình nguyên muộn (ep9-18). Hơn baseline ~0,02 nhưng THUA cả ba nguồn khác cùng phương pháp 0/5 fold ở cả 4 chỉ số. "
                  "Uncommon chặt khác các nguồn kia cả CỠ (128 hàng so với 625-1 333) lẫn NHÃN (CWE-1321 / 843 không áp cho Python) - không tách được." % (
                      " / ".join(num(a["test_roc_auc"]) for a in A11), num(st.mean(a["test_roc_auc"] for a in A11)),
                      " / ".join(num(p["test_roc_auc"]) for p in p11), " / ".join(str(p["best_epoch"]) for p in p11))],
        "table": t11,
        "bullets": ["U1 ĐÚNG: Pha 1 chọn ep ≤ 3 ở 3/5 fold (f2, f3, f4).",
                    "U2 ĐÚNG: TB %s trong [0,900; 0,935], hơn baseline %d/5 (f4 hoà), thua 4:1 val JS common 5/5." % (
                        num(st.mean(a["test_roc_auc"] for a in A11)), sum(v > TIE for v in r11["baseline"])),
                    "U3 ĐÚNG: 0/5 ô sập (min %s); cảnh báo f4 16:15 = báo nhầm (thoát muộn, test 0,943)." % num(min(a["test_roc_auc"] for a in A11))]}]

# 07/10 17:0x người dùng: common MỞ RỘNG - giữ hàm có ≥ 1 nhãn thuộc common, giữ cả hàm unknown
ext_sections = []
EXT = "rasam_jspy41jsval_common_ext"
mext = json.load(open(os.path.join(DATA, "js_common_ext", "MANIFEST.json"), encoding="utf-8"))
ext_done = [k for k in range(1, 6) if res(EXT, k)]
ext_paras = ["Luật common GỐC (tools/build_sources_v2.py:subset_common, luật R1-R6 của tools/cwe_rules.py trên cwec_v4.20.xml, đích Python): giữ hàm khi MỌI "
             "nhãn CWE được giữ, nhãn unknown / NVD-CWE-* / rỗng (R1) ⇒ bỏ. Bản MỞ RỘNG: giữ hàm khi có ÍT NHẤT MỘT nhãn được giữ HOẶC có nhãn R1 ⇒ đúng phần bù "
             "của uncommon chặt trong JS full. %s hàm = %d cặp = %d common gốc + %d (lẫn common + unknown 66, lẫn common + CWE khác 18 - vd CWE-79 + CWE-1321, "
             "có unknown 60: 29 cặp toàn NVD-CWE-noinfo + 1 cặp CWE-248 + NVD-CWE-noinfo). Bị bỏ 120 hàm = 60 cặp, 36 repo: CWE-1321 56 cặp (R5: chỉ khai "
             "JavaScript), CWE-843 4 cặp (R5: chỉ khai C / C++); danh sách: data/mwonly5_sources/js_common_ext/DROPPED.csv. Pha 1: %d JS + 241 Python, val = %d JS." % (
                 "{:,}".format(mext["n"]).replace(",", " "), *[int(x) for x in (mext["pairs"], mext["n_common_goc"], mext["n_them"], mext["train"], mext["val"])])]
ext_bullets = ["Dự đoán (ghi trước): E1 ROC TB trong ±0,010 của 4:1 val JS common gốc (0,944) và không 5/5 cùng dấu; E2 hơn baseline 5/5 fold; E3 0/5 ô sập."]
t12 = None
if len(ext_done) == 5:
    A12, t12, r12 = pair_table(EXT, (("baseline", "- baseline"), ("rasam_jspy41jsval_common", "- 4:1 val JS · common gốc"),
                                     ("rasam_jspy41jsval_full", "- 4:1 val JS · full")))
    m12 = st.mean(a["test_roc_auc"] for a in A12)
    ext_paras.append("ROC theo fold %s (TB %s)." % (" / ".join(num(a["test_roc_auc"]) for a in A12), num(m12)))
    dc = r12["rasam_jspy41jsval_common"]
    ext_bullets = ["E1 %s: Δ với common gốc %s (+%d/-%d)." % ("ĐÚNG" if abs(st.mean(dc)) <= 0.010 and not same(dc) else "SAI", sgn(st.mean(dc)),
                                                             sum(v > TIE for v in dc), sum(v < -TIE for v in dc)),
                   "E2 %s: hơn baseline %d/5 fold." % ("ĐÚNG" if sum(v > TIE for v in r12["baseline"]) == 5 else "SAI", sum(v > TIE for v in r12["baseline"])),
                   "E3 %s: min ROC %s." % ("ĐÚNG" if min(a["test_roc_auc"] for a in A12) >= 0.75 else "SAI", num(min(a["test_roc_auc"] for a in A12)))]
else:
    ext_paras.append("ĐANG CHẠY từ 17:07 (161 f1/f3/f5, 158 f2/f4); xong %d/5 fold Pha 2. ROC từng fold đã xong (common mở rộng / common gốc / baseline): %s. "
                     "Chưa kết luận khi chưa đủ 5 fold." % (len(ext_done), "; ".join("f%d %s / %s / %s" % (k, num(res(EXT, k)["test_roc_auc"]),
                         num(res("rasam_jspy41jsval_common", k)["test_roc_auc"]), num(res("baseline", k)["test_roc_auc"])) for k in ext_done) or "-"))
ext_sections = [{"heading": "12. Common MỞ RỘNG trộn 4:1, val Pha 1 chỉ JS (n = 5)", "paras": ext_paras, "bullets": ext_bullets}]
if t12:
    ext_sections[0]["table"] = t12

doc = {
    "title": "Pha 1 trộn JS:Py theo fold (4:1, 3:1, 4CWE, val chỉ JS, uncommon, common mở rộng)",
    "updated": sys.argv[1] if len(sys.argv) > 1 else "",
    "question": "Người dùng 06/10 11:5x: \"trộn phase 1 cho model transfer tỷ lệ js:py là 4:1 với cả bộ full và common, val sẽ bằng cả js 15% là python của fold đó. Chạy thử 5 fold\"; chọn: Python lấy mẫu cân nhãn từ SVEN train fold k (seed 42), val Pha 1 = JS val + SVEN val fold k, Pha 2 chỉ cột chính.",
    "verdict": verdict,
    "confidence": "thấp (bậc 2: n = 5, một seed, 161 + 158 trùng bit; đối chứng rasam_<mức>_jsonly cùng fold)",
    "tier": "Bậc 2 theo yêu cầu (n = 5 fold, seed 42, TF32, tất định); ghép cặp theo fold với cột chính Pha 1 chỉ JS.",
    "sections": [
        {"heading": "1. Pha 2 cột chính: Pha 1 trộn so với Pha 1 chỉ JS (+k/-k: số fold dương/âm, hoà 0,001)", "table": t_pair},
        {"heading": "2. ROC theo fold", "table": t_fold},
        {"heading": "3. Pha 1: ROC trên val Pha 1 tách theo ngôn ngữ", "table": t_p1,
         "paras": ["Val Pha 1 = JS val (148 / 188) + SVEN val fold k (152, cũng là val Pha 2). Phần JS gần ngẫu nhiên nên checkpoint Pha 1 được chọn chủ yếu theo phần Python của ĐÍCH."]},
        {"heading": "4. Chấm dự đoán ghi trước", "bullets": grades},
    ] + ratio_sections + asam_sections + full31_sections + cwe_sections + sep_sections + ra_sections + unc_sections + ext_sections,
    "next": ["(XONG 07/10 00:41, mục 9) Tách nguồn lợi qua val Pha 1: val CHỈ JS không làm mất lợi TB.",
             "(cần duyệt) Đối chứng Python-only: Pha 1 = 211/267 hàm SVEN train (không JS) - đo phần lợi của riêng dữ liệu đích trong Pha 1."],
    "sources": ["_FinalPaperExperiment/results/{p1,rasam}_jspy41_{common,full}/fold<k>.json (+ probs.npz), đối chứng rasam_<mức>_jsonly",
                "data/mwonly5_sources/js_py41_{common,full}/ (tools/build_p1_mixpy.py, MANIFEST.json)",
                "_FinalPaperExperiment/meta/insights/jspy41/build_jspy41.py (dựng lại trang này)"],
}
out = os.path.dirname(os.path.abspath(__file__))
md = ["# " + doc["title"], "", "_%s · %s_" % (doc["updated"], doc["tier"]), "", "**Câu hỏi:** " + doc["question"], "", "**Kết luận:** " + doc["verdict"], "",
      "**Độ tin:** " + doc["confidence"], ""]
for s in doc["sections"]:
    md += ["## " + s["heading"], ""]
    if s.get("table"):
        t = s["table"]; md += ["| " + " | ".join(t["head"]) + " |", "|" + "---|" * len(t["head"])] + ["| " + " | ".join(r) + " |" for r in t["rows"]] + [""]
    md += [p + "\n" for p in s.get("paras", [])] + ["- " + b for b in s.get("bullets", [])] + ([""] if s.get("bullets") else [])
md += ["## Đề xuất kiểm tiếp (cần duyệt trước khi chạy)", ""] + ["- " + x for x in doc["next"]] + ["", "## Nguồn", ""] + ["- " + x for x in doc["sources"]]
if complete:
    json.dump(doc, open(os.path.join(out, "jspy41.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(out, "jspy41.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
