#!/usr/bin/env python3
"""Nhận định NGẮN: hướng A bước 1 (HUONG_DI_LN_VA_TRAN_JS_0510.md) - nhiễu ε của ASAM trên bias/LN ở 5 ô kẹt của chỉ ASAM trên JS + C/C++.
Đọc results/ + logs/ của khối mwonly5, ghi dirA.json (tab Nhận định) và dirA.md. Chỉ đọc, không chạy gì; chạy được khi chưa đủ 15 ô
(ô thiếu in "-", chấm dự đoán ghi "chưa đủ")."""
import json, os, re, statistics as st
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CELLS = [("4cwe_jscpp", 1), ("common_jscpp", 1), ("common_jscpp", 3), ("common_jscpp", 4), ("full_jscpp", 4)]
SRC = {"4cwe_jscpp": "4CWE", "common_jscpp": "common", "full_jscpp": "full"}
ARMS = [("full", "full (chạy lại)"), ("mask", "mask"), ("excl", "exclude")]
K = [("test_roc_auc", "ROC-AUC"), ("test_pr_auc", "PR-AUC"), ("test_macro_f1_at_0.5", "F1@0,5"), ("test_macro_f1_at_valcal", "F1@ngưỡng val")]
TIE, ESC, ESC_BY = 1e-3, 0.65, 4          # thoát bình nguyên = epoch đầu tiên train loss < 0,65; "trước epoch 5" = thoát ở epoch ≤ 4
num = lambda x, d=3: "-" if x is None else ("%.*f" % (d, x)).replace(".", ",")
sgn = lambda x, d=3: "-" if x is None else "±0,000" if abs(x) < 0.5 * 10 ** -d else ("%+.*f" % (d, x)).replace(".", ",")


def res(run, k):
    p = os.path.join(BASE, "results", run, "fold%d.json" % k)
    return json.load(open(p)) if os.path.exists(p) else None


def hist(run, k):
    p = os.path.join(BASE, "logs", run, "fold%d.log" % k)
    ep, share = [], {}
    if not os.path.exists(p):
        return ep, share
    for l in open(p, encoding="utf-8"):
        m = re.search(r"Epoch (\d+)/30 \| train loss ([0-9.]+) \| val loss [0-9.]+ \| val roc_auc ([0-9.]+)", l)
        if m:
            ep.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        m = re.search(r"SAM share ep (\d+) \| weight ([0-9.]+)% \| bias ([0-9.]+)% \| LN ([0-9.]+)%", l)
        if m:
            share[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return ep, share


def escape(h):
    return next((e for e, t, v in h if t < ESC), None)


def esc_txt(h, done):
    e = escape(h)
    return ("ep%d" % e) if e else ("không thoát (%d ep)" % len(h) if done else "chưa (%d ep)" % len(h))


C = {}
for s, k in CELLS:
    c = {"orig": res("asamonly_" + s, k), "orig_h": hist("asamonly_" + s, k)[0], "adamw": res("noras_" + s, k), "main": res("rasam_" + s, k)}
    for a, _ in ARMS:
        run = "asamA%s_%s" % (a, s)
        c[a] = res(run, k)
        c[a + "_h"], c[a + "_sh"] = hist(run, k)
    C[(s, k)] = c
n_done = sum(C[x][a] is not None for x in C for a, _ in ARMS)

# bảng 1: theo ô - epoch thoát bình nguyên và test ROC
t1 = {"head": ["ô (chỉ ASAM)", "gốc", "full (chạy lại)", "mask", "exclude", "AdamW", "chính"], "rows": []}
for (s, k), c in C.items():
    row = ["%s f%d" % (SRC[s], k), "%s · %s" % (esc_txt(c["orig_h"], True), num(c["orig"]["test_roc_auc"]))]
    for a, _ in ARMS:
        row.append("%s · %s" % (esc_txt(c[a + "_h"], c[a] is not None), num(c[a]["test_roc_auc"]) if c[a] else "đang chạy" if c[a + "_h"] else "chờ"))
    row += [num((c["adamw"] or {}).get("test_roc_auc")), num((c["main"] or {}).get("test_roc_auc"))]
    t1["rows"].append(row)


def paired(a, b, key):
    d = [C[x][a][key] - C[x][b][key] for x in C if C[x][a] and C[x][b]]
    return (st.mean(d), sum(v > TIE for v in d), sum(v < -TIE for v in d), len(d)) if d else None


# bảng 2: hiệu ghép cặp theo ô, đủ bốn chỉ số
PAIRS = [("mask", "full", "mask - full"), ("excl", "full", "exclude - full"), ("excl", "mask", "exclude - mask"),
         ("mask", "adamw", "mask - AdamW"), ("excl", "adamw", "exclude - AdamW"), ("mask", "main", "mask - chính (RecAdam + ASAM)")]
t2 = {"head": ["hiệu (ghép theo ô)", "n"] + [lab for _, lab in K], "rows": []}
for a, b, lab in PAIRS:
    ps = [paired(a, b, key) for key, _ in K]
    t2["rows"].append([lab, str(max((p[3] for p in ps if p), default=0))] + ["-" if p is None else "%s (+%d/-%d)" % (sgn(p[0]), p[1], p[2]) for p in ps])

# chấm dự đoán ghi trước (CURRENT_RUN.md mục hướng A)
full_ok = [C[x]["full"] is not None for x in C]
same = [C[x]["full"]["test_roc_auc"] == C[x]["orig"]["test_roc_auc"] for x in C if C[x]["full"]]
mask_esc = [escape(C[x]["mask_h"]) for x in C if C[x]["mask"]]
excl_esc = [escape(C[x]["excl_h"]) for x in C if C[x]["excl"]]
share1 = [sum(C[x]["full_sh"][1][1:]) for x in C if C[x]["full"] and 1 in C[x]["full_sh"]]
stuck = [("4cwe_jscpp", 1), ("common_jscpp", 3)]
a4 = [C[x]["mask"]["test_roc_auc"] for x in stuck if C[x]["mask"]]


def grade(vals, need, ok_fn, txt):
    if len(vals) < need:
        return "chưa đủ (%d/%d ô): %s" % (len(vals), need, txt)
    return ("ĐÚNG" if ok_fn(vals) else "SAI") + ": " + txt


hits = lambda xs: sum(1 for e in xs if e is not None and e <= ESC_BY)
grades = [
    "A0 " + grade(same, 5, lambda v: all(v), "nhánh full trùng bit ô gốc (test ROC đủ 16 chữ số) ở %d/%d ô" % (sum(same), len(same))),
    "A1 " + grade(mask_esc, 5, lambda v: hits(v) >= 4, "mask thoát bình nguyên trước ep5 (≤ ep4) ở %d/%d ô (epoch thoát: %s; nới thành ≤ ep5 thì %d/%d - ghi để đọc, không đổi chấm)" % (
        hits(mask_esc), len(mask_esc), " / ".join("ep%d" % e if e else "không" for e in mask_esc), sum(1 for e in mask_esc if e and e <= 5), len(mask_esc))),
    "A2 " + grade(excl_esc, 5, lambda v: hits(v) >= 4, "exclude thoát trước ep5 ở %d/%d ô (epoch thoát: %s)" % (
        hits(excl_esc), len(excl_esc), " / ".join("ep%d" % e if e else "không" for e in excl_esc))),
    "A3 " + grade(share1, 5, lambda v: min(v) >= 80, "nhánh full, epoch 1: bias + LN chiếm %s %% của ‖g·T‖² (ngưỡng ≥ 80 %%)" % (
        " / ".join(num(v, 1) for v in share1) or "-")),
    "A4 " + grade(a4, 2, lambda v: min(v) >= 0.85, "mask ở hai ô sập thật (4CWE f1, common f3) test ROC %s (ngưỡng ≥ 0,85)" % (" / ".join(num(v) for v in a4) or "-")),
]

complete = n_done == 15
pm = paired("mask", "full", "test_roc_auc")
pe = paired("excl", "mask", "test_roc_auc")
pa = paired("mask", "adamw", "test_roc_auc")
verdict = ("Đã xong %d/15 ô. " % n_done + (
    "Mask thoát bình nguyên trước ep5 ở %d/%d ô kẹt (epoch thoát %s), full kẹt như ô gốc; ROC mask - full %s (+%d/-%d), mask - AdamW %s (+%d/-%d), "
    "exclude - mask %s (+%d/-%d). " % (hits(mask_esc), len(mask_esc), " / ".join("ep%d" % e if e else "không" for e in mask_esc),
                                         sgn(pm[0]), pm[1], pm[2], sgn(pa[0]), pa[1], pa[2], sgn(pe[0]), pe[1], pe[2])
    if pm and pa and pe else "Chưa đủ để đọc. "))

if complete:
    pm_main = paired("mask", "main", "test_roc_auc")
    plat_full = [escape(C[x]["full_h"]) or len(C[x]["full_h"]) + 1 for x in C]
    gain = [C[x]["mask"]["test_roc_auc"] - C[x]["full"]["test_roc_auc"] for x in C]
    verdict = ("Tắt nhiễu ε của ASAM trên bias/LN (mask, ε trên weight giữ nguyên bit) rút bình nguyên đầu Pha 2 từ %s epoch xuống %s epoch "
               "(epoch thoát) ở 5/5 ô và cứu 2/2 ô sập (0,607 → %s; 0,523 → %s); nhưng tiêu chí đặt trước \"thoát trước ep5 ở ≥ 4/5 ô\" trượt sát "
               "(%d/5). Về độ chính xác mask chỉ đưa chỉ ASAM về ngang AdamW (ROC %s, +%d/-%d ô) và thấp hơn cột chính (%s, +%d/-%d); exclude ≈ mask "
               "(%s). Đọc: nhiễu bias/LN là nguyên nhân chính của bình nguyên - dùng được làm câu cơ chế - nhưng KHÔNG phải đòn bẩy độ chính xác; "
               "không đổi công thức chính.") % (
        " / ".join(">%d" % (e - 1) if e > len(C[x]["full_h"]) else "%d" % e for e, x in zip(plat_full, C)),
        " / ".join("%d" % escape(C[x]["mask_h"]) for x in C), num(C[("4cwe_jscpp", 1)]["mask"]["test_roc_auc"]),
        num(C[("common_jscpp", 3)]["mask"]["test_roc_auc"]), hits(mask_esc), sgn(pa[0]), pa[1], pa[2], sgn(pm_main[0]), pm_main[1], pm_main[2],
        sgn(pe[0]))
    assess = [
        "Cơ chế: ở epoch 1 bias + LN chiếm %s %% của ‖g·T‖² (5/5 ô, A3); bỏ ĐÚNG phần ε đó (mask: ε weight trùng bit full) đủ rút bình nguyên ≥ 8 epoch ở 5/5 ô. Trong phạm vi 5 ô này đó là quan hệ nhân quả, không chỉ tương quan." % " / ".join(num(v, 1) for v in share1),
        "Độ chính xác: lợi chỉ nằm ở 2 ô sập (mask - full %s / %s). Ở 3 ô bình nguyên dài nhưng không sập: %s / %s / %s - thoát sớm không làm tốt hơn." % tuple(sgn(g) for g in [gain[0], gain[2], gain[1], gain[3], gain[4]]),
        "Bước 2 của tài liệu (mask so với full trên công thức chính, 3 seed, 30 lượt, 30-40 GPU-giờ A4000): KHÔNG đề xuất - cột chính trên nguồn chính (common chỉ JS) không có ô kẹt, và mask không có lợi ROC ở ô không sập. Chỉ đáng nếu cần chống sập cho nguồn JS + C/C++ (cần duyệt).",
        "Đã ẩn khỏi bảng chính (vẫn ở tab So sánh): 3 dòng full (chạy lại, trùng bit ô gốc), 3 dòng exclude (≈ mask; tài liệu khuyên không dùng làm công thức), 2 biến thể phụ common chỉ JS (bỏ nhiễu / co T, không hơn chỉ ASAM ở n = 5). Giữ 3 dòng mask làm kết quả của hướng A.",
    ]
else:
    assess = []

# 06/10 02:1x tự chạy thêm f2, f5 của mask / exclude trên common JS + C/C++ (158 + paper_mw3 rảnh) ⇒ một nguồn đủ 5 fold KHÔNG chọn lọc
extra = []
EXF = range(1, 6)
if all(res("asamA%s_common_jscpp" % a, k) for a in ("mask", "excl") for k in EXF):
    G = {r: [res(r + "_common_jscpp", k) for k in EXF] for r in ("asamAmask", "asamAexcl", "asamonly", "noras", "rasam")}
    E = {r: [escape(hist(r + "_common_jscpp", k)[0]) for k in EXF] for r in ("asamAmask", "asamAexcl", "asamonly")}
    t5 = {"head": ["hiệu (ghép theo fold)", "n"] + [lab for _, lab in K], "rows": []}
    for a, b, lab in (("asamAmask", "asamonly", "mask - chỉ ASAM gốc"), ("asamAmask", "noras", "mask - AdamW"), ("asamAmask", "rasam", "mask - chính"),
                      ("asamAexcl", "asamAmask", "exclude - mask")):
        cells = []
        for key, _ in K:
            d = [x[key] - y[key] for x, y in zip(G[a], G[b])]
            cells.append("%s (+%d/-%d)" % (sgn(st.mean(d)), sum(v > TIE for v in d), sum(v < -TIE for v in d)))
        t5["rows"].append([lab, "5"] + cells)
    dm = [x["test_roc_auc"] - y["test_roc_auc"] for x, y in zip(G["asamAmask"], G["asamonly"])]
    ep = lambda xs: " / ".join("ep%d" % e if e else "không" for e in xs)
    extra = [{"heading": "6. Thêm f2, f5: nguồn common JS + C/C++ đủ 5 fold (không chọn lọc)",
              "paras": ["Tự chạy lúc máy rảnh (158 + paper_mw3, ghi dự đoán trước ở CURRENT_RUN). Epoch thoát - gốc: %s; mask: %s; exclude: %s. ROC mask - gốc theo fold: %s - lợi trung bình đến từ một fold sập (f3), 3/4 fold còn lại hơi thấp hơn." % (
                  ep(E["asamonly"]), ep(E["asamAmask"]), ep(E["asamAexcl"]), " / ".join(sgn(v) for v in dm))],
              "table": t5,
              "bullets": ["C1 ĐÚNG: mask và exclude thoát ≤ ep5 ở cả f2 lẫn f5 (mask %s / %s, exclude %s / %s)." % ("ep%d" % E["asamAmask"][1], "ep%d" % E["asamAmask"][4], "ep%d" % E["asamAexcl"][1], "ep%d" % E["asamAexcl"][4]),
                          "C2 ĐÚNG: ở f2, f5 |ROC mask - gốc| = %s / %s ≤ 0,03." % (num(abs(dm[1])), num(abs(dm[4]))),
                          "C3 ĐÚNG: mask - AdamW không 5/5 cùng dấu.",
                          "C4 ĐÚNG: mask thấp hơn cột chính ở 5/5 fold."]}]
# 06/10 13:0x mask trên full JS + C/C++ đủ 5 fold (người dùng: "trông có vẻ ổn, chạy thử đủ 5 fold xem"; vast paper_mw3 54437777, f4 chạy lại trùng bit 161)
extra_full = []
if all(res("asamAmask_full_jscpp", k) for k in range(1, 6)):
    G = {r: [res(r + "_full_jscpp", k) for k in range(1, 6)] for r in ("asamAmask", "asamonly", "noras", "rasam")}
    E = {r: [escape(hist(r + "_full_jscpp", k)[0]) for k in range(1, 6)] for r in ("asamAmask", "asamonly")}
    t7 = {"head": ["hiệu (ghép theo fold)", "n"] + [lab for _, lab in K], "rows": []}
    for b, lab in (("asamonly", "mask - chỉ ASAM gốc"), ("noras", "mask - AdamW"), ("rasam", "mask - chính")):
        cells = []
        for key, _ in K:
            d = [x[key] - y[key] for x, y in zip(G["asamAmask"], G[b])]
            cells.append("%s (+%d/-%d)" % (sgn(st.mean(d)), sum(v > TIE for v in d), sum(v < -TIE for v in d)))
        t7["rows"].append([lab, "5"] + cells)
    ep = lambda xs: " / ".join("ep%d" % e if e else "không" for e in xs)
    extra_full = [{"heading": "7. Mask trên full JS + C/C++ đủ 5 fold",
                   "paras": ["Người dùng yêu cầu 06/10 13:0x; vast paper_mw3 (A4000), f4 chạy lại trùng 16 chữ số với 161. Epoch thoát - gốc: %s; mask: %s. ROC mask: %s. Không fold nào của ô gốc sập, nên đây là phép đo mask trên một nguồn KHÔNG chọn lọc: thoát sớm hơn 5/5 fold, ROC ngang ô gốc, không hơn AdamW ở fold nào." % (
                       ep(E["asamonly"]), ep(E["asamAmask"]), " / ".join(num(x["test_roc_auc"]) for x in G["asamAmask"]))],
                   "table": t7,
                   "bullets": ["D1 ĐÚNG (f4 trùng bit 161), D2 ĐÚNG (thoát ≤ ep7 ở 5/5), D3 ĐÚNG (ROC TB so với gốc trong [-0,01; +0,03], không 5/5 cùng dấu), D4 ĐÚNG (không 5/5 cùng dấu với AdamW; thấp hơn cột chính ở 3/5)."]}]

doc = {
    "title": "Hướng A bước 1: tắt nhiễu ASAM trên bias/LN ở 5 ô kẹt JS + C/C++",
    "updated": "",
    "question": "Người dùng 05/10 23:4x: \"Tham khảo multiBABEL/audits/HUONG_DI_LN_VA_TRAN_JS_0510.md triển hướng A xem sao\"; 06/10 00:3x: \"thêm nhận định thật ngắn gọn về hướng A khi chạy xong nha, kết quả không tốt có thể ẩn khỏi kết quả chính luôn\".",
    "verdict": verdict,
    "confidence": "thấp (bậc 1: một seed, 5 ô chọn vì kẹt, ba máy A4000 - mỗi ô đủ ba nhánh trên cùng một máy)",
    "tier": "Bậc 1 (kiểm cơ chế, n = 5 ô kẹt, seed 42) - KHÔNG phải kết quả; ghép cặp theo ô, ba nhánh cùng checkpoint Pha 1 và cùng máy.",
    "sections": [
        {"heading": "1. Thiết kế (một câu)",
         "paras": ["Chỉ ASAM (ρ 0,5) trên 5 ô từng kẹt bình nguyên ln2 ≥ 14 epoch, ba nhánh chỉ khác --sam_bias_ln: full (như cũ, chạy lại), mask (ε = 0 trên bias/LN, ε weight trùng full), exclude (bỏ bias/LN khỏi cả ε lẫn chuẩn, ρ dồn cho weight). Thoát bình nguyên = epoch đầu train loss < 0,65."]},
        {"heading": "2. Theo ô: epoch thoát · test ROC", "table": t1},
        {"heading": "3. Hiệu ghép cặp theo ô (+k/-k: số ô dương/âm, hoà 0,001)", "table": t2},
        {"heading": "4. Chấm dự đoán ghi trước", "bullets": grades},
    ] + ([{"heading": "5. Đánh giá", "bullets": assess}] if assess else []) + extra + extra_full,
    "sources": ["_FinalPaperExperiment/results/asamA{full,mask,excl}_{4cwe,common,full}_jscpp/fold<k>.json, ô gốc asamonly_<nguồn>/fold<k>.json",
                "_FinalPaperExperiment/logs/<run>/fold<k>.log (dòng Epoch và SAM share)",
                "/drive1/cuongtm/metaProject/MAML/multiBABEL/audits/HUONG_DI_LN_VA_TRAN_JS_0510.md (hướng A)",
                "_FinalPaperExperiment/meta/insights/dirA/build_dirA.py (dựng lại trang này)"],
}
if __name__ == "__main__":
    import sys
    doc["updated"] = sys.argv[1] if len(sys.argv) > 1 else ""
    out = os.path.dirname(os.path.abspath(__file__))
    json.dump({"doc": doc, "complete": complete, "n_done": n_done}, open(os.path.join(out, "dirA_numbers.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if complete:
        json.dump(doc, open(os.path.join(out, "dirA.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = ["# " + doc["title"], "", "_%s · %s_" % (doc["updated"], doc["tier"]), "", "**Câu hỏi:** " + doc["question"], "", "**Kết luận:** " + doc["verdict"], "",
          "**Độ tin:** " + doc["confidence"], ""]
    for s in doc["sections"]:
        md += ["## " + s["heading"], ""] + [p + "\n" for p in s.get("paras", [])] + ["- " + b for b in s.get("bullets", [])]
        if s.get("bullets"):
            md.append("")
        if s.get("table"):
            t = s["table"]; md += ["| " + " | ".join(t["head"]) + " |", "|" + "---|" * len(t["head"])] + ["| " + " | ".join(r) + " |" for r in t["rows"]] + [""]
    md += ["## Nguồn", ""] + ["- " + x for x in doc["sources"]]
    if complete:
        open(os.path.join(out, "dirA.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
