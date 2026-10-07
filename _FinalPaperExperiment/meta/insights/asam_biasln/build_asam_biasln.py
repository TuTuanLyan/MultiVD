#!/usr/bin/env python3
"""Nhận định: hai biến thể phụ của ASAM trên nguồn common chỉ JS (kiểm mục 2.3 PHA2_SUA_THEO_BANG_CHUNG_0510.md).
Đọc results/ + logs/ của khối mwonly5, ghi asam_biasln.json (cho tab Nhận định) và asam_biasln.md. Chỉ đọc, không chạy gì."""
import json, os, re, statistics as st
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = [("asamonly_common_jsonly", "chỉ ASAM"), ("asamblnex_common_jsonly", "bỏ nhiễu bias/LN"),
        ("asamblnsh_common_jsonly", "co T ×0,1"), ("noras_common_jsonly", "AdamW")]
K = [("test_roc_auc", "ROC-AUC"), ("test_pr_auc", "PR-AUC"), ("test_macro_f1_at_0.5", "F1@0,5"), ("test_macro_f1_at_valcal", "F1@ngưỡng val")]
TIE = 1e-3
f = lambda x, d=3: ("%+.*f" % (d, x) if x else "0,000").replace(".", ",") if isinstance(x, float) and x != 0 else ("%.*f" % (d, x)).replace(".", ",")
num = lambda x, d=3: ("%.*f" % (d, x)).replace(".", ",")
sgn = lambda x, d=3: "±0,000" if abs(x) < 0.5 * 10 ** -d else ("%+.*f" % (d, x)).replace(".", ",")


def hist(run, fold):
    out = []
    for l in open(os.path.join(BASE, "logs", run, "fold%d.log" % fold), encoding="utf-8"):
        m = re.search(r"Epoch (\d+)/30 \| train loss ([0-9.]+) \| val loss ([0-9.]+) \| val roc_auc ([0-9.]+)", l)
        if m:
            out.append((int(m.group(1)), float(m.group(2)), float(m.group(4))))
    return out


R = {r: [json.load(open(os.path.join(BASE, "results", r, "fold%d.json" % k))) for k in range(1, 6)] for r, _ in RUNS}
H = {r: [hist(r, k) for k in range(1, 6)] for r, _ in RUNS}
name = dict(RUNS)


def paired(a, b, key):
    d = [x[key] - y[key] for x, y in zip(R[a], R[b])]
    return st.mean(d), sum(v > TIE for v in d), sum(v < -TIE for v in d), d


# bảng 1: chỉ số trung bình + hiệu ghép cặp
t1 = {"head": ["chỉ số", "chỉ ASAM", "bỏ nhiễu", "co T ×0,1", "AdamW", "bỏ nhiễu - chỉ ASAM", "co T - chỉ ASAM", "bỏ nhiễu - AdamW", "co T - AdamW"], "rows": []}
for key, lab in K:
    row = [lab] + [num(st.mean(x[key] for x in R[r])) for r, _ in RUNS]
    for a, b in (("asamblnex_common_jsonly", "asamonly_common_jsonly"), ("asamblnsh_common_jsonly", "asamonly_common_jsonly"),
                 ("asamblnex_common_jsonly", "noras_common_jsonly"), ("asamblnsh_common_jsonly", "noras_common_jsonly")):
        m, p, n, _ = paired(a, b, key)
        row.append("%s (+%d/-%d)" % (sgn(m), p, n))
    t1["rows"].append(row)
# bảng 2: ROC theo fold
t2 = {"head": ["fold", "chỉ ASAM", "bỏ nhiễu", "co T ×0,1", "AdamW", "chỉ ASAM - AdamW", "bỏ nhiễu - chỉ ASAM", "co T - chỉ ASAM"], "rows": []}
for k in range(5):
    v = {r: R[r][k]["test_roc_auc"] for r, _ in RUNS}
    t2["rows"].append(["f%d" % (k + 1)] + [num(v[r]) for r, _ in RUNS] +
                      [sgn(v["asamonly_common_jsonly"] - v["noras_common_jsonly"]), sgn(v["asamblnex_common_jsonly"] - v["asamonly_common_jsonly"]),
                       sgn(v["asamblnsh_common_jsonly"] - v["asamonly_common_jsonly"])])
# bảng 3: động học
t3 = {"head": ["", "train loss ep1", "train loss ep3", "val ROC ep1", "epoch chọn (từng fold)", "train loss tại epoch chọn", "bình nguyên (epoch, từng fold)"], "rows": []}
dyn = {}
for r, lab in RUNS:
    be = [x["best_epoch"] for x in R[r]]
    tlb = [next(t for e, t, v in h if e == b) for h, b in zip(H[r], be)]
    pl = []
    for h in H[r]:
        e0 = next((e for e, t, v in h if t < 0.65), None)
        pl.append(e0 - 1 if e0 else len(h))
    dyn[r] = {"tl1": st.mean(h[0][1] for h in H[r]), "tl3": st.mean(h[2][1] for h in H[r]), "v1": st.mean(h[0][2] for h in H[r]), "tlb": st.mean(tlb)}
    t3["rows"].append([lab, num(dyn[r]["tl1"]), num(dyn[r]["tl3"]), num(dyn[r]["v1"]), " / ".join(map(str, be)), num(dyn[r]["tlb"]), " / ".join(map(str, pl))])

gain = paired("asamonly_common_jsonly", "noras_common_jsonly", "test_roc_auc")
ex_ad = paired("asamblnex_common_jsonly", "noras_common_jsonly", "test_roc_auc")
sh_ad = paired("asamblnsh_common_jsonly", "noras_common_jsonly", "test_roc_auc")
ex_sh = paired("asamblnex_common_jsonly", "asamblnsh_common_jsonly", "test_roc_auc")
ex_as = paired("asamblnex_common_jsonly", "asamonly_common_jsonly", "test_roc_auc")
sh_as = paired("asamblnsh_common_jsonly", "asamonly_common_jsonly", "test_roc_auc")

doc = {
    "title": "Nhiễu ε của ASAM trên bias + LayerNorm: bỏ hay co có giúp không (kiểm mục 2.3)",
    "updated": "2026-10-05 22:55",
    "question": "Người dùng 05/10: \"local test thêm cho tôi theo tham khảo mục 2.3 PHA2_SUA_THEO_BANG_CHUNG_0510.md\"; 22:4x: \"bạn đánh giá thế nào về 2 biến thể phụ của ASAM JS common => thêm vào nhận định và các dự đoán trong .md\".",
    "verdict": ("Trên nguồn common chỉ JS (công thức chỉ ASAM, n = 5, một seed), bỏ hoặc co nhiễu ε trên bias + LayerNorm KHÔNG cải thiện: ROC thấp hơn "
                "chỉ ASAM %s (+%d/-%d) và %s (+%d/-%d), dưới sàn nhiễu 0,010; vẫn hơn AdamW %s (+%d/-%d) và %s (+%d/-%d), tức giữ khoảng một nửa lợi ROC "
                "của ASAM (%s). Cơ chế thấy rõ hơn độ chính xác: hai biến thể học nhanh gần như AdamW (train loss ep3 %s / %s so với chỉ ASAM %s, AdamW %s) "
                "- phần lớn cái 'phanh' lúc khởi động của ASAM nằm ở nhiễu trên bias/LN. Mục 2.3 chưa được ủng hộ về độ chính xác trên nguồn này; phần "
                "'cứu bình nguyên' chưa kiểm được vì nguồn này không có ô kẹt.") % (
        sgn(ex_as[0]), ex_as[1], ex_as[2], sgn(sh_as[0]), sh_as[1], sh_as[2], sgn(ex_ad[0]), ex_ad[1], ex_ad[2], sgn(sh_ad[0]), sh_ad[1], sh_ad[2],
        sgn(gain[0]), num(dyn["asamblnex_common_jsonly"]["tl3"]), num(dyn["asamblnsh_common_jsonly"]["tl3"]), num(dyn["asamonly_common_jsonly"]["tl3"]),
        num(dyn["noras_common_jsonly"]["tl3"])),
    "confidence": "thấp - trung bình (một nguồn, một seed, một máy; mẫu hình đã co lại một lần từ n = 3 sang n = 5)",
    "tier": "Bậc 2 (xác nhận): n = 5 fold, seed 42, máy 161 (A4000, TF32, tất định); ghép cặp theo fold với asamonly_common_jsonly và noras_common_jsonly cùng checkpoint Pha 1 p1_common_jsonly.",
    "sections": [
        {"heading": "1. Thiết kế",
         "paras": ["ASAM của khối: ε = ρ T² g / ‖T g‖, ρ 0,5, T = |w| + 0,01 cho tham số có 'weight' trong tên, T = 1 cho bias. Gain của LayerNorm ≈ 1 nên T ≈ 1 cho cả LayerNorm. Thử trên mô hình đồ chơi: 92 % ‖ε‖² rơi vào bias + LayerNorm (khớp con số 72-81 % ‖T·g‖² ở mục 2.3).",
                   "Hai biến thể, cờ mới --sam_bias_ln của src_mwonly (mặc định full = như cũ, đã thử: ε và w + ε trùng bit mã cũ): (a) exclude - bias + LayerNorm không bị nhiễu, bỏ khỏi cả ε lẫn chuẩn ‖T g‖, nên ρ dồn hết cho các weight; (b) shrink - T của bias + LayerNorm nhân 0,1. Mọi cờ khác giữ chỉ ASAM (AdamW, lr 5,66e-5, không warmup, min 10, patience 8). hyperparameters của 10 ô chỉ khác chỉ ASAM ở hai cờ này."]},
        {"heading": "2. Độ chính xác: không hơn chỉ ASAM, vẫn hơn AdamW",
         "paras": ["Trung bình 5 fold và hiệu ghép cặp theo fold (+k/-k: số fold dương/âm, ngưỡng hoà 0,001). Mọi hiệu so với chỉ ASAM nằm dưới hoặc sát sàn nhiễu 0,010 và đổi dấu giữa các fold; so với AdamW, ROC dương 4/5 ở cả hai biến thể."],
         "table": t1},
        {"heading": "3. Theo fold: lợi của ASAM tập trung ở f1-f3, và biến thể mất đúng chỗ đó",
         "paras": ["Trên nguồn này ASAM hơn AdamW %s ROC TB, nhưng toàn bộ nằm ở f1-f3 (+0,030..+0,034); f4-f5 ASAM ngang AdamW. Hai biến thể thấp hơn chỉ ASAM đúng ở f1-f3 (-0,016..-0,033) và cao hơn ở f5 (+0,017 / +0,022), nơi chỉ ASAM chọn epoch 18 với train loss 0,026 (khớp quá). Không suy tương quan giữa hai cột cuối với cột 'chỉ ASAM - AdamW': chúng chung số hạng chỉ ASAM nên tương quan âm sẵn có." % sgn(gain[0]),
                   "Ở n = 3 (f1-f3) cả hai biến thể kém chỉ ASAM 3/3 (ROC TB -0,022 / -0,021) và đã được báo là 'lợi của ASAM phần lớn đến từ nhiễu bias/LN'. f4-f5 kéo trung bình về -0,009: câu đó đã RÚT LẠI (CURRENT_RUN.md mục kiểm 2.3)."],
         "table": t2},
        {"heading": "4. Động học: nhiễu bias/LN là cái phanh lúc khởi động",
         "paras": ["Trung bình 5 fold. Hai biến thể bắt đầu gần như AdamW (val ROC ep1 0,82 so với 0,73 của chỉ ASAM), train loss ep3 chỉ hơn AdamW 0,04 trong khi chỉ ASAM hơn 0,22. Tại epoch được chọn, mức khớp train của biến thể nằm giữa chỉ ASAM và AdamW. Bình nguyên (epoch đầu train loss < 0,65, trừ 1) là 0-1 epoch ở mọi ô - nguồn này không có ô kẹt nên không đo được tác dụng chống kẹt.",
                   "Đọc: phần lớn việc ASAM làm chậm khởi động đến từ ε trên bias/LN, đúng cơ chế mục 2.3 mô tả; nhưng chính phần 'chậm và khớp ít hơn' đó đi kèm lợi ROC trên nguồn này, nên bỏ nó không có lợi về độ chính xác."],
         "table": t3},
        {"heading": "5. Bỏ hẳn hay co T: không phân biệt được",
         "paras": ["Bỏ nhiễu - co T: ROC %s (từng fold %s). Núm vặn thực chất là 'bao nhiêu ε rơi vào bias/LN', không phải cách loại." % (sgn(ex_sh[0]), " / ".join(sgn(v) for v in ex_sh[3]))]},
        {"heading": "6. Đối chiếu dự đoán ghi trước (CURRENT_RUN.md)",
         "bullets": ["B1 ĐÚNG: dòng 'SAM |' của 10/10 ô in đúng bias_ln=exclude / shrink.",
                     "B2 ĐÚNG: bình nguyên ≤ 1 epoch ở 10/10 ô.",
                     "B3 ĐÚNG: ROC so với chỉ ASAM TB -0,009 / -0,009, trong [-0,020; +0,010], dương ≤ 2/5.",
                     "B4 ĐÚNG: cả hai biến thể hơn AdamW về ROC ở 4/5 fold.",
                     "Nhận định 18:3x (từ n = 3) 'lợi của ASAM phần lớn đến từ nhiễu bias/LN' SAI ở n = 5 - thêm một lần mẫu hình ở n = 3 co lại khi lên n = 5."]},
        {"heading": "7. Đánh giá cho bài",
         "bullets": ["Không thay ASAM chuẩn bằng biến thể bỏ / co nhiễu bias-LN để lấy độ chính xác: trên nguồn chính không có lợi (n = 5, một seed).",
                     "Có thể dùng làm bằng chứng cơ chế (phân tích phụ): nhiễu ε trên bias/LN là thành phần làm ASAM khởi động chậm; cùng lúc nó gánh khoảng một nửa lợi ROC của ASAM so với AdamW trên nguồn này. Phát biểu ở mức quan sát, chưa lặp seed / phần cứng.",
                     "Giá trị thật sự của mục 2.3 nằm ở nguồn / đích có kẹt bình nguyên (JS + C/C++, đích JS) - chưa kiểm."]},
    ],
    "next": [
        "P1 (cần duyệt): bỏ nhiễu bias/LN trên nguồn common JS + C/C++, công thức chỉ ASAM, f1-f5 (ô gốc chỉ ASAM kẹt 4/5 fold). Dự đoán: bình nguyên ≤ 2 epoch ở ≥ 4/5 fold; 0/5 sập; ROC dương so với chỉ ASAM ở mọi fold mà chỉ ASAM kẹt.",
        "P2 (cần duyệt): bỏ nhiễu bias/LN trên đích JS full với cột chính (RecAdam + ASAM), nguồn SVEN, f1-f5 (cả 5 fold gốc kẹt 10-18 epoch, f4 sập). Dự đoán: bình nguyên ≤ 3 epoch ở ≥ 4/5 fold; 0/5 sập; ROC TB hơn bản gốc (0,553).",
        "P3 (cần duyệt): lặp seed 1234 trên common chỉ JS (3 nhánh: chỉ ASAM, bỏ nhiễu, AdamW). Dự đoán: |ROC bỏ nhiễu - chỉ ASAM| TB < 0,010, không 5/5 cùng dấu; bỏ nhiễu vẫn hơn AdamW ở ≥ 3/5.",
    ],
    "sources": ["_FinalPaperExperiment/results/{asamonly,asamblnex,asamblnsh,noras}_common_jsonly/fold1-5.json", "_FinalPaperExperiment/logs/<run>/fold<k>.log (động học)",
                "src_mwonly/mwg/optim.py SAMStep(bias_ln) + train.py --sam_bias_ln (bản trước ở _FinalPaperExperiment/state/*.before_biasln_0510)",
                "/drive1/cuongtm/metaProject/MAML/multiBABEL/audits/PHA2_SUA_THEO_BANG_CHUNG_0510.md mục 2.3",
                "_FinalPaperExperiment/meta/insights/asam_biasln/build_asam_biasln.py (dựng lại trang này)"],
}
out = os.path.dirname(os.path.abspath(__file__))
json.dump(doc, open(os.path.join(out, "asam_biasln.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
md = ["# " + doc["title"], "", "_%s · %s_" % (doc["updated"], doc["tier"]), "", "**Câu hỏi:** " + doc["question"], "", "**Kết luận:** " + doc["verdict"], "",
      "**Độ tin:** " + doc["confidence"], ""]
for s in doc["sections"]:
    md += ["## " + s["heading"], ""] + [p + "\n" for p in s.get("paras", [])] + ["- " + b for b in s.get("bullets", [])]
    if s.get("bullets"):
        md.append("")
    if s.get("table"):
        t = s["table"]; md += ["| " + " | ".join(t["head"]) + " |", "|" + "---|" * len(t["head"])] + ["| " + " | ".join(r) + " |" for r in t["rows"]] + [""]
md += ["## Đề xuất kiểm tiếp (cần duyệt trước khi chạy)", ""] + ["- " + x for x in doc["next"]] + ["", "## Nguồn", ""] + ["- " + x for x in doc["sources"]]
open(os.path.join(out, "asam_biasln.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("ok")
