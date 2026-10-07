# Nội dung viết tay của báo cáo; được build_asam_collapse_report.py exec() với các biến report, P, O, NUM, SE, row, orow, vn, pfmt.
# Mọi số lấy từ asam_collapse_numbers.json (asam_collapse_analysis.py) hoặc ghi rõ nguồn khác.

report["verdict"] = (
    "ASAM ρ 0,5 bật từ bước đầu là nguyên nhân của bình nguyên ln2 ở Pha 2: ghép cặp cùng checkpoint và cùng fold, thêm ASAM làm "
    "bình nguyên dài hơn ở 33/35 cặp khi không có RecAdam và 24/30 cặp khi có RecAdam (0 cặp ngắn hơn), lặp ở khối cũ (49/49, 64/64), "
    "và khi tắt ASAM tới lúc loss rời ln2 thì bình nguyên biến mất (15/15 ô, 0/15 sập so với 8/15); 'sập' chỉ là bình nguyên dài bị luật "
    "dừng sớm cắt. Phát biểu 'RecAdam giúp' thì KHÔNG đứng: RecAdam + ASAM so với chỉ ASAM có bình nguyên ngắn hơn 8 / dài hơn 9 / bằng 13 "
    "(trung vị 0), số ô kẹt > 10 epoch 14/45 so với 10/35; chênh lệch 0/45 với 3/35 ô sập chỉ nằm ở đuôi (p ≈ 0,08, một seed), còn ở khối "
    "cũ RecAdam lại làm bình nguyên DÀI hơn (22/1). Warmup ngầm của RecAdam có thật trong mã (λ(t) nhân vào bước Adam, λ = 0,46 ở bước 1, "
    "≥ 0,9 từ epoch 4) và đo được ở nhánh không ASAM, nhưng không rút ngắn bình nguyên của ASAM một cách nhất quán."
)
report["confidence"] = "trung bình"
report["tier"] = (
    "Bậc 2 (xác nhận): khối mwonly5, n = 5 fold mỗi nguồn, MỘT seed (42); 6 nguồn có đủ thiết kế 2×2 (120 ô) + 10 ô không Pha 1 + 15 ô "
    "cột chính chỉ C/C++. Bốn máy RTX A4000 đã kiểm trùng bit nên ghép cặp khác máy không mang lệch phần cứng. Sáu nguồn dùng CHUNG 5 fold "
    "đích SVEN nên 30 cặp không độc lập: p cấp ô là lạc quan, phải đọc kèm đếm theo fold (5/5 là sàn p 0,0625) và theo nguồn. Bổ sung: khối "
    "cũ trong archive_20261004 (batch 8, lr 2e-5, warmup 10 %, min_epochs 3; 444 ô ghép được) và phép can thiệp ASAM muộn K15 (bậc 1, n = 3). "
    "Mức tin: phần 'ASAM gây bình nguyên' cao; phần 'RecAdam đỡ' thấp (dữ liệu không ủng hộ); phần cơ chế warmup ngầm là giả thuyết. "
    "Sàn nhiễu chạy lại ~0,010 ROC; chưa có số nào ở đây đủ để viết vào bài."
)

S = report["sections"]

# ----------------------------------------------------------------------------- 1. 2×2
S.append({
    "heading": "1. Thiết kế 2×2 trên 6 nguồn: ASAM kéo dài bình nguyên, RecAdam không rút ngắn",
    "paras": [
        "Định nghĩa: độ dài bình nguyên = (epoch đầu tiên có train loss < 0,65) - 1. Ô không bao giờ xuống dưới ngưỡng thì bị kiểm duyệt "
        "(gán epoch cuối + 1; chỉ có ở 3 ô chỉ ASAM sập, nên các hiệu dính tới chúng là cận dưới). 'Kẹt' = bình nguyên > 10 epoch; 'sập' = "
        "test ROC < 0,75. Hiệu ghép cặp theo (nguồn, fold), cùng checkpoint Pha 1, cùng mã tất định; dấu + nghĩa là vế đầu kẹt lâu hơn. "
        "Hai biến thể của cùng ô chỉ khác cờ --recadam / --sam_rho (đã đối chiếu hyperparameters). 6 nguồn = {4CWE, common, full} × {chỉ JS, "
        "JS + C/C++}; hàng 'không có RA' của ASAM gộp thêm 5 cặp không Pha 1 (nop1_asamonly - nop1_adamw).",
    ],
    "table": {
        "head": ["phép so (A - B), đơn vị epoch", "n", "+/-/hoà", "TB (trung vị)", "min .. max", "Wilcoxon p", "theo nguồn",
                 "theo fold", "kẹt > 10: chỉ A vs chỉ B", "sập: chỉ A vs chỉ B"],
        "rows": [
            row("A", "ASAM_noRA (asamonly - noras)", "ASAM khi không RA: chỉ ASAM - noRAS"),
            row("B", "ASAM_withRA (rasam - raonly)", "ASAM khi có RA: RA+ASAM - chỉ RA"),
            row("C", "RA_withASAM (rasam - asamonly)", "RA khi có ASAM: RA+ASAM - chỉ ASAM"),
            row("D", "RA_noASAM (raonly - noras)", "RA khi không ASAM: chỉ RA - noRAS"),
            ["tương tác: (RA+ASAM - chỉ RA) - (chỉ ASAM - noRAS)", str(NUM["new_interaction_plat_0.65"]["n"]),
             f"{NUM['new_interaction_plat_0.65']['pos']}/{NUM['new_interaction_plat_0.65']['neg']}/{NUM['new_interaction_plat_0.65']['tie']}",
             f"{vn(NUM['new_interaction_plat_0.65']['mean'], 1, True)} ({vn(NUM['new_interaction_plat_0.65']['median'], 0, True)})",
             f"{vn(NUM['new_interaction_plat_0.65']['min'], 0, True)} .. {vn(NUM['new_interaction_plat_0.65']['max'], 0, True)}",
             pfmt(NUM["new_interaction_plat_0.65"]["wilcoxon_p"]),
             f"{NUM['new_interaction_plat_0.65']['by_source']['pos']}+/{NUM['new_interaction_plat_0.65']['by_source']['neg']}- trên 6",
             f"{NUM['new_interaction_plat_0.65']['by_fold']['pos']}+/{NUM['new_interaction_plat_0.65']['by_fold']['neg']}- trên 5", "-", "-"],
        ],
    },
    "bullets": [
        "Độ nhạy theo ngưỡng (0,60 / 0,65 / 0,68), dạng +/-/hoà: A 33/0/2 - 33/0/2 - 32/0/3; B 30/0/0 - 24/0/6 - 19/0/11; C 13/8/9 - 8/9/13 - "
        "7/11/12 (dấu ĐỔI theo ngưỡng); D 11/1/18 - 8/1/21 - 9/1/20. A và B vững; C không có hướng ổn định.",
        "C theo nguồn (TB epoch): common JS + C/C++ -5,8 (4/5 ngắn hơn), 4CWE JS + C/C++ -2,4 (cả phần này là f1: 4 so với ≥ 17), full "
        "JS + C/C++ +1,2 (3/5 dài hơn), ba nguồn chỉ JS từ -1,0 tới +0,4. Hiệu lớn chỉ có ở nơi bình nguyên dài, và dấu đổi theo nguồn.",
        "Đo bằng mốc khác, không dùng train loss (epoch đầu tiên val ROC ≥ 0,80; +/-/hoà, trung vị): A 35/0/0 (+3); B 30/0/0 (+2,5); "
        "D 23/0/7 (+1); C 18/8/4 (+1), TB -0,6, Wilcoxon p 0,18. Tức là ngay cả khi có ASAM, RecAdam thường làm mô hình bắt đầu xếp hạng "
        "đúng MUỘN hơn khoảng 1 epoch (đúng kiểu warmup làm chậm), và chỉ ở vài ô có bình nguyên rất dài thì nó rút ngắn mạnh (4CWE JS + "
        "C/C++ f1, common JS + C/C++ f1, f3) - hai tác dụng ngược chiều nên trung vị ≈ 0. Ngưỡng 0,75 cho cùng hình: C 16/8/6.",
        "D nhỏ nhưng nhất quán: chỉ RA so với noRAS có train loss ep1 cao hơn 28/30, val ROC ep1 thấp hơn 29/30 (TB -0,061), bình nguyên "
        "+0,23 epoch (8/1/21) - đây là dấu vết đo được của warmup ngầm (mục 4).",
        "Bốn chỉ số test (TB, số cặp dương/n). A: ROC -0,025 (21/35), F1@0,5 -0,028 (16/35), F1@val -0,023 (17/35), PR -0,025 (19/35), "
        "trung bình bị 3 ô sập kéo. C: ROC +0,025 (17/30), F1@0,5 +0,028 (17/30), F1@val +0,028 (17/30), PR +0,024 (15/30), trung vị "
        "≈ 0 ở cả bốn: phần hơn của cột chính so với chỉ ASAM nằm gần trọn ở 2 ô chỉ ASAM sập. B: ROC +0,016 (25/30), F1@0,5 +0,019 "
        "(25/30), F1@val +0,017 (24/30), PR +0,017 (24/30) - ASAM có ích cho độ chính xác khi không sập.",
    ],
})

# ----------------------------------------------------------------------------- 2. sập = bình nguyên bị cắt
VS = NUM["new_variant_summary"]


def vrow(label, v):
    return [label, str(v["n"]), str(v["stuck10_0.65"]), str(v["collapse"]), f"{vn(v['plat_median'], 0)} ({v['plat_min']} .. {v['plat_max']})"]


S.append({
    "heading": "2. 'Sập' là bình nguyên dài bị luật dừng sớm cắt; RecAdam không làm ít ô kẹt hơn",
    "table": {
        "head": ["biến thể (khối mwonly5)", "n", "kẹt > 10 epoch", "sập (ROC < 0,75)", "bình nguyên trung vị (min .. max), epoch"],
        "rows": [
            vrow("noRAS (Pha 1 + AdamW), 6 nguồn", VS["noras"]),
            vrow("chỉ RecAdam, 6 nguồn", VS["raonly"]),
            vrow("chỉ ASAM, 6 nguồn", VS["asamonly"]),
            vrow("RA + ASAM (cột chính), 6 nguồn", VS["rasam"]),
            vrow("RA + ASAM, 3 nguồn chỉ C/C++ (không có ô chỉ ASAM để ghép)", VS["rasam_ccpp3"]),
            vrow("không Pha 1, AdamW", VS["nop1_adamw"]),
            vrow("không Pha 1, chỉ ASAM", VS["nop1_asamonly"]),
        ],
    },
    "paras": [
        "Cả 3 ô sập đều KHÔNG rời ngưỡng 0,65 tới lúc dừng ở epoch 17. Với min_epochs 10 + patience 8, epoch dừng sớm nhất là 17 (khi best "
        "ở ≤ ep9), nên một ô chỉ sập khi bình nguyên ≥ 17 epoch VÀ val ROC không lập đỉnh mới trong lúc kẹt. Các ô kẹt > 10 epoch mà sống "
        "đều thoát ở ep12-22 và đạt test ROC 0,85-0,95, nên bình nguyên phần lớn là tạm thời; ô sập chưa được chạy tiếp nên chưa biết chúng "
        "có thoát không (phép kiểm 1).",
        "Tỉ lệ kẹt gần như bằng nhau: RA + ASAM 14/45 (31 %) so với các biến thể chỉ ASAM 10/35 (29 %), Fisher p 1,0. Khác biệt chỉ ở đuôi: "
        "bình nguyên ≥ 16 epoch 1/45 so với 5/35 (p 0,08); sập khi đã kẹt 0/14 so với 3/10 (p 0,06); sập tổng 0/45 so với 3/35 (Fisher p 0,08; "
        "trên 30 cặp ghép được thì 0 so với 2, McNemar p 0,5). Một seed, ba ô: chưa phân biệt được với may rủi.",
        "Trong lúc kẹt, val ROC của ô sống nhích dần (+0,015 .. +0,045 mỗi epoch) nên patience được đặt lại; ở 3 ô sập val ROC đi ngang hoặc "
        "trôi xuống (4CWE JS + C/C++ f1: 0,626 ở ep1 rồi TB 0,485, 12 epoch < 0,5; common JS + C/C++ f3: TB 0,450, 13 epoch < 0,5). Độ dốc "
        "val ROC trong bình nguyên: ô chỉ ASAM kẹt ≥ 6 epoch 4/16 có độ dốc ≤ 0,0014 (3 trong đó là 3 ô sập); ô RA + ASAM kẹt 0/24 dưới "
        "0,0016. Ranh giới sát nhau, và val nhích cũng là dấu hiệu sắp thoát nên phép so này có tính vòng.",
        "ASAM khuếch đại độ chậm sẵn có của checkpoint: tương quan hạng Spearman giữa bình nguyên chỉ ASAM và noRAS cùng ô là 0,80 (35 ô). "
        "Nguồn chỉ JS: noRAS 0-1 epoch, chỉ ASAM 1-3 (0 ô kẹt); JS + C/C++: 1-4 → 3-17 (7/15 kẹt); không Pha 1: 2-3 → 9-21 (3/5 kẹt). "
        "Rủi ro vì vậy tập trung ở nguồn JS + C/C++, chỉ C/C++ và không Pha 1 (khớp collapse_A.md của khối cũ: ρ +0,61).",
    ],
})

# ----------------------------------------------------------------------------- 3. epoch đầu
C = NUM["new_cells"]
import numpy as _np


def med3(v, srcs, k):
    xs = [C[f"{s}|{v}|f{f}"][k] for s in srcs for f in range(1, 6) if f"{s}|{v}|f{f}" in C]
    return vn(float(_np.median(xs)), 2)


_s6 = [f"{a}_{b}" for a in ("4cwe", "common", "full") for b in ("jsonly", "jscpp")]
_rows3 = []
for lab, v, ss in [("noRAS, 6 nguồn", "noras", _s6), ("chỉ RecAdam, 6 nguồn", "raonly", _s6), ("chỉ ASAM, 6 nguồn", "asamonly", _s6),
                   ("RA + ASAM, 6 nguồn", "rasam", _s6), ("RA + ASAM, chỉ C/C++", "rasam", ["4cwe_ccpp", "common_ccpp", "full_ccpp"]),
                   ("không Pha 1, AdamW", "noras", ["nop1"]), ("không Pha 1, chỉ ASAM", "asamonly", ["nop1"])]:
    _rows3.append([lab, " / ".join(med3(v, ss, k) for k in ("tl1", "tl2", "tl3")), " / ".join(med3(v, ss, k) for k in ("vr1", "vr2", "vr3"))])
S.append({
    "heading": "3. Những epoch đầu: không học nổi ngay từ đầu, không phải học rồi sụp",
    "table": {"head": ["biến thể (trung vị)", "train loss ep1 / ep2 / ep3", "val ROC ep1 / ep2 / ep3"], "rows": _rows3},
    "paras": [
        "0/165 ô của khối này (và 0/444 ô khối cũ) từng xuống dưới 0,65 rồi quay lại ≥ 0,69: không có kiểu 'học rồi sụp' ở train loss. "
        "Train loss ep1 > ln2 ở mọi biến thể (head mới ngẫu nhiên, 'head luôn mới' trong init_from); khác biệt nằm ở ep2-3: noRAS rơi "
        "0,70 → 0,47 → 0,30 còn chỉ ASAM đứng 0,74 → 0,70 → 0,67 (JS + C/C++, C/C++ và không Pha 1 đứng 0,71-0,73). Ghép cặp chỉ ASAM - "
        "noRAS: train loss ep2 cao hơn 33/35, val ROC ep1 thấp hơn 31/35 (TB -0,105). Trong lúc kẹt, train loss TB ep2-10 = 0,70-0,72, tức "
        "đứng QUANH và phía trên ln2 = 0,693 (dropout cũng góp phần), không nằm đúng ln2.",
        "Không có cú vọt sau epoch đầu: train loss ep2 hoặc ep3 > ep1 chỉ ở 2/30 ô chỉ ASAM và 0/30 ô RA + ASAM (không Pha 1: 3/5). Vì vậy "
        "giả thuyết 'Pha 2 không warmup nên các bước đầu với lr đủ 5,66e-5 phá đặc trưng, RecAdam đỡ nhờ bước nhỏ' không để lại dấu vết ở mức "
        "epoch. History chỉ lưu theo epoch (15 bước/epoch) nên không loại được một cú vọt bên trong epoch 1.",
        "Riêng val ROC: ở 2 ô chỉ ASAM sập có Pha 1, val ROC giảm sau ep1 trong khi train loss đứng yên - tín hiệu từ đặc trưng Pha 1 bị trôi "
        "mất khi head chưa học được. Đây là 'trôi đặc trưng trong bình nguyên', không phải 'học rồi quên'.",
    ],
})

# ----------------------------------------------------------------------------- 4. cơ chế
pe = sn["per_epoch"]
pe_o = so["per_epoch"]
S.append({
    "heading": "4. Cơ chế trong mã (src_mwonly) và lịch λ(t) theo bước thật",
    "paras": [
        "ASAM (mwg/optim.py, SAMStep._ascend_asam): ε = ρ·T²·g/‖T·g‖ với ρ = 0,5 cố định, T = |w| + 0,01 cho mọi tham số có chữ 'weight' "
        "trong tên (kể cả LayerNorm, embedding và head), T = 1 cho bias, đúng bản chính thức. Thứ tự trong train.py: backward tại w, leo tới "
        "w + ε, backward lần hai, trả về w, clip 1,0, rồi optimizer bước với gradient TẠI w + ε. ρ không co theo lr và không co theo độ lớn "
        "gradient, nên ngay từ bước 1, khi head mới ngẫu nhiên và đầu ra gần hằng (loss ≈ ln2), mọi bước đều được tính ở một điểm cách w một "
        "khoảng cố định. Probe CPU cũ (review_pack/collapse_A.md, khối cũ, 4 checkpoint × 1 batch): loss tại w + ε gấp 1,2-2,6 lần tại w; ở "
        "checkpoint có bình nguyên dài cos(g(w), g(w + ε)) của backbone chỉ 0,18 và -0,11 (head 0,90-0,96); 72-81 % ngân sách ‖T·g‖² rơi vào "
        "bias. Tức bước cập nhật backbone gần như không tương quan với hướng giảm loss tại w: chỉ head học được, nên thoát chậm.",
        "RecAdam (mwg/optim.py, RecAdam.step): λ(t) = sigmoid(k·(t - t0)) NHÂN thẳng vào bước Adam (p -= step_size·λ·m/√v) và thêm kéo về "
        "neo p -= lr·(1 - λ)·500·(p - p0). Neo = trọng số ngay sau init_from, GỒM CẢ head mới ngẫu nhiên (mã gốc của Chen et al. đặt "
        "anneal_w = 0 cho head). t0 = max(1, int(0,01 × 450)) = 4. Một khác biệt kéo dài cả quá trình: eps của RecAdam 1e-6, của AdamW 1e-8 "
        "(nhiều khả năng nhỏ, chưa đo). ε của ASAM tính TRƯỚC bước optimizer nên RecAdam không làm nhỏ nhiễu loạn, chỉ làm nhỏ bước và kéo về neo.",
        "Ghi chú cũ 'λ(t) nhân vào bước nên t0 là đóng băng': đúng về cơ chế, nhưng với t0 = 4 thì KHÔNG đóng băng (λ(1) = 0,46). Đóng băng "
        "chỉ có khi t0 lớn: quét khối cũ (chỉ RecAdam, MWG common chỉ JS, f1-f3) t0 0,01 / 0,05 / 0,10 / 0,30 cho bình nguyên 2 / 2 / 3-4 / "
        "10 epoch.",
        "Warmup ngầm là có thật (bảng): ở khối này RecAdam là một warmup bắt đầu từ 46 % (không từ 0), đạt 90 % ở epoch 4, cộng một vùng "
        "tin cậy kiểu L2-SP: nếu bước Adam giữ một hướng thì độ dời mỗi toạ độ bị chặn quanh λ/((1 - λ)·500) ≈ 0,002 (ep1), 0,005 (ep2), "
        "0,01 (ep3) (suy luận từ công thức). Từ epoch 7 (λ ≥ 0,99, hệ số kéo < 0,02 % mỗi bước) RecAdam gần như là AdamW, nên mọi bình nguyên "
        "dài hơn 7 epoch phần lớn diễn ra khi RecAdam đã 'tắt': RecAdam chỉ có thể tác động qua trạng thái nó để lại ở khoảng epoch 5-7. Ở "
        "khối cũ có warmup thật 3 epoch thì RecAdam gần như không thêm gì (kéo tích luỹ 0,036).",
    ],
    "table": {
        "head": ["đại lượng", "khối mwonly5 (batch 32, lr 5,66e-5)", "khối cũ (batch 8, lr 2e-5)"],
        "rows": [
            ["bước mỗi epoch / tổng bước", f"{sn['steps_per_epoch']} / {sn['total_steps']}", f"{so['steps_per_epoch']} / 1 710"],
            ["warmup tường minh", "0", f"{so['warmup_steps']} bước (3 epoch, lr từ 0)"],
            ["t0 (bước); λ ở bước 1", f"{sn['t0']}; {vn(sn['lam_step1'], 2)}", f"{so['t0']}; {vn(so['lam_step1'], 2)}"],
            ["λ ≥ 0,9 / λ ≥ 0,99", f"bước {sn['lam_ge']['0.9']['step']} (ep{sn['lam_ge']['0.9']['epoch']}) / bước {sn['lam_ge']['0.99']['step']} (ep{sn['lam_ge']['0.99']['epoch']})",
             f"bước {so['lam_ge']['0.9']['step']} (ep{so['lam_ge']['0.9']['epoch']}) / bước {so['lam_ge']['0.99']['step']} (ep{so['lam_ge']['0.99']['epoch']})"],
            ["λ trung bình epoch 1 / 2 / 3 / 4", " / ".join(vn(e["lam_mean"], 2) for e in pe[:4]), " / ".join(vn(e["lam_mean"], 2) for e in pe_o[:4])],
            ["lr hiệu dụng bước 1: AdamW vs RecAdam", "5,66e-5 vs 2,62e-5", "0 vs 0 (warmup)"],
            ["Σ lr hiệu dụng epoch 1-3: AdamW vs RecAdam", "2,42e-3 vs 1,70e-3 (-30 %)", "1,70e-3 vs 1,63e-3 (-4 %)"],
            ["hệ số kéo về neo mỗi bước, epoch 1 / 2 / 3", " / ".join(vn(100 * e["pull_coef_mean"], 2) + " %" for e in pe[:3]),
             " / ".join(vn(100 * e["pull_coef_mean"], 3) + " %" for e in pe_o[:3])],
            ["kéo tích luỹ Σ lr·(1 - λ)·500", vn(sn["cum_pull_total"], 2), vn(so["cum_pull_total"], 3)],
        ],
    },
})

# ----------------------------------------------------------------------------- 5. khối cũ + ASAM muộn
S.append({
    "heading": "5. Khối cũ (batch 8, lr 2e-5, warmup 10 %, min_epochs 3) và phép can thiệp ASAM muộn",
    "table": {
        "head": ["phép so (khối cũ), đơn vị epoch", "n", "+/-/hoà", "TB (trung vị)", "min .. max", "Wilcoxon p"],
        "rows": [
            orow("ASAM_noRA (asamonly - noras)", "ASAM khi không RA"),
            orow("ASAM_withRA (rasam - raonly)", "ASAM khi có RA"),
            orow("RA_withASAM (rasam - asamonly)", "RA khi có ASAM"),
            orow("RA_noASAM (raonly - noras)", "RA khi không ASAM"),
            ["tương tác", str(oi["n"]), f"{oi['pos']}/{oi['neg']}/{oi['tie']}", f"{vn(oi['mean'], 1, True)} ({vn(oi['median'], 0, True)})",
             f"{vn(oi['min'], 0, True)} .. {vn(oi['max'], 0, True)}", pfmt(oi["wilcoxon_p"])],
            ["ASAM muộn (K15) - RA + ASAM gốc", str(late["n"]), f"{late['pos']}/{late['neg']}/{late['tie']}",
             f"{vn(late['mean'], 1, True)} ({vn(late['median'], 0, True)})", f"{vn(late['min'], 0, True)} .. {vn(late['max'], 0, True)}",
             pfmt(late["wilcoxon_p"])],
        ],
    },
    "paras": [
        "Ghép cặp tự động theo cùng checkpoint Pha 1, cùng seed, cùng TF32/FP32, cùng kiến trúc (MW hoặc MWG), cùng fold, chỉ khác cờ "
        "optimizer (archive_20261004: meta/dbrows + results + dòng TF32 trong log). ASAM kéo dài bình nguyên ở MỌI cặp (49/49, 64/64) dù khối "
        "cũ CÓ warmup tường minh 3 epoch và lr nhỏ hơn ⇒ warmup không ngăn được bình nguyên của ASAM.",
        "Hướng của RecAdam ĐẢO giữa hai khối: khối cũ RA + ASAM - chỉ ASAM +1,2 epoch (22 dài hơn / 1 ngắn hơn / 6 bằng), tương tác +0,7 "
        "(16/4/9, p 0,008), và RA + ASAM sập 20/180 ô so với chỉ ASAM 11/84 (phần lớn ô chỉ ASAM sập là seed 1234); khối mới -1,2 (8/9/13), "
        "tương tác -1,5 (6/11/13, p 0,15). Không Pha 1 ở khối cũ: AdamW 3 epoch, chỉ RA 3-4, chỉ ASAM 6-7, RA + ASAM 7-10 (RecAdam thêm vào "
        "ASAM dài hơn ở 5/5 fold; chỉ tham khảo vì chỉ ASAM chạy FP32 còn RA + ASAM chạy TF32). Hai khối khác nhau ở batch, lr, warmup, min_epochs và Pha 1 (nguồn, pair loss) nên KHÔNG tách được vì "
        "sao dấu đảo. Một giả thuyết khớp cả hai: làm nhỏ bước (warmup, λ(t)) trong lúc ASAM bật kéo dài thời gian ở vùng yên ngựa (rõ ở khối "
        "cũ); ở khối mới hiệu ứng đó bị lấn bởi khác biệt quỹ đạo theo từng nguồn.",
        "Phép can thiệp sạch nhất trong repo là K15 (khối cũ, bậc 1, n = 3, CURRENT_RUN.md): RA + ASAM nhưng ρ = 0 cho tới khi train loss một "
        "epoch < 0,65. Trên 15 ô (5 nguồn × f1-f3): bình nguyên ngắn hơn RA + ASAM gốc 15/15 (TB -7,9 epoch), BẰNG ĐÚNG bình nguyên chỉ "
        "RecAdam ở 15/15 (các epoch trước khi bật ASAM trùng bit), sập 0/15 so với 8/15 ở ô gốc; ROC so với ô gốc không sụp -0,007 (1/7), "
        "so với AdamW +0,025 (15/15). Đây là bằng chứng nhân quả: bỏ ASAM ở giai đoạn đầu thì bỏ được bình nguyên, còn RecAdam (giữ nguyên ở cả "
        "hai nhánh) không quyết định gì.",
    ],
})

# ----------------------------------------------------------------------------- 6. tài liệu ML
S.append({
    "heading": "6. Tài liệu học máy (đã kiểm; [PDF] = đọc thân bài, [TT] = chỉ tóm tắt)",
    "bullets": [
        "Kwon et al., ASAM, ICML 2021 [PDF]: ρ = 0,5 là giá trị chọn cho CIFAR-10 với SGD 'because it gives moderately good performance across "
        "various models'; với Transformer + Adam (IWSLT'14) họ chọn 'ρ = 0.2 for ASAM' bằng lưới trên val. Chú thích bảng ghi có run 'completely "
        "failed' (VGG19-BN, CIFAR-10: thành công SGD 3/5, SAM 1/5, ASAM 3/5). Ta dùng ρ 0,5 cho Adam + CodeBERT từ bước 1: ỦNG HỘ việc ρ 0,5 "
        "quá lớn cho bối cảnh này.",
        "Kim et al., Stability Analysis of Sharpness-Aware Minimization, ICML 2026 [TT]: 'the saddle point can become an attractor under SAM "
        "dynamics'; momentum và batch size quan trọng để giảm bất ổn. Theo collapse_fix_D.md (đã đọc thân bài): SAM thoát yên ngựa chậm hơn "
        "SGD, tệ hơn khi ρ lớn, batch nhỏ giúp thoát; khối mới tăng batch 8 → 32. ỦNG HỘ P1 (cơ chế yên ngựa).",
        "Compagnoni et al., An SDE for Modeling SAM, ICML 2023 [TT]: 'SAM is attracted to saddle points under some realistic conditions'. "
        "ỦNG HỘ P1.",
        "Mosbach et al., On the Stability of Fine-tuning BERT, ICLR 2021 [PDF]: run thất bại có train loss 'close to - ln(1/2)', do khó tối "
        "ưu dẫn tới gradient biến mất; 'What is crucial is rather the number of training iterations'. Khớp hiện tượng (ln2, 456 mẫu, chỉ 15 "
        "bước/epoch ở batch 32), nhưng ở ta AdamW trên cùng checkpoint không kẹt nên tác nhân là ASAM; BỔ SUNG. Bài gọi bias correction của "
        "Adam là 'implicit warmup' ('The implicit warmup of ADAM is likely to be an important factor'); cả hai nhánh của ta đều có bias "
        "correction nên đó không phải chỗ khác nhau - warmup ngầm riêng của RecAdam là λ(t).",
        "Chen et al., Recall and Learn (RecAdam), EMNLP 2020 [TT + mã gốc]: 'objective shifting' chuyển dần trọng tâm sang tác vụ đích. Mã "
        "gốc mặc định k 0,5, t0 250 bước, γ 5 000 và anneal_w = 0 cho head (head học Adam thường). Cấu hình ta (k 0,05, t0 4, γ 500, head bị "
        "neo) khác hẳn; với mặc định gốc, khoảng 240 bước đầu backbone gần như đứng yên còn head học - gần LP-FT hơn là warmup. Bài không bàn "
        "SAM: KHÔNG có cơ sở tài liệu cho P2.",
        "Zhou et al., SAM Efficiently Selects Flatter Minima Late in Training, ICLR 2025 [TT]: 'Even a few epochs of SAM applied at the end of "
        "training yield nearly the same generalization ... as full SAM training'. ỦNG HỘ cách sửa 'ASAM muộn' (K15) thay vì dựa vào RecAdam.",
        "Dodge et al., arXiv 2002.06305, 2020 [TT]: trên tập nhỏ 'many fine-tuning trials diverge part of the way through training'; khởi "
        "tạo head và thứ tự dữ liệu góp phương sai ngang nhau. Seed 42 ghim cùng head cho mọi fold ⇒ khối này chưa lấy mẫu phương sai do "
        "khởi tạo head; BỔ SUNG (lý do phải lặp seed trước khi tin 0/45 vs 3/35).",
        "Không tìm thấy (ở đây và ở collapse_fix_D.md) công trình nào đo riêng cặp RecAdam + SAM/ASAM.",
    ],
})

# ----------------------------------------------------------------------------- 7. SE / AI4SE
if SE:
    S.append(SE)
else:
    S.append({"heading": "7. Nghiên cứu liên quan (SE / AI4SE)", "paras": ["(đang tra cứu)"]})

# ----------------------------------------------------------------------------- 8. phát biểu
S.append({
    "heading": "8. Phát biểu nào đứng, cái gì còn là giả thuyết",
    "bullets": [
        "ĐỨNG, mức cao: ASAM ρ 0,5 bật từ bước đầu là nguyên nhân của bình nguyên ln2 ở Pha 2. Khối mới 33/35 và 24/30 cặp, 0 cặp ngược, "
        "5/5 fold ở cả hai phép so (sàn p 0,0625); khối cũ 49/49 và 64/64; can thiệp ASAM muộn 15/15 và 0/15 sập. Vững với ngưỡng 0,60-0,68.",
        "ĐỨNG, mức cao: 'sập' là bình nguyên dài bị luật dừng sớm cắt (3/3 ô sập không rời 0,65 tới epoch dừng 17; ô kẹt mà sống thoát ở "
        "ep12-22, ROC 0,85-0,95). Không có kiểu 'học rồi sụp' ở train loss (0/165, 0/444).",
        "ĐỨNG, mức trung bình: ASAM khuếch đại độ chậm sẵn có của checkpoint (Spearman 0,80) nên rủi ro tập trung ở nguồn JS + C/C++, chỉ "
        "C/C++ và không Pha 1; nguồn chỉ JS gần như miễn nhiễm.",
        "KHÔNG ĐỨNG: 'RecAdam làm giảm vấn đề'. Không rút ngắn bình nguyên (8/9/13, trung vị 0, dấu đổi theo ngưỡng và theo nguồn), không "
        "giảm số ô kẹt (14/45 so với 10/35), và ngược chiều với khối cũ (+1,2 epoch, 22/1). 0/45 so với 3/35 ô sập là một quan sát ở đuôi, "
        "p ≈ 0,08, một seed; 15/45 ô cột chính nằm ở nguồn chỉ C/C++ không có ô chỉ ASAM để so.",
        "GIẢ THUYẾT (H1, chưa kiểm): warmup ngầm của RecAdam đỡ ASAM khi Pha 2 không warmup. Cơ chế có thật trong mã và đo được ở nhánh không "
        "ASAM (mục 1, D), nhưng không thấy bình nguyên ASAM ngắn lại; khối cũ có warmup thật mà ASAM vẫn kẹt 49/49 và RecAdam còn làm dài thêm. "
        "Phép kiểm 2 trả lời trực tiếp.",
        "GIẢ THUYẾT (H2): neo RecAdam giữ đặc trưng Pha 1 nên val ROC không trôi xuống trong lúc kẹt, nhờ vậy ô kẹt không bị luật dừng cắt. "
        "Khớp số ở mục 2 (độ dốc val ROC) nhưng sát ranh giới và có tính vòng.",
        "GIẢ THUYẾT (H3): chênh lệch số ô sập là ngẫu nhiên (thời điểm thoát khỏi yên ngựa rất nhạy với nhiễu nhỏ, collapse_B.md: độ tản tăng "
        "theo độ dài bình nguyên). Phép kiểm 3 (lặp seed) phân biệt H3 với H1/H2.",
        "Không nên viết 'RecAdam ổn định Pha 2' như một phát hiện. Nếu nhắc, ghi đúng: 'RA + ASAM 0/45 ô sập so với chỉ ASAM 3/35, một seed, "
        "p ≈ 0,08; độ dài bình nguyên và tỉ lệ kẹt không khác'. Cách sửa có cơ sở nhất cho bình nguyên là ASAM muộn hoặc ρ nhỏ hơn, không phải RecAdam.",
    ],
})

report["next"] = [
    "1 (rẻ nhất, làm trước): chạy lại 3 ô chỉ ASAM đã sập (asamonly_4cwe_jscpp f1, asamonly_common_jscpp f3, nop1_asamonly f4) với "
    "--min_epochs 30, giữ --epochs 30 nên lịch LR không đổi; mã tất định nên 17 epoch đầu phải trùng bit log cũ (cổng kiểm sẵn). Trả lời: "
    "bình nguyên tạm thời bị cắt hay kẹt vĩnh viễn. 3 ô × ~25 phút ≈ 1,3 GPU-giờ, không sửa mã.",
    "2 (kiểm thẳng warmup ngầm): chỉ ASAM + warmup tuyến tính 60 bước (--warmup_ratio 0.1333, đúng mốc λ ≥ 0,9 của RecAdam) trên 3 nguồn "
    "JS + C/C++, f1-f3. Nếu H1 đúng: bình nguyên ≤ RA + ASAM cùng ô ở ≥ 7/9; nếu sai: ≈ chỉ ASAM hoặc dài hơn (như khối cũ). 9 ô × ~20 phút "
    "≈ 3 GPU-giờ, không sửa mã; lưu ý warmup cũng đổi lịch giảm LR phía sau.",
    "3 (kiểm 0/45 vs 3/35 có thật không): lặp seed Pha 2 (1234, rồi 7; giữ checkpoint Pha 1) cho RA + ASAM và chỉ ASAM trên 3 nguồn JS + "
    "C/C++, f1-f3. Seed 1234: 18 ô ≈ 6 GPU-giờ; thêm seed 7 ≈ 12 GPU-giờ. Đọc theo fold, không gộp seed thành cặp độc lập.",
    "4 (cách sửa có cơ sở nhất): ASAM muộn trong cấu hình mới (ρ = 0 tới khi train loss < 0,65) cho chỉ ASAM và RA + ASAM, 3 nguồn JS + "
    "C/C++, f1-f3: lặp K15. Cần chuyển cờ --sam_start_loss từ src_final_lateasam sang src_mwonly (sửa mã, phải hỏi). 18 ô ≈ 4,5 GPU-giờ. "
    "Dự đoán: bình nguyên = chỉ RA / noRAS, 0 sập.",
    "5: ρ 0,2 (giá trị ASAM chọn cho Adam + Transformer) cho chỉ ASAM, 3 nguồn JS + C/C++, f1-f3: kiểm 'bình nguyên giảm theo ρ'. 9 ô ≈ "
    "2,5 GPU-giờ, không sửa mã.",
    "Không cần GPU: thêm cột 'bình nguyên (epoch)' và 'kẹt / sập' vào bảng artifact cho mọi ô Pha 2, để tỉ lệ kẹt được báo như một chỉ số "
    "của phương pháp thay vì chỉ đếm ô sập.",
]

report["sources"] = [
    "Số liệu: meta/insights/asam_collapse_numbers.json (sinh bởi meta/insights/asam_collapse_analysis.py từ meta/dbrows/*.json, "
    "results/RUN/foldK.json, archive_20261004/{meta/dbrows,results,logs})",
    "Mã: /drive1/cuongtm/ntat/MultiVD/src_mwonly/train.py (build_optimizer, vòng train, init_from), src_mwonly/mwg/optim.py (RecAdam, SAMStep); "
    "src_final/RecAdam.py, src_final/sam.py (khối cũ, cùng phép tính)",
    "Cấu hình: _FinalPaperExperiment/scripts/runs.json (flag_sets), archive_20261004/scripts_snapshot/runs.json",
    "Ghi chép: /drive1/cuongtm/ntat/MultiVD/CURRENT_RUN.md (mục mwonly5, CẢNH BÁO, K15), FACTS.md §87.2, §87.6",
    "Phân tích cũ: archive_20261004/meta/review_pack/collapse_A.md (probe CPU, ICC), collapse_B.md (Kaplan-Meier), collapse_fix_D.md (tài liệu)",
    "Kwon et al., ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks, ICML 2021, https://arxiv.org/abs/2102.11600",
    "Kim et al., Stability Analysis of Sharpness-Aware Minimization, ICML 2026, https://arxiv.org/abs/2301.06308",
    "Compagnoni et al., An SDE for Modeling SAM: Theory and Insights, ICML 2023, https://arxiv.org/abs/2301.08203",
    "Mosbach et al., On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines, ICLR 2021, https://arxiv.org/abs/2006.04884",
    "Chen et al., Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting, EMNLP 2020, https://doi.org/10.18653/v1/2020.emnlp-main.634; "
    "mã gốc https://github.com/Sanyuan-Chen/RecAdam (run_glue_with_RecAdam.py)",
    "Zhou et al., Sharpness-Aware Minimization Efficiently Selects Flatter Minima Late in Training, ICLR 2025, https://arxiv.org/abs/2410.10373",
    "Dodge et al., Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping, arXiv 2020, https://arxiv.org/abs/2002.06305",
]
if SE and SE.get("_sources"):
    report["sources"] += SE.pop("_sources")
