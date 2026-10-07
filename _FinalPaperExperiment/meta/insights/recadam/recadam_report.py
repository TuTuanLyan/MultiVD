#!/usr/bin/env python3
"""Dựng recadam_contribution.json / .md từ numbers.json (do make_recadam_figures.py sinh).

Mọi con số trong văn bản lấy từ numbers.json, nên chạy lại make_recadam_figures.py (ví dụ sau khi 3 ô me30 xong)
là văn bản tự cập nhật. Phần diễn giải định tính cố định; riêng đoạn về ô me30 đổi theo dữ liệu có sẵn.
Văn bản tiếng Việt: số thập phân dấu phẩy, chỉ gạch ngắn '-'.
"""
import json
import os
import re
import sys
from datetime import datetime

OUT = os.path.dirname(os.path.abspath(__file__))
QUESTION = ("Gọi agent song song phân tích chi tiết kết quả để xem đóng góp của recadam, có thể vẽ biểu đồ + bảng phù hợp "
            "nếu đưa vào paper để phân tích")
SRC6 = ["4cwe_jsonly", "common_jsonly", "full_jsonly", "4cwe_jscpp", "common_jscpp", "full_jscpp"]
SRC_VI = {"4cwe_jsonly": "4CWE chỉ JS", "common_jsonly": "common chỉ JS", "full_jsonly": "full chỉ JS",
          "4cwe_jscpp": "4CWE JS + C/C++", "common_jscpp": "common JS + C/C++", "full_jscpp": "full JS + C/C++"}
VAR_VI = {"noras": "AdamW (noRAS)", "raonly": "chỉ RecAdam", "asamonly": "chỉ ASAM", "rasam": "RecAdam + ASAM"}
MK = [("f1_05", "F1@0,5"), ("f1_val", "F1@val"), ("roc", "ROC-AUC"), ("pr", "PR-AUC")]


def v(x, nd=3, sign=False):
    if x is None:
        return "-"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if sign and float(s) == 0:
        s = ("±" + s[1:])
    return s.replace(".", ",")


def ci(st, nd=3):
    c = st.get("ci_boot")
    return f"[{v(c[0], nd, True)}; {v(c[1], nd, True)}]" if c else "[-]"


def full(st, nd=3, folds=True):
    """'+0,001 [-0,008; +0,011], 14/28 cặp, 3/5 fold'."""
    if not st or st.get("n", 0) == 0:
        return "-"
    s = f"{v(st['mean'], nd, True)} {ci(st, nd)}, {st['pos']}/{st['n']} cặp"
    if folds:
        s += f", {st['fold_pos']}/{st['fold_n']} fold"
    return s


def short(st, nd=3):
    """'+0,001 (14/28)'."""
    if not st or st.get("n", 0) == 0:
        return "-"
    return f"{v(st['mean'], nd, True)} ({st['pos']}/{st['n']})"


def cell(st, nd=3):
    """ô bảng: 'Δ [CI] k/n; fold k/5'."""
    if not st or st.get("n", 0) == 0:
        return "-"
    return f"{v(st['mean'], nd, True)} {ci(st, nd)} {st['pos']}/{st['n']}; fold {st['fold_pos']}/{st['fold_n']}"


def build(num):
    D, DD, S = num["deltas"], num["derived_deltas"], num["stability"]
    d = lambda con, g, mode, k: D.get(f"{con}|{g}|{mode}|{k}") or DD.get(f"{con}|{g}|{mode}|{k}")
    pv, sch, agree = num["per_variant"], num["schedule"], num["pairwise_agreement"]
    seed = num.get("seed_reference")
    fam = num["family_interaction"]
    isim = num["inter_source_similarity"]
    ind = num.get("indirect_figure", {})
    main = num["main_table"]
    me30, prog = num.get("me30", {}), num.get("me30_progress", {})
    has_me30_2x2 = any(k.startswith(("asamonly_4cwe_jscpp f1", "asamonly_common_jscpp f3")) for k in me30)

    # ------------------------------------------------------------------ các số then chốt
    ra_w_all, ra_w_ex = d("ra_with_asam", "all", "all", "roc"), d("ra_with_asam", "all", "excl", "roc")
    ra_o_all, ra_o_cpp = d("ra_without_asam", "all", "all", "roc"), d("ra_without_asam", "jscpp", "all", "roc")
    ra_o_js = d("ra_without_asam", "jsonly", "all", "roc")
    inter_all, inter_ex = d("interaction", "all", "all", "roc"), d("interaction", "all", "excl", "roc")
    asam_js = d("asam_without_ra", "jsonly", "all", "roc")
    full_js, full_cpp = d("full_vs_noras", "jsonly", "all", "roc"), d("full_vs_noras", "jscpp", "all", "roc")
    ra_w_js = d("ra_with_asam", "jsonly", "all", "roc")
    vr1_o, ep85_o = d("ra_without_asam", "all", "all", "vr1"), d("ra_without_asam", "all", "all", "ep_to_val85")
    vrm_o = d("ra_without_asam", "all", "all", "vr_mean_1_10")
    tl2_o = d("ra_without_asam", "all", "all", "tl2")
    coll_a, coll_r = S["asamonly"]["collapse"], S["rasam"]["collapse"]
    st_a, st_r = S["asamonly"]["stuck10"], S["rasam"]["stuck10"]
    sdp = S["paired"]

    # đoạn me30 đổi theo dữ liệu
    me30_lines = []
    for key in ("asamonly_4cwe_jscpp f1", "asamonly_common_jscpp f3", "nop1_asamonly f4"):
        run, f = key.split(" f")
        if key in me30:
            m = me30[key]
            esc = "rời bình nguyên" if not m["plateau_censored"] else "KHÔNG rời bình nguyên tới epoch 30"
            me30_lines.append(f"{run}_me30 f{f}: test ROC {v(m['roc'])}, F1@val {v(m['f1_val'])}, chọn epoch {m['best_epoch']}, "
                              f"dừng epoch {m['stop_epoch']}, {esc} (bình nguyên {m['plateau']} epoch)")
        elif f"{run}_me30 f{f}" in prog:
            p = prog[f"{run}_me30 f{f}"]
            me30_lines.append(f"{run}_me30 f{f}: CHƯA xong, log tới epoch {p['last_epoch']}, train loss {v(p['train_loss'], 4)}, "
                              f"val ROC {v(p['val_roc'], 4)} (train loss nhỏ nhất {v(p['min_train_loss'], 4)})")
        else:
            me30_lines.append(f"{run}_me30 f{f}: chưa có log ở máy này (chạy trên 158, chưa đồng bộ về)")
    # diễn giải: ô nào kẹt VĨNH VIỄN (không rời 0,65 tới epoch 30 hoặc vẫn sập), ô nào chỉ bị luật dừng cắt
    src_of = {"asamonly_4cwe_jscpp f1": ("4cwe_jscpp", 1), "asamonly_common_jscpp f3": ("common_jscpp", 3)}
    perm, temp = [], []
    for key, (s_, f_) in src_of.items():
        if key not in me30:
            continue
        m = me30[key]
        ra = main[f"rasam|{s_}"]["roc"]["folds"][f_ - 1]
        orig = main[f"asamonly|{s_}"]["roc"]["folds"][f_ - 1]
        if m["plateau_censored"] or m["roc"] < 0.75:
            perm.append(f"chỉ ASAM {SRC_VI[s_]} f{f_} kẹt suốt 30 epoch (test ROC {v(m['roc'])}, ô gốc {v(orig)}) trong khi "
                        f"RecAdam + ASAM cùng nguồn, cùng fold đạt {v(ra)}")
        else:
            temp.append(f"chỉ ASAM {SRC_VI[s_]} f{f_} thoát ở epoch {m['plateau'] + 1} (test ROC {v(m['roc'])}, ô gốc {v(orig)}; "
                        f"RecAdam + ASAM cùng fold {v(ra)})")
    nop = me30.get("nop1_asamonly f4")
    nop_txt = (f"ô không Pha 1 chỉ ASAM f4 (ngoài thiết kế 2×2) thoát ở epoch {nop['plateau'] + 1} và đạt test ROC {v(nop['roc'])} "
               f"(ô gốc dừng ở epoch 17)") if nop and not nop["plateau_censored"] else ""
    pending = [k for k in src_of if k not in me30]
    me30_interp = ""
    if perm or temp or nop_txt:
        parts = perm + temp + ([nop_txt] if nop_txt else [])
        me30_interp = "Chạy lại với --min_epochs 30 (17 epoch đầu trùng ô gốc): " + "; ".join(parts) + "."
        if perm:
            me30_interp += (" Ở ô kẹt vĩnh viễn, khác biệt giữa có và không RecAdam KHÔNG phải do luật dừng sớm: cùng checkpoint, "
                            "cùng fold, chỉ ASAM không bao giờ rời ln2 còn RecAdam + ASAM thoát. Đây là bằng chứng cấp một ô (n = 1, "
                            "một seed) cho vai trò ổn định hoá của RecAdam khi đi cùng ASAM.")
        if temp or nop_txt:
            me30_interp += (" Ô thoát được cho thấy một phần 'sập' chỉ là bình nguyên dài bị cắt, nên tỉ lệ sập phụ thuộc luật "
                            "dừng chứ không chỉ phụ thuộc tối ưu.")
        if pending:
            me30_interp += " Còn chờ: " + ", ".join(k.replace(" f", "_me30 f") for k in pending) + "."
    if has_me30_2x2:
        me30_d = d("ra_with_asam", "all", "me30", "roc")
        repl = [k for k in src_of if k in me30]
        same = [k for k in repl if abs(me30[k]["roc"] - main[f"asamonly|{src_of[k][0]}"]["roc"]["folds"][src_of[k][1] - 1]) < 1e-9]
        me30_sent = (f"Phép nhạy 'thay ô sập bằng bản chạy lại me30' (đã có: {', '.join(repl)}): RecAdam khi có ASAM "
                     f"{full(me30_d)} (ROC)."
                     + (f" Bản me30 của {', '.join(same)} cho đúng ROC cũ (kẹt suốt 30 epoch) nên hiệu không đổi so với bản "
                        f"tính đủ." if same else "")
                     + (f" Còn chờ: {', '.join(pending)}." if pending else ""))
    else:
        me30_sent = ("Ô me30 của 2 ô sập trong thiết kế 2×2 chưa có kết quả khi sinh báo cáo này; chạy lại "
                     "make_recadam_figures.py sau khi chúng xong để thêm phép nhạy 'thay ô sập bằng bản me30'.")

    verdict = (
        f"RecAdam KHÔNG đóng góp độ chính xác trong khối mwonly5: bỏ 2 ô sập thì RecAdam + ASAM - chỉ ASAM là "
        f"{v(ra_w_ex['mean'], 3, True)} ROC ({ra_w_ex['pos']}/{ra_w_ex['n']} cặp, {ra_w_ex['fold_pos']}/5 fold) và chỉ RecAdam - "
        f"AdamW là {v(ra_o_all['mean'], 3, True)} ROC ({ra_o_all['pos']}/{ra_o_all['n']}), riêng nguồn JS + C/C++ "
        f"{v(ra_o_cpp['mean'], 3, True)} ({ra_o_cpp['pos']}/{ra_o_cpp['n']} cặp, 0/3 nguồn); cả bốn chỉ số đều nằm trong sàn "
        f"nhiễu 0,010, còn phần hơn của cột chính so với AdamW trên nguồn chỉ JS ({v(full_js['mean'], 3, True)} ROC, "
        f"{full_js['fold_pos']}/5 fold) là của ASAM. Đóng góp duy nhất nhìn thấy là ổn định hoá khi đi cùng ASAM: {coll_r}/30 ô "
        f"RecAdam + ASAM sập so với {coll_a}/30 ô chỉ ASAM, và chính 2 ô đó tạo ra toàn bộ hiệu trung bình "
        f"{v(ra_w_all['mean'], 3, True)} ROC lẫn hiệu tương tác; ngoài 2 ô đó độ tản theo fold không giảm, và đây là một seed"
        + ((". Chạy lại với min_epochs 30: " + "; ".join(perm) + " - tức ít nhất một ô là kẹt thật chứ không do luật dừng. ")
           if perm else ". ") +
        f"Tác dụng đo chắc nhất là động học: RecAdam là một warmup ngầm làm mô hình lên val ROC chậm khoảng 1 epoch "
        f"({ep85_o['pos']}/{ep85_o['n']} cặp chậm hơn, 0 cặp nhanh hơn, {ep85_o['fold_pos']}/5 fold), không phải bằng chứng "
        f"'chống quên' (dữ liệu hiện có không đo được việc quên nguồn).")

    tier = ("Bậc 2 (xác nhận): khối mwonly5, MỘT seed (42), 5 fold đích SVEN Python, 6 nguồn Pha 1 có đủ thiết kế 2×2 (120 ô; "
            "kiểm hyperparameters: 120/120 ô chỉ khác cờ --recadam / --sam_rho so với ô AdamW cùng nguồn, cùng fold, cùng "
            "checkpoint Pha 1). 4 máy RTX A4000 đã kiểm trùng bit. 6 nguồn dùng CHUNG 5 fold đích nên 30 cặp không độc lập: "
            "khoảng tin cậy là bootstrap theo cụm fold (5 cụm, hẹp hơn thực tế), sàn Wilcoxon theo fold là p 0,0625 (5/5). "
            "Sàn nhiễu chạy lại khác máy 0,010 ROC. Chưa có số nào ở đây đủ để viết như phát hiện vào bài; dùng được cho "
            "bảng ablation kèm ghi chú 'một seed'.")

    sections = []
    coll_pairs = []
    for s_, f_ in (("4cwe_jscpp", 1), ("common_jscpp", 3)):
        a_ = main[f"asamonly|{s_}"]["roc"]["folds"][f_ - 1]
        r_ = main[f"rasam|{s_}"]["roc"]["folds"][f_ - 1]
        coll_pairs.append((s_, f_, a_, r_, r_ - a_))
    # khoảng t (thận trọng) có chứa 0 không, cho các phép so chính
    ct_out = []
    for con, mode, lab in (("ra_with_asam", "excl", "RecAdam khi có ASAM (bỏ ô sập)"),
                           ("ra_without_asam", "all", "RecAdam khi không ASAM"),
                           ("ra_main", "excl", "tác dụng chính (bỏ ô sập)"), ("interaction", "excl", "tương tác (bỏ ô sập)")):
        for g, glab in (("all", "6 nguồn"), ("jsonly", "chỉ JS"), ("jscpp", "JS + C/C++")):
            for k, klab in MK:
                st_ = d(con, g, mode, k)
                c_ = st_.get("ci_t") if st_ else None
                if c_ and (c_[0] > 0 or c_[1] < 0):
                    ct_out.append(f"{lab}, {glab}, {klab} {v(st_['mean'], 3, True)} [{v(c_[0], 4, True)}; {v(c_[1], 4, True)}]")

    # ------------------------------------------------------------------ 1. thiết kế
    rows = []
    for s in SRC6:
        r = [SRC_VI[s]]
        for vv in ("noras", "raonly", "asamonly", "rasam"):
            m = main[f"{vv}|{s}"]
            r.append(f"{v(m['roc']['mean'])} ± {v(m['roc']['sd'])}" + (f" ({m['collapse']} sập)" if m["collapse"] else ""))
        for vv in ("noras", "raonly", "asamonly", "rasam"):
            m = main[f"{vv}|{s}"]
            r.append(f"{v(m['f1_val']['mean'])} ± {v(m['f1_val']['sd'])}")
        rows.append(r)
    for key, lab in (("extra|baseline", "không Pha 1: CodeBERT baseline"), ("extra|nop1_adamw", "không Pha 1: MW + AdamW")):
        if key in main:
            m = main[key]
            rows.append([lab, f"{v(m['roc']['mean'])} ± {v(m['roc']['sd'])}", "", "", "",
                         f"{v(m['f1_val']['mean'])} ± {v(m['f1_val']['sd'])}", "", "", ""])
    sections.append({
        "heading": "1. Thiết kế, dữ liệu và cách đọc",
        "paras": [
            "Bốn biến thể Pha 2 trên cùng checkpoint Pha 1 và cùng fold đích, chỉ khác hai cờ: AdamW (noRAS, Pha 1 rồi fine-tune "
            "thường), chỉ RecAdam, chỉ ASAM, RecAdam + ASAM (cột chính). Cấu hình RecAdam: γ (pretrain_cof) 500, sigmoid k 0,05, "
            "t0 = 1 % tổng bước = bước 4; ASAM ρ 0,5, η 0,01; batch 32, lr 5,66e-5, không warmup, 30 epoch, min_epochs 10, "
            "patience 8, chọn checkpoint theo val ROC. Mã: src_mwonly/mwg/optim.py (RecAdam, SAMStep), src_mwonly/train.py.",
            "Mọi hiệu là GHÉP CẶP theo (nguồn, fold): 'RecAdam khi có ASAM' = cột chính - chỉ ASAM; 'RecAdam khi không ASAM' = "
            "chỉ RecAdam - AdamW; tác dụng chính = trung bình hai hiệu đó; tương tác = hiệu thứ nhất - hiệu thứ hai. Mỗi số đi "
            "kèm khoảng tin cậy 95 % bootstrap theo cụm fold, số cặp dương / số cặp và số fold dương (trung bình qua các nguồn "
            "trong fold). 'Sập' = test ROC < 0,75 (2/120 ô, đều là chỉ ASAM: 4CWE JS + C/C++ f1 0,607 và common JS + C/C++ f3 "
            "0,523); bản 'bỏ ô sập' bỏ cặp có ô sập ở bất kỳ vế nào.",
            "Bảng dưới: trung bình ± độ lệch chuẩn qua 5 fold (ROC-AUC rồi F1 tại ngưỡng val). Bảng đủ 4 chỉ số ở table_main.csv; "
            "bản tiếng Anh cho bài ở table_main.tex / .md. Ba nguồn chỉ C/C++ chỉ có cột chính (ROC 0,921 / 0,917 / 0,923) nên "
            "không vào phép so RecAdam.",
        ],
        "table": {"head": ["nguồn Pha 1", "ROC AdamW", "ROC chỉ RA", "ROC chỉ ASAM", "ROC RA + ASAM", "F1@val AdamW",
                           "F1@val chỉ RA", "F1@val chỉ ASAM", "F1@val RA + ASAM"], "rows": rows},
    })

    # ------------------------------------------------------------------ 2. Q1 chỉ số cuối
    rows = []
    spec = [("ra_with_asam", "all", "all", "RecAdam khi có ASAM, mọi cặp"),
            ("ra_with_asam", "all", "excl", "RecAdam khi có ASAM, bỏ cặp có ô sập"),
            ("ra_with_asam", "all", "me30", "RecAdam khi có ASAM, ô sập thay bằng bản me30"),
            ("ra_with_asam", "jsonly", "all", "RecAdam khi có ASAM, nguồn chỉ JS"),
            ("ra_with_asam", "jscpp", "excl", "RecAdam khi có ASAM, JS + C/C++, bỏ ô sập"),
            ("ra_without_asam", "all", "all", "RecAdam khi không ASAM, 6 nguồn"),
            ("ra_without_asam", "jsonly", "all", "RecAdam khi không ASAM, nguồn chỉ JS"),
            ("ra_without_asam", "jscpp", "all", "RecAdam khi không ASAM, JS + C/C++"),
            ("ra_main", "all", "all", "tác dụng chính, mọi cặp"),
            ("ra_main", "all", "excl", "tác dụng chính, bỏ ô sập"),
            ("interaction", "all", "all", "tương tác RA × ASAM, mọi cặp"),
            ("interaction", "all", "excl", "tương tác RA × ASAM, bỏ ô sập")]
    for con, g, mode, lab in spec:
        if d(con, g, mode, "roc") is None:
            continue
        if mode == "me30" and all(abs(d(con, g, "me30", k)["mean"] - d(con, g, "all", k)["mean"]) < 1e-12 for k, _ in MK):
            continue
        rows.append([lab] + [cell(d(con, g, mode, k)) for k, _ in MK])
    per_src = []
    for s in SRC6:
        a, b = d("ra_with_asam", s, "all", "roc"), d("ra_without_asam", s, "all", "roc")
        per_src.append(f"{SRC_VI[s]}: có ASAM {short(a)}, không ASAM {short(b)}")
    sections.append({
        "heading": "2. Câu 1 - tác dụng của RecAdam lên chỉ số cuối (ghép cặp nguồn, fold)",
        "paras": [
            f"Khi có ASAM, tính đủ 30 cặp thì RecAdam trông như có ích: ROC {full(ra_w_all)}, F1@val "
            f"{full(d('ra_with_asam', 'all', 'all', 'f1_val'))}. Nhưng trung vị chỉ {v(ra_w_all['median'], 4, True)} và toàn bộ phần "
            f"hơn đến từ 2 cặp mà ô chỉ ASAM sập (hiệu {v(coll_pairs[0][4], 3, True)} và {v(coll_pairs[1][4], 3, True)} ROC). "
            f"Bỏ 2 cặp đó: ROC {full(ra_w_ex)}, F1@0,5 "
            f"{full(d('ra_with_asam', 'all', 'excl', 'f1_05'))}, F1@val {full(d('ra_with_asam', 'all', 'excl', 'f1_val'))}, PR "
            f"{full(d('ra_with_asam', 'all', 'excl', 'pr'))}. Trên nguồn chỉ JS (không có ô sập): ROC {full(ra_w_js)}.",
            f"Khi không có ASAM, RecAdam hơi KÉM fine-tune thường: ROC {full(ra_o_all)}, PR "
            f"{full(d('ra_without_asam', 'all', 'all', 'pr'))}, F1@0,5 {full(d('ra_without_asam', 'all', 'all', 'f1_05'))}. Phần âm "
            f"nằm ở nguồn JS + C/C++: ROC {full(ra_o_cpp)} (0/3 nguồn dương; 4CWE JS + C/C++ -0,017, 0/5 fold), trong khi nguồn "
            f"chỉ JS ≈ 0 ({full(ra_o_js)}). Độ lớn vẫn dưới hoặc ngang sàn nhiễu 0,010.",
            f"Tác dụng chính (trung bình hai hiệu đơn, đúng nghĩa 'main effect' của thiết kế giai thừa) là "
            f"{full(d('ra_main', 'all', 'all', 'roc'))} khi tính đủ và {full(d('ra_main', 'all', 'excl', 'roc'))} khi bỏ ô sập; "
            f"riêng JS + C/C++ bỏ ô sập là {full(d('ra_main', 'jscpp', 'excl', 'roc'))}. Tức là ngoài việc cứu 2 ô sập, RecAdam "
            f"không thêm gì vào độ chính xác và hơi bớt đi ở nguồn có C/C++.",
            "Khoảng tin cậy t trên 5 trung bình fold (bản thận trọng, df = 4; cột ci_t trong table_recadam_delta.csv) của mọi phép "
            "so gộp 6 nguồn (bỏ ô sập) đều chứa 0, ở cả bốn chỉ số. "
            + (("Các khoảng t KHÔNG chứa 0 chỉ có: " + "; ".join(ct_out) + " - đều là hiệu ÂM nhỏ của RecAdam ở nhóm nguồn JS + "
                "C/C++, dưới sàn nhiễu 0,010.") if ct_out else "Không khoảng t nào (theo nhóm nguồn) loại được 0."),
            me30_sent,
        ],
        "bullets": per_src,
        "table": {"head": ["phép so (RecAdam bật - tắt)", "F1@0,5", "F1@val", "ROC-AUC", "PR-AUC"], "rows": rows},
        "figure": {"file": "fig_recadam_forest.svg",
                   "caption": "Hiệu ghép cặp RecAdam bật - tắt theo nguồn Pha 1 và gộp, cho 4 chỉ số. Chấm đặc: khi có ASAM "
                              "(RecAdam + ASAM - chỉ ASAM); chấm rỗng, nét đứt: khi không ASAM (chỉ RecAdam - AdamW); thoi xám: "
                              "khi có ASAM nhưng bỏ cặp có ô sập. Thanh ngang: khoảng tin cậy 95 % bootstrap theo cụm fold "
                              "(hàng một nguồn: n = 5, khoảng rất thô); mũi tên: khoảng vượt khung. Dải xám: sàn nhiễu chạy lại "
                              "±0,010.",
                   "caption_en": "Paired effect of RecAdam (on minus off) per Phase-1 source and pooled, for four test metrics. "
                                 "Filled markers: with ASAM (RecAdam+ASAM minus ASAM only); open markers, dashed: without ASAM "
                                 "(RecAdam minus AdamW); grey diamonds: with ASAM after removing pairs that contain a collapsed run. "
                                 "Bars: 95% fold-cluster bootstrap CIs (per-source rows use n = 5 folds); arrows: CI beyond the "
                                 "axis. Grey band: rerun noise floor (±0.010). Single seed."},
    })

    # ------------------------------------------------------------------ 3. tương tác theo nhóm nguồn
    im = num.get("interaction_means", {})
    rows = []
    for vv in ("noras", "raonly", "asamonly", "rasam"):
        r = [VAR_VI[vv]]
        for k in ("roc", "f1_val"):
            for fm in ("jsonly", "jscpp"):
                e = im.get(f"{k}|{fm}|{vv}")
                r.append(f"{v(e['mean'])} [{v(e['ci'][0])}; {v(e['ci'][1])}]" if e else "-")
        rows.append(r)
    e1, e2 = im.get("roc|jscpp|asamonly_excl"), im.get("f1_val|jscpp|asamonly_excl")
    rows.append(["chỉ ASAM, bỏ 2 ô sập", "-", f"{v(e1['mean'])} [{v(e1['ci'][0])}; {v(e1['ci'][1])}]" if e1 else "-", "-",
                 f"{v(e2['mean'])} [{v(e2['ci'][0])}; {v(e2['ci'][1])}]" if e2 else "-"])
    fi_o, fi_w = fam["ra_without_asam|all|roc"], fam["ra_with_asam|excl|roc"]
    sections.append({
        "heading": "3. Tương tác RecAdam × ASAM và theo nhóm nguồn",
        "paras": [
            f"Tương tác tính đủ cặp là dương ở cả bốn chỉ số (ROC {full(inter_all)}), nhưng như mục 2, nó gần như trọn vẹn là 2 ô "
            f"chỉ ASAM sập; bỏ chúng còn {full(inter_ex)}. Ở nguồn chỉ JS không có ô sập và tương tác ROC "
            f"{full(d('interaction', 'jsonly', 'all', 'roc'))}: RecAdam thêm vào ASAM hay thêm vào AdamW đều ≈ 0.",
            f"Phần còn lại sau khi bỏ ô sập là một mẫu hình theo NHÓM NGUỒN, không theo ASAM: khi không ASAM, hiệu của RecAdam "
            f"trên nguồn chỉ JS trừ hiệu trên nguồn JS + C/C++ (cùng fold) là {v(fi_o['mean'], 3, True)} ROC, dương "
            f"{fi_o['fold_pos']}/{fi_o['fold_n']} fold; khi có ASAM (bỏ ô sập) là {v(fi_w['mean'], 3, True)} ({fi_w['fold_pos']}/"
            f"{fi_w['fold_n']} fold). Tức là RecAdam một mình làm hại nhẹ đúng ở các checkpoint Pha 1 học nhiều C/C++, còn ASAM "
            f"che mất điều đó. Đây là phép so THĂM DÒ (chọn sau khi xem dữ liệu, 5/5 là sàn p 0,0625), không phải phát hiện.",
            f"Đóng góp của cột chính so với chuyển giao thường (AdamW) tách theo thiết kế 2×2: nguồn chỉ JS cột chính - AdamW "
            f"{full(full_js)}, trong đó ASAM khi không RecAdam {full(asam_js)} và RecAdam khi có ASAM {full(ra_w_js)}; nguồn JS + "
            f"C/C++ cột chính - AdamW chỉ {full(full_cpp)}. Phần ăn được là của ASAM, và trên JS + C/C++ cả gói không hơn AdamW.",
        ],
        "table": {"head": ["biến thể", "ROC chỉ JS", "ROC JS + C/C++", "F1@val chỉ JS", "F1@val JS + C/C++"], "rows": rows},
        "figure": {"file": "fig_recadam_interaction.svg",
                   "caption": "Biểu đồ tương tác 2×2: trung bình 15 ô (3 nguồn × 5 fold) của mỗi biến thể, theo nhóm nguồn, cho "
                              "ROC-AUC và F1@val. Màu: RecAdam tắt (cam) / bật (xanh); đặc, nét liền: ASAM bật; rỗng, nét đứt: "
                              "ASAM tắt; thoi xám: chỉ ASAM sau khi bỏ 2 ô sập. Thanh dọc: khoảng tin cậy 95 % bootstrap theo cụm "
                              "fold. Đường cắt nhau ở JS + C/C++ là do 2 ô sập kéo trung bình chỉ ASAM xuống.",
                   "caption_en": "Interaction plot of the 2x2 ablation: mean test score over 15 runs (3 sources x 5 folds) per "
                                 "variant and source family. Colour: RecAdam off (orange) / on (blue); filled, solid: ASAM on; "
                                 "open, dashed: ASAM off; grey diamond: ASAM only without its two collapsed runs. Bars: 95% "
                                 "fold-cluster bootstrap CIs. The crossing on JS+C/C++ is driven by the two collapsed ASAM-only "
                                 "runs."},
    })

    # ------------------------------------------------------------------ 4a. ổn định: sập
    rows = []
    for vv in ("noras", "raonly", "asamonly", "rasam"):
        st = S[vv]
        rows.append([VAR_VI[vv], str(st["n"]), str(st["collapse"]), str(st["stuck10"]),
                     f"{st['plateau_median']:.0f} ({st['plateau_min']}-{st['plateau_max']})", v(st["sd_roc_mean"]),
                     v(st["sd_roc_adj_mean"]), v(st["sd_f1val_mean"]), v(st["worst_roc_mean"]), v(st["min_roc"])])
    sections.append({
        "heading": "4. Câu 2 - độ ổn định (a): sập, kẹt, khoảng min-max",
        "paras": [
            f"Đóng góp nhìn thấy duy nhất của RecAdam là ở đuôi: RecAdam + ASAM {coll_r}/30 ô sập, chỉ ASAM {coll_a}/30 (cộng ô "
            f"không Pha 1: chỉ ASAM 1/5). Agent ASAM (asam_collapse.md mục 2) đã tính trên toàn khối: 0/45 so với 3/35, Fisher p ≈ "
            f"0,08; trên 30 cặp ghép được 0 so với 2, McNemar p 0,5. Số ô kẹt ở ln2 > 10 epoch thì không khác: {st_r}/30 so với "
            f"{st_a}/30, và bình nguyên của cột chính ngắn hơn chỉ ASAM ở 8 cặp, dài hơn ở 9, bằng ở 13 (asam_collapse.md). "
            f"Tức là RecAdam không làm ô bớt kẹt ở mức chung; khác biệt chỉ nằm ở 2 ô kẹt lâu nhất của chỉ ASAM (xem phép chạy "
            f"lại me30 dưới đây).",
            f"Vì vậy điểm thấp nhất mỗi nguồn ('fold tệ nhất') của cột chính cao hơn chỉ ASAM ở 5/6 nguồn (TB "
            f"{v(sdp['with_asam|worst_roc_by_src']['mean'], 3, True)} ROC), nhưng trên nguồn chỉ JS chỉ "
            f"{v(sdp['with_asam|worst_roc_by_src']['mean_jsonly'], 3, True)}; khi không ASAM thì fold tệ nhất tăng 3/6, giảm 3/6 "
            f"(TB {v(sdp['without_asam|worst_roc_by_src']['mean'], 3, True)}).",
            "Ba ô sập được chạy lại với --min_epochs 30 (cùng mọi cờ, --epochs vẫn 30 nên lịch LR không đổi; bậc 1, n = 1 mỗi "
            "ô) để biết bình nguyên là tạm thời bị luật dừng cắt hay kẹt vĩnh viễn. Ô nào thoát được thì 'RecAdam chống sập' ở ô "
            "đó thu về 'RecAdam làm bình nguyên ngắn hơn ngưỡng luật dừng' (tính chất của cặp RecAdam và luật dừng); ô nào vẫn kẹt "
            "thì khác biệt là của tối ưu. Trạng thái lúc sinh báo cáo: " + "; ".join(me30_lines) + ".",
        ] + ([me30_interp] if me30_interp else []),
        "table": {"head": ["biến thể", "ô", "sập", "kẹt > 10 epoch", "bình nguyên TV (min-max)", "SD ROC theo fold (TB 6 nguồn)",
                           "SD ROC đã trừ hiệu fold", "SD F1@val theo fold", "ROC fold tệ nhất (TB)", "ROC thấp nhất"],
                  "rows": rows},
        "figure": {"file": "fig_recadam_paired_folds.svg",
                   "caption": "Từng cặp (nguồn, fold) trước và sau khi bật RecAdam, khi không có ASAM (AdamW → chỉ RecAdam) và khi "
                              "có ASAM (chỉ ASAM → RecAdam + ASAM), cho ROC-AUC và F1@val. Đường xanh nhạt: RecAdam cao hơn; xám: "
                              "thấp hơn; đỏ: cặp có ô sập (giá trị thật ghi ở mép dưới). Tròn: nguồn chỉ JS; tam giác: JS + C/C++. "
                              "Ngoài 2 cặp đỏ, số cặp lên và xuống gần bằng nhau ở cả bốn ô.",
                   "caption_en": "Each (source, fold) pair before and after enabling RecAdam, without ASAM (AdamW to RecAdam) "
                                 "and with ASAM (ASAM only to RecAdam+ASAM), for ROC-AUC and F1 at the validation threshold. Light "
                                 "blue: RecAdam higher; grey: lower; red: pair containing a collapsed run (true values printed at "
                                 "the bottom). Circles: JS-only sources; triangles: JS+C/C++ sources. Apart from the two red pairs, "
                                 "ups and downs are balanced in all four panels."},
    })

    # ------------------------------------------------------------------ 4b. ổn định: độ tản theo fold
    rows = []
    for s in SRC6:
        rows.append([SRC_VI[s]] + [v(S[vv]["sd_roc_by_src"][s]) for vv in ("noras", "raonly", "asamonly", "rasam")] +
                    [v(S[vv]["sd_f1val_by_src"][s]) for vv in ("noras", "raonly", "asamonly", "rasam")])
    w, o, wx = sdp["with_asam|sd_roc_by_src"], sdp["without_asam|sd_roc_by_src"], sdp["with_asam|sd_roc_by_src_excl"]
    wa, oa = sdp["with_asam|sd_roc_adj_by_src"], sdp["without_asam|sd_roc_adj_by_src"]
    sections.append({
        "heading": "4. Câu 2 - độ ổn định (b): RecAdam có làm kết quả ít dao động theo fold hơn không",
        "paras": [
            f"Ghép theo nguồn (6 cặp độ lệch chuẩn qua 5 fold): khi có ASAM, RecAdam làm SD ROC nhỏ hơn ở {w['n_lower']}/6 nguồn "
            f"(TB {v(w['mean'], 3, True)}), nhưng phần lớn là 2 nguồn có ô sập ({v(w['by_src']['4cwe_jscpp'], 3, True)} và "
            f"{v(w['by_src']['common_jscpp'], 3, True)}); tính lại trên các fold không sập thì còn {wx['n_lower']}/6 nguồn nhỏ hơn, "
            f"TB {v(wx['mean'], 3, True)}. Trên nguồn chỉ JS, SD thô của cột chính nhỏ hơn chỉ ASAM ở 3/3 nguồn "
            f"(TB {v(w['mean_jsonly'], 3, True)}), nhưng sau khi trừ hiệu fold (trừ trung vị 24 ô 2×2 của fold) thì "
            f"{v(wa['mean_jsonly'], 3, True)}: cột chính chỉ ít bám độ khó chung của fold hơn, không ít nhiễu hơn.",
            f"Khi không ASAM: SD ROC nhỏ hơn ở {o['n_lower']}/6 nguồn, TB {v(o['mean'], 3, True)}; đã trừ hiệu fold thì "
            f"{oa['n_lower']}/6, TB {v(oa['mean'], 3, True)}. SD của 5 giá trị có sai số tương đối khoảng 35 %, nên các hiệu "
            f"cỡ 0,002-0,008 này không phân biệt được với 0.",
            f"Độ nhạy theo NGUỒN trong cùng fold (SD ROC giữa 3 nguồn cùng họ): khi không ASAM RecAdam nhỏ hơn ở "
            f"{sdp['without_asam|src_sensitivity']['n_lower']}/10 (họ × fold), TB {v(sdp['without_asam|src_sensitivity']['mean'], 4, True)}; "
            f"khi có ASAM {sdp['with_asam|src_sensitivity']['n_lower']}/10 nhưng do ô sập.",
            "Kết luận câu 2: RecAdam không làm kết quả ít dao động hơn một cách đo được; thứ duy nhất khác là không có ô sập khi "
            "đi cùng ASAM (một seed). Văn liệu cũng không ủng hộ 'RecAdam giảm phương sai': chạy lại độc lập trên RoBERTa-large "
            "(Zheng et al., ACL 2023, Bảng 1) cho RecAdam std CAO hơn fine-tune thường ở 6/7 tập; chỉ L2-SP (cùng họ phạt bậc "
            "hai về neo) có bằng chứng giảm std qua 25 seed (Hua et al., TNNLS).",
        ],
        "table": {"head": ["nguồn", "SD ROC AdamW", "SD ROC chỉ RA", "SD ROC chỉ ASAM", "SD ROC RA + ASAM", "SD F1@val AdamW",
                           "SD F1@val chỉ RA", "SD F1@val chỉ ASAM", "SD F1@val RA + ASAM"], "rows": rows},
        "figure": {"file": "fig_recadam_fold_sd.svg",
                   "caption": "Độ lệch chuẩn qua 5 fold của mỗi nguồn khi tắt (trục ngang) và bật (trục dọc) RecAdam, thang log. "
                              "Đặc: khi có ASAM; rỗng: khi không ASAM; tròn: nguồn chỉ JS; tam giác: JS + C/C++. Điểm dưới đường "
                              "chéo: RecAdam ít dao động hơn. Hai tam giác đặc xa bên phải là hai nguồn có ô chỉ ASAM sập.",
                   "caption_en": "Standard deviation over the five folds for each Phase-1 source with RecAdam off (x) and on "
                                 "(y), log scale. Filled: with ASAM; open: without ASAM; circles: JS-only sources; triangles: "
                                 "JS+C/C++ sources. Points below the diagonal: RecAdam less variable. The two filled triangles on "
                                 "the far right are the sources with a collapsed ASAM-only run."},
    })

    # ------------------------------------------------------------------ 5. Q3 nhớ nguồn
    def ix(key, con):
        e = ind.get(f"{key}|{con}")
        if not e:
            return "-"
        return f"{v(e['mean'], 3, True)} [{v(e['ci_boot'][0], 3, True)}; {v(e['ci_boot'][1], 3, True)}] {e['pos']}/{e['n']}; fold {e['fold_pos']}/{e['fold_n']}"
    rows = [
        ["Spearman với mô hình KHÔNG Pha 1 (MW + AdamW)", ix("rho_nop1", "ra_without_asam"), ix("rho_nop1", "ra_with_asam"),
         "thấp hơn = giữ dấu Pha 1 nhiều hơn"],
        ["Spearman giữa hai nguồn Pha 1 khác nhau (15 cặp nguồn × 5 fold)", ix("inter_src", "ra_without_asam"),
         ix("inter_src", "ra_with_asam"), "thấp hơn = dự đoán phụ thuộc nguồn nhiều hơn (hoặc nhiễu hơn)"],
        ["ROC trên CWE-022 + CWE-079 (CWE cần chuyển giao)", ix("cwe_hard", "ra_without_asam"), ix("cwe_hard", "ra_with_asam"),
         "cao hơn = tri thức nguồn giúp nhiều hơn"],
        ["ROC trên CWE-078 + CWE-089", cell(d("ra_without_asam", "all", "all", "cwe_easy")),
         cell(d("ra_with_asam", "all", "excl", "cwe_easy")), "đối chứng: CWE mô hình đã làm tốt"],
        ["ECE (10 khoảng)", ix("ece", "ra_without_asam"), ix("ece", "ra_with_asam"), "thấp hơn = hiệu chỉnh tốt hơn"],
        ["Brier", cell(d("ra_without_asam", "all", "all", "brier")), cell(d("ra_with_asam", "all", "excl", "brier")), ""],
        ["val ROC (epoch chọn) - test ROC", cell(d("ra_without_asam", "all", "all", "val_test_gap")),
         cell(d("ra_with_asam", "all", "excl", "val_test_gap")), "chênh val-test"],
    ]
    hj_o, hj_w = d("ra_without_asam", "jsonly", "all", "cwe_hard"), d("ra_with_asam", "jsonly", "all", "cwe_hard")
    c22 = d("ra_without_asam", "jsonly", "all", "cwe022")
    fh_o, fh_w = fam["ra_without_asam|all|cwe_hard"], fam["ra_with_asam|excl|cwe_hard"]
    seed_txt = (f"Để đặt độ lớn: đổi seed cả hai pha (khối cũ, chỉ ASAM, common chỉ JS, {seed['n']} cặp seed) đổi "
                f"{v(100 * seed['flip_rate_mean'], 1)} % quyết định, Spearman trung vị {v(seed['spearman_median'])}, |Δ ROC| TB "
                f"{v(seed['abs_droc_mean'])}. RecAdam (cùng seed) đổi {v(100 * agree['without_asam']['flip_rate_mean'], 1)} % "
                f"quyết định khi không ASAM và {v(100 * agree['with_asam']['flip_rate_mean'], 1)} % khi có ASAM, |Δ ROC| TB "
                f"{v(agree['without_asam']['abs_droc_mean'])} / {v(agree['with_asam']['abs_droc_mean'])}: RecAdam đổi dự đoán "
                f"cỡ một lần đổi seed, nhưng không theo hướng nào nhất quán.") if seed else ""
    sections.append({
        "heading": "5. Câu 3 - 'nhớ nguồn' (chống quên): dữ liệu đo được gì và không đo được gì",
        "paras": [
            "Nói thẳng: khối này KHÔNG đo được việc quên. Mục đích gốc của RecAdam là giữ năng lực trên tác vụ nguồn (Chen et al., "
            "2020), mà muốn đo thì phải chấm mô hình SAU Pha 2 trên dữ liệu nguồn (val JS / C/C++), hoặc đo khoảng cách tham số "
            "tới checkpoint Pha 1; checkpoint Pha 2 đã xoá và không có lượt chấm nào như vậy. Mọi thứ dưới đây là chỉ báo GIÁN "
            "TIẾP từ xác suất test (probs.npz) và theo CWE (nhãn CWE từ test.jsonl, thứ tự khớp nhãn 152/152).",
            f"(a) Giống mô hình không Pha 1: khi không ASAM, dự đoán của chỉ RecAdam xa mô hình không Pha 1 hơn một chút so với "
            f"AdamW ({ix('rho_nop1', 'ra_without_asam')}), đúng hướng 'giữ dấu Pha 1' nhưng nhỏ; khi có ASAM thì không có "
            f"({ix('rho_nop1', 'ra_with_asam')}, đã bỏ ô sập).",
            f"(b) Dự đoán giữa các nguồn: khi không ASAM, RecAdam làm dự đoán của hai nguồn Pha 1 khác nhau (cùng fold) KHÁC nhau "
            f"hơn: {ix('inter_src', 'ra_without_asam')} (0/5 fold dương). Đây là tín hiệu nhất quán nhất cho 'giữ dấu nguồn', "
            f"nhưng cũng khớp với giải thích thứ hai: RecAdam chỉ thêm nhiễu riêng mỗi lần chạy (nhiễu nào cũng làm tương quan "
            f"giữa hai lần chạy giảm). Hai giải thích không tách được nếu không có dự đoán của chính checkpoint Pha 1 trên SVEN "
            f"(đề xuất 2).",
            f"(c) Theo CWE: trên nguồn chỉ JS, RecAdam nâng ROC trên CWE-022 + CWE-079 ở cả hai bối cảnh (không ASAM "
            f"{full(hj_o)}; có ASAM {full(hj_w)}; riêng CWE-022 khi không ASAM {full(c22)}), còn trên JS + C/C++ thì ≈ 0 hoặc âm. "
            f"Hiệu theo họ nguồn của RecAdam trên CWE-022 + 079: {v(fh_o['mean'], 3, True)} ({fh_o['fold_pos']}/5 fold) khi không "
            f"ASAM, {v(fh_w['mean'], 3, True)} ({fh_w['fold_pos']}/5 fold) khi có ASAM. Khớp giả thuyết 'RecAdam giữ tri thức "
            f"web (XSS, path traversal) học từ JS', nhưng mỗi tập CWE chỉ 8-20 hàm mỗi fold (ROC rất nhiễu) và đây là một trong "
            f"nhiều phép so thăm dò đã nhìn, chưa hiệu chỉnh so sánh bội.",
            f"(d) Hiệu chỉnh xác suất: RecAdam không làm tốt hơn; khi không ASAM còn hơi tệ hơn (ECE {ix('ece', 'ra_without_asam')}, "
            f"Brier {cell(d('ra_without_asam', 'all', 'all', 'brier'))}), tập trung ở JS + C/C++. Thứ cải thiện hiệu chỉnh là "
            f"ASAM: ECE trung bình AdamW {v(pv['noras']['ece']['mean'])}, chỉ RecAdam {v(pv['raonly']['ece']['mean'])}, chỉ ASAM "
            f"{v(pv['asamonly']['ece']['mean'])}, RecAdam + ASAM {v(pv['rasam']['ece']['mean'])}; mọi biến thể đều quá tự tin "
            f"(|p - 0,5| × 2 trung bình 0,81-0,93), khớp Zhou et al. (ICSE 2024) về mô hình mã.",
            f"(e) Chênh val - test: không khác (khi không ASAM {short(d('ra_without_asam', 'all', 'all', 'val_test_gap'))}, khi có "
            f"ASAM {short(d('ra_with_asam', 'all', 'excl', 'val_test_gap'))}), nên RecAdam không làm chọn checkpoint lạc quan hơn "
            f"hay bớt lạc quan. {seed_txt}",
        ],
        "table": {"head": ["đại lượng (RecAdam bật - tắt)", "khi không ASAM", "khi có ASAM (bỏ ô sập)", "đọc thế nào"],
                  "rows": rows},
        "figure": {"file": "fig_recadam_indirect.svg",
                   "caption": "Chỉ báo gián tiếp cho 'nhớ nguồn' và hiệu chỉnh, hiệu ghép cặp RecAdam bật - tắt: tương quan "
                              "Spearman với mô hình không Pha 1, tương quan giữa hai nguồn Pha 1 trong cùng fold, ROC trên "
                              "CWE-022 + CWE-079, và ECE. Rỗng: khi không ASAM; đặc: khi có ASAM; đỏ: cặp có ô sập (phần lớn nằm "
                              "ngoài khung). Gạch đen: trung bình và khoảng tin cậy 95 % bootstrap theo cụm fold, đã bỏ cặp có ô "
                              "sập.",
                   "caption_en": "Indirect indicators of source retention and calibration (paired difference RecAdam on minus "
                                 "off): Spearman correlation with the no-Phase-1 model, correlation between two different Phase-1 "
                                 "sources in the same fold, ROC-AUC on CWE-022 and CWE-079, and expected calibration error. Open: "
                                 "without ASAM; filled: with ASAM; red: pairs containing a collapsed run (mostly off-scale). Black: "
                                 "mean and 95% fold-cluster bootstrap CI excluding collapsed pairs. None of these measures "
                                 "forgetting of the source task directly."},
    })

    # ------------------------------------------------------------------ 6. Q4 động học
    pe = sch["per_epoch"]
    rows = []
    for k, lab, nd in (("vr1", "val ROC epoch 1", 3), ("vr_mean_1_10", "val ROC trung bình epoch 1-10", 3),
                       ("ep_to_val85", "số epoch tới val ROC ≥ 0,85", 2), ("ep_to_val80", "số epoch tới val ROC ≥ 0,80", 2),
                       ("tl1", "train loss epoch 1", 3), ("tl2", "train loss epoch 2", 3),
                       ("best_epoch", "epoch được chọn", 2), ("stop_epoch", "epoch dừng", 2)):
        rows.append([lab, cell(d("ra_without_asam", "all", "all", k), nd), cell(d("ra_with_asam", "all", "excl", k), nd),
                     cell(d("ra_with_asam", "jsonly", "all", k), nd)])
    sections.append({
        "heading": "6. Câu 4 - động học: RecAdam là một warmup ngầm làm chậm khoảng 1 epoch",
        "paras": [
            f"Lịch thật của khối (456 hàm train, batch 32 ⇒ {sch['steps_per_epoch']} bước / epoch, {sch['total']} bước): λ(t) = "
            f"sigmoid(k (t - t0)) với k {v(sch['k'], 2)}, t0 = {sch['t0']}; λ nhân thẳng vào bước Adam và phần 1 - λ kéo về neo "
            f"với hệ số lr(t)·(1 - λ)·γ, γ {sch['cof']:.0f}. λ ở bước 1 = {v(sch['lam_step1'], 2)}; λ ≥ 0,9 từ bước "
            f"{sch['step_lam90']} (epoch 4); λ ≥ 0,99 từ bước {sch['step_lam99']} (epoch 7). λ trung bình epoch 1-4: "
            f"{', '.join(v(e['lambda_mean'], 2) for e in pe[:4])}. Tổng lr hiệu dụng epoch 1-3: AdamW "
            f"{v(sch['cum_eff_lr_ep1_3_adamw'] * 1e3, 2)}e-3, RecAdam {v(sch['cum_eff_lr_ep1_3_recadam'] * 1e3, 2)}e-3 "
            f"({v(100 * (sch['cum_eff_lr_ep1_3_recadam'] / sch['cum_eff_lr_ep1_3_adamw'] - 1), 0, True)} %). Kéo về neo mỗi bước: "
            f"{', '.join(v(100 * e['pull_mean'], 2) + ' %' for e in pe[:3])} ở epoch 1-3; tích luỹ {v(sch['cum_pull'], 2)}. Từ "
            f"epoch 7 RecAdam gần như là AdamW (khác duy nhất: eps 1e-6 thay vì 1e-8). Neo gồm cả head mới khởi tạo ngẫu nhiên "
            f"(mã gốc của Chen et al. không neo head).",
            f"Khi không ASAM, dấu vết warmup đo được rất rõ và nhất quán: val ROC epoch 1 {full(vr1_o)}; val ROC trung bình epoch "
            f"1-10 {full(vrm_o)}; số epoch tới val ROC ≥ 0,85 {full(ep85_o, 2)} (0 cặp nhanh hơn, 6/6 nguồn); train loss epoch 2 "
            f"{full(tl2_o)}. Đây là hiệu ứng mạnh nhất và chắc nhất của RecAdam trong khối, nhưng nó không biến thành độ chính "
            f"xác cuối: tới epoch được chọn (muộn hơn {v(d('ra_without_asam', 'all', 'all', 'best_epoch')['mean'], 1)} epoch) hai "
            f"nhánh đã ngang nhau.",
            f"Khi có ASAM thì khác: val ROC epoch 1 hơi CAO hơn ({full(d('ra_with_asam', 'all', 'excl', 'vr1'))}), có lẽ vì bước "
            f"nhỏ hơn làm nhẹ cú xáo của ASAM ở bước đầu, nhưng trung bình epoch 1-10 lại thấp hơn "
            f"({full(d('ra_with_asam', 'all', 'excl', 'vr_mean_1_10'))}) và bình nguyên không ngắn đi (asam_collapse.md mục 1). "
            f"Trên nguồn chỉ JS, cột chính tới val ROC 0,85 chậm hơn chỉ ASAM {full(d('ra_with_asam', 'jsonly', 'all', 'ep_to_val85'), 2)} "
            f"và chọn epoch muộn hơn {full(d('ra_with_asam', 'jsonly', 'all', 'best_epoch'), 2)}.",
        ],
        "table": {"head": ["đại lượng (RecAdam bật - tắt)", "khi không ASAM (30 cặp)", "khi có ASAM, bỏ ô sập (28 cặp)",
                           "khi có ASAM, nguồn chỉ JS (15 cặp)"], "rows": rows},
        "figure": {"file": "fig_recadam_dynamics.svg",
                   "caption": "Động học Pha 2. Trái và giữa: trung vị (đường) và khoảng tứ phân vị (dải) của val ROC-AUC (trên) "
                              "và train loss (dưới) theo epoch cho 4 biến thể, tách nguồn chỉ JS và JS + C/C++; vẽ tới epoch 17, "
                              "trước đó không ô nào dừng (min_epochs 10 + patience 8). Phải trên: λ(t), bước Adam tương đối của "
                              "AdamW (lr(t)/lr0) và RecAdam (λ·lr(t)/lr0); phải dưới: hệ số kéo về neo mỗi bước (thang log).",
                   "caption_en": "Phase-2 training dynamics. Left and middle: median (line) and inter-quartile range (band) of "
                                 "validation ROC-AUC (top) and training loss (bottom) per epoch for the four variants, by source "
                                 "family; shown up to epoch 17, before which no run can stop (min. 10 epochs + patience 8). Top "
                                 "right: RecAdam's annealing coefficient lambda(t) and the relative Adam step of AdamW (lr(t)/lr0) "
                                 "and RecAdam (lambda*lr(t)/lr0); bottom right: per-step pull towards the anchor (log scale)."},
    })

    # ------------------------------------------------------------------ 7. Q5 kết luận cho bài
    sections.append({
        "heading": "7. Câu 5 - RecAdam đóng góp gì cho bài, và nên trình bày thế nào",
        "paras": [
            "Khuyến nghị: KHÔNG trình bày RecAdam là thành phần đóng góp độ chính xác hay 'chống quên'. Trình bày nó là một thành "
            "phần của công thức tối ưu Pha 2 kèm bảng ablation 2×2 (table_main + table_recadam_delta), nói rõ phần ăn là ASAM, "
            "RecAdam trung tính về độ chính xác, và quan sát ổn định là một seed. Nếu bài cần gọn và không muốn bảo vệ RecAdam "
            "trước phản biện, bỏ RecAdam và sửa bình nguyên của ASAM bằng cách có cơ sở hơn (ASAM muộn: K15 xoá bình nguyên 15/15; "
            "hoặc ρ nhỏ hơn), như asam_collapse.md đề xuất.",
        ],
        "bullets": [
            f"ĐỨNG (trung bình, một seed): RecAdam không đổi độ chính xác cuối ngoài sàn nhiễu - bỏ ô sập, gộp 6 nguồn, cả bốn chỉ "
            f"số |Δ| ≤ 0,010 và mọi khoảng tin cậy chứa 0 (ROC khi có ASAM {short(ra_w_ex)}, khi không ASAM {short(ra_o_all)}).",
            f"ĐỨNG (cao): RecAdam với lịch này là warmup ngầm làm chậm việc lên val ROC khoảng 1 epoch khi không ASAM "
            f"({ep85_o['pos']}/{ep85_o['n']} cặp chậm hơn, 0 nhanh hơn, {ep85_o['fold_pos']}/5 fold, 6/6 nguồn).",
            f"ĐỨNG (trung bình): trên nguồn JS + C/C++, chỉ RecAdam kém AdamW một chút ({short(ra_o_cpp)}, 0/3 nguồn dương; ROC "
            f"-0,017 0/5 fold ở 4CWE); dưới sàn nhiễu nên chỉ nên ghi như mô tả.",
            f"QUAN SÁT, CHƯA ĐỨNG: RecAdam + ASAM {coll_r}/30 ô sập so với chỉ ASAM {coll_a}/30 (toàn khối 0/45 so với 3/35, p ≈ "
            f"0,08). Không kèm giảm số ô kẹt ({st_r} so với {st_a}) hay giảm độ tản ngoài ô sập. "
            + (f"Chạy lại min_epochs 30 cho thấy {len(perm)} ô chỉ ASAM kẹt vĩnh viễn trong khi RecAdam + ASAM cùng ô thoát "
               f"(không phải do luật dừng), {len(temp)} ô 2×2 thoát khi cho chạy đủ"
               + (f", còn {len(pending)} ô chờ" if pending else "") + "; vẫn là một seed, cần lặp seed (đề xuất 4)."
               if (perm or temp) else "Có thể là tương tác với luật dừng sớm (chờ me30) hoặc may rủi của một seed."),
            "GIẢ THUYẾT (thăm dò): RecAdam giữ dấu nguồn - dự đoán giữa các nguồn khác nhau hơn (0/5 fold dương), lợi trên "
            "CWE-022 + 079 với nguồn chỉ JS, hại nhẹ với nguồn nhiều C/C++. Chưa tách được khỏi 'RecAdam thêm nhiễu'.",
            "KHÔNG ĐƯỢC VIẾT: 'RecAdam chống quên' (không đo quên), 'RecAdam cải thiện F1/ROC' (hiệu trung bình +0,025 ROC là 2 ô "
            "sập), 'RecAdam giảm phương sai' (không đo được ngoài ô sập; văn liệu chạy lại độc lập còn thấy ngược lại).",
            "Câu gợi ý cho bài (tiếng Anh): \"In a 2x2 ablation (6 Phase-1 sources x 5 folds, one seed), RecAdam did not change "
            "test performance beyond run-to-run noise (paired ROC-AUC difference " + v(ra_w_ex['mean'], 3, True).replace(',', '.') +
            " with ASAM after excluding the two collapsed ASAM-only runs, " + v(ra_o_all['mean'], 3, True).replace(',', '.') +
            " without ASAM; all pooled 95% CIs include zero); the accuracy gain over plain AdamW comes from ASAM. RecAdam acts "
            "as an implicit warm-up that delays the rise of validation ROC-AUC by about one epoch. With ASAM, none of the 30 "
            "RecAdam+ASAM runs collapsed versus 2 of 30 ASAM-only runs"
            + ("; when allowed to train for all 30 epochs, " + str(len(perm)) + " of these ASAM-only runs still never left the "
               "ln 2 plateau, while RecAdam+ASAM from the same checkpoint and fold did" if perm else "")
            + ". We report this as a single-seed stability observation, not as an established effect.\"",
        ],
    })

    # ------------------------------------------------------------------ 8. tài liệu
    lit = [
        ["Chen et al., Recall and Learn (RecAdam), https://doi.org/10.18653/v1/2020.emnlp-main.634", "EMNLP 2020",
         "gốc của phương pháp; mục đích là giảm quên tác vụ tiền huấn luyện. Mã gốc không neo head (anneal_w = 0), k 0,5, t0 250 bước, γ 5 000 - khác cấu hình ta (asam_collapse.md mục 6)", "tóm tắt + mã"],
        ["Xu et al., Raise a Child in Large Language Model (ChildTuning), https://aclanthology.org/2021.emnlp-main.749.pdf",
         "EMNLP 2021", "BERT-large, 4 tác vụ GLUE, TB 10 seed: RecAdam 79,17 so với fine-tune thường 78,42 (+0,75), MRPC +0,08; weight decay về trọng số tiền huấn luyện +0,67, Mixout +0,84, R3F +0,88. ỦNG HỘ 'lợi ích trung bình nhỏ'", "Bảng 3 (đã đối chiếu)"],
        ["Zheng et al., Preserving Commonsense Knowledge from PLMs via Causal Inference, https://doi.org/10.18653/v1/2023.acl-long.509",
         "ACL 2023", "RoBERTa-large, 7 tập QA: RecAdam thấp hơn fine-tune thường ở 4/7 (CSQA -0,31, SIQA -0,75), cao hơn 3/7 (OBQA +2,56), std CAO hơn ở 6/7 (PIQA 1,25 so với 0,53). NGƯỢC 'RecAdam ổn định hoá'", "Bảng 1 (đã đối chiếu)"],
        ["Hua et al., Improving PLM Fine-tuning with Noise Stability Regularization, https://arxiv.org/abs/2206.05658",
         "IEEE TNNLS (theo arXiv; chưa mở trang IEEE)", "L2-SP (cùng họ phạt bậc hai về neo) qua 25 seed: trung bình đổi ít và lẫn dấu, std giảm ở 3/4 - 4/4 tác vụ. Ủng hộ gián tiếp khả năng 'ổn định hoá' của neo, không phải của RecAdam", "theo agent tra cứu (Bảng II)"],
        ["Li et al., Explicit Inductive Bias for Transfer Learning (L2-SP), https://proceedings.mlr.press/v80/li18a.html",
         "ICML 2018", "gốc của phạt bậc hai về trọng số tiền huấn luyện mà RecAdam dùng; ảnh, không phải mã", "tóm tắt"],
        ["Lee et al., Mixout, https://arxiv.org/abs/1909.11299", "ICLR 2020",
         "kéo về trọng số tiền huấn luyện tăng ổn định fine-tune BERT-large (có lần chạy suy biến); bối cảnh", "tóm tắt"],
        ["Zhang et al., Revisiting Few-sample BERT Fine-tuning, https://arxiv.org/abs/2006.05987", "ICLR 2021",
         "sửa quy trình tối ưu thì tác dụng của các kỹ thuật ổn định hoá đề xuất trước đó giảm đáng kể: ủng hộ đọc RecAdam như sửa bất ổn chứ không nâng trần", "tóm tắt"],
        ["Mosbach et al., On the Stability of Fine-tuning BERT, https://arxiv.org/abs/2006.04884", "ICLR 2021",
         "run thất bại kẹt ở loss ≈ ln 2 do khó tối ưu; bias correction của Adam là 'implicit warmup' - cả hai nhánh của ta đều có, warmup riêng của RecAdam là λ(t)", "thân bài (asam_collapse.md)"],
        ["Zhou et al., On Calibration of Pre-trained Code Models, https://doi.org/10.1145/3597503.3639126", "ICSE 2024",
         "mô hình mã tiền huấn luyện quá tự tin, tệ hơn khi lệch phân phối: khớp mục 5(d); lý do báo ECE / Brier cạnh F1", "tóm tắt"],
        ["Liu et al., On the Reproducibility and Replicability of DL in SE, https://doi.org/10.1145/3477535", "ACM TOSEM 2022",
         "kết quả DL trong SE khó tái lập vì tối ưu bất ổn: căn cứ báo ghép cặp và không kết luận từ một seed", "tóm tắt"],
        ["Steenhoek et al., An Empirical Study of DL Models for Vulnerability Detection, https://doi.org/10.1109/ICSE48619.2023.00188",
         "ICSE 2023", "chỉ đổi seed đã đổi dự đoán của nhiều mẫu test: cùng hướng với mục 5 (RecAdam đổi dự đoán cỡ một lần đổi seed)", "thân bài (asam_collapse.md)"],
        ["Weyssow et al., Continual Learning for OOD Generalization in PLMs of Code, https://doi.org/10.1145/3611643.3616244",
         "ESEC/FSE 2023", "EWC (neo về trọng số cũ) kém replay trên mô hình mã: neo không tự động có lợi", "một phần thân bài (asam_collapse.md)"],
        ["Gao et al., REPEAT - Continual Learning of Code Intelligence Models, https://arxiv.org/abs/2302.03482", "ICSE 2023",
         "điều chuẩn kiểu EWC giảm quên trên CodeBERT / CodeT5 (có Big-Vul) - đo quên trực tiếp trên tác vụ cũ, đúng loại phép đo khối này thiếu", "một phần thân bài (asam_collapse.md)"],
        ["Wang et al., One Adapter for All Programming Languages?, https://arxiv.org/abs/2303.15822", "ICSE 2023",
         "fine-tune đa ngôn ngữ làm giảm hiệu năng, quy cho quên; chống quên bằng adapter + đóng băng backbone - đối chứng khác RecAdam", "tóm tắt"],
        ["Shi et al., Towards Efficient Fine-tuning of Pre-trained Code Models (Telly), https://doi.org/10.1145/3597926.3598036",
         "ISSTA 2023", "đóng băng lớp dưới giữ hoặc tăng hiệu năng: đối chứng rẻ để tách 'neo' khỏi 'warmup' của RecAdam", "tóm tắt (asam_collapse.md)"],
    ]
    sections.append({
        "heading": "8. Tài liệu liên quan (đã kiểm link; mức đọc ghi ở cột cuối)",
        "paras": [
            "Tra cứu thêm cho câu hỏi này (agent tra cứu song song, ~25 truy vấn; hai bài then chốt Xu 2021 và Zheng 2023 đã đối "
            "chiếu lại bảng số trong PDF). Không tìm thấy bài nào dùng RecAdam hoặc SAM/ASAM cho mô hình mã hay phát hiện lỗ hổng, "
            "cũng không bài nào so RecAdam với L2-SP / EWC trong cùng thí nghiệm. Văn liệu ủng hộ vế 'lợi ích trung bình của "
            "RecAdam nhỏ' (Xu 2021: +0,75 điểm; Zheng 2023: thua ở 4/7 tập), còn vế 'ổn định hoá' thì chính RecAdam không có bằng "
            "chứng (Zheng 2023: std cao hơn 6/7), chỉ có bằng chứng gián tiếp từ L2-SP. Đây là 'chưa tìm thấy' qua tìm kiếm web, "
            "chưa dò toàn văn ACM DL / IEEE Xplore.",
        ],
        "table": {"head": ["bài (link)", "venue", "liên quan tới kết luận", "mức đọc"], "rows": lit},
    })

    nxt = [
        ("1 (đang chạy, không tốn thêm): còn " + ", ".join(k.replace(" f", "_me30 f") for k in pending) + " với --min_epochs 30; "
         "xong thì chạy lại make_recadam_figures.py để cập nhật phép nhạy 'me30' và đoạn diễn giải mục 4."
         if pending else
         "1 (xong): 3 ô sập đã chạy lại với --min_epochs 30; kết quả đã vào mục 4 và phép nhạy 'me30'."),
        "2 (rẻ nhất cho câu 'nhớ nguồn', ~10-15 phút GPU, không huấn luyện, không sửa mã nếu --phase test nhận checkpoint Pha 1): "
        "chấm 6 checkpoint Pha 1 (zero-shot, head nhị phân của Pha 1) trên 5 fold test SVEN, lưu probs. Rồi đo Spearman giữa dự "
        "đoán Pha 2 và dự đoán Pha 1 cùng nguồn: nếu RecAdam giữ nguồn thì chỉ RecAdam gần Pha 1 hơn AdamW ở ≥ 4/5 fold; nếu "
        "không khác thì tín hiệu 5(b) là nhiễu. Cần kiểm head Pha 1 có nạp được không (init_from đang dùng 'head luôn mới').",
        "3 (đo quên trực tiếp, ~3 GPU-giờ): AdamW và chỉ RecAdam trên 2 nguồn (common chỉ JS, common JS + C/C++) × f1-f3, GIỮ "
        "checkpoint Pha 2, chấm trên val của nguồn Pha 1. Đây là phép đo duy nhất cho phép viết 'chống quên'. Bậc 1 (n = 3).",
        "4 (kiểm 0/30 vs 2/30 sập, ~6 GPU-giờ): lặp seed Pha 2 1234 cho chỉ ASAM và RecAdam + ASAM trên 3 nguồn JS + C/C++, "
        "f1-f3 (trùng phép kiểm 3 của asam_collapse.md - chạy một lần cho cả hai câu hỏi).",
        "5 (tách 'warmup' khỏi 'neo', ~3 GPU-giờ): chỉ ASAM + warmup tuyến tính 60 bước (đúng mốc λ ≥ 0,9) trên 3 nguồn JS + "
        "C/C++, f1-f3 (phép kiểm 2 của asam_collapse.md). Nếu bằng RecAdam + ASAM thì phần 'ổn định' (nếu có) là warmup, không "
        "phải neo.",
        "Không cần GPU: nếu giữ RecAdam trong bài, báo kèm cấu hình khác bản gốc (neo cả head, k 0,05, t0 = bước 4, γ 500) và "
        "bảng ablation 2×2 đủ 4 chỉ số; mọi phép kiểm trên đều phải hỏi người dùng trước khi chạy.",
    ]

    sources = [
        "Số liệu: meta/insights/recadam/numbers.json (sinh bởi meta/insights/recadam/make_recadam_figures.py từ results/<run>/"
        "fold<k>.json, results/<run>/fold<k>.probs.npz, meta/dbrows/<run>__f<k>.json, logs/<run>/fold<k>.log, "
        "data/final_experiment_data/sven_python_folds_nocomment/fold<k>/test.jsonl)",
        "Bảng: table_main, table_recadam_delta, table_stability, table_indirect (.csv / .tex / .md, tiếng Anh cho bài)",
        "Hình: fig_recadam_forest, fig_recadam_interaction, fig_recadam_paired_folds, fig_recadam_fold_sd, fig_recadam_dynamics, "
        "fig_recadam_indirect (.svg / .pdf)",
        "Phân tích liền trước (tái dùng định nghĩa bình nguyên, kẹt, sập và số toàn khối): meta/insights/asam_collapse.md, "
        "asam_collapse_numbers.json",
        "Mã: /drive1/cuongtm/ntat/MultiVD/src_mwonly/mwg/optim.py (RecAdam, anneal_function, SAMStep), src_mwonly/train.py "
        "(build_optimizer, vòng train)",
        "Ghi chép khối: /drive1/cuongtm/ntat/MultiVD/CURRENT_RUN.md (mwonly5, các mục XONG và chấm dự đoán)",
        "Tham chiếu nhiễu seed: archive_20261004/results/{mw,mwg}_assemble_asamonly_jsonly{,_s1234,_s7}",
    ] + [r[0] + " - " + r[1] for r in lit]

    doc = {
        "title": "Đóng góp của RecAdam trong Pha 2",
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "question": QUESTION,
        "verdict": verdict,
        "confidence": "trung bình",
        "tier": tier,
        "sections": sections,
        "next": nxt,
        "sources": sources,
    }
    return doc


def to_md(doc):
    L = [f"# {doc['title']}", "", f"Cập nhật: {doc['updated']}", "", f"**Câu hỏi (nguyên văn):** {doc['question']}", "",
         f"**Kết luận:** {doc['verdict']}", "", f"**Mức tin:** {doc['confidence']}", "", f"**Độ mạnh dữ liệu:** {doc['tier']}", ""]
    for s in doc["sections"]:
        L += [f"## {s['heading']}", ""]
        for p in s.get("paras", []):
            L += [p, ""]
        for b in s.get("bullets", []):
            L.append(f"- {b}")
        if s.get("bullets"):
            L.append("")
        if s.get("table"):
            t = s["table"]
            L.append("| " + " | ".join(t["head"]) + " |")
            L.append("|" + "---|" * len(t["head"]))
            for r in t["rows"]:
                L.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
            L.append("")
        if s.get("figure"):
            f = s["figure"]
            L += [f"![{f['file']}]({f['file']})", "", f"*Hình:* {f['caption']}", "", f"*Caption (EN):* {f['caption_en']}", ""]
    L += ["## Phép kiểm đề xuất (chưa chạy)", ""] + [f"- {x}" for x in doc["next"]] + ["", "## Nguồn", ""] + \
         [f"- {x}" for x in doc["sources"]] + [""]
    return "\n".join(L)


def check_text(s):
    bad = [ch for ch in ("\u2013", "\u2014", "\u2012", "\u2015") if ch in s]
    if bad:
        raise SystemExit(f"có gạch dài trong văn bản: {bad}")
    if "<" in s and re.search(r"<(p|div|span|br|b|i)[ >/]", s):
        raise SystemExit("có thẻ HTML trong văn bản")


def write(num):
    doc = build(num)
    js = json.dumps(doc, ensure_ascii=False, indent=1)
    check_text(js)
    json.loads(js)
    md = to_md(doc)
    check_text(md)
    with open(os.path.join(OUT, "recadam_contribution.json"), "w", encoding="utf-8") as fh:
        fh.write(js + "\n")
    with open(os.path.join(OUT, "recadam_contribution.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    return [os.path.join(OUT, "recadam_contribution.json"), os.path.join(OUT, "recadam_contribution.md")]


if __name__ == "__main__":
    num = json.load(open(os.path.join(OUT, "numbers.json")))
    for p in write(num):
        print("ghi", p)
