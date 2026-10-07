#!/usr/bin/env python3
"""Hướng A - mask trên full JS + C/C++ đủ 5 fold (người dùng 06/10 13:0x): hiệu ghép cặp theo fold với chỉ ASAM gốc / AdamW / cột chính,
epoch thoát bình nguyên (train loss < 0,65), chấm D1-D4 (CURRENT_RUN). Chỉ đọc results/ + logs/."""
import json, os, re, statistics as st
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
K = [("test_roc_auc", "ROC"), ("test_pr_auc", "PR"), ("test_macro_f1_at_0.5", "F1@0,5"), ("test_macro_f1_at_valcal", "F1@val")]
TIE = 1e-3
R = lambda r, k: json.load(open(os.path.join(BASE, "results", r, "fold%d.json" % k)))
def esc(r, k):
    for l in open(os.path.join(BASE, "logs", r, "fold%d.log" % k), encoding="utf-8"):
        m = re.search(r"Epoch (\d+)/30 \| train loss ([0-9.]+)", l)
        if m and float(m.group(2)) < 0.65: return int(m.group(1))
F = [k for k in range(1, 6) if os.path.exists(os.path.join(BASE, "results", "asamAmask_full_jscpp", "fold%d.json" % k))]
print("fold có:", F)
print("epoch thoát | gốc:", [esc("asamonly_full_jscpp", k) for k in F], "| mask:", [esc("asamAmask_full_jscpp", k) for k in F])
print("ROC mask:", [round(R("asamAmask_full_jscpp", k)["test_roc_auc"], 4) for k in F])
res = {}
for ref, lab in (("asamonly_full_jscpp", "mask - chỉ ASAM gốc"), ("noras_full_jscpp", "mask - AdamW"), ("rasam_full_jscpp", "mask - chính")):
    cells = []
    for key, kl in K:
        d = [R("asamAmask_full_jscpp", k)[key] - R(ref, k)[key] for k in F]
        cells.append("%s %+.4f (+%d/-%d)" % (kl, st.mean(d), sum(x > TIE for x in d), sum(x < -TIE for x in d)))
        if key == "test_roc_auc": res[ref] = d
    print("%-22s" % lab, " | ".join(cells))
    print("%-22s" % "", "ROC theo fold:", " / ".join("%+.4f" % x for x in res[ref]))
if len(F) == 5:
    e = [esc("asamAmask_full_jscpp", k) for k in F]
    f4 = R("asamAmask_full_jscpp", 4)["test_roc_auc"]
    d3 = res["asamonly_full_jscpp"]; d4a = res["noras_full_jscpp"]; d4b = res["rasam_full_jscpp"]
    print("D1", "ĐÚNG" if repr(f4) == "0.9439497118910425" else "SAI", "f4 =", repr(f4))
    print("D2", "ĐÚNG" if all(x is not None and x <= 7 for x in e) else "SAI", "thoát:", e)
    same = lambda d: all(x > TIE for x in d) or all(x < -TIE for x in d)
    print("D3", "ĐÚNG" if (-0.01 <= st.mean(d3) <= 0.03 and not same(d3)) else "SAI", "TB %+.4f" % st.mean(d3))
    print("D4", "ĐÚNG" if (not same(d4a) and sum(x < -TIE for x in d4b) >= 3) else "SAI",
          "AdamW +%d/-%d, chính thấp hơn ở %d/5" % (sum(x > TIE for x in d4a), sum(x < -TIE for x in d4a), sum(x < -TIE for x in d4b)))
