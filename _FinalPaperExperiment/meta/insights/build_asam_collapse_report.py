#!/usr/bin/env python3
"""Dựng báo cáo asam_collapse.json + asam_collapse.md từ số đã tính (asam_collapse_numbers.json) và nội dung viết tay.

Chỉ đọc / ghi trong meta/insights/. Kiểm: JSON hợp lệ, không có gạch dài, không có thẻ HTML.
"""
import json
import os
import re

HERE = "/drive1/cuongtm/ntat/MultiVD/_FinalPaperExperiment/meta/insights"
NUM = json.load(open(os.path.join(HERE, "asam_collapse_numbers.json")))
SE = json.load(open(os.path.join(HERE, "se_related.json"))) if os.path.exists(os.path.join(HERE, "se_related.json")) else None


def vn(x, nd=1, sign=False):
    """Số thập phân kiểu Việt (dấu phẩy)."""
    if sign and round(x, nd) == 0:
        return f"{0:.{nd}f}".replace(".", ",")
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",")


def pfmt(p):
    if p is None:
        return "-"
    if p < 0.001:
        return "< 0,001"
    return vn(p, 3)


P = NUM["new_paired"]


def row(name, key, label):
    b = P[key]
    x = b["plat_0.65"]
    st, co = b["mcnemar_stuck10_0.65"], b["mcnemar_collapse"]
    bs, bf = b["by_source"], b["by_fold"]
    return [label, str(x["n"]), f"{x['pos']}/{x['neg']}/{x['tie']}", f"{vn(x['mean'], 1, True)} ({vn(x['median'], 0, True)})",
            f"{vn(x['min'], 0, True)} .. {vn(x['max'], 0, True)}", pfmt(x.get("wilcoxon_p")),
            f"{bs['pos']}+/{bs['neg']}- trên {bs['n_clusters']}", f"{bf['pos']}+/{bf['neg']}- trên {bf['n_clusters']}",
            f"{st['a_only']} vs {st['b_only']} (p {pfmt(st['p_exact'])})", f"{co['a_only']} vs {co['b_only']} (p {pfmt(co['p_exact'])})"]


inter = NUM["new_interaction_plat_0.65"]
O = NUM["old_paired"]


def orow(key, label):
    x = O[key]["plat_0.65"]
    return [label, str(x["n"]), f"{x['pos']}/{x['neg']}/{x['tie']}", f"{vn(x['mean'], 1, True)} ({vn(x['median'], 0, True)})",
            f"{vn(x['min'], 0, True)} .. {vn(x['max'], 0, True)}", pfmt(x.get("wilcoxon_p"))]


oi = NUM["old_interaction_plat_0.65"]
late = NUM["old_lateasam_paired"]
sn, so = NUM["schedule_new"], NUM["schedule_old"]
pe = sn["per_epoch"]

report = {
    "title": "ASAM có gây sập Pha 2 không, RecAdam có đỡ không",
    "updated": "2026-10-05 12:57",
    "question": "nghiên cứu xem có phải asam gây sập pha 2 và recadam giúp giảm vấn đề này. Nếu chỉ dùng kết quả vì 5 fold là chưa đủ.",
    "verdict": "",
    "confidence": "",
    "tier": "",
    "sections": [],
    "next": [],
    "sources": [],
}

exec(open(os.path.join(HERE, "asam_collapse_text.py")).read())   # điền verdict / sections / next / sources

# ----------------------------------------------------------------------------- kiểm
blob = json.dumps(report, ensure_ascii=False)
assert "\u2013" not in blob and "\u2014" not in blob, "có gạch dài"
assert not re.search(r"<[a-zA-Z/][^>]*>", blob), "có thẻ HTML"
assert set(report) == {"title", "updated", "question", "verdict", "confidence", "tier", "sections", "next", "sources"}
assert report["confidence"] in ("thấp", "trung bình", "cao")
for s in report["sections"]:
    assert set(s) <= {"heading", "paras", "bullets", "table"}, s.keys()
    if "table" in s:
        n = len(s["table"]["head"])
        assert all(len(r) == n for r in s["table"]["rows"]), s["heading"]
with open(os.path.join(HERE, "asam_collapse.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, ensure_ascii=False, indent=1)

# ----------------------------------------------------------------------------- Markdown
md = [f"# {report['title']}", "", f"Cập nhật: {report['updated']}", "", f"**Câu hỏi (nguyên văn):** {report['question']}", "",
      f"**Kết luận:** {report['verdict']}", "", f"**Mức tin:** {report['confidence']}", "", f"**Độ mạnh dữ liệu:** {report['tier']}", ""]
for s in report["sections"]:
    md += [f"## {s['heading']}", ""]
    for p in s.get("paras", []):
        md += [p, ""]
    for b in s.get("bullets", []):
        md.append(f"- {b}")
    if s.get("bullets"):
        md.append("")
    if "table" in s:
        h = s["table"]["head"]
        md.append("| " + " | ".join(h) + " |")
        md.append("|" + "|".join(["---"] * len(h)) + "|")
        for r in s["table"]["rows"]:
            md.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")
        md.append("")
md += ["## Phép kiểm đề xuất (chưa chạy)", ""] + [f"- {x}" for x in report["next"]] + ["", "## Nguồn", ""] + [f"- {x}" for x in report["sources"]] + [""]
txt = "\n".join(md)
assert "\u2013" not in txt and "\u2014" not in txt
with open(os.path.join(HERE, "asam_collapse.md"), "w", encoding="utf-8") as fh:
    fh.write(txt)
json.load(open(os.path.join(HERE, "asam_collapse.json"), encoding="utf-8"))
print("ok", len(report["sections"]), "mục,", len(blob), "ký tự")
