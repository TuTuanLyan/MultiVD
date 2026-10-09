#!/usr/bin/env python3
"""Dựng dữ liệu nguồn từ dữ liệu THÔ: PrimeVul v0.1 (paired + unpaired) và CleanVul mức 3–4 (Java, JS).

Bốn bộ, mỗi bộ ghi `<bộ>_{full,common,4cwe}.jsonl` + `.meta.jsonl` (xuất xứ, mã thô, cùng thứ tự dòng):
    ccpp_primevul_paired     đơn vị = CẶP (bản lỗi, bản vá) — hai dòng liền nhau của *_paired.jsonl
    ccpp_primevul_unpaired   đơn vị = HÀNG
    java_cleanvul_3-4        đơn vị = CẶP (func_before = 1, func_after = 0)
    js_cleanvul_3-4          đơn vị = CẶP

Sau khi xoá comment, luật áp theo thứ tự; đơn vị dính luật nào trước mang lý do đó:
    T0  rỗng sau khi xoá comment
    T1  mã trùng nguyên văn (bỏ khoảng trắng) mà có cả nhãn 0 và 1 -> bỏ MỌI bản (lỗi nhãn)
    T2  mã sinh / nén (chỉ JS) và hàm không có thân (xem filters.py)
    T3  trùng nguyên văn, cùng nhãn -> giữ bản gặp trước
    T4  trùng dạng abstract (định danh -> ID, chuỗi -> S, số -> N), cùng nhãn -> giữ bản gặp trước
        (ngược nhãn ở dạng abstract thì GIỮ: thường là bản vá thật chỉ đổi tên)
Bộ theo cặp: bỏ một nửa là bỏ cả cặp. `common` / `4cwe` lấy từ các hàng đã giữ của `full`.

    python dataset/build_sources.py --primevul_raw <thư mục PrimeVul> --cleanvul_dir <thư mục CSV CleanVul> \
        --cwe_xml data/cwe_spec/cwec_v4.20.xml --out data/sources_v4 [--workers 4] [--only <bộ,...>]
"""
import argparse
import collections
import csv
import json
import os
import re
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cwe as C                                           # noqa: E402
import filters as FL                                      # noqa: E402
from mwg.comments import strip                           # noqa: E402

csv.field_size_limit(10 ** 9)
nows = lambda s: re.sub(r"\s+", "", s)                    # noqa: E731
LX = {"ccpp": "c", "java": "java", "js": "js"}


# ----------------------------------------------------------------------------- đọc dữ liệu thô
def read_primevul_paired(d):
    """Hai dòng liền nhau là một cặp, bắt buộc hai nhãn {0, 1} (commit có thể khác: PrimeVul ghép bản vá của cùng hàm)."""
    rows = []
    for split in ("train", "valid", "test"):
        ds = [json.loads(l) for l in open(os.path.join(d, "primevul_%s_paired.jsonl" % split), encoding="utf-8")]
        assert len(ds) % 2 == 0, split
        for k in range(0, len(ds), 2):
            a, b = ds[k], ds[k + 1]
            assert {int(a["target"]), int(b["target"])} == {0, 1}, (split, k)
            pid = "primevul:%s-%s" % (a["idx"], b["idx"])
            commits_differ = a["commit_id"] != b["commit_id"]
            for x in (a, b):
                rows.append({"code_raw": x["func"], "label": int(x["target"]), "pair_id": pid, "row_id": "primevul:%s" % x["idx"],
                             "cwe_raw": x.get("cwe"), "cve": x.get("cve"), "project": x.get("project"),
                             "commit_url": x.get("commit_url"), "file_name": x.get("file_name"), "orig_split": split,
                             "pair_commits_differ": commits_differ})
    return rows


def read_primevul_unpaired(d):
    rows = []
    for split in ("train", "valid", "test"):
        for l in open(os.path.join(d, "primevul_%s.jsonl" % split), encoding="utf-8"):
            x = json.loads(l)
            rows.append({"code_raw": x["func"], "label": int(x["target"]), "pair_id": None, "row_id": "primevul:%s" % x["idx"],
                         "cwe_raw": x.get("cwe"), "cve": x.get("cve"), "project": x.get("project"),
                         "commit_url": x.get("commit_url"), "file_name": x.get("file_name"), "orig_split": split})
    return rows


def read_cleanvul(d, ext):
    """Mức 3 + 4, lọc theo đuôi file. Không khử trùng ở đây: bản trùng đi qua T3 để có lý do ghi lại."""
    rows = []
    for s in (3, 4):
        with open(os.path.join(d, "vulnerability_score_%d.csv" % s), newline="", encoding="utf-8") as fh:
            for k, r in enumerate(csv.DictReader(fh)):
                if (r.get("extension") or "").strip().lower() != ext:
                    continue
                pid = "cleanvul:%d:%d" % (s, k)
                base = {"pair_id": pid, "cwe_raw": r.get("cwe_id"), "cve": r.get("cve_id"), "commit_url": r.get("commit_url"),
                        "file_name": r.get("file_name"), "score": s}
                rows.append(dict(base, code_raw=r.get("func_before") or "", label=1, half="before", row_id=pid + ":1"))
                rows.append(dict(base, code_raw=r.get("func_after") or "", label=0, half="after", row_id=pid + ":0"))
    return rows


def _strip_one(args):
    code, lx = args
    out, st = strip(code.replace("\r\n", "\n").replace("\r", "\n"), lx)
    return out, st["n_spans"], st["chars_removed"]


# ----------------------------------------------------------------------------- luật T0–T4
def apply_rules(rows, lang, paired):
    """-> (đơn vị giữ theo thứ tự đọc, {đơn vị: (lý do, chi tiết)}, thống kê, {đơn vị: [chỉ số hàng]})."""
    lx = LX[lang]
    unit = [r["pair_id"] if paired else r["row_id"] for r in rows]
    order = list(dict.fromkeys(unit))
    rows_of = collections.defaultdict(list)
    for i, u in enumerate(unit):
        rows_of[u].append(i)
    pos = {u: k for k, u in enumerate(order)}
    drop, stat = {}, collections.Counter()

    def drop_unit(u, why, detail=""):
        if u not in drop:
            drop[u] = (why, detail)
            stat[why] += 1

    for i, r in enumerate(rows):                                                   # T0
        if not r["code"].strip():
            drop_unit(unit[i], "T0_empty_after_comment_removal")

    grp = collections.defaultdict(list)                                            # T1
    for i, r in enumerate(rows):
        if r["code"].strip():
            grp[nows(r["code"])].append(i)
    n_conflict_groups = 0
    for idx in grp.values():
        if len({rows[i]["label"] for i in idx}) > 1:
            n_conflict_groups += 1
            units = sorted({unit[i] for i in idx}, key=pos.__getitem__)
            detail = "xung_dot_%d: %d bản (nhãn 1: %d, nhãn 0: %d) — %s" % (
                n_conflict_groups, len(idx), sum(rows[j]["label"] == 1 for j in idx), sum(rows[j]["label"] == 0 for j in idx),
                ", ".join(units[:6]) + (" …" if len(units) > 6 else ""))
            for i in idx:
                drop_unit(unit[i], "T1_label_conflict", detail)
    stat["(note) label_conflict_groups"] = n_conflict_groups

    for i, r in enumerate(rows):                                                   # T2
        if unit[i] in drop:
            continue
        why = FL.quality_reason(r["code"], r["code_raw"], r.get("file_name"), lx)
        if why:
            drop_unit(unit[i], why, (r.get("file_name") or "") if why == "T2_build_path" else "")

    for tier, keyf in (("T3_exact_duplicate", lambda r: nows(r["code"])),         # T3, T4
                       ("T4_abstract_duplicate", lambda r: FL.alpha_norm(r["code"], lx))):
        owner = {}
        for u in order:
            if u in drop:
                continue
            ks = [(keyf(rows[i]), rows[i]["label"]) for i in rows_of[u]]
            hit = next((owner[k] for k in ks if k in owner), None)
            if hit is not None:
                drop_unit(u, tier, "trùng với %s (bản đó ĐƯỢC GIỮ)" % hit)
            else:
                for k in ks:
                    owner.setdefault(k, u)

    ab = collections.defaultdict(set)                   # chỉ ghi nhận: ngược nhãn ở dạng abstract giữa các đơn vị giữ
    for u in order:
        if u not in drop:
            for i in rows_of[u]:
                ab[FL.alpha_norm(rows[i]["code"], lx)].add(rows[i]["label"])
    stat["(note) abstract_groups_with_both_labels_kept"] = sum(1 for v in ab.values() if len(v) > 1)
    return [u for u in order if u not in drop], drop, stat, rows_of


# ----------------------------------------------------------------------------- ghi
def keep_complete_pairs(recs, metas):
    """Chỉ giữ cặp còn đủ một hàng nhãn 1 và một hàng nhãn 0 (PrimeVul đôi khi gán CWE khác nhau cho hai nửa)."""
    labs = collections.defaultdict(list)
    for r, m in zip(recs, metas):
        labs[m["pair_id"]].append(r["label"])
    ok = {p for p, v in labs.items() if sorted(v) == [0, 1]}
    R = [r for r, m in zip(recs, metas) if m["pair_id"] in ok]
    M = [m for m in metas if m["pair_id"] in ok]
    return R, M, len(recs) - len(R)


def write_set(outdir, name, recs, metas):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, name + ".jsonl"), "w", encoding="utf-8") as f1, \
         open(os.path.join(outdir, name + ".meta.jsonl"), "w", encoding="utf-8") as f2:
        for r, m in zip(recs, metas):
            f1.write(json.dumps(r, ensure_ascii=False) + "\n")
            f2.write(json.dumps(m, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primevul_raw", required=True, help="thư mục primevul_{train,valid,test}[_paired].jsonl")
    ap.add_argument("--cleanvul_dir", required=True, help="thư mục vulnerability_score_{3,4}.csv")
    ap.add_argument("--cwe_xml", default="data/cwe_spec/cwec_v4.20.xml")
    ap.add_argument("--pillars", default=os.path.join(HERE, "cwe_root_parents.txt"))
    ap.add_argument("--out", default="data/sources_v4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="", help="chỉ dựng các bộ này (phẩy)")
    a = ap.parse_args()
    if not os.path.isfile(a.cwe_xml):
        sys.exit("thiếu %s — xem README (cwec_v4.20.xml, md5 efd2581cfe58dd678965c8e582945c48)" % a.cwe_xml)
    pil, _ = C.load_pillars(a.pillars)
    jobs = [("ccpp_primevul_paired", "ccpp", True, lambda: read_primevul_paired(a.primevul_raw)),
            ("ccpp_primevul_unpaired", "ccpp", False, lambda: read_primevul_unpaired(a.primevul_raw)),
            ("java_cleanvul_3-4", "java", True, lambda: read_cleanvul(a.cleanvul_dir, "java")),
            ("js_cleanvul_3-4", "js", True, lambda: read_cleanvul(a.cleanvul_dir, "js"))]
    if a.only:
        jobs = [j for j in jobs if j[0] in a.only.split(",")]
    report = []
    os.makedirs(os.path.join(a.out, "_dropped"), exist_ok=True)
    for name, lang, paired, reader in jobs:
        t0 = time.time()
        rows = reader()
        with Pool(a.workers) as p:
            res = p.map(_strip_one, [(r["code_raw"], LX[lang]) for r in rows], chunksize=256)
        for r, (code, nsp, nch) in zip(rows, res):
            r["code"], r["n_comment_spans"], r["chars_removed"] = code, nsp, nch
        keep, drop, stat, rows_of = apply_rules(rows, lang, paired)
        recs, metas = [], []
        for u in keep:
            for i in rows_of[u]:
                r = rows[i]
                cids = C.cwe_all(r.get("cwe_raw"))
                recs.append(C.make_row(r["code"], r["label"], cids[0] if cids else None, lang, pil))
                m = {k: v for k, v in r.items() if k != "code"}
                m.update({"cwe_all": cids, "cwe_labels": C.cwe_labels(r.get("cwe_raw"))})
                if not paired:
                    m["pair_id"] = None
                metas.append(m)
        write_set(a.out, name + "_full", recs, metas)
        mm = [dict(m, pair_id=m["pair_id"] or m["row_id"]) for m in metas]         # đơn vị ghép cặp cho luật common
        rc, mc, common_dropped = C.subset_common(recs, mm, lang, a.cwe_xml)
        if rc is not None:
            mc = [dict(m, pair_id=m["pair_id"] if paired else None) for m in mc]
            if paired:
                rc, mc, n_orphans = keep_complete_pairs(rc, mc)
                common_dropped["(note) orphan_halves_dropped"] = n_orphans
            write_set(a.out, name + "_common", rc, mc)
        r4, m4 = C.subset_4cwe(recs, metas)
        if r4 and paired:
            r4, m4, _ = keep_complete_pairs(r4, m4)
        if r4:
            write_set(a.out, name + "_4cwe", r4, m4)
        with open(os.path.join(a.out, "_dropped", name + ".jsonl"), "w", encoding="utf-8") as fh:
            for u, (why, det) in drop.items():
                fh.write(json.dumps({"unit": u, "reason": why, "detail": det,
                                     "rows": [{k: rows[i].get(k) for k in ("row_id", "label", "code", "code_raw", "file_name", "cwe_raw",
                                                                             "cve", "commit_url", "orig_split")} for i in rows_of[u]]},
                                    ensure_ascii=False) + "\n")
        lab = collections.Counter(r["label"] for r in recs)
        rep = {"set": name, "lang": lang, "unit_type": "pair" if paired else "row", "rows_in": len(rows), "units_in": len(rows_of),
               "units_kept": len(keep), "rows_kept": len(recs), "labels_kept": {str(k): v for k, v in sorted(lab.items())},
               "common_rows": len(rc) if rc is not None else None, "cwe4_rows": len(r4) if r4 else 0,
               "dropped_by_reason": dict(stat), "common_dropped_by": common_dropped, "seconds": round(time.time() - t0, 1)}
        report.append(rep)
        print(json.dumps(rep, ensure_ascii=False), flush=True)
    with open(os.path.join(a.out, "_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
