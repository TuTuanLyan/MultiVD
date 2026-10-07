#!/usr/bin/env python3
"""Dựng dữ liệu nguồn v4 từ dữ liệu THÔ: PrimeVul (paired + unpaired) và CleanVul mức 3–4 (Java, JS).

Bốn bộ ra, mỗi bộ có `full`, `common` và (nếu có nhãn CWE) `4cwe`:
    ccpp_primevul_paired     đơn vị = CẶP (bản lỗi, bản vá) — hai dòng liền nhau cùng commit
    ccpp_primevul_unpaired   đơn vị = HÀNG
    ccpp_primevul_merged     paired + unpaired GỘP rồi lọc T0–T4 trên toàn bộ: hàng thuộc một cặp -> đơn vị = CẶP,
                             còn lại -> đơn vị = HÀNG; cặp đứng TRƯỚC nên bản trong cặp được giữ khi trùng
                             (paired nằm trọn trong unpaired: 9 398/9 398 hàm cùng idx, cùng code, cùng nhãn)
    java_cleanvul_3-4        đơn vị = CẶP (func_before = 1, func_after = 0)
    js_cleanvul_3-4          đơn vị = CẶP

Thứ tự luật (người dùng chốt 23/09) — một đơn vị dính luật nào trước thì mang lý do đó:
    T0  rỗng sau khi xoá comment
    T1  NHÃN MÂU THUẪN: mã trùng nguyên văn (bỏ khoảng trắng) mà có cả nhãn 0 lẫn 1
        → bỏ MỌI bản. Đặt trước lọc chất lượng: lỗi nhãn không phụ thuộc chất lượng mã,
        lọc chất lượng trước có thể xoá mất một phía và để sót bản mang nhãn sai.
    T2  CHẤT LƯỢNG MÃ: mã sinh/nén (chỉ JS, luật của clean_v3) và hàm không có thân (cả 3).
        Đặt trước khử trùng để bản được giữ lại luôn là bản tốt (vd. bản ở src/ chứ không
        phải bản trong dist/).
    T3  TRÙNG NGUYÊN VĂN, CÙNG NHÃN → giữ bản đầu tiên, bỏ phần còn lại.
    T4  TRÙNG DẠNG ABSTRACT (định danh → ID1.., chuỗi → S, số → N), CÙNG NHÃN → giữ bản đầu.
        Ngược nhãn ở dạng abstract thì KHÔNG bỏ: đó thường là bản vá thật chỉ đổi tên hàm/biến
        (đo 22/09: 153 cặp như `g_free` → `vcard_apdu_delete`). Chỉ ghi số lượng để biết.

Với bộ theo CẶP, bỏ một nửa là bỏ CẢ CẶP — không để lại nửa mồ côi.
Chỉ thị tiền xử lý C (`#define`, `#if`…) là logic mã, được GIỮ NGUYÊN; chỉ xoá comment.
"""
import argparse, collections, csv, hashlib, json, os, re, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # tools/
sys.path.insert(0, HERE)
from strip_comments import strip                     # noqa: E402  bộ xoá comment duy nhất (= babel/strip_comments.py)
import build_sources_v2 as B                         # noqa: E402  nhãn CWE, pillar, bộ common / 4cwe
import clean_v3 as C3                                # noqa: E402  luật mã sinh/nén của JS, alpha_norm

csv.field_size_limit(10 ** 9)
NOWS = re.compile(r"\s+")
nows = lambda s: NOWS.sub("", s)
LX = {"ccpp": "c", "java": "java", "js": "js"}


# ------------------------------------------------------------------ đọc dữ liệu thô
def read_primevul_paired(d):
    """Ba file *_paired.jsonl; hai dòng liền nhau là một cặp. Kiểm tra: cùng commit, nhãn {0, 1}."""
    rows = []
    for split in ("train", "valid", "test"):
        ds = [json.loads(l) for l in open(os.path.join(d, "primevul_%s_paired.jsonl" % split), encoding="utf-8")]
        assert len(ds) % 2 == 0, split
        for k in range(0, len(ds), 2):
            a, b = ds[k], ds[k + 1]
            # Bắt buộc hai nhãn {0, 1}. Commit có thể KHÁC: PrimeVul ghép bản lỗi với bản vá của cùng
            # hàm dù commit khác (đo 23/09: 17 cặp ở train, vd. ReadOneJNGImage, ghostpdl ↔ ghostscript).
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


def read_primevul_merged(d):
    """Paired TRƯỚC (đơn vị cặp), unpaired SAU (đơn vị hàng) — luật giữ-bản-gặp-trước ưu tiên bản trong cặp."""
    return read_primevul_paired(d) + read_primevul_unpaired(d)


def read_cleanvul(d, ext):
    """Mức 3 + 4, lọc theo đuôi file. KHÔNG khử trùng ở đây: mọi bản trùng đi qua T3 để có lý do ghi lại."""
    rows = []
    for s in (3, 4):
        with open(os.path.join(d, "vulnerability_score_%d.csv" % s), newline="", encoding="utf-8") as fh:
            for k, r in enumerate(csv.DictReader(fh)):
                if (r.get("extension") or "").strip().lower() != ext:
                    continue
                before, after = r.get("func_before") or "", r.get("func_after") or ""
                pid = "cleanvul:%d:%d" % (s, k)
                base = {"pair_id": pid, "cwe_raw": r.get("cwe_id"), "cve": r.get("cve_id"), "commit_url": r.get("commit_url"),
                        "file_name": r.get("file_name"), "score": s}
                rows.append(dict(base, code_raw=before, label=1, half="before", row_id=pid + ":1"))
                rows.append(dict(base, code_raw=after, label=0, half="after", row_id=pid + ":0"))
    return rows


# ------------------------------------------------------------------ xoá comment (song song)
def _strip_one(args):
    code, lx = args
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    out, st = strip(code, lx)
    return out, st["n_spans"], st["chars_removed"]


# ------------------------------------------------------------------ luật
_GENERATED_REASON = {"gen_marker": "T2_generator_marker", "lime_artifact": "T2_lime_artifact",
                     "hinh_dang_kem_mangle": "T2_compressed_shape_mangled"}   # giá trị trả về của clean_v3.is_generated


def quality_reason(r, lx):
    """T2 — trả về lý do loại hoặc None. Luật mã sinh/nén CHỈ áp cho JS (bản build của Java là
    .class, của C là .o nên file .java/.c luôn là mã nguồn); luật không-thân áp cho cả 3."""
    code, raw, fn = r["code"], r["code_raw"], r.get("file_name") or ""
    if lx == "js":
        if C3.is_build_path(fn, code, lx):
            return "T2_build_path"
        if C3.COMPRESS_FN.match(code):
            return "T2_short_function_name"
        if C3.is_mangled(code, lx, min_ids=4) and C3.shape_compressed(code):
            return "T2_mangled_content"
        g = C3.is_generated(raw, lx)
        if g:
            return _GENERATED_REASON[g]
        if len(raw.split("\n")) == 1:
            return "T2_one_line"
    if has_no_body(code, lx):
        return "T2_no_body"
    return None


_INIT = re.compile(r"\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?:(?!:)")


def _matching_brace(code, k):
    """Vị trí `}` khớp với `{` tại k (bỏ qua chuỗi/ký tự); -1 nếu hàm cụt."""
    d, n, q = 0, len(code), None
    while k < n:
        ch = code[k]
        if q:
            if ch == "\\":
                k += 2
                continue
            if ch == q or ch == "\n":
                q = None
        elif ch in "\"'`":
            q = ch
        elif ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
            if d == 0:
                return k
        k += 1
    return -1


def _body_brace(code, lx):
    """Vị trí `{` mở THÂN hàm: `{` đầu tiên nằm NGOÀI chuỗi và NGOÀI ngoặc tròn/vuông — tức sau khi
    danh sách tham số đã đóng. Bỏ qua: `{}` của tham số mặc định JS (`opts = {}`), `{` trong chuỗi
    annotation Java (`"asc/{}/x"`), và (chỉ C++) khởi tạo bằng ngoặc nhọn trong danh sách khởi tạo
    (`X() : m_{0} {}`). -1 nếu không có (khai báo)."""
    n, k, par, q, first_brace = len(code), 0, 0, None, -1
    while k < n:
        ch = code[k]
        if q:
            if ch == "\\":
                k += 2
                continue
            if ch == q or ch == "\n":
                q = None
        elif ch in "\"'`":
            q = ch
        elif ch in "([":
            par += 1
        elif ch in ")]":
            par = max(0, par - 1)
        elif ch == "{" and first_brace < 0:
            first_brace = k                                   # `{` đầu tiên ngoài chuỗi — dùng khi ngoặc tròn lệch
        if not q and ch == "{" and par == 0:
            before = code[:k].rstrip()
            if lx == "c" and before and (before[-1].isalnum() or before[-1] == "_") and _INIT.search(before):
                e = _matching_brace(code, k)
                if e < 0:
                    return -1
                k = e + 1
                continue
            return k
        k += 1
    # Ngoặc tròn không bao giờ về 0: bản vá PrimeVul ghép hỏng còn dính dòng chữ ký cũ
    # (`f(int *mtu,\nf(unsigned int *mtu, ...)\n{`) — lấy `{` đầu tiên ngoài chuỗi.
    return first_brace


def has_no_body(code, lx="c"):
    """Hàm KHÔNG có thân: không có `{` mở thân (khai báo), hoặc thân — từ `{` mở thân tới `}` KHỚP
    với nó — rỗng. Constructor chỉ có danh sách khởi tạo mà thân rỗng cũng tính là không thân
    (người dùng chốt 23/09: bản vá của 4/5 ca như vậy không nhìn thấy được trong hàm).
    Hàm CỤT thiếu `}` cuối (PrimeVul: 5 559 hàm unpaired) KHÔNG phải rỗng — luật cũ `rfind("}")`
    đọc nhầm những hàm không có `}` nào thành rỗng, vd. `ext4_isize`."""
    i = _body_brace(code, lx)
    if i < 0:
        return True
    e = _matching_brace(code, i)
    return not (code[i + 1:e] if e >= 0 else code[i + 1:]).strip()


def apply_rules(rows, lang, paired):
    """Áp T0–T4. Trả về (tập đơn vị giữ, {đơn vị: (lý do, chi tiết)}, thống kê)."""
    lx = LX[lang]
    unit = [(r["pair_id"] or r["row_id"]) if paired else r["row_id"] for r in rows]   # bộ gộp: hàng lẻ là đơn vị riêng
    order, seen_u = [], set()
    for u in unit:                                    # thứ tự đơn vị ổn định = thứ tự đọc
        if u not in seen_u:
            seen_u.add(u); order.append(u)
    rows_of = collections.defaultdict(list)
    for i, u in enumerate(unit):
        rows_of[u].append(i)
    pos = {u: k for k, u in enumerate(order)}
    drop, stat = {}, collections.Counter()

    def drop_unit(u, why, detail=""):
        if u not in drop:
            drop[u] = (why, detail)
            stat[why] += 1

    # T0 — rỗng sau khi xoá comment
    for i, r in enumerate(rows):
        if not r["code"].strip():
            drop_unit(unit[i], "T0_empty_after_comment_removal")

    # T1 — nhãn mâu thuẫn trên mã nguyên văn (bỏ khoảng trắng): bỏ MỌI bản
    grp = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["code"].strip():
            grp[nows(r["code"])].append(i)
    n_conflict_groups = 0
    for k, idx in grp.items():
        labs = {rows[i]["label"] for i in idx}
        if len(labs) > 1:
            n_conflict_groups += 1
            units = sorted({unit[i] for i in idx}, key=pos.__getitem__)
            group_id = "xung_dot_%d" % n_conflict_groups
            for i in idx:
                drop_unit(unit[i], "T1_label_conflict", "%s: %d bản (nhãn 1: %d, nhãn 0: %d) — %s" % (
                    group_id, len(idx), sum(rows[j]["label"] == 1 for j in idx), sum(rows[j]["label"] == 0 for j in idx),
                    ", ".join(units[:6]) + (" …" if len(units) > 6 else "")))
    stat["(note) label_conflict_groups"] = n_conflict_groups

    # T2 — chất lượng mã
    for i, r in enumerate(rows):
        if unit[i] in drop:
            continue
        why = quality_reason(r, lx)
        if why:
            drop_unit(unit[i], why, (r.get("file_name") or "") if why == "T2_build_path" else "")

    # T3 rồi T4 — khử trùng CÙNG NHÃN, giữ đơn vị gặp trước
    for tier, keyf in (("T3_exact_duplicate", lambda r: nows(r["code"])),
                       ("T4_abstract_duplicate", lambda r: C3.alpha_norm(r["code"], lx))):
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

    # Ghi nhận (không bỏ): ngược nhãn ở dạng abstract giữa các đơn vị còn giữ
    ab = collections.defaultdict(set)
    for u in order:
        if u in drop:
            continue
        for i in rows_of[u]:
            ab[C3.alpha_norm(rows[i]["code"], lx)].add(rows[i]["label"])
    stat["(note) abstract_groups_with_both_labels_kept"] = sum(1 for v in ab.values() if len(v) > 1)
    keep = [u for u in order if u not in drop]
    return keep, drop, stat, rows_of


def load_stripped(src_dir, lang):
    """row_id -> (code_raw, code, n_comment_spans, chars_removed) từ các bộ *_full đã dựng cùng ngôn ngữ."""
    out = {}
    for fn in sorted(os.listdir(src_dir)):
        if fn.startswith(lang + "_") and fn.endswith("_full.jsonl"):
            with open(os.path.join(src_dir, fn), encoding="utf-8") as f1, \
                 open(os.path.join(src_dir, fn[:-6] + ".meta.jsonl"), encoding="utf-8") as f2:
                for l, m in zip(f1, f2):
                    r, m = json.loads(l), json.loads(m)
                    out[m["row_id"]] = (m["code_raw"], r["code"], m["n_comment_spans"], m["chars_removed"])
    return out


# ------------------------------------------------------------------ ghi
def keep_complete_pairs(recs, metas):
    """Bộ theo cặp: chỉ giữ cặp còn ĐỦ hai nửa (một nhãn 1, một nhãn 0). Luật CWE của `common` xét
    từng hàng, mà PrimeVul đôi khi gán CWE khác nhau cho bản lỗi và bản vá của cùng cặp (23/09: 6 nửa
    mồ côi) — bỏ một nửa là bỏ cả cặp."""
    labs = collections.defaultdict(list)
    for r, m in zip(recs, metas):
        if m["pair_id"]:
            labs[m["pair_id"]].append(r["label"])
    ok = {p for p, v in labs.items() if sorted(v) == [0, 1]}
    keep = [not m["pair_id"] or m["pair_id"] in ok for m in metas]        # hàng lẻ (bộ gộp) luôn qua
    R = [r for r, k in zip(recs, keep) if k]
    M = [m for m, k in zip(metas, keep) if k]
    return R, M, len(recs) - len(R)


def write_set(outdir, name, recs, metas):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, name + ".jsonl"), "w", encoding="utf-8") as f1, \
         open(os.path.join(outdir, name + ".meta.jsonl"), "w", encoding="utf-8") as f2:
        for r, m in zip(recs, metas):
            f1.write(json.dumps(r, ensure_ascii=False) + "\n")
            f2.write(json.dumps(m, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primevul_raw", default="/drive1/cuongtm/ntat/Archive/PrimeVulRaw")
    ap.add_argument("--cleanvul_dir", required=True)
    ap.add_argument("--out", default="data/sources_v4")
    ap.add_argument("--pillars", default=os.path.join(os.path.dirname(HERE), "..", "data", "cwe_root_parents.txt"))
    ap.add_argument("--cwe_xml", default="data/cwe_spec/cwec_v4.20.xml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="", help="chỉ dựng các bộ này (phẩy), để thử nhanh")
    ap.add_argument("--reuse_stripped", default="", help="thư mục nguồn đã dựng: lấy lại `code` đã xoá comment của các hàng "
                    "đã giữ (*_full + .meta) khi `code_raw` trùng từng ký tự; hàng khác vẫn xoá comment như thường")
    a = ap.parse_args()
    pil, _ = B.load_pillars(a.pillars)
    jobs = [("ccpp_primevul_paired", "ccpp", True, lambda: read_primevul_paired(a.primevul_raw)),
            ("ccpp_primevul_unpaired", "ccpp", False, lambda: read_primevul_unpaired(a.primevul_raw)),
            ("ccpp_primevul_merged", "ccpp", True, lambda: read_primevul_merged(a.primevul_raw)),
            ("java_cleanvul_3-4", "java", True, lambda: read_cleanvul(a.cleanvul_dir, "java")),
            ("js_cleanvul_3-4", "js", True, lambda: read_cleanvul(a.cleanvul_dir, "js"))]
    if a.only:
        jobs = [j for j in jobs if j[0] in a.only.split(",")]
    report = []
    os.makedirs(os.path.join(a.out, "_dropped"), exist_ok=True)
    for name, lang, paired, reader in jobs:
        t0 = time.time()
        rows = reader()
        cache = load_stripped(a.reuse_stripped, lang) if a.reuse_stripped else {}
        todo = [i for i, r in enumerate(rows) if cache.get(r["row_id"], (None,))[0] != r["code_raw"]]
        with Pool(a.workers) as p:
            res = p.map(_strip_one, [(rows[i]["code_raw"], LX[lang]) for i in todo], chunksize=256)
        for i, (code, nsp, nch) in zip(todo, res):
            rows[i]["code"] = code; rows[i]["n_comment_spans"] = nsp; rows[i]["chars_removed"] = nch
        for i in set(range(len(rows))) - set(todo):
            _, rows[i]["code"], rows[i]["n_comment_spans"], rows[i]["chars_removed"] = cache[rows[i]["row_id"]]
        print("%s: %d hàng, xoá comment %d, lấy lại %d" % (name, len(rows), len(todo), len(rows) - len(todo)), flush=True)
        keep, drop, stat, rows_of = apply_rules(rows, lang, paired)
        keep_set = set(keep)
        recs, metas = [], []
        for u in keep:
            for i in rows_of[u]:
                r = rows[i]
                cids = B.cwe_all(r.get("cwe_raw"))
                recs.append(B.make_row(r["code"], r["label"], cids[0] if cids else None, lang, pil))
                m = {k: v for k, v in r.items() if k not in ("code",)}
                m.update({"cwe_all": cids, "cwe_labels": B.cwe_labels(r.get("cwe_raw"))})
                if not paired:
                    m["pair_id"] = None
                metas.append(m)
        write_set(a.out, name + "_full", recs, metas)
        # common / 4cwe thừa hưởng quyết định của full (tính trên các hàng ĐÃ GIỮ)
        mm = [dict(m, pair_id=m["pair_id"] or m["row_id"]) for m in metas]
        rc, mc, common_dropped = B.subset_common(recs, mm, lang, a.cwe_xml)
        if rc is not None:
            mc = [dict(m, pair_id=m["pair_id"] if paired and m["pair_id"] != m["row_id"] else None) for m in mc]
            if paired:
                rc, mc, n_orphans = keep_complete_pairs(rc, mc)
                common_dropped["(note) orphan_halves_dropped"] = n_orphans
            write_set(a.out, name + "_common", rc, mc)
        r4, m4 = B.subset_4cwe(recs, metas)
        if r4 and paired:
            r4, m4, _ = keep_complete_pairs(r4, m4)
        if r4:
            write_set(a.out, name + "_4cwe", r4, m4)
        with open(os.path.join(a.out, "_dropped", name + ".jsonl"), "w", encoding="utf-8") as fh:
            for u, (why, det) in drop.items():
                fh.write(json.dumps({"unit": u, "reason": why, "detail": det,
                                     "rows": [{k: rows[i].get(k) for k in ("row_id", "label", "code", "code_raw", "file_name", "cwe_raw", "cve", "commit_url", "orig_split")}
                                              for i in rows_of[u]]}, ensure_ascii=False) + "\n")
        n_units = len(rows_of)
        lab = collections.Counter(r["label"] for r in recs)
        rep = {"set": name, "lang": lang, "unit_type": "pair" if paired else "row", "rows_in": len(rows), "units_in": n_units,
               "units_kept": len(keep), "rows_kept": len(recs), "labels_kept": {str(k): v for k, v in sorted(lab.items())},
               "common_rows": len(rc) if rc is not None else None, "cwe4_rows": len(r4) if r4 else 0,
               "dropped_by_reason": dict(stat), "common_dropped_by": common_dropped, "seconds": round(time.time() - t0, 1)}
        report.append(rep)
        print(json.dumps(rep, ensure_ascii=False))
    rp = os.path.join(a.out, "_report.json")
    old = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else []
    names = {x["set"] for x in report}
    report = [x for x in old if x["set"] not in names] + report       # bộ không dựng lại lần này giữ nguyên báo cáo cũ
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
