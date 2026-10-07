#!/usr/bin/env python3
"""Sinh `data/check_data/` de hai ben cung soi ket qua loc.

Moi ngon ngu -> MOT log jsonl (moi cap mot dong: giu hay loai + ly do) + MOT trang html.
Rieng phan BI LOC con tach theo tung ly do de nhin nhanh.
"""
import collections, difflib, html, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_v3 import clean_full, pair_key, load

SRC = "data/sources_v2"
OUT = "data/check_data"
BOS = [("ccpp_primevul_from-paired", "ccpp", "c"),
       ("java_cleanvul_3-4", "java", "java"),
       ("js_cleanvul_3-4", "js", "js")]
REASON_NOTES = {
 "L1_mot_dong":              "Mã THÔ (trước khi xoá comment) chỉ có một dòng. Tiêu chí 1 của bộ lọc đồng tác giả (GraphTransferVD@d45c913 build_jsc.py), áp nguyên văn, chỉ cho JS. Người dùng chốt 23/09. Sau mọi tầng khác chỉ còn 1 cặp lọt: sending_profiles.min.js trong static/js/dist/.",
 "L1_duong_dan_build":       "Đường dẫn file nằm trong thư mục BUILD/VENDOR (dist/, build/, out/, node_modules/, bower_components/, vendor/, third_party/) hoặc tên *.min.js / *.bundle.js — mã do webpack/uglify sinh ra.",
 "L1_ten_ham_1-2_ky_tu":     "Hàm mở đầu bằng `function` + tên chỉ 1–2 ký tự (function t(, function ab(, function $a(). Với hàm ngắn kiểu này, lỗ hổng thường nằm ở HÀM NÓ GỌI BÊN TRONG chứ không nằm tại chỗ, nên đọc riêng nó cũng không học được gì. Loại hẳn, chấp nhận mất vài mẫu thật: đo 23/09 có 258 cặp dính mẫu này, 256 đã bị tầng khác bắt, chỉ còn 2 — một là mã nén thật đang lọt lưới, một là mã người viết (function at(target, path, update)).",
 "L1_noi_dung_mangle":       "Nội dung đã bị minify: từ 60 % trở lên số định danh (không tính từ khoá) chỉ dài 1–2 ký tự. Đổi tên định danh chính là bản chất của việc nén, nên mã kiểu function $r(e,t,n){…} không còn ngữ nghĩa để học.",
 "L1c_gen_marker":           "Chứa dấu hiệu SINH MÃ trực tiếp: __webpack_, webpackJsonp, __esModule, _classCallCheck, _interopRequireDefault, _createClass, _slicedToArray, _typeof=. Người không bao giờ gõ tay những chuỗi này.",
 "L1c_lime_artifact":        "Chứa chuỗi lime25 / limedev — hiện vật build của một dự án cụ thể.",
 "L1c_hinh_dang_kem_mangle": "Hình dạng nghi là mã nén (một dòng, hoặc dòng dài quá 250 ký tự, hoặc trung bình trên 120 ký tự mỗi dòng, hoặc hơn 3 dấu chấm phẩy mỗi dòng, hoặc tên hàm dài không quá 2 ký tự) VÀ có bằng chứng mangle từ 40 % trở lên. Hình dạng một mình không đủ — xem RULES.md.",
 "L2_ham_rac":               "Hàm rác: thân hàm rỗng (khai báo abstract/interface, hoặc thân chỉ có {}), hoặc dưới 10 token mã — quá ngắn để mang thông tin về lỗ hổng.",
 "L2_khong_co_than":         "Không có thân hàm — khai báo abstract/interface, hoặc thân chỉ có {}. Không có bằng chứng lỗ hổng nằm trong chính hàm này.",
 "L5_cap_ghep_sai":          "Hai nửa mang tên hàm khác nhau, VÀ cả hai tên đều xuất hiện độc lập ở bản ghi khác — tức là hai hàm riêng biệt, chắc chắn không phải một cặp (bản lỗi, bản vá). Ví dụ _TIFFmalloc (nhãn 1) ghép với _TIFFrealloc (nhãn 0): cả hai hàm đều tồn tại trong libtiff.",
 "L5_nghi_ghep_sai_hoac_doi_ten": "Hai nửa khác tên hàm, nhưng một trong hai tên chỉ xuất hiện ở đúng bản ghi này — nên KHÔNG kết luận được: có thể là ghép sai (dcn20 ↔ dcn10_clock_source_create, hai chip khác nhau), cũng có thể là ĐỔI TÊN thật trong bản vá (x2c → _x2c, fill_threshhold_buffer → fill_threshold_buffer sửa lỗi chính tả). Đã đo: độ giống thân hàm KHÔNG tách được hai trường hợp này (_TIFFmalloc/_TIFFrealloc giống 0,615 trong khi có bản vá thật chỉ giống 0,12). ĐANG LOẠI, cần người xem lại.",
 "L5_cap_ghep_sai_CU":       "Hai nửa mang TÊN HÀM khác nhau nên không phải một cặp (bản lỗi, bản vá). Lỗi xếp dòng trong dữ liệu PrimeVul gốc: một commit vá nhiều hàm thì thứ tự bị xen kẽ, ví dụ _TIFFmalloc (nhãn 1) đứng cạnh _TIFFrealloc (nhãn 0). Để lại thì pair loss được dạy hai hàm không liên quan là một cặp.",
 "L3_trung_lap":             "Cặp trùng với một cặp đã giữ, so sau khi bỏ hết khoảng trắng — nên bắt được cả bản sao chỉ khác thụt lề (cùng một đoạn mã bị crawl hai lần, hoặc hai repo dùng chung đúng đoạn mã đó).",
 "L2_than rong":             "Thân hàm rỗng — khai báo abstract/interface, hoặc thân chỉ có {}. Không có bằng chứng lỗ hổng nằm trong chính hàm này.",
 "L2_qua ngan (<10 token)":  "Dưới 10 token mã — quá ngắn để mang thông tin về lỗ hổng.",
 "L3_trung_chinh_xac":       "Cả cặp trùng nguyên văn với một cặp đã được giữ trước đó.",
 "L3_hai_nua_giong_het":     "Hai nửa của cặp giống hệt nhau nguyên văn, nghĩa là hai nhãn ngược nhau gán trên cùng một chuỗi — không thể học được.",
 "L3b_nua_cap_da_xuat_hien": "Một nửa của cặp đã xuất hiện ở một cặp khác: cùng một hàm lỗi nhưng có hai commit vá khác nhau.",
 "L4_trung_sau_chuan_hoa":   "Sau khi phi-ngữ-nghĩa-hoá (định danh → ID1, ID2…; số → N; chuỗi → S) thì trùng NGUYÊN VĂN với một cặp đã giữ. Đây là clone sao-chép, ví dụ do_siocgstamp và do_siocgstampns — so nguyên văn không bắt được.",
}

REASON_LABELS = {
 "L1_duong_dan_build":       "L1 · đường dẫn build",
 "L1_noi_dung_mangle":       "L1 · nội dung đã minify",
 "L1_ten_ham_1-2_ky_tu":     "L1 · tên hàm 1–2 ký tự",
 "L1_mot_dong":              "L1 · một dòng (mã thô)",
 "L1c_gen_marker":           "L1c · dấu hiệu sinh mã",
 "L1c_lime_artifact":        "L1c · hiện vật build",
 "L1c_hinh_dang_kem_mangle": "L1c · hình dạng kèm mangle",
 "L2_ham_rac":               "L2 · hàm rác (rỗng hoặc quá ngắn)",
 "L2_khong_co_than":         "L2 · không có thân hàm",
 "L5_cap_ghep_sai":          "L5 · cặp ghép sai (chắc chắn)",
 "L5_nghi_ghep_sai_hoac_doi_ten": "L5? · khác tên hàm — GHÉP SAI hay ĐỔI TÊN, cần review",
 "L3_trung_lap":             "L3 · trùng lặp (sau khi bỏ khoảng trắng)",
 "L2_than rong":             "L2 · thân hàm rỗng",
 "L2_qua ngan (<10 token)":  "L2 · quá ngắn (<10 token)",
 "L3_trung_chinh_xac":       "L3 · trùng chính xác",
 "L3_hai_nua_giong_het":     "L3 · hai nửa giống hệt",
 "L3b_nua_cap_da_xuat_hien": "L3b · nửa cặp đã xuất hiện",
 "L4_trung_sau_chuan_hoa":   "L4 · trùng sau chuẩn hoá",
}
def reason_label(w): return REASON_LABELS.get(w, w)

def esc(s): return html.escape(s or "")

PAGE = """<!doctype html><meta charset="utf-8"><title>check_data · %(lang)s</title>
<style>
:root{--bg:#fbfaf8;--fg:#1d1b19;--mut:#6d6862;--line:#e3ded7;--keep:#2f7d4f;--drop:#b3402f;--code:#f4f1ec;
--dbg:#fdeceb;--ibg:#eaf6ed;--dhi:#f9c3bd;--ihi:#b6e6c4;--dfg:#b3402f;--ifg:#2f7d4f}
@media(prefers-color-scheme:dark){:root{--bg:#16150f;--fg:#ece8e1;--mut:#9a948c;--line:#33302a;--keep:#7fc79a;--drop:#e8907f;--code:#1f1e17;
--dbg:#3a201c;--ibg:#1c3326;--dhi:#6e3229;--ihi:#2c5c3d;--dfg:#e8907f;--ifg:#7fc79a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:16px 24px 12px;border-bottom:1px solid var(--line);background:var(--bg)}
.stick{position:sticky;top:0;z-index:20;background:var(--bg);box-shadow:0 1px 0 var(--line)}
h1{margin:0 0 4px;font-size:19px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px}
nav{padding:9px 24px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px;
background:var(--bg);max-height:26vh;overflow-y:auto}
nav a.cur{border-color:var(--fg)}
h2{scroll-margin-top:34vh}section{scroll-margin-top:34vh}
nav a{padding:4px 10px;border:1px solid var(--line);border-radius:999px;text-decoration:none;color:var(--fg);font-size:12px}
nav a:hover{border-color:var(--mut)}
main{padding:20px 24px;max-width:1400px}
section{margin:0 0 34px}
h2{font-size:15px;margin:0 0 4px;display:flex;align-items:baseline;gap:10px}
h2 .n{color:var(--mut);font-weight:400;font-size:13px}
.why{color:var(--mut);font-size:13px;margin:0 0 12px;max-width:80ch}
.pair{border:1px solid var(--line);border-radius:8px;margin:0 0 12px;overflow:hidden}
.meta{padding:6px 12px;background:var(--code);border-bottom:1px solid var(--line);
font-size:12px;color:var(--mut);display:flex;flex-wrap:wrap;gap:14px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.col{background:var(--bg);min-width:0}
.tag{padding:4px 12px;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);
display:flex;justify-content:space-between;gap:10px;align-items:center}
.rs{text-transform:none;letter-spacing:0;font-size:11px;padding:1px 8px;border-radius:999px;
border:1px solid var(--line);white-space:nowrap}
.rs.x{color:var(--drop);border-color:var(--dhi);background:var(--dbg)}
.rs.k{color:var(--keep);border-color:var(--ihi);background:var(--ibg)}
pre{margin:0;padding:10px 12px;overflow-x:auto;background:var(--code);
font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre;tab-size:4}
.keep h2{color:var(--keep)}.drop h2{color:var(--drop)}
.kc{color:var(--keep)}.dc{color:var(--drop)}
.l{display:block;padding:0 2px;border-radius:2px}
.l.del{background:var(--dbg)}.l.ins{background:var(--ibg)}
.l.del::before{content:"\2212 ";color:var(--dfg);font-weight:700}
.l.ins::before{content:"+ ";color:var(--ifg);font-weight:700}
.l.eq::before{content:"  ";color:transparent}
mark.d{background:var(--dhi);color:inherit;border-radius:2px;padding:0 1px}
mark.i{background:var(--ihi);color:inherit;border-radius:2px;padding:0 1px}
.legend{font-size:12px;color:var(--mut);margin:0 0 14px;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
.legend i{font-style:normal;padding:1px 7px;border-radius:3px}
.nodiff{padding:5px 12px;font-size:12px;color:var(--mut);background:var(--code);border-top:1px solid var(--line)}
footer{padding:20px 24px;color:var(--mut);font-size:12px;border-top:1px solid var(--line)}
a{color:inherit}
</style>
<div class="stick"><header><h1><a href="../index.html" style="text-decoration:none">check_data</a> · %(lang)s</h1>
<div class="sub">%(n_in)s cặp vào · <b class="kc">%(n_keep)s giữ</b> · <b class="dc">%(n_drop)s loại</b> · log: <code>%(log)s</code>
&nbsp;·&nbsp; <span class="legend" style="display:inline-flex;margin:0">
<i style="background:var(--dbg);color:var(--dfg)">− bản lỗi</i><i style="background:var(--ibg);color:var(--ifg)">+ bản vá</i>
<i style="background:var(--dhi)">ký tự đổi</i></span></div></header>
<nav>%(nav)s</nav></div><main>%(body)s</main>
<footer>Mỗi khối dưới đây là MỘT CẶP (bản lỗi / bản vá). Loại thì loại cả cặp.
Trang này hiện toàn bộ phần bị loại, và %(nsample)s cặp mẫu của phần giữ lại. Dữ liệu đầy đủ ở các file jsonl.</footer>
"""

_WORD = re.compile(r"\w+|\s+|.")

def _hl(a, b):
    """Diff theo TU trong MOT cap dong (theo ky tu thi highlight bi vun, kho doc)."""
    A, B = _WORD.findall(a), _WORD.findall(b)
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    L = R = ""
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            L += esc("".join(A[i1:i2])); R += esc("".join(B[j1:j2]))
        else:
            if i2 > i1: L += '<mark class="d">%s</mark>' % esc("".join(A[i1:i2]))
            if j2 > j1: R += '<mark class="i">%s</mark>' % esc("".join(B[j1:j2]))
    return L, R


def _hl_char(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    L = R = ""
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            L += esc(a[i1:i2]); R += esc(b[j1:j2])
        else:
            if i2 > i1: L += '<mark class="d">%s</mark>' % esc(a[i1:i2])
            if j2 > j1: R += '<mark class="i">%s</mark>' % esc(b[j1:j2])
    return L, R


def diff_cols(a, b):
    """Diff theo DONG giua ban loi va ban va. Tra ve (pre_trai, pre_phai, co_thay_doi)."""
    A, B = a.split("\n"), b.split("\n")
    L, R, changed = [], [], False
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i1, i2): L.append('<span class="l eq">%s</span>' % esc(A[k]))
            for k in range(j1, j2): R.append('<span class="l eq">%s</span>' % esc(B[k]))
        elif tag == "replace":
            changed = True
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                la = A[i1 + k] if i1 + k < i2 else None
                lb = B[j1 + k] if j1 + k < j2 else None
                if la is not None and lb is not None:
                    x, y = _hl(la, lb)
                    L.append('<span class="l del">%s</span>' % x)
                    R.append('<span class="l ins">%s</span>' % y)
                elif la is not None:
                    L.append('<span class="l del">%s</span>' % esc(la))
                else:
                    R.append('<span class="l ins">%s</span>' % esc(lb))
        elif tag == "delete":
            changed = True
            for k in range(i1, i2): L.append('<span class="l del">%s</span>' % esc(A[k]))
        else:
            changed = True
            for k in range(j1, j2): R.append('<span class="l ins">%s</span>' % esc(B[k]))
    return "".join(L), "".join(R), changed


def block(pair, why, note=""):
    a = b = None
    for r in pair:
        if r["label"] == 1 and a is None: a = r
        elif b is None: b = r
    a = a or pair[0]
    bcode = b.get("code", "") if b else ""
    meta = ['<span>pair <code>%s</code></span>' % esc(str(a.get("pair_id"))[:34])]
    if a.get("file_name"): meta.append("<span>%s</span>" % esc(a["file_name"][:90]))
    if a.get("cwe"): meta.append("<span>%s</span>" % esc(str(a["cwe"])))
    if note: meta.append("<span>%s</span>" % esc(note))
    if b is None:
        left, right, changed = esc(a["code"]), "(khong co nua kia)", None
    else:
        left, right, changed = diff_cols(a["code"], bcode)
    foot = ""
    if changed is False:
        foot = '<div class="nodiff">hai nửa GIỐNG HỆT nhau — không có thay đổi nào</div>'
    if why == "giu":
        badge = '<span class="rs k">GIỮ LẠI</span>'
    else:
        badge = '<span class="rs x">loại bởi: %s</span>' % esc(reason_label(why))
    return ('<div class="pair"><div class="meta">%s</div><div class="cols">'
            '<div class="col"><div class="tag"><span>bản lỗi · label 1</span>%s</div><pre>%s</pre></div>'
            '<div class="col"><div class="tag"><span>bản vá · label 0</span>%s</div><pre>%s</pre></div>'
            '</div>%s</div>') % ("".join(meta), badge, left, badge, right, foot)


def main():
    os.makedirs(OUT, exist_ok=True)
    for base, lang, lx in BOS:
        recs, metas = load(SRC, base, "full")
        keep, cnt, _ex, pid, drop = clean_full(recs, metas, lang, lx, 10)
        d = os.path.join(OUT, lang); os.makedirs(os.path.join(d, "bi_loc"), exist_ok=True)

        by = collections.defaultdict(list)
        for i, (r, m) in enumerate(zip(recs, metas)):
            by[pid[i]].append({"code": r["code"], "label": r["label"], "lang": lang,
                               "cwe": r.get("cwe"), "pair_id": pid[i],
                               "file_name": m.get("file_name"), "score": m.get("score"),
                               "commit_url": m.get("commit_url")})

        # ---- MOT log cho ca ngon ngu: moi CAP mot dong
        with open(os.path.join(d, "%s.log.jsonl" % lang), "w", encoding="utf-8") as fh:
            for p, rows in by.items():
                why = drop.get(p)
                fh.write(json.dumps({
                    "pair_id": p, "lang": lang, "ket_qua": "loai" if why else "giu",
                    "tang": (why or "").split("_")[0] or None,
                    "ly_do": why, "ten_hien": reason_label(why) if why else None,
                    "giai_thich": REASON_NOTES.get(why, ""),
                    "file_name": rows[0].get("file_name"), "cwe": rows[0].get("cwe"),
                    "so_nua": len(rows),
                    "code_ban_loi": next((r["code"] for r in rows if r["label"] == 1), None),
                    "code_ban_va":  next((r["code"] for r in rows if r["label"] == 0), None),
                }, ensure_ascii=False) + "\n")

        with open(os.path.join(d, "giu_lai.jsonl"), "w", encoding="utf-8") as fh:
            for p in keep:
                for r in by[p]:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        per = collections.defaultdict(list)
        for p, why in drop.items():
            per[why].append(p)
        for why, ps in per.items():
            fn = why.replace(" ", "_").replace("(", "").replace(")", "").replace("<", "d")
            with open(os.path.join(d, "bi_loc", fn + ".jsonl"), "w", encoding="utf-8") as fh:
                for p in ps:
                    for r in by[p]:
                        fh.write(json.dumps(dict(r, ly_do=why, ten_hien=reason_label(why),
                                                 giai_thich=REASON_NOTES.get(why, "")),
                                            ensure_ascii=False) + "\n")

        # ---- trang html
        NS = 40
        secs, nav = [], []
        for why in sorted(per, key=lambda w: -len(per[w])):
            aid = why.replace(" ", "_").replace("(", "").replace(")", "").replace("<", "d")
            nav.append('<a href="#%s">%s <b>%d</b></a>' % (aid, esc(reason_label(why)), len(per[why])))
            secs.append('<section class="drop" id="%s"><h2>%s <span class="n">· %d cặp bị loại</span></h2>'
                        '<p class="why">%s</p>%s</section>'
                        % (aid, esc(reason_label(why)), len(per[why]), esc(REASON_NOTES.get(why, "")),
                           "".join(block(by[p], why) for p in per[why])))
        ks = sorted(keep)[:NS]
        nav.append('<a href="#giu">giữ lại <b>%d</b></a>' % len(keep))
        secs.append('<section class="keep" id="giu"><h2>giữ lại <span class="n">· %d cặp, hiện %d mẫu đầu</span></h2>'
                    '<p class="why">Đây là dữ liệu sẽ dùng để huấn luyện. Soi xem có cặp nào đáng lẽ phải loại không. '
                    'Đầy đủ ở <code>giu_lai.jsonl</code>.</p>%s</section>'
                    % (len(keep), len(ks), "".join(block(by[p], "giu") for p in ks)))
        with open(os.path.join(d, "%s.html" % lang), "w", encoding="utf-8") as fh:
            fh.write(PAGE % {"lang": lang, "n_in": len(by), "n_keep": len(keep),
                             "n_drop": len(drop), "log": "%s.log.jsonl" % lang,
                             "nav": "".join(nav), "body": "".join(secs), "nsample": len(ks)})
        print("  %-6s %5d cặp → giữ %5d · loại %4d" % (lang, len(by), len(keep), len(drop)))


if __name__ == "__main__":
    main()
