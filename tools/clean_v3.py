#!/usr/bin/env python3
"""Loc `data/sources_v2` -> `data/sources_v3`: bo build artifact, ham rac, va trung lap.

Nguoi dung neu 22/09/2026. Bon loai loc, theo dung thu tu:

  L1 BUILD/MINIFIED  ma sinh ra tu webpack/uglify: ten file o `dist/ build/ vendor/
                     node_modules/ *.min.js`, HOAC noi dung da bi mangle (>=60% dinh
                     danh dai 1-2 ky tu). Ma nay khong con ngu nghia de hoc.
  L2 HAM RAC         than ham rong, hoac qua ngan de mang thong tin lo hong.
  L3 TRUNG CHINH XAC hai hang co `code` giong het nhau.
  L4 TRUNG SAU PHI-NGU-NGHIA-HOA
                     doi moi dinh danh -> ID1, ID2..., so -> N, chuoi -> S, rom chu ->
                     C, roi ep khoang trang. Hai ham khac ten bien nhung cung khung
                     suon se trung o buoc nay. Day la loai trung ma L3 KHONG bat duoc.

QUY TAC CAP: moi hang thuoc mot cap (ban loi, ban va). Bo mot nua thi BO CA CAP —
neu khong se lech nhan va pair loss mat doi tac. Nguoc lai, hai nua cua CUNG mot cap
gan giong nhau la CHU Y (do la ca kho chung minh mo hinh hoc dac trung lo hong chu
khong hoc mau van ban), KHONG bao gio bi coi la trung lap.

KHONG dung cham cach chia train/val: nguoi dung yeu cau giu chia NGAU NHIEN THEO HANG.
"""
import argparse, collections, json, os, re, sys

# --------------------------------------------------------------- phi-ngu-nghia-hoa
KW = {
 "c": set("""auto break case char const continue default do double else enum extern float for goto
 if inline int long register restrict return short signed sizeof static struct switch typedef union
 unsigned void volatile while _Bool _Complex bool true false NULL class public private protected
 virtual template typename namespace using new delete this operator friend explicit throw try catch
 nullptr constexpr static_cast dynamic_cast reinterpret_cast const_cast""".split()),
 "java": set("""abstract assert boolean break byte case catch char class const continue default do
 double else enum extends final finally float for goto if implements import instanceof int interface
 long native new package private protected public return short static strictfp super switch
 synchronized this throw throws transient try void volatile while true false null var record yield
 sealed permits""".split()),
 "js": set("""await break case catch class const continue debugger default delete do else export
 extends finally for function if import in instanceof let new return super switch this throw try
 typeof var void while with yield async of static get set true false null undefined""".split()),
}
TOK = re.compile(r"""
    (?P<str>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)
  | (?P<num>\b\d[\w.']*\b)
  | (?P<id>[A-Za-z_$][A-Za-z0-9_$]*)
  | (?P<op>\S)
""", re.X | re.S)

def tokens(code):
    return [(m.lastgroup, m.group()) for m in TOK.finditer(code)]

def alpha_norm(code, lang):
    """Doi ten moi dinh danh/hang so -> ky hieu vi tri. Giu nguyen tu khoa va toan tu."""
    kw = KW.get(lang, KW["c"])
    out, ren = [], {}
    for kind, t in tokens(code):
        if kind == "str":
            out.append("S")
        elif kind == "num":
            out.append("N")
        elif kind == "id":
            if t in kw:
                out.append(t)
            else:
                if t not in ren:
                    ren[t] = "ID%d" % (len(ren) + 1)
                out.append(ren[t])
        else:
            out.append(t)
    return " ".join(out)

def n_code_tokens(code):
    return sum(1 for k, _ in tokens(code) if k in ("id", "num", "str"))

# ------------------------------------------------------------------------ L1, L2
# CHI duong dan san pham BUILD. Duong dan VENDOR (node_modules/ vendor/ third_party/)
# DA BO khoi luat: nguoi dung chot 23/09 — "model can hoc lo hong chu khong quan tam no
# o repo nao". Ma thu vien ben thu ba do NGUOI VIET, co ban va that thi van la du lieu tot.
BUILD = re.compile(r"(\.min\.js$|-min\.js$|\.bundle\.|"
                   r"(^|/)(dist|build|out|_build)/)", re.I)

# Ngoai le: `packages/node_modules/` la kieu bo tri MONOREPO — du an tu dat cac goi cua
# CHINH NO o day de Node tim thay khi phat trien, khong phai thu vien tai ve. Vi du da
# kiem 22/09: packages/node_modules/@node-red/runtime/... la ma nguon cua node-red, vet
# la commit nam trong chinh repo ho (node-red/node-red@74db3e17, CVE-2021-21298 CWE-22,
# va `fspath.join` -> `fspath.resolve(fspath.join(...))`) va con nguyen 1105 ky tu comment.
# Khac han Modules/.../node_modules/socket.io-parser/build/cjs/... — thu vien tai ve, va
# con nam trong thu muc build/ nen van bi bat boi ve `build` cua BUILD.
MONOREPO = re.compile(r"(^|/)packages/node_modules/", re.I)


def is_build_path(fn, code=None, lang="js"):
    """`fn` khop mau build. Rieng TEN *.min.js thi phai co NOI DUNG xac nhan, vi ten
    file noi doi: do 22/09 co 23/258 hang mang ten *.min.js ma noi dung khong he nen
    (`dwz.min.js` co 239 dinh danh, chi 17 % ngan; `jquery.uploadfile.min.js` tuong tu)."""
    fn = fn or ""
    if MONOREPO.search(fn):
        return False
    if not BUILD.search(fn):
        return False
    if code is None:
        return True
    # Duong dan chi la GOI Y. Phai co bang chung trong NOI DUNG thi moi bo — nguoi dung
    # chot 23/09 qua ca `cleanvul:f492d101590f7938` (socket.io-parser/build/cjs/index.js):
    # do la ban bien dich TS->JS, doc duoc, va vá CVE-2023-36485 that su nam trong ham.
    # Dung DUNG dieu kien nhu tang L1: mangle VA hinh dang (mangle mot minh bo nham ma
    # nguoi viet ten ngan), hoac dau vet trinh sinh ma.
    return ((is_mangled(code, lang, min_ids=4) and shape_compressed(code))
            or bool(is_generated(code, lang)))

def is_mangled(code, lang, min_ids=8):
    """CHI dung cho js. `min_ids` ha xuong 4 de bat ham nen ngan nhu
    `function ae(e){return new Pr(e)}` (chi 4 dinh danh)."""
    kw = KW.get(lang, KW["c"])
    ids = [t for k, t in tokens(code) if k == "id" and t not in kw]
    if len(ids) < min_ids:
        return False
    return sum(1 for t in ids if len(t) <= 2) / len(ids) >= 0.60

# --- L1c: luat cua dong tac gia (GraphTransferVD@d45c913 scripts/build_jsc.py:28) ---------
# Ho dung 8 tieu chi tren MA THO. Do lai 22/09 tren chinh du lieu nay: ap NGUYEN XI thi
# tieu chi hinh dang (`long-line>250`, `dense>120`, `;/dong>3`, `one-liner`) co ty le
# duong tinh GIA rat cao — 44/48 hang js bi bo la ma NGUOI VIET doc duoc chi co mot dong
# dai; tren C no bo nham 43 ham (`bool const_item() const { return true; }`) va tren Java
# 100 ham (khai bao interface mot dong). Ho chi chay no cho JS nen khong gap dieu do.
#
# Nen tach lam hai: dau hieu SINH MA (webpack/babel/lime) la bang chung truc tiep -> bo
# ngay; dau hieu HINH DANG chi la goi y -> phai KEM bang chung mangle (>=0.40) moi bo.
# Do lai: luat ghep nay bat du 4/4 ca THAT ma minh dang sot, va 0/46 ca GIA.
GEN_MARKER = re.compile(r"__webpack_|webpackJsonp|__esModule|_interopRequireDefault|"
                        r"_classCallCheck|_createClass|_typeof2?=|_slicedToArray|"
                        # dau vet cua TRINH SINH MA: PEG.js, Closure Compiler, Emscripten.
                        # `function peg$parsepartial(){ var s0,s1,...; peg$currPos ... }` —
                        # khong nguoi nao viet ten nhu vay. Do 23/09: bat dung 26 hang cua
                        # dist/dust-full.js, KHONG dinh hang nguoi viet nao.
                        r"peg\$|\$jscomp|goog\.provide|Module\[\"asm\"\]|wasmExports")

def mangle_frac(code, lang):
    kw = KW.get(lang, KW["c"])
    ids = [t for k, t in tokens(code) if k == "id" and t not in kw]
    return sum(1 for t in ids if len(t) <= 2) / len(ids) if len(ids) >= 8 else 0.0

def is_generated(code, lang):
    """Chay tren MA THO (`code_raw`) — dung thu tu ho neu ra: loc truoc khi xoa comment."""
    if GEN_MARKER.search(code):
        return "gen_marker"
    if "lime25" in code or "limedev" in code:
        return "lime_artifact"
    lines = code.split("\n"); nl = len(lines)
    shape = (nl == 1
             or max((len(x) for x in lines), default=0) > 250
             or len(code) / max(1, nl) > 120
             or code.count(";") / max(1, nl) > 3
             or bool(re.match(r"\s*function\s+[\w$]{1,2}\s*\(", code)))
    if shape and mangle_frac(code, lang) >= 0.40:
        return "hinh_dang_kem_mangle"
    return None

def shape_compressed(code):
    """Dau hieu HINH DANG cua ma nen. Mot minh KHONG du de ket luan — phai di kem
    bang chung mangle. Dung de doi chung cho ca luat mangle lan luat ten *.min.js."""
    ls = code.split("\n")
    nl = len(ls)
    return (nl == 1
            or max((len(x) for x in ls), default=0) > 250
            or len(code) / max(1, nl) > 120
            or code.count(";") / max(1, nl) > 3)


MINNAME = re.compile(r"(\.min\.js$|-min\.js$)", re.I)

# Ham mo dau bang `function` + ten 1-2 ky tu: `function t(`, `function ab(`, `function $a(`.
# Nguoi dung chot 23/09 (theo tieu chi 8 cua GraphTransferVD scripts/build_jsc.py:28):
# bo han, chap nhan mat vai sample that. Ly do: voi ham ngan the nay lo hong thuong nam o
# HAM NO GOI BEN TRONG chu khong nam tai cho, nen doc rieng no cung khong hoc duoc gi.
# Do 23/09: 258 cap dinh mau nay, 256 da bi cac tang khac bat; chi con 2 — mot la
# `function t(e,n){var i=this;for(var o in s(this,t),...}` (nen that, dang lot luoi) va mot
# la `function at(target, path, update)` (ma nguoi viet). Doi mot lay mot.
COMPRESS_FN = re.compile(r"^\s*(?:async\s+)?function\s*\*?\s*[\w$]{1,2}\s*\(")


def no_body(code):
    """Khong co than ham: khai bao abstract/interface, hoac than chi co {}."""
    if "{" not in code:
        return True
    b = code[code.find("{") + 1: code.rfind("}")] if "}" in code else ""
    return not b.strip()


def is_junk(code, min_tok):        # giu lai cho cong cu cu, khong con dung trong clean_full
    body = code[code.find("{") + 1: code.rfind("}")] if "{" in code and "}" in code else code
    if not body.strip():
        return True, "than rong"
    if n_code_tokens(code) < min_tok:
        return True, "qua ngan (<%d token)" % min_tok
    return False, ""


# --------------------------------------------------------------------------- main
def pair_key(m, i, lang):
    pid = m.get("pair_id")
    if not pid:                      # PrimeVul: hai dong lien tiep la mot cap
        pid = "primevul:%s:%d" % (m.get("split_goc", "?"), m.get("idx_goc", i) // 2)
    return pid


def load(src_dir, base, tag):
    pj = os.path.join(src_dir, "%s_%s.jsonl" % (base, tag))
    pm = os.path.join(src_dir, "%s_%s.meta.jsonl" % (base, tag))
    if not os.path.exists(pj):
        return None, None
    return ([json.loads(l) for l in open(pj, encoding="utf-8")],
            [json.loads(l) for l in open(pm, encoding="utf-8")])


# Token dung truoc dau ( nhung KHONG phai ten ham: annotation cua Java va macro thuoc
# tinh cua kernel. Do 22/09: `fname` cu bat `@SuppressWarnings` va `__acquires` lam ten
# ham, khien L5 so nham annotation voi nhau.
_NOTNAME = {"__acquires", "__releases", "__must_hold", "__attribute__", "__init", "__exit",
            "__user", "__kernel", "__weak", "__always_inline", "if", "for", "while",
            "switch", "return", "sizeof", "catch", "synchronized"}

def fname(code):
    """Ten ham, hoac None neu khong xac dinh duoc (chu ky bi cat cut / chi con annotation).
    Tra None thi L5 BO QUA cap do — tha khong ket luan con hon ket luan sai."""
    head = code[:code.find("{")] if "{" in code else code   # chi tim trong CHU KY
    for m in re.finditer(r"(@?)([A-Za-z_][\w:~]*)\s*\(", head):
        at, name = m.group(1), m.group(2)
        if at or name in _NOTNAME:      # @Annotation( hoac macro thuoc tinh -> bo qua
            continue
        return name
    return None


_NOWS = lambda x: re.sub(r"\s+", "", x)


def clean_full(recs, metas, lang, lx, min_tok):
    """Loc bo `full`. Tra ve (tap pair_id GIU LAI, dem, vi du, pid, drop).

    DON VI LOC LA CAP, luon luon. Khong bao giờ bỏ lẻ một nửa — nửa còn lại sẽ
    mất đối tác của pair loss và làm lệch nhãn. Thứ tự tầng (người dùng chốt 22/09):

      L3  trùng lặp        -> ƯU TIÊN ĐẦU: bỏ bản sao trước khi xét gì khác
      L5  cặp ghép sai     -> hai nửa là hai hàm khác nhau, không phải (lỗi, vá)
      L1  mã sinh/nén
      L2  hàm không có thân
    """
    n0 = len(recs)
    pid = [pair_key(m, i, lx) for i, m in enumerate(metas)]
    drop, cnt, ex = {}, collections.Counter(), collections.defaultdict(list)

    def kill(p, why, sample=None):
        if p in drop:
            return
        drop[p] = why
        cnt[why] += 1
        if sample is not None and len(ex[why]) < 3:
            ex[why].append(sample)

    by_pair = collections.defaultdict(list)
    for i in range(n0):
        by_pair[pid[i]].append(i)

    # ---- L3 TRUNG LAP (uu tien dau tien) --------------------------------------
    # So sau khi BO HET KHOANG TRANG, nen bat duoc ca ban sao chi khac thut le.
    # Do 22/09: cach nay bat tron 5 nhom ma phep phi-ngu-nghia-hoa goi la "trung
    # that" (SPL_METHOD, dex_parse_debug_item, phar_parse_zipfile, CSoundFile::
    # GetLength, S_AL_Init — dinh danh, so, chuoi deu giong het). Con 19 nhom kia
    # la HAM ANH EM cung mot commit (dce80/dce100/dcn10/dcn20_clock_source_create,
    # do_siocgstamp/do_siocgstampns) — moi cai la mot thuc the lo hong rieng, KHONG
    # phai ban sao, nen KHONG bo. Vi vay khong con tang phi-ngu-nghia-hoa nua.
    seen = {}
    for p, idxs in sorted(by_pair.items()):
        codes = [recs[i]["code"] for i in idxs]
        if len(idxs) > 1 and len(set(codes)) == 1:
            kill(p, "L3_hai_nua_giong_het", codes[0][:110])
            continue
        k = "\x00".join(sorted(_NOWS(c) for c in codes))
        if k in seen:
            # Ghi ro cap NAO duoc giu thay the — de nguoi doc khong tuong la mat du lieu.
            kill(p, "L3_trung_lap", "trung voi %s (cap do DA DUOC GIU)" % seen[k])
        else:
            seen[k] = p

    # ---- L5 CAP GHEP SAI: DA BO HAN (nguoi dung chot 23/09) -------------------
    # PrimeVul co 57/4704 cap (1,2 %) ma hai nua la hai ham khac nhau — mot commit va
    # nhieu ham thi thu tu dong bi xen ke. Truoc day loai chung.
    # Nguoi dung chot 23/09: GIU. Ly do: nhan cua TUNG HANG van dung va lo hong van
    # that, do la thu mo hinh can hoc. Chi rieng viec ghep cap la lech, va cai gia do
    # nho hon cai gia mat du lieu that (trong 57 cap co ca doi ten that: x2c -> _x2c,
    # fill_threshhold_buffer -> fill_threshold_buffer).
    # Giu lai `fname()` vi cong cu khac con dung.

    # ---- L1 MA SINH / MA NEN ---------------------------------------------------
    # mangle CHI ap cho js: minify la chuyen cua web. C/C++ dung ten bien mot chu
    # (`p, s, d, l`) la THANH NGU, khong phai nen — do 22/09 luat cu bo nham 65 ham
    # C va 8 ham Java.
    for i, (r, m) in enumerate(zip(recs, metas)):
        if pid[i] in drop:
            continue
        fn = m.get("file_name") or ""
        # Luat duong dan CHI ap cho js. Nguoi dung neu 22/09: build cua Java ra .class/.jar,
        # cua C ra .o/.so — nen mot file .java hay .c LUON la ma nguon, du nam o thu muc ten
        # gi. Chi rieng js moi co san pham build TRUNG DUOI voi nguon, nen duong dan la tin
        # hieu duy nhat. Do lai: luat cu bo nham CA 9 file Java, trong do 6 la ma nguon cua
        # chinh Bazel (`com/google/devtools/BUILD/lib/...` — `build` la TEN PACKAGE Java),
        # 2 la module ten `build-caching` co `src/main/java` ben trong, 1 la ma nguon Shiro.
        if lx == "js" and is_build_path(fn, r["code"], lx):
            kill(pid[i], "L1_duong_dan_build", fn)
        elif lx == "js" and COMPRESS_FN.match(r["code"]):
            kill(pid[i], "L1_ten_ham_1-2_ky_tu", r["code"][:110])
        elif lx == "js" and is_mangled(r["code"], lx, min_ids=4) and shape_compressed(r["code"]):
            # Phai KEM hinh dang. Do 22/09 (dong tac gia neu): mangle mot minh bo nham
            # ma NGUOI VIET theo phong cach ten ngan — `minimatch.js`, `ecc/math.js`,
            # `moment/from-string.js`, `semver-regex`, `undici/headers.js`:
            #     function ext (a, b) { a = a || {} b = b || {} var t = {} ...
            # 74 % dinh danh ngan nhung xuong dong va cach chu dang hoang.
            kill(pid[i], "L1_noi_dung_mangle", r["code"][:110])
        elif lx == "js":
            # Cung ly le: marker webpack/babel va dau hieu hinh dang deu la chuyen cua js.
            # Tren java no bo nham `static void loadComments(){ c.addComment("...") ... }`
            # — chuoi comment dai kich hoat "dong >250", bien `c` lap nhieu day mangle len 40%.
            raw = m.get("code_raw") or r["code"]
            g = is_generated(raw, lx)
            if g:
                kill(pid[i], "L1c_" + g, (m.get("code_raw") or "")[:110])
            elif len(raw.split("\n")) == 1:
                # MOT DONG tren MA THO — tieu chi 1 cua GraphTransferVD@d45c913 build_jsc.py,
                # ap nguyen van. Nguoi dung chot 23/09: "cai oneline thi bo". Do 23/09: sau
                # moi tang khac chi con dung 1 cap lot (static/js/dist/app/sending_profiles.min.js,
                # `function sendTestEmail(){var o=[];$.each(...)` — ten thuoc tinh dai keo ty le
                # mangle xuong duoi nguong nen hai luat mangle khong bat). Chi ap cho js.
                kill(pid[i], "L1_mot_dong", raw[:110])

    # ---- L2 HAM KHONG CO THAN --------------------------------------------------
    # Chi bat "khong co than", KHONG dung nguong do dai: nguong <10 token bo nham
    # _TIFFmalloc(tmsize_t s){ return malloc((size_t)s); } — ngan nhung co noi dung
    # va co lo hong tran so nguyen that.
    for i, r in enumerate(recs):
        if pid[i] in drop:
            continue
        if no_body(r["code"]):
            kill(pid[i], "L2_khong_co_than", r["code"][:110])

    keep = {p for p in by_pair if p not in drop}
    return keep, cnt, ex, pid, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/sources_v2")
    ap.add_argument("--out_dir", default="data/sources_v3")
    ap.add_argument("--min_tok", type=int, default=10)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if not a.dry:
        os.makedirs(a.out_dir, exist_ok=True)

    BOS = [("ccpp_primevul_from-paired", "ccpp", "c"),
           ("java_cleanvul_3-4", "java", "java"),
           ("js_cleanvul_3-4", "js", "js")]
    report = []

    for base, lang, lx in BOS:
        # ---- BUOC 1: loc bo `full` (nguoi dung neu 22/09: full truoc, roi dan xuat)
        recs, metas = load(a.src_dir, base, "full")
        keep, cnt, ex, pid, drop = clean_full(recs, metas, lang, lx, a.min_tok)
        n_pair0 = len(set(pid))

        # ---- BUOC 2: `common` va `4cwe` THUA HUONG quyet dinh cua `full`
        for tag in ("full", "common", "4cwe"):
            rs, ms = load(a.src_dir, base, tag)
            if rs is None:
                continue
            pids = [pair_key(m, i, lx) for i, m in enumerate(ms)]
            idx = [i for i in range(len(rs)) if pids[i] in keep]
            lab = collections.Counter(rs[i]["label"] for i in idx)
            report.append({"bo": base, "lang": lang, "tag": tag,
                           "n_vao": len(rs), "n_ra": len(idx), "bo_di": len(rs) - len(idx),
                           "label_ra": dict(lab),
                           "chi_tiet": dict(cnt) if tag == "full" else "thua huong tu full",
                           "cap_vao": n_pair0 if tag == "full" else len(set(pids)),
                           "cap_ra": len({pids[i] for i in idx}),
                           "vi_du": {k: v for k, v in ex.items()} if tag == "full" else {}})
            if not a.dry:
                with open(os.path.join(a.out_dir, "%s_%s.jsonl" % (base, tag)), "w", encoding="utf-8") as f1, \
                     open(os.path.join(a.out_dir, "%s_%s.meta.jsonl" % (base, tag)), "w", encoding="utf-8") as f2:
                    for i in idx:
                        f1.write(json.dumps(rs[i], ensure_ascii=False) + "\n")
                        m = dict(ms[i]); m["pair_id"] = pids[i]
                        f2.write(json.dumps(m, ensure_ascii=False) + "\n")

    if not a.dry:
        with open(os.path.join(a.out_dir, "_clean_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("%-30s %-7s %7s %7s %7s %8s %8s  %s" %
          ("bo", "tag", "vao", "ra", "bo", "cap vao", "cap ra", "nhan"))
    print("-" * 108)
    for r in report:
        print("%-30s %-7s %7d %7d %7d %8d %8d  %s" %
              (r["bo"], r["tag"], r["n_vao"], r["n_ra"], r["bo_di"],
               r["cap_vao"], r["cap_ra"], r["label_ra"]))
    print()
    for r in report:
        if r["tag"] == "full":
            print("%-30s %s" % (r["bo"], r["chi_tiet"]))
    return report


if __name__ == "__main__":
    main()
