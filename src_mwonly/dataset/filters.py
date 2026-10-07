"""Luật chất lượng mã (tầng T2) và khoá khử trùng dạng abstract (tầng T4).

T2 — mã sinh / nén (CHỈ JS: bản build của Java là .class, của C là .o, nên file .java/.c luôn là mã nguồn):
  T2_build_path                đường dẫn build (dist/ build/ out/ *.min.js ...) VÀ nội dung xác nhận nén/sinh
  T2_short_function_name       `function t(`, `function ab(` ... (tên 1–2 ký tự)
  T2_mangled_content           >= 60 % định danh dài <= 2 ký tự VÀ hình dạng nén
  T2_generator_marker          dấu vết trình sinh mã (webpack, babel, PEG.js, Closure, Emscripten)
  T2_lime_artifact             artifact của lime
  T2_compressed_shape_mangled  hình dạng nén VÀ >= 40 % định danh ngắn
  T2_one_line                  cả hàm một dòng (trên mã thô)
T2 — không có thân (mọi ngôn ngữ): khai báo, thân rỗng, constructor chỉ có danh sách khởi tạo. Hàm CỤT thiếu `}`
cuối (PrimeVul) không tính là rỗng.
"""
import re

# ----------------------------------------------------------------------------- dạng abstract (T4)
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
    """Định danh -> ID1, ID2… (theo thứ tự gặp), chuỗi -> S, số -> N; giữ từ khoá và toán tử."""
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


# ----------------------------------------------------------------------------- mã sinh / nén (JS)
BUILD = re.compile(r"(\.min\.js$|-min\.js$|\.bundle\.|(^|/)(dist|build|out|_build)/)", re.I)
MONOREPO = re.compile(r"(^|/)packages/node_modules/", re.I)      # gói của CHÍNH dự án trong monorepo, không phải build
GEN_MARKER = re.compile(r"__webpack_|webpackJsonp|__esModule|_interopRequireDefault|"
                        r"_classCallCheck|_createClass|_typeof2?=|_slicedToArray|"
                        r"peg\$|\$jscomp|goog\.provide|Module\[\"asm\"\]|wasmExports")
COMPRESS_FN = re.compile(r"^\s*(?:async\s+)?function\s*\*?\s*[\w$]{1,2}\s*\(")


def is_mangled(code, lang, min_ids=8):
    kw = KW.get(lang, KW["c"])
    ids = [t for k, t in tokens(code) if k == "id" and t not in kw]
    if len(ids) < min_ids:
        return False
    return sum(1 for t in ids if len(t) <= 2) / len(ids) >= 0.60


def mangle_frac(code, lang):
    kw = KW.get(lang, KW["c"])
    ids = [t for k, t in tokens(code) if k == "id" and t not in kw]
    return sum(1 for t in ids if len(t) <= 2) / len(ids) if len(ids) >= 8 else 0.0


def shape_compressed(code):
    """Hình dạng mã nén: một dòng, dòng > 250 ký tự, > 120 ký tự/dòng, hoặc > 3 dấu `;`/dòng."""
    ls = code.split("\n")
    nl = len(ls)
    return (nl == 1
            or max((len(x) for x in ls), default=0) > 250
            or len(code) / max(1, nl) > 120
            or code.count(";") / max(1, nl) > 3)


def is_generated(code, lang):
    """Chạy trên MÃ THÔ. -> "gen_marker" | "lime_artifact" | "shape_and_mangled" | None."""
    if GEN_MARKER.search(code):
        return "gen_marker"
    if "lime25" in code or "limedev" in code:
        return "lime_artifact"
    if (shape_compressed(code) or bool(re.match(r"\s*function\s+[\w$]{1,2}\s*\(", code))) and mangle_frac(code, lang) >= 0.40:
        return "shape_and_mangled"
    return None


def is_build_path(fn, code, lang="js"):
    """Đường dẫn build chỉ là GỢI Ý: phải kèm bằng chứng trong nội dung (nén và mangle, hoặc dấu vết trình sinh)."""
    fn = fn or ""
    if MONOREPO.search(fn) or not BUILD.search(fn):
        return False
    return (is_mangled(code, lang, min_ids=4) and shape_compressed(code)) or bool(is_generated(code, lang))


# ----------------------------------------------------------------------------- không có thân
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
    """Vị trí `{` mở THÂN hàm: `{` đầu tiên ngoài chuỗi và ngoài ngoặc tròn/vuông; với C++ bỏ qua khởi tạo bằng ngoặc
    nhọn trong danh sách khởi tạo (`X() : m_{0} {}`). Ngoặc tròn không về 0 (bản vá PrimeVul ghép hỏng) thì lấy `{`
    đầu tiên ngoài chuỗi. -1 nếu không có."""
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
            first_brace = k
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
    return first_brace


def has_no_body(code, lx="c"):
    i = _body_brace(code, lx)
    if i < 0:
        return True
    e = _matching_brace(code, i)
    return not (code[i + 1:e] if e >= 0 else code[i + 1:]).strip()


# ----------------------------------------------------------------------------- T2
_GENERATED_REASON = {"gen_marker": "T2_generator_marker", "lime_artifact": "T2_lime_artifact",
                     "shape_and_mangled": "T2_compressed_shape_mangled"}


def quality_reason(code, code_raw, file_name, lx):
    """Lý do loại ở tầng T2, hoặc None. `code` đã xoá comment, `code_raw` là mã thô."""
    if lx == "js":
        if is_build_path(file_name or "", code, lx):
            return "T2_build_path"
        if COMPRESS_FN.match(code):
            return "T2_short_function_name"
        if is_mangled(code, lx, min_ids=4) and shape_compressed(code):
            return "T2_mangled_content"
        g = is_generated(code_raw, lx)
        if g:
            return _GENERATED_REASON[g]
        if len(code_raw.split("\n")) == 1:
            return "T2_one_line"
    if has_no_body(code, lx):
        return "T2_no_body"
    return None
