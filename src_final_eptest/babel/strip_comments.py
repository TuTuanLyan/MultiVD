r'''Xoá comment cho C/C++, Java, JavaScript và Python — MỘT bản duy nhất dùng chung cho bước dựng
dữ liệu (scripts/clean_v3) và cho BABEL (code_snapshot/src/babel/strip_comments.py là bản sao y hệt file này).

API:
    remove_comments(code, lang)    -> mã đã xoá comment (tên giữ như bản cũ để mọi chỗ gọi vẫn chạy)
    blank_comments(code, lang)     -> thay ký tự comment bằng dấu cách, GIỮ NGUYÊN số dòng và vị trí
    comment_spans(code, lang)      -> [(đầu, cuối), ...] vị trí comment trong chuỗi gốc (C/Java/JS)
`lang`: "c" (cả C và C++; nhận cả "ccpp", "cpp", "c++"), "java", "js" ("javascript"), "python" ("py").

Vì sao KHÔNG dùng regex: `//.*$` cắt `"http://host/path"` thành `"http:` — đúng dòng một bản vá hay
chạm vào, và cắt GIỐNG HỆT nhau ở bản lỗi lẫn bản vá nên xoá mất chính chỗ khác nhau. Các bẫy cùng
loại mà máy trạng thái này xử lý: `'/'`, `"/*"`, regex literal của JS (`/…\//g`, `/["']/`), template
`${…}` lồng nhau, text block `"""…"""` của Java, raw string `R"d(…)d"` của C++, dấu `'` phân cách chữ
số của C++14 (`1'000`), và `//` nối dòng bằng `\` cuối dòng trong C.

Chỉ thị tiền xử lý C (`#define`, `#if`, `#include`…) là LOGIC MÃ nên được GIỮ; chỉ xoá comment
(đo 23/09: 100 bản vá PrimeVul sửa chính dòng chỉ thị, vd. thêm `#define VPN_PREFIXLEN_MIN_BYTES`).

Đuôi comment MỒ CÔI (chỉ có `*/`, mất `/*` mở) — lỗi trích xuất của PrimeVul:
    luật A  đầu hàm: đoạn đứng trước `*/` đầu tiên không có `/*`, không có `;{}`, và sau `*/` chỉ
            còn khoảng trắng tới hết dòng → là đuôi comment của đoạn mã đứng trước hàm, xoá.
            (đo 23/09: ~1 190 hàm PrimeVul, vd. ` */\nint sock_queue_err_skb(...)`)
    luật B  giữa thân (chỉ C): `*/` nằm NGOÀI chuỗi và ngoài comment thật → xoá từ đầu dòng đó
            (hoặc sau dấu `;{}` cuối cùng trên dòng) và các dòng liền trên bắt đầu bằng `*`.
            Phải hiểu chuỗi: `"Accept: */*"`, `string.find("*/")` là MÃ, không phải comment.

Quy tắc giữ cấu trúc dòng (để đồ thị BABEL theo dòng vẫn khớp):
  * chỉ xoá KÝ TỰ của comment; ký tự xuống dòng NẰM TRONG comment được giữ;
  * một comment nằm giữa hai token thành MỘT dấu cách (`a/**/b` là hai token `a` `b`);
  * mọi dòng rstrip; dòng nào rỗng SAU khi xoá VÀ trước đó có comment thì bỏ hẳn;
    dòng vốn rỗng từ đầu thì GIỮ.

Python: `tokenize` cho comment `#` THẬT, `ast` cho docstring THẬT (chuỗi đứng một mình làm câu lệnh
đầu tiên của module/hàm/lớp). Chuỗi ba nháy GÁN cho biến (vd. câu SQL) KHÔNG phải docstring.
Python giữ nguyên số dòng (docstring/comment thành dòng trống).

Đã chấm bằng đáp án độc lập (23/09): gcc (C/C++), lexer javac 17 (Java), Babel parser (JS).
'''
import ast
import io
import re
import sys
import textwrap
import tokenize

__all__ = ["remove_comments", "blank_comments", "comment_spans", "strip", "strip_comments_py"]

LANGS = ("c", "java", "js", "python")
_ALIAS = {"c": "c", "cpp": "c", "ccpp": "c", "c++": "c", "java": "java", "js": "js", "javascript": "js",
          "python": "python", "py": "python"}

_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$")
_DIGIT = set("0123456789abcdefABCDEF")
# Từ khoá mà sau nó dấu `/` là REGEX chứ không phải phép chia (JS).
_JS_KW_BEFORE_RE = {
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "throw", "case", "do", "else", "yield", "await", "if", "while", "for",
    "switch", "catch", "with",
}
# Từ khoá là GIÁ TRỊ: sau nó dấu `/` là phép chia.
_JS_KW_VALUE = {"this", "super", "true", "false", "null", "undefined"}


def _record_string(warn, a, b):
    """Ghi lại vị trí một chuỗi/ký tự/template đã quét qua — để KHÔNG rstrip dòng nằm trong chuỗi."""
    warn.setdefault("_strings", []).append((a, b))


def _lines_ending_in_strings(code, strings):
    """Chỉ số các dòng mà ký tự xuống dòng của nó NẰM TRONG một chuỗi (chuỗi nhiều dòng)."""
    out = set()
    for a, b in strings:
        k = code.find("\n", a, b)
        while k >= 0:
            out.add(code.count("\n", 0, k))
            k = code.find("\n", k + 1, b)
    return out


def _skip_splices(code, i):
    """Nhảy qua chuỗi `\\` + xuống dòng (nối dòng của C/C++). Trả về vị trí mới."""
    n = len(code)
    while i < n and code[i] == "\\":
        j = i + 1
        while j < n and code[j] in " \t":   # khoảng trắng thừa sau `\` vẫn nối dòng
            j += 1
        if j < n and code[j] == "\n":
            i = j + 1
        elif j + 1 < n and code[j] == "\r" and code[j + 1] == "\n":
            i = j + 2
        else:
            break
    return i


# --------------------------------------------------------------------------- C / C++
def _spans_c(code):
    """C và C++. Giữ nguyên chỉ thị tiền xử lý (#define/#if/...); chỉ xoá comment."""
    n, i = len(code), 0
    spans, warn = [], {"unterminated_block": 0, "unterminated_string": 0,
                       "raw_string": 0, "splice_in_comment": 0, "digit_sep": 0}
    while i < n:
        ch = code[i]

        # --- raw string C++11:  [tiền tố]R"delim( ... )delim"
        if ch == "R" and i + 1 < n and code[i + 1] == '"':
            k = i - 1
            pre = ""
            while k >= 0 and code[k] in "uUL8":
                pre = code[k] + pre
                k -= 1
            if k < 0 or code[k] not in _ID:      # tiền tố hợp lệ, không dính vào tên biến
                j = code.find("(", i + 2)
                if 0 <= j <= i + 18:             # delimiter tối đa 16 ký tự
                    delim = code[i + 2:j]
                    end = code.find(")" + delim + '"', j + 1)
                    if end >= 0:
                        warn["raw_string"] += 1
                        _record_string(warn, i, end + len(delim) + 2)
                        i = end + len(delim) + 2
                        continue

        # --- chuỗi
        if ch == '"':
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j = _skip_splices(code, j)
                    if j < n and code[j] == "\\":
                        j += 2
                    continue
                if code[j] == '"':
                    j += 1
                    break
                if code[j] == "\n":              # chuỗi không đóng -> phục hồi ở cuối dòng
                    warn["unterminated_string"] += 1
                    break
                j += 1
            _record_string(warn, i, j)
            i = j
            continue

        # --- ký tự (cảnh giác dấu nháy đơn làm dấu phân cách chữ số của C++14)
        if ch == "'":
            if (i > 0 and code[i - 1] in _DIGIT and i + 1 < n and code[i + 1] in _DIGIT):
                warn["digit_sep"] += 1
                i += 1
                continue
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j = _skip_splices(code, j)
                    if j < n and code[j] == "\\":
                        j += 2
                    continue
                if code[j] == "'":
                    j += 1
                    break
                if code[j] == "\n":
                    warn["unterminated_string"] += 1
                    break
                j += 1
            i = j
            continue

        # --- comment khối
        if ch == "/" and i + 1 < n and code[i + 1] == "*":
            j = code.find("*/", i + 2)
            if j < 0:
                warn["unterminated_block"] += 1
                spans.append((i, n))
                break
            spans.append((i, j + 2))
            i = j + 2
            continue

        # --- comment dòng, có thể nối dòng bằng `\`
        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            j = i + 2
            while j < n:
                if code[j] == "\\":
                    k = _skip_splices(code, j)
                    if k != j:
                        warn["splice_in_comment"] += 1
                        j = k
                        continue
                    j += 1
                    continue
                if code[j] == "\n":
                    break
                j += 1
            spans.append((i, j))
            i = j
            continue

        i += 1
    return spans, warn


# ------------------------------------------------------------------------------ Java
def _spans_java(code):
    n, i = len(code), 0
    spans, warn = [], {"unterminated_block": 0, "unterminated_string": 0,
                       "text_block": 0, "unicode_escape_slash": 0}
    # `//` là comment THẬT trong Java (unicode escape được xử lý trước khi tách token).
    low = code.lower()
    for tok in ("\\u002f", "\\u002a"):
        warn["unicode_escape_slash"] += low.count(tok)
    while i < n:
        ch = code[i]

        # --- text block  """ ... """   (Java 15+)
        if ch == '"' and code.startswith('"""', i):
            j = i + 3
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code.startswith('"""', j):
                    j += 3
                    break
                j += 1
            else:
                warn["unterminated_string"] += 1
            warn["text_block"] += 1
            _record_string(warn, i, j)
            i = j
            continue

        if ch == '"':
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == '"':
                    j += 1
                    break
                if code[j] == "\n":
                    warn["unterminated_string"] += 1
                    break
                j += 1
            _record_string(warn, i, j)
            i = j
            continue

        if ch == "'":
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == "'":
                    j += 1
                    break
                if code[j] == "\n":
                    warn["unterminated_string"] += 1
                    break
                j += 1
            i = j
            continue

        if ch == "/" and i + 1 < n and code[i + 1] == "*":
            j = code.find("*/", i + 2)
            if j < 0:
                warn["unterminated_block"] += 1
                spans.append((i, n))
                break
            spans.append((i, j + 2))
            i = j + 2
            continue

        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            j = code.find("\n", i)
            j = n if j < 0 else j
            spans.append((i, j))
            i = j
            continue

        i += 1
    return spans, warn


# -------------------------------------------------------------------------------- JS
def _js_regex_allowed(code, i, spans):
    """Dấu `/` ở vị trí i có thể mở một REGEX literal không?

    Nhìn lùi ký tự có nghĩa gần nhất, BỎ QUA khoảng trắng và các comment ĐÃ nhận dạng
    (nằm trong `spans`). Sau một giá trị (tên biến, số, chuỗi, `)`, `]`, `++`) thì `/` là
    phép CHIA; còn lại là regex.
    """
    j = i - 1
    while j >= 0:
        # lùi qua comment đã nhận dạng
        hit = None
        for a, b in reversed(spans):
            if a <= j < b:
                hit = a
                break
        if hit is not None:
            j = hit - 1
            continue
        if code[j] in " \t\r\n":
            j -= 1
            continue
        break
    if j < 0:
        return True
    c = code[j]
    if c in ")]":
        return False            # `(a+b)/2`, `arr[0]/2` -> chia (bỏ sót `if(x) /re/`)
    if c == "}":
        return True             # kết thúc khối -> regex; `({}/2)` hiếm gặp
    if c in "+-":
        return not (j > 0 and code[j - 1] == c)     # `a++ /2` là chia
    if c in _ID:
        k = j
        while k >= 0 and code[k] in _ID:
            k -= 1
        word = code[k + 1:j + 1]
        if word in _JS_KW_BEFORE_RE:
            return True
        if word in _JS_KW_VALUE:
            return False
        if word[0].isdigit():
            return False        # số -> chia
        return False            # tên biến/hàm -> chia
    if c in "'\"`":
        return False            # sau một chuỗi -> chia
    return True                 # `= / ( , : ; ! & | ? { [ ...` -> regex


def _js_scan(code, start, end, spans, warn, depth=0):
    """Quét [start, end) của mã JS, ghi comment vào `spans`."""
    i = start
    while i < end:
        ch = code[i]

        # --- template literal, hỗ trợ `${ ... }` lồng nhau
        if ch == "`":
            j = i + 1
            while j < end:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == "`":
                    j += 1
                    break
                if code[j] == "$" and j + 1 < end and code[j + 1] == "{":
                    k, dep = j + 2, 1
                    while k < end and dep:
                        if code[k] in "'\"`" or (code[k] == "/" and k + 1 < end and code[k + 1] in "/*"):
                            break                       # để vòng quét con xử lý
                        if code[k] == "{":
                            dep += 1
                        elif code[k] == "}":
                            dep -= 1
                        k += 1
                    if dep and depth < 8:
                        # có chuỗi/comment bên trong -> quét đệ quy phần biểu thức
                        m, dep2 = j + 2, 1
                        m = _js_scan_expr(code, j + 2, end, spans, warn, depth + 1)
                        j = m
                        continue
                    j = k
                    continue
                j += 1
            _record_string(warn, i, j)
            i = j
            continue

        if ch in "'\"":
            j = i + 1
            while j < end:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == ch:
                    j += 1
                    break
                if code[j] == "\n":
                    warn["unterminated_string"] += 1
                    break
                j += 1
            _record_string(warn, i, j)
            i = j
            continue

        if ch == "/" and i + 1 < end and code[i + 1] == "*":
            j = code.find("*/", i + 2)
            if j < 0 or j >= end:
                warn["unterminated_block"] += 1
                spans.append((i, end))
                return end
            spans.append((i, j + 2))
            i = j + 2
            continue

        if ch == "/" and i + 1 < end and code[i + 1] == "/":
            j = code.find("\n", i)
            j = end if (j < 0 or j > end) else j
            spans.append((i, j))
            i = j
            continue

        if ch == "/":
            if _js_regex_allowed(code, i, spans):
                j, incls, ok = i + 1, False, False
                while j < end:
                    c = code[j]
                    if c == "\\":
                        j += 2
                        continue
                    if c == "\n":
                        break                       # regex không bắc qua dòng -> là phép chia
                    if c == "[":
                        incls = True
                    elif c == "]":
                        incls = False
                    elif c == "/" and not incls:
                        ok = True
                        j += 1
                        break
                    j += 1
                if ok:
                    warn["regex"] += 1
                    while j < end and code[j] in "dgimsuvy":
                        j += 1
                    i = j
                    continue
            i += 1
            continue

        i += 1
    return end


def _js_scan_expr(code, start, end, spans, warn, depth):
    """Quét phần `${ ... }` của template: dừng đúng ở dấu `}` cân bằng."""
    i, dep = start, 1
    while i < end:
        ch = code[i]
        if ch == "{":
            dep += 1
            i += 1
            continue
        if ch == "}":
            dep -= 1
            i += 1
            if dep == 0:
                return i
            continue
        nxt = _js_scan(code, i, i + 1, spans, warn, depth)
        if ch in "'\"`" or (ch == "/" and i + 1 < end and code[i + 1] in "/*"):
            j = _js_scan_one(code, i, end, spans, warn, depth)
            i = j
            continue
        i += 1
    return end


def _js_scan_one(code, i, end, spans, warn, depth):
    """Xử lý ĐÚNG MỘT đơn vị (chuỗi / template / comment) bắt đầu tại i, trả vị trí sau nó."""
    ch = code[i]
    if ch == "`":
        j = i + 1
        while j < end:
            if code[j] == "\\":
                j += 2
                continue
            if code[j] == "`":
                _record_string(warn, i, j + 1)
                return j + 1
            if code[j] == "$" and j + 1 < end and code[j + 1] == "{" and depth < 8:
                j = _js_scan_expr(code, j + 2, end, spans, warn, depth + 1)
                continue
            j += 1
        return j
    if ch in "'\"":
        j = i + 1
        while j < end:
            if code[j] == "\\":
                j += 2
                continue
            if code[j] == ch:
                _record_string(warn, i, j + 1)
                return j + 1
            if code[j] == "\n":
                warn["unterminated_string"] += 1
                return j
            j += 1
        return j
    if ch == "/" and i + 1 < end and code[i + 1] == "*":
        j = code.find("*/", i + 2)
        if j < 0 or j >= end:
            spans.append((i, end))
            return end
        spans.append((i, j + 2))
        return j + 2
    if ch == "/" and i + 1 < end and code[i + 1] == "/":
        j = code.find("\n", i)
        j = end if (j < 0 or j > end) else j
        spans.append((i, j))
        return j
    return i + 1


def _spans_js(code):
    spans, warn = [], {"unterminated_block": 0, "unterminated_string": 0, "regex": 0}
    _js_scan(code, 0, len(code), spans, warn, 0)
    spans.sort()
    # hợp nhất span lồng nhau (có thể sinh ra từ nhánh đệ quy của template)
    out = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out, warn


# ----------------------------------------------------------- đuôi comment mồ côi (luật A, B)
_CODE_CHARS = re.compile(r"[;{}]")


def _orphan_closers_c(code, spans):
    """Vị trí các `*/` nằm NGOÀI chuỗi/ký tự/raw string và ngoài comment thật (mã C)."""
    n, i, si, out = len(code), 0, 0, []
    spans = sorted(spans)
    while i < n:
        while si < len(spans) and spans[si][1] <= i:
            si += 1
        if si < len(spans) and spans[si][0] <= i < spans[si][1]:
            i = spans[si][1]
            continue
        ch = code[i]
        if ch == "R" and i + 1 < n and code[i + 1] == '"' and (i == 0 or code[i - 1] not in _ID or code[i - 1] in "uUL8"):
            j = code.find("(", i + 2)
            if 0 <= j <= i + 18:
                end = code.find(")" + code[i + 2:j] + '"', j + 1)
                if end >= 0:
                    i = end + (j - i - 2) + 2
                    continue
        if ch in "\"'":
            if ch == "'" and i > 0 and code[i - 1] in _DIGIT and i + 1 < n and code[i + 1] in _DIGIT:
                i += 1
                continue
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j = _skip_splices(code, j)
                    if j < n and code[j] == "\\":
                        j += 2
                    continue
                if code[j] == ch:
                    j += 1
                    break
                if code[j] == "\n":
                    break
                j += 1
            i = j
            continue
        if code.startswith("*/", i):
            out.append(i)
            i += 2
            continue
        i += 1
    return out


def _orphan_comment_tails(code, lang, spans):
    """Vùng [a, b) là đuôi comment mồ côi (luật A cho mọi ngôn ngữ, luật B chỉ C)."""
    out = []
    # luật A — đầu hàm
    i = code.find("*/")
    if i >= 0 and not any(a <= i < b for a, b in spans):
        j = code.find("/*")
        eol = code.find("\n", i + 2)
        eol = len(code) if eol < 0 else eol
        if (j < 0 or j > i) and not _CODE_CHARS.search(code[:i]) and not code[i + 2:eol].strip():
            out.append((0, i + 2))
    if lang != "c":
        return out
    # luật B — giữa thân
    for p in _orphan_closers_c(code, spans):
        if out and out[0][0] <= p < out[0][1]:
            continue
        bol = code.rfind("\n", 0, p) + 1
        seg = code[bol:p]
        last = max(seg.rfind(";"), seg.rfind("{"), seg.rfind("}"))
        if last >= 0:
            a = bol + last + 1
        else:
            a = bol + (len(seg) - len(seg.lstrip()))
            # lên các dòng liền trên bắt đầu bằng `*` (thân comment khối) mà không mang mã
            while a > 0:
                pb = code.rfind("\n", 0, bol - 1) + 1 if bol > 0 else 0
                prev = code[pb:bol - 1] if bol > 0 else ""
                if bol == 0 or not prev.strip().startswith("*") or _CODE_CHARS.search(prev):
                    break
                bol = pb
                a = pb + (len(prev) - len(prev.lstrip()))
        out.append((a, p + 2))
    return out


# ----------------------------------------------------------------------------- chung
def comment_spans(code, lang):
    lang = _ALIAS.get(lang, lang)
    if lang == "c":
        return _spans_c(code)
    if lang == "java":
        return _spans_java(code)
    if lang == "js":
        return _spans_js(code)
    raise ValueError("lang không hỗ trợ: %r (chỉ có %s)" % (lang, LANGS))


def _removal_spans(code, lang):
    spans, warn = comment_spans(code, lang)
    orphans = _orphan_comment_tails(code, lang, spans)
    warn["orphan_comment_tails"] = len(orphans)
    all_spans = sorted(spans + orphans)
    merged = []
    for a, b in all_spans:                        # hợp nhất vùng chồng nhau
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, warn


def strip(code, lang):
    """C/Java/JS: trả về (mã đã xoá comment, thống kê)."""
    lang = _ALIAS.get(lang, lang)
    spans, warn = _removal_spans(code, lang)
    keep_trailing = _lines_ending_in_strings(code, warn.pop("_strings", []))
    n = len(code)
    marked = [True] * n
    had = set()
    nl_before = [0] * (n + 1)
    ln = 0
    for k, c in enumerate(code):
        nl_before[k] = ln
        if c == "\n":
            ln += 1
    nl_before[n] = ln
    # Comment là DẤU PHÂN CÁCH TOKEN nên phải thành MỘT dấu cách chứ không biến mất:
    # `a/**/b` là hai token `a` `b`, không phải `ab`. Chỉ chèn khi cả hai bên KHÔNG phải
    # khoảng trắng; còn lại xoá hẳn cho sạch.
    gap = set()
    for a, b in spans:
        for k in range(a, b):
            if code[k] != "\n":            # giữ ký tự xuống dòng nằm trong comment
                marked[k] = False
            had.add(nl_before[k])
        if a > 0 and b < n and not code[a - 1].isspace() and not code[b].isspace():
            gap.add(a)
    buf = []
    for k, (c, m) in enumerate(zip(code, marked)):
        if k in gap:
            buf.append(" ")               # comment -> đúng MỘT dấu cách
        if m:
            buf.append(c)
    kept = "".join(buf)
    lines = kept.split("\n")
    out = []
    # Chỉ số dòng trong `kept` trùng chỉ số dòng gốc: ký tự xuống dòng luôn được giữ.
    for idx, li in enumerate(lines):
        s = li if idx in keep_trailing else li.rstrip()   # dòng kết thúc TRONG chuỗi: giữ nguyên khoảng trắng cuối
        if s == "" and idx in had:
            continue                        # dòng chỉ có comment -> bỏ hẳn
        out.append(s)
    res = "\n".join(out)
    stats = {"n_spans": len(spans), "chars_removed": n - len(kept),
             "lines_in": code.count("\n") + 1, "lines_out": res.count("\n") + 1, **warn}
    return res, stats


# ---------------------------------------------------------------------------- Python
TQ = ('"""', "'" * 3)


def _blank(lines, r1, c1, r2, c2):
    if r1 == r2:
        s = lines[r1 - 1]
        lines[r1 - 1] = s[:c1] + " " * (c2 - c1) + s[c2:]
        return
    s = lines[r1 - 1]; lines[r1 - 1] = s[:c1] + " " * (len(s) - c1)
    for r in range(r1 + 1, r2):
        lines[r - 1] = " " * len(lines[r - 1])
    s = lines[r2 - 1]; lines[r2 - 1] = " " * c2 + s[c2:]


def _docstring_spans(src):
    """Vị trí docstring THẬT. None nếu không parse được."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str) and first.value.end_lineno is not None):
            v = first.value
            out.append((v.lineno, v.col_offset, v.end_lineno, v.end_col_offset))
    return out


def _line_start_tq_spans(src):
    """Chuỗi ba nháy MỞ ĐẦU DÒNG -> coi là docstring. Đường lùi khi ast thất bại."""
    spans = []
    for q in TQ:
        k = 0
        while True:
            a = src.find(q, k)
            if a < 0:
                break
            bol = src.rfind("\n", 0, a) + 1
            if src[bol:a].strip() == "":
                b = src.find(q, a + 3)
                b = len(src) if b < 0 else b + 3
                r1 = src.count("\n", 0, a) + 1; c1 = a - bol
                r2 = src.count("\n", 0, b) + 1; c2 = b - (src.rfind("\n", 0, b) + 1)
                spans.append((r1, c1, r2, c2)); k = b
            else:
                # chuỗi GIÁ TRỊ (`q = <3 nháy>`): nhảy qua CẢ nháy đóng của nó, nếu không
                # nháy đóng nằm đầu dòng sẽ bị đọc nhầm thành một docstring mới.
                b = src.find(q, a + 3)
                k = (a + 3) if b < 0 else (b + 3)
    return spans


def _token_py(src):
    """(comment, dòng nằm trong chuỗi nhiều dòng) theo `tokenize`; None nếu không tách token được."""
    cmt, in_string = [], set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                cmt.append((tok.start[0], tok.start[1], tok.end[0], tok.end[1]))
            elif tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                in_string.update(range(tok.start[0], tok.end[0]))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return cmt, in_string, False
    return cmt, in_string, True


def strip_comments_py(src, drop_docstring=True):
    """Trả về (mã đã xoá comment, parse_được_bằng_ast). Số dòng KHÔNG đổi.

    Hàm trích từ trong class bị thụt lề nên `ast.parse` báo IndentationError: khi đó phân tích trên
    bản `textwrap.dedent` để TÌM VỊ TRÍ, rồi dời cột về mã gốc và xoá trắng TRÊN MÃ GỐC. Không dựng
    lại mã từ bản dedent, vì dedent biến dòng chỉ có khoảng trắng TRONG CHUỖI thành rỗng.
    Dòng kết thúc bên trong một chuỗi nhiều dòng (vd. câu SQL ba nháy) được giữ nguyên khoảng trắng
    cuối dòng; mọi dòng khác rstrip."""
    lines = src.split("\n")
    ded = textwrap.dedent(src)
    dedent_width = 0                                   # số cột dedent đã cắt ở mỗi dòng không trống
    if ded != src:
        f0 = next((l for l in lines if l.strip()), "")
        g0 = next((l for l in ded.split("\n") if l.strip()), "")
        dedent_width = len(f0) - len(g0)

    def shift_spans(spans, d):                        # dời cột từ bản dedent về mã gốc
        return [(r1, c1 + d, r2, c2 + d) for r1, c1, r2, c2 in spans]

    ds = _docstring_spans(src) if drop_docstring else []
    if drop_docstring and ds is None and dedent_width:
        d2 = _docstring_spans(ded)
        ds = shift_spans(d2, dedent_width) if d2 is not None else None
    parsed_ok = ds is not None
    cmt, in_string, ok_tok = _token_py(src)
    if not ok_tok and dedent_width:
        c2, t2, ok2 = _token_py(ded)
        if ok2:
            cmt, in_string = shift_spans(c2, dedent_width), t2
    ds_all = list(ds or [])
    if drop_docstring and not parsed_ok:
        ds_all.extend(_line_start_tq_spans(src))     # đường lùi: chuỗi ba nháy mở đầu dòng
    spans = ds_all + cmt
    ds_set = set(ds_all)
    for r1, c1, r2, c2 in sorted(spans, reverse=True):
        if 1 <= r1 <= len(lines) and 1 <= r2 <= len(lines):
            _blank(lines, r1, c1, r2, c2)
            if (r1, c1, r2, c2) in ds_set:     # docstring đã xoá: dòng của nó không còn "trong chuỗi"
                in_string.difference_update(range(r1, r2))
    return "\n".join(l if (k + 1) in in_string else l.rstrip() for k, l in enumerate(lines)), parsed_ok


# ------------------------------------------------------------------------------- API
def remove_comments(code, lang="c"):
    """Mã đã xoá comment. Chạy lại trên mã đã sạch thì KHÔNG đổi gì (đã đo trên toàn bộ dữ liệu)."""
    lang = _ALIAS.get(lang)
    if lang is None:
        raise ValueError("lang không hỗ trợ (chỉ có %s)" % (LANGS,))
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    if lang == "python":
        return strip_comments_py(code)[0]
    return strip(code, lang)[0]


def blank_comments(code, lang="c"):
    """Thay ký tự comment bằng dấu cách, GIỮ số dòng và vị trí — dùng để tìm định danh theo dòng
    (babel/clean_gadget) mà không làm lệch chỉ số dòng với đồ thị."""
    lang = _ALIAS.get(lang)
    if lang is None:
        raise ValueError("lang không hỗ trợ (chỉ có %s)" % (LANGS,))
    if lang == "python":
        return strip_comments_py(code)[0]
    spans, _ = _removal_spans(code, lang)
    b = list(code)
    for a, e in spans:
        for k in range(a, e):
            if b[k] != "\n":
                b[k] = " "
    return "".join(b)


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "c"
    sys.stdout.write(remove_comments(sys.stdin.read(), lang))
