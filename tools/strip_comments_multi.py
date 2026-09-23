#!/usr/bin/env python3
r"""Xoa comment cho C/C++, Java, JavaScript — quet trang thai, KHONG dung regex.

Vi sao khong dung regex: mot regex `//.*$` cat dut `"http://host/path"` thanh
`"http:` — dung cai dong ma mot ban va lo hong hay cham vao, va cat GIONG HET
nhau o ban vulnerable lan ban fixed nen xoa mat chinh cho khac nhau giua hai
ban. Cac bay khac cung loai: `'/'`, `"/*"`, regex literal cua JS, text block
cua Java, raw string cua C++, dau nhay don lam dau phan cach chu so trong C++
(`1'000'000`), va `//` noi dong bang `\` cuoi dong trong C.

Ham chinh:
    comment_spans(code, lang) -> [(start, end), ...]   vi tri comment trong chuoi goc
    strip(code, lang)         -> (code_da_xoa, stats)

Quy tac xoa (giu nguyen cau truc dong, de do thi BABEL theo dong con khop duoc):
  * chi xoa KY TU cua comment; ky tu xuong dong NAM TRONG comment duoc giu lai;
  * moi dong rstrip;
  * mot dong bi BO HAN neu sau khi xoa no rong VA truoc do no co chua comment.
    Dong von da rong tu dau thi GIU — nen cau truc dong trong cua ma goc khong doi.

`lang`: "c" (dung cho ca C va C++), "java", "js".
"""

import sys

__all__ = ["comment_spans", "strip", "LANGS"]

LANGS = ("c", "java", "js")

_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$")
_DIGIT = set("0123456789abcdefABCDEF")
# Tu khoa sau no dau `/` la REGEX chu khong phai phep chia (JS).
_JS_KW_BEFORE_RE = {
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "throw", "case", "do", "else", "yield", "await", "if", "while", "for",
    "switch", "catch", "with",
}
# Tu khoa la GIA TRI: sau no dau `/` la phep chia.
_JS_KW_VALUE = {"this", "super", "true", "false", "null", "undefined"}


def _skip_splices(code, i):
    """Nhay qua chuoi `\\` + xuong dong (noi dong cua C/C++). Tra ve vi tri moi."""
    n = len(code)
    while i < n and code[i] == "\\":
        j = i + 1
        while j < n and code[j] in " \t":   # khoang trang thua sau `\` van noi dong
            j += 1
        if j < n and code[j] == "\n":
            i = j + 1
        elif j + 1 < n and code[j] == "\r" and code[j + 1] == "\n":
            i = j + 2
        else:
            break
    return i


# --------------------------------------------------------------------------- C
def _spans_c(code):
    """C va C++. Giu nguyen chi thi tien xu ly (#define/#if/...); chi xoa comment."""
    n, i = len(code), 0
    spans, warn = [], {"unterminated_block": 0, "unterminated_string": 0,
                       "raw_string": 0, "splice_in_comment": 0, "digit_sep": 0}
    while i < n:
        ch = code[i]

        # --- raw string C++11:  [prefix]R"delim( ... )delim"
        if ch == "R" and i + 1 < n and code[i + 1] == '"':
            k = i - 1
            pre = ""
            while k >= 0 and code[k] in "uUL8":
                pre = code[k] + pre
                k -= 1
            if k < 0 or code[k] not in _ID:      # prefix hop le, khong dinh vao ten bien
                j = code.find("(", i + 2)
                if 0 <= j <= i + 18:             # delimiter toi da 16 ky tu
                    delim = code[i + 2:j]
                    end = code.find(")" + delim + '"', j + 1)
                    if end >= 0:
                        warn["raw_string"] += 1
                        i = end + len(delim) + 2
                        continue

        # --- string literal
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
                if code[j] == "\n":              # chuoi khong dong -> phuc hoi o cuoi dong
                    warn["unterminated_string"] += 1
                    break
                j += 1
            i = j
            continue

        # --- char literal (canh dau nhay don lam dau phan cach chu so cua C++14)
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

        # --- block comment
        if ch == "/" and i + 1 < n and code[i + 1] == "*":
            j = code.find("*/", i + 2)
            if j < 0:
                warn["unterminated_block"] += 1
                spans.append((i, n))
                break
            spans.append((i, j + 2))
            i = j + 2
            continue

        # --- line comment, co the noi dong bang `\`
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


# ------------------------------------------------------------------------ Java
def _spans_java(code):
    n, i = len(code), 0
    spans, warn = [], {"unterminated_block": 0, "unterminated_string": 0,
                       "text_block": 0, "unicode_escape_slash": 0}
    # `//` la comment THAT trong Java (unicode escape xu ly truoc khi lex).
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


# -------------------------------------------------------------------------- JS
def _js_regex_allowed(code, i, spans):
    """Dau `/` o vi tri i co the mo mot REGEX literal khong?

    Nhin lui ky tu co nghia gan nhat, BO QUA khoang trang va cac comment DA nhan
    dang (nam trong `spans`). Sau mot gia tri (ten bien, so, chuoi, `)`, `]`, `++`)
    thi `/` la phep CHIA; con lai thi la regex.
    """
    j = i - 1
    while j >= 0:
        # lui qua comment da nhan dang
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
        return False            # `(a+b)/2`, `arr[0]/2`  -> chia (bo sot `if(x) /re/`)
    if c == "}":
        return True             # ket thuc block -> regex; `({}/2)` hiem gap
    if c in "+-":
        return not (j > 0 and code[j - 1] == c)     # `a++ /2` la chia
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
            return False        # so -> chia
        return False            # ten bien/ham -> chia
    if c in "'\"`":
        return False            # sau mot chuoi -> chia
    return True                 # `= / ( , : ; ! & | ? { [ ...` -> regex


def _js_scan(code, start, end, spans, warn, depth=0):
    """Quet [start,end) cua ma JS, ghi comment vao `spans`."""
    i = start
    while i < end:
        ch = code[i]

        # --- template literal, ho tro `${ ... }` long nhau
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
                            break                       # de vong quet con xu ly
                        if code[k] == "{":
                            dep += 1
                        elif code[k] == "}":
                            dep -= 1
                        k += 1
                    if dep and depth < 8:
                        # co chuoi/comment ben trong -> quet de quy phan bieu thuc
                        m, dep2 = j + 2, 1
                        m = _js_scan_expr(code, j + 2, end, spans, warn, depth + 1)
                        j = m
                        continue
                    j = k
                    continue
                j += 1
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
                        break                       # regex khong bac qua dong -> la phep chia
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
    """Quet phan `${ ... }` cua template: dung dung o dau `}` can bang."""
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
    """Xu ly DUNG MOT don vi (chuoi / template / comment) bat dau tai i, tra vi tri sau no."""
    ch = code[i]
    if ch == "`":
        j = i + 1
        while j < end:
            if code[j] == "\\":
                j += 2
                continue
            if code[j] == "`":
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
    # hop nhat span long nhau (co the sinh ra tu nhanh de quy cua template)
    out = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out, warn


# ----------------------------------------------------------------------- chung
def comment_spans(code, lang):
    if lang in ("c", "cpp", "ccpp", "c++"):
        return _spans_c(code)
    if lang == "java":
        return _spans_java(code)
    if lang in ("js", "javascript"):
        return _spans_js(code)
    raise ValueError("lang khong ho tro: %r (chi co %s)" % (lang, LANGS))


def strip(code, lang):
    """Tra ve (code_da_xoa_comment, stats)."""
    spans, warn = comment_spans(code, lang)
    n = len(code)
    keep = bytearray(b"\x01") * 0          # placeholder tranh nham kieu
    marked = [True] * n
    had = set()
    nl_before = [0] * (n + 1)
    ln = 0
    for k, c in enumerate(code):
        nl_before[k] = ln
        if c == "\n":
            ln += 1
    nl_before[n] = ln
    # Quy tac cua C/Java/JS: mot comment la DAU PHAN CACH TOKEN, phai thanh MOT dau
    # cach chu khong bien mat — `a/**/b` la hai token `a` `b`, khong phai `ab`.
    # Chi chen khi ca hai ben deu KHONG phai khoang trang; con lai xoa han cho sach.
    gap = set()
    for a, b in spans:
        for k in range(a, b):
            if code[k] != "\n":            # giu ky tu xuong dong nam trong comment
                marked[k] = False
            had.add(nl_before[k])
        if a > 0 and b < n and not code[a - 1].isspace() and not code[b].isspace():
            gap.add(a)
    _buf = []
    for k, (c, m) in enumerate(zip(code, marked)):
        if k in gap:
            _buf.append(" ")               # comment -> dung MOT dau cach
        if m:
            _buf.append(c)
    kept = "".join(_buf)
    lines = kept.split("\n")
    out = []
    for idx, li in enumerate(lines):
        s = li.rstrip()
        if s == "" and idx in had:
            continue                        # dong chi co comment -> bo han
        out.append(s)
    res = "\n".join(out)
    stats = {
        "n_spans": len(spans),
        "chars_removed": n - len(kept),
        "lines_in": code.count("\n") + 1,
        "lines_out": res.count("\n") + 1,
        **warn,
    }
    return res, stats


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "c"
    src = sys.stdin.read()
    o, st = strip(src, lang)
    sys.stdout.write(o)
    sys.stderr.write(repr(st) + "\n")
