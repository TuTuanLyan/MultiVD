import ast
import io
import re
import textwrap
import tokenize

__all__ = ["remove_comments", "blank_comments", "comment_spans", "strip", "strip_comments_py"]

LANGS = ("c", "java", "js", "python")
_ALIAS = {"c": "c", "cpp": "c", "ccpp": "c", "c++": "c", "java": "java", "js": "js", "javascript": "js",
          "python": "python", "py": "python"}

_ID = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$")
_DIGIT = set("0123456789abcdefABCDEF")
_JS_KW_BEFORE_RE = {
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "throw", "case", "do", "else", "yield", "await", "if", "while", "for",
    "switch", "catch", "with",
}
_JS_KW_VALUE = {"this", "super", "true", "false", "null", "undefined"}


def _record_string(warn, a, b):
    warn.setdefault("_strings", []).append((a, b))


def _lines_ending_in_strings(code, strings):
    out = set()
    for a, b in strings:
        k = code.find("\n", a, b)
        while k >= 0:
            out.add(code.count("\n", 0, k))
            k = code.find("\n", k + 1, b)
    return out


def _skip_splices(code, i):
    n = len(code)
    while i < n and code[i] == "\\":
        j = i + 1
        while j < n and code[j] in " \t":
            j += 1
        if j < n and code[j] == "\n":
            i = j + 1
        elif j + 1 < n and code[j] == "\r" and code[j + 1] == "\n":
            i = j + 2
        else:
            break
    return i


def _spans_c(code):
    n, i = len(code), 0
    spans, warn = [], {"unterminated_block": 0, "unterminated_string": 0,
                       "raw_string": 0, "splice_in_comment": 0, "digit_sep": 0}
    while i < n:
        ch = code[i]

        if ch == "R" and i + 1 < n and code[i + 1] == '"':
            k = i - 1
            pre = ""
            while k >= 0 and code[k] in "uUL8":
                pre = code[k] + pre
                k -= 1
            if k < 0 or code[k] not in _ID:
                j = code.find("(", i + 2)
                if 0 <= j <= i + 18:
                    delim = code[i + 2:j]
                    end = code.find(")" + delim + '"', j + 1)
                    if end >= 0:
                        warn["raw_string"] += 1
                        _record_string(warn, i, end + len(delim) + 2)
                        i = end + len(delim) + 2
                        continue

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
                if code[j] == "\n":
                    warn["unterminated_string"] += 1
                    break
                j += 1
            _record_string(warn, i, j)
            i = j
            continue

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


def _spans_java(code):
    n, i = len(code), 0
    spans, warn = [], {"unterminated_block": 0, "unterminated_string": 0,
                       "text_block": 0, "unicode_escape_slash": 0}
    low = code.lower()
    for tok in ("\\u002f", "\\u002a"):
        warn["unicode_escape_slash"] += low.count(tok)
    while i < n:
        ch = code[i]

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


def _js_regex_allowed(code, i, spans):
    j = i - 1
    while j >= 0:
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
        return False
    if c == "}":
        return True
    if c in "+-":
        return not (j > 0 and code[j - 1] == c)
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
            return False
        return False
    if c in "'\"`":
        return False
    return True


def _js_scan(code, start, end, spans, warn, depth=0):
    i = start
    while i < end:
        ch = code[i]

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
                            break
                        if code[k] == "{":
                            dep += 1
                        elif code[k] == "}":
                            dep -= 1
                        k += 1
                    if dep and depth < 8:
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
                        break
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
        if ch in "'\"`" or (ch == "/" and i + 1 < end and code[i + 1] in "/*"):
            j = _js_scan_one(code, i, end, spans, warn, depth)
            i = j
            continue
        i += 1
    return end


def _js_scan_one(code, i, end, spans, warn, depth):
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
    out = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out, warn


_CODE_CHARS = re.compile(r"[;{}]")


def _orphan_closers_c(code, spans):
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
    out = []
    i = code.find("*/")
    if i >= 0 and not any(a <= i < b for a, b in spans):
        j = code.find("/*")
        eol = code.find("\n", i + 2)
        eol = len(code) if eol < 0 else eol
        if (j < 0 or j > i) and not _CODE_CHARS.search(code[:i]) and not code[i + 2:eol].strip():
            out.append((0, i + 2))
    if lang != "c":
        return out
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
            while a > 0:
                pb = code.rfind("\n", 0, bol - 1) + 1 if bol > 0 else 0
                prev = code[pb:bol - 1] if bol > 0 else ""
                if bol == 0 or not prev.strip().startswith("*") or _CODE_CHARS.search(prev):
                    break
                bol = pb
                a = pb + (len(prev) - len(prev.lstrip()))
        out.append((a, p + 2))
    return out


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
    for a, b in all_spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, warn


def strip(code, lang):
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
    gap = set()
    for a, b in spans:
        for k in range(a, b):
            if code[k] != "\n":
                marked[k] = False
            had.add(nl_before[k])
        if a > 0 and b < n and not code[a - 1].isspace() and not code[b].isspace():
            gap.add(a)
    buf = []
    for k, (c, m) in enumerate(zip(code, marked)):
        if k in gap:
            buf.append(" ")
        if m:
            buf.append(c)
    kept = "".join(buf)
    lines = kept.split("\n")
    out = []
    for idx, li in enumerate(lines):
        s = li if idx in keep_trailing else li.rstrip()
        if s == "" and idx in had:
            continue
        out.append(s)
    res = "\n".join(out)
    stats = {"n_spans": len(spans), "chars_removed": n - len(kept),
             "lines_in": code.count("\n") + 1, "lines_out": res.count("\n") + 1, **warn}
    return res, stats


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
                b = src.find(q, a + 3)
                k = (a + 3) if b < 0 else (b + 3)
    return spans


def _token_py(src):
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
    lines = src.split("\n")
    ded = textwrap.dedent(src)
    dedent_width = 0
    if ded != src:
        f0 = next((l for l in lines if l.strip()), "")
        g0 = next((l for l in ded.split("\n") if l.strip()), "")
        dedent_width = len(f0) - len(g0)

    def shift_spans(spans, d):
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
        ds_all.extend(_line_start_tq_spans(src))
    spans = ds_all + cmt
    ds_set = set(ds_all)
    for r1, c1, r2, c2 in sorted(spans, reverse=True):
        if 1 <= r1 <= len(lines) and 1 <= r2 <= len(lines):
            _blank(lines, r1, c1, r2, c2)
            if (r1, c1, r2, c2) in ds_set:
                in_string.difference_update(range(r1, r2))
    return "\n".join(l if (k + 1) in in_string else l.rstrip() for k, l in enumerate(lines)), parsed_ok


def remove_comments(code, lang="c"):
    lang = _ALIAS.get(lang)
    if lang is None:
        raise ValueError("lang không hỗ trợ (chỉ có %s)" % (LANGS,))
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    if lang == "python":
        return strip_comments_py(code)[0]
    return strip(code, lang)[0]


def blank_comments(code, lang="c"):
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
