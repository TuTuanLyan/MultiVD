#!/usr/bin/env python3
"""Xoa comment + docstring cua Python MA KHONG dung regex tho.

Vi sao phai viet lai: `babel/strip_comments` cua dong nghiep coi MOI chuoi ba nhay la
docstring, nen no xoa ca `query = <<3 nhay>>select * from usr where email like...<<3 nhay>>`
— dung cau SQL tao ra lo hong CWE-089. Do duoc 230 cho nhu vay trong
`data/sven_python_folds_nocmt` cua ho (xem FACTS §74).

Cach lam:
  * `tokenize` -> token COMMENT la comment `#` THAT (khong bao gio la `#` trong chuoi).
  * `ast`      -> docstring la chuoi dung MOT MINH lam cau lenh dau tien cua
                  Module/FunctionDef/AsyncFunctionDef/ClassDef. Chuoi gan cho bien KHONG phai.
  * Hon mot nua ham SVEN duoc trich tu trong class nen thut dau dong => `ast.parse` bao
    IndentationError. Bo thut chung ra, xu ly, roi tra thut lai.
  * Khi van khong parse duoc (~1.6%): quy tac bao thu — chuoi ba nhay MO DAU DONG (truoc no
    chi co khoang trang) la docstring; co dau `=` phia truoc thi la GIA TRI, giu nguyen.
  * Xoa bang cach thay khoang trang, GIU NGUYEN SO DONG, de do thi BABEL dung tren dong
    van khop hang voi ma goc.
"""
import ast, io, textwrap, tokenize

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
    """Vi tri docstring THAT. None neu khong parse duoc."""
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
    """Chuoi ba nhay mo dau dong -> coi la docstring. Duong lui khi ast that bai."""
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
                # chuoi GIA TRI (`q = <3 nhay>`): nhay qua CA nhay dong cua no, neu khong
                # nhay dong nam dau dong se bi doc nham thanh mot docstring moi.
                b = src.find(q, a + 3)
                k = (a + 3) if b < 0 else (b + 3)
    return spans


def strip_comments_py(src, drop_docstring=True):
    """Tra ve (ma da xoa comment, parse_duoc_bang_ast). So dong KHONG doi."""
    ded = textwrap.dedent(src)
    if ded != src and _docstring_spans(src) is None and _docstring_spans(ded) is not None:
        first = src.split("\n")[0]
        pre = first[: len(first) - len(ded.split("\n")[0])]
        if not pre.strip():
            out, ok = strip_comments_py(ded, drop_docstring)
            return "\n".join((pre + l).rstrip() for l in out.split("\n")), ok

    ds = _docstring_spans(src) if drop_docstring else []
    parsed_ok = ds is not None
    spans = list(ds or [])
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                spans.append((tok.start[0], tok.start[1], tok.end[0], tok.end[1]))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        pass
    if drop_docstring and not parsed_ok:
        spans.extend(_line_start_tq_spans(src))

    lines = src.split("\n")
    for r1, c1, r2, c2 in sorted(spans, reverse=True):
        if 1 <= r1 <= len(lines) and 1 <= r2 <= len(lines):
            _blank(lines, r1, c1, r2, c2)
    return "\n".join(l.rstrip() for l in lines), parsed_ok
