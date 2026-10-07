"""String-aware comment stripping.

The regex version in BABEL cut at the first `//` or `#` anywhere in the line, so

    const u = "https://host/path";   ->   const u = "https:
    x = "#token"                     ->   x = "

That deletes exactly the kind of line a vulnerability patch touches, and it does so
identically in the vulnerable and the fixed twin, which erases the difference the
paired task depends on. This scanner walks the text and only strips comments that
start outside a string or character literal.

Behaviour kept from the original: C preprocessor directives are removed, and Python
docstrings collapse to an empty literal.
"""

DQ = '"'
SQ = "'"
BT = "`"
BS = "\\"
NL = "\n"


def remove_comments(code, lang="c"):
    n = len(code)
    out = []
    i = 0
    at_line_start = True
    triple = lang == "python"
    quotes = DQ + SQ + (BT if lang in ("js", "java") else "")
    while i < n:
        ch = code[i]

        # Python docstring
        if triple and code[i:i + 3] in (DQ * 3, SQ * 3):
            q = code[i:i + 3]
            j = code.find(q, i + 3)
            out.append(DQ * 2)
            i = n if j < 0 else j + 3
            at_line_start = False
            continue

        # string / char / template literal: copied through untouched
        if ch in quotes:
            j = i + 1
            while j < n:
                if code[j] == BS:
                    j += 2
                    continue
                if code[j] == ch:
                    break
                if code[j] == NL and ch != BT:      # unterminated literal
                    break
                j += 1
            out.append(code[i:min(j + 1, n)])
            i = j + 1
            at_line_start = False
            continue

        # block comment
        if lang != "python" and code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue

        # line comment
        if lang != "python" and code.startswith("//", i):
            j = code.find(NL, i)
            i = n if j < 0 else j
            continue

        # Python comment, or a C preprocessor directive at the start of a line
        if ch == "#" and (lang == "python" or (lang == "c" and at_line_start)):
            j = code.find(NL, i)
            i = n if j < 0 else j
            continue

        out.append(ch)
        at_line_start = (ch == NL) or (at_line_start and ch in " \t")
        i += 1
    return "".join(out)
