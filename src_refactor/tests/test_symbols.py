#!/usr/bin/env python3
"""Kiểm hồi quy chuẩn hoá định danh (mwg.symbols) và xoá comment (mwg.comments). Chạy: python tests/test_symbols.py

Mỗi ca: (ngôn ngữ, dòng, tên PHẢI thành VAR/FUN, chữ trong literal KHÔNG được thành VAR/FUN)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mwg.comments import remove_comments      # noqa: E402
from mwg.symbols import normalize_identifiers  # noqa: E402

CASES = [
    # 6 ca đầu của fix_gadget: chuỗi nhiều dòng, template JS, text block Java, chuỗi nối dòng C, f-string nhiều dòng
    ("python", ['    q = """SELECT id, name', '        FROM people WHERE name = {0}"""', '    cur.execute(q.format(name))'], ["q", "name"], ["people", "id"]),
    ("js", ["const q = `SELECT *", " FROM t WHERE id=${user.id}", " AND name = x`;", "db.run(q, user);"], ["q", "user"], ["SELECT", "AND"]),
    ("js", ["const m = `Hello world ${name}`;", "send(m, name);"], ["m", "name"], ["world", "Hello"]),
    ("java", ['String s = """', '    select a from people', '    """;', 'q(s, people);'], ["s", "people"], ["select"]),
    ("c", ['char *s = "abc \\', ' def ghi";', 'f(s, ghi);'], ["s", "ghi"], ["def", "abc"]),
    ("python", ['    q = f"""SELECT * FROM issue', '        WHERE issue.id = {id} AND x = {{lit}}"""', '    cur.execute(q)', '    return id'], ["id", "q"], ["issue", "lit", "WHERE"]),
    # f-string MỘT dòng: giữ biến nội suy, không để tiền tố `f` thành VAR
    ("python", ['    cur.execute(f"SELECT * FROM t WHERE name=\'{name}\'")', '    return name'], ["name"], ["f", "SELECT", "t"]),
    ("python", ['    query = (f"UPDATE users "', '             f"SET first_name=\'{user.first_name}\', "', '             f"WHERE chat_id={user.chat_id}")', '    db.run(query)'], ["user", "query"], ["f", "users", "SET", "WHERE"]),
    # nháy lồng: `"` trong chuỗi nháy đơn không được ghép cặp với `"` của chuỗi khác
    ("python", ['    html = \'<a href="\' + url + \'">\' + label', '    return html'], ["url", "label", "html"], ["href", "a"]),
    ("python", ['    cursor.execute("""UPDATE games SET state="%s:resignation" WHERE id=%d;""" % (res, game))'], ["res", "game"], ["resignation", "s", "games"]),
    # tiền tố chuỗi Python
    ("python", ['    data = b"abc" + r"\\d+" + u"x" + rb"y"', '    use(data)'], ["data"], ["b", "r", "u", "rb", "abc"]),
    # gadget bị cắt ở Lmax giữa chuỗi ba nháy (tokenize báo lỗi, chuỗi không đóng)
    ("python", ['    page = req.page', '    req.write(u\'\'\'', '<select id="sctPagename" size="1">', ' * FCKeditor - The text editor'], ["req", "page"], ["select", "FCKeditor", "sctPagename", "u"]),
    # f-string: chuỗi lồng, !r, định dạng lồng {w}, {{ }}
    ("python", ['    s = f"{d[\'key\']!r:>10} {x:{w}} {{lit}}"', '    out(s)'], ["d", "x", "w", "s"], ["key", "lit", "r", "f"]),
    # C: ký tự '"', tiền tố L / u8 / R, chuỗi chứa /* */, dấu phân cách chữ số C++14, L'\n'
    ("c", ["if (c == '\"') { x = \"abc\"; }", "y = x;"], ["c", "x", "y"], ["abc"]),
    ("c", ['s = L"wide"; t = u8"x"; r = R"del(a "b" c)del";', "use(s, t, r);"], ["s", "t", "r"], ["L", "u8", "R", "wide", "del", "b"]),
    ("c", ['(void) WriteBlobString(image,"/* XPM */\\n");', "image = 0;"], ["image", "WriteBlobString"], ["XPM"]),
    ("c", ["int x = 1'000'000;", "y = x;"], ["x", "y"], []),
    ("c", ["if (wc == L'\\n') n++;", "wc = n;"], ["wc", "n"], ["L"]),
    # Java: ký tự '"', chuỗi chứa /* */
    ("java", ["if (c == '\"') { s = \"q r\"; }", "t = s;"], ["c", "s", "t"], ["q", "r"]),
    ("java", ['prefix = "/*<![CDATA[<!--*/\\n";', "use(prefix);"], ["prefix"], ["CDATA"]),
    # JS: regex literal, nháy lồng, template lồng, chuỗi chứa /* */, phép chia
    ("js", ["out = s.replace(/\"/g, \"&quot;\").replace(/'/g, \"&#39;\");", "send(out, s);"], ["s", "out"], ["quot", "g"]),
    ("js", ["defaultOption = dialog.down('option[value=\"'+defaultValue+'\"]').selected = true;"], ["defaultValue", "dialog", "defaultOption"], ["option", "value"]),
    ("js", ["const re = /^\\/[a-z]\\//i;", "if (re.test(p)) go(p);"], ["re", "p"], ["z", "a", "i"]),
    ("js", ["const h = `a ${b ? `c${d}` : \"}\"} e`;", "use(h, b, d);"], ["h", "b", "d"], ["a", "c", "e"]),
    ("js", ["out += '{/*[Circular]*/}'", "send(out);"], ["out"], ["Circular"]),
    ("js", ["a = b / c / d;", "e = a;"], ["a", "b", "c", "d", "e"], []),
    # JS: regex nằm TRONG `${...}` của template nhiều dòng
    ("js", ["const row = `<tr>", '<input value="${entry.name.replace(/"/g, "&quot;")}" type="text">', "</tr>`;", "put(row, entry);"], ["row", "entry"], ["input", "value", "type", "quot", "g"]),
]

COMMENT_CASES = [   # (ngôn ngữ, mã, kết quả mong đợi)
    ("c", 'url = "http://a/b"; // c\nx = 1; /* d */ y = 2;', 'url = "http://a/b";\nx = 1;  y = 2;'),
    ("c", "#define N 4 // kích thước\nint a[N];", "#define N 4\nint a[N];"),
    ("js", "const r = /\\/\\//g; // c\nf(r);", "const r = /\\/\\//g;\nf(r);"),
    ("java", 'String s = """\n  // không phải comment\n  """; // c', 'String s = """\n  // không phải comment\n  """;'),
    ("python", 'def f(x):\n    """doc"""\n    q = """SELECT 1"""  # c\n    return q', "def f(x):\n\n    q = \"\"\"SELECT 1\"\"\"\n    return q"),
]


def main():
    bad = 0
    for lang, lines, must, forbid in CASES:
        out, names = normalize_identifiers(lines, lang, return_symbols=True)
        miss, fake = [m for m in must if m not in names], [f for f in forbid if f in names]
        if len(out) != len(lines) or miss or fake:
            bad += 1
            print(f"SAI [{lang}] {lines}\n    -> {out}\n    thiếu {miss} | giả {fake}")
    for lang, code, want in COMMENT_CASES:
        got = remove_comments(code, lang)
        if got != want:
            bad += 1
            print(f"SAI comment [{lang}] {code!r}\n    -> {got!r}\n    cần {want!r}")
    n = len(CASES) + len(COMMENT_CASES)
    print(f"{n - bad}/{n} ca đạt")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
