"""Chấm strip_comments trên Java (lexer javac 17), JS (Babel parser) — CleanVul thô — và Python
(AST của Python: xoá comment/docstring không được đổi AST ngoài docstring) trên SVEN."""
import json, csv, re, sys, subprocess, base64, collections, glob, ast, textwrap
sys.path.insert(0, "tools/v4"); from strip_comments import remove_comments
csv.field_size_limit(10 ** 9)
S = sys.argv[1]; nows = lambda s: re.sub(r"\s+", "", s)
rows = {"java": [], "js": []}
for s in (3, 4):
    for r in csv.DictReader(open("/drive1/cuongtm/dgmoe/MAML/data/js_java_cleanvul/vulnerability_score_%d.csv" % s, newline="", encoding="utf-8")):
        e = (r.get("extension") or "").strip().lower()
        if e in rows:
            rows[e] += [(r.get("func_before") or "").replace("\r\n", "\n"), (r.get("func_after") or "").replace("\r\n", "\n")]
out = {}
# ---- Java
J = ["java"] + sum([["--add-exports", "jdk.compiler/com.sun.tools.javac.%s=ALL-UNNAMED" % k] for k in ("parser", "util", "file")], []) + ["-cp", S + "/jtok", "JavaTok"]
inp = []
for raw in rows["java"]:
    for x in (raw, remove_comments(raw, "java")): inp.append(base64.b64encode(x.encode()).decode())
o = subprocess.run(J, input="\n".join(inp) + "\n", capture_output=True, text=True).stdout.splitlines()
c = collections.Counter()
for k in range(len(rows["java"])):
    t, m = o[2 * k], o[2 * k + 1]
    c["loi_dap_an" if t.startswith("ERR") else ("dung" if t.split()[0] == m.split()[0] else "SAI")] += 1
out["java"] = dict(c)
# ---- JS
p = subprocess.run(["node", S + "/babel_comments.js"], input="\n".join(json.dumps({"raw": x}) for x in rows["js"]) + "\n", capture_output=True, text=True)
res = [json.loads(l) for l in p.stdout.splitlines()]
def cut(raw, spans):
    o_, last = [], 0
    for a, b in sorted(spans): o_.append(raw[last:a]); o_.append(" "); last = b
    o_.append(raw[last:]); return "".join(o_)
c = collections.Counter()
for raw, g in zip(rows["js"], res):
    if not g["ok"]:
        c["parser_khong_doc_duoc"] += 1; continue
    c["dung" if nows(cut(raw, g["spans"])) == nows(remove_comments(raw, "js")) else "SAI"] += 1
out["js"] = dict(c)
# ---- Python (SVEN, mọi fold, bỏ trùng)
class _BoDoc(ast.NodeTransformer):
    def _f(self, n):
        self.generic_visit(n)
        if n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
        return n
    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = visit_Module = _f
def dump(src, bo_doc):
    t = ast.parse(textwrap.dedent(src))
    if bo_doc: t = _BoDoc().visit(t)
    return ast.dump(t)
seen, c = set(), collections.Counter()
for f in sorted(glob.glob("data/sven_python_folds_norm/fold*/*.jsonl")):
    for l in open(f):
        raw = json.loads(l)["code"]
        if raw in seen: continue
        seen.add(raw)
        m = remove_comments(raw, "python")
        try: a = dump(raw, True)
        except Exception: c["dap_an_khong_parse_duoc"] += 1; continue
        try: b = dump(m, False)
        except Exception:
            try: b = dump(m + "\n    pass", False)       # thân chỉ có docstring -> rỗng sau khi xoá
            except Exception: c["SAI_(ma_sau_xoa_khong_parse)"] += 1; continue
        b2 = b
        c["dung" if a == b2 or a.replace("Pass()", "") == b2.replace("Pass()", "") else "SAI"] += 1
        c["bat_bien"] += remove_comments(m, "python") == m
        c["co_comment_hoac_docstring"] += nows(m) != nows(raw)
out["python_sven"] = dict(c)
print(json.dumps(out, ensure_ascii=False, indent=1)); json.dump(out, open(S + "/cham_2.json", "w"), ensure_ascii=False, indent=1)
