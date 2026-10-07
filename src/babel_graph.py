#!/usr/bin/env python3
"""Dung DO THI DONG NHAT kieu BABEL (ICSME 2024) tu ma nguon, KHONG dung code parser.

BABEL (`gdufsnlp/BABEL`) tra loi cau hoi "rao can ngon ngu" o muc BIEU DIEN: bo han parser
(Joern/tree-sitter) vi chinh parser moi la thu khoa chat vao mot ngon ngu. Thay vao do:

  1. moi CAU LENH mot DONG  (ho dung clang-format ColumnLimit lon; Python von da vay)
  2. dinh = DONG, khong phai token
  3. hai do thi DONG NHAT tren cac dong:
       data_matrix    : noi i-j neu hai dong DUNG CHUNG mot dinh danh   -> DOI XUNG
       control_matrix : tu DO THUT LE, ngan xep noi cha->con + anh em   -> BAT DOI XUNG

Khac ban goc mot cho co CHU Y, da ghi ro: BABEL symbol hoa dinh danh (`buf` -> VAR1) vi
encoder cua ho la bang nhung TINH. O day encoder la CodeBERT/CodeT5+ da tien huan luyen
tren code, ma ten dinh danh chinh la phan lon tin hieu -> symbol hoa rat co the LAM HONG.
Nen `--symbolize none` la mac dinh; do thi VAN dung ten dinh danh de noi canh (dung luat
cua BABEL), chi phan VAN BAN dua vao encoder la giu nguyen. Muon chay dung ban goc thi
`--symbolize vars`.

Chay truc tiep de DO do thi trc khi ton GPU:
    python3 src/babel_graph.py --data_root data/sven_python_folds_norm
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np

# Tu khoa + ten dung san. Khong loai thi `if`/`for`/`self`/`len` noi TAT CA cac dong voi nhau
# va do thi data thanh gan nhu day du -> GCN khong con thong tin gi.
STOP = set("""
False None True and as assert async await break class continue def del elif else except
finally for from global if import in is lambda nonlocal not or pass raise return try while
with yield self cls print len range str int float bool list dict set tuple type object
super isinstance hasattr getattr setattr enumerate zip map filter sorted sum min max abs
open format join append extend items keys values get split strip replace startswith
endswith Exception ValueError TypeError KeyError IndexError RuntimeError OSError IOError
auto bool break case char const continue default do double else enum extern float for goto
if inline int long register restrict return short signed sizeof static struct switch
typedef union unsigned void volatile while
var let function const new this typeof instanceof null undefined true false
""".split())

IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
BLANKISH = re.compile(r"^[\s\}\{\)\(,;]*$")


def strip_noise(code: str) -> str:
    """Bo chu thich va NOI DUNG chuoi. Dinh danh nam trong chuoi khong phai phu thuoc
    du lieu — de nguyen thi mot thong bao loi dung chung se noi buc xa ca ham."""
    # GIU NGUYEN SO XUONG DONG. Thay mot chuoi/chu thich NHIEU DONG bang mot dong don
    # lam `clean` NGAN hon `raw` => ghep theo chi so LECH HANG va do thi noi nham dong.
    # Loi nay da xay ra that, chi lo ra khi thu voi docstring 3 dong.
    keepnl = lambda m: "\n" * m.group(0).count("\n")
    code = re.sub(r"/\*.*?\*/", keepnl, code, flags=re.S)
    code = re.sub(r'"""(?:.|\n)*?"""', keepnl, code)
    code = re.sub(r"'''(?:.|\n)*?'''", keepnl, code)
    out = []
    for line in code.split("\n"):
        line = re.sub(r"(#|//).*$", "", line)
        line = re.sub(r'"(?:[^"\\]|\\.)*"', ' "" ', line)
        line = re.sub(r"'(?:[^'\\]|\\.)*'", " '' ", line)
        out.append(line)
    return "\n".join(out)


def to_lines(code: str, max_lines: int):
    """Tra ve (dong_GOC_de_dua_vao_encoder, dong_DA_LAM_SACH_de_dung_do_thi)."""
    raw = [l for l in code.split("\n") if l.strip()]
    clean = [l for l in strip_noise(code).split("\n") if l.strip()]
    # strip_noise giu nguyen so dong nen hai danh sach co the lech khi mot dong chi co
    # chu thich. Ghep lai theo chi so dong GOC de khong bao gio lech hang.
    rawA = code.split("\n")
    clnA = strip_noise(code).split("\n")
    if len(clnA) != len(rawA):      # cong chan lech hang, khong bao gio duoc im lang
        raise SystemExit(f"babel_graph: strip_noise doi so dong {len(rawA)} -> {len(clnA)}")
    keep = [i for i in range(len(rawA)) if rawA[i].strip()]
    raw = [rawA[i] for i in keep][:max_lines]
    clean = [(clnA[i] if i < len(clnA) else "") for i in keep][:max_lines]
    return raw, clean


def idents_per_line(clean_lines):
    return [{t for t in IDENT.findall(l) if t not in STOP} for l in clean_lines]


def data_adj(clean_lines, max_degree: int = 0) -> np.ndarray:
    """Noi i-j neu hai dong dung chung it nhat mot dinh danh. DOI XUNG + tu vong.

    `max_degree > 0` cat bot canh cua dinh qua day, giu lai cac lang gieng GAN NHAT theo
    khoang cach dong — dinh danh dung chung nhieu lan (bien lap) neu khong cat se noi
    ca ham va lam GCN thanh phep trung binh toan cuc.
    """
    ids = idents_per_line(clean_lines)
    n = len(ids)
    A = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if ids[i] & ids[j]:
                A[i, j] = A[j, i] = 1.0
    if max_degree and n > max_degree:
        for i in range(n):
            nb = [j for j in range(n) if j != i and A[i, j] > 0]
            if len(nb) > max_degree:
                nb.sort(key=lambda j: abs(j - i))
                for j in nb[max_degree:]:
                    A[i, j] = 0.0
        A = np.maximum(A, A.T)        # giu DOI XUNG sau khi cat
    return A


def depth_list(raw_lines):
    """Do thut le. Dong chi co dau ngoac/cham phay khong mang cau truc -> -2 (luat BABEL)."""
    d = []
    for l in raw_lines:
        if BLANKISH.match(l):
            d.append(-2.0)
        else:
            d.append((len(l) - len(l.lstrip())) / 2.0)
    return d


def control_adj(depths) -> np.ndarray:
    """Ngan xep: noi CHA -> CON theo bac long, va noi ANH EM cung bac. BAT DOI XUNG."""
    n = len(depths)
    A = np.eye(n, dtype=np.float32)
    stack = []                                    # (chi so, do sau)
    last_at_depth = {}
    for i, d in enumerate(depths):
        if d == -2.0:                             # dong khong mang cau truc: gan vao cha gan nhat
            if stack:
                A[stack[-1][0], i] = 1.0
            continue
        while stack and stack[-1][1] >= d:
            stack.pop()
        if stack:
            A[stack[-1][0], i] = 1.0              # cha -> con
        if last_at_depth.get(d) is not None:
            A[last_at_depth[d], i] = 1.0          # anh em truoc -> anh em sau
        last_at_depth[d] = i
        for k in list(last_at_depth):
            if k > d:
                last_at_depth[k] = None
        stack.append((i, d))
    return A


def build(code: str, max_lines: int, max_degree: int = 0):
    raw, clean = to_lines(code, max_lines)
    if not raw:
        raw, clean = [""], [""]
    return raw, data_adj(clean, max_degree), control_adj(depth_list(raw))


# ------------------------------------------------- CHUNK: gop dong truoc khi dung do thi
def chunk_lines(raw_lines, clean_lines, tok_len, budget, max_nodes):
    """Gop cac dong LIEN TIEP thanh khuc <= `budget` token. Tra ve (khuc_goc, khuc_sach, nhom).

    VI SAO CO HAM NAY. Do duoc o khoi `mw_assemble_babel` (n=5): khi dinh la MOT DONG
    (~20 token) thi encoder tien huan luyen mat gan het ngu canh, va hai tang GCN khong
    mua lai du — ROC 0.8910 so voi 0.9069 cua multi-window. Gop dong thanh khuc giu lai
    ngu canh THAT trong tung dinh ma van con nhieu dinh de do thi co nghia.

    Mot dong dai hon `budget` van giu nguyen thanh mot khuc: cat giua cau lenh thi ca hai
    nua deu vo nghia voi encoder.
    """
    chunks, cur, cur_tok = [], [], 0
    for i, n in enumerate(tok_len):
        if cur and cur_tok + n > budget:
            chunks.append(cur); cur, cur_tok = [], 0
        cur.append(i); cur_tok += n
    if cur:
        chunks.append(cur)
    chunks = chunks[:max_nodes]
    raw = ["\n".join(raw_lines[i] for i in c) for c in chunks]
    cln = ["\n".join(clean_lines[i] for i in c) for c in chunks]
    return raw, cln, chunks


def build_chunked(code, tok, budget, max_nodes, max_lines=400, max_degree=0):
    """Nhu `build` nhung DINH LA KHUC chu khong phai DONG.

      canh DATA    : hai khuc dung chung mot dinh danh              -> DOI XUNG
      canh CONTROL : do sau cua khuc = do sau NHO NHAT trong cac dong cua no, roi dung
                     dung thuat toan ngan xep cua BABEL             -> BAT DOI XUNG

    Lay min vi dong NONG nhat quyet dinh khuc do nam o dau trong cay long nhau.
    """
    raw_lines, clean_lines = to_lines(code, max_lines)
    if not raw_lines:
        raw_lines, clean_lines = [""], [""]
    tl = [max(1, len(tok(l, add_special_tokens=False)["input_ids"])) for l in raw_lines]
    raw, cln, groups = chunk_lines(raw_lines, clean_lines, tl, budget, max_nodes)
    d_line = depth_list(raw_lines)
    depths = [min(d_line[i] for i in g) for g in groups]
    return raw, data_adj(cln, max_degree), control_adj(depths)


# ------------------------------------------------------------------ do thu, 0 GPU
def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/sven_python_folds_norm")
    ap.add_argument("--max_lines", type=int, default=150)
    ap.add_argument("--max_degree", type=int, default=0)
    ap.add_argument("--tokenizer", default="microsoft/codebert-base")
    a = ap.parse_args()

    rows, seen = [], set()
    for f in range(1, 6):
        for sp in ("train", "val", "test"):
            for l in open(Path(a.data_root) / f"fold{f}" / f"{sp}.jsonl"):
                r = json.loads(l)
                if r["code"] not in seen:
                    seen.add(r["code"]); rows.append(r)
    print(f"{len(rows)} ham duy nhat | max_lines={a.max_lines} max_degree={a.max_degree}\n")

    nl, dd, dc, iso, cut = [], [], [], 0, 0
    for r in rows:
        raw, Ad, Ac = build(r["code"], a.max_lines, a.max_degree)
        n = len(raw); nl.append(n)
        off = Ad - np.eye(n)
        dd.append(off.sum() / max(n, 1))
        dc.append((Ac - np.eye(n)).sum() / max(n, 1))
        iso += int((off.sum(1) == 0).any())
        cut += int(len([l for l in r["code"].split("\n") if l.strip()]) > a.max_lines)
    nl, dd, dc = np.array(nl), np.array(dd), np.array(dc)
    print(f"  dong/ham      : TB {nl.mean():6.1f} | trung vi {np.median(nl):5.0f} | p90 {np.percentile(nl,90):5.0f} | max {nl.max()}")
    print(f"  bi cat o {a.max_lines} dong: {100*cut/len(rows):.1f}%")
    print(f"  bac do thi DATA    : TB {dd.mean():5.2f} canh/dinh | p90 {np.percentile(dd,90):5.2f} | max {dd.max():.1f}")
    print(f"  bac do thi CONTROL : TB {dc.mean():5.2f} canh/dinh | p90 {np.percentile(dc,90):5.2f} | max {dc.max():.1f}")
    print(f"  ham co dinh CO LAP trong do thi data: {100*iso/len(rows):.1f}%")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    lens = []
    for r in rows[:300]:
        raw, _, _ = build(r["code"], a.max_lines, a.max_degree)
        lens += [len(tok(l, add_special_tokens=False)["input_ids"]) for l in raw]
    lens = np.array(lens)
    print(f"\n  token/DONG ({a.tokenizer}, 300 ham): TB {lens.mean():5.1f} | p90 {np.percentile(lens,90):4.0f} "
          f"| p99 {np.percentile(lens,99):4.0f} | max {lens.max()}")
    for L in (24, 32, 48, 64):
        print(f"     max_line_tokens={L:3d} -> giu duoc {100*(lens<=L).mean():5.1f}% so dong tron ven")


if __name__ == "__main__":
    _main()
