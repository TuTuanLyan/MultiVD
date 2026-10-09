"""Đồ thị dòng không cần parser (theo BABEL, Ye et al. ICSME 2024), hợp nhất giữa các ngôn ngữ.

Mỗi dòng (không rỗng) là một đỉnh. Quan hệ, mỗi cái là ma trận uint8 [L, L]:
  def_use : dòng i định nghĩa VARk (gán / biến lặp / tham số), dòng j > i dùng nó trước lần định nghĩa lại (có hướng)
  co_use  : hai dòng dùng chung VARk / FUNk (đối xứng). co_mode="all": mọi cặp dòng (BABEL gốc);
            co_mode="chain": chỉ hai lần xuất hiện LIÊN TIẾP của mỗi ký hiệu (mật độ không phụ thuộc độ dài hàm)
  ctrl    : cây thụt lề (cha -> con) + xích anh em cùng độ sâu
  next    : i -> i+1 qua các dòng lệnh (bỏ dòng chỉ có ngoặc)
typed=False: 2 quan hệ gốc BABEL [co_use, ctrl]; typed=True: 4 quan hệ [def_use, co_use, ctrl, next].
"""
import re

import numpy as np

BRACKET_ONLY = re.compile(r'^[\s\}\{\)\(,;]*$')
VAR_RE = re.compile(r'\bVAR\d+\b')
FUN_RE = re.compile(r'\bFUN\d+\b')
# assignment operators, excluding comparisons (==, <=, >=, !=) and arrows (=>)
ASSIGN_RE = re.compile(r'(?<![=!<>])(?:=(?![=>])|\+=|-=|\*=|/=|%=|\|=|&=|\^=|<<=|>>=|:=)')
FOR_RE = re.compile(r'\bfor\b\s*\(?\s*(?:[\w:<>\[\]\*&]+\s+)?(VAR\d+(?:\s*,\s*VAR\d+)*)\s*(?:\bin\b|=|:|of\b)')


# ----------------------------------------------------------------------------- BABEL gốc
def construct_adjacency_matrix(code_list):
    n = len(code_list)
    m = np.eye(n, dtype='uint8')
    syms = [set(VAR_RE.findall(l)) | {'f' + s for s in FUN_RE.findall(l)} for l in code_list]
    for i in range(n):
        if not syms[i]:
            continue
        for j in range(i + 1, n):
            if syms[i] & syms[j]:
                m[i][j] = 1
                m[j][i] = 1
    return m


def get_depth_list(code):
    indents = []
    for line in code:
        if BRACKET_ONLY.match(line):
            indents.append(-2)
        else:
            indents.append(len(line) - len(line.lstrip(' ')))
    positive = [i for i in indents if i > 0]
    unit = max(1, min(positive)) if positive else 1
    return [-1 if i == -2 else i // unit for i in indents]


def connect_elements(depth_list):
    n = len(depth_list)
    conn = np.zeros((n, n), dtype='uint8')

    def connect(i, j):
        if 0 <= i < n and 0 <= j < n:
            conn[i][j] = 1

    stack = []

    def process_element(index):
        if not stack:
            stack.append(index)
        else:
            while stack and depth_list[stack[-1]] >= depth_list[index]:
                stack.pop()
            if stack:
                connect(stack[-1], index)
            stack.append(index)

    skip_depth = None
    for i in range(n):
        depth = depth_list[i]
        if depth == -1:
            continue
        if depth == 0:
            if stack:
                connect(stack[-1], i)
            stack.append(i)
        else:
            if skip_depth is not None and depth >= skip_depth:
                continue
            process_element(i)
            if depth_list[i] == -1:
                skip_depth = depth

    depth_to_nodes = {}
    for i in range(n):
        if depth_list[i] == -1:
            continue
        depth_to_nodes.setdefault(depth_list[i], []).append(i)
    for nodes in depth_to_nodes.values():
        for a, b in zip(nodes, nodes[1:]):
            connect(a, b)
    if n > 1:
        conn[0][:] = 0
        conn[0][1] = 1
    return conn


# ----------------------------------------------------------------------------- quan hệ có kiểu
def _definitions(line, is_signature):
    """Symbols this (normalised) line defines."""
    defs = set()
    if is_signature:
        return set(VAR_RE.findall(line))
    m = FOR_RE.search(line)
    if m:
        defs.update(VAR_RE.findall(m.group(1)))
    # every VAR that appears before an assignment operator (handles a = b = c and a[i] = ...)
    pos = 0
    for am in ASSIGN_RE.finditer(line):
        lhs = line[pos:am.start()]
        last = VAR_RE.findall(lhs)
        if last:
            defs.add(last[0] if len(last) == 1 else last[-1] if '[' in lhs else last[0])
        pos = am.end()
    return defs


def def_use_matrix(code_symbolic):
    n = len(code_symbolic)
    m = np.zeros((n, n), dtype='uint8')
    last_def = {}
    sig_seen = False
    for j, line in enumerate(code_symbolic):
        if BRACKET_ONLY.match(line):
            continue
        uses = set(VAR_RE.findall(line))
        for s in uses:
            i = last_def.get(s)
            if i is not None and i != j:
                m[i][j] = 1
        is_sig = (not sig_seen) and bool(FUN_RE.search(line)) and '(' in line
        for s in _definitions(line, is_sig):
            last_def[s] = j
        if is_sig:
            sig_seen = True
    return m


def next_matrix(code_lines):
    n = len(code_lines)
    m = np.zeros((n, n), dtype='uint8')
    prev = None
    for i, line in enumerate(code_lines):
        if BRACKET_ONLY.match(line):
            continue
        if prev is not None:
            m[prev][i] = 1
        prev = i
    return m


def co_use_chain(code_symbolic):
    """co_use theo CHUỖI: mỗi ký hiệu chỉ nối hai lần xuất hiện liên tiếp (đối xứng, có đường chéo như BABEL)."""
    n = len(code_symbolic)
    m = np.eye(n, dtype="uint8")
    last = {}
    for i, line in enumerate(code_symbolic):
        for sym in set(VAR_RE.findall(line)) | {"f" + x for x in FUN_RE.findall(line)}:
            j = last.get(sym)
            if j is not None and j != i:
                m[i][j] = 1
                m[j][i] = 1
            last[sym] = i
    return m


def build_relations(code_lines, code_symbolic, typed=True, co_mode="chain"):
    """uint8 [R, L, L]; R = 2 (co_use, ctrl) khi typed=False, 4 (def_use, co_use, ctrl, next) khi typed=True.

    R-GCN tính A @ H nên hàng i gom H[j] ở mọi j có A[i, j] = 1. Các hàm dựng ghi A[nguồn, đích]; chuyển vị các
    quan hệ có hướng để đỉnh nhận kéo từ đỉnh nguồn (dòng dùng nhận từ dòng định nghĩa, con nhận từ cha,
    dòng i nhận từ dòng i-1)."""
    if len(code_symbolic) != len(code_lines):
        raise ValueError(f"lệch số dòng: symbolic {len(code_symbolic)} vs gốc {len(code_lines)}")
    co = co_use_chain(code_symbolic) if co_mode == "chain" else construct_adjacency_matrix(code_symbolic)
    ctrl = connect_elements(get_depth_list(code_lines))
    if not typed:
        return np.stack([co, ctrl])
    return np.stack([def_use_matrix(code_symbolic).T, co, ctrl.T, next_matrix(code_lines).T])
