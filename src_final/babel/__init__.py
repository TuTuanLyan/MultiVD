"""Bieu dien dong cua BABEL (Ye et al., ICSME 2024; https://github.com/gdufsnlp/BABEL) — chep tu ban
/data/tranmanhcuong/multibabel (preprocess/graph.py, evaluator/clean_gadget.py) de khong phu thuoc repo do.
  clean_gadget(lines, lang) : chuan hoa dinh danh -> VARk / FUNk (bo string/char literal), theo tu khoa cua ngon ngu
  build_relations(lines, sym_lines, typed) : ma tran ke uint8 [R, L, L]
      typed=False : 2 quan he goc BABEL  (co_use = hai dong dung chung VAR/FUN; ctrl = cay thut le + xich anh em)
      typed=True  : 4 quan he            (def_use, co_use, ctrl, next) — ban MultiBABEL
"""
from .clean_gadget import clean_gadget, KEYWORDS_BY_LANG   # noqa: F401
from .graph import build_relations, BRACKET_ONLY           # noqa: F401
