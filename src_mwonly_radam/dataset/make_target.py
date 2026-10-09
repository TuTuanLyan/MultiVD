#!/usr/bin/env python3
"""Đích SVEN (Python) đã xoá comment: đọc data/sven_python_folds_norm (5 fold, chia sẵn, giữ nguyên), chỉ thay `code`
bằng bản đã xoá comment/docstring (mwg.comments, giữ số dòng); thứ tự, nhãn, fold và mọi trường khác giữ nguyên.

    python dataset/make_target.py [data/sven_python_folds_norm] [data/sven_python_folds_v4]
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mwg.comments import remove_comments  # noqa: E402

IN = sys.argv[1] if len(sys.argv) > 1 else "data/sven_python_folds_norm"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/sven_python_folds_v4"
st = collections.Counter()
for f in sorted(glob.glob(IN + "/fold*/*.jsonl")):
    o = f.replace(IN, OUT, 1)
    os.makedirs(os.path.dirname(o), exist_ok=True)
    with open(o, "w") as w:
        for l in open(f):
            r = json.loads(l)
            c = remove_comments(r["code"], "python")
            assert c.strip(), "hàm rỗng sau khi xoá comment"
            st["functions"] += 1
            st["changed"] += c != r["code"]
            r["code"] = c
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
print(OUT, dict(st))
