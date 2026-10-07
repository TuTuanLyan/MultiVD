#!/usr/bin/env python3
"""Dựng nguồn Pha 1 `phase1_4cwe_withjava` = `phase1_4cwe` (C/C++ + JS, nhãn 1 thuộc CWE-022/078/079/089) + TOÀN BỘ 5 184 hàm Java của common.
Người dùng 04/10: "Java không lọc được chấp nhận nhét vào toàn bộ" - CleanVul Java không có nhãn CWE nên không lọc được theo 4 CWE.
KHÔNG chia lại: hàng 4cwe giữ đúng train/val của phase1_4cwe, hàng Java giữ đúng train/val của phase1_common_javaonly (= cách chia Java
trong phase1_common, vì 4cwe lấy nhãn 0 từ common); test.jsonl = bản sao val.jsonl như mọi nguồn Pha 1.
    python3 scripts/build_4cwe_withjava.py      (ghi vào data/final_experiment_data/, từ chối nếu thư mục đích đã có)"""
import json, os, sys

D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "final_experiment_data")
A, B, OUT = "phase1_4cwe", "phase1_common_javaonly", "phase1_4cwe_withjava"
if os.path.exists(os.path.join(D, OUT)) or os.path.exists(os.path.join(D, OUT + ".jsonl")):
    sys.exit("đã có %s - xoá tay trước nếu muốn dựng lại" % OUT)


def read(p):
    return open(os.path.join(D, p), encoding="utf-8").read().splitlines()


os.makedirs(os.path.join(D, OUT, "fold1"))
ids = set()
for name in ["%s.jsonl", "%s/fold1/train.jsonl", "%s/fold1/val.jsonl"]:
    a, b = read(name % A), read(name % B)
    assert all(json.loads(l)["lang"] != "java" for l in a) and all(json.loads(l)["lang"] == "java" for l in b)
    rows = a + b
    if name.startswith("%s/"):
        ids |= {json.loads(l)["id"] for l in rows}
    open(os.path.join(D, name % OUT), "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print("%-40s %5d = %d (%s) + %d (%s)" % (name % OUT, len(rows), len(a), A, len(b), B))
open(os.path.join(D, OUT, "fold1", "test.jsonl"), "w", encoding="utf-8").write(open(os.path.join(D, OUT, "fold1", "val.jsonl"), encoding="utf-8").read())
allids = [json.loads(l)["id"] for l in read(OUT + ".jsonl")]
assert len(allids) == len(set(allids)) == len(ids), "id trùng hoặc train ∪ val ≠ file gộp"
print("ok: %d hàm, train ∪ val = file gộp, không trùng id" % len(allids))
