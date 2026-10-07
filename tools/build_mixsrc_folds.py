#!/usr/bin/env python3
"""Dựng dữ liệu MỘT PHA TRỘN NGUỒN cho mô hình MW không đồ thị (người dùng 05/10 18:5x: "trộn nguồn sau đó train 1 pha kiểu baseline
nhưng trộn nguồn khác, cái này sẽ không dùng graph. Chạy thêm vào n=5"; chọn nguồn common chỉ JS).

Mỗi fold k: train.jsonl = MỌI hàng của pool nguồn (train + val của pool, giữ nguyên thứ tự) rồi tới train.jsonl của fold k ở tập đích;
val.jsonl / test.jsonl = val / test của fold k ở tập đích (chọn checkpoint và đánh giá hoàn toàn trên đích, như mixsrc_common cũ).
Hàng được chép NGUYÊN VĂN từng dòng (không tuần tự hoá lại) để giữ đúng byte của file gốc.

    python tools/build_mixsrc_folds.py --pool data/mwonly5_sources/js_common \\
        --target data/final_experiment_data/sven_python_folds_nocomment --out data/final_experiment_data/mixsrc_jsonly_common
"""
import argparse
import collections
import hashlib
import json
import os


def lines(path):
    return [l if l.endswith("\n") else l + "\n" for l in open(path, encoding="utf-8") if l.strip()]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="pool nguồn có fold1/{train,val}.jsonl (test = bản sao val nên bỏ qua)")
    ap.add_argument("--target", required=True, help="tập đích có fold<k>/{train,val,test}.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    pool = lines(os.path.join(a.pool, "fold1", "train.jsonl")) + lines(os.path.join(a.pool, "fold1", "val.jsonl"))
    assert len({json.loads(l)["code"] for l in pool}) == len(pool), "pool có hàm trùng nguyên văn"
    stat = {"pool": a.pool, "pool_n": len(pool), "pool_md5": {s: md5(os.path.join(a.pool, "fold1", s + ".jsonl")) for s in ("train", "val")},
            "target": a.target, "rule": "train = pool (train + val, giữ thứ tự) + đích train fold k; val/test = đích val/test fold k; chép nguyên dòng",
            "script": "tools/build_mixsrc_folds.py", "script_md5": md5(os.path.abspath(__file__)), "folds": []}
    for k in range(1, a.k + 1):
        src = os.path.join(a.target, "fold%d" % k)
        out = os.path.join(a.out, "fold%d" % k)
        os.makedirs(out, exist_ok=True)
        parts = {"train": pool + lines(os.path.join(src, "train.jsonl")),
                 "val": lines(os.path.join(src, "val.jsonl")), "test": lines(os.path.join(src, "test.jsonl"))}
        # hàm của pool không được trùng nguyên văn với val/test của đích (rò rỉ giữa nguồn và đích)
        held = {json.loads(l)["code"] for l in parts["val"] + parts["test"]}
        leak = sum(json.loads(l)["code"] in held for l in pool)
        assert leak == 0, "fold%d: %d hàm của pool trùng val/test đích" % (k, leak)
        fs = {"fold": k}
        for name, rows in parts.items():
            p = os.path.join(out, name + ".jsonl")
            open(p, "w", encoding="utf-8").writelines(rows)
            recs = [json.loads(l) for l in rows]
            fs[name] = {"n": len(rows), "label1": sum(r["label"] for r in recs), "langs": dict(collections.Counter(r.get("lang") for r in recs)),
                        "md5": md5(p)}
        stat["folds"].append(fs)
        print("fold%d  train %d %s (nhãn 1: %d) | val %d | test %d" % (k, fs["train"]["n"], fs["train"]["langs"], fs["train"]["label1"],
                                                                     fs["val"]["n"], fs["test"]["n"]))
    json.dump(stat, open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
