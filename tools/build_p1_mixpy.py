#!/usr/bin/env python3
"""Dựng dữ liệu Pha 1 TRỘN JS + Python theo fold (người dùng 06/10 11:5x: "trộn phase 1 cho model transfer tỷ lệ js:py là 4:1 với cả bộ full
và common, val sẽ bằng cả js 15% là python của fold đó. Chạy thử 5 fold"; chọn: Python lấy mẫu từ SVEN train fold k, val = JS val 15 % + SVEN val
fold k, Pha 2 chỉ cột chính).

Mỗi fold k:
  train = TOÀN BỘ JS train của pool (giữ thứ tự) + n_py = round(n_js_train / ratio) hàm Python lấy mẫu từ SVEN train fold k,
          cân nhãn (ceil(n_py/2) nhãn 1, phần còn lại nhãn 0), random.Random(seed = 42) mới cho mỗi fold, giữ thứ tự gốc của SVEN train;
  val   = JS val 15 % của pool + SVEN val fold k (152 hàm, cũng là val Pha 2 của fold k);
  test  = bản sao val (quy ước của mọi pool Pha 1: test Pha 1 chỉ để chấm lại val).
SVEN test fold k KHÔNG được đụng tới (assert: không hàm Python nào của train/val trùng nguyên văn test fold k).
Hàng chép NGUYÊN VĂN từng dòng để giữ đúng byte của file gốc.

    python tools/build_p1_mixpy.py --pool data/mwonly5_sources/js_common \\
        --target data/final_experiment_data/sven_python_folds_nocomment --out data/mwonly5_sources/js_py41_common
"""
import argparse
import collections
import hashlib
import json
import os
import random


def lines(path):
    return [l if l.endswith("\n") else l + "\n" for l in open(path, encoding="utf-8") if l.strip()]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def sample_python(rows, n, rng):
    """Lấy n dòng cân nhãn (ceil(n/2) nhãn 1), giữ thứ tự gốc của rows."""
    by_label = {0: [], 1: []}
    for i, l in enumerate(rows):
        by_label[int(json.loads(l)["label"])].append(i)
    n1 = (n + 1) // 2
    n0 = n - n1
    assert len(by_label[1]) >= n1 and len(by_label[0]) >= n0, "SVEN train không đủ hàm để lấy mẫu"
    keep = set(rng.sample(by_label[1], n1)) | set(rng.sample(by_label[0], n0))
    return [rows[i] for i in sorted(keep)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="pool JS có fold1/{train,val}.jsonl (val = 15 %%)")
    ap.add_argument("--target", required=True, help="tập đích SVEN có fold<k>/{train,val,test}.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=4.0, help="JS : Python trong train")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", choices=("js+target", "js", "js+py41"), default="js+target",
                    help="val Pha 1: JS val + SVEN val fold k (mặc định) hoặc CHỈ JS val (người dùng 06/10 22:0x: \"trộn ngẫu nhiên 4:1 lấy val js\") "
                         "hoặc js+py41 = JS val + Python lấy từ SVEN val fold k theo tỉ lệ JS:Py = --ratio, cân nhãn bằng random.Random(seed) như train "
                         "(người dùng 08/10 09:1x: \"val gồm cả 2 ngôn ngữ, tỷ lệ 4:1 random giống train\")")
    a = ap.parse_args()

    js_train = lines(os.path.join(a.pool, "fold1", "train.jsonl"))
    js_val = lines(os.path.join(a.pool, "fold1", "val.jsonl"))
    n_py = int(len(js_train) / a.ratio + 0.5)                     # làm tròn nửa lên: 842/4 = 210,5 -> 211; 1066/4 = 266,5 -> 267
    stat = {"pool": a.pool, "pool_md5": {s: md5(os.path.join(a.pool, "fold1", s + ".jsonl")) for s in ("train", "val")},
            "target": a.target, "ratio": a.ratio, "seed": a.seed, "val": a.val, "n_js_train": len(js_train), "n_js_val": len(js_val), "n_py_train": n_py,
            "rule": ("train = JS train của pool (giữ thứ tự) + n_py hàm SVEN train fold k lấy mẫu cân nhãn bằng random.Random(seed) ở mọi fold (giữ thứ tự SVEN); "
                     "val = JS val + SVEN val fold k (--val js: CHỈ JS val; --val js+py41: JS val + round(n_js_val / ratio) hàm SVEN val fold k cân nhãn bằng "
                     "random.Random(seed)); test = bản sao val; chép nguyên dòng"),
            "script": "tools/build_p1_mixpy.py", "script_md5": md5(os.path.abspath(__file__)), "folds": []}
    for k in range(1, a.k + 1):
        src = os.path.join(a.target, "fold%d" % k)
        sv_train, sv_val, sv_test = (lines(os.path.join(src, s + ".jsonl")) for s in ("train", "val", "test"))
        py = sample_python(sv_train, n_py, random.Random(a.seed))       # seed 42 ở MỌI fold (người dùng 06/10: "seed 42 đầy đủ")
        if a.val == "js+target":
            val = js_val + sv_val
        elif a.val == "js+py41":
            val = js_val + sample_python(sv_val, int(len(js_val) / a.ratio + 0.5), random.Random(a.seed))   # Random(seed) mới, không dẫn xuất
        else:
            val = list(js_val)
        parts = {"train": js_train + py, "val": val}
        parts["test"] = list(parts["val"])
        # không hàm Python nào của train/val Pha 1 được trùng nguyên văn test fold k của đích
        held = {json.loads(l)["code"] for l in sv_test}
        leak = sum(json.loads(l)["code"] in held for l in py + sv_val + val)
        assert leak == 0, "fold%d: %d hàm Python của Pha 1 trùng test đích" % (k, leak)
        assert set(py) <= set(sv_train)
        out = os.path.join(a.out, "fold%d" % k)
        os.makedirs(out, exist_ok=True)
        fs = {"fold": k}
        for name, rows in parts.items():
            p = os.path.join(out, name + ".jsonl")
            open(p, "w", encoding="utf-8").writelines(rows)
            recs = [json.loads(l) for l in rows]
            fs[name] = {"n": len(rows), "label1": sum(int(r["label"]) for r in recs),
                        "langs": dict(collections.Counter(r.get("lang") for r in recs)), "md5": md5(p)}
        stat["folds"].append(fs)
        print("fold%d  train %d %s (nhãn 1: %d) | val %d %s" % (k, fs["train"]["n"], fs["train"]["langs"], fs["train"]["label1"],
                                                            fs["val"]["n"], fs["val"]["langs"]))
    json.dump(stat, open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
