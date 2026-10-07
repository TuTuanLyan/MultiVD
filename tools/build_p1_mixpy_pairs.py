#!/usr/bin/env python3
"""Dựng dữ liệu Pha 1 TRỘN JS + Python RÚT THEO CẶP, val CHỈ JS (người dùng 06/10 21:0x: "chạy thêm 1 bản JS common + 1/4 số lượng của JS
hàm SVEN train_k (rút theo cặp). pha 1 dùng best roc js").

Khác tools/build_p1_mixpy.py ở hai chỗ:
  1. Python lấy theo CẶP SVEN (hàm lỗi + bản đã vá của nó), chỉ lấy cặp có CẢ HAI nửa nằm trong SVEN train fold k;
     n_pair = round_half_up(n_js_train / ratio / 2) -> 842 / 4 / 2 = 105,25 -> 105 cặp = 210 hàm (nhãn tự cân 105/105).
  2. val = CHỈ JS val của pool (không thêm SVEN val) => checkpoint Pha 1 chọn theo ROC trên JS, không nhìn val của đích.
  test = bản sao val (quy ước của mọi pool Pha 1).

Cặp SVEN: tập đích không lưu mã cặp nên ghép lại trên toàn bộ data.jsonl (760 hàm, mã nguồn không trùng): nhóm theo (tên hàm `def`, cwe);
mỗi nhóm có đúng k hàm nhãn 1 và k hàm nhãn 0; nhóm k = 1 là một cặp, nhóm k > 1 ghép tham lam theo tỉ số SequenceMatcher lớn nhất.
Lấy mẫu: random.Random(seed = 42) MỚI cho mỗi fold (không dẫn xuất seed), rút n_pair cặp trong các cặp đủ hai nửa ở train fold k,
giữ thứ tự gốc của SVEN train. Hàng chép NGUYÊN VĂN từng dòng.

    python tools/build_p1_mixpy_pairs.py --pool data/mwonly5_sources/js_common \\
        --target data/final_experiment_data/sven_python_folds_nocomment --out data/mwonly5_sources/js_py41pair_common
"""
import argparse
import collections
import difflib
import hashlib
import json
import os
import random
import re


def lines(path):
    return [l if l.endswith("\n") else l + "\n" for l in open(path, encoding="utf-8") if l.strip()]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def func_name(code):
    m = re.search(r"def\s+(\w+)", code)
    return m.group(1) if m else None


def build_pairs(rows):
    """Trả về (danh sách cặp (mã nhãn 1, mã nhãn 0, tỉ số giống), thống kê nhóm)."""
    groups = collections.defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        groups[(func_name(r["code"]), r["cwe"])][int(r["label"])].append(r["code"])
    pairs, sizes = [], collections.Counter()
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
        g = groups[key]
        assert len(g[0]) == len(g[1]), "nhóm %s lệch nhãn: %d / %d" % (key, len(g[1]), len(g[0]))
        sizes[len(g[1])] += 1
        cand = sorted(((difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), i, j)
                       for i, a in enumerate(g[1]) for j, b in enumerate(g[0])), reverse=True)
        used1, used0 = set(), set()
        for ratio, i, j in cand:
            if i in used1 or j in used0:
                continue
            used1.add(i)
            used0.add(j)
            pairs.append((g[1][i], g[0][j], ratio))
    return pairs, sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="pool JS có fold1/{train,val}.jsonl (val = 15 %%)")
    ap.add_argument("--target", required=True, help="tập đích SVEN có data.jsonl và fold<k>/{train,val,test}.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=4.0, help="JS : Python trong train")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    js_train = lines(os.path.join(a.pool, "fold1", "train.jsonl"))
    js_val = lines(os.path.join(a.pool, "fold1", "val.jsonl"))
    n_pair = int(len(js_train) / a.ratio / 2 + 0.5)                # 842 / 4 / 2 = 105,25 -> 105
    full = [json.loads(l) for l in lines(os.path.join(a.target, "data.jsonl"))]
    assert len({r["code"] for r in full}) == len(full), "data.jsonl có mã nguồn trùng - không ghép cặp theo mã được"
    pairs, sizes = build_pairs(full)
    assert len(pairs) * 2 == len(full)
    stat = {"pool": a.pool, "pool_md5": {s: md5(os.path.join(a.pool, "fold1", s + ".jsonl")) for s in ("train", "val")},
            "target": a.target, "target_md5": md5(os.path.join(a.target, "data.jsonl")), "ratio": a.ratio, "seed": a.seed,
            "n_js_train": len(js_train), "n_js_val": len(js_val), "n_pair": n_pair, "n_py_train": 2 * n_pair,
            "sven_pairs": len(pairs), "sven_group_sizes": dict(sizes), "pair_min_ratio": round(min(p[2] for p in pairs), 4),
            "rule": ("train = JS train của pool (giữ thứ tự) + n_pair cặp SVEN (hàm lỗi + bản vá) có cả hai nửa ở SVEN train fold k, rút bằng "
                     "random.Random(seed) mới ở mọi fold (giữ thứ tự SVEN); val = CHỈ JS val; test = bản sao val; chép nguyên dòng"),
            "script": "tools/build_p1_mixpy_pairs.py", "script_md5": md5(os.path.abspath(__file__)), "folds": []}
    for k in range(1, a.k + 1):
        src = os.path.join(a.target, "fold%d" % k)
        sv_train, sv_test = (lines(os.path.join(src, s + ".jsonl")) for s in ("train", "test"))
        line_of = {json.loads(l)["code"]: l for l in sv_train}
        order = {json.loads(l)["code"]: i for i, l in enumerate(sv_train)}
        complete = [p for p in pairs if p[0] in line_of and p[1] in line_of]
        complete.sort(key=lambda p: min(order[p[0]], order[p[1]]))    # thứ tự xuất hiện trong SVEN train
        assert len(complete) >= n_pair, "fold%d chỉ có %d cặp đủ hai nửa" % (k, len(complete))
        pick = random.Random(a.seed).sample(range(len(complete)), n_pair)   # seed 42 ở MỌI fold
        keep = {c for i in pick for c in complete[i][:2]}
        py = [l for l in sv_train if json.loads(l)["code"] in keep]
        assert len(py) == 2 * n_pair
        held = {json.loads(l)["code"] for l in sv_test}
        assert not keep & held, "fold%d: hàm Python của Pha 1 trùng test đích" % k
        parts = {"train": js_train + py, "val": list(js_val)}
        parts["test"] = list(parts["val"])
        out = os.path.join(a.out, "fold%d" % k)
        os.makedirs(out, exist_ok=True)
        fs = {"fold": k, "complete_pairs_in_train": len(complete),
              "picked_pair_min_ratio": round(min(complete[i][2] for i in pick), 4)}
        for name, rows in parts.items():
            p = os.path.join(out, name + ".jsonl")
            open(p, "w", encoding="utf-8").writelines(rows)
            recs = [json.loads(l) for l in rows]
            fs[name] = {"n": len(rows), "label1": sum(int(r["label"]) for r in recs),
                        "langs": dict(collections.Counter(r.get("lang") for r in recs)), "md5": md5(p)}
        stat["folds"].append(fs)
        print("fold%d  cặp đủ hai nửa ở train %d, rút %d | train %d %s (nhãn 1: %d) | val %d %s" % (
            k, len(complete), n_pair, fs["train"]["n"], fs["train"]["langs"], fs["train"]["label1"], fs["val"]["n"], fs["val"]["langs"]))
    print("cặp SVEN %d, cỡ nhóm %s, tỉ số giống nhỏ nhất %.3f" % (len(pairs), dict(sizes), stat["pair_min_ratio"]))
    json.dump(stat, open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
