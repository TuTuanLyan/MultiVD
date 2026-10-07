#!/usr/bin/env python3
"""Ghép các bộ nguồn (theo cặp) thành pool Pha 1: `<out_root>/mwsrc_<tên>/fold1/{train,val,test}.jsonl`.

Pha 1 chạy MỘT lần trên pool; `test.jsonl` là bản sao `val.jsonl` (val nguồn chỉ dùng để chọn checkpoint,
đánh giá thật nằm ở Pha 2 trên đích). Chia val NGẪU NHIÊN theo hàm (cặp có thể bị tách giữa train/val).

    v4jsCjv_rand   (SOTA) JS common + Java, val 10 %, seed 36
    v4full         C/C++ paired + Java + JS, bộ full,   val 20 %, seed 42
    v4common       C/C++ paired + Java + JS, bộ common, val 20 %, seed 42
    v4cwe4         C/C++ paired + JS, bộ 4cwe,          val 20 %, seed 42
    v4common_nocc  Java + JS, bộ common,                val 20 %, seed 42

    python dataset/build_pools.py --src_dir data/sources_v4 --out_root data
"""
import argparse
import collections
import json
import os
import random

POOLS = {
    "v4full":   [("ccpp_primevul_paired_full", "ccpp"), ("java_cleanvul_3-4_full", "java"), ("js_cleanvul_3-4_full", "js")],
    "v4common": [("ccpp_primevul_paired_common", "ccpp"), ("java_cleanvul_3-4_common", "java"), ("js_cleanvul_3-4_common", "js")],
    "v4cwe4":   [("ccpp_primevul_paired_4cwe", "ccpp"), ("js_cleanvul_3-4_4cwe", "js")],
    "v4common_nocc": [("java_cleanvul_3-4_common", "java"), ("js_cleanvul_3-4_common", "js")],
}
SOTA_POOL = ("v4jsCjv_rand", ["js_cleanvul_3-4_common", "java_cleanvul_3-4_full"])   # Java: CleanVul không có CWE -> full = common


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def write_pool(out_root, name, train, val):
    d = os.path.join(out_root, "mwsrc_" + name, "fold1")
    os.makedirs(d, exist_ok=True)
    for fn, data in (("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", val)):
        with open(os.path.join(d, fn), "w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_pairs(train, val):
    """Số cặp có một nửa ở train, nửa kia ở val."""
    side = collections.defaultdict(set)
    for part, rows in (("train", train), ("val", val)):
        for r in rows:
            side[r["pair_id"]].add(part)
    return sum(1 for v in side.values() if len(v) == 2)


def build_sota_pool(src_dir, out_root):
    name, bases = SOTA_POOL
    rows = []
    for base in bases:
        recs, metas = load(os.path.join(src_dir, base + ".jsonl")), load(os.path.join(src_dir, base + ".meta.jsonl"))
        assert len(recs) == len(metas)
        for r, m in zip(recs, metas):
            assert r["label"] == m["label"]
            rows.append(dict(code=r["code"], label=int(r["label"]), lang=r["lang"], cwe=r.get("cwe", ""),
                             pair_id=f"{r['lang']}:{m['pair_id']}", source_set=base))
    labs = collections.defaultdict(list)
    for r in rows:
        labs[r["pair_id"]].append(r["label"])
    assert all(sorted(v) == [0, 1] for v in labs.values()), "cặp không đủ nhãn 0/1"
    random.Random(36).shuffle(rows)
    n_val = int(round(0.1 * len(rows)))
    val, train = rows[:n_val], rows[n_val:]
    write_pool(out_root, name, train, val)
    langs = collections.Counter(r["lang"] for r in rows)
    print("%-14s %8d %8d %8d %9d %9d %8d %12d" % (name, langs["ccpp"], langs["java"], langs["js"], len(rows), len(train), len(val),
                                                 split_pairs(train, val)))


def build_pool(src_dir, out_root, name, parts, val_ratio, seed):
    rows, per_lang = [], collections.Counter()
    for base, lang in parts:
        recs, metas = load(os.path.join(src_dir, base + ".jsonl")), load(os.path.join(src_dir, base + ".meta.jsonl"))
        for r, m in zip(recs, metas):
            if not m.get("pair_id"):
                raise ValueError("thiếu pair_id trong %s — pool chỉ dựng từ bộ theo cặp" % base)
            rows.append({"code": r["code"], "label": r["label"], "lang": lang, "cwe": r["cwe"], "cwe_id": r["cwe_id"],
                         "cwe_class": r["cwe_class"], "pair_id": m["pair_id"], "source_set": base})
        per_lang[lang] += len(recs)
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    val_idx = set(idx[:int(round(len(rows) * val_ratio))])
    train = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val = [rows[i] for i in range(len(rows)) if i in val_idx]
    write_pool(out_root, name, train, val)
    print("%-14s %8d %8d %8d %9d %9d %8d %12d" % (name, per_lang["ccpp"], per_lang["java"], per_lang["js"], len(rows), len(train),
                                                 len(val), split_pairs(train, val)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", default="data/sources_v4")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="cho các pool v4* (không áp cho pool SOTA)")
    ap.add_argument("--seed", type=int, default=42, help="cho các pool v4* (không áp cho pool SOTA)")
    a = ap.parse_args()
    print("%-14s %8s %8s %8s %9s %9s %8s %12s" % ("pool", "ccpp", "java", "js", "tổng", "train", "val", "cặp bị tách"))
    build_sota_pool(a.src_dir, a.out_root)
    for name, parts in POOLS.items():
        build_pool(a.src_dir, a.out_root, name, parts, a.val_ratio, a.seed)


if __name__ == "__main__":
    main()
