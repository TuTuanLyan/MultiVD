#!/usr/bin/env python3
"""Dựng pool Pha 1 từ bộ nguồn chuẩn đã xuất (`export_clean_sources.py`, bản công khai: HuggingFace
LyanDumpling/MultiDataSource4VD, thư mục `by_source/<nguồn>/{full,common,4cwe}.jsonl`) theo ĐÚNG quy tắc pool SOTA:
nối các file theo thứ tự cho trước, xáo bằng random.Random(seed), val = 10 % đầu (chia ngẫu nhiên theo hàm, cặp có thể
bị tách), test = bản sao val. Mặc định chỉ nhận bộ theo cặp (mỗi pair_id đủ nhãn 0 và 1); --allow_unpaired nhận thêm hàm
không cặp (vd. C/C++ PrimeVul đầy đủ đã undersample) — hàm đó là mẫu lẻ, pair loss chỉ áp cho các cặp.

    python dataset/pool_from_export.py --name hfjsCcc_rand --out_root data \\
        EXPORT/by_source/cleanvul_js/full.jsonl EXPORT/by_source/primevul_paired/full.jsonl
    -> data/mwsrc_hfjsCcc_rand/fold1/{train,val,test}.jsonl

Với `cleanvul_js/common.jsonl` + `cleanvul_java/full.jsonl` cho ra đúng thứ tự hàm và nhãn của pool SOTA `mwsrc_v4jsCjv_rand`.
"""
import argparse
import collections
import json
import os
import random

from build_pools import split_pairs, write_pool


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="các file by_source/<nguồn>/<loại>.jsonl, theo thứ tự nối")
    ap.add_argument("--name", required=True, help="tên pool -> <out_root>/mwsrc_<name>")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=36)
    ap.add_argument("--allow_unpaired", action="store_true", help="nhận hàm có pair_id rỗng (không cặp)")
    a = ap.parse_args()
    rows = []
    for path in a.files:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("pair_id") and not a.allow_unpaired:
                raise ValueError("thiếu pair_id trong %s — pool chỉ dựng từ bộ theo cặp (thêm --allow_unpaired)" % path)
            rows.append(dict(code=r["code"], label=int(r["label"]), lang=r["lang"], cwe=r.get("cwe", ""),
                             pair_id=f"{r['lang']}:{r['pair_id']}" if r.get("pair_id") else None, source_set="/".join(path.split(os.sep)[-2:])[:-len(".jsonl")]))
    labs = collections.defaultdict(list)
    for r in rows:
        if r["pair_id"] is not None:
            labs[r["pair_id"]].append(r["label"])
    assert all(sorted(v) == [0, 1] for v in labs.values()), "cặp không đủ nhãn 0/1"
    random.Random(a.seed).shuffle(rows)
    n_val = int(round(a.val_ratio * len(rows)))
    val, train = rows[:n_val], rows[n_val:]
    write_pool(a.out_root, a.name, train, val)
    langs = collections.Counter(r["lang"] for r in rows)
    paired = lambda rs: [r for r in rs if r["pair_id"] is not None]
    print("mwsrc_%s: %s | tổng %d (không cặp %d), train %d, val %d, cặp bị tách %d"
          % (a.name, dict(langs), len(rows), sum(r["pair_id"] is None for r in rows), len(train), len(val),
             split_pairs(paired(train), paired(val))))


if __name__ == "__main__":
    main()
