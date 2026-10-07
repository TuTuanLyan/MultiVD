#!/usr/bin/env python3
"""Dựng pool Pha 1 `js_uncommon_strict` cho khối mwonly5 (người dùng 07/10 15:0x: "tab chung, chạy thêm uncommon 4:1, val Pha 1 chỉ JS";
chọn bản CHẶT).

Tập hàm: các hàng JS của HF `merged_undersampled/full.jsonl` (đọc bằng `build_hf_pools.load_rows`, giữ thứ tự gốc) có mã nguồn thuộc
`data/final_experiment_data/phase1_uncommon_jsonly_strict.jsonl` - bộ đó dựng bằng luật CHẶT 01/10 (`tools/v4/build_final_experiment_data.py
--uncommon_strict_only`: JS full bỏ JS common, bỏ mẫu có bất kỳ nhãn CWE nào thuộc common hoặc nhãn không phải mã CWE; nhãn 0 theo nửa nhãn 1
cùng cặp). Chia ĐÚNG quy tắc mọi pool mwonly5 (`build_hf_pools.py`): random.Random(seed).shuffle, val = round(val_ratio × n) hàng đầu,
test = bản sao val.

    python tools/build_js_uncommon_strict_pool.py --hf_root data/hf_MultiDataSource4VD --out data/mwonly5_sources/js_uncommon_strict
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_hf_pools import load_rows, write_jsonl   # noqa: E402

ORACLE = "data/final_experiment_data/phase1_uncommon_jsonly_strict.jsonl"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle", default=ORACLE, help="file gộp của bộ uncommon chỉ JS CHẶT (luật 01/10) - chỉ dùng tập mã nguồn")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    keep = [json.loads(l)["code"] for l in open(a.oracle, encoding="utf-8") if l.strip()]
    assert len(set(keep)) == len(keep), "bộ chặt có mã nguồn trùng"
    full = load_rows(a.hf_root, "merged_undersampled/full.jsonl", {"js"})
    common = {r["code"] for r in load_rows(a.hf_root, "merged_undersampled/common.jsonl", {"js"})}
    assert len({r["code"] for r in full}) == len(full), "JS full trên HF có mã nguồn trùng"
    rows = [r for r in full if r["code"] in set(keep)]
    assert len(rows) == len(keep), "chỉ khớp %d / %d hàm của bộ chặt" % (len(rows), len(keep))
    assert not any(r["code"] in common for r in rows), "có hàm thuộc JS common"
    pairs = collections.defaultdict(list)
    for r in rows:
        pairs[r["pair_id"]].append(r["label"])
    assert all(p is not None and sorted(v) == [0, 1] for p, v in pairs.items()), "có cặp thiếu nửa"

    random.Random(a.seed).shuffle(rows)
    n_val = int(round(a.val_ratio * len(rows)))
    val, train = rows[:n_val], rows[n_val:]
    base = os.path.join(a.out, "fold1")
    for split, part in (("train", train), ("val", val), ("test", val)):
        write_jsonl(os.path.join(base, split + ".jsonl"), part)
    vpairs = {r["pair_id"] for r in val}
    stat = {"pool": "js_uncommon_strict", "hf_file": "merged_undersampled/full.jsonl", "oracle": a.oracle, "oracle_md5": md5(a.oracle),
            "n": len(rows), "pairs": len(pairs), "train": len(train), "val": len(val),
            "label1_train": sum(r["label"] for r in train), "label1_val": sum(r["label"] for r in val),
            "pairs_split_train_val": len(vpairs & {r["pair_id"] for r in train}),
            "cwe_label1": collections.Counter(r["cwe"] for r in rows if r["label"] == 1).most_common(),
            "rule": "JS của HF merged_undersampled/full có mã thuộc bộ chặt 01/10 (giữ thứ tự HF) rồi như build_hf_pools.py: random.Random(seed).shuffle, "
                    "val = round(val_ratio*n) hàng đầu, test = bản sao val", "seed": a.seed, "val_ratio": a.val_ratio,
            "script": "tools/build_js_uncommon_strict_pool.py", "script_md5": md5(os.path.abspath(__file__)),
            "md5": {s: md5(os.path.join(base, s + ".jsonl")) for s in ("train", "val", "test")}}
    json.dump(stat, open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("js_uncommon_strict n=%d (%d cặp) train=%d (nhãn 1: %d) val=%d (nhãn 1: %d) | cặp bị tách giữa train/val: %d" % (
        len(rows), len(pairs), len(train), stat["label1_train"], len(val), stat["label1_val"], stat["pairs_split_train_val"]))
    print("CWE nhãn 1:", stat["cwe_label1"])


if __name__ == "__main__":
    main()
