#!/usr/bin/env python3
"""Dựng pool Pha 1 `js_common_ext` (common MỞ RỘNG) cho khối mwonly5 (người dùng 07/10 17:0x: "chạy thêm vào nhánh chính 1 bản SOTA nguồn js
common, nhưng bản common này dùng: cả những CWE có đa nhãn, trong đó chỉ cần duy nhất 1 nhãn nằm trong common thì giữ cả hàm; cả những CWE không có
thông tin CWE, unknown").

Luật common GỐC (`tools/build_sources_v2.py:subset_common`, luật R1-R6 của `tools/cwe_rules.py` đọc `data/cwe_spec/cwec_v4.20.xml`, đích Python):
giữ hàm khi MỌI nhãn CWE được giữ; nhãn không phải mã CWE (unknown / NVD-CWE-* / rỗng, luật R1) ⇒ bỏ.
Luật MỞ RỘNG ở đây: giữ hàm khi CÓ ÍT NHẤT MỘT nhãn được giữ, HOẶC có nhãn R1 (không có thông tin CWE). Phần bị bỏ là ĐÚNG phần bù đã dựng làm
uncommon chặt (mọi nhãn là mã CWE thật và không nhãn nào thuộc common) ⇒ js_common_ext ⊔ uncommon chặt = JS full (kiểm bằng assert).

Nhãn đầy đủ của từng hàm lấy từ `data/sources_v4/js_cleanvul_3-4_full.meta.jsonl` (trường `cwe_labels`, như `tools/v4/build_final_experiment_data.py`);
hàng lấy từ HF `merged_undersampled/full.jsonl` (giữ thứ tự HF, `build_hf_pools.load_rows`). Chia như mọi pool mwonly5: random.Random(seed).shuffle,
val = round(val_ratio × n) hàng đầu, test = bản sao val.

    python tools/build_js_common_ext_pool.py --hf_root data/hf_MultiDataSource4VD --out data/mwonly5_sources/js_common_ext
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_hf_pools import load_rows, write_jsonl   # noqa: E402
from build_sources_v2 import get_rules              # noqa: E402

META = "data/sources_v4/js_cleanvul_3-4_full.meta.jsonl"
STRICT = "data/final_experiment_data/phase1_uncommon_jsonly_strict.jsonl"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xml", default="data/cwe_spec/cwec_v4.20.xml")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rules = get_rules(a.xml, "Python")
    labels = {}
    for line in open(META, encoding="utf-8"):
        m = json.loads(line)
        if m.get("row_id"):
            labels[m["row_id"]] = m.get("cwe_labels") or [None]
    full = load_rows(a.hf_root, "merged_undersampled/full.jsonl", {"js"})
    common = {r["code"] for r in load_rows(a.hf_root, "merged_undersampled/common.jsonl", {"js"})}
    strict = {json.loads(l)["code"] for l in open(STRICT, encoding="utf-8") if l.strip()}
    assert all(r["id"] in labels for r in full), "có hàm JS full không tra được nhãn CWE"

    why = collections.Counter()
    rows = []
    for r in full:
        d = [rules.decide(x if x is not None else "") for x in labels[r["id"]]]
        any_common = any(k for k, _, _ in d)
        any_unknown = any(rule == "R1" for _, rule, _ in d)
        keep = any_common or any_unknown
        kind = ("mọi nhãn common" if all(k for k, _, _ in d) else "có nhãn common (lẫn)" if any_common else "") + \
               (" + có unknown" if any_unknown else "")
        why[(kind.strip(" +") or "chỉ CWE không thuộc common", keep)] += 1
        if keep:
            rows.append(r)
    codes = {r["code"] for r in rows}
    assert common <= codes, "common gốc phải nằm trọn trong bản mở rộng"
    assert not codes & strict and len(codes) + len(strict) == len(full), "bản mở rộng phải là phần bù đúng của uncommon chặt trong JS full"
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
    stat = {"pool": "js_common_ext", "hf_file": "merged_undersampled/full.jsonl", "meta": META, "meta_md5": md5(META), "xml": a.xml,
            "n": len(rows), "pairs": len(pairs), "n_common_goc": len(common), "n_them": len(rows) - len(common),
            "train": len(train), "val": len(val), "label1_train": sum(r["label"] for r in train), "label1_val": sum(r["label"] for r in val),
            "pairs_split_train_val": len(vpairs & {r["pair_id"] for r in train}),
            "phan_loai": [{"loai": k, "giu": g, "n": n} for (k, g), n in sorted(why.items(), key=str)],
            "rule": "giữ hàm JS full có ÍT NHẤT MỘT nhãn được luật R1-R6 giữ HOẶC có nhãn R1 (unknown / NVD-CWE-* / rỗng); rồi như build_hf_pools.py: "
                    "random.Random(seed).shuffle, val = round(val_ratio*n) hàng đầu, test = bản sao val",
            "seed": a.seed, "val_ratio": a.val_ratio, "script": "tools/build_js_common_ext_pool.py", "script_md5": md5(os.path.abspath(__file__)),
            "md5": {s: md5(os.path.join(base, s + ".jsonl")) for s in ("train", "val", "test")}}
    json.dump(stat, open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for (k, g), n in sorted(why.items(), key=str):
        print("  %-36s %s %4d hàm" % (k, "GIỮ" if g else "bỏ ", n))
    print("js_common_ext n=%d (%d cặp; common gốc %d + thêm %d) train=%d (nhãn 1: %d) val=%d (nhãn 1: %d) | cặp bị tách train/val: %d" % (
        len(rows), len(pairs), len(common), len(rows) - len(common), len(train), stat["label1_train"], len(val), stat["label1_val"],
        stat["pairs_split_train_val"]))


if __name__ == "__main__":
    main()
