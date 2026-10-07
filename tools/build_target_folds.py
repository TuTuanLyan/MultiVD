#!/usr/bin/env python3
"""Dựng tập ĐÍCH Pha 2 chia 5 fold 6:2:2 kiểu XOAY VÒNG như SVEN (người dùng 05/10 16:4x: "lên lịch chạy task nguồn SVEN, đích JS
nhé. JS chia lại 5 fold, tỷ lệ 6:2:2. lưu data thí nghiệm này cẩn thận"; chọn bộ JS full).

Cách chia giống hệt `sven_python_folds_nocomment` (đã kiểm: test của 5 fold rời nhau, hợp lại đủ 760 hàm, val fold k = test fold k+1):
xáo MỘT lần bằng random.Random(seed), cắt thành k khúc liền nhau (khúc đầu dài hơn khi n không chia hết), fold i có test = khúc i,
val = khúc i+1 (vòng), train = các khúc còn lại theo thứ tự khúc. Chia theo HÀM: hai nửa của một cặp lỗi/vá có thể rơi vào hai tập,
y như SVEN (chủ ý của reviewer, CLAUDE.md mục 6). Hàng lấy từ HF giữ thứ tự gốc và chuẩn hoá trường y như tools/build_hf_pools.py
(pair_id = "<lang>:<pair_id>", giữ mọi trường gốc) để khớp từng hàm với pool Pha 1 cùng tên.

    python tools/build_target_folds.py --hf_root data/hf_MultiDataSource4VD --file merged_undersampled/full.jsonl --langs js \\
        --out data/final_experiment_data/js_full_folds_622
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_hf_pools import load_rows, md5, write_jsonl  # noqa: E402

HF_COMMIT = "387d9dffbdedb1e74391851afcee419d97f1e326"


def chunk_bounds(n, k):
    """Biên của k khúc liền nhau phủ n hàng; n % k khúc đầu dài hơn 1 hàng."""
    q, r = divmod(n, k)
    out, start = [], 0
    for i in range(k):
        size = q + (1 if i < r else 0)
        out.append((start, start + size))
        start += size
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_root", required=True)
    ap.add_argument("--file", required=True, help="file trong hf_root, vd merged_undersampled/full.jsonl")
    ap.add_argument("--langs", required=True, help="lang giữ lại, phẩy ngăn cách")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check_pool", default=None, help="pool Pha 1 cùng hàng (thư mục có fold1/train,val) để đối chiếu tập hàm")
    a = ap.parse_args()

    rows = load_rows(a.hf_root, a.file, set(a.langs.split(",")))
    write_jsonl(os.path.join(a.out, "data.jsonl"), rows)                 # mọi hàng, đúng thứ tự file HF
    order = list(range(len(rows)))
    random.Random(a.seed).shuffle(order)
    chunks = [order[s:e] for s, e in chunk_bounds(len(rows), a.k)]
    for c in chunks:
        assert c, "khúc rỗng"
    folds = []
    for i in range(a.k):
        test, val = chunks[i], chunks[(i + 1) % a.k]
        train = [j for c in range(a.k) if c not in (i, (i + 1) % a.k) for j in chunks[c]]
        assert len(set(train) | set(val) | set(test)) == len(rows) and not (set(train) & set(val)) and not (set(train) & set(test))
        base = os.path.join(a.out, "fold%d" % (i + 1))
        parts = {"train": train, "val": val, "test": test}
        for split, ids in parts.items():
            write_jsonl(os.path.join(base, split + ".jsonl"), [rows[j] for j in ids])
        # nửa kia của cặp nằm ở train: tính chất được giữ CỐ Ý như SVEN, ghi lại để đọc kết quả
        train_pairs = collections.Counter(rows[j]["pair_id"] for j in train if rows[j]["pair_id"])
        stat = {"fold": i + 1}
        for split, ids in parts.items():
            lab = collections.Counter(rows[j]["label"] for j in ids)
            other_half = sum(1 for j in ids if rows[j]["pair_id"] and train_pairs[rows[j]["pair_id"]] > (1 if split == "train" else 0))
            stat[split] = {"n": len(ids), "label1": lab[1], "label0": lab[0], "pair_other_half_in_train": other_half,
                           "md5": md5(os.path.join(base, split + ".jsonl"))}
        folds.append(stat)
        print("fold%d  " % (i + 1) + " | ".join("%s %d (1/0 = %d/%d, nửa cặp kia ở train %d)" % (
            s, stat[s]["n"], stat[s]["label1"], stat[s]["label0"], stat[s]["pair_other_half_in_train"]) for s in ("train", "val", "test")))
    tests = [set(chunks[i]) for i in range(a.k)]
    assert set().union(*tests) == set(range(len(rows))) and sum(len(t) for t in tests) == len(rows), "test các fold phải rời nhau và phủ đủ"

    check = None
    if a.check_pool:
        key = lambda r: (r["code"], r["label"])
        pool = [json.loads(l) for s in ("train", "val") for l in open(os.path.join(a.check_pool, "fold1", s + ".jsonl"), encoding="utf-8")]
        check = {"pool": a.check_pool, "same_rows": sorted(map(key, pool)) == sorted(map(key, rows))}
        print("đối chiếu với %s: %s" % (a.check_pool, "TRÙNG tập hàm" if check["same_rows"] else "LỆCH"))
        assert check["same_rows"]

    lab = collections.Counter(r["label"] for r in rows)
    manifest = {"source": "huggingface.co/datasets/LyanDumpling/MultiDataSource4VD", "hf_commit": HF_COMMIT, "file": a.file,
                "file_md5": md5(os.path.join(a.hf_root, a.file)), "langs_kept": sorted(a.langs.split(",")), "n": len(rows),
                "label1": lab[1], "label0": lab[0], "pairs": len({r["pair_id"] for r in rows if r["pair_id"]}),
                "k": a.k, "seed": a.seed, "chunk_sizes": [len(c) for c in chunks],
                "rule": "random.Random(seed).shuffle(range(n)) một lần; k khúc liền nhau; fold i: test = khúc i, val = khúc i+1 (vòng), train = phần còn lại",
                "data_md5": md5(os.path.join(a.out, "data.jsonl")), "check": check, "folds": folds,
                "script": "tools/build_target_folds.py", "script_md5": hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    json.dump(manifest, open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("n=%d (1/0 = %d/%d), khúc %s -> %s" % (len(rows), lab[1], lab[0], manifest["chunk_sizes"], a.out))


if __name__ == "__main__":
    main()
