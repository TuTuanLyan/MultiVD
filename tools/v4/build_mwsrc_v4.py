#!/usr/bin/env python3
"""Ghép các bộ nguồn v4 (theo cặp) thành pool Pha 1 theo quy ước `data/mwsrc_<tên>/fold1/{train,val,test}.jsonl`.

Pha 1 chạy MỘT lần trên pool, ra một checkpoint; Pha 2 dùng lại cho cả 5 fold đích.
`test.jsonl` là BẢN SAO của `val.jsonl` — đúng quy ước các pool nguồn cũ của dự án (đánh giá thật nằm ở
Pha 2 trên tập đích Python).

Chia val: NGẪU NHIÊN THEO HÀNG, tỉ lệ `--val_ratio` (mặc định 0,2), seed 42. In số cặp bị tách giữa
train/val để biết mức rò rỉ song sinh trong val NGUỒN — val nguồn chỉ dùng để CHỌN CHECKPOINT.

`train_mwg.py` chỉ đọc: code, label, lang, pair_id. Vẫn ghi thêm cwe / cwe_id / cwe_class / source_set.
"""
import argparse, collections, json, os, random

POOLS = {
    "v4full":   [("ccpp_primevul_paired_full", "ccpp"), ("java_cleanvul_3-4_full", "java"), ("js_cleanvul_3-4_full", "js")],
    "v4common": [("ccpp_primevul_paired_common", "ccpp"), ("java_cleanvul_3-4_common", "java"), ("js_cleanvul_3-4_common", "js")],
    "v4cwe4":   [("ccpp_primevul_paired_4cwe", "ccpp"), ("js_cleanvul_3-4_4cwe", "js")],
    # common KHÔNG có C/C++ — để so common có và không có ccpp (người dùng nêu 22/09).
    "v4common_nocc": [("java_cleanvul_3-4_common", "java"), ("js_cleanvul_3-4_common", "js")],
}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print("%-14s %8s %8s %8s %9s %9s %8s %12s" % ("pool", "ccpp", "java", "js", "tổng", "train", "val", "cặp bị tách"))
    print("-" * 86)
    for name, parts in POOLS.items():
        rows, per_lang = [], collections.Counter()
        for base, lang in parts:
            recs = load(os.path.join(a.src_dir, base + ".jsonl"))
            metas = load(os.path.join(a.src_dir, base + ".meta.jsonl"))
            for r, m in zip(recs, metas):
                if not m.get("pair_id"):
                    raise ValueError("thiếu pair_id trong %s — pool chỉ dựng từ bộ theo cặp" % base)
                rows.append({"code": r["code"], "label": r["label"], "lang": lang,
                             "cwe": r["cwe"], "cwe_id": r["cwe_id"], "cwe_class": r["cwe_class"],
                             "pair_id": m["pair_id"], "source_set": base})
            per_lang[lang] += len(recs)
        rng = random.Random(a.seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n_val = int(round(len(rows) * a.val_ratio))
        val_idx = set(idx[:n_val])
        train = [rows[i] for i in range(len(rows)) if i not in val_idx]
        val = [rows[i] for i in range(len(rows)) if i in val_idx]
        # số cặp có một nửa ở train, nửa kia ở val
        side = {}
        for i, r in enumerate(rows):
            side.setdefault(r["pair_id"], []).append(i in val_idx)
        split_pairs = sum(1 for v in side.values() if len(v) == 2 and v[0] != v[1])
        d = os.path.join(a.out_root, "mwsrc_" + name, "fold1")
        os.makedirs(d, exist_ok=True)
        for fn, data in (("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", val)):
            with open(os.path.join(d, fn), "w", encoding="utf-8") as fh:
                for r in data:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("%-14s %8d %8d %8d %9d %9d %8d %12d" %
              (name, per_lang["ccpp"], per_lang["java"], per_lang["js"], len(rows), len(train), len(val), split_pairs))


if __name__ == "__main__":
    main()
