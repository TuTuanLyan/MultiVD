#!/usr/bin/env python3
"""Dựng 6 pool Pha 1 CHIA CỐ ĐỊNH từ bộ HuggingFace LyanDumpling/MultiDataSource4VD (khối mwonly5, người dùng 04/10 21:2x:
"Lấy HuggingFace là cái chuẩn ... Để 15% và xuất ra lại file cố định, tôi sẽ update vào HF sau"; 21:3x: "Dùng các bộ trong
merged undersample và lọc lại ra thành các source theo lang"). Mỗi pool = các hàng của merged_undersampled/<mức>.jsonl có lang thuộc
tập cho trước, GIỮ thứ tự gốc.

Quy tắc chia giữ ĐÚNG src_mwonly/dataset/pool_from_export.py (nhánh refactor): nối các file theo thứ tự cho trước, xáo bằng
random.Random(seed), val = phần đầu round(val_ratio × n) hàng (chia ngẫu nhiên theo HÀM, cặp có thể bị tách), test = bản sao val,
pair_id đổi thành "<lang>:<pair_id>". Khác duy nhất: giữ thêm mọi trường gốc (id, cwe_id, cwe_class, cve, project, ...) để truy ngược
khi tải lên HF - thứ tự xáo chỉ phụ thuộc số hàng và seed nên không đổi phép chia (kiểm bằng --check_against).

    python tools/build_hf_pools.py --hf_root data/hf_MultiDataSource4VD --out_root data/mwonly5_sources
(--hf_root phải là bản tải ĐÚNG commit HF ghi trong MANIFEST.json; kiểm sha256 với LFS oid trước khi dựng)
"""
import argparse
import collections
import hashlib
import json
import os
import random

# tên pool -> (file của HF, tập lang giữ lại); hàng giữ đúng thứ tự trong file
POOLS = {
    "js_full":       ("merged_undersampled/full.jsonl",   {"js"}),
    "js_common":     ("merged_undersampled/common.jsonl", {"js"}),
    "js_4cwe":       ("merged_undersampled/4cwe.jsonl",   {"js"}),
    "jsccpp_full":   ("merged_undersampled/full.jsonl",   {"js", "ccpp"}),
    "jsccpp_common": ("merged_undersampled/common.jsonl", {"js", "ccpp"}),
    "jsccpp_4cwe":   ("merged_undersampled/4cwe.jsonl",   {"js", "ccpp"}),
    # 05/10 người dùng: "sau bộ các task chính, chạy thêm nguồn CCPP"
    "ccpp_full":     ("merged_undersampled/full.jsonl",   {"ccpp"}),
    "ccpp_common":   ("merged_undersampled/common.jsonl", {"ccpp"}),
    "ccpp_4cwe":     ("merged_undersampled/4cwe.jsonl",   {"ccpp"}),
    # 05/10 người dùng: "chạy thêm nguồn java + js common" - Java trong common.jsonl là đủ 5 184 hàm (CleanVul Java không có CWE nên
    # common = full ở phần Java), JS là 990 hàm common
    "jsjava_common": ("merged_undersampled/common.jsonl", {"js", "java"}),
    # 05/10 16:4x người dùng: "lên lịch chạy task nguồn SVEN, đích JS" - nguồn Pha 1 = toàn bộ 760 hàm SVEN Python đã xoá comment
    # (bản HF trùng từng byte data/final_experiment_data/sven_python_folds_nocomment/data.jsonl), chia 85/15 như mọi nguồn
    "sven_python":   ("sven_python_folds_nocomment/data.jsonl", {"python"}),
}
HEAD = ("id", "code", "label", "lang", "cwe", "cwe_id", "cwe_class", "pair_id", "source_set")


def load_rows(hf_root, rel, langs):
    """Hàng có lang thuộc `langs`. Phần C/C++ của merged_undersampled có hàm lẻ (pair_id rỗng) - giữ nguyên là mẫu lẻ
    (như --allow_unpaired của pool_from_export.py); cặp nào có pair_id thì phải đủ nhãn 0/1."""
    rows = []
    for line in open(os.path.join(hf_root, rel), encoding="utf-8"):
        r = json.loads(line)
        if r["lang"] not in langs:
            continue
        out = {"id": r.get("id"), "code": r["code"], "label": int(r["label"]), "lang": r["lang"], "cwe": r.get("cwe", ""),
               "cwe_id": r.get("cwe_id"), "cwe_class": r.get("cwe_class"),
               "pair_id": "%s:%s" % (r["lang"], r["pair_id"]) if r.get("pair_id") else None,
               "source_set": rel[:-len(".jsonl")]}
        out.update({k: v for k, v in r.items() if k not in out and k != "pair_id"})   # trường gốc còn lại
        rows.append(out)
    labs = collections.defaultdict(list)
    for r in rows:
        if r["pair_id"] is not None:
            labs[r["pair_id"]].append(r["label"])
    assert all(sorted(v) == [0, 1] for v in labs.values()), "cặp không đủ nhãn 0/1"
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_root", required=True, help="bản tải về của LyanDumpling/MultiDataSource4VD (giữ nguyên cây thư mục)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None, help="chỉ dựng các pool này (phẩy ngăn cách); pool khác và mục của chúng trong MANIFEST giữ nguyên")
    ap.add_argument("--check_against", default=None, help="thư mục mwsrc_<tên> do pool_from_export.py dựng (để đối chiếu phép chia)")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    mpath = os.path.join(a.out_root, "MANIFEST.json")
    old = json.load(open(mpath, encoding="utf-8"))["pools"] if (only and os.path.exists(mpath)) else []
    manifest = [m for m in old if m["pool"] not in (only or set())]
    for name, (rel, langs) in POOLS.items():
        if only and name not in only:
            continue
        rows = load_rows(a.hf_root, rel, langs)
        random.Random(a.seed).shuffle(rows)
        n_val = int(round(a.val_ratio * len(rows)))
        val, train = rows[:n_val], rows[n_val:]
        base = os.path.join(a.out_root, name, "fold1")
        for split, part in (("train", train), ("val", val), ("test", val)):
            write_jsonl(os.path.join(base, split + ".jsonl"), part)
        pairs_val = {r["pair_id"] for r in val if r["pair_id"]}
        split_pairs = len(pairs_val & {r["pair_id"] for r in train if r["pair_id"]})
        stat = {"pool": name, "file": rel, "langs_kept": sorted(langs), "unpaired": sum(r["pair_id"] is None for r in rows), "n": len(rows), "train": len(train), "val": len(val),
                "langs": dict(collections.Counter(r["lang"] for r in rows)),
                "label1_train": sum(r["label"] for r in train), "label1_val": sum(r["label"] for r in val),
                "pairs_split_train_val": split_pairs,
                "md5": {s: md5(os.path.join(base, s + ".jsonl")) for s in ("train", "val", "test")}}
        manifest.append(stat)
        lab = collections.Counter((r["lang"], r["label"]) for r in rows)
        print("%-14s n=%5d train=%5d val=%4d nhãn(lang,label)=%s lẻ=%d | cặp bị tách giữa train/val: %d"
              % (name, len(rows), len(train), len(val), dict(sorted(lab.items())), stat["unpaired"], split_pairs))
    json.dump({"source": "huggingface.co/datasets/LyanDumpling/MultiDataSource4VD", "hf_commit": "387d9dffbdedb1e74391851afcee419d97f1e326", "val_ratio": a.val_ratio, "seed": a.seed,
               "rule": "lọc merged_undersampled/<mức> theo lang (giữ thứ tự) rồi như pool_from_export.py: random.Random(seed).shuffle, val = round(val_ratio*n) hàng đầu, test = bản sao val",
               "pools": manifest}, open(os.path.join(a.out_root, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
