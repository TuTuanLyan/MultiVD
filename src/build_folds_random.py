#!/usr/bin/env python3
"""Chia fold NGẪU NHIÊN nhưng không rò rỉ, bằng StratifiedGroupKFold chuẩn.

Vì sao cần bộ này thay cho hai bộ đã có:

`data/sven_python_folds_norm` chia theo TỪNG DÒNG. Corpus dựng từ cặp
vulnerable/fixed nên cách chia đó đẩy bản vá của một hàm test vào train — đo được
**43%** hàm test có bản gần-giống (>0.75) nằm trong train. Không dùng được.

`data/sven_python_twin` không rò rỉ (7%, và 0/384 cặp bị tách) nhưng cách gán fold
là **tất định**: `assign_folds` shuffle rồi `sort(key=len, reverse=True)` và gán
tham lam vào fold nhẹ nhất. Sắp xếp theo cỡ ghi đè lên shuffle, nên ngẫu nhiên chỉ
còn phá hoà giữa các cụm cùng kích thước. Người phản biện có quyền gọi đó là fold
được dàn xếp chứ không phải chia ngẫu nhiên.

Bộ này giữ phần đúng của cả hai: **nhóm các dòng gần-giống lại trước** (nên không
rò rỉ), rồi **giao cho `StratifiedGroupKFold` gán** — một hàm chuẩn của
scikit-learn, có shuffle thật với `random_state`, phân tầng theo CWE, và tôn trọng
ranh giới nhóm. Không có bước cân bằng thủ công nào.

Val cũng tách theo NHÓM bằng `GroupShuffleSplit`, không tách theo dòng — nếu không
thì rò rỉ quay lại giữa train và val, chỗ dùng để chọn checkpoint.

Phần gom cụm dùng lại nguyên logic của build_folds.py: union-find trên cặp
gần-giống, chỉ so trong cùng CWE và cùng dải độ dài.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

LENGTH_BAND = 0.5


def normalize(code):
    return re.sub(r"\s+", " ", code).strip()


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("code", "label", "cwe"):
                if field not in record:
                    raise ValueError(f"{path}:{number}: thiếu trường {field!r}")
            record["_normalized"] = normalize(record["code"])
            records.append(record)
    if not records:
        raise ValueError(f"{path}: không có bản ghi nào")
    return records


def cluster_records(records, threshold):
    """Union-find trên các cặp gần-giống; trả về nhãn cụm cho từng dòng."""
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_cwe = defaultdict(list)
    for index, record in enumerate(records):
        by_cwe[record["cwe"]].append(index)

    for indices in by_cwe.values():
        indices.sort(key=lambda i: len(records[i]["_normalized"]))
        for position, left in enumerate(indices):
            left_length = len(records[left]["_normalized"])
            for right in indices[position + 1:]:
                right_length = len(records[right]["_normalized"])
                # Danh sách đã sắp theo độ dài, nên vượt dải là dừng được luôn.
                if right_length > left_length / LENGTH_BAND:
                    break
                matcher = SequenceMatcher(
                    None, records[left]["_normalized"], records[right]["_normalized"])
                if matcher.quick_ratio() < threshold:
                    continue
                if matcher.ratio() >= threshold:
                    union(left, right)

    return [find(i) for i in range(len(records))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sven_python_folds_norm/data.jsonl")
    parser.add_argument("--output_dir", default="data/sven_python_random")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="phần của train tách ra làm val, tách theo nhóm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records(args.input)
    groups = cluster_records(records, args.threshold)
    sizes = Counter(Counter(groups).values())
    print(f"nạp {len(records)} dòng, gom thành {len(set(groups))} cụm")
    print(f"  phân bố cỡ cụm: {dict(sorted(sizes.items()))}")

    labels = [f"{r['cwe']}|{r['label']}" for r in records]
    splitter = StratifiedGroupKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    output = Path(args.output_dir)
    manifest = {
        "source": args.input, "threshold": args.threshold, "seed": args.seed,
        "n_folds": args.n_folds, "n_records": len(records),
        "n_groups": len(set(groups)),
        "splitter": "sklearn StratifiedGroupKFold(shuffle=True) + GroupShuffleSplit cho val",
        "folds": {},
    }

    for fold, (rest_idx, test_idx) in enumerate(
            splitter.split(records, labels, groups), start=1):
        rest_groups = [groups[i] for i in rest_idx]
        inner = GroupShuffleSplit(
            n_splits=1, test_size=args.val_fraction, random_state=args.seed + fold)
        train_pos, val_pos = next(inner.split(rest_idx, groups=rest_groups))
        train_idx = [rest_idx[p] for p in train_pos]
        val_idx = [rest_idx[p] for p in val_pos]

        fold_dir = output / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            with open(fold_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
                for i in indices:
                    row = {k: v for k, v in records[i].items() if not k.startswith("_")}
                    # Giữ nhãn cụm để kiểm tra được về sau; nó không đi vào huấn luyện.
                    row["pair_id"] = f"g{groups[i]}"
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[name] = len(indices)

        overlap = (set(groups[i] for i in train_idx) | set(groups[i] for i in val_idx)) \
            & set(groups[i] for i in test_idx)
        label_mix = Counter(records[i]["label"] for i in test_idx)
        print(f"fold{fold}: {counts}  nhóm chung train/val với test: {len(overlap)}  "
              f"test labels {dict(sorted(label_mix.items()))}")
        manifest["folds"][f"fold{fold}"] = {**counts, "group_overlap": len(overlap)}

    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"\nghi ra {output}")
    print("nhóm chung phải bằng 0 ở mọi fold — nếu khác thì có rò rỉ, không dùng được.")


if __name__ == "__main__":
    main()
