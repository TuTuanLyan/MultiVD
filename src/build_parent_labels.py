#!/usr/bin/env python3
"""Thay nhãn CWE chi tiết bằng pillar gốc của CWE Research Concept.

Ý tưởng: head phụ hiện học nhãn CWE cụ thể. §34.3 đo được tín hiệu đó gần như hết
giá trị gia tăng trên backbone mạnh — backbone đã tự biểu diễn được lớp đó rồi.
Một nhãn ở mức trừu tượng cao hơn (pillar) là thông tin khác chứ không phải cùng
thông tin ở độ phân giải thấp hơn: nó nói *kiểu sai lầm*, không nói *lỗ hổng nào*.

Chỉ ghi đè trường `cwe_class`, giữ nguyên `cwe` và mọi trường khác, đúng cách
`build_edit_labels.py` đã làm. Nhờ vậy `AUX_MODE=cwe` trên file mới là học tín
hiệu mới mà không phải sửa một dòng code model nào, và so sánh chỉ đổi đúng một
biến.

Hai chỗ phải quyết định vì bảng nguồn không đơn trị:

1. **CWE có nhiều pillar.** CWE-13 thuộc cả CWE-284 lẫn CWE-664. Chọn pillar có
   ID nhỏ nhất — tất định, không phụ thuộc thứ tự đọc file. Số dòng bị ảnh hưởng
   được in ra để biết quyết định này nặng đến đâu.
2. **CWE không có pillar** (`ROOT - No parent`) và CWE không có trong bảng. Gán
   -100 để `ignore_index` của cross-entropy bỏ qua, giống hệt cách các dòng không
   ghép cặp được xử lý ở `build_edit_labels.py`. Không bịa một lớp "khác", vì
   một lớp gom rác sẽ dạy head phụ đúng thứ không nên học.

KHÔNG sửa file nguồn: luôn ghi ra đường dẫn mới.
"""

import argparse
import json
import re
from collections import Counter, defaultdict

IGNORE_INDEX = -100
LINE = re.compile(r"^CWE-\s*(\d+):\s*(?:CWE-\s*(\d+)|\(ROOT)")


def load_parents(path):
    """CWE -> tập pillar gốc. Dòng ROOT ghi nhận nhưng không có pillar."""
    parents = defaultdict(set)
    roots = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = LINE.match(line.strip())
            if not match:
                continue
            child, parent = match.group(1), match.group(2)
            if parent is None:
                roots.add(int(child))
            else:
                parents[int(child)].add(int(parent))
    return parents, roots


def cwe_number(value):
    match = re.search(r"(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parents", default="data/cwe_root_parents.txt")
    args = parser.parse_args()

    if args.output == args.input:
        parser.error("output trùng input — script này không ghi đè file nguồn")

    parents, roots = load_parents(args.parents)
    print(f"bảng ánh xạ: {len(parents)} CWE có pillar, {len(roots)} CWE là gốc")
    multi = {c: p for c, p in parents.items() if len(p) > 1}
    print(f"  trong đó {len(multi)} CWE có nhiều hơn một pillar")

    records = []
    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    seen = Counter(cwe_number(r.get("cwe")) for r in records)
    pillar_of, ambiguous_rows, self_rooted, missing = {}, 0, Counter(), Counter()
    for number in seen:
        options = parents.get(number)
        if not options:
            # Một pillar tự nó không có parent, nên nó nằm ở nhánh ROOT của bảng
            # chứ không nằm ở nhánh ánh xạ. Bản đầu của hàm này gán -100 cho
            # chúng và ném đi 419 dòng CWE-703 của ccpp_common — 14% dữ liệu,
            # và ném đi đúng những dòng đã ở sẵn mức trừu tượng cần đến.
            if number in roots:
                pillar_of[number] = number
                self_rooted[number] = seen[number]
            else:
                missing[number] = seen[number]
            continue
        pillar_of[number] = min(options)
        if len(options) > 1:
            ambiguous_rows += seen[number]

    pillars = sorted(set(pillar_of.values()))
    index_of = {p: i for i, p in enumerate(pillars)}
    print(f"\n{len(seen)} CWE trong dữ liệu -> {len(pillars)} pillar")

    counts = Counter()
    for record in records:
        pillar = pillar_of.get(cwe_number(record.get("cwe")))
        record["cwe_class"] = IGNORE_INDEX if pillar is None else index_of[pillar]
        counts[pillar] += 1

    total = sum(counts.values())
    print(f"\n{'pillar':<12}{'lớp':>5}{'dòng':>8}{'tỉ lệ':>9}")
    for pillar in pillars:
        print(f"CWE-{pillar:<8}{index_of[pillar]:>5}{counts[pillar]:>8}"
              f"{counts[pillar] / total:>8.1%}")
    if counts[None]:
        print(f"{'bỏ qua':<12}{IGNORE_INDEX:>5}{counts[None]:>8}{counts[None] / total:>8.1%}")
    if ambiguous_rows:
        print(f"\n{ambiguous_rows} dòng ({ambiguous_rows / total:.1%}) thuộc CWE có "
              f"nhiều pillar; đã lấy pillar ID nhỏ nhất")
    if self_rooted:
        print(f"{sum(self_rooted.values())} dòng có CWE vốn đã là pillar, ánh xạ về chính nó: "
              f"{', '.join(f'CWE-{c}({n})' for c, n in self_rooted.most_common(8))}")
    if missing:
        print(f"{sum(missing.values())} dòng có CWE không tra được: "
              f"{', '.join(f'CWE-{c}({n})' for c, n in missing.most_common(8))}")

    with open(args.output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nghi {len(records)} dòng ra {args.output}")
    usable = total - counts[None]
    print(f"dùng được cho head phụ: {usable}/{total} ({usable / total:.1%})")
    if len(pillars) < 3:
        print("\nCẢNH BÁO: dưới 3 lớp. Head phụ ở độ phân giải này gần như không "
              "mang thêm thông tin nào so với nhãn nhị phân của task chính.")


if __name__ == "__main__":
    main()
