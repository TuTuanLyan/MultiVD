#!/usr/bin/env python3
"""Thay nhãn CWE của source bằng nhãn suy ra từ diff của chính cặp vul/fix.

Vì sao cần cái này: §34.3 đo được rằng giá trị gia tăng của head phụ —
`Δ(cwe) − Δ(none)` — là +0.076 và +0.080 trên hai lớp hiếm của CodeBERT nhưng chỉ
−0.014 và +0.012 trên CodeT5+. Nghĩa là tín hiệu CWE **chỉ có tác dụng khi backbone
chưa biểu diễn được lớp đó**. Trên backbone mạnh, nhãn CWE là thông tin nó đã có,
nên head phụ không thêm gì mà chỉ nhiễu.

Hướng đi là tìm tín hiệu phụ mà backbone **chưa** có. Số dòng phải sửa để vá lỗi là
một ứng viên: nó không nằm trong bất kỳ mục tiêu pretrain nào, và nó suy ra được
hoàn toàn từ dữ liệu đã có — source ghép cặp liền kề ở 638 cặp, phủ 99% số dòng.

Đo trên `data/train_ccpp_js.jsonl`, chia bốn nhóm cho phân bố 30/22/27/21 phần
trăm — cân bằng hơn hẳn phân bố CWE gốc, và đủ xa ngưỡng suy biến.

Thiết kế cố ý tối giản: script chỉ **ghi đè trường `cwe_class`** bằng nhóm kích
thước sửa đổi. `resolve_cwe_class` trong train_transfer.py vốn ưu tiên `cwe_class`
khi trường này có sẵn, nên chạy với `AUX_MODE=cwe` là head phụ học tín hiệu mới —
**không phải sửa dòng code model nào**. Nhờ vậy phép so "cùng kiến trúc, khác tín
hiệu" là sạch: chỉ đúng một biến thay đổi.

Cảnh báo đã đo: 26% số cặp có bản lỗi dài hơn 60 dòng. Ở đó chỗ sửa có thể nằm
ngoài cửa sổ 512 token, và nhãn trở thành thứ model không nhìn thấy được. Đây là
giả thuyết cần kiểm, không phải khiếm khuyết đã biết — nhưng phải theo dõi, vì
§18.3 cho thấy giả thuyết truncation từng nghe hợp lý mà đo ra chỉ 1.8%.
"""

import argparse
import collections
import difflib
import json

# Ranh giới nhóm chọn từ tứ phân vị đo được (q1=1, median=3, q3=8), không phải
# số tròn tự nghĩ ra.
def bucket(changed_lines):
    if changed_lines <= 1:
        return 0
    if changed_lines <= 3:
        return 1
    if changed_lines <= 10:
        return 2
    return 3


BUCKET_NAMES = {0: "1 dòng", 1: "2-3 dòng", 2: "4-10 dòng", 3: ">10 dòng"}


def changed_line_count(vulnerable, fixed):
    left, right = vulnerable.splitlines(), fixed.splitlines()
    matcher = difflib.SequenceMatcher(None, left, right)
    return sum(
        max(op[2] - op[1], op[4] - op[3])
        for op in matcher.get_opcodes()
        if op[0] != "equal"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    # Cặp nằm liền kề nhau: nhãn khác nhau và cùng CWE. Dòng nào không ghép được
    # thì gán -100, đúng giá trị ignore_index mà cross_entropy dùng, nên nó không
    # đóng góp vào loss thay vì âm thầm thành một lớp giả.
    for record in records:
        record["cwe_class"] = -100

    paired = 0
    index = 0
    while index < len(records) - 1:
        left, right = records[index], records[index + 1]
        if left.get("label") != right.get("label") and left.get("cwe") == right.get("cwe"):
            vulnerable, fixed = (left, right) if left["label"] == 1 else (right, left)
            label = bucket(changed_line_count(vulnerable["code"], fixed["code"]))
            left["cwe_class"] = right["cwe_class"] = label
            paired += 1
            index += 2
        else:
            index += 1

    with open(args.output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = collections.Counter(r["cwe_class"] for r in records)
    unlabelled = counts.pop(-100, 0)
    total = sum(counts.values())
    print(f"nguồn   : {args.input}")
    print(f"cặp     : {paired}  ({2 * paired}/{len(records)} dòng = "
          f"{100 * 2 * paired / len(records):.0f}%)")
    print(f"bỏ nhãn : {unlabelled} dòng không ghép cặp được (đặt -100, không vào loss)")
    for key in sorted(counts):
        print(f"  lớp {key} ({BUCKET_NAMES[key]:<9}) {counts[key]:5d} dòng  "
              f"{100 * counts[key] / total:4.0f}%")
    largest = 100 * max(counts.values()) / total
    print(f"lớp lớn nhất chiếm {largest:.0f}% — "
          f"{'cân bằng, dùng được' if largest < 60 else 'SUY BIẾN, không dùng'}")
    print(f"ghi ra  : {args.output}")
    print("\nChạy với AUX_MODE=cwe và PHASE1_DATA_PATH trỏ vào file này; "
          "không cần sửa code model.")


if __name__ == "__main__":
    main()
