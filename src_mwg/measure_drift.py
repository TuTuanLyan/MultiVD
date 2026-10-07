#!/usr/bin/env python3
"""Phase 1 đẩy backbone đi bao xa khỏi trọng số pretrained gốc?

Đây là phép đo phải làm TRƯỚC khi chạy thí nghiệm đổi neo RecAdam. Ý tưởng đổi neo
chỉ có nghĩa nếu hai điểm neo thật sự khác nhau. Nếu Phase 1 gần như không dịch
chuyển backbone thì neo về Phase 1 và neo về pretrained là cùng một chỗ, và thí
nghiệm đó không đáng một giờ GPU nào.

Đại lượng đo là độ dịch tương đối theo từng tensor:

    ||θ_phase1 − θ_pretrained|| / ||θ_pretrained||

Con số cần đối chiếu là giữa CodeBERT và CodeT5+. §34.2 đo được Phase 1 gây hại
−0.0312 trên CWE-078 của CodeT5+ mà head phụ chỉ gỡ lại +0.0056. Nếu độ dịch của
CodeT5+ lớn hơn CodeBERT thì có một cơ chế cụ thể đứng sau con số đó: cùng một
lịch RecAdam, nhưng backbone mạnh bị kéo ra xa hơn khỏi biểu diễn nó vốn có.

Chạy trên CPU, không đụng vào GPU đang chạy.
"""

import argparse
import json
import sys

import torch

sys.path.insert(0, "src")
from model import build_backbone  # noqa: E402


def drift(checkpoint_path, model_name, device="cpu"):
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = blob.get("model_state_dict", blob)
    pretrained = build_backbone(model_name).to(device).state_dict()

    rows, skipped = [], 0
    total_sq, total_norm_sq = 0.0, 0.0
    for name, original in pretrained.items():
        trained = state.get(f"backbone.{name}")
        if trained is None or trained.shape != original.shape:
            skipped += 1
            continue
        if not original.is_floating_point():
            skipped += 1
            continue
        original = original.to(device).float()
        trained = trained.to(device).float()
        difference = (trained - original).norm().item()
        reference = original.norm().item()
        total_sq += difference ** 2
        total_norm_sq += reference ** 2
        if reference > 0:
            rows.append((name, difference / reference, original.numel()))
    return rows, skipped, (total_sq ** 0.5) / (total_norm_sq ** 0.5 or 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", required=True,
                        help="các cặp label=checkpoint=model_name")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    summary = {}
    for pair in args.pairs:
        label, checkpoint, model_name = pair.split("=", 2)
        rows, skipped, overall = drift(checkpoint, model_name)
        weighted = sum(r[1] * r[2] for r in rows) / max(1, sum(r[2] for r in rows))
        summary[label] = {"overall": overall, "weighted_mean": weighted,
                          "tensors": len(rows), "skipped": skipped}
        print(f"\n### {label}  ({model_name})")
        print(f"  tensor so được: {len(rows)}   bỏ qua: {skipped}")
        print(f"  độ dịch TỔNG THỂ (chuẩn Frobenius toàn backbone): {overall:.6f}")
        print(f"  trung bình có trọng số theo số tham số:           {weighted:.6f}")
        rows.sort(key=lambda r: r[1], reverse=True)
        print(f"  {args.top} tensor dịch nhiều nhất:")
        for name, value, numel in rows[:args.top]:
            print(f"     {value:8.4f}  {numel:>10,}  {name}")

    if len(summary) > 1:
        print("\n" + "=" * 62)
        print("ĐỐI CHIẾU")
        print("=" * 62)
        for label, info in summary.items():
            print(f"  {label:<12} tổng thể {info['overall']:.6f}   "
                  f"trọng số {info['weighted_mean']:.6f}")
        print("\n  Neo về pretrained chỉ đáng chạy nếu độ dịch ĐỦ LỚN để hai điểm neo")
        print("  khác nhau thật. Dịch càng lớn thì neo hiện tại càng giữ mô hình ở xa")
        print("  biểu diễn gốc — và đó chính là thứ §34.2 đo được là gây hại.")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
