#!/usr/bin/env python3
"""Chốt chặn chất lượng Phase 1: nhánh nào có checkpoint nguồn hỏng thì loại.

Vì sao cần: `latent_bottleneck` của CodeT5-base dừng Phase 1 ở **epoch 1** với val
Macro-F1 **0.5079** — ngang ngẫu nhiên. Checkpoint nguồn gần như chưa được huấn
luyện, nên con số Phase 2 của nó (−0.2732) đo một Phase 1 hỏng chứ không đo phương
pháp. Không ai phát hiện ra cho tới khi đọc log bằng tay.

Một nhánh có Phase 1 hỏng phải bị **loại khỏi bảng**, không phải báo cáo như kết
quả âm — hai thứ đó dẫn tới hai kết luận khác nhau: "phương pháp không hiệu quả"
so với "lần chạy này hỏng, chạy lại".

Đọc thẳng từ checkpoint chứ không parse log, vì log có thể bị xoay vòng hoặc mất.
"""

import argparse
import glob
import os

import torch

FAIL_F1 = 0.55      # dưới mức này thì head nhị phân của Phase 1 gần như đoán mù
MIN_EPOCH = 2       # best epoch 1 nghĩa là không epoch nào cải thiện được


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_prefix", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_root", default="model")
    args = parser.parse_args()

    pattern = f"{args.model_root}/{args.run_prefix}_*/transfer_*/seed_{args.seed}/source/best.pt"
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"khong tim thay checkpoint nao khop {pattern}")
        return

    print(f"\n{'run / nhanh':<52}{'best epoch':>12}{'val Macro-F1':>14}  ket luan")
    print("-" * 100)
    bad = []
    for path in paths:
        parts = path.split(os.sep)
        label = f"{parts[-5]}/{parts[-4]}"
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:                                   # noqa: BLE001
            print(f"{label:<52}{'—':>12}{'—':>14}  KHONG DOC DUOC: {error}")
            bad.append(label)
            continue
        epoch = blob.get("best_epoch")
        score = blob.get("best_val_macro_f1")
        if score is None:
            verdict, flag = "thieu best_val_macro_f1", True
        elif score < FAIL_F1:
            verdict, flag = f"PHASE 1 HONG — val {score:.4f} < {FAIL_F1}", True
        elif epoch is not None and epoch < MIN_EPOCH:
            verdict, flag = f"PHASE 1 HONG — dung o epoch {epoch}", True
        else:
            verdict, flag = "dung duoc", False
        print(f"{label:<52}{str(epoch):>12}{(f'{score:.4f}' if score is not None else '—'):>14}"
              f"  {verdict}")
        if flag:
            bad.append(label)

    print("-" * 100)
    if bad:
        print(f"\n{len(bad)} nhanh co Phase 1 hong — LOAI khoi bang, khong bao cao nhu ket qua am:")
        for label in bad:
            print(f"   {label}")
        print("\nChay lai Phase 1 cua chung (doi seed hoac tang patience) roi chay lai Phase 2.")
    else:
        print("\nMoi nhanh co Phase 1 dung duoc.")


if __name__ == "__main__":
    main()
