#!/usr/bin/env python3
"""Cực tiểu Phase 1 của backbone nào NHỌN hơn? — phép thử trước khi tính đến SAM.

SAM (Foret et al., ICLR 2021) cực tiểu giá trị lớn nhất của loss trong một lân cận
bán kính ρ quanh trọng số, tức nó đi tìm cực tiểu **phẳng**. Nó chỉ đáng chạy nếu
cực tiểu hiện tại thật sự nhọn — và nhọn hơn ở đúng backbone đang hỏng.

Vì sao câu hỏi này có nghĩa ở đây, chứ không phải thử SAM cho có: §40 đo trực tiếp
trên trọng số rằng Phase 1 đẩy CodeT5+ đi xa khỏi trọng số gốc **gấp 1.86 lần**
CodeBERT, và head phụ hoạt động **bằng cách hãm dịch chuyển** (−11.1% trên CodeBERT
so với chỉ −5.7% trên CodeT5+). Cực tiểu phẳng theo nghĩa của SAM chính là cực tiểu
**ít nhạy với dịch chuyển trọng số**. Nếu Phase 1 của CodeT5+ nhọn hơn thì Phase 2
phá nó nhiều hơn, và SAM có một mục tiêu thật.

Hai chẩn đoán đang cạnh tranh và phép đo này phân biệt chúng:

  §34  tín hiệu CWE THỪA vì backbone mạnh đã biểu diễn được lớp đó
       -> SAM không chữa được, vì nó không đổi tín hiệu
  §40  Phase 1 đẩy backbone mạnh ra xa, nghiệm nhọn nên Phase 2 phá nhiều
       -> SAM có thể chữa

Đo hai đại lượng ở mỗi bán kính, cả hai chuẩn hoá theo ||w|| nên so được giữa các
backbone có số tham số khác nhau:

  ngẫu nhiên   trung bình L(w+ε) − L(w) trên k hướng ngẫu nhiên, ||ε|| = ρ·||w||
  đối kháng    L(w+ε) − L(w) với ε = ρ·||w||·g/||g||  — chính là bước leo của SAM,
               và là đại lượng SAM thật sự cực tiểu

Chỉ nạp checkpoint và chạy forward/backward, KHÔNG huấn luyện gì. Vài phút GPU.
"""

import argparse
import copy
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")

from dataset import CodeDataset                       # noqa: E402
from torch.utils.data import DataLoader               # noqa: E402
from transformers import AutoTokenizer                # noqa: E402

from train_transfer import (                          # noqa: E402
    load_jsonl, make_model, split_source_records, adopt_checkpoint_shape,
)


def build_val_loader(args, tokenizer):
    """Đúng tập val mà Phase 1 đã dùng — cùng hàm chia, cùng seed."""
    records = load_jsonl(args.data_path)
    _, val_records = split_source_records(records, args.seed)
    dataset = CodeDataset(val_records, tokenizer, args.max_length, args.truncation_strategy)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


@torch.no_grad()
def binary_loss(model, loader, device):
    total, seen = 0.0, 0
    model.eval()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(ids, mask, return_cwe=False)
        loss = F.cross_entropy(out["vul_logits"], labels, reduction="sum")
        total += loss.item()
        seen += labels.numel()
    return total / max(1, seen)


def grad_direction(model, loader, device):
    """Gradient của loss theo trọng số, chuẩn hoá — hướng leo của SAM."""
    model.eval()
    model.zero_grad(set_to_none=True)
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(ids, mask, return_cwe=False)
        F.cross_entropy(out["vul_logits"], labels).backward()
    grads = [(p, p.grad.detach().clone()) for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    norm = torch.sqrt(sum((g ** 2).sum() for _, g in grads))
    model.zero_grad(set_to_none=True)
    return grads, norm


def weight_norm(model):
    return torch.sqrt(sum((p.detach() ** 2).sum()
                          for p in model.parameters() if p.requires_grad))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint nguồn Phase 1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--data_path", default="data/train_ccpp_js.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--truncation_strategy", default="head_middle_tail")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    # Mã gốc của Google dùng ||eps|| = rho TUYỆT ĐỐI: dual_vector chuẩn hoá gradient
    # về chuẩn 1 rồi nhân rho, nên ||eps|| đúng bằng rho và KHÔNG tỉ lệ theo ||w||.
    # Ở đây mặc định là 'relative' vì mục đích của phép đo là SO GIỮA các backbone,
    # mà chúng có ||w|| khác nhau. Hai quy ước không được lẫn: các giá trị rho trong
    # bài báo (0.05, 0.1) là theo 'absolute'. Xem docs/SAM_REFERENCE.md.
    parser.add_argument("--rho_mode", choices=("relative", "absolute"), default="relative",
                        help="relative: ||eps||=rho*||w|| (so giữa backbone). "
                             "absolute: ||eps||=rho (đúng quy ước bài báo)")
    parser.add_argument("--n_random", type=int, default=3)
    parser.add_argument("--aux_mode", default="cwe")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.num_cwes = adopt_checkpoint_shape(args.checkpoint, device)
    args.phase = "phase2"
    model = make_model(args.model_name, device, args)
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    loader = build_val_loader(args, tokenizer)

    base_loss = binary_loss(model, loader, device)
    w_norm = weight_norm(model).item()
    print(f"\n### {args.model_name}   pooling={args.pooling}")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  best epoch     {blob.get('best_epoch')}   val Macro-F1 {blob.get('best_val_macro_f1'):.4f}")
    print(f"  ||w||          {w_norm:.2f}")
    print(f"  loss tại w     {base_loss:.6f}\n")

    grads, g_norm = grad_direction(model, loader, device)
    saved = copy.deepcopy(model.state_dict())

    print(f"  che do rho: {args.rho_mode}"
          f"   ({'||eps|| = rho*||w||' if args.rho_mode == 'relative' else '||eps|| = rho, dung quy uoc bai bao'})")
    print(f"  {'rho':>7}{'||eps||':>11}{'rho tuyet doi':>15}{'ngẫu nhiên Δloss':>20}{'đối kháng Δloss':>19}")
    for rho in args.rhos:
        eps_norm = rho * w_norm if args.rho_mode == "relative" else rho

        increases = []
        for k in range(args.n_random):
            torch.manual_seed(args.seed + 1000 * k)
            with torch.no_grad():
                directions = [torch.randn_like(p) for p in model.parameters() if p.requires_grad]
                d_norm = torch.sqrt(sum((d ** 2).sum() for d in directions))
                idx = 0
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(directions[idx] * (eps_norm / d_norm))
                        idx += 1
            increases.append(binary_loss(model, loader, device) - base_loss)
            model.load_state_dict(saved)

        with torch.no_grad():
            for p, g in grads:
                p.add_(g * (eps_norm / g_norm))
        adversarial = binary_loss(model, loader, device) - base_loss
        model.load_state_dict(saved)

        mean_random = sum(increases) / len(increases)
        ratio = adversarial / mean_random if abs(mean_random) > 1e-9 else float("nan")
        # In cả hai quy ước ở mọi dòng để không bao giờ phải đoán đơn vị về sau.
        del ratio
        print(f"  {rho:>7.3f}{eps_norm:>11.3f}{eps_norm:>15.3f}"
              f"{mean_random:>+20.6f}{adversarial:>+19.6f}")

    print("\n  Δloss càng lớn ở cùng rho -> cực tiểu càng NHỌN.")
    print("  So CodeBERT với CodeT5+ ở cùng rho. SAM chỉ đáng chạy nếu backbone")
    print("  đang hỏng (CodeT5+) nhọn hơn rõ rệt; nếu không thì §34 mới là chẩn đoán")
    print("  đúng và SAM không chạm tới nó.")


if __name__ == "__main__":
    main()
