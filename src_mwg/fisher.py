#!/usr/bin/env python3
"""Fisher cheo cua checkpoint Pha 1 — do "tham so nao Pha 1 that su hoc duoc".

Vi sao co file nay: RecAdam keo MOI tham so ve diem neo voi CUNG mot he so gamma.
Do duoc 06/09 tren t5p, gamma cang nho Delta cang cao (gamma 5 -> +0.0295 so voi AdamW
thuan; gamma 5000 -> -0.0086), tuc "neo deu tay" chu yeu la GANH NANG. Nhung neo yeu di
thi RecAdam tien ve AdamW va phuong phap mat loi.

Huong thu ba: giu tong luc keo, nhung PHAN BO lai theo do quan trong — keo manh dung
nhung tham so Pha 1 hoc duoc, tha long phan con lai. Do la penalty cua EWC (Kirkpatrick
et al., PNAS 2017) dat vao lich anneal cua RecAdam.

    L = lambda(t)*L_target + (1-lambda(t)) * (gamma/2) * sum_i  F_i * (theta_i - theta*_i)^2

F la Fisher CHEO uoc luong MOT LAN tren du lieu Pha 1 tai chinh checkpoint Pha 1.

Ba quyet dinh cai dat, deu la cho de sai:

1. **Fisher thuc nghiem hay Fisher that.** Dung `--fisher_mode true` (mac dinh) thi nhan
   duoc LAY MAU tu phan phoi du doan cua mo hinh — dung dinh nghia Fisher. `empirical`
   dung nhan that; re hon nhung chech, va do chech do khong dong deu giua cac lop.

2. **Chuan hoa ve trung binh 1.** Fisher tho trai hang chuc bac do lon nen `gamma` cu
   khong con y nghia gi. Chia cho trung binh co trong so (theo so phan tu) giu
   `sum(F_i)/N = 1`, nen **tong luc keo bang dung RecAdam thuong o cung gamma** —
   phep so vi the doi DUNG MOT BIEN: cach PHAN BO luc keo, khong phai do lon cua no.

3. **Kep tren, va no RANG BUOC VOI gamma.** Luc keo la  theta <- theta - c*(theta-theta*)
   voi  c = lr*(1-lambda)*gamma*F_i.  Day la lap diem co dinh: on dinh khi 0 < c < 2, va
   c gan 1 thi mot buoc keo gan het ve neo. Voi lr = 2e-5:

       gamma 5000 -> lr*gamma = 0.1   -> F_max phai < 5   (clip 10 cho c = 1.0, HONG)
       gamma  500 -> lr*gamma = 0.01  -> F_max < 50
       gamma   50 -> lr*gamma = 0.001 -> gan nhu tu do

   `--fisher_clip` mac dinh **5** vi the, va `recadam_fisher.check_pull_stability()` kiem
   lai truoc khi chay. Do duoc 06/09: lr=1e-2, gamma=500, F=10 cho c = 50 va trong so no
   ra 9.7e31 sau 25 buoc — phep kiem hai chieu bat duoc ngay, chay that thi khong.

Ghi ra `<checkpoint>.fisher.pt`: dict {ten_tham_so: tensor cung shape}, kem sieu du lieu.
Ghi nguyen tu (file tam roi os.replace) — cung ly do nhu runtime_env.py.

    python src/fisher.py --source_checkpoint model/n48/phase1/.../best.pt \\
        --data_path data/phase1_4cwe.jsonl --cwe_vocab fixed4 \\
        --model_name Salesforce/codet5p-220m-bimodal --pooling mean --aux_mode latent_bottleneck
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from logging_utils import get_logger

logger = get_logger()


def estimate_diagonal_fisher(model, dataloader, device, max_batches=None,
                             mode="true", n_samples=1):
    """Fisher cheo = E[ (d log p / d theta)^2 ].

    `mode="true"`: nhan lay mau tu p(y|x) cua chinh mo hinh — dung dinh nghia.
    `mode="empirical"`: dung nhan that trong du lieu.

    Chi lay gradient cua LOSS NHI PHAN (vul_head), khong lay cua head phu: Pha 2 chi
    dung vul_head, nen "quan trong" phai do theo dung task se duoc chuyen giao.
    """
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()
              if p.requires_grad}
    model.eval()          # tat dropout: Fisher phai do tai DIEM, khong phai trung binh nhieu
    seen = 0
    for index, batch in enumerate(dataloader):
        if max_batches is not None and index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask, return_cwe=False)
        logits = outputs["vul_logits"]
        if mode == "empirical":
            targets = [batch["labels"].to(device)]
        else:
            probs = torch.softmax(logits.detach(), dim=-1)
            targets = [torch.multinomial(probs, 1).squeeze(-1) for _ in range(n_samples)]
        for target in targets:
            model.zero_grad(set_to_none=True)
            # `sum` chu khong phai `mean`: Fisher la trung binh TREN MAU, nen chia cho
            # tong so mau o cuoi. Dung `mean` roi lai chia nua la chia hai lan.
            F.cross_entropy(logits, target, reduction="sum").backward(retain_graph=True)
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
        seen += input_ids.size(0) * len(targets)
    model.zero_grad(set_to_none=True)
    if seen:
        for name in fisher:
            fisher[name] /= seen
    return fisher, seen


def normalize_fisher(fisher, clip=5.0):
    """Chuan hoa ve trung binh 1 (theo phan tu), kep tren, roi chuan hoa LAI.

    Giu `sum(F_i)/N = 1` nen tong luc keo bang RecAdam thuong o cung gamma; chi khac
    cach phan bo. Neu khong lam vay thi doi tu RecAdam sang RecAdam-Fisher se doi HAI
    bien (phan bo VA do lon), va khong doc duoc bien nao gay ra thay doi.
    """
    total = sum(float(v.sum()) for v in fisher.values())
    count = sum(v.numel() for v in fisher.values())
    mean = total / count if count else 0.0
    stats = {"mean_raw": mean, "count": count}
    if mean <= 0:
        logger.warning("Fisher toan 0 — tra ve toan 1 (tuong duong RecAdam thuong)")
        return {k: torch.ones_like(v) for k, v in fisher.items()}, stats
    out = {k: v / mean for k, v in fisher.items()}
    stats["clipped"] = 0
    if clip and clip > 0:
        stats["clipped"] = int(sum(int((v > clip).sum()) for v in out.values()))
        out = {k: v.clamp(max=clip) for k, v in out.items()}
        mean2 = sum(float(v.sum()) for v in out.values()) / count
        if mean2 > 0:
            out = {k: v / mean2 for k, v in out.items()}
    flat = torch.cat([v.flatten() for v in out.values()])
    stats.update({
        "mean_after": float(flat.mean()),
        "median": float(flat.median()),
        "p99": float(torch.quantile(flat[torch.randperm(flat.numel())[:1000000]], 0.99)),
        "frac_below_0p01": float((flat < 0.01).float().mean()),
    })
    return out, stats


def assess_fisher(stats):
    """Fisher co dung duoc khong, hay da suy bien? Kiem TRUOC khi ton GPU.

    Rui ro lon nhat cua huong nay (L2-SP §5.2, ICML 2018; va arXiv:2603.18596): tai mot
    checkpoint DA HOI TU, so hang (p_k - y_k) tien ve 0 nen Fisher TIEU BIEN. Hau qua:
      * Pha 1 TOT  -> F ~ 0 cho gan het mang -> gamma*F suy bien thanh AdamW, va ta se
        doc nham ket qua "khong khac gi" thanh "Fisher khong giup", trong khi that ra
        Fisher khong duoc ap dung.
      * Pha 1 SAP  -> F lon -> qua cung.
    Ca hai deu phai bao TRUOC, khong de chay xong 30 o roi moi doan.

    Tra ve (dung_duoc, danh_sach_canh_bao).
    """
    warn = []
    below = stats.get("frac_below_0p01")
    median = stats.get("median")
    p99 = stats.get("p99")
    if below is not None and below > 0.90:
        warn.append(f"SUY BIEN: {100*below:.1f}% tham so co F < 0.01 — gamma*F ~ 0 cho gan "
                    f"het mang, tuong duong AdamW. Ket qua se KHONG doc duoc nhu 'Fisher khong giup'.")
    elif below is not None and below > 0.75:
        warn.append(f"canh bao: {100*below:.1f}% tham so co F < 0.01 — luc keo don vao thieu so tham so")
    if median is not None and median < 1e-3:
        warn.append(f"canh bao: median F = {median:.2e} — phan bo lech rat manh ve 0")
    if p99 is not None and median and p99 / max(median, 1e-12) > 1e4:
        warn.append(f"canh bao: p99/median = {p99/median:.1e} — duoi tren rat dai, kep chat hon")
    return (not any(w.startswith("SUY BIEN") for w in warn)), warn


def sidecar_path(checkpoint_path):
    return Path(str(checkpoint_path) + ".fisher.pt")


def main():
    parser = argparse.ArgumentParser(description="Fisher cheo cua checkpoint Pha 1")
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--data_path", required=True, help="JSONL cua Pha 1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--aux_mode", default="latent_bottleneck")
    parser.add_argument("--cwe_vocab", choices=("fixed4", "source", "precomputed"), default="fixed4")
    parser.add_argument("--num_latent", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=64,
                        help="so batch dung de uoc luong; 64x16 = 1024 mau la du on dinh")
    parser.add_argument("--fisher_mode", choices=("true", "empirical"), default="true")
    parser.add_argument("--fisher_clip", type=float, default=5.0,
                        help="kep F sau chuan hoa; PHAI di cung gamma, xem docstring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--truncation_strategy", default="head_middle_tail")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="tinh lai du da co sidecar")
    args = parser.parse_args()

    out_path = sidecar_path(args.source_checkpoint)
    if out_path.is_file() and not args.force:
        logger.info("Da co %s — bo qua (dung --force de tinh lai)", out_path)
        return

    from transformers import AutoTokenizer
    from train_transfer import (adopt_checkpoint_shape, apply_cwe_vocab, build_cwe_vocab,
                                build_dataloader, load_checkpoint, load_jsonl, make_model,
                                set_seed)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_jsonl(args.data_path, trust_precomputed=(args.cwe_vocab == "precomputed"))
    if args.cwe_vocab == "source":
        apply_cwe_vocab(records, build_cwe_vocab(records))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.num_cwes = adopt_checkpoint_shape(args.source_checkpoint, device)
    model = make_model(args.model_name, device, args)
    checkpoint = load_checkpoint(args.source_checkpoint, model, device)
    logger.info("Checkpoint Pha 1: epoch %s | val %.6f",
                checkpoint.get("best_epoch"), float(checkpoint.get("best_val_macro_f1") or 0))

    loader = build_dataloader(records, tokenizer, args.max_length, args.batch_size,
                              False, args.seed, args.num_workers, args.truncation_strategy)
    started = time.perf_counter()
    fisher, seen = estimate_diagonal_fisher(model, loader, device,
                                            max_batches=args.max_batches, mode=args.fisher_mode)
    fisher, stats = normalize_fisher(fisher, clip=args.fisher_clip)
    elapsed = time.perf_counter() - started
    logger.info("Fisher xong | %d mau | %.1fs | trung binh sau chuan hoa %.4f | "
                "median %.4f | p99 %.2f | %.1f%% duoi 0.01 | kep %d phan tu",
                seen, elapsed, stats["mean_after"], stats["median"], stats["p99"],
                100 * stats["frac_below_0p01"], stats.get("clipped", 0))

    payload = {"fisher": {k: v.cpu() for k, v in fisher.items()},
               "stats": stats, "samples": seen, "seconds": round(elapsed, 2),
               "fisher_mode": args.fisher_mode, "fisher_clip": args.fisher_clip,
               "source_checkpoint": str(args.source_checkpoint),
               "data_path": args.data_path,
               "source_best_val_macro_f1": float(checkpoint.get("best_val_macro_f1") or 0)}
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    usable, warnings = assess_fisher(stats)
    for w in warnings:
        logger.warning("Fisher | %s", w)
    if not usable:
        logger.warning("Fisher | => KHONG NEN chay khoi nay truoc khi hieu vi sao F suy bien. "
                       "Thu --fisher_mode empirical, hoac lay Fisher o mot checkpoint SOM hon "
                       "trong Pha 1 (luc chua hoi tu han).")
    logger.info("Ghi %s (%.1f MB)", out_path, out_path.stat().st_size / 1048576)
    print(json.dumps({k: v for k, v in stats.items() if isinstance(v, (int, float))}, indent=2))


if __name__ == "__main__":
    main()
