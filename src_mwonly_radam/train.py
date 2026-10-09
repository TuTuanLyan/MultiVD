#!/usr/bin/env python3
"""Huấn luyện / kiểm tra MWG trên một fold.

    python train.py --phase train --run_name <tên> --fold <k> --data_root <thư mục chứa fold<k>/> --model_name <codebert> ...
    python train.py --phase test  ...cùng cờ...  --result_path results/<tên>/.../fold<k>.json

Chuyển giao hai pha: pha 1 huấn luyện trên pool nguồn (1 lần, có --pair_loss), pha 2 khởi tạo từ checkpoint
pha 1 (--init all --init_ckpt), chống quên bằng --recadam 1 và tìm cực tiểu phẳng bằng --sam_rho (SAM / ASAM).
Target-only: pha 2 không --init, AdamW trần (không --recadam, --sam_rho 0).
Cờ, mặc định, checkpoint và file kết quả giữ đúng như `train_mwg.py` cũ (archive/code_snapshot/src).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from mwg.data import GraphWindowDataset, PairBatchSampler
from mwg.metrics import classification_metrics, find_best_threshold
from mwg.model import ARCHITECTURE, MWGraphModel, build_backbone, pair_margin_loss
from mwg.optim import RecAdam, SAMStep
from mwg.utils import configure_logging, get_logger, limit_records, load_jsonl, set_seed, train_loss_stalled

logger = get_logger()
KEYS = ("input_ids", "attention_mask", "window_mask", "window_pos", "tok_line", "graphs", "line_mask", "line_ids", "pair")
GRAPH_PREFIX = ("graph.", "word_att.")
INIT_PARTS = {"encoder": ("backbone.",), "encoder_agg": ("backbone.", "agg.", "win_pos_emb.", "rel_pos."),
              "graph": GRAPH_PREFIX, "graph_agg": GRAPH_PREFIX + ("agg.", "win_pos_emb.", "rel_pos."),
              "all": ("backbone.", "agg.", "win_pos_emb.", "rel_pos.") + GRAPH_PREFIX}
SELECT_METRICS = ("roc_auc", "macro_f1", "pr_auc", "train_loss")
STUCK_EXIT = 3   # mã thoát khi --stuck_epoch phát hiện kẹt (scripts/run_p1.sh bắt mã này để chạy lại với seed khác)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("train", "test"), required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--data_root", required=True, help="thư mục chứa fold<k>/{train,val,test}.jsonl")
    p.add_argument("--target_lang", default=None, help="nếu đặt: mọi bản ghi phải có lang này")
    p.add_argument("--model_name", required=True)
    p.add_argument("--seed", type=int, default=36)
    # 04/10 (khối MW-only n=5): tái lập từng bit + TF32
    p.add_argument("--deterministic", type=int, default=1,
                   help="1 = thuật toán tất định (use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG, PYTHONHASHSEED)")
    p.add_argument("--tf32", type=int, default=0, help="1 = bật TF32 ở matmul và cuDNN")
    # multi-window
    p.add_argument("--window", type=int, default=510)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--max_windows", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--agg", choices=("transformer", "mean"), default="mean")
    p.add_argument("--agg_layers", type=int, default=2)
    p.add_argument("--pooling", default="cls")
    # đồ thị dòng
    p.add_argument("--graph", choices=("typed", "babel"), default="typed", help="typed: 4 quan hệ; babel: 2 quan hệ gốc")
    p.add_argument("--max_lines", type=int, default=150, help="số dòng tối đa của đồ thị (BABEL max_sentnum)")
    p.add_argument("--graph_layers", type=int, default=2)
    p.add_argument("--graph_lstm", type=int, default=1)
    p.add_argument("--fusion", choices=("cat", "sum", "graph", "mw"), default="cat", help="mw = chỉ nhánh multi-window")
    p.add_argument("--norm_text", type=int, default=0, help="1 = đưa văn bản đã chuẩn hoá VARk/FUNk vào CodeBERT")
    p.add_argument("--line_enc", choices=("codebert", "babel"), default="codebert",
                   help="vector dòng: codebert = gộp trạng thái token theo dòng; babel = WordAttNet gốc")
    p.add_argument("--drop_bracket", type=int, default=0, help="1 = bỏ dòng chỉ có ngoặc khỏi tập đỉnh")
    p.add_argument("--co_mode", choices=("all", "chain"), default="all", help="chain = co_use chỉ nối hai lần xuất hiện liên tiếp")
    p.add_argument("--max_words", type=int, default=60, help="số token tối đa mỗi dòng (chỉ dùng khi --line_enc babel)")
    p.add_argument("--graph_lr", type=float, default=1e-4, help="lr riêng của nhánh đồ thị")
    # chuyển giao
    p.add_argument("--init", choices=tuple(["none"] + list(INIT_PARTS)), default="none",
                   help="nạp phần nào từ --init_ckpt (head luôn mới)")
    p.add_argument("--init_ckpt", default=None)
    p.add_argument("--freeze_backbone", type=int, default=0, help="1 = đóng băng CodeBERT")
    p.add_argument("--pair_loss", type=float, default=0.0, help=">0: batch xếp theo cặp + lambda * softplus(margin - (s_lỗi - s_vá))")
    p.add_argument("--pair_margin", type=float, default=1.0)
    p.add_argument("--recadam", type=int, default=0)
    # 07/10 người dùng: thử RAdam (Liu et al. 2019, arXiv:1908.03265) cho ô Pha 1 kẹt ở ln2; mặc định adamw = hành vi cũ
    p.add_argument("--p1_optimizer", choices=("adamw", "radam"), default="adamw",
                   help="optimizer khi --recadam 0: adamw (cũ) hoặc radam (torch.optim.RAdam, weight decay tách rời như AdamW)")
    # 08/10 người dùng: thử RAdam beta2 = 0,99 (warmup ngầm khoảng 2/(1-beta2) = 200 bước thay vì 2 000); mặc định 0,999 = hành vi cũ
    p.add_argument("--p1_beta2", type=float, default=0.999,
                   help="beta2 của RAdam khi --p1_optimizer radam (mặc định 0.999 như torch)")
    p.add_argument("--pretrain_cof", type=float, default=5000.0)
    p.add_argument("--anneal_t0_ratio", type=float, default=0.05)
    p.add_argument("--anneal_k", type=float, default=0.05)
    p.add_argument("--anneal_fun", default="sigmoid")
    p.add_argument("--anneal_t0_steps", type=int, default=0,
                   help=">0: t0 tính bằng BƯỚC tuyệt đối như bài gốc (Chen et al. 2020), bỏ qua --anneal_t0_ratio")
    p.add_argument("--recadam_free_head", type=int, default=0,
                   help="1 = vul_head chạy Adam thường (anneal_w=0, không neo) như mã gốc RecAdam; 0 = neo cả head (hành vi cũ)")
    p.add_argument("--sam_rho", type=float, default=0.0)
    p.add_argument("--sam_variant", choices=("sam", "asam"), default="sam")
    p.add_argument("--sam_eta", type=float, default=0.01)
    p.add_argument("--sam_bias_ln", choices=("full", "exclude", "shrink", "mask"), default="full",
                   help="ASAM: nhiễu trên bias + LayerNorm - full (cũ), exclude (không nhiễu, ρ dồn cho weight), shrink (T nhân --sam_bias_ln_scale), "
                        "mask (chuẩn như full, ε = 0 trên bias/LN ⇒ ε weight trùng full)")
    p.add_argument("--sam_log_share", type=int, default=0, help="1 = ghi mỗi epoch tỉ lệ ‖g·T‖² theo nhóm weight / bias / LN (T của ASAM gốc)")
    p.add_argument("--sam_bias_ln_scale", type=float, default=0.1)
    # tối ưu
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--min_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--micro", type=int, default=16, help="số cửa sổ mỗi lượt qua CodeBERT")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--selection_metric", choices=SELECT_METRICS, default="roc_auc",
                   help="chọn checkpoint theo metric val (lớn là tốt) hoặc train_loss (nhỏ là tốt)")
    p.add_argument("--also_select", choices=SELECT_METRICS, default=None,
                   help="giữ thêm checkpoint tốt nhất theo metric này ở <thư mục checkpoint>/best_<metric>.pt")
    # 04/10 (khối mwonly5, người dùng): "trong khi pha 1 chưa giảm train loss > 2% thì lấy checkpoint theo train loss, sau đó mới lấy theo val ROC"
    p.add_argument("--select_after_drop", type=float, default=0.0,
                   help=">0: chọn checkpoint theo train loss nhỏ nhất cho tới epoch đầu tiên có train loss < (1 - giá trị) × train loss epoch 1, "
                        "từ đó theo --selection_metric (điểm cũ bỏ đi vì hai thang không so được); 0 = tắt (như cũ)")
    p.add_argument("--patience_after_loss", type=float, default=0.0,
                   help=">0: chỉ đếm patience (early stopping) từ epoch đầu tiên có train loss < giá trị này; "
                        "tránh cắt lượt còn đứng ở loss ~ln 2 (0 = tắt, như cũ)")
    p.add_argument("--stuck_epoch", type=int, default=0,
                   help=">=2: cuối epoch này, nếu train loss giảm chưa tới --stuck_min_drop so với epoch trước thì dừng, "
                        "thoát mã 3 (0 = tắt)")
    p.add_argument("--stuck_min_drop", type=float, default=0.02, help="độ giảm tương đối tối thiểu của train loss giữa hai epoch")
    p.add_argument("--grad_checkpoint", type=int, default=1)
    # vào / ra
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--result_path", default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.stuck_epoch == 1 or args.stuck_epoch < 0:
        p.error("--stuck_epoch phải là 0 (tắt) hoặc >= 2")
    return args


def to_dev(b, device):
    return {k: b[k].to(device) for k in KEYS}


def run_eval(model, loader, device, micro):
    model.eval()
    labels, probs, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for b in loader:
            out = model(to_dev(b, device), micro)
            y = b["labels"].to(device)
            loss_sum += F.cross_entropy(out["vul_logits"], y).item() * y.size(0)
            n += y.size(0)
            labels.extend(y.cpu().tolist())
            probs.extend(torch.softmax(out["vul_logits"], -1)[:, 1].cpu().tolist())
    res = classification_metrics(labels, probs, 0.5)
    res.update(loss=loss_sum / max(1, n), labels=labels, probabilities=probs)
    return res


def save_ckpt(path, model, epoch, score, args):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "best_epoch": epoch, "best_val_score": score,
                "architecture": ARCHITECTURE, "model_name": args.model_name, "window": args.window, "stride": args.stride,
                "max_windows": args.max_windows, "agg": args.agg, "seed": args.seed, "fold": args.fold,
                "training_args": {k: v for k, v in vars(args).items() if not k.startswith("_")}}, path)


def init_from(model, path, parts, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    assert ck.get("architecture") in (ARCHITECTURE, "codebert_multiwindow"), f"{path}: không phải checkpoint MWG"
    keep = {k: v for k, v in ck["model_state_dict"].items() if k.startswith(INIT_PARTS[parts])}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    logger.info("Khởi tạo từ %s | parts=%s | nạp %d tensor | head mới | thiếu (ngoài backbone): %s", path, parts, len(keep),
                [m for m in missing if not m.startswith("backbone.")][:8])
    assert not unexpected, unexpected


def build_optimizer(model, args, total_steps):
    """Hai nhóm tham số: nhánh đồ thị (lr riêng) và phần còn lại. Khi đóng băng CodeBERT, mọi phần mới đều theo lr đồ thị."""
    prefix = GRAPH_PREFIX + (("vul_head.", "win_pos_emb.", "rel_pos.") if args.freeze_backbone else ())
    g_params = [q for n, q in model.named_parameters() if n.startswith(prefix) and q.requires_grad]
    o_params = [q for n, q in model.named_parameters() if not n.startswith(prefix) and q.requires_grad]
    if not args.recadam:
        if args.p1_optimizer == "radam":
            logger.info("RAdam | lr=%g graph_lr=%g | betas=(0.9, %g) | weight_decay=%g tách rời (decoupled)",
                        args.learning_rate, args.graph_lr, args.p1_beta2, args.weight_decay)
            return torch.optim.RAdam([{"params": o_params, "lr": args.learning_rate}, {"params": g_params, "lr": args.graph_lr}],
                                     lr=args.learning_rate, betas=(0.9, args.p1_beta2), weight_decay=args.weight_decay,
                                     decoupled_weight_decay=True)
        return torch.optim.AdamW([{"params": o_params, "lr": args.learning_rate}, {"params": g_params, "lr": args.graph_lr}],
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    t0 = args.anneal_t0_steps if args.anneal_t0_steps > 0 else max(1, int(args.anneal_t0_ratio * total_steps))
    # mã gốc RecAdam đặt anneal_w=0 cho tham số không có trong mô hình tiền huấn luyện: head mới học Adam thường, không neo
    h_params = [q for n, q in model.named_parameters() if args.recadam_free_head and n.startswith("vul_head.") and q.requires_grad]
    o_params = [q for q in o_params if all(q is not h for h in h_params)]
    g_params = [q for q in g_params if all(q is not h for h in h_params)]
    anchor = [q.detach().clone() for q in o_params + g_params + h_params]  # điểm neo = trọng số ngay sau khởi tạo
    logger.info("RecAdam | cof=%.0f | k=%g | t0=%d/%d | neo %d tensor | head tự do %d tensor", args.pretrain_cof, args.anneal_k,
                t0, total_steps, len(o_params) + len(g_params), len(h_params))
    groups = [{"params": o_params, "lr": args.learning_rate, "pretrain_params": anchor[: len(o_params)]},
              {"params": g_params, "lr": args.graph_lr, "pretrain_params": anchor[len(o_params): len(o_params) + len(g_params)]}]
    if h_params:
        groups.append({"params": h_params, "lr": args.learning_rate, "anneal_w": 0.0,
                       "pretrain_params": anchor[len(o_params) + len(g_params):]})
    return RecAdam(groups, lr=args.learning_rate, weight_decay=args.weight_decay, anneal_fun=args.anneal_fun, anneal_k=args.anneal_k,
                   anneal_t0=t0, anneal_w=1.0, pretrain_cof=args.pretrain_cof, pretrain_params=anchor)


def selection_score(metric, v, train_loss):
    """Điểm chọn checkpoint, càng lớn càng tốt: train_loss lấy âm; metric val thiếu hoặc NaN thì dùng macro_f1."""
    if metric == "train_loss":
        return -train_loss
    score = v.get(metric)
    if score is None or score != score:
        score = v["macro_f1"]
    return score


def train(model, args, device, make_loader, load_split):
    t_prep = time.perf_counter()
    tr = make_loader(load_split("train", args.max_train_samples), args.batch_size, True)
    va = make_loader(load_split("val", args.max_eval_samples), args.eval_batch_size, False)
    if args.pair_loss > 0:
        sampler = PairBatchSampler(tr.dataset, args.batch_size, args.seed)
        tr = DataLoader(tr.dataset, batch_sampler=sampler, num_workers=args.num_workers)
        logger.info("Pair loss lambda=%.2f margin=%.2f | %d cặp đủ hai nhãn, %d mẫu lẻ", args.pair_loss, args.pair_margin,
                    len(sampler.pairs), len(sampler.singles))
    cov = {"train": tr.dataset.coverage(), "val": va.dataset.coverage()}
    logger.info("Độ phủ W=%d S=%d K=%d Lmax=%d | tiền xử lý %.0fs | %s", args.window, args.stride, args.max_windows, args.max_lines,
                time.perf_counter() - t_prep, json.dumps(cov))
    if args.init != "none":
        init_from(model, args.init_ckpt, args.init, device)
    total = len(tr) * args.epochs
    opt = build_optimizer(model, args, total)
    sam = None
    if args.sam_rho > 0:
        sam = SAMStep(args.sam_rho, variant=args.sam_variant, eta=args.sam_eta, bias_ln=args.sam_bias_ln,
                      bias_ln_scale=args.sam_bias_ln_scale, log_share=bool(args.sam_log_share))
        logger.info("SAM | variant=%s rho=%.4f eta=%.4f | bias_ln=%s scale=%g", args.sam_variant, args.sam_rho, args.sam_eta,
                    args.sam_bias_ln, args.sam_bias_ln_scale)
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)

    def loss_of(logits, y, bd):
        loss = F.cross_entropy(logits, y)
        if args.pair_loss > 0:
            loss = loss + args.pair_loss * pair_margin_loss(logits, y, bd["pair"], args.pair_margin)
        return loss

    best, best_ep, wait, t0, losses = -math.inf, 0, 0, time.perf_counter(), []
    best2, also_path = -math.inf, str(Path(args.checkpoint_path).with_name(f"best_{args.also_select}.pt"))
    counting = args.patience_after_loss <= 0
    gate_open = args.select_after_drop <= 0          # đã qua mốc giảm train loss chưa (tắt cờ = mở sẵn, hành vi cũ)
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        tl, tn, te = 0.0, 0, time.perf_counter()
        for b in tr:
            bd, y = to_dev(b, device), b["labels"].to(device)
            loss = loss_of(model(bd, args.micro)["vul_logits"], y, bd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if sam is not None:
                named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
                params = [q for _, q in named]
                if sam.ascend(params, names=[n for n, _ in named]):
                    opt.zero_grad(set_to_none=True)
                    loss_of(model(bd, args.micro)["vul_logits"], y, bd).backward()
                    sam.restore(params)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            tl += loss.item() * y.size(0)
            tn += y.size(0)
        losses.append(tl / max(1, tn))
        if not counting and losses[-1] < args.patience_after_loss:
            counting = True
            logger.info("Bắt đầu đếm patience từ epoch %d (train loss %.4f < %.2f)", ep, losses[-1], args.patience_after_loss)
        v = run_eval(model, va, device, args.micro)
        for k in ("roc_auc", "macro_f1", "pr_auc"):
            if v.get(k) is None:
                v[k] = float("nan")
        if not gate_open and losses[-1] < (1 - args.select_after_drop) * losses[0]:
            gate_open = True
            best, wait = -math.inf, 0                    # đổi tiêu chí: điểm train loss cũ không so được với val ROC
            logger.info("Chọn checkpoint theo %s từ epoch %d (train loss %.4f < %.0f%% train loss epoch 1 = %.4f)", args.selection_metric, ep,
                        losses[-1], 100 * (1 - args.select_after_drop), losses[0])
        metric = args.selection_metric if gate_open else "train_loss"
        score = selection_score(metric, v, losses[-1])
        if args.also_select:
            score2 = selection_score(args.also_select, v, losses[-1])
            if score2 > best2:
                best2 = score2
                save_ckpt(also_path, model, ep, score2, args)
                logger.info("Checkpoint %s saved | Epoch: %d | %s", args.also_select, ep, also_path)
        if score > best:
            best, best_ep, wait = score, ep, 0
            save_ckpt(args.checkpoint_path, model, ep, score, args)
            logger.info("Best checkpoint saved | Epoch: %d | Val %s: %.6f", ep, metric, score)
        elif ep >= args.min_epochs and counting:
            wait += 1
        logger.info("Epoch %d/%d | train loss %.4f | val loss %.4f | val roc_auc %.4f | val macro_f1 %.4f | best ep %d | patience %d/%d | %.0fs",
                    ep, args.epochs, losses[-1], v["loss"], v["roc_auc"], v["macro_f1"], best_ep, wait, args.patience,
                    time.perf_counter() - te)
        if sam is not None and args.sam_log_share:
            sh = sam.group_shares()
            if sh:
                logger.info("SAM share ep %d | weight %.1f%% | bias %.1f%% | LN %.1f%% | %d bước (‖g·T‖², T của ASAM gốc)",
                            ep, 100 * sh[0]["weight"], 100 * sh[0]["bias"], 100 * sh[0]["LN"], sh[1])
        if ep == args.stuck_epoch:
            stalled, drop = train_loss_stalled(losses, args.stuck_min_drop)
            logger.info("Kiểm kẹt | epoch %d | train loss %.4f -> %.4f | giảm %.2f%% (ngưỡng %.2f%%) | %s", ep, losses[-2],
                        losses[-1], 100 * drop, 100 * args.stuck_min_drop, "KẸT" if stalled else "đang học")
            if stalled:
                sys.exit(STUCK_EXIT)
        if ep >= args.min_epochs and wait >= args.patience:
            logger.info("Early stopping | Epoch: %d | Best epoch: %d", ep, best_ep)
            break
    logger.info("Train xong | best epoch %d | %.0fs", best_ep, time.perf_counter() - t0)


def test(model, args, device, make_loader, load_split):
    ck = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    va = make_loader(load_split("val", args.max_eval_samples), args.eval_batch_size, False)
    te = make_loader(load_split("test", args.max_eval_samples), args.eval_batch_size, False)
    t0 = time.perf_counter()
    v = run_eval(model, va, device, args.micro)
    thr, vf1 = find_best_threshold(v["labels"], v["probabilities"])
    t1 = time.perf_counter()
    t = run_eval(model, te, device, args.micro)
    t2 = time.perf_counter()
    m05 = classification_metrics(t["labels"], t["probabilities"], 0.5)
    mvc = classification_metrics(t["labels"], t["probabilities"], thr)
    ds = te.dataset
    nt, lab, pr = np.asarray(ds.n_tokens), np.asarray(t["labels"]), np.asarray(t["probabilities"])

    def sub(mask):
        ok = mask.sum() > 1 and len(set(lab[mask])) == 2
        return (float(roc_auc_score(lab[mask], pr[mask])) if ok else None, int(mask.sum()))

    res = {"experiment_name": f"{args.run_name}/multiwindow", "architecture": ARCHITECTURE, "phase": "test", "fold": args.fold,
           "seed": args.seed, "best_epoch": ck["best_epoch"], "best_val_score": ck["best_val_score"],
           "val_calibrated_threshold": thr, "val_macro_f1_at_valcal": vf1,
           "test_roc_auc": m05["roc_auc"], "test_pr_auc": m05["pr_auc"], "test_macro_f1_at_0.5": m05["macro_f1"],
           "test_macro_f1_at_valcal": mvc["macro_f1"], "test_inference_seconds": t2 - t1, "validation_and_threshold_seconds": t1 - t0,
           "subgroup_roc": {"le510": sub(nt <= 510), "gt510": sub(nt > 510), "one_window": sub(np.asarray(ds.n_windows_total) == 1),
                            "truncated_at_K": sub(np.asarray(ds.n_windows_total) > args.max_windows),
                            "lines_gt_Lmax": sub(np.asarray(ds.n_lines) > args.max_lines)},
           "coverage": {"val": va.dataset.coverage(), "test": ds.coverage()}, "hyperparameters": ck["training_args"]}
    Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("test_roc_auc", "test_macro_f1_at_valcal", "best_epoch", "subgroup_roc")}))
    if os.environ.get("DUMP_PROBS", "1") == "1":
        np.savez_compressed(Path(args.result_path).with_suffix(".probs.npz"), probabilities=pr.astype(np.float32),
                            labels=lab.astype(np.int8), n_tokens=nt, val_probabilities=np.asarray(v["probabilities"], np.float32),
                            val_labels=np.asarray(v["labels"], np.int8), threshold=np.float32(thr))
    logger.info("Kết quả: %s", args.result_path)


def main():
    args = parse_args()
    if args.deterministic and os.environ.get("PYTHONHASHSEED") != "42":
        # thứ tự duyệt set/dict theo hash của chuỗi phải cố định TRƯỚC khi trình thông dịch khởi động ⇒ tự chạy lại chính nó
        os.environ["PYTHONHASHSEED"] = "42"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    root = Path("model") / args.run_name / "multiwindow" / f"seed_{args.seed}"
    args.checkpoint_path = args.checkpoint_path or str(root / f"fold{args.fold}" / "best.pt")
    args.result_path = args.result_path or str(Path("results") / args.run_name / "multiwindow" / f"seed_{args.seed}" / f"fold{args.fold}.json")
    configure_logging()
    set_seed(args.seed, bool(args.deterministic), bool(args.tf32))
    logger.info("Tái lập | deterministic=%s CUBLAS=%s PYTHONHASHSEED=%s cudnn.det=%s benchmark=%s tf32(matmul/cudnn)=%s/%s",
                torch.are_deterministic_algorithms_enabled(), os.environ.get("CUBLAS_WORKSPACE_CONFIG"), os.environ.get("PYTHONHASHSEED"),
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    fold_dir = Path(args.data_root) / f"fold{args.fold}"
    typed = args.graph == "typed"

    def load_split(split, cap):
        return limit_records(load_jsonl(fold_dir / f"{split}.jsonl", args.target_lang), cap, args.seed)

    def make_loader(records, batch_size, shuffle):
        ds = GraphWindowDataset(records, tok, args.window, args.stride, args.max_windows, args.max_lines, typed, args.max_length,
                                bool(args.norm_text), args.max_words, bool(args.drop_bracket), args.co_mode)
        # thứ tự trộn lấy từ bộ sinh RIÊNG theo seed: không phụ thuộc lượng số ngẫu nhiên các phần khác đã rút
        gen = torch.Generator().manual_seed(args.seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers, generator=gen)

    backbone = build_backbone(args.model_name)
    if args.freeze_backbone:
        for q in backbone.parameters():
            q.requires_grad = False
        backbone.eval()
        logger.info("CodeBERT đóng băng: chỉ học nhánh đồ thị + vị trí cửa sổ + head")
    elif args.grad_checkpoint and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    model = MWGraphModel(backbone, args.max_windows, args.agg, args.agg_layers, pooling=args.pooling, num_rel=4 if typed else 2,
                         graph_layers=args.graph_layers, fusion=args.fusion, graph_lstm=bool(args.graph_lstm),
                         max_lines=args.max_lines, line_enc=args.line_enc).to(device)
    n_graph = sum(q.numel() for n, q in model.named_parameters() if n.startswith(GRAPH_PREFIX) and q.requires_grad)
    logger.info("MWGraph | graph=%s R=%d Lmax=%d layers=%d lstm=%d fusion=%s norm_text=%d line_enc=%s drop_bracket=%d co_mode=%s | "
                "tham số nhánh đồ thị %.2fM", args.graph, 4 if typed else 2, args.max_lines, args.graph_layers, args.graph_lstm,
                args.fusion, args.norm_text, args.line_enc, args.drop_bracket, args.co_mode, n_graph / 1e6)
    (train if args.phase == "train" else test)(model, args, device, make_loader, load_split)


if __name__ == "__main__":
    main()
