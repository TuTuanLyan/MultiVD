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
from transformers import AutoTokenizer, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from multiwindow.data import WindowDataset
from multiwindow.metrics import classification_metrics, find_best_threshold
from multiwindow.model import ARCHITECTURE, MultiWindowModel, build_backbone
from multiwindow.optim import RecAdam, SAMStep
from multiwindow.utils import configure_logging, get_logger, limit_records, load_jsonl, set_seed, train_loss_stalled

logger = get_logger()
KEYS = ("input_ids", "attention_mask", "window_mask", "window_pos")
INIT_PARTS = {"encoder": ("backbone.",), "encoder_agg": ("backbone.", "agg.", "win_pos_emb.", "rel_pos."),
              "all": ("backbone.", "agg.", "win_pos_emb.", "rel_pos."),
              "all_head": ("backbone.", "agg.", "win_pos_emb.", "rel_pos.", "vul_head.")}
SELECT_METRICS = ("roc_auc", "macro_f1", "pr_auc", "train_loss")
STUCK_EXIT = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("train", "test"), required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--data_root", required=True, help="directory containing fold<k>/{train,val,test}.jsonl")
    p.add_argument("--target_lang", default=None, help="if set, every record must have this lang")
    p.add_argument("--model_name", required=True)
    p.add_argument("--seed", type=int, default=36)
    p.add_argument("--deterministic", type=int, default=1,
                   help="1 = deterministic algorithms (use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG, PYTHONHASHSEED)")
    p.add_argument("--tf32", type=int, default=0, help="1 = enable TF32 in matmul and cuDNN")
    p.add_argument("--window", type=int, default=510)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--max_windows", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--agg", choices=("transformer", "mean"), default="mean")
    p.add_argument("--agg_layers", type=int, default=2)
    p.add_argument("--pooling", default="cls")
    p.add_argument("--init", choices=tuple(["none"] + list(INIT_PARTS)), default="none",
                   help="which parts to load from --init_ckpt (new classifier head, except all_head, which also loads vul_head)")
    p.add_argument("--init_ckpt", default=None)
    p.add_argument("--recadam", type=int, default=0)
    p.add_argument("--pretrain_cof", type=float, default=5000.0)
    p.add_argument("--anneal_t0_ratio", type=float, default=0.05)
    p.add_argument("--anneal_k", type=float, default=0.05)
    p.add_argument("--anneal_fun", default="sigmoid")
    p.add_argument("--anneal_t0_steps", type=int, default=0,
                   help=">0: t0 in absolute steps as in the original paper (Chen et al. 2020), ignores --anneal_t0_ratio")
    p.add_argument("--recadam_free_head", type=int, default=0,
                   help="1 = vul_head uses plain Adam (anneal_w=0, no anchoring) as in the original RecAdam code; 0 = anchor the head too")
    p.add_argument("--sam_rho", type=float, default=0.0)
    p.add_argument("--sam_variant", choices=("sam", "asam"), default="sam")
    p.add_argument("--sam_eta", type=float, default=0.01)
    p.add_argument("--sam_bias_ln", choices=("full", "exclude", "shrink", "mask"), default="full",
                   help="ASAM perturbation on bias + LayerNorm: full, exclude (no perturbation, rho goes to weights), "
                        "shrink (T scaled by --sam_bias_ln_scale), mask (normalized as full, eps = 0 on bias/LN)")
    p.add_argument("--sam_log_share", type=int, default=0, help="1 = log per epoch the share of ||g*T||^2 for weight / bias / LN")
    p.add_argument("--sam_bias_ln_scale", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--min_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--micro", type=int, default=16, help="number of windows per encoder pass")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--lr_schedule", choices=("linear", "constant"), default="linear",
                   help="linear: warm-up then linear decay to 0; constant: warm-up then constant")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--selection_metric", choices=SELECT_METRICS, default="roc_auc",
                   help="checkpoint selection by a validation metric (higher is better) or train_loss (lower is better)")
    p.add_argument("--also_select", choices=SELECT_METRICS, default=None,
                   help="also keep the best checkpoint by this metric at <checkpoint dir>/best_<metric>.pt")
    p.add_argument("--select_after_drop", type=float, default=0.0,
                   help=">0: select by the lowest train loss until the first epoch whose train loss < (1 - value) x the epoch-1 train loss, "
                        "then by --selection_metric (earlier scores are discarded); 0 = off")
    p.add_argument("--patience_after_loss", type=float, default=0.0,
                   help=">0: count patience only from the first epoch whose train loss < this value; 0 = off")
    p.add_argument("--stuck_epoch", type=int, default=0,
                   help=">=2: at the end of this epoch, stop with exit code 3 if the train loss dropped less than --stuck_min_drop "
                        "relative to the previous epoch (0 = off)")
    p.add_argument("--stuck_min_drop", type=float, default=0.02, help="minimum relative drop of the train loss between two epochs")
    p.add_argument("--grad_checkpoint", type=int, default=1)
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--result_path", default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.stuck_epoch == 1 or args.stuck_epoch < 0:
        p.error("--stuck_epoch must be 0 (off) or >= 2")
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
    assert ck.get("architecture") == ARCHITECTURE, f"{path}: not a multi-window checkpoint"
    keep = {k: v for k, v in ck["model_state_dict"].items() if k.startswith(INIT_PARTS[parts])}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    head = "head loaded from Phase 1" if any(k.startswith("vul_head.") for k in keep) else "new head"
    logger.info("Initialized from %s | parts=%s | loaded %d tensors | %s | missing (outside backbone): %s", path, parts, len(keep), head,
                [m for m in missing if not m.startswith("backbone.")][:8])
    assert not unexpected, unexpected


def build_optimizer(model, args, total_steps):
    o_params = [q for n, q in model.named_parameters() if q.requires_grad]
    if not args.recadam:
        return torch.optim.AdamW([{"params": o_params, "lr": args.learning_rate}], lr=args.learning_rate, weight_decay=args.weight_decay)
    t0 = args.anneal_t0_steps if args.anneal_t0_steps > 0 else max(1, int(args.anneal_t0_ratio * total_steps))
    h_params = [q for n, q in model.named_parameters() if args.recadam_free_head and n.startswith("vul_head.") and q.requires_grad]
    o_params = [q for q in o_params if all(q is not h for h in h_params)]
    anchor = [q.detach().clone() for q in o_params + h_params]
    logger.info("RecAdam | cof=%.0f | k=%g | t0=%d/%d | anchored %d tensors | free head %d tensors", args.pretrain_cof, args.anneal_k,
                t0, total_steps, len(o_params), len(h_params))
    groups = [{"params": o_params, "lr": args.learning_rate, "pretrain_params": anchor[: len(o_params)]}]
    if h_params:
        groups.append({"params": h_params, "lr": args.learning_rate, "anneal_w": 0.0, "pretrain_params": anchor[len(o_params):]})
    return RecAdam(groups, lr=args.learning_rate, weight_decay=args.weight_decay, anneal_fun=args.anneal_fun, anneal_k=args.anneal_k,
                   anneal_t0=t0, anneal_w=1.0, pretrain_cof=args.pretrain_cof, pretrain_params=anchor)


def selection_score(metric, v, train_loss):
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
    cov = {"train": tr.dataset.coverage(), "val": va.dataset.coverage()}
    logger.info("Coverage W=%d S=%d K=%d | preprocessing %.0fs | %s", args.window, args.stride, args.max_windows,
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
    if args.lr_schedule == "constant":
        sched = get_constant_schedule_with_warmup(opt, int(total * args.warmup_ratio))
    else:
        sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
    logger.info("LR schedule: %s | warm-up %d/%d steps | initial lr %.3e", args.lr_schedule, int(total * args.warmup_ratio), total,
                opt.param_groups[0]["lr"])

    best, best_ep, wait, t0, losses = -math.inf, 0, 0, time.perf_counter(), []
    best2, also_path = -math.inf, str(Path(args.checkpoint_path).with_name(f"best_{args.also_select}.pt"))
    counting = args.patience_after_loss <= 0
    gate_open = args.select_after_drop <= 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, tn, te = 0.0, 0, time.perf_counter()
        for b in tr:
            bd, y = to_dev(b, device), b["labels"].to(device)
            loss = F.cross_entropy(model(bd, args.micro)["vul_logits"], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if sam is not None:
                named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
                params = [q for _, q in named]
                if sam.ascend(params, names=[n for n, _ in named]):
                    opt.zero_grad(set_to_none=True)
                    F.cross_entropy(model(bd, args.micro)["vul_logits"], y).backward()
                    sam.restore(params)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            tl += loss.item() * y.size(0)
            tn += y.size(0)
        losses.append(tl / max(1, tn))
        if not counting and losses[-1] < args.patience_after_loss:
            counting = True
            logger.info("Patience counting starts at epoch %d (train loss %.4f < %.2f)", ep, losses[-1], args.patience_after_loss)
        v = run_eval(model, va, device, args.micro)
        for k in ("roc_auc", "macro_f1", "pr_auc"):
            if v.get(k) is None:
                v[k] = float("nan")
        if not gate_open and losses[-1] < (1 - args.select_after_drop) * losses[0]:
            gate_open = True
            best, wait = -math.inf, 0
            logger.info("Selecting checkpoints by %s from epoch %d (train loss %.4f < %.0f%% of the epoch-1 train loss %.4f)",
                        args.selection_metric, ep, losses[-1], 100 * (1 - args.select_after_drop), losses[0])
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
                logger.info("SAM share ep %d | weight %.1f%% | bias %.1f%% | LN %.1f%% | %d steps (||g*T||^2, T of the original ASAM)",
                            ep, 100 * sh[0]["weight"], 100 * sh[0]["bias"], 100 * sh[0]["LN"], sh[1])
        if ep == args.stuck_epoch:
            stalled, drop = train_loss_stalled(losses, args.stuck_min_drop)
            logger.info("Stall check | epoch %d | train loss %.4f -> %.4f | drop %.2f%% (threshold %.2f%%) | %s", ep, losses[-2],
                        losses[-1], 100 * drop, 100 * args.stuck_min_drop, "STALLED" if stalled else "learning")
            if stalled:
                sys.exit(STUCK_EXIT)
        if ep >= args.min_epochs and wait >= args.patience:
            logger.info("Early stopping | Epoch: %d | Best epoch: %d", ep, best_ep)
            break
    logger.info("LR schedule: %s | final lr %.3e", args.lr_schedule, opt.param_groups[0]["lr"])
    logger.info("Training done | best epoch %d | %.0fs", best_ep, time.perf_counter() - t0)


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
                            "truncated_at_K": sub(np.asarray(ds.n_windows_total) > args.max_windows)},
           "coverage": {"val": va.dataset.coverage(), "test": ds.coverage()}, "hyperparameters": ck["training_args"]}
    Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("test_roc_auc", "test_macro_f1_at_valcal", "best_epoch", "subgroup_roc")}))
    if os.environ.get("DUMP_PROBS", "1") == "1":
        np.savez_compressed(Path(args.result_path).with_suffix(".probs.npz"), probabilities=pr.astype(np.float32),
                            labels=lab.astype(np.int8), n_tokens=nt, val_probabilities=np.asarray(v["probabilities"], np.float32),
                            val_labels=np.asarray(v["labels"], np.int8), threshold=np.float32(thr))
    logger.info("Results: %s", args.result_path)


def main():
    args = parse_args()
    if args.deterministic and os.environ.get("PYTHONHASHSEED") != "42":
        os.environ["PYTHONHASHSEED"] = "42"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    root = Path("model") / args.run_name / "multiwindow" / f"seed_{args.seed}"
    args.checkpoint_path = args.checkpoint_path or str(root / f"fold{args.fold}" / "best.pt")
    args.result_path = args.result_path or str(Path("results") / args.run_name / "multiwindow" / f"seed_{args.seed}" / f"fold{args.fold}.json")
    configure_logging()
    set_seed(args.seed, bool(args.deterministic), bool(args.tf32))
    logger.info("Reproducibility | deterministic=%s CUBLAS=%s PYTHONHASHSEED=%s cudnn.det=%s benchmark=%s tf32(matmul/cudnn)=%s/%s",
                torch.are_deterministic_algorithms_enabled(), os.environ.get("CUBLAS_WORKSPACE_CONFIG"), os.environ.get("PYTHONHASHSEED"),
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    fold_dir = Path(args.data_root) / f"fold{args.fold}"

    def load_split(split, cap):
        return limit_records(load_jsonl(fold_dir / f"{split}.jsonl", args.target_lang), cap, args.seed)

    def make_loader(records, batch_size, shuffle):
        ds = WindowDataset(records, tok, args.window, args.stride, args.max_windows, args.max_length)
        gen = torch.Generator().manual_seed(args.seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers, generator=gen)

    backbone = build_backbone(args.model_name)
    if args.grad_checkpoint and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    model = MultiWindowModel(backbone, args.max_windows, args.agg, args.agg_layers, pooling=args.pooling).to(device)
    logger.info("Multi-window | window %d | stride %d | max windows %d | agg %s | pooling %s", args.window, args.stride,
                args.max_windows, args.agg, args.pooling)
    (train if args.phase == "train" else test)(model, args, device, make_loader, load_split)


if __name__ == "__main__":
    main()
