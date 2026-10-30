import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.append(str(ROOT / "baseline"))
from evaluate import classification_metrics, find_best_threshold, per_cwe_metrics
from train_transfer import CLASS_TO_CWE

logger = logging.getLogger("train_mulvuln")
LANG_NAME = {"js": "JavaScript", "python": "Python"}
NEW_PARAM_PREFIXES = ("pool", "keys", "classifier.")


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MulVulnClassifier(nn.Module):
    def __init__(self, encoder, num_languages, prompt_length, use_pool, query, pool_init="normal"):
        super().__init__()
        self.encoder = encoder
        self.use_pool = use_pool
        self.query_mode = query
        self.num_languages = num_languages
        self.prompt_length = prompt_length
        hidden = encoder.config.d_model
        if use_pool:
            pool, keys = torch.empty(num_languages, prompt_length, hidden), torch.empty(num_languages, hidden)
            for tensor in (pool, keys):
                if pool_init == "uniform":
                    tensor.uniform_(-1.0, 1.0)
                else:
                    tensor.normal_(0.0, 0.02)
            self.pool, self.keys = nn.Parameter(pool), nn.Parameter(keys)
        self.classifier = nn.Linear(hidden, 2)

    def query(self, embeds, mask):
        if self.query_mode == "cls_embed":
            return embeds[:, 0]
        weight = mask.unsqueeze(-1).to(embeds.dtype)
        return (embeds * weight).sum(1) / weight.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask, language_ids=None, selection="eq1"):
        embeds = self.encoder.get_input_embeddings()(input_ids)
        if not self.use_pool:
            hidden = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
            weight = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weight).sum(1) / weight.sum(1).clamp(min=1.0)
            return self.classifier(pooled), None, None
        q = F.normalize(self.query(embeds, attention_mask), dim=-1)
        k = F.normalize(self.keys, dim=-1)
        sim = (q.unsqueeze(1) * k.unsqueeze(0)).sum(-1)
        if selection == "eq2":
            if language_ids is None:
                raise ValueError("Eq. 2 cần nhãn ngôn ngữ của mẫu")
            index = language_ids
        else:
            index = sim.argmax(dim=1)
        onehot = (index.unsqueeze(1) == torch.arange(self.num_languages, device=index.device)).to(sim.dtype)
        chosen_sim = (sim * onehot).sum(1)
        prompt = (onehot[:, :, None, None] * self.pool.unsqueeze(0)).sum(1)
        inputs = torch.cat([prompt.to(embeds.dtype), embeds], dim=1)
        ones = torch.ones(input_ids.size(0), self.prompt_length, dtype=attention_mask.dtype, device=attention_mask.device)
        mask = torch.cat([ones, attention_mask], dim=1)
        hidden = self.encoder(inputs_embeds=inputs, attention_mask=mask).last_hidden_state
        pooled = hidden[:, : self.prompt_length].mean(1)
        return self.classifier(pooled), chosen_sim, index


def build_model(args, device):
    config = AutoConfig.from_pretrained(args.model_name)
    if config.model_type not in ("t5",):
        raise SystemExit(f"--model_name phải là checkpoint họ T5 (CodeT5), nhận model_type={config.model_type!r}")
    encoder = T5EncoderModel.from_pretrained(args.model_name)
    if args.grad_checkpoint:
        encoder.gradient_checkpointing_enable()
    model = MulVulnClassifier(encoder, len(args.languages), args.prompt_length, bool(args.pool), args.query, args.pool_init)
    return model.to(device)


def encode(records, tokenizer, args):
    body = args.max_code_length - 2
    ids = []
    for record in records:
        tokens = [tokenizer.cls_token] + tokenizer.tokenize(str(record["code"]))[:body] + [tokenizer.sep_token]
        row = tokenizer.convert_tokens_to_ids(tokens)
        ids.append(row + [tokenizer.pad_token_id] * (args.max_code_length - len(row)))
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.tensor([int(r["label"]) for r in records], dtype=torch.long)
    languages = torch.tensor([args.languages.index(LANG_NAME[r["lang"]]) for r in records], dtype=torch.long)
    return TensorDataset(input_ids, labels, languages)


def fold_records(args):
    src = ROOT / args.source_root if not os.path.isabs(args.source_root) else Path(args.source_root)
    tgt = (ROOT / args.data_root if not os.path.isabs(args.data_root) else Path(args.data_root)) / f"fold{args.fold}"
    js = {s: load_jsonl(src / f"{s}.jsonl") for s in ("train", "val")}
    py = {s: load_jsonl(tgt / f"{s}.jsonl") for s in ("train", "val", "test")}
    if args.limit_samples > 0:
        js = {k: v[: args.limit_samples] for k, v in js.items()}
        py = {k: v[: args.limit_samples] for k, v in py.items()}
    return js, py


def lr_lambda(kind, warmup, total):
    def factor(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        if kind == "constant":
            return 1.0
        progress = float(step - warmup) / float(max(1, total - warmup))
        if kind == "linear":
            return max(0.0, 1.0 - progress)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return factor


def build_optimizer(args, model, steps_per_epoch):
    new_params = [p for n, p in model.named_parameters() if n.startswith(NEW_PARAM_PREFIXES)]
    enc_params = [p for n, p in model.named_parameters() if not n.startswith(NEW_PARAM_PREFIXES)]
    pool_lr = args.learning_rate if args.pool_lr is None else args.pool_lr
    pool_wd = args.weight_decay if args.pool_weight_decay is None else args.pool_weight_decay
    groups = [{"params": enc_params, "lr": args.learning_rate, "weight_decay": args.weight_decay, "name": "backbone"},
              {"params": new_params, "lr": pool_lr, "weight_decay": pool_wd, "name": "pool_key_classifier"}]
    cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = cls(groups, lr=args.learning_rate)
    total = steps_per_epoch * args.epochs
    warm = int(total * args.warmup_ratio)
    pool_kind = args.scheduler if args.pool_scheduler == "same" else "constant"
    pool_warm = warm if args.pool_scheduler == "same" else 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda(args.scheduler, warm, total),
                                                              lr_lambda(pool_kind, pool_warm, total)])
    info = {"total_steps": total, "steps_per_epoch": steps_per_epoch, "backbone_warmup_steps": warm, "backbone_schedule": args.scheduler,
            "backbone_lr": args.learning_rate, "backbone_weight_decay": args.weight_decay, "new_group_lr": pool_lr,
            "new_group_weight_decay": pool_wd, "new_group_warmup_steps": pool_warm, "new_group_schedule": pool_kind,
            "n_backbone_tensors": len(enc_params), "n_new_tensors": len(new_params),
            "n_backbone_params": sum(p.numel() for p in enc_params), "n_new_params": sum(p.numel() for p in new_params)}
    logger.info("Optimizer %s | backbone: lr %.3g wd %g, %s, warmup %d/%d bước (%.2f × %d bước/epoch × %d epoch) | "
                "pool + khoá + classifier: lr %.3g wd %g, %s, warmup %d/%d bước | %d + %d tham số",
                args.optimizer, args.learning_rate, args.weight_decay, args.scheduler, warm, total, args.warmup_ratio, steps_per_epoch,
                args.epochs, pool_lr, pool_wd, pool_kind, pool_warm, total, info["n_backbone_params"],
                info["n_new_params"])
    return optimizer, scheduler, info


def log_lrs(optimizer, step, total, tag):
    logger.info("LR %s | bước %d/%d | %s", tag, step, total,
                " | ".join("%s %.4e" % (g["name"], g["lr"]) for g in optimizer.param_groups))


@torch.no_grad()
def score_split(model, dataset, args, device):
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size, num_workers=0)
    model.eval()
    out = {"prob": [], "label": [], "lang": [], "pick": [], "sim": [], "ce": []}
    for input_ids, labels, languages in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits, sim, pick = model(input_ids, input_ids.ne(args.pad_token_id).long(), selection="eq1")
        logits = logits.float()
        out["prob"].append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        out["ce"].append(F.cross_entropy(logits, labels, reduction="none").cpu().numpy())
        out["label"].append(labels.cpu().numpy())
        out["lang"].append(languages.numpy())
        out["pick"].append(pick.cpu().numpy() if pick is not None else np.full(len(labels), -1))
        out["sim"].append(sim.float().cpu().numpy() if sim is not None else np.full(len(labels), np.nan))
    return {k: np.concatenate(v) for k, v in out.items()}


def pick_rates(scored, languages):
    if (scored["pick"] < 0).all():
        return None
    out = {}
    for index, name in enumerate(languages):
        sel = scored["lang"] == index
        if sel.any():
            out[name] = {"n": int(sel.sum()), "own_matrix_rate": float((scored["pick"][sel] == index).mean()),
                         "counts": {languages[j]: int((scored["pick"][sel] == j).sum()) for j in range(len(languages))}}
    return out


def val_summary(scored, args):
    py = scored["lang"] == args.languages.index("Python")
    pred_all = (scored["prob"] >= 0.5).astype(int)
    py_metrics = classification_metrics(scored["label"][py].tolist(), scored["prob"][py].tolist(), 0.5) if py.any() else {}
    all_metrics = classification_metrics(scored["label"].tolist(), scored["prob"].tolist(), 0.5)
    summary = {
        "loss": float(scored["ce"].mean()),
        "f1_binary_all": float(f1_score(scored["label"], pred_all, zero_division=0)),
        "roc_auc_all": all_metrics["roc_auc"],
        "f1_binary_python": py_metrics.get("positive_f1"),
        "macro_f1_python": py_metrics.get("macro_f1"),
        "roc_auc_python": py_metrics.get("roc_auc"),
        "eq1_pick": pick_rates(scored, args.languages),
        "mean_cos_chosen": None if np.isnan(scored["sim"]).all() else float(np.nanmean(scored["sim"])),
    }
    key = {("roc_auc", "python"): "roc_auc_python", ("roc_auc", "all"): "roc_auc_all",
           ("f1", "python"): "f1_binary_python", ("f1", "all"): "f1_binary_all"}[(args.selection_metric, args.selection_split)]
    summary["selection_key"] = key
    summary["selection_score"] = summary[key]
    return summary


def fmt(value, spec="%.4f"):
    return "nan" if value is None or value != value else spec % value


def finite_or_none(obj):
    if isinstance(obj, dict):
        return {k: finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_or_none(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_state(model, path):
    tmp = str(path) + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def run_train(args, device):
    js, py = fold_records(args)
    ckpt = Path(args.checkpoint_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(str(ckpt) + ".mulvuln.json")
    if meta_path.exists():
        meta_path.unlink()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.pad_token_id = tokenizer.pad_token_id
    model = build_model(args, device)
    train_set = encode(js["train"] + py["train"], tokenizer, args)
    val_set = encode(js["val"] + py["val"], tokenizer, args)
    logger.info("MulVuln data | fold %d | train %d (JS %d + Python %d, dương %d) | val %d (JS %d + Python %d) | languages %s | "
                "pool %d | query %s | selection train %s, val/test eq1 | λ %s | Lp %d | max_code_length %d",
                args.fold, len(train_set), len(js["train"]), len(py["train"]), int(train_set.tensors[1].sum()), len(val_set),
                len(js["val"]), len(py["val"]), args.languages, args.pool, args.query, args.selection, args.lambda_key,
                args.prompt_length, args.max_code_length)
    loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=args.batch_size, num_workers=0)
    optimizer, scheduler, opt_info = build_optimizer(args, model, len(loader))
    total = opt_info["total_steps"]
    log_points = {0, total // 2, total - 1}
    lam = args.lambda_key if args.pool else 0.0
    best, best_epoch, wait, history = -math.inf, 0, 0, []
    best_summary = None
    step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.perf_counter()
        sums = {"loss": 0.0, "ce": 0.0, "cos": 0.0, "n": 0, "agree": 0}
        for input_ids, labels, languages in loader:
            input_ids, labels, languages = input_ids.to(device), labels.to(device), languages.to(device)
            optimizer.zero_grad(set_to_none=True)
            n_batch = labels.size(0)
            for lo in range(0, n_batch, args.micro_batch):
                hi = min(n_batch, lo + args.micro_batch)
                ids = input_ids[lo:hi]
                logits, sim, pick = model(ids, ids.ne(args.pad_token_id).long(), languages[lo:hi], selection=args.selection)
                ce = F.cross_entropy(logits.float(), labels[lo:hi])
                key_term = sim.mean() if sim is not None else torch.zeros((), device=device)
                loss = ce - lam * key_term
                (loss * ((hi - lo) / n_batch)).backward()
                n = hi - lo
                sums["loss"] += loss.item() * n
                sums["ce"] += ce.item() * n
                sums["cos"] += key_term.item() * n
                sums["n"] += n
                if sim is not None:
                    with torch.no_grad():
                        q = F.normalize(model.query(model.encoder.get_input_embeddings()(ids), ids.ne(args.pad_token_id).long()), dim=-1)
                        eq1 = (q.unsqueeze(1) * F.normalize(model.keys, dim=-1).unsqueeze(0)).sum(-1).argmax(1)
                        sums["agree"] += int((eq1 == pick).sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if step in log_points:
                log_lrs(optimizer, step, total, "đầu" if step == 0 else ("giữa" if step == total // 2 else "cuối"))
            optimizer.step()
            scheduler.step()
            step += 1
        train_loss = sums["loss"] / max(1, sums["n"])
        scored = score_split(model, val_set, args, device)
        summary = val_summary(scored, args)
        score = summary["selection_score"]
        score = -math.inf if score is None or score != score else score
        if score > best:
            best, best_epoch, wait, best_summary = score, epoch, 0, summary
            save_state(model, ckpt)
            logger.info("Best checkpoint saved | Epoch: %d | Val %s: %.6f", epoch, summary["selection_key"], score)
        elif epoch >= args.min_epochs:
            wait += 1
        seconds = time.perf_counter() - t_epoch
        picks = summary["eq1_pick"]
        pick_txt = "-" if picks is None else " ".join("%s %.2f" % (k, v["own_matrix_rate"]) for k, v in picks.items())
        record = {"epoch": epoch, "train_loss": train_loss, "train_ce": sums["ce"] / max(1, sums["n"]),
                  "train_cos": sums["cos"] / max(1, sums["n"]) if args.pool else None,
                  "train_eq1_agree": sums["agree"] / max(1, sums["n"]) if args.pool else None,
                  "seconds": seconds, **{f"val_{k}": v for k, v in summary.items() if k not in ("selection_key",)}}
        history.append(record)
        logger.info("Epoch %d/%d | train loss %.4f | val loss %.4f | val roc_auc %s | val macro_f1 %s | best ep %d | patience %d/%d | %.0fs"
                    " | val JS+Py: F1 %s ROC %s | train CE %.4f cos %s Eq1=Eq2 %s | val Eq.1 chọn đúng: %s",
                    epoch, args.epochs, train_loss, summary["loss"], fmt(summary["roc_auc_python"]), fmt(summary["macro_f1_python"]),
                    best_epoch, wait, args.patience, seconds, fmt(summary["f1_binary_all"]), fmt(summary["roc_auc_all"]),
                    record["train_ce"], fmt(record["train_cos"]), fmt(record["train_eq1_agree"], "%.2f"), pick_txt)
        if epoch >= args.min_epochs and wait >= args.patience:
            logger.info("Early stopping | Epoch: %d | Best epoch: %d", epoch, best_epoch)
            break
    seconds = time.perf_counter() - started
    if best_epoch == 0:
        raise SystemExit("không epoch nào cho điểm val hữu hạn - không có checkpoint")
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
    meta = {"best_epoch": best_epoch, "best_val_selection_score": best, "best_val": best_summary, "history": history,
            "train_seconds": seconds, "peak_vram_allocated_mib": peak, "optimizer_info": opt_info,
            "training_args": training_args(args)}
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(finite_or_none(meta), handle, indent=1, allow_nan=False)
    logger.info("Train xong | best epoch %d | %.0fs | VRAM đỉnh %s MiB", best_epoch, seconds, fmt(peak, "%.0f"))


def training_args(args):
    out = {k: v for k, v in vars(args).items() if k not in ("languages", "pad_token_id")}
    out.update({
        "mv_language_index": {name: i for i, name in enumerate(args.languages)},
        "mv_backbone": "T5EncoderModel (chỉ encoder), fine-tune toàn bộ",
        "mv_query": {"mean_embed": "trung bình đầu ra tầng embedding trên token thật (người dùng chốt 10/10)",
                     "cls_embed": "đầu ra tầng embedding tại token đầu <s> (chữ của bài; hằng với CodeT5)"}[args.query] if args.pool else None,
        "mv_pool_init": ({"normal": "N(0, 0.02²)", "uniform": "U(-1, 1) như L2P"}[args.pool_init] + " cho pool và khoá") if args.pool else None,
        "mv_classifier": ("Linear(D, 2) trên trung bình đầu ra encoder tại Lp vị trí pool" if args.pool
                          else "Linear(D, 2) trên trung bình đầu ra encoder tại token thật"),
        "mv_selection_eval": "eq1",
        "mv_truncation": "cắt đầu: <s> + %d token mã + </s>, đệm tới %d" % (args.max_code_length - 2, args.max_code_length),
        "mv_checkpoint_selection": "val %s trên %s, tăng nghiêm ngặt" % (args.selection_metric, args.selection_split),
        "mv_adam_betas_eps": "(0.9, 0.999), 1e-8",
        "mv_source": "tự cài theo arXiv 2510.04397 (không có mã công bố)",
    })
    return out


def run_test(args, device):
    _js, py = fold_records(args)
    ckpt = Path(args.checkpoint_path)
    meta = json.load(open(str(ckpt) + ".mulvuln.json", encoding="utf-8"))
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.pad_token_id = tokenizer.pad_token_id
    model = build_model(args, device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)

    started = time.perf_counter()
    val = score_split(model, encode(py["val"], tokenizer, args), args, device)
    threshold, calibrated_val_f1 = find_best_threshold(val["label"].tolist(), val["prob"].tolist())
    validation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    test = score_split(model, encode(py["test"], tokenizer, args), args, device)
    test_seconds = time.perf_counter() - started
    test_y, test_p = test["label"].tolist(), test["prob"].tolist()
    assert test_y == [int(r["label"]) for r in py["test"]], "thứ tự / nhãn test lệch test.jsonl"
    cwe = [r.get("cwe_class") for r in py["test"]]
    at05, atval = classification_metrics(test_y, test_p, 0.5), classification_metrics(test_y, test_p, threshold)
    test_pick, val_pick = pick_rates(test, args.languages), pick_rates(val, args.languages)
    py_rate = None if test_pick is None else test_pick["Python"]["own_matrix_rate"]
    logger.info("Test | ROC %.4f | PR %.4f | macro-F1@0.5 %.4f | macro-F1@val %.4f (ngưỡng %.2f) | F1 nhị phân@0.5 %.4f P %.4f R %.4f | "
                "Eq.1: Python test chọn ma trận Python %s", at05["roc_auc"], at05["pr_auc"], at05["macro_f1"], atval["macro_f1"],
                threshold, at05["positive_f1"], at05["precision"], at05["recall"], fmt(py_rate, "%.3f"))
    result = {
        "experiment_name": f"{args.run_name}/{'mulvuln' if args.pool else 't5joint'}", "phase": "test", "fold": args.fold,
        "seed": args.seed, "source_checkpoint": None, "target_checkpoint": str(ckpt), "best_epoch": meta["best_epoch"],
        "best_val_selection_score": meta["best_val_selection_score"], "best_val": meta["best_val"],
        "val_calibrated_threshold": threshold, "val_macro_f1_at_valcal": calibrated_val_f1,
        "validation_and_threshold_seconds": validation_seconds, "test_inference_seconds": test_seconds,
        "train_seconds": meta["train_seconds"], "peak_vram_allocated_mib": meta["peak_vram_allocated_mib"],
        "test_macro_f1_at_0.5": at05["macro_f1"], "test_macro_f1_at_valcal": atval["macro_f1"],
        "test_positive_f1_at_0.5": at05["positive_f1"], "test_positive_f1_at_valcal": atval["positive_f1"],
        "test_precision_at_0.5": at05["precision"], "test_precision_at_valcal": atval["precision"],
        "test_recall_at_0.5": at05["recall"], "test_recall_at_valcal": atval["recall"],
        "test_accuracy_at_0.5": at05["accuracy"], "test_accuracy_at_valcal": atval["accuracy"],
        "test_roc_auc": at05["roc_auc"], "test_pr_auc": at05["pr_auc"],
        "per_cwe": per_cwe_metrics(test_y, test_p, cwe, threshold, CLASS_TO_CWE),
        "per_cwe_at_0.5": per_cwe_metrics(test_y, test_p, cwe, 0.5, CLASS_TO_CWE),
        "per_cwe_at_valcal": per_cwe_metrics(test_y, test_p, cwe, threshold, CLASS_TO_CWE),
        "test_eq1_python_own_matrix_rate": py_rate, "test_eq1_pick": test_pick, "val_python_eq1_pick": val_pick,
        "test_mean_cos_chosen": None if np.isnan(test["sim"]).all() else float(np.nanmean(test["sim"])),
        "mulvuln_history": meta["history"], "optimizer_info": meta["optimizer_info"],
        "hyperparameters": meta["training_args"],
    }
    out = Path(args.result_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".probs.npz"), probabilities=np.asarray(test_p, dtype=np.float32),
                        labels=np.asarray(test_y, dtype=np.int8), cwe_classes=np.asarray(cwe),
                        val_probabilities=val["prob"].astype(np.float32), val_labels=val["label"].astype(np.int8),
                        threshold=np.float32(threshold), pool_index=test["pick"].astype(np.int8),
                        val_pool_index=val["pick"].astype(np.int8), cos_chosen=test["sim"].astype(np.float32))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(finite_or_none(result), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    logger.info("MulVuln result saved: %s", out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("train", "test"), required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--model_name", required=True, help="thư mục CodeT5 (tokenizer + trọng số), chỉ dùng encoder")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--result_path")
    p.add_argument("--data_root", required=True, help="fold SVEN Python (đích), tương đối từ gốc repo hoặc tuyệt đối")
    p.add_argument("--source_root", required=True, help="pool JS (train.jsonl + val.jsonl)")
    p.add_argument("--language", default="Python,JavaScript", help="thứ tự ma trận trong pool: chỉ số k = ngôn ngữ thứ k")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pool", type=int, choices=(0, 1), default=1, help="0 = đối chứng RQ2 (ext_t5joint): không pool, không loss khoá")
    p.add_argument("--prompt_length", type=int, default=5, help="Lp (bài: 5)")
    p.add_argument("--lambda_key", type=float, default=None, help="λ của loss khoá (bài quét 0.1 / 0.3 / 0.01 / 0.03); bắt buộc khi --pool 1")
    p.add_argument("--selection", choices=("eq1", "eq2"), default="eq2", help="cách chọn ma trận LÚC TRAIN; val / test luôn Eq. 1")
    p.add_argument("--query", choices=("mean_embed", "cls_embed"), default="mean_embed",
                   help="q(X): trung bình tầng embedding trên token thật (chốt 10/10) hoặc token <s> của tầng embedding (chữ của bài)")
    p.add_argument("--max_code_length", type=int, default=507, help="số token mã gồm <s> và </s> (507 + Lp 5 = 512)")
    p.add_argument("--pool_init", choices=("normal", "uniform"), default="normal",
                   help="khởi tạo pool + khoá (bài không nêu): normal = N(0, 0.02²); uniform = U(-1, 1) như L2P")
    p.add_argument("--optimizer", choices=("adam", "adamw"), default="adam", help="bài: Adam")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="lr của encoder CodeT5 (bài: 1e-4 cố định)")
    p.add_argument("--weight_decay", type=float, default=0.0, help="weight decay của encoder (Adam của bài: 0)")
    p.add_argument("--warmup_ratio", type=float, default=0.0, help="tỉ lệ bước warmup của encoder (trên tổng bước = bước/epoch × epochs)")
    p.add_argument("--scheduler", choices=("constant", "linear", "cosine"), default="constant", help="lịch lr của encoder sau warmup")
    p.add_argument("--pool_lr", type=float, default=None, help="lr của pool + khoá + classifier; bỏ trống = --learning_rate")
    p.add_argument("--pool_weight_decay", type=float, default=None, help="weight decay của pool + khoá + classifier; bỏ trống = --weight_decay")
    p.add_argument("--pool_scheduler", choices=("same", "constant"), default="same",
                   help="same = cùng lịch + warmup với encoder; constant = lr hằng, không warmup")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=999)
    p.add_argument("--min_epochs", type=int, default=10, help="patience chỉ đếm từ epoch này (khuôn trainer SOTA)")
    p.add_argument("--selection_metric", choices=("roc_auc", "f1"), default="roc_auc", help="f1 = F1 nhị phân (lớp 1) ở argmax")
    p.add_argument("--selection_split", choices=("python", "all"), default="python", help="chấm điểm chọn trên phần Python của val hay toàn val")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--micro_batch", type=int, default=16, help="cỡ lượt forward; tích luỹ gradient cho đủ --batch_size")
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--grad_checkpoint", type=int, choices=(0, 1), default=1)
    p.add_argument("--limit_samples", type=int, default=0, help="CHỈ để thử khói: lấy N bản ghi đầu của mỗi tập; 0 = tắt")
    args = p.parse_args()
    if args.phase == "test" and not args.result_path:
        p.error("--phase test cần --result_path")
    if args.pool and args.lambda_key is None:
        p.error("--pool 1 cần --lambda_key")
    if args.micro_batch < 1 or args.micro_batch > args.batch_size:
        p.error("--micro_batch phải trong [1, --batch_size]")
    if args.max_code_length < 3 or args.max_code_length + (args.prompt_length if args.pool else 0) > 512:
        p.error("--max_code_length (+ Lp nếu có pool) phải ≤ 512")
    if args.epochs < 1 or args.min_epochs < 1 or args.patience < 1:
        p.error("epochs, min_epochs, patience phải dương")
    args.languages = [x.strip() for x in args.language.split(",")]
    if sorted(args.languages) != ["JavaScript", "Python"]:
        p.error("--language phải gồm đúng Python và JavaScript")
    return args


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, force=True)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("MulVuln | phase %s | fold %d | seed %d | device %s | tf32 %s/%s | deterministic %s | torch %s",
                args.phase, args.fold, args.seed, device, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.are_deterministic_algorithms_enabled(), torch.__version__)
    (run_train if args.phase == "train" else run_test)(args, device)


if __name__ == "__main__":
    main()
