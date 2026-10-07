#!/usr/bin/env python3
"""Plain CodeBERT binary fine-tuning baseline on the five Python folds."""

import argparse
import json
import os
import time
from pathlib import Path

import sklearn
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from evaluate import (
    classification_metrics,
    evaluate,
    find_best_threshold,
    log_per_cwe_reports,
    per_cwe_metrics,
    print_classification_report,
)
from logging_utils import configure_logging, get_logger
from model import BaselineModel, build_backbone
from train import train_loop
from train_transfer import (
    CLASS_TO_CWE,
    build_dataloader,
    limit_records,
    load_jsonl,
    print_dataset_stats,
    python_paths,
    set_seed,
    training_args_dict,
)

logger = get_logger()


# ====================== [RESOURCE] do tai nguyen tung o =======================
# Tap chi so theo Ni et al. 2024 Bang 12 (arXiv:2408.07526), DeepDFA Bang 5
# (arXiv:2212.08108) va Green AI (arXiv:1907.10597). Chi tiet: clean/RESOURCE_LOGGING.md
try:
    from resource_log import ResourceLog
except Exception:                                   # thieu module -> chay nhu cu
    ResourceLog = None


class _NullRL:
    """Khong do gi. Mot o co so ma thieu gio van dung hon mot o trong."""
    def stage(self, name):
        import contextlib
        return contextlib.nullcontext()
    def epoch(self, *a, **k): pass
    def model_stats(self, *a, **k): return {}
    def macs(self, *a, **k): return None
    def tokens(self, *a, **k): return {}
    def dump(self, *a, **k): return {}


def _mk_reslog(device, args):
    if ResourceLog is None:
        return _NullRL()
    try:
        return ResourceLog(device, args,
                           usd_per_hour=float(os.environ.get("USD_PER_HOUR", "0") or 0))
    except Exception as exc:
        logger.warning("[RESOURCE] khong khoi tao duoc: %s", exc)
        return _NullRL()


def _sync():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()      # khong sync thi do trung luc PHONG kernel
    except Exception:
        pass


def _token_budget(rl, split, loader, tokenizer, args):
    """Do NGAN SACH TOKEN that: bao nhieu vi tri thuc su chay qua model, va cat cut
    bao nhieu. Day la cot tra loi "512 so voi khong cat cut thi doi bao nhieu"."""
    ds = getattr(loader, "dataset", None)
    if ds is None or not hasattr(ds, "records"):
        return
    try:
        specials = tokenizer.num_special_tokens_to_add(pair=False)
        budget = args.max_length - specials
        raw = []
        for r in ds.records:
            raw.append(len(tokenizer(r["code"], add_special_tokens=False, truncation=False,
                                     return_attention_mask=False, verbose=False)["input_ids"]))
        used = [min(n, budget) + specials for n in raw]
        d = rl.tokens(split, lengths=used, windows=[1] * len(used))
        if d:
            d["tok_raw_mean"] = round(sum(raw) / max(len(raw), 1), 1)
            d["tok_raw_max"] = max(raw) if raw else 0
            d["n_truncated"] = sum(1 for n in raw if n > budget)
            d["frac_truncated"] = round(d["n_truncated"] / max(len(raw), 1), 4)
            d["tokens_discarded"] = sum(max(0, n - budget) for n in raw)
            d["max_length"] = args.max_length
    except Exception as exc:
        logger.warning("[RESOURCE] khong do duoc token budget cho %s: %s", split, exc)
# =============================================================================


def make_model(model_name, device, pooling="cls"):
    return BaselineModel(build_backbone(model_name), pooling=pooling).to(device)


def parameter_counts(model):
    output = {}
    for name in ("backbone", "vul_head"):
        parameters = list(getattr(model, name).parameters())
        output[name] = {
            "total": sum(parameter.numel() for parameter in parameters),
            "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            "frozen": sum(parameter.numel() for parameter in parameters if not parameter.requires_grad),
        }
    return output


def log_environment(args, model, device, mode):
    components = parameter_counts(model)
    logger.info(
        "Environment: %s",
        json.dumps(
            {
                "mode": mode,
                "seed": args.seed,
                "fold": args.fold,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "sklearn": sklearn.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "model_name": args.model_name,
                "components": components,
                "trainable_parameters": sum(item["trainable"] for item in components.values()),
                "frozen_parameters": sum(item["frozen"] for item in components.values()),
                "hyperparameters": training_args_dict(args),
            },
            sort_keys=True,
        ),
    )


def save_checkpoint(path, model, epoch, score, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": epoch,
            "best_val_macro_f1": score,
            "seed": args.seed,
            "fold": args.fold,
            "model_name": args.model_name,
            "architecture": "codebert_binary_cls",
            "training_args": training_args_dict(args),
        },
        path,
    )


def load_checkpoint(path, model, device, args):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "codebert_binary_cls":
        raise ValueError(f"{path}: not a CodeBERT binary baseline checkpoint")
    if checkpoint.get("model_name") != args.model_name:
        raise ValueError(
            f"checkpoint model={checkpoint.get('model_name')!r} != current model={args.model_name!r}"
        )
    saved_strategy = checkpoint.get("training_args", {}).get("truncation_strategy", "head")
    if saved_strategy != args.truncation_strategy:
        raise ValueError(
            f"checkpoint truncation_strategy={saved_strategy!r} != current "
            f"strategy={args.truncation_strategy!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def loaders(args, tokenizer, include_test=False):
    train_path, val_path, test_path = python_paths(args)
    logger.info("Loading Python validation data: %s", val_path)
    val_records = limit_records(
        load_jsonl(val_path, args.target_lang), args.max_eval_samples, args.seed
    )
    val_loader = build_dataloader(
        val_records,
        tokenizer,
        args.max_length,
        args.eval_batch_size,
        False,
        args.seed,
        args.num_workers,
        args.truncation_strategy,
    )
    if include_test:
        logger.info("Loading Python test data: %s", test_path)
        print_dataset_stats("python_val", val_records)
        test_records = limit_records(
            load_jsonl(test_path, args.target_lang), args.max_eval_samples, args.seed
        )
        print_dataset_stats("python_test", test_records)
        test_loader = build_dataloader(
            test_records,
            tokenizer,
            args.max_length,
            args.eval_batch_size,
            False,
            args.seed,
            args.num_workers,
            args.truncation_strategy,
        )
        return val_loader, test_loader

    logger.info("Loading Python training data: %s", train_path)
    train_records = limit_records(
        load_jsonl(train_path, args.target_lang), args.max_train_samples, args.seed
    )
    print_dataset_stats("python_train", train_records)
    print_dataset_stats("python_val", val_records)
    train_loader = build_dataloader(
        train_records,
        tokenizer,
        args.max_length,
        args.batch_size,
        True,
        args.seed,
        args.num_workers,
        args.truncation_strategy,
    )
    return train_loader, val_loader


def run_train(args, device):
    rl = _mk_reslog(device, args)                       # [RESOURCE]
    args._reslog = rl
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    with rl.stage("preprocess"):                        # [RESOURCE] Ni Bang 12 cot Pre-processing
        train_loader, val_loader = loaders(args, tokenizer)
    with rl.stage("tokencount"):                        # [RESOURCE] do rieng, khong tinh vao preprocess
        _token_budget(rl, "train", train_loader, tokenizer, args)
        _token_budget(rl, "val", val_loader, tokenizer, args)
    # set_seed is called before this fresh model initialization for every fold.
    model = make_model(args.model_name, device, args.pooling)
    rl.model_stats(model, components={                  # [RESOURCE] Ni cot Parameter
        "backbone": getattr(model, "encoder", None) or getattr(model, "backbone", None),
        "vul_head": getattr(model, "vul_head", None),
    })
    log_environment(args, model, device, "baseline_train")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # Warmup tuyen tinh roi giam tuyen tinh. Mac dinh --warmup_ratio 0 -> KHONG tao
    # scheduler, duong chay cu giu nguyen tung buoc.
    args._scheduler = None
    if getattr(args, "warmup_ratio", 0.0) > 0:
        from transformers import get_linear_schedule_with_warmup
        total = len(train_loader) * args.epochs
        warm = int(total * args.warmup_ratio)
        args._scheduler = get_linear_schedule_with_warmup(optimizer, warm, total)
        logger.info("Warmup bat | ty le %.2f -> %d/%d buoc", args.warmup_ratio, warm, total)
    args._val_keep = None
    if getattr(args, "val_keep_mask", None):
        import numpy as _np; args._val_keep = _np.load(args.val_keep_mask).astype(bool)
    with rl.stage("train"):                             # [RESOURCE] Ni cot Training
        train_loop(
            args,
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            "baseline",
            save_checkpoint,
        )
        _sync()
    try:                                                # [RESOURCE]
        rl.macs(model, (next(iter(train_loader))["input_ids"][:1].to(device),
                        next(iter(train_loader))["attention_mask"][:1].to(device)))
    except Exception:
        pass
    rl.dump((args.checkpoint_path or "baseline") + ".resource_train.json",
            extra={"phase": "train", "run_name": args.run_name, "fold": args.fold,
                   "seed": args.seed})


def run_test(args, device):
    rl = _mk_reslog(device, args)                       # [RESOURCE]
    args._reslog = rl
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    with rl.stage("preprocess"):
        val_loader, test_loader = loaders(args, tokenizer, include_test=True)
    with rl.stage("tokencount"):
        _token_budget(rl, "val", val_loader, tokenizer, args)
        _token_budget(rl, "test", test_loader, tokenizer, args)
    model = make_model(args.model_name, device, args.pooling)
    rl.model_stats(model, components={
        "backbone": getattr(model, "encoder", None) or getattr(model, "backbone", None),
        "vul_head": getattr(model, "vul_head", None),
    })
    checkpoint = load_checkpoint(args.checkpoint_path, model, device, args)
    log_environment(args, model, device, "baseline_inference_no_optimizer")

    validation_started = time.perf_counter()
    with rl.stage("infer_val"):                         # [RESOURCE] DeepDFA Bang 5: ms/mau
        val = evaluate(model, val_loader, device)
        _sync()
    threshold, calibrated_val_f1 = find_best_threshold(val["labels"], val["probabilities"])
    validation_seconds = time.perf_counter() - validation_started
    logger.info(
        "Evaluation time | Split: validation+threshold | Seconds: %.2f",
        validation_seconds,
    )
    print_classification_report("validation", val["labels"], val["probabilities"], 0.5)
    print_classification_report("validation_calibrated", val["labels"], val["probabilities"], threshold)

    test_started = time.perf_counter()
    with rl.stage("infer_test"):                        # [RESOURCE] Ni cot Inferring (ca split)
        test = evaluate(model, test_loader, device)
        _sync()
    test_seconds = time.perf_counter() - test_started
    logger.info("Evaluation time | Split: test | Seconds: %.2f", test_seconds)
    test_at_05 = classification_metrics(test["labels"], test["probabilities"], 0.5)
    test_at_valcal = classification_metrics(test["labels"], test["probabilities"], threshold)
    print_classification_report("test", test["labels"], test["probabilities"], 0.5)
    print_classification_report("test_valcal", test["labels"], test["probabilities"], threshold)
    log_per_cwe_reports(
        test["labels"], test["probabilities"], test["cwe_classes"], 0.5,
        CLASS_TO_CWE, "test_at_0.5"
    )
    log_per_cwe_reports(
        test["labels"], test["probabilities"], test["cwe_classes"], threshold,
        CLASS_TO_CWE, "test_at_valcal"
    )
    per_cwe_at_05 = per_cwe_metrics(
        test["labels"], test["probabilities"], test["cwe_classes"], 0.5, CLASS_TO_CWE
    )
    per_cwe_at_valcal = per_cwe_metrics(
        test["labels"], test["probabilities"], test["cwe_classes"], threshold, CLASS_TO_CWE
    )

    result = {
        "experiment_name": f"{args.run_name}/{args.method_name}",
        "phase": "test",
        "fold": args.fold,
        "seed": args.seed,
        "source_checkpoint": None,
        "target_checkpoint": str(args.checkpoint_path),
        "best_epoch": checkpoint["best_epoch"],
        "best_val_macro_f1_at_0.5": checkpoint["best_val_macro_f1"],
        "val_calibrated_threshold": threshold,
        "val_macro_f1_at_valcal": calibrated_val_f1,
        "validation_and_threshold_seconds": validation_seconds,
        "test_inference_seconds": test_seconds,
        "test_macro_f1_at_0.5": test_at_05["macro_f1"],
        "test_macro_f1_at_valcal": test_at_valcal["macro_f1"],
        "test_positive_f1_at_0.5": test_at_05["positive_f1"],
        "test_positive_f1_at_valcal": test_at_valcal["positive_f1"],
        "test_precision_at_0.5": test_at_05["precision"],
        "test_precision_at_valcal": test_at_valcal["precision"],
        "test_recall_at_0.5": test_at_05["recall"],
        "test_recall_at_valcal": test_at_valcal["recall"],
        "test_accuracy_at_0.5": test_at_05["accuracy"],
        "test_accuracy_at_valcal": test_at_valcal["accuracy"],
        "test_roc_auc": test_at_05["roc_auc"],
        "test_pr_auc": test_at_05["pr_auc"],
        "per_cwe": per_cwe_at_valcal,
        "per_cwe_at_0.5": per_cwe_at_05,
        "per_cwe_at_valcal": per_cwe_at_valcal,
        "hyperparameters": checkpoint["training_args"],
    }
    # DIAGNOSTIC: luu xac suat test+val de chay duoc paired bootstrap / DeLong cho HIEU
    # hai mo hinh. Nhanh chuyen da co doan nay (train_transfer.py:968); moc thi chua,
    # nen truoc 13/09 khong tinh duoc SE(Delta). Khong dung so nao trong result/.
    if os.environ.get("DUMP_PROBS", "1") == "1":
        try:
            import numpy as _np
            _dump = Path(args.result_path).with_suffix(".probs.npz")
            _dump.parent.mkdir(parents=True, exist_ok=True)
            _np.savez_compressed(
                _dump,
                probabilities=_np.asarray(test["probabilities"], dtype=_np.float32),
                labels=_np.asarray(test["labels"], dtype=_np.int8),
                cwe_classes=_np.asarray(test["cwe_classes"]),
                val_probabilities=_np.asarray(val["probabilities"], dtype=_np.float32),
                val_labels=_np.asarray(val["labels"], dtype=_np.int8),
                threshold=_np.float32(threshold),
            )
            logger.info("Diagnostic probabilities saved | Path: %s", _dump)
        except Exception as _e:
            logger.warning("Diagnostic probability dump failed (harmless): %s", _e)

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    logger.info("Baseline inference result saved: %s", result_path)
    # [RESOURCE] gop ban ghi tai nguyen cua pha train (neu con) vao mot file duy nhat
    # canh ket qua, de doc bang <result>.resource.json mot lan la du.
    try:
        _tr = (args.checkpoint_path or "") + ".resource_train.json"
        _extra = {"phase": "test", "run_name": args.run_name, "fold": args.fold,
                  "seed": args.seed, "n_test": len(test["labels"]),
                  "n_val": len(val["labels"])}
        if os.path.exists(_tr):
            with open(_tr, encoding="utf-8") as _fh:
                _extra["train_phase"] = json.load(_fh)
        _rp = rl.dump(str(result_path) + ".resource.json", extra=_extra)
        logger.info("[RESOURCE] ghi %s | suy luan test %.2fs | dinh VRAM %s MiB | %s Wh",
                    str(result_path) + ".resource.json",
                    (_rp.get("stages_seconds") or {}).get("infer_test", -1),
                    (_rp.get("memory") or {}).get("gpu_alloc_peak_mb"),
                    (_rp.get("energy") or {}).get("gpu_wh"))
    except Exception as exc:
        logger.warning("[RESOURCE] khong ghi duoc ban ghi tai nguyen: %s", exc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plain CodeBERT binary fine-tuning baseline on Python",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase", choices=("train", "infer", "test"), required=True,
        help="train or run validation-calibrated test inference; test is an infer alias"
    )
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True, help="Python fold")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--run_name", default="default_run", help="parent experiment folder")
    parser.add_argument("--method_name", default="baseline", help="method folder under run_name")
    parser.add_argument("--train_path", help="optional Python train JSONL override")
    parser.add_argument("--val_path", help="optional Python validation JSONL override")
    parser.add_argument("--test_path", help="optional Python test JSONL override")
    parser.add_argument("--checkpoint_path", help="checkpoint to write/read")
    parser.add_argument("--result_path", help="test result JSON")
    parser.add_argument("--target_lang", default="python",
                        help="language every target row must declare")
    parser.add_argument("--data_root", default="data/sven_python_folds_norm",
                        help="directory holding fold1..fold5")
    parser.add_argument("--model_name", default="microsoft/codebert-base", help="backbone")
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls",
                        help="sentence representation; use mean for T5-family encoders "
                             "which have no CLS token")
    parser.add_argument("--epochs", type=int, default=10, help="maximum epochs")
    parser.add_argument("--min_epochs", type=int, default=3, help="minimum epochs before patience")
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="evaluation batch size")
    parser.add_argument("--max_length", type=int, default=512, help="token sequence length")
    parser.add_argument(
        "--truncation_strategy",
        choices=("head", "head_middle_tail"),
        default="head_middle_tail",
        help="long-code token selection",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="ty le buoc warmup tuyen tinh (0 = tat, giu nguyen hanh vi cu)")
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clip norm")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--val_keep_mask", default=None, help="file .npy bool theo thu tu val.jsonl; True = dung de chon checkpoint")
    parser.add_argument("--selection_metric", choices=("macro_f1", "pr_auc", "roc_auc"),
                        default="macro_f1",
                        help="chi so chon checkpoint tren val; train.py doc bang getattr. "
                             "Mac dinh macro_f1 GIU NGUYEN hanh vi cu — dat roc_auc de "
                             "chon epoch bang dung chi so cua goal.")
    parser.add_argument("--max_train_samples", type=int, help="smoke training cap")
    parser.add_argument("--max_eval_samples", type=int, help="smoke evaluation cap")
    args = parser.parse_args()
    if args.epochs < 1 or args.min_epochs < 1 or args.patience < 1:
        parser.error("epochs, min_epochs, and patience must be positive")
    model_root = Path("model") / args.run_name / args.method_name / f"seed_{args.seed}"
    if args.checkpoint_path is None:
        args.checkpoint_path = str(model_root / f"fold{args.fold}" / "best.pt")
    if args.result_path is None:
        args.result_path = (
            f"results/{args.run_name}/{args.method_name}/seed_{args.seed}/fold{args.fold}.json"
        )
    if args.phase in ("infer", "test") and not Path(args.checkpoint_path).is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint_path}")
    return args


def main():
    args = parse_args()
    configure_logging()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    logger.info("Using device: %s", device)
    logger.info(
        "Run started | Run: %s | Method: %s | Phase: baseline_%s | Fold: %d | "
        "Seed: %d | Device: %s",
        args.run_name,
        args.method_name,
        args.phase,
        args.fold,
        args.seed,
        device,
    )
    try:
        if args.phase == "train":
            run_train(args, device)
        else:
            run_test(args, device)
    finally:
        logger.info(
            "Run finished | Phase: baseline_%s | Fold: %d | Seed: %d | Elapsed: %.2fs",
            args.phase,
            args.fold,
            args.seed,
            time.perf_counter() - started,
        )


if __name__ == "__main__":
    main()
