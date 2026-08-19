#!/usr/bin/env python3
"""Plain CodeBERT binary fine-tuning baseline on the five Python folds."""

import argparse
import json
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
        load_jsonl(val_path, "python"), args.max_eval_samples, args.seed
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
            load_jsonl(test_path, "python"), args.max_eval_samples, args.seed
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
        load_jsonl(train_path, "python"), args.max_train_samples, args.seed
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = loaders(args, tokenizer)
    # set_seed is called before this fresh model initialization for every fold.
    model = make_model(args.model_name, device, args.pooling)
    log_environment(args, model, device, "baseline_train")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
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


def run_test(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    val_loader, test_loader = loaders(args, tokenizer, include_test=True)
    model = make_model(args.model_name, device, args.pooling)
    checkpoint = load_checkpoint(args.checkpoint_path, model, device, args)
    log_environment(args, model, device, "baseline_inference_no_optimizer")

    validation_started = time.perf_counter()
    val = evaluate(model, val_loader, device)
    threshold, calibrated_val_f1 = find_best_threshold(val["labels"], val["probabilities"])
    validation_seconds = time.perf_counter() - validation_started
    logger.info(
        "Evaluation time | Split: validation+threshold | Seconds: %.2f",
        validation_seconds,
    )
    print_classification_report("validation", val["labels"], val["probabilities"], 0.5)
    print_classification_report("validation_calibrated", val["labels"], val["probabilities"], threshold)

    test_started = time.perf_counter()
    test = evaluate(model, test_loader, device)
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
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    logger.info("Baseline inference result saved: %s", result_path)


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
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clip norm")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
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
