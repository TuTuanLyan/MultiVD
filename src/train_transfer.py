#!/usr/bin/env python3
"""CLI for generic source multitask pretraining and Python RecAdam transfer."""

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
import torch
import transformers
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from dataset import CodeDataset
from evaluate import (
    classification_metrics,
    evaluate,
    find_best_threshold,
    log_per_cwe_reports,
    per_cwe_metrics,
    print_classification_report,
)
from logging_utils import configure_logging, get_logger
from model import (TransferModel, build_backbone, freeze_backbone_layers,
                   inject_lora, merge_lora)
from RecAdam import RecAdam, anneal_function
from train import assert_recadam_setup, train_loop


CWE_MAPPING = {"CWE-022": 0, "CWE-078": 1, "CWE-079": 2, "CWE-089": 3}
CWE_ID_MAPPING = {22: 0, 78: 1, 79: 2, 89: 3}
CLASS_TO_CWE = {value: key for key, value in CWE_MAPPING.items()}
GROUP_FIELDS = ("pair_id", "commit_id", "project", "repo", "CVE_ID")
logger = get_logger()


def set_seed(seed, strict=False):
    """Seed every generator, and optionally demand deterministic kernels too.

    Seeding alone does not pin Phase 1 here. Three runs at seed 36 with identical
    arguments produced three different checkpoints -- source validation 0.5722,
    0.6591, 0.6654 -- and the spread across fixed-seed repeats (sd 0.0521) matched
    the spread across three different seeds (sd 0.0496). The seed was buying no
    control at all.

    The amplifier is early stopping: patience 5 over at most 15 epochs turns a
    drift of order 1e-7 into a different stopping decision, and one of those three
    runs stopped at epoch 3 while the others ran to 11 and 13. cudnn.deterministic
    does not cover the non-cuDNN reductions that produce that drift.

    strict=True closes the remaining sources. It is opt-in because it changes
    results relative to every run recorded in RESULT.md, and because determinism
    is not the same thing as low variance -- it makes one draw repeatable, while
    the spread across draws stays real and still has to be averaged over.
    """
    if strict:
        # cuBLAS reads this once, when it creates its handle, so it has to be set
        # before any CUDA work happens.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only: a few backward kernels have no deterministic implementation,
        # and warning beats refusing to run at all.
        torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_cwe_class(record, trust_precomputed=False):
    """Prefer normalized cwe_class; fall back only when that field is absent.

    `trust_precomputed` takes the field verbatim instead of checking it against
    the four-way mapping. Needed whenever a build script writes an auxiliary
    label that is not a CWE index at all -- the CWE-pillar grouping has nine
    classes, and the default path would send every class above 3 to -100 while
    reporting nothing, so the head would silently train on a quarter of the
    taxonomy. `build_edit_labels.py` escaped that only because it happened to
    emit exactly four buckets.
    """
    if "cwe_class" in record:
        try:
            value = int(record["cwe_class"])
        except (TypeError, ValueError):
            return -100
        if trust_precomputed:
            return value if value >= 0 else -100
        return value if value in CLASS_TO_CWE else -100
    try:
        cwe_id = int(record.get("cwe_id"))
    except (TypeError, ValueError):
        cwe_id = None
    if cwe_id in CWE_ID_MAPPING:
        return CWE_ID_MAPPING[cwe_id]
    cwe = str(record.get("cwe", "")).upper()
    if cwe.startswith("CWE-"):
        try:
            return CWE_ID_MAPPING.get(int(cwe.split("-", 1)[1]), -100)
        except ValueError:
            pass
    return -100


def build_cwe_vocab(records):
    """Map every CWE present in the source onto a contiguous auxiliary class index.

    The fixed four-way mapping silently sent everything outside
    {22,78,79,89} to -100, so a source like PrimeVul (121 CWEs here) would
    contribute nothing to the auxiliary task beyond those four. Building the
    vocabulary from the data lets any source use its whole taxonomy; the target
    never sees this head, so its size does not have to match anything.

    Sorted by name so the mapping is stable across runs and machines.
    """
    names = sorted({
        str(record.get("cwe", "")).upper()
        for record in records
        if str(record.get("cwe", "")).upper().startswith("CWE-")
    })
    return {name: index for index, name in enumerate(names)}


def apply_cwe_vocab(records, vocab):
    for record in records:
        name = str(record.get("cwe", "")).upper()
        record["cwe_class"] = vocab.get(name, -100)


def load_jsonl(path, expected_lang=None, trust_precomputed=False):
    expected_languages = None
    if expected_lang is not None:
        expected_languages = {expected_lang} if isinstance(expected_lang, str) else set(expected_lang)
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if "code" not in record or not isinstance(record["code"], str) or not record["code"].strip():
                raise ValueError(f"{path}:{line_number}: missing or empty string field 'code'")
            if "label" not in record:
                raise ValueError(f"{path}:{line_number}: missing field 'label'")
            try:
                record["label"] = int(record["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: label must be 0 or 1") from error
            if record["label"] not in (0, 1):
                raise ValueError(f"{path}:{line_number}: label must be 0 or 1, got {record['label']}")
            lang = record.get("lang")
            legacy_language = record.get("language")
            if lang is not None and legacy_language is not None and lang != legacy_language:
                raise ValueError(
                    f"{path}:{line_number}: conflicting 'lang'={lang!r} and "
                    f"'language'={legacy_language!r}"
                )
            normalized_lang = lang if lang is not None else legacy_language
            if not isinstance(normalized_lang, str) or not normalized_lang.strip():
                raise ValueError(
                    f"{path}:{line_number}: missing or empty language field ('lang' or 'language')"
                )
            record["lang"] = normalized_lang
            if expected_languages is not None and record.get("lang") not in expected_languages:
                raise ValueError(
                    f"{path}:{line_number}: expected lang in {sorted(expected_languages)!r}, "
                    f"got {record.get('lang')!r}"
                )
            record["cwe_class"] = resolve_cwe_class(record, trust_precomputed)
            record["_source_index"] = len(records)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def print_dataset_stats(name, records):
    labels = Counter(record["label"] for record in records)
    cwes = Counter(
        CLASS_TO_CWE[record["cwe_class"]]
        for record in records
        if record["cwe_class"] in CLASS_TO_CWE
    )
    languages = Counter(record.get("lang", "missing") for record in records)
    invalid_cwes = sum(record["cwe_class"] == -100 for record in records)
    logger.info(
        "Data split: %s | Total: %d | Labels: %s | CWEs: %s | Languages: %s | Invalid CWE: %d",
        name,
        len(records),
        dict(sorted(labels.items())),
        dict(sorted(cwes.items())),
        dict(sorted(languages.items())),
        invalid_cwes,
    )


def limit_records(records, maximum, seed):
    if maximum is None or maximum <= 0 or len(records) <= maximum:
        return records
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(records), size=maximum, replace=False).tolist())
    return [records[index] for index in indices]


def source_groups(records):
    """Conservatively group adjacent vulnerable/fixed rows with the same CWE/language."""
    groups = []
    index = 0
    while index < len(records):
        if (
            index + 1 < len(records)
            and records[index + 1]["_source_index"] == records[index]["_source_index"] + 1
            and records[index]["label"] == 1
            and records[index + 1]["label"] == 0
            and records[index]["cwe_class"] == records[index + 1]["cwe_class"]
            and records[index].get("lang") == records[index + 1].get("lang")
        ):
            groups.append(records[index : index + 2])
            index += 2
        elif (
            index + 1 < len(records)
            and records[index + 1]["_source_index"] == records[index]["_source_index"] + 1
            and records[index]["label"] == 0
            and records[index + 1]["label"] == 1
            and records[index]["cwe_class"] == records[index + 1]["cwe_class"]
            and records[index].get("lang") == records[index + 1].get("lang")
            and not (
                index + 2 < len(records)
                and records[index + 2]["_source_index"] == records[index + 1]["_source_index"] + 1
                and records[index + 2]["label"] == 0
                and records[index + 1]["cwe_class"] == records[index + 2]["cwe_class"]
                and records[index + 1].get("lang") == records[index + 2].get("lang")
            )
        ):
            # Accept reversed safe->vulnerable only when the vulnerable row
            # cannot begin the usual 1->0 pair.
            groups.append(records[index : index + 2])
            index += 2
        else:
            groups.append(records[index : index + 1])
            index += 1
    return groups


def limit_source_groups(records, maximum, seed):
    """Smoke-test cap that never separates conservative source groups."""
    if maximum is None or maximum <= 0 or len(records) <= maximum:
        return records
    groups = source_groups(records)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(groups)).tolist()
    selected, count = [], 0
    for index in indices:
        group = groups[index]
        if selected and count + len(group) > maximum:
            continue
        selected.append(index)
        count += len(group)
        if count >= maximum:
            break
    return [record for index in sorted(selected) for record in groups[index]]


def split_source_language(records, seed, language):
    for field in GROUP_FIELDS:
        if all(record.get(field) not in (None, "") for record in records):
            groups = [str(record[field]) for record in records]
            if len(set(groups)) > 1:
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
                train_indices, val_indices = next(splitter.split(records, groups=groups))
                logger.info(
                    "Source split | Language: %s | Group field: %s | Ratio: 90/10 | Seed: %d",
                    language,
                    field,
                    seed,
                )
                return [records[i] for i in train_indices], [records[i] for i in val_indices]

    groups = source_groups(records)
    pair_count = sum(len(group) == 2 for group in groups)
    singleton_count = sum(len(group) == 1 for group in groups)
    if pair_count:
        group_indices = list(range(len(groups)))
        group_strata = [group[0]["cwe_class"] for group in groups]
        group_counts = Counter(group_strata)
        test_group_count = math.ceil(0.10 * len(groups))
        can_stratify = (
            min(group_counts.values()) >= 2 and test_group_count >= len(group_counts)
        )
        train_groups, val_groups = train_test_split(
            group_indices,
            test_size=0.10,
            random_state=seed,
            shuffle=True,
            stratify=group_strata if can_stratify else None,
        )
        logger.info(
            "Source split | Language: %s | Conservative pairs: %d | Singletons: %d | "
            "Stratified: %s | Ratio: 90/10 | Seed: %d",
            language,
            pair_count,
            singleton_count,
            can_stratify,
            seed,
        )
        return (
            [record for index in train_groups for record in groups[index]],
            [record for index in val_groups for record in groups[index]],
        )
    strata = [f"{record['label']}:{record['cwe_class']}" for record in records]
    counts = Counter(strata)
    labels = [record["label"] for record in records]
    label_counts = Counter(labels)
    test_record_count = math.ceil(0.10 * len(records))
    if min(counts.values()) >= 2 and test_record_count >= len(counts):
        stratify = strata
        stratify_name = "label+cwe_class"
    elif min(label_counts.values()) >= 2 and test_record_count >= len(label_counts):
        stratify = labels
        stratify_name = "label"
    else:
        stratify = None
        stratify_name = "none"
    train_records, val_records = train_test_split(
        records, test_size=0.10, random_state=seed, shuffle=True, stratify=stratify
    )
    logger.info(
        "Source split | Language: %s | Fallback stratification: %s | Ratio: 90/10 | Seed: %d",
        language,
        stratify_name,
        seed,
    )
    return train_records, val_records


def split_source_records(records, seed):
    """Split every source language independently while preserving detected groups."""
    by_language = {}
    for record in records:
        by_language.setdefault(record["lang"], []).append(record)
    train_records, val_records = [], []
    for offset, language in enumerate(sorted(by_language)):
        language_train, language_val = split_source_language(
            by_language[language], seed + offset, language
        )
        train_records.extend(language_train)
        val_records.extend(language_val)
    logger.info(
        "Combined source split | Languages: %s | Train: %d | Validation: %d",
        sorted(by_language),
        len(train_records),
        len(val_records),
    )
    logger.warning(
        "No upstream pair/repo/commit IDs: adjacent groups are protected, but distant "
        "near-duplicate families may still cross source train/validation"
    )
    return train_records, val_records


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    records, tokenizer, max_length, batch_size, shuffle, seed, num_workers, truncation_strategy
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        CodeDataset(records, tokenizer, max_length, truncation_strategy),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )


def make_model(model_name, device, args=None):
    backbone = build_backbone(model_name)
    aux_mode = getattr(args, "aux_mode", "cwe") if args else "cwe"
    num_latent = getattr(args, "num_latent", 8) if args else 8
    num_cwes = getattr(args, "num_cwes", 4) if args else 4
    temperature = getattr(args, "latent_temperature", 0.1) if args else 0.1
    pooling = getattr(args, "pooling", "cls") if args else "cls"
    model = TransferModel(
        backbone,
        num_classes=2,
        num_cwes=num_cwes,
        aux_mode=aux_mode,
        num_latent=num_latent,
        latent_temperature=temperature,
        pooling=pooling,
    ).to(device)
    if args is not None and getattr(args, "lora_rank", 0) > 0 and args.phase == "phase1":
        count = inject_lora(model.backbone, args.lora_rank)
        # inject_lora builds fresh parameters on CPU, so the model has to move
        # again; it was already on the device before the wrapping happened.
        model.to(device)
        logger.info(
            "LoRA injected | Rank: %d | Projections wrapped: %d | Backbone otherwise frozen",
            args.lora_rank, count,
        )
    if args is not None and getattr(args, "freeze_backbone_layers", 0) > 0:
        info = freeze_backbone_layers(model.backbone, args.freeze_backbone_layers)
        logger.info("Backbone partially frozen | %s", json.dumps(info, sort_keys=True))
    if hasattr(model, "freeze_prototypes_steps") and args is not None:
        model.freeze_prototypes_steps = getattr(args, "freeze_prototypes_steps", 0)
    return model


def freeze_aux_head(model):
    """Freeze every auxiliary parameter; Phase 2 optimizes backbone + vul_head only."""
    for parameter in model.aux_parameters():
        parameter.requires_grad = False


def component_parameter_counts(model):
    output = {}
    for name in ("backbone", "vul_head", "latent_proj", "cwe_head"):
        if not hasattr(model, name):
            continue
        parameters = list(getattr(model, name).parameters())
        output[name] = {
            "total": sum(parameter.numel() for parameter in parameters),
            "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            "frozen": sum(parameter.numel() for parameter in parameters if not parameter.requires_grad),
        }
    return output


def training_args_dict(args):
    scalar_types = (str, int, float, bool, type(None))
    return {key: value for key, value in vars(args).items() if isinstance(value, scalar_types)}


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
            "cwe_mapping": CWE_MAPPING,
            "aux_mode": model.aux_mode,
            "num_cwes": getattr(args, "num_cwes", 4),
            "pooling": model.pooling,
            "num_latent": getattr(model, "num_latent", None),
            "training_args": training_args_dict(args),
        },
        path,
    )


def adopt_checkpoint_shape(path, device):
    """Read the auxiliary head width the checkpoint was written with."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    num_cwes = saved.get("num_cwes")
    if num_cwes is None:
        weight = saved.get("model_state_dict", {}).get("cwe_head.weight")
        num_cwes = weight.shape[0] if weight is not None else 4
    logger.info("Auxiliary head width taken from checkpoint: %d", num_cwes)
    return int(num_cwes)


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def assert_checkpoint_compatible(checkpoint, args, checkpoint_name):
    saved_args = checkpoint.get("training_args", {})
    saved_model = checkpoint.get("model_name")
    if saved_model != args.model_name:
        raise ValueError(
            f"{checkpoint_name} model_name={saved_model!r} does not match --model_name={args.model_name!r}"
        )
    saved_aux = checkpoint.get("aux_mode", "cwe")
    if saved_aux != args.aux_mode:
        raise ValueError(
            f"{checkpoint_name} used aux_mode={saved_aux!r}, but current run uses "
            f"{args.aux_mode!r}; the auxiliary head shapes differ so the state dict "
            f"cannot be loaded"
        )
    saved_pooling = checkpoint.get("pooling", "cls")
    if saved_pooling != args.pooling:
        raise ValueError(
            f"{checkpoint_name} used pooling={saved_pooling!r}, but current run uses "
            f"{args.pooling!r}; the representation would change meaning"
        )
    saved_strategy = saved_args.get("truncation_strategy", "head")
    if saved_strategy != args.truncation_strategy:
        raise ValueError(
            f"{checkpoint_name} used truncation_strategy={saved_strategy!r}, but current run uses "
            f"{args.truncation_strategy!r}; retrain the preceding phase to avoid incompatible inputs"
        )


def log_environment(args, model, device, mode):
    components = component_parameter_counts(model)
    trainable = sum(item["trainable"] for item in components.values())
    frozen = sum(item["frozen"] for item in components.values())
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
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
                "gpu": gpu,
                "model_name": args.model_name,
                "trainable_parameters": trainable,
                "frozen_parameters": frozen,
                "components": components,
                "hyperparameters": training_args_dict(args),
            },
            sort_keys=True,
        ),
    )


def run_phase1(args, device):
    logger.info("Loading Phase-1 source data: %s", args.data_path)
    records = load_jsonl(args.data_path, trust_precomputed=args.cwe_vocab == "precomputed")
    logger.info("Detected Phase-1 languages: %s", sorted({record["lang"] for record in records}))
    if args.cwe_vocab == "source":
        vocab = build_cwe_vocab(records)
        apply_cwe_vocab(records, vocab)
        args.num_cwes = max(1, len(vocab))
        logger.info(
            "Auxiliary CWE vocabulary built from source | Classes: %d | Sample: %s",
            len(vocab),
            dict(list(vocab.items())[:6]),
        )
    elif args.cwe_vocab == "precomputed":
        # The file already carries the auxiliary label. Rebuilding it from `cwe`
        # would overwrite exactly what the build script wrote, which is how the
        # pillar grouping would have silently run as plain CWE classification.
        present = sorted({r["cwe_class"] for r in records if r["cwe_class"] >= 0})
        if present != list(range(len(present))):
            raise ValueError(
                f"cwe_class in {args.data_path} must be 0..N-1 (or -100); got {present[:12]}"
            )
        args.num_cwes = max(1, len(present))
        counts = Counter(r["cwe_class"] for r in records)
        logger.info(
            "Auxiliary labels taken from file | Classes: %d | Counts: %s | Ignored: %d",
            args.num_cwes, {c: counts[c] for c in present}, counts[-100],
        )
    else:
        args.num_cwes = 4
        logger.info("Auxiliary CWE vocabulary: fixed four-way %s", CWE_MAPPING)
    train_records, val_records = split_source_records(records, args.seed)
    train_records = limit_source_groups(train_records, args.max_train_samples, args.seed)
    val_records = limit_source_groups(val_records, args.max_eval_samples, args.seed)
    print_dataset_stats("source_train", train_records)
    print_dataset_stats("source_val", val_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = make_model(args.model_name, device, args)
    log_environment(args, model, device, "phase1_train")
    train_loader = build_dataloader(
        train_records, tokenizer, args.max_length, args.batch_size, True, args.seed, args.num_workers,
        args.truncation_strategy
    )
    val_loader = build_dataloader(
        val_records, tokenizer, args.max_length, args.eval_batch_size, False, args.seed, args.num_workers,
        args.truncation_strategy
    )
    aux_weighter = None
    if getattr(args, "aux_weight_mode", "fixed") == "uncertainty":
        if args.aux_mode == "none":
            logger.info("aux_weight_mode=uncertainty nhung aux_mode=none — khong co loss phu "
                        "de can, chay nhu binh thuong")
        else:
            from uncertainty_weighting import UncertaintyWeights
            aux_weighter = UncertaintyWeights(init_lambda=args.lambda_cwe).to(device)
            logger.info("lambda hoc duoc BAT | khoi tao tai lambda_eff=%.4f (dung bang --lambda_cwe, "
                        "de day nay la mo rong that su cua baseline)", args.lambda_cwe)

    param_groups = [
        {"params": list(model.parameters()), "weight_decay": args.weight_decay},
    ]
    if aux_weighter is not None:
        # KHONG weight decay tren hai vo huong log-phuong sai: decay se keo chung
        # ve 0, tuc keo sigma^2 ve 1, tuc ap dat mot lambda cu the — dung thu ma
        # thi nghiem nay dang co gang khong ap dat.
        #
        # VA learning rate RIENG, lon hon nhieu. Do truc tiep: voi lr chung 2e-5,
        # Adam di ~lr moi buoc bat ke do lon gradient, nen qua ~1200 buoc cua mot
        # Phase 1 that thi s chi dich duoc ~0.024 — lambda_eff doi ~2%. "Lambda hoc
        # duoc" khi do that ra la lambda co dinh, va thi nghiem se tra ve mot ket
        # qua null khong co y nghia gi. Thu don vi cho thay lr ~5e-2 hoi tu trong
        # ~100 buoc.
        param_groups.append({
            "params": list(aux_weighter.parameters()),
            "weight_decay": 0.0,
            "lr": args.aux_weight_lr,
        })
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    train_loop(
        args, model, train_loader, val_loader, optimizer, device, "phase1", save_checkpoint,
        aux_weighter=aux_weighter,
    )
    if aux_weighter is not None:
        d = aux_weighter.diagnostics()
        logger.info("lambda hoc duoc KET THUC | lambda_eff %.4f (khoi tao %.4f) | s_bin %+.4f | s_aux %+.4f",
                    d["lambda_eff"], args.lambda_cwe, d["s_binary"], d["s_aux"])
        args.learned_lambda_eff = d["lambda_eff"]
        args.learned_s_binary = d["s_binary"]
        args.learned_s_aux = d["s_aux"]

    if args.lora_rank > 0:
        # The best checkpoint was written mid-training and still carries LoRA
        # modules. Reload it, fold the update into the frozen weights, and write
        # it back so Phase 2 loads a plain backbone.
        saved = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        merged = merge_lora(model.backbone)
        saved["model_state_dict"] = model.state_dict()
        saved["lora_merged"] = merged
        torch.save(saved, args.checkpoint_path)
        logger.info("LoRA merged into the saved checkpoint | Projections: %d", merged)


def python_paths(args):
    base = Path(getattr(args, "data_root", None) or "data/sven_python_folds_norm")
    base = base / f"fold{args.fold}"
    return (
        Path(args.train_path) if args.train_path else base / "train.jsonl",
        Path(args.val_path) if args.val_path else base / "val.jsonl",
        Path(args.test_path) if args.test_path else base / "test.jsonl",
    )


def print_recadam_schedule(args, total_steps, anneal_t0, steps_per_epoch):
    checkpoints = sorted(set([1, steps_per_epoch, anneal_t0, total_steps]))
    values = {
        step: anneal_function(args.anneal_fun, step, args.anneal_k, anneal_t0, args.anneal_w)
        for step in checkpoints
    }
    logger.info(
        "RecAdam schedule | Total steps: %d | Steps/epoch: %d | Anneal t0: %d | Weights: %s",
        total_steps,
        steps_per_epoch,
        anneal_t0,
        values,
    )


def build_recadam_anchor(named_trainable, args, device):
    """Clone the point RecAdam pulls Phase 2 back toward.

    The default anchors on the Phase-1 weights, which is what RecAdam was
    written for. That choice is not neutral once the backbone is strong. §34.2
    measured, per CWE class, that bare source pretraining costs CodeT5+ -0.0312
    on CWE-078 -- 204 samples, the class it already handled well -- and that the
    auxiliary head recovers only +0.0056 of it, against +0.0195 on CodeBERT.
    So on a strong backbone the anchor is holding the model at the very solution
    that damaged the common classes.

    `pretrained` anchors on the untouched HF weights instead. Initialisation
    still comes from Phase 1, so whatever the source stage bought on the rare
    classes is still there at step 0; only the pull changes direction, back
    toward the general representation the common classes rely on.

    This is not the interpolation that §15 rejected. That one blended the
    *starting weights* and so gave up part of the source solution before
    training began. Here Phase 1 arrives intact and only the regulariser moves.
    """
    anchor = [parameter.detach().clone() for _, parameter in named_trainable]
    if args.recadam_anchor == "source":
        return anchor

    pretrained = build_backbone(args.model_name).to(device)
    state = pretrained.state_dict()
    replaced, kept = 0, 0
    for index, (name, parameter) in enumerate(named_trainable):
        prefix = "backbone."
        original = state.get(name[len(prefix):]) if name.startswith(prefix) else None
        if original is None or original.shape != parameter.shape:
            # Heads have no pretrained counterpart, so they stay anchored on
            # Phase 1. Leaving them unanchored would let the classifier drift
            # freely while the backbone is held, which is a different
            # experiment from the one being run.
            kept += 1
            continue
        anchor[index] = original.detach().clone().to(parameter.device)
        replaced += 1
    del pretrained
    logger.info(
        "RecAdam anchor: pretrained | Tensors from original weights: %d | Left on Phase 1: %d",
        replaced, kept,
    )
    return anchor


def interpolate_backbone(model, args, device):
    """Blend the Phase-1 backbone back toward its original pretrained weights.

    Phase 1 helps a weak encoder and damages a strong one, so rather than
    choosing between running it and not, dial it: alpha=1 keeps Phase 1 whole,
    alpha=0 restores the untouched pretrained weights, and values between trade
    source knowledge against the pretrained features Phase 1 would overwrite.

    This is the weight-ensembling idea from WiSE-FT (Wortsman et al., 2021),
    applied to the source stage rather than to a zero-shot model, and it costs
    one extra model load rather than another training run.
    """
    alpha = args.source_interpolation
    pretrained = build_backbone(args.model_name).to(device)
    source_state = model.backbone.state_dict()
    pretrained_state = pretrained.state_dict()
    blended, skipped = 0, 0
    for name, tensor in source_state.items():
        original = pretrained_state.get(name)
        if original is None or original.shape != tensor.shape:
            skipped += 1
            continue
        if not tensor.is_floating_point():
            skipped += 1
            continue
        tensor.mul_(alpha).add_(original.to(tensor.device), alpha=1.0 - alpha)
        blended += 1
    del pretrained
    logger.info(
        "Backbone interpolated toward pretrained | alpha: %.2f | Tensors blended: %d | Skipped: %d",
        alpha, blended, skipped,
    )


def run_linear_probe(args, model, train_records, val_records, tokenizer, device):
    """Fit vul_head with the backbone frozen, before any full fine-tuning.

    Kumar et al. (ICLR 2022) show full fine-tuning distorts pretrained features
    when they are already good, which is exactly the pattern measured here: the
    transfer helps CodeBERT and hurts the stronger CodeT5+. Fitting the head
    first means the initial full-model gradients are no longer dominated by a
    randomly-initialised head pulling the backbone apart.

    The RecAdam anchor is cloned after this runs, so the probed head is part of
    the source solution the optimizer holds on to.
    """
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    head_params = [p for p in model.vul_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        head_params, lr=args.lp_learning_rate, weight_decay=args.weight_decay
    )
    loader = build_dataloader(
        train_records, tokenizer, args.max_length, args.batch_size, True, args.seed,
        args.num_workers, args.truncation_strategy
    )
    logger.info(
        "Linear probe | Epochs: %d | LR: %.2e | Trainable: %d parameters",
        args.lp_epochs,
        args.lp_learning_rate,
        sum(p.numel() for p in head_params),
    )
    model.train()
    for epoch in range(1, args.lp_epochs + 1):
        total, seen = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                return_cwe=False,
            )
            labels = batch["labels"].to(device)
            loss = torch.nn.functional.cross_entropy(outputs["vul_logits"], labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.size(0)
            seen += labels.size(0)
        logger.info("Linear probe | Epoch %d/%d | Loss: %.6f", epoch, args.lp_epochs, total / seen)

    for parameter in model.backbone.parameters():
        parameter.requires_grad = True
    freeze_aux_head(model)


def run_phase2(args, device):
    train_path, val_path, _ = python_paths(args)
    logger.info("Loading Python training data: %s", train_path)
    logger.info("Loading Python validation data: %s", val_path)
    train_records = limit_records(load_jsonl(train_path, args.target_lang), args.max_train_samples, args.seed)
    val_records = limit_records(load_jsonl(val_path, args.target_lang), args.max_eval_samples, args.seed)
    print_dataset_stats("python_train", train_records)
    print_dataset_stats("python_val", val_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # The auxiliary head's width is a property of the Phase-1 data, not of this
    # run's flags, so read it off the checkpoint before building the model.
    # Rebuilding with the four-way default makes load_state_dict fail on a
    # source trained with a larger CWE vocabulary.
    args.num_cwes = adopt_checkpoint_shape(args.source_checkpoint, device)
    model = make_model(args.model_name, device, args)
    source_checkpoint = load_checkpoint(args.source_checkpoint, model, device)
    if args.source_interpolation < 1.0:
        interpolate_backbone(model, args, device)
    assert_checkpoint_compatible(source_checkpoint, args, "source checkpoint")
    logger.info(
        "Loaded source checkpoint: %s | Best epoch: %s | Best Val Macro-F1: %.6f",
        args.source_checkpoint,
        source_checkpoint["best_epoch"],
        source_checkpoint["best_val_macro_f1"],
    )
    freeze_aux_head(model)

    if args.lp_epochs > 0:
        run_linear_probe(args, model, train_records, val_records, tokenizer, device)

    # Take names alongside the tensors: the anchor may need to swap individual
    # backbone entries for their pretrained originals, which needs the name.
    # named_parameters() and parameters() iterate in the same order, so the
    # index alignment RecAdam relies on is preserved.
    named_trainable = [(name, parameter) for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    current_params = [parameter for _, parameter in named_trainable]
    # This is the immutable RecAdam anchor, cloned before any Python optimizer step.
    pretrain_params = build_recadam_anchor(named_trainable, args, device)
    assert_recadam_setup(current_params, pretrain_params, model)

    train_loader = build_dataloader(
        train_records, tokenizer, args.max_length, args.batch_size, True, args.seed, args.num_workers,
        args.truncation_strategy
    )
    val_loader = build_dataloader(
        val_records, tokenizer, args.max_length, args.eval_batch_size, False, args.seed, args.num_workers,
        args.truncation_strategy
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    anneal_t0 = max(1, int(args.anneal_t0_ratio * total_steps))

    if args.phase2_optimizer == "adamw":
        # RecAdam scales the target gradient by lambda(t), which is calibrated to
        # total_steps. With a small target set there are too few steps for lambda
        # to rise, so the model stays anchored at the source solution -- visible
        # as test probabilities collapsing toward 0.5. Plain AdamW isolates that.
        optimizer = torch.optim.AdamW(
            current_params, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        logger.info("Phase 2 optimizer: AdamW (RecAdam anchoring disabled)")
        log_environment(args, model, device, "phase2_train")
        train_loop(
            args, model, train_loader, val_loader, optimizer, device, "phase2",
            save_checkpoint, pretrain_params,
        )
        return

    optimizer = RecAdam(
        current_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        anneal_fun=args.anneal_fun,
        anneal_k=args.anneal_k,
        anneal_t0=anneal_t0,
        anneal_w=args.anneal_w,
        pretrain_cof=args.pretrain_cof,
        pretrain_params=pretrain_params,
    )
    print_recadam_schedule(args, total_steps, anneal_t0, len(train_loader))
    log_environment(args, model, device, "phase2_train")
    train_loop(
        args,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        "phase2",
        save_checkpoint,
        pretrain_params,
    )


def _collect_runtime(args, device, n_test, validation_seconds, test_seconds):
    """Gom gio cua CA HAI pha vao ket qua, vi mot o Pha 2 khong the tach khoi Pha 1
    da sinh ra checkpoint cho no. Doc tu file phu canh checkpoint; thieu thi de None
    chu khong dung lan chay."""
    from runtime_env import hardware_fingerprint, read_runtime_sidecar
    out = {
        "phase2": read_runtime_sidecar(args.checkpoint_path),
        "phase1": read_runtime_sidecar(args.source_checkpoint) if args.source_checkpoint else None,
        "inference": {
            "samples_test": n_test,
            "validation_and_threshold_seconds": round(float(validation_seconds), 3),
            "test_seconds": round(float(test_seconds), 3),
            "ms_per_test_sample": round(1000.0 * float(test_seconds) / n_test, 3) if n_test else None,
        },
        "hardware_at_inference": hardware_fingerprint(device),
    }
    return out


def run_test(args, device):
    _, val_path, test_path = python_paths(args)
    logger.info("Loading Python validation data: %s", val_path)
    logger.info("Loading Python test data: %s", test_path)
    val_records = limit_records(load_jsonl(val_path, args.target_lang), args.max_eval_samples, args.seed)
    test_records = limit_records(load_jsonl(test_path, args.target_lang), args.max_eval_samples, args.seed)
    print_dataset_stats("python_val", val_records)
    print_dataset_stats("python_test", test_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.num_cwes = adopt_checkpoint_shape(args.checkpoint_path, device)
    model = make_model(args.model_name, device, args)
    checkpoint = load_checkpoint(args.checkpoint_path, model, device)
    assert_checkpoint_compatible(checkpoint, args, "target checkpoint")
    freeze_aux_head(model)
    log_environment(args, model, device, "test_inference_no_optimizer")
    val_loader = build_dataloader(
        val_records, tokenizer, args.max_length, args.eval_batch_size, False, args.seed, args.num_workers,
        args.truncation_strategy
    )
    test_loader = build_dataloader(
        test_records, tokenizer, args.max_length, args.eval_batch_size, False, args.seed, args.num_workers,
        args.truncation_strategy
    )
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
    # Exactly one test inference pass; both reports reuse these fixed predictions.
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
        "source_checkpoint": args.source_checkpoint or checkpoint["training_args"].get("source_checkpoint"),
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
        "runtime": _collect_runtime(args, device, len(test["labels"]),
                                    validation_seconds, test_seconds),
    }
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    logger.info("Test result saved: %s", result_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic source multitask pretraining -> Python RecAdam transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    execution = parser.add_argument_group("execution")
    execution.add_argument("--phase", choices=("phase1", "phase2", "test"), required=True,
                           help="pipeline phase to execute")
    execution.add_argument("--seed", type=int, default=42, help="random seed")
    execution.add_argument("--strict_determinism", action="store_true",
                           help="demand deterministic kernels as well as seeding; the seed "
                                "alone does not pin Phase 1 (see set_seed)")
    execution.add_argument("--fold", type=int, default=1, choices=range(1, 6), help="Python fold")
    execution.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    execution.add_argument("--run_name", default="default_run", help="parent experiment folder")
    execution.add_argument("--method_name", default="transfer", help="method folder under run_name")

    paths = parser.add_argument_group("data and output paths")
    paths.add_argument("--data_path", default="data/train_ccpp_filtered.jsonl", help="Phase-1 JSONL")
    paths.add_argument("--target_lang", default="python",
                       help="language every target row must declare; guards against pointing "
                            "--data_root at folds from a different corpus by mistake")
    paths.add_argument("--data_root", default="data/sven_python_folds_norm",
                       help="directory holding fold1..fold5; point at the pair-preserving "
                            "folds to evaluate without near-duplicate leakage")
    paths.add_argument("--train_path", help="optional Python train JSONL override")
    paths.add_argument("--val_path", help="optional validation JSONL override")
    paths.add_argument("--test_path", help="optional test JSONL override")
    paths.add_argument("--source_checkpoint", help="best Phase-1 checkpoint")
    paths.add_argument("--checkpoint_path", help="checkpoint to write or read")
    paths.add_argument("--output_dir", help="result root; derived from run_name when omitted")
    paths.add_argument("--result_path", help="test JSON output path")

    training = parser.add_argument_group("model and training")
    training.add_argument("--model_name", default="microsoft/codebert-base", help="Hugging Face model")
    training.add_argument("--epochs", type=int, default=10, help="maximum epochs")
    training.add_argument("--min_epochs", type=int, default=3,
                          help="do not count early-stopping patience before this epoch")
    training.add_argument("--batch_size", type=int, default=8, help="training batch size")
    training.add_argument("--eval_batch_size", type=int, default=16, help="evaluation batch size")
    training.add_argument("--max_length", type=int, default=512, help="token sequence length")
    training.add_argument(
        "--truncation_strategy",
        choices=("head", "head_middle_tail"),
        default="head_middle_tail",
        help="how to retain tokens from code longer than max_length",
    )
    training.add_argument("--learning_rate", type=float, default=2e-5, help="optimizer learning rate")
    training.add_argument("--weight_decay", type=float, default=0.01, help="decoupled weight decay")
    training.add_argument("--lambda_cwe", type=float, default=0.2, help="Phase-1 auxiliary loss weight")
    training.add_argument(
        "--aux_mode",
        choices=("cwe", "latent_bottleneck", "latent_proto", "none"),
        default="cwe",
        help="Phase-1 auxiliary task: explicit CWE head, latent bottleneck, "
             "label-free latent prototypes, or none for the lambda=0 ablation",
    )
    training.add_argument("--pooling", choices=("cls", "mean"), default="cls",
                          help="sentence representation; use mean for T5-family encoders "
                               "which have no CLS token")
    training.add_argument("--num_latent", type=int, default=8,
                          help="latent units or prototypes, held fixed across source configs")
    training.add_argument("--latent_temperature", type=float, default=0.1,
                          help="prototype assignment temperature for aux_mode=latent_proto")
    training.add_argument("--lora_rank", type=int, default=0,
                          help="Phase-1 LoRA rank; the backbone is frozen and only a rank-r "
                               "update trains, then it is merged before the checkpoint is "
                               "written so Phase 2 sees an ordinary backbone")
    training.add_argument("--source_interpolation", type=float, default=1.0,
                          help="blend the Phase-1 backbone toward its original pretrained "
                               "weights before Phase 2: 1.0 keeps Phase 1 whole, 0.0 "
                               "restores the untouched checkpoint")
    training.add_argument("--lp_epochs", type=int, default=0,
                          help="Phase-2 linear-probe epochs before unfreezing the backbone "
                               "(LP-FT). Guards a strong backbone against distortion by a "
                               "freshly initialised head")
    training.add_argument("--lp_learning_rate", type=float, default=1e-3,
                          help="learning rate for the linear-probe stage; the head alone "
                               "tolerates a far larger step than the backbone")
    training.add_argument("--cwe_vocab", choices=("fixed4", "source", "precomputed"), default="fixed4",
                          help="fixed4 keeps the original four-way head; source builds the "
                               "auxiliary label space from whatever CWEs the Phase-1 data "
                               "contains, so a 121-CWE corpus is usable in full")
    training.add_argument("--freeze_backbone_layers", type=int, default=0,
                          help="freeze embeddings and the lowest N encoder layers during "
                               "Phase 1, bounding how far source pretraining can move a "
                               "backbone that is already strong")
    training.add_argument("--freeze_prototypes_steps", type=int, default=0,
                          help="hold prototypes fixed for this many steps so the projection "
                               "settles before assignment targets move")
    training.add_argument("--selection_metric", choices=("macro_f1", "pr_auc", "roc_auc"),
                          default="macro_f1",
                          help="validation metric for best-checkpoint and early stopping; "
                               "pr_auc is threshold-free and catches a model that only wins "
                               "at 0.5")
    training.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    training.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clipping norm")

    recadam = parser.add_argument_group("RecAdam phase 2")
    recadam.add_argument("--phase2_optimizer", choices=("recadam", "adamw"), default="recadam",
                         help="adamw drops the source anchor entirely, isolating what RecAdam "
                              "contributes and testing whether its step-count-calibrated "
                              "annealing is what fails on a small target set")
    recadam.add_argument("--anneal_fun", choices=("sigmoid", "linear", "constant"), default="sigmoid",
                         help="RecAdam annealing curve")
    recadam.add_argument("--anneal_k", type=float, default=0.05, help="sigmoid steepness")
    recadam.add_argument("--anneal_t0_ratio", type=float, default=0.05,
                         help="anneal midpoint as a fraction of total Phase-2 steps")
    recadam.add_argument("--anneal_w", type=float, default=1.0, help="maximum target-task weight")
    recadam.add_argument("--aux_weight_mode", choices=("fixed", "uncertainty"), default="fixed",
                         help="fixed = dung hang so --lambda_cwe (mac dinh, duong chay khong doi). "
                              "uncertainty = hoc trong so tung task theo Kendall/Gal/Cipolla "
                              "(CVPR 2018, arXiv:1705.07115): hai vo huong log-phuong sai hoc "
                              "cung luc voi trong so. Khoi tao tai dung --lambda_cwe nen day la "
                              "mo rong that su cua baseline. Chi co tac dung o Phase 1")
    recadam.add_argument("--aux_weight_lr", type=float, default=1e-2,
                         help="learning rate RIENG cho hai vo huong log-phuong sai khi "
                              "--aux_weight_mode uncertainty. Phai lon hon lr cua backbone: voi "
                              "lr chung 2e-5 thi qua ca mot Phase 1 chung chi dich duoc ~0.024, "
                              "tuc lambda gan nhu khong hoc gi")
    recadam.add_argument("--sam_rho", type=float, default=0.0,
                         help="Sharpness-Aware Minimization o CHINH pha dang chay (Phase 1 hay "
                              "Phase 2 la do --phase quyet dinh, khong phai co nay). 0 = tat (mac dinh, "
                              "duong chay khong doi). >0 bat, moi buoc 2 luot forward-backward "
                              "nen ~2x thoi gian. rho la do dai TUYET DOI cua nhieu loan, "
                              "chuan L2 toan cuc, dung quy uoc bai bao (0.05, 0.1 la pho bien)")
    recadam.add_argument("--sam_variant", choices=("sam", "asam"), default="sam",
                         help="sam = Foret et al. ICLR 2021, ban kinh TUYET DOI, chuan L2 toan cuc. "
                              "asam = Kwon et al. ICML 2021 (arXiv:2102.11600), nhieu loan chuan hoa "
                              "theo |w| nen BAT BIEN VOI THANG DO trong so. rho cua hai bien the "
                              "KHONG cung thang do: bai bao ASAM chon 0.5-1.0 trong khi SAM dung "
                              "0.05-0.1; thi nghiem transformer duy nhat cua ho dung 0.2 cho ASAM "
                              "va 0.1 cho SAM. Bung 0.05 vao ASAM la gan nhu khong lam gi")
    recadam.add_argument("--asam_eta", type=float, default=0.01,
                         help="on dinh so cho ASAM: T_w = |w| + eta. 0.01 la so cua bai bao")
    recadam.add_argument("--recadam_anchor", choices=("source", "pretrained"), default="source",
                         help="what RecAdam pulls back toward: the Phase-1 weights (default) "
                              "or the untouched pretrained weights. Initialisation is the "
                              "Phase-1 checkpoint either way")
    recadam.add_argument("--pretrain_cof", type=float, default=5000.0,
                         help="quadratic source-anchor coefficient")

    smoke = parser.add_argument_group("smoke-test limits")
    smoke.add_argument("--max_train_samples", type=int, help="cap training records")
    smoke.add_argument("--max_eval_samples", type=int, help="cap validation/test records")
    args = parser.parse_args()
    if args.epochs < 1 or args.min_epochs < 1 or args.patience < 1:
        parser.error("epochs, min_epochs, and patience must be positive")
    if not 0.0 <= args.anneal_t0_ratio <= 1.0:
        parser.error("--anneal_t0_ratio must be in [0, 1]")

    if args.output_dir is None:
        args.output_dir = f"results/{args.run_name}/{args.method_name}"
    model_root = Path("model") / args.run_name / args.method_name / f"seed_{args.seed}"
    if args.checkpoint_path is None:
        if args.phase == "phase1":
            args.checkpoint_path = str(model_root / "source" / "best.pt")
        else:
            args.checkpoint_path = str(model_root / f"fold{args.fold}" / "best.pt")
    if args.result_path is None:
        args.result_path = str(Path(args.output_dir) / f"seed_{args.seed}" / f"fold{args.fold}.json")
    if args.phase == "phase2" and args.source_checkpoint is None:
        args.source_checkpoint = str(model_root / "source" / "best.pt")
    if args.phase == "phase2" and not Path(args.source_checkpoint).is_file():
        parser.error(f"source checkpoint does not exist: {args.source_checkpoint}")
    if args.phase == "test" and not Path(args.checkpoint_path).is_file():
        parser.error(f"target checkpoint does not exist: {args.checkpoint_path}")
    return args


def main():
    args = parse_args()
    configure_logging()
    set_seed(args.seed, strict=args.strict_determinism)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    logger.info("Using device: %s", device)
    logger.info(
        "Run started | Run: %s | Method: %s | Phase: %s | Fold: %d | Seed: %d | Device: %s",
        args.run_name,
        args.method_name,
        args.phase,
        args.fold,
        args.seed,
        device,
    )
    try:
        if args.phase == "phase1":
            run_phase1(args, device)
        elif args.phase == "phase2":
            run_phase2(args, device)
        else:
            run_test(args, device)
    finally:
        logger.info(
            "Run finished | Phase: %s | Fold: %d | Seed: %d | Elapsed: %.2fs",
            args.phase,
            args.fold,
            args.seed,
            time.perf_counter() - started,
        )


if __name__ == "__main__":
    main()
