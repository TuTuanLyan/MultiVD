import json
import logging
import os
import random
import sys

import numpy as np
import torch

LOGGER_NAME = "guidevul"


def configure_logging():
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def get_logger():
    return logging.getLogger(LOGGER_NAME)


def set_seed(seed, deterministic=False, tf32=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(path, expected_lang=None):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record.get("code"), str) or not record["code"].strip():
                raise ValueError(f"{path}:{line_number}: missing or empty field 'code'")
            try:
                record["label"] = int(record["label"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: 'label' must be 0 or 1") from error
            if record["label"] not in (0, 1):
                raise ValueError(f"{path}:{line_number}: 'label' must be 0 or 1, got {record['label']}")
            lang, legacy = record.get("lang"), record.get("language")
            if lang is not None and legacy is not None and lang != legacy:
                raise ValueError(f"{path}:{line_number}: 'lang'={lang!r} conflicts with 'language'={legacy!r}")
            lang = lang if lang is not None else legacy
            if not isinstance(lang, str) or not lang.strip():
                raise ValueError(f"{path}:{line_number}: missing field 'lang'")
            record["lang"] = lang
            if expected_lang is not None and lang != expected_lang:
                raise ValueError(f"{path}:{line_number}: expected lang {expected_lang!r}, got {lang!r}")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def limit_records(records, maximum, seed):
    if maximum is None or maximum <= 0 or len(records) <= maximum:
        return records
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(records), size=maximum, replace=False).tolist())
    return [records[i] for i in indices]


def train_loss_stalled(losses, min_drop):
    prev, last = losses[-2], losses[-1]
    drop = (prev - last) / prev
    return drop < min_drop, drop
