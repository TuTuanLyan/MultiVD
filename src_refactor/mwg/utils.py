"""Seed, logging, đọc dữ liệu jsonl."""
import json
import logging
import random
import sys

import numpy as np
import torch

LOGGER_NAME = "mwg"


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


def set_seed(seed):
    """Seed mọi bộ sinh số ngẫu nhiên. KHÔNG bật thuật toán tất định của cuBLAS: cùng seed hai lần chạy vẫn có thể
    lệch nhau (early stopping khuếch đại sai số làm tròn), nên kết quả phải đọc trên nhiều fold / nhiều seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(path, expected_lang=None):
    """Mỗi dòng một hàm: cần `code` (chuỗi khác rỗng), `label` (0/1), `lang` (hoặc `language`)."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: JSON hỏng: {error}") from error
            if not isinstance(record.get("code"), str) or not record["code"].strip():
                raise ValueError(f"{path}:{line_number}: thiếu hoặc rỗng trường 'code'")
            try:
                record["label"] = int(record["label"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: 'label' phải là 0 hoặc 1") from error
            if record["label"] not in (0, 1):
                raise ValueError(f"{path}:{line_number}: 'label' phải là 0 hoặc 1, nhận {record['label']}")
            lang, legacy = record.get("lang"), record.get("language")
            if lang is not None and legacy is not None and lang != legacy:
                raise ValueError(f"{path}:{line_number}: 'lang'={lang!r} mâu thuẫn 'language'={legacy!r}")
            lang = lang if lang is not None else legacy
            if not isinstance(lang, str) or not lang.strip():
                raise ValueError(f"{path}:{line_number}: thiếu trường 'lang'")
            record["lang"] = lang
            if expected_lang is not None and lang != expected_lang:
                raise ValueError(f"{path}:{line_number}: cần lang {expected_lang!r}, nhận {lang!r}")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: không có bản ghi nào")
    return records


def limit_records(records, maximum, seed):
    """Lấy ngẫu nhiên tối đa `maximum` bản ghi, giữ thứ tự gốc (để chạy thử nhanh)."""
    if maximum is None or maximum <= 0 or len(records) <= maximum:
        return records
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(records), size=maximum, replace=False).tolist())
    return [records[i] for i in indices]


def train_loss_stalled(losses, min_drop):
    """Pha 1 kẹt ở nghiệm tầm thường: train loss epoch cuối giảm chưa tới `min_drop` (tương đối) so với epoch trước.
    Chỉ đo độ giảm giữa hai epoch, không so với một mức loss cố định (mức đó đổi theo pool và pair loss).
    Trả về (kẹt?, độ giảm tương đối)."""
    prev, last = losses[-2], losses[-1]
    drop = (prev - last) / prev
    return drop < min_drop, drop
