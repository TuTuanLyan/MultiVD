"""Seed, logging, đọc dữ liệu jsonl."""
import json
import logging
import os
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


def set_seed(seed, deterministic=False, tf32=False):
    """Seed mọi bộ sinh số ngẫu nhiên.

    deterministic=True (04/10, khối MW-only n=5; lấy từ set_seed(strict=True) của mã latent bottleneck cũ và
    det_launch.py của khối final): bật thêm thuật toán tất định của PyTorch/cuBLAS để cùng seed, cùng máy cho ra
    cùng từng bit. Chỉ seed thôi là KHÔNG đủ: early stopping khuếch đại sai số làm tròn ~1e-7 của các phép rút gọn
    không tất định thành một quyết định dừng khác. Không dùng warn_only: gặp kernel không có bản tất định thì
    ném lỗi và dừng, thay vì chạy tiếp mà không tái lập được.
    tf32=True: bật TF32 cho matmul và cuDNN (nhanh ~x1,8 trên A4000); tất định vẫn giữ, nhưng số học đổi như đổi
    hạt giống nên chỉ so các lượt cùng bật hoặc cùng tắt."""
    if deterministic:
        # cuBLAS chỉ đọc biến này MỘT lần lúc tạo handle, nên phải đặt trước mọi phép tính CUDA
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
