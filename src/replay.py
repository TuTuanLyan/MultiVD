"""replay.py — đưa dữ liệu NGUỒN vào chính Pha 2 (cầu CWE, RESEARCH_2026-09-06 §11.4 Đ5).

Vì sao có file này. Bốn khối OPT1 / RET1 / SPD1 / INT1 (RESEARCH_2026-09-06 §12.2) cùng
nói một điều: tri thức nguồn đi vào bài toán đích CHỈ qua điểm khởi tạo; mọi thao tác trên
không gian trọng số sau đó (RecAdam, SPD, Fisher, WiSE-FT) đều không thêm được gì, vì cái
neo chỉ là RÀNG BUỘC — nó không mang thêm một bit thông tin nào từ dữ liệu nguồn. Cách duy
nhất để gradient của dữ liệu nguồn nặn trực tiếp nghiệm đích là cho nó xuất hiện trong Pha 2.

Cơ chế:
  * mỗi bước Pha 2 lấy thêm MỘT batch nguồn, cộng μ(e)·L_nguồn vào mục tiêu;
  * μ(e) giảm tuyến tính từ μ₀ về 0 sau `epochs` epoch (μ=0 ⇒ Pha 2 thuần đích ở cuối,
    nên hàm quyết định cuối cùng vẫn là của đích; nguồn chỉ nặn ĐẶC TRƯNG ở giai đoạn đầu);
  * `stratify`: lấy mẫu cân bằng theo tầng (label, cwe_class) — nguồn 4cwe lệch 74% CWE-79,
    còn hai lớp đích được lợi nhất (022/079) lại là hai lớp NHỎ nhất của đích (FACTS §35).
    Cân tầng để mỗi CWE của đích nhận cùng lượng ví dụ nguồn.

Mặc định mọi cờ = tắt ⇒ `train_one_epoch_phase2` chạy đúng đường cũ từng byte.
"""
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import CodeDataset
from logging_utils import get_logger

logger = get_logger()


def replay_mu(epoch, mu0, epochs):
    """μ(e): hằng μ₀ nếu `epochs<=0`; ngược lại tuyến tính μ₀·(1 − (e−1)/epochs), kẹp ≥ 0.

    e=1 → μ₀ ; e=epochs+1 → 0. Ví dụ μ₀=0.5, epochs=6: 0.5 0.417 0.333 0.25 0.167 0.083 0 …
    """
    if mu0 <= 0:
        return 0.0
    if epochs <= 0:
        return float(mu0)
    return float(max(0.0, mu0 * (1.0 - (epoch - 1) / float(epochs))))


def stratum_weights(records, stratify):
    """Trọng số lấy mẫu từng dòng. `stratify` ⇒ mỗi tầng (label, cwe_class) có tổng xác suất
    bằng nhau; không ⇒ đều (mọi dòng 1.0). Trả (weights, counts)."""
    keys = [(int(r["label"]), int(r.get("cwe_class", -100))) for r in records]
    counts = Counter(keys)
    if not stratify:
        return [1.0] * len(records), counts
    return [1.0 / counts[k] for k in keys], counts


class SourceReplay:
    """Bộ phát batch nguồn vô hạn cho Pha 2. Gọi `next_batch()` mỗi bước; `mu(epoch)` cho hệ số."""

    def __init__(self, records, tokenizer, max_length, truncation_strategy, batch_size,
                 mu0, epochs, stratify, seed, lambda_cwe=0.0):
        if not records:
            raise ValueError("replay: khong co dong nguon nao")
        self.records = records
        self.mu0 = float(mu0)
        self.epochs = int(epochs)
        self.stratify = bool(stratify)
        self.lambda_cwe = float(lambda_cwe)
        self.batch_size = int(batch_size)
        weights, self.counts = stratum_weights(records, stratify)
        self._generator = torch.Generator()
        self._generator.manual_seed(int(seed) + 7919)   # lệch seed đích để hai luồng không trùng nhịp
        dataset = CodeDataset(records, tokenizer, max_length, truncation_strategy)
        # Một "epoch" của sampler dài bằng số dòng nguồn; hết thì tạo lại iterator (vô hạn).
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=max(len(records), self.batch_size),
            replacement=True,
            generator=self._generator,
        )
        self._loader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        self._iter = iter(self._loader)
        self.n_drawn = 0
        logger.info(
            "Replay nguon | %d dong | batch %d | mu0 %.3f | anneal %d epoch | stratify=%s | "
            "lambda_cwe_nguon %.3f | tang (label,cwe): %s",
            len(records), self.batch_size, self.mu0, self.epochs, self.stratify, self.lambda_cwe,
            dict(sorted(self.counts.items())),
        )

    def mu(self, epoch):
        return replay_mu(epoch, self.mu0, self.epochs)

    def next_batch(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            batch = next(self._iter)
        self.n_drawn += int(batch["labels"].numel())
        return batch

    def describe(self):
        """Chuỗi ngắn cho JSON/log: tầng và số dòng."""
        return " ".join(f"{l}/{c}:{n}" for (l, c), n in sorted(self.counts.items()))
