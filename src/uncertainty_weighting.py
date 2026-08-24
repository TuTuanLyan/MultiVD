#!/usr/bin/env python3
"""λ học được, theo Kendall, Gal, Cipolla (CVPR 2018, arXiv:1705.07115).

Thay hằng số `lambda_cwe` chọn tay bằng hai vô hướng học được. Lý do cụ thể từ
số liệu của dự án, không phải vì nó mới: đi từ λ=0.2 xuống λ=0.05 cải thiện
**8/9 ô** của họ backbone T5, biên độ tới +0.066 — gấp ba sd giữa seed. Tức giá
trị λ tối ưu KHÁC NHAU theo backbone, và một hằng số toàn cục không thể vừa cho
tất cả.

--------------------------------------------------------------------------
Công thức, trích đúng từ bài
--------------------------------------------------------------------------
Dạng HỒI QUY (công thức 7) được hệ số `1/(2σ²)`; dạng PHÂN LOẠI (công thức 10)
được `1/σ²`, KHÔNG có số 2. Cả hai loss ở đây đều là cross-entropy, nên dùng
dạng phân loại cho cả hai:

    L = (1/σ_bin²)·L_bin + (1/σ_aux²)·L_aux + log σ_bin + log σ_aux

Bài học `s := log σ²` chứ không học σ², nguyên văn: *"it is more numerically
stable than regressing the variance, σ², as the loss avoids any division by
zero"*. Với `s` thì `1/σ² = exp(−s)` và `log σ = s/2`, cho dạng cài đặt:

    L = exp(−s_bin)·L_bin + exp(−s_aux)·L_aux + s_bin/2 + s_aux/2

Bài cũng trả lời sẵn lo ngại "trọng số sẽ tụt về 0": *"This loss is smoothly
differentiable, and is well formed such that the task weights will not converge
to zero. In contrast, directly learning the weights using a simple linear sum of
losses would result in weights which quickly converge to zero."* Số hạng
`+ s/2` chính là thứ phạt việc đẩy σ lên cao để bỏ qua một task.

Công thức 10 còn kèm một xấp xỉ tường minh —
`(1/σ)·Σ exp(f_c/σ²) ≈ (Σ exp f_c)^(1/σ²)`, thành đẳng thức khi σ → 1 — mà bài
nói là vừa đơn giản hoá tối ưu vừa *"empirically improving results"*. Xấp xỉ đó
đã nằm sẵn trong dạng trên; ở đây không cài lại softmax có scale.

--------------------------------------------------------------------------
Suy diễn của dự án này, KHÔNG phải của bài
--------------------------------------------------------------------------
Chia cả hai vế cho `exp(−s_bin)` thì mục tiêu tỉ lệ thuận với
`L_bin + exp(s_bin − s_aux)·L_aux`, tức

    λ_eff = exp(s_bin − s_aux)

ánh xạ thẳng vào chính đại lượng `lambda_cwe` đang quét. Nhờ vậy con số học được
so trực tiếp được với λ=0.2 và λ=0.05 đã chạy, và ghi lại λ_eff theo epoch cho
biết mô hình *muốn* λ bằng bao nhiêu — đó mới là kết quả đáng đọc, chứ không phải
riêng điểm F1 cuối.

Giới hạn phải nói trước khi đọc kết quả: σ được học từ **loss huấn luyện Phase 1**,
nên nó cân bằng độ khó/độ nhiễu của hai task. Nó KHÔNG nhìn thấy đại lượng ta thật
sự cần — Macro-F1 trên target sau Phase 2. Muốn tối ưu đúng đại lượng đó thì phải
dùng bilevel (xem RESEARCH §4.4). Nên phép thử này trả lời câu "λ tự cân bằng có
đủ không", không phải câu "λ tối ưu cho transfer là bao nhiêu".
"""

import math

import torch
import torch.nn as nn


class UncertaintyWeights(nn.Module):
    """Hai log-phương sai học được cho loss nhị phân và loss phụ."""

    def __init__(self, init_lambda=0.2, learn_binary=True):
        super().__init__()
        if init_lambda <= 0:
            raise ValueError(f"init_lambda phai duong, nhan duoc {init_lambda}")
        # Khởi tạo sao cho λ_eff bắt đầu ĐÚNG BẰNG λ cố định đang dùng, để phép
        # thử là một mở rộng thật sự của baseline chứ không phải một điểm xuất
        # phát khác. Bắt đầu ở λ_eff=1 sẽ trộn "λ học được" với "λ khởi tạo khác".
        self.s_binary = nn.Parameter(torch.zeros(()), requires_grad=learn_binary)
        self.s_aux = nn.Parameter(torch.full((), -math.log(init_lambda)))

    def forward(self, binary_loss, aux_loss):
        """Trả về (loss tổng, dict chẩn đoán). aux_loss=None thì trả nguyên loss nhị phân."""
        if aux_loss is None:
            return binary_loss, {"lambda_eff": float("nan")}
        total = (
            torch.exp(-self.s_binary) * binary_loss
            + torch.exp(-self.s_aux) * aux_loss
            + 0.5 * self.s_binary
            + 0.5 * self.s_aux
        )
        return total, self.diagnostics()

    @torch.no_grad()
    def diagnostics(self):
        s_b = float(self.s_binary)
        s_a = float(self.s_aux)
        return {
            "s_binary": s_b,
            "s_aux": s_a,
            "w_binary": math.exp(-s_b),
            "w_aux": math.exp(-s_a),
            # Đại lượng so được với `lambda_cwe`; xem chú thích đầu file.
            "lambda_eff": math.exp(s_b - s_a),
        }

    def extra_repr(self):
        d = self.diagnostics()
        return f"s_binary={d['s_binary']:.4f}, s_aux={d['s_aux']:.4f}, lambda_eff={d['lambda_eff']:.4f}"
