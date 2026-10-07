"""AdamW voi learning rate nhan lambda(t) cua RecAdam — doi chung 'warmup ngam'.

RecAdam (src/RecAdam.py, dong `p.data.addcdiv_(exp_avg, denom, value=-step_size *
anneal_lambda)`) nhan buoc Adam DA CHUAN HOA voi lambda(t), roi cong them luc keo ve
neo `-lr*(w-lambda)*gamma*(theta-theta*)`. Nhan buoc Adam voi lambda(t) tuong duong
nhan learning rate voi lambda(t). Lop nay giu DUNG phan thu nhat va bo phan thu hai.

Vi sao can no: RecAdam o du an nay chi neo ~43 buoc tren 870 (RESEARCH_2026-09-06),
nen phan lon tac dung cua no co the chi la mot lich warmup. Neu nhanh nay bang
RecAdam thi phan neo hien tai khong mua gi; neu kem hon thi neo co tac dung du ngan.
Khong co doi chung nay thi 'RecAdam thang/thua AdamW' khong tach duoc hai co che.

Weight decay: RecAdam ap wd voi lr GOC, khong nhan lambda
(`p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])`). AdamW cua torch
ap wd = lr * weight_decay. De tich lr*wd giu nguyen bang lr_goc*wd_goc nhu RecAdam,
o day nhan lr voi lambda va chia weight_decay cho lambda. lambda >= 0.109 ngay buoc 1
voi k=0.05, t0=43 nen phep chia an toan; van kep duoi 1e-8 cho chac.

Khac biet con lai so voi RecAdam-khong-neo: eps (torch AdamW 1e-8, RecAdam 1e-6) —
giong het nhanh `--phase2_optimizer adamw` thuan dang dung lam doi chung, nen hai
nhanh AdamW so duoc voi nhau tung byte ngoai lich lr.
"""

import torch

from RecAdam import anneal_function


class AnnealedAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, anneal_fun="sigmoid",
                 anneal_k=0.05, anneal_t0=1, anneal_w=1.0, **kwargs):
        super().__init__(params, lr=lr, weight_decay=weight_decay, **kwargs)
        self._anneal = (anneal_fun, float(anneal_k), int(anneal_t0), float(anneal_w))
        self._step_count = 0
        for group in self.param_groups:
            group["base_lr"] = group["lr"]
            group["base_weight_decay"] = group["weight_decay"]

    def step(self, closure=None):
        self._step_count += 1
        fun, k, t0, w = self._anneal
        lam = max(float(anneal_function(fun, self._step_count, k, t0, w)), 1e-8)
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * lam
            group["weight_decay"] = group["base_weight_decay"] / lam
            group["last_anneal_lambda"] = lam
        return super().step(closure)
