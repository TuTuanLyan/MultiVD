"""RecAdam voi luc keo neo PHAN BO theo Fisher — khong sua src/RecAdam.py.

`src/RecAdam.py` la ban tham chieu tu repo chinh chu va README noi ro khong sua no khi
mo rong phuong phap. Lop duoi day chep dung phan toan cua no va doi **mot cho duy nhat**:

    RecAdam :  theta -= lr * (w - lambda) * gamma        * (theta - theta*)
    o day   :  theta -= lr * (w - lambda) * gamma * F_i  * (theta - theta*)

`F_i` la Fisher cheo da chuan hoa ve trung binh 1 (src/fisher.py), nen **tong luc keo
khong doi** o cung gamma — chi doi cach PHAN BO. Nho vay phep so RecAdam vs RecAdam-Fisher
doi dung mot bien.

Vi sao khong dung param_groups voi pretrain_cof rieng cho tung tensor: `alpha=` cua
`Tensor.add_` chi nhan so vo huong, nen he so theo TUNG PHAN TU khong the di qua duong do.
Theo tensor thi duoc, nhung do la Fisher THEO LOP — tho hon han, va chinh cai ta muon do
la su khac biet BEN TRONG mot lop.

Kiem tra tinh dung: dat F = 1 cho moi tham so thi lop nay phai cho ra quy dao TRUNG KHIT
RecAdam goc. `tests/test_recadam_fisher.py` kiem dieu do bang mot bai toan hai chieu.
"""

import math

import numpy as np
import torch
from torch.optim import Optimizer

from RecAdam import anneal_function


class RecAdamFisher(Optimizer):
    """Xem docstring module. Tham so giong RecAdam, them `fisher_params`.

    fisher_params: list tensor cung thu tu va shape voi `params`. `None` cho mot tensor
    nghia la he so 1 (khong doi xu dac biet) — de tien khi vai tensor khong co Fisher.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0,
                 correct_bias=True, anneal_fun='sigmoid', anneal_k=0, anneal_t0=0,
                 anneal_w=1.0, pretrain_cof=5000.0, pretrain_params=None, fisher_params=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias, anneal_fun=anneal_fun, anneal_k=anneal_k,
                        anneal_t0=anneal_t0, anneal_w=anneal_w, pretrain_cof=pretrain_cof,
                        pretrain_params=pretrain_params, fisher_params=fisher_params)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            fishers = group.get("fisher_params") or [None] * len(group["params"])
            for p, pp, fi in zip(group["params"], group["pretrain_params"], fishers):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                if group['anneal_w'] > 0.0:
                    anneal_lambda = anneal_function(group['anneal_fun'], state["step"],
                                                    group['anneal_k'], group['anneal_t0'],
                                                    group['anneal_w'])
                    group["last_anneal_lambda"] = anneal_lambda
                    assert anneal_lambda <= group['anneal_w']
                    p.data.addcdiv_(exp_avg, denom, value=-step_size * anneal_lambda)
                    # >>> KHAC BIET DUY NHAT so voi RecAdam goc <<<
                    coef = group["lr"] * (group['anneal_w'] - anneal_lambda) * group["pretrain_cof"]
                    pull = p.data - pp.data
                    if fi is not None:
                        pull = pull * fi          # he so theo TUNG PHAN TU
                    p.data.add_(pull, alpha=-coef)
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
        return loss


def check_pull_stability(lr, pretrain_cof, fisher_max=1.0, anneal_lambda=0.0):
    """Luc keo neo cap nhat HIEN co nguong on dinh — kiem TRUOC khi chay, khong doi no no.

    Buoc keo la  theta <- theta - c*(theta - theta*)  voi  c = lr*(w-lambda)*gamma*F_i.
    Do la lap diem co dinh: |1 - c| < 1  <=>  0 < c < 2. c >= 2 thi dao dau va PHAN KY.
    c gan 1 thi nhay thang ve neo trong mot buoc — dung ve so hoc nhung xoa sach quy dao.

    Do duoc 06/09 khi viet tests/test_recadam_fisher.py: lr=1e-2, gamma=500, F=10 cho
    c = 50 va trong so no ra 9.7e31 chi sau 25 buoc. Voi tham so THAT cua du an
    (lr=2e-5, gamma=5000) thi lr*gamma = 0.1, nen F phai kep duoi ~10 moi an toan —
    do dung la ly do `--fisher_clip 10` cua src/fisher.py, va hai con so nay PHAI di cung nhau.

    Tra ve (on_dinh, thong_diep).
    """
    c = float(lr) * float(pretrain_cof) * float(fisher_max) * (1.0 - float(anneal_lambda))
    if c >= 2.0:
        return False, f"PHAN KY: c = {c:.3g} >= 2"
    if c >= 1.0:
        return False, f"NGUY HIEM: c = {c:.3g} >= 1 — mot buoc keo gan het ve neo"
    if c >= 0.5:
        return True, f"sat bien: c = {c:.3g}"
    return True, f"on dinh: c = {c:.3g}"


def load_fisher_for(named_trainable, fisher_path, device):
    """Doc sidecar Fisher va xep DUNG THU TU cua named_trainable.

    Tra ve (list tensor hoac None, so tensor khop). Tensor nao khong co trong sidecar
    thi de None = he so 1; xay ra voi head moi hoac khi doi kien truc, va phai bao ro
    chu khong im lang coi nhu 0.
    """
    payload = torch.load(fisher_path, map_location="cpu", weights_only=False)
    table = payload["fisher"]
    out, matched, missing = [], 0, []
    for name, p in named_trainable:
        f = table.get(name)
        if f is not None and f.shape == p.shape:
            out.append(f.to(device=p.device, dtype=p.dtype))
            matched += 1
        else:
            out.append(None)
            missing.append(name)
    return out, matched, missing, payload.get("stats", {})
