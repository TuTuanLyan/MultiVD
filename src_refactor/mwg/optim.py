"""Bộ tối ưu cho pha chuyển giao: RecAdam (chống quên) và SAM / ASAM (tìm cực tiểu phẳng).

RecAdam — Chen et al., EMNLP 2020 (https://github.com/Sanyuan-Chen/RecAdam), Apache-2.0.
SAM     — Foret et al., ICLR 2021 (port từ https://github.com/google-research/sam).
ASAM    — Kwon et al., ICML 2021 (https://github.com/SamsungLabs/ASAM).
"""
import math

import numpy as np
import torch
from torch.optim import Optimizer


# ----------------------------------------------------------------------------- RecAdam
def anneal_function(function, step, k, t0, weight):
    if function == "sigmoid":
        return float(1 / (1 + np.exp(-k * (step - t0)))) * weight
    if function == "linear":
        return min(1, step / t0) * weight
    if function == "constant":
        return weight
    raise ValueError("Unsupported anneal function: {}".format(function))


class RecAdam(Optimizer):
    """Adam + phạt bậc hai kéo trọng số về điểm neo `pretrain_params`, trọng số lambda(t) ủ theo `anneal_fun`:

        Loss = lambda(t) * Loss_target + (1 - lambda(t)) * pretrain_cof / 2 * sum((theta - theta_anchor)^2)

    Mỗi param group cần khoá `pretrain_params` (danh sách tensor neo cùng thứ tự với `params`).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True,
                 anneal_fun="sigmoid", anneal_k=0, anneal_t0=0, anneal_w=1.0, pretrain_cof=5000.0, pretrain_params=None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias,
                        anneal_fun=anneal_fun, anneal_k=anneal_k, anneal_t0=anneal_t0, anneal_w=anneal_w,
                        pretrain_cof=pretrain_cof, pretrain_params=pretrain_params)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p, pp in zip(group["params"], group["pretrain_params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
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
                if group["anneal_w"] > 0.0:
                    anneal_lambda = anneal_function(group["anneal_fun"], state["step"], group["anneal_k"],
                                                    group["anneal_t0"], group["anneal_w"])
                    group["last_anneal_lambda"] = anneal_lambda
                    assert anneal_lambda <= group["anneal_w"]
                    p.data.addcdiv_(exp_avg, denom, value=-step_size * anneal_lambda)       # lambda(t) * Loss_target
                    p.data.add_(p.data - pp.data,                                          # phạt bậc hai về điểm neo
                                alpha=-group["lr"] * (group["anneal_w"] - anneal_lambda) * group["pretrain_cof"])
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                if group["weight_decay"] > 0.0:                                            # weight decay tách rời
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
        return loss


# ----------------------------------------------------------------------------- SAM / ASAM
@torch.no_grad()
def _grad_global_norm(params):
    total = None
    for p in params:
        if p.grad is None:
            continue
        s = (p.grad.detach() ** 2).sum()
        total = s if total is None else total + s
    return None if total is None else torch.sqrt(total)


class SAMStep:
    """Bước leo / khôi phục của SAM quanh một lượt backward có sẵn:

        loss.backward(); sam.ascend(params, names)      # tới w + eps
        loss2 = <forward lần 2>; loss2.backward()       # gradient TẠI w + eps
        sam.restore(params); optimizer.step()           # về w gốc rồi mới bước

    variant="sam":  eps = rho * g / ||g||, chuẩn L2 TOÀN CỤC qua mọi tensor.
    variant="asam": eps = rho * T^2 * g / ||T * g||, T = |w| + eta cho tham số có 'weight' trong tên, T = 1 cho
                    các tham số khác (như bản chính thức) — rho bất biến theo thang trọng số.
    Ghép được với bất kỳ optimizer nào (RecAdam hoặc AdamW) vì mọi thứ xảy ra trước optimizer.step().
    """

    def __init__(self, rho, variant="sam", eta=0.01):
        if rho <= 0:
            raise ValueError("rho phải dương")
        if variant not in ("sam", "asam"):
            raise ValueError(f"variant phải là sam hoặc asam, nhận {variant!r}")
        self.rho, self.variant, self.eta = rho, variant, eta
        self._eps = []

    @torch.no_grad()
    def ascend(self, params, names=None):
        """Đi tới w + eps. Trả về False (bỏ qua bước) nếu gradient bằng 0."""
        if self.variant == "asam":
            return self._ascend_asam(params, names)
        self._eps = []
        norm = _grad_global_norm(params)
        if norm is None or not torch.isfinite(norm) or norm.item() == 0.0:
            return False
        scale = self.rho / norm
        for p in params:
            if p.grad is None:
                self._eps.append(None)
                continue
            e = p.grad.detach() * scale
            p.add_(e)
            self._eps.append(e)
        return True

    @torch.no_grad()
    def _ascend_asam(self, params, names):
        if names is None:
            raise ValueError("ASAM cần `names` để biết tham số nào là 'weight'")
        self._eps = []
        tws = []
        for p, n in zip(params, names):
            if p.grad is None:
                tws.append(None)
                continue
            tws.append(p.detach().abs().add(self.eta) if "weight" in n else None)
        total = None
        for p, tw in zip(params, tws):
            if p.grad is None:
                continue
            g = p.grad.detach() if tw is None else p.grad.detach() * tw
            sq = (g ** 2).sum()
            total = sq if total is None else total + sq
        if total is None:
            return False
        norm = torch.sqrt(total)
        if not torch.isfinite(norm) or norm.item() == 0.0:
            return False
        scale = self.rho / norm
        for p, tw in zip(params, tws):
            if p.grad is None:
                self._eps.append(None)
                continue
            g = p.grad.detach()
            e = (g if tw is None else g * tw * tw) * scale
            p.add_(e)
            self._eps.append(e)
        return True

    @torch.no_grad()
    def restore(self, params):
        """Trừ đúng eps đã cộng, đưa trọng số về w trước khi optimizer bước."""
        for p, e in zip(params, self._eps):
            if e is not None:
                p.sub_(e)
        self._eps = []
