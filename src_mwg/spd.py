"""Adam + Selective Projection Decay (SPD) — Tian, Huang, Kira, NeurIPS 2024, arXiv:2411.01713.

VI SAO CO FILE NAY (do duoc 07/09, RESEARCH §10): RecAdam quyet dinh neo **theo thoi gian** —
lambda(t) khong biet luc nay cai neo dang giup hay dang can. Do duoc tren 100 o ghep cap,
ba backbone, ba nguon: RecAdam - AdamW = -0.0041 (44/100). Quet gamma cung khong cuu
(nut phang trong dai 0.5..50). Nen thu doi CO CHE thay vi doi tham so.

SPD thay lich thoi gian bang mot DIEU KIEN theo gradient, tinh RIENG cho tung lop:

    c_t   = -g_t^T (theta_{t-1} - theta_0)                                   (Eq. 3)
    r_t   = max(0, gamma_t - gamma_{t-1}) / gamma_t                          (Eq. 9)
            voi gamma_t     = ||theta~_t   - theta_0||_2   (do lech SAU buoc Adam, TRUOC phat)
                gamma_{t-1} = ||theta_{t-1} - theta_0||_2   (do lech truoc buoc)
    neu c_t < 0:  theta_t = theta~_t - lambda * r_t * (theta~_t - theta_0)   (Eq. 4)
    neu c_t >= 0: theta_t = theta~_t          (KHONG phat gi ca)

Doc cho de hieu: c_t < 0 nghia la buoc dang di **cung huong** voi do lech khoi neo, tuc
lop nay dang bi keo ngay cang xa ma van giam duoc loss -> SPD ham lai. c_t >= 0 nghia la
do lech va gradient nghich nhau -> de yen. Va r_t chi khac 0 khi do lech **tang them**
trong chinh buoc nay, nen luc lop da dung yen thi phat tu tat.

Voi lambda = 1, phep phat DUNG BANG phep chieu len qua cau ban kinh = do lech cua buoc
truoc (Eq. 10) — do la ly do bai khuyen bat dau tu lambda = 1 chu khong phai mot so nho.

BA CHO PHAI GHI RO VI KHAC BAI GOC:

1. **theta_0 cua ta la checkpoint Pha 1**, khong phai model pretrained. Do la chu dich:
   neo vao Pha 1 moi la transfer. Dung `--recadam_anchor pretrained` de neo ve pretrained
   nhu bai goc — dung chung duong `build_recadam_anchor` voi RecAdam.
2. **"Lop" = mot tensor tham so.** Bai goc noi "layer"; ma tham chieu (GT-RIPL/
   Selective-Projection-Decay) cung lam theo tung tensor trong param group.
3. **Weight decay giu nguyen 0.01 cua du an** (mac dinh), du bai goc dat SPD nhu vat
   THAY THE weight decay. Ly do: doi chung cua ta la AdamW wd=0.01, giu nguyen thi phep
   so chi doi DUNG MOT bien la so hang chieu. Dat `--spd_replace_wd` de chay dung nhu bai.

Kiem tinh dung: lambda = 0 thi lop nay phai cho quy dao TRUNG KHIT AdamW.
`tests/test_spd.py` kiem dieu do va ba cong hai chieu khac.
"""

import math

import torch
from torch.optim import Optimizer


class AdamSPD(Optimizer):
    """Adam voi Selective Projection Decay. Tham so giong AdamW, them `anchor_params`.

    anchor_params: list tensor cung thu tu va shape voi `params` (theta_0). `None` cho ca
    nhom nghia la khong neo -> lop nay suy bien thanh AdamW, dung de kiem tinh dung.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0,
                 correct_bias=True, spd_lambda=1.0, anchor_params=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        if spd_lambda < 0.0:
            raise ValueError(f"spd_lambda phai >= 0, nhan {spd_lambda}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias, spd_lambda=spd_lambda,
                        anchor_params=anchor_params)
        super().__init__(params, defaults)
        # Dem de doc log: bao nhieu lan dieu kien c_t < 0 duoc kich hoat.
        self.n_fired = 0
        self.n_checked = 0

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            anchors = group.get("anchor_params") or [None] * len(group["params"])
            lam = group["spd_lambda"]
            for p, p0 in zip(group["params"], anchors):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamSPD does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # --- do TRUOC khi p doi: c_t va gamma_{t-1} deu dung theta_{t-1} ---
                fire = False
                gamma_prev = None
                if p0 is not None and lam > 0.0:
                    dev_prev = p.data - p0.data
                    c_t = -torch.sum(grad * dev_prev)
                    gamma_prev = torch.linalg.vector_norm(dev_prev)
                    fire = bool(c_t < 0)
                    self.n_checked += 1
                    del dev_prev

                # --- buoc Adam thuong -> theta~_t ---
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # --- chieu chon loc, chi khi c_t < 0 ---
                if fire:
                    dev_new = p.data - p0.data
                    gamma_t = torch.linalg.vector_norm(dev_new)
                    if float(gamma_t) > 0.0:
                        r_t = torch.clamp(gamma_t - gamma_prev, min=0.0) / gamma_t
                        if float(r_t) > 0.0:
                            p.data.add_(dev_new, alpha=-float(lam * r_t))
                            self.n_fired += 1
                    del dev_new

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
        return loss

    def fire_rate(self):
        """Ti le lan dieu kien c_t < 0 duoc kich hoat. 0 nghia la SPD chua bao gio phat —
        khi do no DUNG BANG AdamW va moi ket qua phai doc nhu AdamW, khong phai 'SPD kem'."""
        return self.n_fired / self.n_checked if self.n_checked else 0.0
