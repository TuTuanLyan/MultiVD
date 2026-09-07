"""Cong hai chieu cho src/spd.py (AdamSPD) — SPD, NeurIPS 2024, arXiv:2411.01713.

Moi cong co MOT truong hop phai KHOP va MOT truong hop phai LECH. Cong chi biet bao
"khop" thi khong chung minh duoc gi (CLAUDE.md muc 8).

HIEU SAI DA MAC khi viet lan dau (07/09): tuong SPD se phat khi mo hinh di RA XA neo.
Nguoc lai. c_t = (-g_t)·(theta_{t-1} - theta_0) do do KHOP giua huong giam loss va huong
da di duoc. c_t > 0 = "van dang tien bo nhat quan" -> KHONG phat. c_t < 0 = gradient da
quay dau ma quan tinh (momentum) van day di tiep -> phat. Nen mot bai toan hai chieu tron
tru KHONG BAO GIO kich hoat SPD, va test dau tien "that bai" vi bai toan sai chu khong
phai vi code sai. Muon cham duong phat thi phai dung MOMENTUM lech pha voi gradient.

    python3 tests/test_spd.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from RecAdam import RecAdam          # noqa: E402
from spd import AdamSPD              # noqa: E402

LR = 2e-5          # tham so THAT cua du an
WD = 0.01
STEPS = 40
OK = True


def report(name, passed, detail=""):
    global OK
    OK = OK and passed
    print(f"  [{'OK ' if passed else 'SAI'}] {name}{('  | ' + detail) if detail else ''}")


def run_quadratic(kind, spd_lambda=1.0, anchor=True, steps=STEPS, lr=LR, weight_decay=WD):
    """Bai toan hai chieu tron tru: gradient LUON cung huong voi tien bo -> SPD khong phat.
    Dung de kiem 'khong phat thi phai trung khit Adam'."""
    g = torch.Generator().manual_seed(0)
    theta0 = torch.randn(2, generator=g)
    target = theta0 + torch.tensor([3.0, -2.0])
    p = torch.nn.Parameter(theta0.clone())
    a = [theta0.clone()] if anchor else None
    if kind == "spd":
        opt = AdamSPD([p], lr=lr, weight_decay=weight_decay, spd_lambda=spd_lambda, anchor_params=a)
    else:
        opt = RecAdam([p], lr=lr, weight_decay=weight_decay, anneal_w=0.0,
                      pretrain_params=[theta0.clone()])
    traj = []
    for _ in range(steps):
        opt.zero_grad()
        ((p - target) ** 2).sum().backward()
        opt.step()
        traj.append(p.detach().clone())
    return torch.stack(traj), theta0, opt


print("1) lambda = 0 phai TRUNG KHIT Adam thuan (RecAdam anneal_w=0)")
a, _, _ = run_quadratic("spd", spd_lambda=0.0)
b, _, _ = run_quadratic("recadam")
d = (a - b).abs().max().item()
report("lambda=0 == Adam", d == 0.0, f"lech toi da {d:.3e}")

print("\n2) anchor=None cung phai TRUNG KHIT")
a2, _, _ = run_quadratic("spd", spd_lambda=1.0, anchor=False)
report("anchor=None == Adam", (a2 - b).abs().max().item() == 0.0)

print("\n3) tien bo NHAT QUAN (c_t > 0 moi buoc) => SPD KHONG duoc phat, va phai == Adam")
c, _, opt3 = run_quadratic("spd", spd_lambda=1.0)
report("khong kich hoat lan nao", opt3.n_fired == 0, f"n_fired={opt3.n_fired}/{opt3.n_checked}")
report("nen trung khit Adam", (c - b).abs().max().item() == 0.0)

print("\n4) chieu LECH: ep gradient QUAY DAU trong khi momentum van day tiep")
# 1 chieu, neo tai 0. Nam buoc dau g = -1 (day theta LEN, do lech tang, c_t > 0).
# Tu buoc 6 doi g = +1: c_t < 0 (gradient nguoc voi tien bo) nhung m_hat van am nen
# buoc VAN di len -> do lech VAN tang -> r_t > 0 -> phat that su chay.
grads = [-1.0] * 5 + [1.0] * 5
def run_flip(spd_lambda):
    theta0 = torch.zeros(1)
    p = torch.nn.Parameter(torch.zeros(1))
    opt = AdamSPD([p], lr=0.1, weight_decay=0.0, spd_lambda=spd_lambda,
                  anchor_params=[theta0.clone()])
    hist = []
    for gv in grads:
        p.grad = torch.tensor([gv])
        before = p.data.clone()
        fired_before = opt.n_fired
        opt.step()
        hist.append((before.clone(), p.data.clone(), opt.n_fired > fired_before))
        p.grad = None
    return p, opt, hist

p_l1, opt_l1, hist1 = run_flip(1.0)
p_l0, opt_l0, hist0 = run_flip(0.0)
report("co kich hoat", opt_l1.n_fired > 0, f"n_fired={opt_l1.n_fired}/{opt_l1.n_checked}")
report("lambda=0 khong bao gio phat", opt_l0.n_fired == 0)
report("lambda=1 ket thuc GAN NEO hon lambda=0",
       abs(float(p_l1.data)) < abs(float(p_l0.data)),
       f"|theta|: {abs(float(p_l1.data)):.6f} < {abs(float(p_l0.data)):.6f}")

print("\n5) CONG THUC — buoc phat DAU TIEN phai khop theta~ - lambda*r_t*(theta~ - theta0)")
# Truoc buoc phat dau tien hai quy dao van trung khit, nen theta~ cua lambda=1 chinh la
# theta sau buoc do cua lambda=0. Kiem tra dung buoc do, khong doc lai code cua optimizer.
k = next(i for i, h in enumerate(hist1) if h[2])
theta_prev = hist1[k][0]
theta_tilde = hist0[k][1]                      # buoc Adam thuan, chua phat
gamma_prev = theta_prev.norm()
gamma_t = theta_tilde.norm()
r_t = torch.clamp(gamma_t - gamma_prev, min=0.0) / gamma_t
expect = theta_tilde - 1.0 * r_t * theta_tilde   # theta0 = 0
report("phat dau tien o buoc dung luc g doi dau", k == 5, f"buoc {k} (0-based)")
report("r_t > 0 (do lech VAN tang du gradient da quay dau)", float(r_t) > 0.0,
       f"gamma {float(gamma_prev):.6f} -> {float(gamma_t):.6f}, r_t={float(r_t):.4f}")
report("theta khop cong thuc tay", torch.allclose(hist1[k][1], expect, atol=1e-7),
       f"thuc {hist1[k][1].tolist()} vs tay {expect.tolist()}")

print("\n6) don dieu: lambda lon hon => ket thuc gan neo hon")
fars = [(lam, abs(float(run_flip(lam)[0].data))) for lam in (0.0, 0.5, 1.0, 2.0)]
mono = all(fars[i][1] >= fars[i + 1][1] for i in range(len(fars) - 1))
report("lambda tang => do lech giam", mono, " > ".join(f"l={l}:{f:.6f}" for l, f in fars))

print("\n" + ("TAT CA CONG DEU DAT" if OK else "CO CONG THAT BAI"))
sys.exit(0 if OK else 1)
