#!/usr/bin/env python3
"""Kiem RecAdamFisher HAI CHIEU truoc khi tin no.

    python tests/test_recadam_fisher.py

Chieu 1 — KHOP: dat F = 1 cho moi tham so thi quy dao phai TRUNG KHIT RecAdam goc.
          Neu khong khop thi lop moi da lam sai phan toan, khong phai lam khac.
Chieu 2 — LECH: dat F khac 1 thi quy dao phai KHAC. Mot phep kiem chi thu chieu 1 se
          bo lot loi "fisher_params bi bo qua hoan toan" — luc do chieu 1 van khop.

Bai hoc chung cua du an: cong nao cung phai duoc cho mot truong hop dung VA mot truong
hop sai, roi xem no tra loi dung ca hai chua (memory record-general-technical-mistakes).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from RecAdam import RecAdam                      # noqa: E402
from recadam_fisher import RecAdamFisher          # noqa: E402

# THAM SO THUC TE cua du an: lr 2e-5, gamma 5000 -> lr*gamma = 0.1.
# KHONG dung lr=1e-2 voi gamma=500 (lr*gamma = 5): luc keo cap nhat HIEN co dieu kien
# on dinh |lr*gamma*F*(1-lambda)| < 2, nen bo tham so do phan ky ngay ca khi ma dung.
KW = dict(lr=2e-5, weight_decay=0.01, anneal_fun="sigmoid",
          anneal_k=0.05, anneal_t0=5, anneal_w=1.0, pretrain_cof=5000.0)


def run(optimizer_cls, fisher=None, steps=25, seed=0):
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.tensor([1.5, -0.7, 0.3, 2.0]))
    anchor = torch.tensor([0.0, 0.0, 0.0, 0.0])
    target = torch.tensor([1.0, 1.0, 1.0, 1.0])
    kwargs = dict(KW)
    kwargs["pretrain_params"] = [anchor]
    if optimizer_cls is RecAdamFisher:
        kwargs["fisher_params"] = [fisher]
    opt = optimizer_cls([w], **kwargs)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        ((w - target) ** 2).sum().backward()
        opt.step()
    return w.detach().clone()


def main():
    ok = True

    a = run(RecAdam)
    b = run(RecAdamFisher, fisher=torch.ones(4))
    diff = (a - b).abs().max().item()
    print(f"chieu 1 (F=1 phai TRUNG KHIT RecAdam goc): lech toi da = {diff:.3e}")
    if diff > 1e-12:
        print(f"  !! KHONG KHOP\n     RecAdam       {a.tolist()}\n     RecAdamFisher {b.tolist()}")
        ok = False
    else:
        print("  OK")

    c = run(RecAdamFisher, fisher=None)
    diff_none = (a - c).abs().max().item()
    print(f"chieu 1b (fisher=None cung phai trung khit): lech toi da = {diff_none:.3e}")
    if diff_none > 1e-12:
        print("  !! KHONG KHOP")
        ok = False
    else:
        print("  OK")

    f = torch.tensor([10.0, 0.01, 1.0, 0.01])       # trung binh ~ 2.75, khong phai 1
    # lr*gamma*F_max = 2e-5*5000*10 = 1.0 < 2 -> on dinh
    d = run(RecAdamFisher, fisher=f)
    diff2 = (a - d).abs().max().item()
    print(f"chieu 2 (F khac 1 phai KHAC quy dao): lech toi da = {diff2:.3e}")
    if diff2 < 1e-6:
        print("  !! KHONG KHAC — fisher_params dang bi BO QUA")
        ok = False
    else:
        print("  OK")
        # tham so co F lon bi ghim gan neo hon tham so co F nho
        pull_strong = abs(d[0].item())      # F=10  -> keo manh ve 0
        pull_weak = abs(d[1].item())        # F=0.01 -> gan nhu tu do
        print(f"     |w0| (F=10)   = {pull_strong:.4f}   <- phai NHO hon")
        print(f"     |w1| (F=0.01) = {pull_weak:.4f}")
        if pull_strong >= pull_weak:
            print("  !! HUONG SAI: F lon phai keo ve neo MANH hon")
            ok = False

    # chieu 3: nguong on dinh phai duoc phat hien, khong duoc phan ky im lang
    print("chieu 3 (cong on dinh phai phan biet dung BA muc):")
    from recadam_fisher import check_pull_stability
    cases = [
        # (lr, gamma, Fmax, mong doi on dinh, ghi chu)
        (2e-5, 500.0,  10.0, True,  "gamma 500 + clip 10 -> c = 0.1"),
        (2e-5, 5000.0, 5.0,  True,  "gamma 5000 + clip 5 -> c = 0.5, sat bien"),
        (2e-5, 5000.0, 10.0, False, "gamma 5000 + clip 10 -> c = 1.0, keo het ve neo"),
        (1e-2, 500.0,  10.0, False, "lr lon -> c = 50, PHAN KY"),
    ]
    for lr, g, fm, want, note in cases:
        got, msg = check_pull_stability(lr=lr, pretrain_cof=g, fisher_max=fm)
        mark = "OK " if got == want else "!! SAI"
        print(f"     {mark} {note:<46} -> {msg}")
        if got != want:
            ok = False

    print("\n" + ("TAT CA DAT" if ok else "CO PHEP KIEM THAT BAI"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
