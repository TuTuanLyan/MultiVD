#!/usr/bin/env python3
"""Kiểm luật phát hiện kẹt Pha 1 (mwg.utils.train_loss_stalled, ngưỡng mặc định 2 %) trên train loss epoch 1–3 của
các lượt Pha 1 đã chạy (log ở ../archive và multiBABEL). Chạy: python tests/test_stuck.py

Kẹt = sau epoch 3 loss vẫn nằm ở mức nghiệm tầm thường (xác nhận bằng các epoch sau hoặc val ROC ~0,5)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mwg.utils import train_loss_stalled  # noqa: E402

MIN_DROP = 0.02
# (lượt, train loss epoch 1, 2, 3, kẹt?)
RUNS = [
    # kẹt: các epoch sau vẫn không giảm
    ("refactor seed 36 lần 1", 1.9076, 1.8947, 1.8859, True),
    ("refactor warmup 0,2 (e5 1,8678)", 1.9163, 1.8970, 1.8770, True),
    ("refactor graph_lr 1e-5 seed 1234 (e6 1,8692)", 1.9197, 1.8757, 1.8748, True),
    ("v4 jsCjv_rand fixg seed 38 (158)", 1.8845, 1.8892, 1.8869, True),
    ("jsCjv seed 36 (158, e4 1,9405)", 1.9645, 1.9596, 1.9583, True),
    ("jsCjv seed 37 (158)", 1.9492, 1.9597, 1.9589, True),
    ("jsCjv seed 36 (local)", 1.9586, 1.9692, 1.9569, True),
    ("ccpyR (158, e6 1,8913)", 1.9192, 1.8915, 1.8945, True),
    ("graph_lr 5e-5", 1.9553, 1.9604, 1.9576, True),
    ("không pair loss (chỉ CE, mức loss khác hẳn)", 0.7336, 0.7106, 0.7028, True),
    # học: các lượt thoát chậm nhất
    ("refactor warmup 0,25", 1.9191, 1.8833, 1.7275, False),
    ("refactor graph_lr 1e-5 seed 7", 1.8779, 1.8495, 1.6232, False),
    ("v4 jsCjv_rand (158)", 1.9107, 1.8787, 1.7041, False),
    ("ccjscjv theo cặp", 1.4452, 1.4210, 1.3854, False),
    ("jsC", 1.5364, 1.5009, 1.4228, False),
    # học: lượt SOTA thường gặp
    ("B / R1 (code cũ, seed 36)", 1.9097, 1.7135, 1.3707, False),
    ("refactor seed 36 lần 2", 1.9107, 1.7623, 1.4923, False),
    ("refactor seed 42", 1.8744, 1.6701, 1.4580, False),
    ("v4 jsCjv_rand fixg (local)", 1.9090, 1.7283, 1.4090, False),
    ("v4 jsCjv_rand seed 38 (158, không fixg)", 1.8728, 1.6726, 1.3767, False),
]

bad = 0
for name, *losses, stuck in RUNS:
    got, drop = train_loss_stalled(losses, MIN_DROP)
    ok = got == stuck
    bad += not ok
    print(f"{'ok ' if ok else 'SAI'} giảm {100 * drop:5.2f}%  {'kẹt' if got else 'học'}  {name}")
print(f"\n{len(RUNS) - bad}/{len(RUNS)} ca đạt")
sys.exit(1 if bad else 0)
