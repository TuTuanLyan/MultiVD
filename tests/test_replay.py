#!/usr/bin/env python3
"""Kiem src/replay.py HAI CHIEU truoc khi tin no.

    python tests/test_replay.py

1. Lich mu(e): dung gia tri o dau, giua, cuoi; mu0=0 hoac epochs=0 phai ra dung dang.
2. Trong so tang: stratify=False -> deu; stratify=True -> moi tang tong = 1 (bang nhau).
3. Lay mau that qua SourceReplay (tokenizer gia): stratify -> tan suat tang xap xi deu;
   khong stratify -> tan suat ti le voi so dong. CA HAI chieu phai dung.
4. Hai backward NOI TIEP (dich roi nguon) cho DUNG gradient cua backward tong loss — day la
   ly do bo nho cua cach cai trong train.py; neu sai thi replay dang toi uu mot muc tieu khac.
"""
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replay import SourceReplay, replay_mu, stratum_weights   # noqa: E402


class FakeTok:
    pad_token_id = 0

    def __call__(self, text, **kw):
        return {"input_ids": [1 + (ord(c) % 50) for c in text[:20]]}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [101] + ids + [102]


def make_records():
    # 4 tang lech manh: (1,2) 60 dong, (0,2) 60, (1,0) 6, (0,0) 6 -> stratify phai san bang.
    recs = []
    for label, cwe, n in [(1, 2, 60), (0, 2, 60), (1, 0, 6), (0, 0, 6)]:
        recs += [{"code": f"x{label}{cwe}{i}", "label": label, "cwe_class": cwe} for i in range(n)]
    return recs


def test_schedule():
    assert abs(replay_mu(1, 0.5, 6) - 0.5) < 1e-12
    assert abs(replay_mu(4, 0.5, 6) - 0.25) < 1e-12
    assert replay_mu(7, 0.5, 6) == 0.0 and replay_mu(30, 0.5, 6) == 0.0
    assert replay_mu(1, 0.0, 6) == 0.0 and replay_mu(5, 0.0, 0) == 0.0
    assert replay_mu(9, 0.3, 0) == 0.3, "epochs=0 phai la mu hang"
    print("1. lich mu(e)                         OK")


def test_weights():
    recs = make_records()
    w, counts = stratum_weights(recs, False)
    assert set(w) == {1.0}
    w, counts = stratum_weights(recs, True)
    tot = Counter()
    for r, x in zip(recs, w):
        tot[(r["label"], r["cwe_class"])] += x
    assert all(abs(v - 1.0) < 1e-9 for v in tot.values()), tot
    print("2. trong so tang (deu / can bang)     OK")


def draw(stratify, n_batches=150):
    rp = SourceReplay(make_records(), FakeTok(), 24, "head", 16, 0.5, 6, stratify, seed=1)
    c = Counter()
    for _ in range(n_batches):
        b = rp.next_batch()
        for l, k in zip(b["labels"].tolist(), b["cwe_class"].tolist()):
            c[(l, k)] += 1
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def test_sampling():
    f_s = draw(True)
    f_u = draw(False)
    # stratify: 4 tang ~ 0.25 moi tang
    assert all(abs(v - 0.25) < 0.05 for v in f_s.values()), f_s
    # khong stratify: tang (1,2) ~ 60/132 = 0.4545, tang (1,0) ~ 6/132 = 0.045
    assert abs(f_u[(1, 2)] - 60 / 132) < 0.05 and abs(f_u[(1, 0)] - 6 / 132) < 0.03, f_u
    assert f_s[(1, 0)] > 3 * f_u[(1, 0)], "hai che do phai KHAC nhau ro: tang hiem phai duoc nang len"
    print(f"3. lay mau  stratify={ {k: round(v, 3) for k, v in sorted(f_s.items())} }")
    print(f"            uniform  ={ {k: round(v, 3) for k, v in sorted(f_u.items())} }   OK")


def test_sequential_backward():
    torch.manual_seed(0)
    lin = torch.nn.Linear(5, 2)
    xt, yt = torch.randn(8, 5), torch.randint(0, 2, (8,))
    xs, ys = torch.randn(8, 5), torch.randint(0, 2, (8,))
    mu = 0.37
    # (a) mot backward cua tong
    lin.zero_grad()
    (torch.nn.functional.cross_entropy(lin(xt), yt)
     + mu * torch.nn.functional.cross_entropy(lin(xs), ys)).backward()
    g_sum = [p.grad.clone() for p in lin.parameters()]
    # (b) hai backward noi tiep, nhu train_one_epoch_phase2
    lin.zero_grad()
    torch.nn.functional.cross_entropy(lin(xt), yt).backward()
    (mu * torch.nn.functional.cross_entropy(lin(xs), ys)).backward()
    g_seq = [p.grad.clone() for p in lin.parameters()]
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(g_sum, g_seq))
    # chieu LECH: bo phan nguon thi gradient phai KHAC
    lin.zero_grad()
    torch.nn.functional.cross_entropy(lin(xt), yt).backward()
    g_only = [p.grad.clone() for p in lin.parameters()]
    assert not all(torch.allclose(a, b, atol=1e-7) for a, b in zip(g_sum, g_only))
    print("4. hai backward noi tiep == backward tong; bo nguon thi khac   OK")


if __name__ == "__main__":
    test_schedule()
    test_weights()
    test_sampling()
    test_sequential_backward()
    print("TAT CA OK")
