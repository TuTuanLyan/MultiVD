"""Kiem `src/pairloss.py` — CA HAI CHIEU cho moi phep.

Bai hoc da tra gia: `bash -n` / "chay khong loi" khong chung minh gi. Moi phep o day deu co
mot truong hop PHAI dat va mot truong hop PHAI hong, de biet phep kiem con phan biet duoc.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from pairloss import PairBatchSampler, build_pair_index, pair_margin_loss  # noqa: E402


def rec(pid, label):
    return {"pair_id": pid, "label": label}


# ---------------------------------------------------------------- build_pair_index
def test_bat_dung_cap_day_du():
    recs = [rec("a", 1), rec("a", 0), rec("b", 0), rec("b", 1)]
    group, role, n = build_pair_index(recs)
    assert n == 2
    assert group[0] == group[1] and group[2] == group[3]
    assert group[0] != group[2]
    assert role[0] == 1 and role[1] == 0
    assert role[3] == 1 and role[2] == 0


def test_bo_nhom_ba_hang_va_nhom_cung_nhan():
    """Chieu nguoc: nhung nhom KHONG phai cap truoc/sau phai bi loai, khong duoc gop bua."""
    recs = [rec("a", 1), rec("a", 0), rec("a", 1),      # ba hang
            rec("b", 1), rec("b", 1),                   # cung nhan
            rec("c", 1), rec("c", 0),                   # hop le
            {"pair_id": None, "label": 1}]              # khong co pair_id
    group, role, n = build_pair_index(recs)
    assert n == 1, "chi 'c' la cap day du"
    assert (group[:5] == -1).all(), "nhom ba hang va nhom cung nhan phai bi loai"
    assert group[5] == group[6] == 0
    assert group[7] == -1


# ---------------------------------------------------------------- PairBatchSampler
def _pairs_intact(batches, group):
    """Moi cap xuat hien phai nam TRON trong mot batch."""
    seen = {}
    for bi, b in enumerate(batches):
        for i in b:
            g = int(group[i])
            if g < 0:
                continue
            seen.setdefault(g, set()).add(bi)
    return all(len(v) == 1 for v in seen.values())


def test_hai_nua_cua_mot_cap_luon_cung_batch():
    recs = [r for k in range(20) for r in (rec(f"p{k}", 1), rec(f"p{k}", 0))]
    group, _, n = build_pair_index(recs)
    assert n == 20
    s = PairBatchSampler(group, batch_size=6, seed=1)
    batches = list(iter(s))
    assert sum(len(b) for b in batches) == len(recs), "khong duoc mat hang nao"
    assert _pairs_intact(batches, group)


def test_chieu_nguoc_sampler_ngau_nhien_thuong_PHA_cap():
    """Neu khong co sampler nay thi phep kiem tren se HONG — chung to no khong rong."""
    recs = [r for k in range(20) for r in (rec(f"p{k}", 1), rec(f"p{k}", 0))]
    group, _, _ = build_pair_index(recs)
    g = torch.Generator()
    g.manual_seed(0)
    order = torch.randperm(len(recs), generator=g).tolist()
    batches = [order[i:i + 6] for i in range(0, len(order), 6)]
    assert not _pairs_intact(batches, group), "xao ngau nhien le ra phai lam vo cap"


def test_sampler_xac_dinh_theo_seed_va_epoch():
    recs = [r for k in range(12) for r in (rec(f"p{k}", 1), rec(f"p{k}", 0))]
    group, _, _ = build_pair_index(recs)
    a = PairBatchSampler(group, batch_size=4, seed=7)
    b = PairBatchSampler(group, batch_size=4, seed=7)
    assert list(iter(a)) == list(iter(b)), "cung seed phai cho cung thu tu"
    a.set_epoch(1)
    assert list(iter(a)) != list(iter(b)), "doi epoch phai doi thu tu"


def test_len_khop_so_batch_thuc_te():
    for n_pair, n_single, bs in [(20, 0, 6), (7, 5, 4), (3, 11, 8), (1, 0, 2)]:
        recs = [r for k in range(n_pair) for r in (rec(f"p{k}", 1), rec(f"p{k}", 0))]
        recs += [{"pair_id": None, "label": k % 2} for k in range(n_single)]
        group, _, _ = build_pair_index(recs)
        s = PairBatchSampler(group, batch_size=bs, seed=3)
        assert len(s) == len(list(iter(s))), f"len() sai o (cap={n_pair}, le={n_single}, bs={bs})"


def test_hang_le_khong_bi_mat():
    recs = [rec("a", 1), rec("a", 0)] + [{"pair_id": None, "label": 1} for _ in range(5)]
    group, _, _ = build_pair_index(recs)
    s = PairBatchSampler(group, batch_size=4, seed=0)
    flat = [i for b in iter(s) for i in b]
    assert sorted(flat) == list(range(len(recs)))


# ---------------------------------------------------------------- pair_margin_loss
def _batch(group, role, idxs, scores):
    """Dung logits sao cho logit[1]-logit[0] = score."""
    logits = torch.zeros(len(idxs), 2)
    logits[:, 1] = torch.tensor(scores, dtype=torch.float)
    return logits, torch.tensor(idxs, dtype=torch.long), group, role


def test_loss_bang_0_khi_da_vuot_bien_va_lon_khi_nguoc_dau():
    recs = [rec("a", 1), rec("a", 0)]
    group, role, _ = build_pair_index(recs)
    # s_vul - s_fixed = 5 >> margin 1  => softplus(1-5) ~ 0.018
    lo, n = pair_margin_loss(*_batch(group, role, [0, 1], [5.0, 0.0]), margin=1.0)
    assert n == 1 and lo.item() < 0.05
    # nguoc dau: s_vul - s_fixed = -5 => softplus(6) ~ 6
    hi, n2 = pair_margin_loss(*_batch(group, role, [0, 1], [0.0, 5.0]), margin=1.0)
    assert n2 == 1 and hi.item() > 5.0
    assert hi.item() > lo.item() * 50, "hai chieu phai khac nhau ro rang"


def test_bo_qua_cap_chi_co_mot_nua_trong_batch():
    recs = [rec("a", 1), rec("a", 0), rec("b", 1), rec("b", 0)]
    group, role, _ = build_pair_index(recs)
    # batch chi chua nua 'vul' cua hai cap
    loss, n = pair_margin_loss(*_batch(group, role, [0, 2], [0.0, 0.0]), margin=1.0)
    assert n == 0, "khong cap nao du hai nua => khong duoc tinh"
    assert loss.item() == 0.0


def test_khong_cap_nao_van_backward_duoc():
    """Batch toan hang le: loss phai la 0 NHUNG van noi voi do thi, khong lam vo backward."""
    recs = [{"pair_id": None, "label": 1}, {"pair_id": None, "label": 0}]
    group, role, _ = build_pair_index(recs)
    logits = torch.zeros(2, 2, requires_grad=True)
    loss, n = pair_margin_loss(logits, torch.tensor([0, 1]), group, role, margin=1.0)
    assert n == 0
    loss.backward()
    assert logits.grad is not None and torch.allclose(logits.grad, torch.zeros_like(logits))


def test_gradient_day_theo_dung_chieu():
    recs = [rec("a", 1), rec("a", 0)]
    group, role, _ = build_pair_index(recs)
    logits = torch.zeros(2, 2, requires_grad=True)
    loss, _ = pair_margin_loss(logits, torch.tensor([0, 1]), group, role, margin=1.0)
    loss.backward()
    # tang score cua ban VUL (hang 0, cot 1) phai LAM GIAM loss => gradient am
    assert logits.grad[0, 1] < 0, "phai day score cua ban co lo hong LEN"
    assert logits.grad[1, 1] > 0, "phai day score cua ban da va XUONG"


def test_bien_lon_hon_thi_loss_lon_hon():
    recs = [rec("a", 1), rec("a", 0)]
    group, role, _ = build_pair_index(recs)
    l1, _ = pair_margin_loss(*_batch(group, role, [0, 1], [1.0, 0.0]), margin=1.0)
    l2, _ = pair_margin_loss(*_batch(group, role, [0, 1], [1.0, 0.0]), margin=3.0)
    assert l2.item() > l1.item()


def test_trung_binh_tren_CAC_CAP_khong_phai_tren_hang():
    """Hai cap giong het nhau phai cho cung loss voi mot cap — neu chia theo HANG thi sai mot nua."""
    one = [rec("a", 1), rec("a", 0)]
    two = one + [rec("b", 1), rec("b", 0)]
    g1, r1, _ = build_pair_index(one)
    g2, r2, _ = build_pair_index(two)
    a, _ = pair_margin_loss(*_batch(g1, r1, [0, 1], [0.0, 0.0]), margin=1.0)
    b, n = pair_margin_loss(*_batch(g2, r2, [0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0]), margin=1.0)
    assert n == 2
    assert abs(a.item() - b.item()) < 1e-6
