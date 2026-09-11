"""Kiem thu MANH 2 — duong quyet dinh kep co cong (model.enable_latent_gate).

Moi phep deu kiem CA HAI CHIEU: mot truong hop PHAI khop va mot truong hop PHAI lech.
Cong chi bao "an toan" thi khong chung minh duoc gi — bay da tra gia o CLAUDE.md muc 8.

    python tests/test_latent_gate.py
"""
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn


@contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"cho doi {exc.__name__} nhung khong co ngoai le nao")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.model import TransferModel  # noqa: E402
from src.train import _phase2_target_loss  # noqa: E402


class TinyBackbone(nn.Module):
    """Backbone gia, du de co `config.hidden_size` va `last_hidden_state`."""

    class _Cfg:
        hidden_size = 16

    def __init__(self):
        super().__init__()
        self.config = self._Cfg()
        self.emb = nn.Embedding(50, 16)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.emb(input_ids)
        return type("O", (), {"last_hidden_state": out})()


def make(num_latent=8, num_classes=2):
    torch.manual_seed(0)
    return TransferModel(TinyBackbone(), num_classes=num_classes, num_cwes=4,
                         aux_mode="latent_bottleneck", num_latent=num_latent, pooling="cls")


def batch(n=6):
    torch.manual_seed(1)
    return {"input_ids": torch.randint(0, 50, (n, 7)),
            "attention_mask": torch.ones(n, 7, dtype=torch.long),
            "labels": torch.randint(0, 2, (n,)),
            "cwe_class": torch.randint(0, 4, (n,)),
            "index": torch.arange(n)}


# ---------------------------------------------------------------- chieu 1: TAT
def test_tat_thi_khong_co_gi_doi():
    """Khong bat cong => dict tra ve KHONG co khoa moi, va logits khong doi."""
    m = make().eval()
    b = batch()
    out = m(b["input_ids"], b["attention_mask"])
    assert "gate_mean" not in out and "lat_vul_logits" not in out
    assert m.gate_value() is None
    # va `latent_proj` van huan luyen duoc (chua ai dong bang no)
    assert all(p.requires_grad for p in m.latent_proj.parameters())


# ---------------------------------------------------------------- chieu 2: BAT
def test_bat_thi_dong_bang_dung_cai_can_dong_bang():
    m = make()
    before = [p.detach().clone() for p in m.latent_proj.parameters()]
    m.enable_latent_gate(mode="scalar")
    # `latent_proj` PHAI dong bang — do la ca co che
    assert all(not p.requires_grad for p in m.latent_proj.parameters())
    # ...nhung gia tri KHONG duoc doi: dong bang, khong phai khoi tao lai
    for a, b_ in zip(before, m.latent_proj.parameters()):
        assert torch.equal(a, b_.detach())
    # nhanh moi PHAI huan luyen duoc
    assert all(p.requires_grad for p in m.lat_vul_head.parameters())
    assert m.gate_logit.requires_grad
    # va nhanh 768 chieu van huan luyen duoc
    assert all(p.requires_grad for p in m.vul_head.parameters())


def test_g_bang_0_va_g_bang_1_la_hai_duong_thuan_tuy():
    """Phep kiem HAI CHIEU quan trong nhat: cong co dung nghia so hoc no khai bao khong."""
    m = make().eval()
    b = batch()
    with torch.no_grad():
        main_only = m(b["input_ids"], b["attention_mask"])["vul_logits"].clone()

    m.enable_latent_gate(mode="scalar")
    m.eval()
    # g -> 0: phai TRUNG KHIT nhanh 768 chieu
    with torch.no_grad():
        m.gate_logit.fill_(-40.0)
        g0 = m(b["input_ids"], b["attention_mask"])
    assert torch.allclose(g0["vul_logits"], main_only, atol=1e-5)
    assert float(g0["gate_mean"]) < 1e-6

    # g -> 1: phai TRUNG KHIT nhanh 8 chieu, va PHAI KHAC nhanh 768 chieu
    with torch.no_grad():
        m.gate_logit.fill_(40.0)
        g1 = m(b["input_ids"], b["attention_mask"])
    assert torch.allclose(g1["vul_logits"], g1["lat_vul_logits"], atol=1e-5)
    assert not torch.allclose(g1["vul_logits"], main_only, atol=1e-3), \
        "g=1 ma van ra dung nhanh 768 chieu => cong khong noi vao dau ca"


def test_cong_phu_thuoc_dau_vao_khoi_tao_trung_voi_cong_vo_huong():
    """mode='input' luc khoi tao phai la mot HANG SO — khong nhay bac giua hai che do."""
    m = make().eval()
    m.enable_latent_gate(mode="input", init_logit=0.3)
    b = batch()
    with torch.no_grad():
        g = torch.sigmoid(m.gate_proj(m.dropout(
            m.backbone(input_ids=b["input_ids"]).last_hidden_state[:, 0])))
    assert torch.allclose(g, torch.full_like(g, float(torch.sigmoid(torch.tensor(0.3)))), atol=1e-6)
    assert g.std() < 1e-6, "trong so gate_proj phai bang 0 luc khoi tao"


# ---------------------------------------------------------------- giam sat nhanh 8 chieu
def test_alpha_co_vao_loss_that():
    # eval() LA BAT BUOC: dropout bat thi hai lan goi da khac nhau san, phep so vo nghia.
    m = make()
    m.enable_latent_gate(mode="scalar")
    m.eval()
    b = batch()
    l0, *_ = _phase2_target_loss(m, b, torch.device("cpu"), 0.0, gate_alpha=0.0)
    l1, *_ = _phase2_target_loss(m, b, torch.device("cpu"), 0.0, gate_alpha=0.5)
    assert not torch.allclose(l0, l1), "alpha>0 ma loss khong doi => giam sat khong vao"
    assert float(l1) > float(l0), "cong them mot cross-entropy phai lam loss LON hon"
    # chieu nguoc: khong bat cong thi alpha KHONG duoc co tac dung nao
    m2 = make().eval()
    a, *_ = _phase2_target_loss(m2, b, torch.device("cpu"), 0.0, gate_alpha=0.0)
    c, *_ = _phase2_target_loss(m2, b, torch.device("cpu"), 0.0, gate_alpha=0.5)
    assert torch.allclose(a, c), "chua bat cong ma alpha van doi loss => ro ri sang duong cu"


def test_cong_THUC_SU_DI_CHUYEN_khi_hoc():
    """Neu g khong bao gio doi thi phep do 'g tang khi dich nho' vo nghia ngay tu goc."""
    m = make()
    m.enable_latent_gate(mode="scalar", init_logit=0.0)
    g_dau = m.gate_value()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.05)
    b = batch(n=16)
    for _ in range(25):
        opt.zero_grad()
        loss, *_ = _phase2_target_loss(m, b, torch.device("cpu"), 0.0, gate_alpha=0.3)
        loss.backward()
        opt.step()
    assert abs(m.gate_value() - g_dau) > 1e-3, f"g dung yen o {g_dau}"


# ---------------------------------------------------------------- vong doi checkpoint
def test_checkpoint_di_va_ve():
    """Luu mo hinh CO cong, dung lai mo hinh KHONG cong, nap lai phai duoc."""
    from src.train_transfer import load_checkpoint

    m = make()
    m.enable_latent_gate(mode="scalar")
    with torch.no_grad():
        m.gate_logit.fill_(1.25)
    p = Path("/tmp/_test_gate_ckpt.pt")
    torch.save({"model_state_dict": m.state_dict()}, p)

    m2 = make()                                   # chua co cong
    assert not hasattr(m2, "lat_vul_head")
    load_checkpoint(str(p), m2, torch.device("cpu"))
    assert m2.gate_mode == "scalar"
    assert abs(m2.gate_value() - float(torch.sigmoid(torch.tensor(1.25)))) < 1e-6

    # chieu nguoc: checkpoint KHONG co cong thi khong duoc tu bat
    m3 = make()
    torch.save({"model_state_dict": m3.state_dict()}, p)
    m4 = make()
    load_checkpoint(str(p), m4, torch.device("cpu"))
    assert getattr(m4, "gate_mode", "off") == "off", "tu bat cong cho checkpoint khong co cong"
    p.unlink()


def test_so_tham_so_moi_la_NHO():
    """Luan diem cua co che la nhanh thu hai gan nhu khong the khop qua. Do lai cho chac."""
    m = make(num_latent=8, num_classes=2)
    m.enable_latent_gate(mode="scalar")
    n_gate = sum(p.numel() for n, p in m.named_parameters()
                 if p.requires_grad and ("lat_vul_head" in n or "gate_logit" in n))
    n_main = sum(p.numel() for p in m.vul_head.parameters())
    assert n_gate == 8 * 2 + 2 + 1, n_gate           # 8*C + C + 1 cong
    assert n_gate < n_main                            # 19 < 34 o backbone gia; 19 << 1538 o that


def test_khong_phai_latent_bottleneck_thi_TU_CHOI():
    torch.manual_seed(0)
    m = TransferModel(TinyBackbone(), num_classes=2, num_cwes=4, aux_mode="none", pooling="cls")
    with raises(ValueError):
        m.enable_latent_gate()


def test_doi_chung_chieu_ngau_nhien():
    """proj='random' phai THAY THAT ma tran chieu, va phai khac han ban da hoc."""
    m = make()
    hoc = [p.detach().clone() for p in m.latent_proj.parameters()]
    m.enable_latent_gate(mode="scalar", proj="random", proj_seed=7)
    assert m.gate_proj_kind == "random"
    W = m.latent_proj.weight.detach()
    assert not torch.allclose(W, hoc[0]), "proj='random' ma ma tran khong doi"
    assert torch.allclose(m.latent_proj.bias.detach(), torch.zeros_like(m.latent_proj.bias))
    assert all(not p.requires_grad for p in m.latent_proj.parameters())
    # CUNG seed => CUNG ma tran; KHAC seed => KHAC. Ca hai chieu.
    m2 = make(); m2.enable_latent_gate(mode="scalar", proj="random", proj_seed=7)
    assert torch.allclose(W, m2.latent_proj.weight.detach())
    m3 = make(); m3.enable_latent_gate(mode="scalar", proj="random", proj_seed=8)
    assert not torch.allclose(W, m3.latent_proj.weight.detach())
    # chieu nguoc: mac dinh 'learned' KHONG duoc dong vao ma tran
    m4 = make()
    hoc4 = [p.detach().clone() for p in m4.latent_proj.parameters()]
    m4.enable_latent_gate(mode="scalar")
    assert m4.gate_proj_kind == "learned"
    for a, b_ in zip(hoc4, m4.latent_proj.parameters()):
        assert torch.equal(a, b_.detach())


def test_nhom_lr_rieng_cho_cong():
    """Cong o lr cua backbone thi DUNG YEN — do that 11/09, g = 0.5000 sau ca hai epoch."""
    from src.train_transfer import split_gate_param_groups

    m = make()
    m.enable_latent_gate(mode="scalar")
    nt = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    groups, n_gate = split_gate_param_groups(nt, 2e-5, 1e-2, 0.01)
    assert len(groups) == 2 and n_gate == 8 * 2 + 2 + 1, (len(groups), n_gate)
    assert groups[0]["lr"] == 2e-5 and groups[1]["lr"] == 1e-2
    # moi tham so xuat hien DUNG MOT lan — khong sot, khong lap
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(nt), (len(ids), len(set(ids)), len(nt))

    # chieu nguoc: khong co cong thi DUNG MOT nhom, y nhu duong cu
    m2 = make()
    nt2 = [(n, p) for n, p in m2.named_parameters() if p.requires_grad]
    g2, n2 = split_gate_param_groups(nt2, 2e-5, 1e-2, 0.01)
    assert len(g2) == 1 and n2 == 0


def test_cong_DUNG_YEN_o_lr_backbone():
    """Tai hien dung loi da gap: cung so buoc, lr 2e-5 thi g khong nhuc nhich; lr 1e-2 thi co."""
    b = batch(n=16)
    moves = {}
    for lr in (2e-5, 1e-2):
        m = make()
        m.enable_latent_gate(mode="scalar", init_logit=0.0)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
        g0 = m.gate_value()
        for _ in range(60):
            opt.zero_grad()
            loss, *_ = _phase2_target_loss(m, b, torch.device("cpu"), 0.0, gate_alpha=0.3)
            loss.backward()
            opt.step()
        moves[lr] = abs(m.gate_value() - g0)
    assert moves[2e-5] < 0.01, f"lr 2e-5 dich {moves[2e-5]:.4f} — phep kiem nay khong con y nghia"
    assert moves[1e-2] > 10 * moves[2e-5], f"lr 1e-2 dich {moves[1e-2]:.4f} vs {moves[2e-5]:.4f}"


# ---------------------------------------------------------------- chay
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fail = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception:
            fail += 1
            print(f"  HONG {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - fail}/{len(tests)} phep dat")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
