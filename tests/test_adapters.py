"""Kiem AdapterFusion. Moi phep deu thu CA HAI CHIEU: mot truong hop phai DOI va mot
truong hop phai KHONG DOI. Cong chi bao "khong doi" thi khong chung minh duoc gi —
mot adapter chet cung cho ket qua nhu the.

Chay: HF_HUB_OFFLINE=1 python tests/test_adapters.py
Dung hai model ti hon cua HF nen chay tren CPU trong vai giay, khong dung GPU.
"""
import os, sys, traceback
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from adapters import (Adapter, AdapterBlock, AdapterFusion, adapter_blocks,  # noqa: E402
                      encoder_layers, enable_fusion, inject_adapters,
                      set_trainable, spec_from_state_dict)

ROBERTA = "hf-internal-testing/tiny-random-roberta"
T5 = "hf-internal-testing/tiny-random-t5"


def _backbone(name):
    from transformers import AutoConfig, AutoModel, T5EncoderModel
    cfg = AutoConfig.from_pretrained(name)
    if getattr(cfg, "is_encoder_decoder", False):
        return T5EncoderModel.from_pretrained(name)
    return AutoModel.from_pretrained(name)


def _ids(bb):
    v = min(bb.config.vocab_size, 100)
    return torch.randint(1, v, (2, 7)), torch.ones(2, 7, dtype=torch.long)


def _out(bb, ids, mask):
    with torch.no_grad():
        return bb(input_ids=ids, attention_mask=mask).last_hidden_state


def test_khoi_tao_la_anh_xa_dong_nhat(name=None):
    """CHIEU 1: adapter moi chen vao KHONG duoc doi dau ra (up khoi tao = 0)."""
    bb = _backbone(name).eval()
    ids, mask = _ids(bb)
    before = _out(bb, ids, mask)
    inject_adapters(bb, names=("src",), dim=4)
    after = _out(bb.eval(), ids, mask)
    assert torch.allclose(before, after, atol=1e-6), "adapter luc khoi tao da lam lech dau ra"


def test_adapter_co_tac_dung_khi_trong_so_khac_khong(name=None):
    """CHIEU 2: nhieu trong so adapter thi dau ra PHAI doi — neu khong, no bi bo qua."""
    bb = _backbone(name).eval()
    ids, mask = _ids(bb)
    inject_adapters(bb, names=("src",), dim=4)
    before = _out(bb, ids, mask)
    with torch.no_grad():
        for blk in adapter_blocks(bb):
            blk.adapters["src"].up.weight.normal_(0, 0.5)
    after = _out(bb, ids, mask)
    assert not torch.allclose(before, after, atol=1e-5), "adapter khong anh huong gi toi dau ra"


def test_boc_lop_khong_lam_hong_tuple(name=None):
    """T5Stack doc layer_outputs[2] lam position_bias. Nuot duoi tuple la hong am tham:
    mo hinh van chay, van ra so, chi la sai. Kiem so lop duoc boc va forward song."""
    bb = _backbone(name).eval()
    n_layers = len(encoder_layers(bb))
    info = inject_adapters(bb, names=("src",), dim=4)
    assert info["layers"] == n_layers
    assert len(adapter_blocks(bb)) == n_layers
    ids, mask = _ids(bb)
    assert _out(bb, ids, mask).shape[:2] == ids.shape


def test_fusion_hai_adapter_va_trong_so_cong_bang_mot(name=None):
    bb = _backbone(name).eval()
    inject_adapters(bb, names=("src",), dim=4)
    inject_adapters(bb, names=("tgt",), dim=4)      # goi lai: them, khong boc lai
    assert set(adapter_blocks(bb)[0].adapters.keys()) == {"src", "tgt"}
    enable_fusion(bb)
    ids, mask = _ids(bb)
    _out(bb, ids, mask)
    w = adapter_blocks(bb)[0].fusion.last_weights
    assert w.shape[-1] == 2
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_fusion_luc_dau_gan_bang_trung_binh_hai_adapter():
    """W_V khoi tao = don vi => fusion luc dau ~ tron tuyen tinh cac z_n, chua bop meo."""
    torch.manual_seed(0)
    f = AdapterFusion(16)
    h = torch.randn(2, 5, 16)
    z1, z2 = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    out = f(h, [z1, z2])
    w = f.last_weights
    expect = w[..., 0:1] * z1 + w[..., 1:2] * z2
    assert torch.allclose(out, expect, atol=1e-3)


def test_gop_truoc_roi_chieu_tuong_duong_chieu_roi_gop():
    """`AdapterFusion.forward` gop  sum_n a_n*z_n  TRUOC roi moi qua W_V, thay vi chieu tung
    z_n roi cong — de bo mot tensor (B,T,N,H) moi lop (t5p OOM neu khong lam vay, 13/09).

    Phep bien doi do dung vi W_V TUYEN TINH. Kiem thang: so dau ra that voi ban tham chieu
    viet theo dang cu. Neu ai do them bias hay phi tuyen vao `value`, phep nay se hong ngay
    — do chinh la dieu can mot bai kiem, chu khong phai mot dong chu thich."""
    torch.manual_seed(0)
    f = AdapterFusion(24)
    h = torch.randn(2, 5, 24)
    zs = [torch.randn(2, 5, 24) for _ in range(3)]
    got = f(h, zs)
    w = f.last_weights
    # ban THAM CHIEU: chieu tung adapter roi moi cong
    ref = sum(w[..., n:n + 1] * f.value(zs[n]) for n in range(len(zs)))
    assert torch.allclose(got, ref, atol=1e-5), \
        f"hai dang KHONG tuong duong, lech max {float((got - ref).abs().max()):.2e}"


def test_spec_doc_lai_dung_hinh_dang_tu_state_dict():
    bb = _backbone(ROBERTA)
    inject_adapters(bb, names=("src",), dim=6)
    spec = spec_from_state_dict(bb.state_dict())
    assert spec["names"] == ["src"] and spec["dim"] == 6 and spec["fusion"] is False
    enable_fusion(bb)
    inject_adapters(bb, names=("tgt",), dim=6)
    spec2 = spec_from_state_dict(bb.state_dict())
    assert set(spec2["names"]) == {"src", "tgt"} and spec2["fusion"] is True


class _Model(torch.nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.backbone = bb
        self.vul_head = torch.nn.Linear(bb.config.hidden_size, 2)


def test_hai_bien_the_pha2_dong_bang_dung_cho():
    """Bien the A: backbone MO, adapter nguon KHOA. Bien the B: backbone KHOA luon.
    Ca hai deu phai khoa adapter `src`, va deu phai mo `tgt` + fusion + head."""
    bb = _backbone(ROBERTA)
    inject_adapters(bb, names=("src", "tgt"), dim=4)
    enable_fusion(bb)
    m = _Model(bb)
    # gia lap `freeze_aux_head`: head phu da bi dong bang TRUOC khi goi set_trainable
    m.aux_head = torch.nn.Linear(bb.config.hidden_size, 3)
    for prm in m.aux_head.parameters():
        prm.requires_grad = False

    a = set_trainable(m, train_backbone=True)
    on_a = {n for n, p in m.named_parameters() if p.requires_grad}
    b = set_trainable(m, train_backbone=False)
    on_b = {n for n, p in m.named_parameters() if p.requires_grad}

    # CHIEU 1 — cai phai KHOA thi khoa, o CA HAI bien the
    for on in (on_a, on_b):
        assert not any(".adapters.src." in n for n in on), "adapter nguon bi mo — sai ca hai bien the"
    # CHIEU 2 — cai phai MO thi mo
    for on in (on_a, on_b):
        assert any(".adapters.tgt." in n for n in on)
        assert any(".fusion." in n for n in on)
        assert "vul_head.weight" in on
    # Khac nhau DUNG o backbone
    assert any(n.startswith("backbone.") and ".adapters." not in n and ".fusion." not in n
               for n in on_a), "bien the A phai mo backbone"
    assert not any(n.startswith("backbone.") and ".adapters." not in n and ".fusion." not in n
                   for n in on_b), "bien the B phai khoa backbone"
    assert a["trainable"] > b["trainable"]
    # CHIEU 3 — set_trainable KHONG duoc mo lai head phu da dong bang (loi da mac 13/09:
    # Pha 2 chet o assert_recadam_setup "auxiliary head not frozen")
    for on in (on_a, on_b):
        assert not any(n.startswith("aux_head.") for n in on), "head phu bi MO LAI"


def test_gradient_chi_chay_vao_phan_duoc_mo():
    """Dem requires_grad chua du — phai thay grad THUC SU khac None dung o cho mong doi."""
    bb = _backbone(ROBERTA)
    inject_adapters(bb, names=("src", "tgt"), dim=4)
    enable_fusion(bb)
    m = _Model(bb)
    set_trainable(m, train_backbone=False)
    ids, mask = _ids(bb)
    h = m.backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
    m.vul_head(h).sum().backward()
    g = {n: (p.grad is not None) for n, p in m.named_parameters()}
    assert any(g[n] for n in g if ".adapters.tgt." in n), "adapter dich khong nhan gradient"
    assert any(g[n] for n in g if ".fusion." in n), "fusion khong nhan gradient"
    assert not any(g[n] for n in g if ".adapters.src." in n), "adapter nguon VAN nhan gradient"


# ----------------------------------------------------------------------------
PARAMETRIZED = ("test_khoi_tao_la_anh_xa_dong_nhat",
                "test_adapter_co_tac_dung_khi_trong_so_khac_khong",
                "test_boc_lop_khong_lam_hong_tuple",
                "test_fusion_hai_adapter_va_trong_so_cong_bang_mot")


def main():
    fails = []
    g = dict(globals())
    for nm in sorted(n for n in g if n.startswith("test_")):
        fn = g[nm]
        cases = [(ROBERTA,), (T5,)] if nm in PARAMETRIZED else [()]
        for c in cases:
            lbl = f"{nm}{'[' + c[0].split('-')[-1] + ']' if c else ''}"
            try:
                fn(*c)
                print(f"  OK   {lbl}")
            except Exception as e:
                fails.append(lbl)
                print(f"  HONG {lbl}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n{'TAT CA DAT' if not fails else str(len(fails)) + ' PHEP HONG: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
