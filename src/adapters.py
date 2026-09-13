"""AdapterFusion (Pfeiffer et al. 2020, arXiv:2005.00247) cho pipeline hai pha cua du an.

Y TUONG. Pha 1 hoc MOT adapter cho task nguon (ccpp+js). Pha 2 them MOT adapter nua cho
task dich (python) va hoc mot lop FUSION tron hai adapter do — adapter nguon **dong bang**.
Vi the tri thuc nguon khong bi ghi de (bai goi la "non-destructive"), khac han fine-tune
tuan tu von ghi de len chinh trong so da hoc.

DAT O DAU. Adapter chen sau MOI lop transformer (12 lop), dung nhu bai bao — fusion hoc
trong so tron RIENG cho tung lop. Cach boc o day la boc CA LOP (`AdapterBlock`) thay vi
sua rieng tung lop RobertaLayer/T5Block: ca hai ho deu tra ve tuple co hidden state o vi
tri 0, nen mot lop boc duy nhat chay dung cho ca codebert (RobertaModel) lan codet5p
(T5Stack) ma khong phai viet hai duong ma.

KHOI TAO GAN-DONG-NHAT. `up` khoi tao bang 0 nen luc bat dau adapter tra ve dung dau vao:
mo hinh khoi dong tai DUNG diem pretrained, khong bi giat. Neu khong lam vay thi Pha 1
phai tieu vai tram buoc chi de go lai cai nhieu do tu gay ra.
"""
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Nut that bottleneck: h + up(GELU(down(h))).

    Residual nam BEN TRONG adapter, nen dau ra dung duoc truc tiep lam hidden state moi.
    """

    def __init__(self, hidden_size, dim=48, init_std=1e-3):
        super().__init__()
        self.down = nn.Linear(hidden_size, dim)
        self.up = nn.Linear(dim, hidden_size)
        self.act = nn.GELU()
        nn.init.normal_(self.down.weight, std=init_std)
        nn.init.zeros_(self.down.bias)
        # up = 0 => adapter la anh xa dong nhat o buoc 0.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(self.act(self.down(h)))


class AdapterFusion(nn.Module):
    """Attention tron nhieu adapter — muc 3 cua arXiv:2005.00247.

        Query = hidden state cua lop (h);  Key, Value = dau ra tung adapter (z_n)
        alpha = softmax_n( <W_Q h, W_K z_n> );   out = sum_n alpha_n * W_V z_n

    `W_V` khoi tao bang DON VI (cong nhieu rat nho) dung theo bai: luc bat dau fusion gan
    nhu chi lay trung binh co trong so cua chinh cac z_n, chua bop meo gi.
    """

    def __init__(self, hidden_size, value_init_std=1e-6, temperature=1.0):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.temperature = temperature
        with torch.no_grad():
            self.value.weight.copy_(
                torch.eye(hidden_size) + torch.randn(hidden_size, hidden_size) * value_init_std)
        self.last_weights = None   # (B, T, N) — de doc xem fusion nghieng ve adapter nao

    def forward(self, h, zs):
        z = torch.stack(zs, dim=2)                          # (B, T, N, H)
        q = self.query(h)                                   # (B, T, H)
        k = self.key(z)                                     # (B, T, N, H)
        # einsum thay cho (q.unsqueeze(2) * k).sum(-1): tranh vat chat hoa mot tensor
        # (B, T, N, H) trung gian chi de roi cong lai theo H.
        scores = torch.einsum("bth,btnh->btn", q, k) / self.temperature
        w = scores.softmax(dim=-1)
        self.last_weights = w.detach()
        # `value` TUYEN TINH nen  sum_n a_n * W_V(z_n) == W_V( sum_n a_n * z_n ).
        # Gop TRUOC roi moi chieu: bo han mot tensor (B, T, N, H) moi lop. Dung ve toan hoc,
        # chi khac thu tu phep cong dau phay dong (~1e-7, duoi san nhieu 0.010 nam bac).
        # Can that: o batch 16 x len 512, t5p-220m OOM o 15.40/15.49 GiB neu khong lam vay.
        return self.value((w.unsqueeze(-1) * z).sum(dim=2))  # (B, T, H)


class AdapterBlock(nn.Module):
    """Boc MOT lop transformer: chay lop that, roi dat adapter/fusion len hidden state.

    Giu nguyen phan con lai cua tuple tra ve. Voi T5 dieu nay BAT BUOC: T5Stack doc
    `layer_outputs[2]` lam `position_bias` cho cac lop sau, nen nuot mat duoi tuple la
    hong ca mo hinh ma khong bao loi.
    """

    def __init__(self, layer, hidden_size, names, dim=48):
        super().__init__()
        self.layer = layer
        self.adapters = nn.ModuleDict({n: Adapter(hidden_size, dim) for n in names})
        self.fusion = None
        self.active = list(names)

    def add_adapter(self, name, hidden_size, dim=48):
        self.adapters[name] = Adapter(hidden_size, dim)
        if name not in self.active:
            self.active.append(name)

    def enable_fusion(self, hidden_size, **kw):
        self.fusion = AdapterFusion(hidden_size, **kw)

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        was_tuple = isinstance(out, tuple)
        h = out[0] if was_tuple else out
        zs = [self.adapters[n](h) for n in self.active]
        if self.fusion is not None:
            h = self.fusion(h, zs)
        elif len(zs) == 1:
            h = zs[0]
        else:
            h = torch.stack(zs, dim=0).mean(0)
        return (h,) + tuple(out[1:]) if was_tuple else h


def encoder_layers(backbone):
    """ModuleList cac lop transformer. RoBERTa: encoder.layer. T5: block."""
    enc = getattr(backbone, "encoder", backbone)
    layers = getattr(enc, "layer", None)
    if layers is None:
        layers = getattr(enc, "block", None)
    if layers is None:
        raise RuntimeError(f"khong tim thay danh sach lop trong {type(backbone).__name__}")
    return layers


def inject_adapters(backbone, names=("src",), dim=48):
    """Boc moi lop bang AdapterBlock. Goi LAI duoc: lop da boc thi chi them adapter."""
    layers = encoder_layers(backbone)
    hidden = backbone.config.hidden_size
    for i, layer in enumerate(layers):
        if isinstance(layer, AdapterBlock):
            for n in names:
                if n not in layer.adapters:
                    layer.add_adapter(n, hidden, dim)
        else:
            layers[i] = AdapterBlock(layer, hidden, names, dim)
    return {"layers": len(layers), "names": list(names), "dim": dim}


def enable_fusion(backbone, **kw):
    n = 0
    for layer in encoder_layers(backbone):
        if isinstance(layer, AdapterBlock):
            layer.enable_fusion(backbone.config.hidden_size, **kw)
            n += 1
    return {"fusion_layers": n}


def adapter_blocks(backbone):
    return [l for l in encoder_layers(backbone) if isinstance(l, AdapterBlock)]


def randomize_adapter(backbone, name, seed=0, match_scale=True):
    """Thay adapter `name` bang NHIEU GAUSS cung thang do — doi chung bat buoc cua khoi fusion.

    VI SAO CAN. Nhanh `fusft` them ~22M tham so fusion + 1,8M tham so adapter so voi doi
    chung. Loi ich do co the den tu THEM SUC CHUA chu khong tu tri thuc cua Pha 1, va hai
    kha nang do KHONG phan biet duoc neu khong co phep doi chung nay. Cung ly do §44 bat
    buoc doi chung "chieu ngau nhien" cho nut that 8 chieu.

    VI SAO PHAI KHOP THANG DO. Neu dung lai khoi tao goc (`up = 0`) thi adapter nguon thanh
    anh xa DONG NHAT — phep do khi do chi tra loi "khong co adapter nguon thi sao", chua
    loai duoc gia thuyet "mot phep bien doi co do lon nhu the, NOI DUNG GI CUNG DUOC, cung
    giup". Khop do lech chuan tung tensor giu nguyen DO LON, pha huy NOI DUNG — do moi dung
    la gia thuyet khong can bac.

    Seed rieng cho tung fold (goi ben ngoai cong fold vao) de ket qua khong cuoc vao mot lan
    boc bai may rui.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    changed = []
    for blk in adapter_blocks(backbone):
        if name not in blk.adapters:
            continue
        for pname, prm in blk.adapters[name].named_parameters():
            with torch.no_grad():
                if match_scale:
                    std = float(prm.detach().float().std())
                    mean = float(prm.detach().float().mean())
                else:
                    std, mean = 0.02, 0.0
                if std == 0.0:
                    # Tensor hang (vi du bias toan 0): nhieu quanh 0 se la mot thien lech
                    # NGAU NHIEN ma ban da hoc khong co => bo qua, giu nguyen.
                    continue
                noise = torch.randn(prm.shape, generator=g, dtype=torch.float32) * std + mean
                prm.copy_(noise.to(prm.dtype).to(prm.device))
            changed.append(pname)
    return {"adapter": name, "tensors": len(changed), "seed": int(seed),
            "match_scale": bool(match_scale)}


def spec_from_state_dict(state_dict, prefix="backbone."):
    """Doc LAI hinh dang adapter/fusion tu state dict cua checkpoint.

    Phai dung cach nay chu khong dung `strict=False`: `strict=False` nuot im moi khoa
    khong khop, nen mot checkpoint CO adapter nap vao mo hinh KHONG adapter van "thanh
    cong" — va Pha 2 chay tiep tren mot mo hinh thieu han phan da hoc o Pha 1.
    """
    names, dim, fusion = [], None, False
    for k, v in state_dict.items():
        if ".adapters." not in k:
            if ".fusion." in k:
                fusion = True
            continue
        nm = k.split(".adapters.")[1].split(".")[0]
        if nm not in names:
            names.append(nm)
        if k.endswith(".down.weight"):
            dim = v.shape[0]
    return {"names": names, "dim": dim, "fusion": fusion}


ADAPTER_MARKERS = (".adapters.", ".fusion.")


def is_adapter_param(name):
    return any(m in name for m in ADAPTER_MARKERS)


def set_trainable(model, train_backbone, train_adapters=("tgt",), train_fusion=True):
    """Bat/tat requires_grad theo DUNG hai bien the Pha 2 cua khoi nay.

    CHI dung toi backbone, adapter va fusion. Moi thu khac — `vul_head`, `latent_proj`,
    `cwe_head` — GIU NGUYEN trang thai da co.

    Vi sao khong "mo het cac head": ban dau ham nay co tham so `train_heads=True` va no
    MO LAI ca head phu, tuc huy bo `freeze_aux_head()` da goi truoc do. Pha 2 chet ngay o
    `assert_recadam_setup`: "auxiliary head not frozen". Bat duoc bang phep thu khoi 32
    mau tren CPU, truoc khi tieu mot giay GPU nao — neu khong, hoac no chet giua khoi,
    hoac te hon la mot nhanh nao do khong co assert do se chay tiep voi head phu DANG HOC
    va phep so am tham doi hai bien.

    Tra ve so tham so mo/khoa de goi ben ngoai DEM va doi chieu — `requires_grad = False`
    viet ra ma khong ai kiem thi khong phai bang chung.
    """
    stats = {"trainable": 0, "frozen": 0, "untouched": 0}
    for name, p in model.named_parameters():
        if is_adapter_param(name):
            if ".fusion." in name:
                on = train_fusion
            else:
                nm = name.split(".adapters.")[1].split(".")[0]
                on = nm in train_adapters
            p.requires_grad = bool(on)
        elif name.startswith("backbone."):
            on = bool(train_backbone)
            p.requires_grad = on
        else:
            on = bool(p.requires_grad)      # GIU NGUYEN
            stats["untouched"] += p.numel()
        stats["trainable" if on else "frozen"] += p.numel()
    return stats
