#!/usr/bin/env python3
"""Kiem neo khong gian dac trung HAI CHIEU truoc khi tin no (FACTS §36).

    python tests/test_feat_anchor.py

Rui ro that su cua co che nay KHONG phai cong thuc loss ma la DIA CHI HOA: bo dem dac trung
tra theo `batch["index"]`. Neu chi so do khong phai chi so hang trong dataset thi moi hang bi
ghep voi dac trung cua hang KHAC — loss van chay, van giam, van ra so dep, va phep neo bien
thanh NHIEU ma khong mot cong nao bao. Nen phep kiem 3 la phep kiem dat nhat o day.

1. KHOP:  mo hinh chua doi + eval mode -> khoang cach cosine ~ 0 (bo dem dung mo hinh do).
2. LECH:  lam nhieu trong so -> khoang cach > 0 ro rang.
3. DIA CHI: loader XAO TRON van tra dung dac trung cua dung hang (so voi bang tinh thang).
4. mse va cos deu 0 luc khop, va cos BAT BIEN khi nhan dac trung voi hang so (mse thi khong).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import CodeDataset  # noqa: E402


class FakeTok:
    pad_token_id = 0

    def __call__(self, text, **kw):
        return {"input_ids": [1 + (ord(c) % 40) for c in text[:12]]}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [101] + ids + [102]


class TinyModel(torch.nn.Module):
    """Backbone gia: embedding + mean pool. Du de kiem dia chi hoa va cong thuc."""

    def __init__(self, vocab=200, dim=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.vul_head = torch.nn.Linear(dim, 2)

    def forward(self, input_ids, attention_mask, return_cwe=False, return_features=False):
        h = self.emb(input_ids)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out = {"vul_logits": self.vul_head(pooled), "cwe_logits": None, "latent": None}
        if return_features:
            out["pooled"] = pooled
        return out


def make_records(n=37):
    return [{"code": f"row{i:03d}xyz", "label": i % 2, "cwe_class": i % 4} for i in range(n)]


def build_cache(model, ds, batch=8):
    """Ban rut gon cua train_transfer.build_feature_cache (cung phep dia chi hoa)."""
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    cache = None
    model.eval()
    with torch.no_grad():
        for b in loader:
            p = model(b["input_ids"], b["attention_mask"], return_features=True)["pooled"].float()
            if cache is None:
                cache = torch.zeros(len(ds), p.shape[-1])
            cache[b["index"]] = p
    return cache


def dist(cur, ref, mode="cos"):
    if mode == "mse":
        return F.mse_loss(cur, ref)
    return (1.0 - F.cosine_similarity(cur, ref, dim=-1)).mean()


def main():
    torch.manual_seed(0)
    recs = make_records()
    ds = CodeDataset(recs, FakeTok(), 16, "head")
    model = TinyModel()
    cache = build_cache(model, ds)

    # 1. KHOP
    model.eval()
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    d = []
    with torch.no_grad():
        for b in loader:
            cur = model(b["input_ids"], b["attention_mask"], return_features=True)["pooled"]
            d.append(dist(cur, cache[b["index"]]).item())
    assert max(d) < 1e-6, f"khop nhung khoang cach {max(d)}"
    print(f"1. KHOP    : d_cos max {max(d):.2e}                       OK")

    # 3. DIA CHI HOA — loader XAO TRON (lam truoc 2, vi 2 lam hong trong so)
    gen = torch.Generator().manual_seed(7)
    sh = DataLoader(ds, batch_size=5, shuffle=True, generator=gen)
    seen, bad = 0, 0
    with torch.no_grad():
        for b in sh:
            cur = model(b["input_ids"], b["attention_mask"], return_features=True)["pooled"]
            ref = cache[b["index"]]
            # bang tinh thang: tra tung hang qua dataset, khong qua bo dem
            truth = torch.stack([
                model(ds[int(i)]["input_ids"][None], ds[int(i)]["attention_mask"][None],
                      return_features=True)["pooled"][0]
                for i in b["index"]
            ])
            bad += int((~torch.isclose(ref, truth, atol=1e-6)).any(dim=-1).sum())
            seen += len(b["index"])
            assert dist(cur, ref).item() < 1e-6
    assert bad == 0, f"{bad}/{seen} hang bi ghep sai dac trung"
    print(f"3. DIA CHI : {seen} hang qua loader XAO TRON, 0 hang ghep sai        OK")

    # 4. cos BAT BIEN thang do, mse thi KHONG
    ref = cache[:8]
    assert dist(ref * 3.0, ref, "cos").item() < 1e-6
    assert dist(ref * 3.0, ref, "mse").item() > 1e-3
    print("4. cos bat bien thang do, mse thi khong                   OK")

    # 2. LECH
    with torch.no_grad():
        model.emb.weight.add_(torch.randn_like(model.emb.weight) * 0.5)
    d2 = []
    with torch.no_grad():
        for b in loader:
            cur = model(b["input_ids"], b["attention_mask"], return_features=True)["pooled"]
            d2.append(dist(cur, cache[b["index"]]).item())
    assert sum(d2) / len(d2) > 0.01, f"lam nhieu trong so ma khoang cach van {sum(d2)/len(d2)}"
    print(f"2. LECH    : sau khi lam nhieu, d_cos trung binh {sum(d2)/len(d2):.4f}   OK")
    print("TAT CA OK")


if __name__ == "__main__":
    main()
