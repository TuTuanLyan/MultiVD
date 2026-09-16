#!/usr/bin/env python3
"""Nhung toan bo tap dich bang backbone GOC (chua fine-tune) de dung cho ROUTER.

    python3 tools/embed_target.py --out emb_codebert.npz

VI SAO dung backbone GOC: router phai doc DOAN CODE, khong duoc doc dau ra cua hai chuyen gia
(thong tin "hang nay thuoc nhom nao" khong nam trong hai so vo huong do). Va dung backbone goc
thi router khong thien vi ben nao, cung khong can checkpoint Pha 2 (da bi xoa).
"""
import argparse, json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import CodeDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--model_name", default="microsoft/codebert-base")
ap.add_argument("--pooling", default="cls")
ap.add_argument("--root", default="data/sven_python_folds_norm")
ap.add_argument("--max_length", type=int, default=512)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model_name)
mod = AutoModel.from_pretrained(a.model_name).to(dev).eval()
print(f"device={dev} model={a.model_name}")

@torch.no_grad()
def emb(rows):
    dl = DataLoader(CodeDataset(rows, tok, a.max_length), batch_size=a.batch_size, shuffle=False)
    out = []
    for b in dl:
        h = mod(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev)).last_hidden_state
        if a.pooling == "cls":
            p = h[:, 0, :]
        else:
            m = b["attention_mask"].to(dev).unsqueeze(-1).to(h.dtype)
            p = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out, 0)

store = {}
for k in range(1, 6):
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(f"{a.root}/fold{k}/{split}.jsonl")]
        store[f"f{k}_{split}"] = emb(rows)
        print(f"  fold{k}/{split}: {store[f'f{k}_{split}'].shape}")
np.savez_compressed(a.out, **store)
print("da ghi", a.out)
