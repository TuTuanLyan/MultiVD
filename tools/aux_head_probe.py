#!/usr/bin/env python3
"""aux_head_probe.py — head phu CO HOC DUOC GI KHONG? (FACTS §30.1)

Ca phuong phap dua tren gia thiet head phu day encoder mot tin hieu huu ich. Nhung do chinh
xac cua head do CHUA TUNG duoc ghi lai o dau: `src/train.py` khong log, checkpoint khong luu.
Khoa duy nhat ve chat luong trong checkpoint la `best_val_macro_f1`, va do la F1 cua dau ra
LO HONG NHI PHAN, khong phai cua head CWE.

Script nay chi doc: nap checkpoint Pha 1, dung LAI DUNG ham chia cua train_transfer.py (nen
tap val trung khop voi tap da chon checkpoint), roi do head phu tren do. So sanh voi hai san:
  - doan lop da so   (khong hoc gi)
  - doan ngau nhien theo phan phoi lop
KHONG huan luyen gi, khong ghi gi.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, "src")
import torch
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score

import train_transfer as T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default="precomputed", choices=("fixed4","source","precomputed"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_length", type=int, default=512)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck.get("training_args") or {}
    seed = int(ta.get("seed", 42))
    print(f"# checkpoint : {a.ckpt}")
    print(f"# backbone   : {ta.get('model_name')} / {ta.get('pooling')} | aux={ck.get('aux_mode')} "
          f"| num_cwes={ck.get('num_cwes')} | lambda={ta.get('lambda_cwe')} | seed={seed}")
    print(f"# val nhi phan da ghi trong checkpoint: {ck.get('best_val_macro_f1'):.4f} (ep {ck.get('best_epoch')})")

    records = T.load_jsonl(a.data, trust_precomputed=(a.vocab == "precomputed"))
    if a.vocab == "source":
        v = T.build_cwe_vocab(records); T.apply_cwe_vocab(records, v)
    _, val_records = T.split_source_records(records, seed)
    print(f"# du lieu    : {a.data} | {len(records)} hang -> val {len(val_records)} hang "
          f"(chia lai bang chinh ham cua train_transfer, seed {seed})")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ta.get("model_name"))
    loader = T.build_dataloader(val_records, tok, a.max_length, a.batch, False, seed, 0,
                                ta.get("truncation_strategy", "head_middle_tail"))

    args_ns = argparse.Namespace(**{**ta, "num_cwes": ck.get("num_cwes"),
                                    "aux_mode": ck.get("aux_mode"),
                                    "num_latent": ck.get("num_latent"),
                                    "pooling": ck.get("pooling")})
    model = T.make_model(ta.get("model_name"), a.device, args_ns)
    sd = ck.get("model_state_dict", ck.get("state_dict"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"# CANH BAO nap trong so: thieu {len(missing)}, thua {len(unexpected)}")
    model.to(a.device).eval()

    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["input_ids"].to(a.device), b["attention_mask"].to(a.device), return_cwe=True)
            logits = out[1] if isinstance(out, (tuple, list)) else out["cwe_logits"]
            ps.append(logits.argmax(-1).cpu().numpy())
            ys.append(b["cwe_class"].numpy() if "cwe_class" in b else b["cwe"].numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    keep = y >= 0
    y, p = y[keep], p[keep]
    print(f"# hang co nhan phu: {len(y)} (bo {int((~keep).sum())} hang -100)")

    K = int(ck.get("num_cwes"))
    cnt = Counter(y.tolist()); n = len(y)
    maj = max(cnt.values()) / n
    f1_maj = f1_score(y, np.full_like(y, cnt.most_common(1)[0][0]), average="macro", zero_division=0)
    rng = np.random.default_rng(0)
    pr = np.array([cnt.get(i,0)/n for i in range(K)])
    rand_acc = float((pr**2).sum())
    print()
    print(f"{'':22}{'do chinh xac':>14}{'macro-F1':>12}")
    print(f"{'HEAD PHU (do that)':22}{accuracy_score(y,p):>14.4f}{f1_score(y,p,average='macro',zero_division=0):>12.4f}")
    print(f"{'san: doan lop da so':22}{maj:>14.4f}{f1_maj:>12.4f}")
    print(f"{'san: doan ngau nhien':22}{rand_acc:>14.4f}{'-':>12}")
    print()
    print("phan bo du doan vs that (lop: that -> doan):")
    for c in sorted(set(y.tolist())):
        m = y == c
        print(f"  lop {c:>3}  n={int(m.sum()):>4}  dung {int((p[m]==c).sum()):>4}  "
              f"({(p[m]==c).mean():.2%})")
    print(f"  head chi dung {len(set(p.tolist()))} lop khac nhau tren {K} lop co the")

if __name__ == "__main__":
    main()
