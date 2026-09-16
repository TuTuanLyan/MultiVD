#!/usr/bin/env python3
"""NHIEU CUA SO (multi-window) — trien khai theo DUNG §B.3 cua DE_XUAT_1_LATE_FUSION.md.

Moi ham -> toi da K cua so `window` token, buoc `stride`; encoder dung chung; vector cua so =
CLS + embedding chi so cua so + chieu vi tri tuong doi; mean-pool co mat na; head 2 lop.

Co chuan cua de xuat:
  --window 510 --stride 384 --max_windows 8 --batch_size 4 --eval_batch_size 8 --micro 16
  --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.10 --epochs 30 --min_epochs 3
  --patience 8 --selection_metric roc_auc --agg mean --sam_rho 0.02

GHI CHU QUAN TRONG tu chinh de xuat (dong 449): tren SVEN co 30,5% ham > 510 token, trung binh
1,58 cua so/ham, va **hieu ung cua so thuan ~ 0 (§131)** — phan tang cua MW target-only den tu
**batch 4 (+2,8)** va **SAM (+1,9)**. Nen khi doc ket qua phai nho MW o day di kem batch 4 + SAM;
muon tach rieng hieu ung cua so thi chay `--max_windows 1` voi DUNG cac co con lai.

Ket qua ghi ra dung dinh dang cua repo (co `val_probabilities` / `test_probabilities`) de
`tools/late_fusion_gate.py` va cac cong cu khac dung duoc ngay.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_backbone, pool_hidden_states          # noqa: E402
from sam import SAMStep                                       # noqa: E402
from evaluate import find_best_threshold                      # noqa: E402
from transformers import AutoTokenizer                        # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score  # noqa: E402


# ----------------------------------------------------------------- du lieu
class WindowDataset(Dataset):
    """Moi ham -> [K, max_length] input_ids/attention_mask + mat na cua so [K]."""

    def __init__(self, records, tokenizer, window, stride, max_windows, max_length=512):
        self.records, self.tok = records, tokenizer
        self.window, self.stride, self.K, self.max_length = window, stride, max_windows, max_length
        self.pad = tokenizer.pad_token_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        ids = self.tok(r["code"], add_special_tokens=False, truncation=False,
                       return_attention_mask=False, verbose=False)["input_ids"]
        if len(ids) <= self.window:
            starts = [0]
        else:
            starts = list(range(0, len(ids) - self.window + self.stride, self.stride))
        starts = starts[: self.K]
        wins = [ids[s: s + self.window] for s in starts]
        input_ids = torch.full((self.K, self.max_length), self.pad, dtype=torch.long)
        attn = torch.zeros((self.K, self.max_length), dtype=torch.long)
        wmask = torch.zeros(self.K, dtype=torch.bool)
        for k, w in enumerate(wins):
            seq = self.tok.build_inputs_with_special_tokens(w)[: self.max_length]
            input_ids[k, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[k, : len(seq)] = 1
            wmask[k] = True
        pos = torch.zeros(self.K, dtype=torch.float)
        for k, s in enumerate(starts):
            pos[k] = s / max(1, len(ids) - 1)
        return {"input_ids": input_ids, "attention_mask": attn, "window_mask": wmask,
                "window_pos": pos, "labels": torch.tensor(int(r["label"]), dtype=torch.long)}


# ----------------------------------------------------------------- mo hinh
class MultiWindowModel(nn.Module):
    def __init__(self, backbone, max_windows, agg="mean", agg_layers=2, agg_heads=8,
                 dropout=0.1, pooling="cls"):
        super().__init__()
        self.backbone, self.pooling, self.agg_kind = backbone, pooling, agg
        H = backbone.config.hidden_size
        self.win_pos_emb = nn.Embedding(max_windows, H)
        self.rel_pos = nn.Linear(1, H)
        if agg == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=H, nhead=agg_heads, dim_feedforward=4 * 256,
                                               dropout=dropout, batch_first=True, activation="gelu")
            self.agg = nn.TransformerEncoder(layer, num_layers=agg_layers)
        elif agg == "mean":
            self.agg = None
        else:
            raise ValueError(agg)
        self.dropout = nn.Dropout(dropout)
        self.vul_head = nn.Linear(H, 2)

    def encode_windows(self, input_ids, attention_mask, window_mask, micro=16):
        B, K, L = input_ids.shape
        flat_idx = window_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        ids = input_ids.view(B * K, L)[flat_idx]
        am = attention_mask.view(B * K, L)[flat_idx]
        outs = []
        for s in range(0, ids.size(0), micro):
            o = self.backbone(input_ids=ids[s: s + micro], attention_mask=am[s: s + micro])
            outs.append(pool_hidden_states(o.last_hidden_state, am[s: s + micro], self.pooling))
        h = torch.zeros(B * K, self.vul_head.in_features, device=input_ids.device, dtype=outs[0].dtype)
        h[flat_idx] = torch.cat(outs, 0)
        return h.view(B, K, -1)

    def forward(self, input_ids, attention_mask, window_mask, window_pos, micro=16):
        h = self.encode_windows(input_ids, attention_mask, window_mask, micro)
        B, K, _ = h.shape
        h = h + self.win_pos_emb(torch.arange(K, device=h.device)).unsqueeze(0) \
              + self.rel_pos(window_pos.unsqueeze(-1))
        if self.agg is not None:
            h = self.agg(h, src_key_padding_mask=~window_mask)
        m = window_mask.unsqueeze(-1).to(h.dtype)
        z = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return {"vul_logits": self.vul_head(self.dropout(z))}


# ----------------------------------------------------------------- chay
def load_split(root, fold, split):
    return [json.loads(l) for l in open(Path(root) / f"fold{fold}" / f"{split}.jsonl")]


@torch.no_grad()
def infer(model, dl, device, micro):
    model.eval()
    P, Y = [], []
    for b in dl:
        out = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                    b["window_mask"].to(device), b["window_pos"].to(device), micro)
        P.append(torch.softmax(out["vul_logits"], -1)[:, 1].float().cpu().numpy())
        Y.append(b["labels"].numpy())
    return np.concatenate(Y), np.concatenate(P)


def metrics(y, p, thr):
    return {"macro_f1": f1_score(y, (p >= thr).astype(int), average="macro"),
            "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
            "pr_auc": average_precision_score(y, p) if len(set(y)) > 1 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--method_name", default="mw")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", default="data/sven_python_folds_norm")
    ap.add_argument("--model_name", default="microsoft/codebert-base")
    ap.add_argument("--pooling", default="cls")
    # co chuan cua de xuat
    ap.add_argument("--window", type=int, default=510)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--max_windows", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--agg", default="mean", choices=("mean", "transformer"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min_epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--selection_metric", default="roc_auc", choices=("roc_auc", "macro_f1"))
    ap.add_argument("--sam_rho", type=float, default=0.02)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_train_samples", type=int)
    ap.add_argument("--output_dir", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = MultiWindowModel(build_backbone(a.model_name), a.max_windows,
                             agg=a.agg, pooling=a.pooling).to(dev)

    tr = load_split(a.data_root, a.fold, "train")
    if a.max_train_samples:
        tr = tr[: a.max_train_samples]
    va = load_split(a.data_root, a.fold, "val")
    te = load_split(a.data_root, a.fold, "test")
    mk = lambda rows: WindowDataset(rows, tok, a.window, a.stride, a.max_windows, a.max_length)
    dtr = DataLoader(mk(tr), batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, drop_last=False)
    dva = DataLoader(mk(va), batch_size=a.eval_batch_size, shuffle=False, num_workers=a.num_workers)
    dte = DataLoader(mk(te), batch_size=a.eval_batch_size, shuffle=False, num_workers=a.num_workers)

    nw = [len(WindowDataset([r], tok, a.window, a.stride, a.max_windows, a.max_length)[0]["window_mask"].nonzero())
          for r in te[:80]]
    print(f"[mw] K={a.max_windows} window={a.window} stride={a.stride} | "
          f"cua so/ham tren 80 hang test: TB {np.mean(nw):.2f}, >1 cua so: {100*np.mean(np.array(nw)>1):.1f}%", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    total = max(1, len(dtr) * a.epochs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.learning_rate, total_steps=total,
                                              pct_start=a.warmup_ratio, anneal_strategy="linear")
    lossf = nn.CrossEntropyLoss()
    sam = SAMStep(a.sam_rho) if a.sam_rho > 0 else None
    params = [p for p in model.parameters() if p.requires_grad]

    best, best_ep, bad, best_state = -1.0, 0, 0, None
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for b in dtr:
            ids, am = b["input_ids"].to(dev), b["attention_mask"].to(dev)
            wm, wp, y = b["window_mask"].to(dev), b["window_pos"].to(dev), b["labels"].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(ids, am, wm, wp, a.micro)["vul_logits"], y)
            loss.backward()
            if sam is not None and sam.ascend(params):
                opt.zero_grad(set_to_none=True)
                lossf(model(ids, am, wm, wp, a.micro)["vul_logits"], y).backward()
                sam.restore(params)
            torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
            opt.step(); sch.step(); tot += float(loss.item()) * y.size(0)
        yv, pv = infer(model, dva, dev, a.micro)
        thr, _ = find_best_threshold(yv.tolist(), pv.tolist())   # ham tra ve (nguong, f1), khong phai mot so
        mv = metrics(yv, pv, 0.5); mv_cal = metrics(yv, pv, thr)
        score = mv["roc_auc"] if a.selection_metric == "roc_auc" else mv_cal["macro_f1"]
        print(f"[mw] epoch {ep:2d} | loss {tot/len(tr):.4f} | val ROC {mv['roc_auc']:.4f} "
              f"| val mF1@cal {mv_cal['macro_f1']:.4f} | {time.time()-t0:.0f}s", flush=True)
        if score > best:
            best, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr, best_yv, best_pv = thr, yv, pv
        else:
            bad += 1
            if ep >= a.min_epochs and bad >= a.patience:
                print(f"[mw] dung som o epoch {ep} (best {best_ep})", flush=True); break

    model.load_state_dict(best_state)
    yt, pt = infer(model, dte, dev, a.micro)
    m05, mcal = metrics(yt, pt, 0.5), metrics(yt, pt, best_thr)
    out = Path(a.output_dir or f"results/{a.run_name}/{a.method_name}") / f"seed_{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "experiment_name": f"{a.run_name}/{a.method_name}", "fold": a.fold, "seed": a.seed,
        "phase": "test", "best_epoch": best_ep, "val_calibrated_threshold": float(best_thr),
        "val_labels": [int(x) for x in best_yv], "val_probabilities": [round(float(x), 6) for x in best_pv],
        "test_labels": [int(x) for x in yt], "test_probabilities": [round(float(x), 6) for x in pt],
        "test_macro_f1_at_0.5": m05["macro_f1"], "test_macro_f1_at_valcal": mcal["macro_f1"],
        "test_roc_auc": m05["roc_auc"], "test_pr_auc": m05["pr_auc"],
        "best_val_macro_f1_at_0.5": metrics(best_yv, best_pv, 0.5)["macro_f1"],
        "source_checkpoint": None, "target_checkpoint": None,
        "hyperparameters": vars(a) | {"architecture": "codebert_multiwindow"},
    }
    (out / f"fold{a.fold}.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    print(f"[mw] XONG fold {a.fold} | test mF1@0.5 {m05['macro_f1']:.4f} | ROC {m05['roc_auc']:.4f} "
          f"| PR {m05['pr_auc']:.4f} -> {out}/fold{a.fold}.json", flush=True)


if __name__ == "__main__":
    main()
