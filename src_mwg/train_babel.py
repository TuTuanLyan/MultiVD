#!/usr/bin/env python3
"""MW_ASSEMBLE_BABEL — thay CUA SO bang DONG LENH, cong hai DO THI DONG NHAT kieu BABEL.

Y tuong ghep: `assemble` cua De xuat 1 giu nguyen (hai model dong cung, tron xac suat tren
val); thu duy nhat thay doi la HOC VIEN CO SO — thay vi multi-window thi la BABEL:

    dong lenh  --(backbone dung chung)-->  h [N, H]
                      |
        +-------------+-------------+
   GCN x2 tren A_data          GCN x2 tren A_control      <- ca hai DONG NHAT (Kipf & Welling)
        +-------------+-------------+
                      |
          cat(h, data, control) -> LSTM -> mean-pool co mat na -> head 2 lop

Hai do thi dung o `src/babel_graph.py`, theo dung luat cua BABEL (ICSME 2024):
  A_data    : noi i-j neu hai dong dung chung mot dinh danh      -> DOI XUNG
  A_control : tu do thut le, ngan xep cha->con + anh em cung bac -> BAT DOI XUNG
Khong dung code parser nao, nen luat nay chay duoc tren moi ngon ngu giong C.

BA CHO DE SAI, da chan san:
  1. **Module moi can learning rate RIENG.** GCN/LSTM/head la tham so ngau nhien; o lr cua
     backbone (2e-5) chung gan nhu khong roi diem khoi tao va ket qua se doc thanh "co che
     khong an gi". Nen co `--graph_lr` (mac dinh 5e-4) va HAI nhom tham so.
  2. **`strict=False` nuot ca khoa THIEU.** `--init_ckpt` in ro so khoa chuyen/thieu/thua
     va DUNG HAN neu thieu qua 8.
  3. **Lech hang.** `babel_graph` co cong chan: neu viec bo chu thich lam doi so dong thi
     bao loi ngay, khong im lang noi nham dong.

Ket qua ghi dung dinh dang repo (co `val_probabilities`/`test_probabilities`) de
`tools/late_fusion_gate.py` va `tools/mw_n5_table.py` dung duoc ngay, khong sua gi.
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
from babel_graph import build as build_graphs                 # noqa: E402
from babel_graph import build_chunked                          # noqa: E402
from transformers import AutoTokenizer                        # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score  # noqa: E402


# ----------------------------------------------------------------- du lieu
class LineDataset(Dataset):
    """Moi ham -> [N, L] input_ids/attention_mask + mat na dong [N] + hai ma tran ke [N, N].

    DUNG SAN MOT LAN trong __init__. Neu dung trong __getitem__ thi moi epoch phai chay lai
    regex + tokenizer cho ca 456 hang, va do thi thi KHONG BAO GIO doi — thuan tuy lang phi.
    Chi luu phan THAT (n dong), viec dem cho vua [N, N] lam luc lay mau.
    """

    def __init__(self, rows, tok, max_lines, max_line_tokens, max_degree, symbolize,
                 node_mode="line", chunk_budget=96):
        self.N, self.L = max_lines, max_line_tokens + 2      # + CLS/SEP
        self.items = []
        for r in rows:
            if node_mode == "chunk":
                lines, Ad, Ac = build_chunked(r["code"], tok, chunk_budget, self.N,
                                              max_lines=400, max_degree=max_degree)
            else:
                lines, Ad, Ac = build_graphs(r["code"], self.N, max_degree)
            if symbolize == "vars":
                lines = _symbolize(lines)
            enc = tok(lines, truncation=True, max_length=self.L)["input_ids"]
            self.items.append((enc, Ad.astype(np.float32), Ac.astype(np.float32), int(r["label"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        enc, Ad, Ac, y = self.items[i]
        n = len(enc)
        ids = torch.zeros(self.N, self.L, dtype=torch.long)
        am = torch.zeros(self.N, self.L, dtype=torch.long)
        for k, e in enumerate(enc):
            ids[k, : len(e)] = torch.tensor(e, dtype=torch.long)
            am[k, : len(e)] = 1
        lm = torch.zeros(self.N, dtype=torch.bool); lm[:n] = True
        D = torch.zeros(self.N, self.N); D[:n, :n] = torch.from_numpy(Ad)
        C = torch.zeros(self.N, self.N); C[:n, :n] = torch.from_numpy(Ac)
        return {"input_ids": ids, "attention_mask": am, "line_mask": lm,
                "A_data": D, "A_ctrl": C, "labels": torch.tensor(y)}


def _symbolize(lines):
    """Duong BAN GOC BABEL: doi ten dinh danh -> VAR1, VAR2... Mac dinh KHONG dung, vi
    backbone o day da tien huan luyen tren code that va ten dinh danh chinh la tin hieu."""
    import re as _re
    from babel_graph import STOP
    ren, nxt = {}, [0]

    def sym(m):
        t = m.group(0)
        if t in STOP:
            return t
        if t not in ren:
            nxt[0] += 1; ren[t] = f"VAR{nxt[0]}"
        return ren[t]

    return [_re.sub(r"[A-Za-z_][A-Za-z_0-9]*", sym, l) for l in lines]


# ----------------------------------------------------------------- mo hinh
class GCN(nn.Module):
    """Tang tich chap do thi DONG NHAT, dung chuan hoa theo bac nhu ban BABEL."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, A):
        h = self.lin(x)                                   # [B, N, d_out]
        deg = A.sum(-1, keepdim=True).clamp(min=1.0)
        return torch.bmm(A, h) / deg


class BabelModel(nn.Module):
    def __init__(self, backbone, graph_hidden=256, pooling="cls", dropout=0.1):
        super().__init__()
        self.backbone, self.pooling = backbone, pooling
        H = backbone.config.hidden_size
        G = graph_hidden
        self.data_gcn1, self.data_gcn2 = GCN(H, G), GCN(G, G)
        self.ctrl_gcn1, self.ctrl_gcn2 = GCN(H, G), GCN(G, G)
        self.gdrop = nn.Dropout(0.5)                      # BABEL dung 0.5 giua hai tang GCN
        self.lstm = nn.LSTM(input_size=H + 2 * G, hidden_size=G, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.vul_head = nn.Sequential(nn.Linear(G, G), nn.GELU(), nn.Linear(G, 2))

    def graph_params(self):
        """Moi thu NGOAI backbone — phai co learning rate rieng."""
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def encode_lines(self, ids, am, lm, micro):
        B, N, L = ids.shape
        flat = lm.view(-1).nonzero(as_tuple=False).squeeze(1)
        i2, a2 = ids.view(B * N, L)[flat], am.view(B * N, L)[flat]
        outs = []
        for s in range(0, i2.size(0), micro):
            o = self.backbone(input_ids=i2[s: s + micro], attention_mask=a2[s: s + micro])
            outs.append(pool_hidden_states(o.last_hidden_state, a2[s: s + micro], self.pooling))
        h = torch.zeros(B * N, self.backbone.config.hidden_size,
                        device=ids.device, dtype=outs[0].dtype)
        h[flat] = torch.cat(outs, 0)
        return h.view(B, N, -1)

    def forward(self, ids, am, lm, A_data, A_ctrl, micro=64):
        h = self.encode_lines(ids, am, lm, micro)
        d = torch.relu(self.data_gcn1(h, A_data))
        d = torch.relu(self.data_gcn2(self.gdrop(d), A_data))
        c = torch.relu(self.ctrl_gcn1(h, A_ctrl))
        c = torch.relu(self.ctrl_gcn2(self.gdrop(c), A_ctrl))
        z, _ = self.lstm(torch.cat((h, d, c), -1))
        m = lm.unsqueeze(-1).to(z.dtype)
        z = (z * m).sum(1) / m.sum(1).clamp(min=1.0)
        return {"vul_logits": self.vul_head(self.dropout(z))}


# ----------------------------------------------------------------- chay
def load_split(root, fold, split):
    return [json.loads(l) for l in open(Path(root) / f"fold{fold}" / f"{split}.jsonl")]


@torch.no_grad()
def infer(model, dl, dev, micro):
    model.eval(); P, Y = [], []
    for b in dl:
        out = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["line_mask"].to(dev),
                    b["A_data"].to(dev), b["A_ctrl"].to(dev), micro)
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
    ap.add_argument("--method_name", default="babel")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", default="data/sven_python_folds_norm")
    ap.add_argument("--model_name", default="microsoft/codebert-base")
    ap.add_argument("--pooling", default="cls")
    # --- do thi ---
    ap.add_argument("--node_mode", default="line", choices=("line", "chunk"),
                    help="line = moi DONG mot dinh (ban BABEL goc); "
                         "chunk = gop dong thanh KHUC <= --chunk_budget token roi moi dung do thi")
    ap.add_argument("--chunk_budget", type=int, default=96,
                    help="chi dung khi --node_mode chunk. Do tren 760 ham dich: 96 token cho "
                         "5.5 dinh/ham, 79.7 token/dinh (gap 4x ban theo dong), 14.8% ham con 1 dinh. "
                         "Dung 256+ thi >50% ham chi con MOT dinh — khong con do thi de hoc.")
    ap.add_argument("--max_lines", type=int, default=150,
                    help="so DINH toi da (dong hoac khuc)")
    ap.add_argument("--max_line_tokens", type=int, default=48)
    ap.add_argument("--max_degree", type=int, default=0, help="0 = khong cat bot canh")
    ap.add_argument("--graph_hidden", type=int, default=256)
    ap.add_argument("--symbolize", default="none", choices=("none", "vars"),
                    help="none = giu ten that (mac dinh, vi backbone da tien huan luyen tren code); "
                         "vars = doi ten nhu ban goc BABEL")
    # --- huan luyen: DUNG co chuan cua khoi mw_assemble de so duoc ---
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--micro", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=2e-5, help="lr cua BACKBONE")
    ap.add_argument("--graph_lr", type=float, default=5e-4, help="lr cua GCN/LSTM/head")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min_epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--selection_metric", default="roc_auc", choices=("roc_auc", "macro_f1"))
    ap.add_argument("--sam_rho", type=float, default=0.02)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--init_ckpt", default=None, help="checkpoint Pha 1: CHI nap `backbone.*`")
    ap.add_argument("--max_train_samples", type=int)
    ap.add_argument("--output_dir", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = BabelModel(build_backbone(a.model_name), a.graph_hidden, a.pooling).to(dev)

    if a.init_ckpt:
        blob = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        sd = blob.get("model_state_dict", blob)
        keep = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        if not keep:
            raise SystemExit(f"{a.init_ckpt}: khong co khoa `backbone.` nao")
        tgt = {k for k in model.state_dict() if k.startswith("backbone.")}
        miss, extra = sorted(tgt - set(keep)), sorted(set(keep) - tgt)
        model.load_state_dict(keep, strict=False); model.to(dev)
        print(f"[babel] nap Pha 1 tu {a.init_ckpt}: chuyen {len(keep)} khoa backbone | "
              f"thieu {len(miss)} | thua {len(extra)}", flush=True)
        if miss:  print(f"[babel]   vd thieu: {miss[:4]}", flush=True)
        if extra: print(f"[babel]   vd thua : {extra[:4]}", flush=True)
        if len(miss) > 8:
            raise SystemExit(f"nap hut {len(miss)} khoa backbone — kiem lai checkpoint")

    tr = load_split(a.data_root, a.fold, "train")
    if a.max_train_samples:
        tr = tr[: a.max_train_samples]
    va, te = load_split(a.data_root, a.fold, "val"), load_split(a.data_root, a.fold, "test")
    mk = lambda rows: LineDataset(rows, tok, a.max_lines, a.max_line_tokens, a.max_degree,
                                  a.symbolize, a.node_mode, a.chunk_budget)
    dtr = DataLoader(mk(tr), batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers)
    dva = DataLoader(mk(va), batch_size=a.eval_batch_size, shuffle=False, num_workers=a.num_workers)
    dte = DataLoader(mk(te), batch_size=a.eval_batch_size, shuffle=False, num_workers=a.num_workers)

    ds = mk(te[:60])
    nl = [int(ds[i]["line_mask"].sum()) for i in range(len(ds))]
    ed = [float((ds[i]["A_data"].sum() - ds[i]["line_mask"].sum())) / max(nl[i], 1) for i in range(len(ds))]
    print(f"[babel] mode={a.node_mode}"
          + (f" budget={a.chunk_budget}" if a.node_mode == "chunk" else "")
          + f" | dinh/ham tren 60 hang test: TB {np.mean(nl):.1f}, cham tran {a.max_lines}: "
          f"{100*np.mean(np.array(nl) >= a.max_lines):.1f}% | 1 dinh: {100*np.mean(np.array(nl)==1):.1f}% | "
          f"canh DATA/dinh TB {np.mean(ed):.2f} | symbolize={a.symbolize}", flush=True)

    gp = model.graph_params()
    gid = {id(p) for p in gp}
    bp = [p for p in model.parameters() if id(p) not in gid]
    opt = torch.optim.AdamW([{"params": bp, "lr": a.learning_rate},
                             {"params": gp, "lr": a.graph_lr}], weight_decay=a.weight_decay)
    print(f"[babel] tham so: backbone {sum(p.numel() for p in bp)/1e6:.1f}M @ lr {a.learning_rate} | "
          f"do thi+head {sum(p.numel() for p in gp)/1e6:.1f}M @ lr {a.graph_lr}", flush=True)
    total = max(1, len(dtr) * a.epochs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.learning_rate, a.graph_lr],
                                              total_steps=total, pct_start=a.warmup_ratio,
                                              anneal_strategy="linear")
    lossf = nn.CrossEntropyLoss()
    sam = SAMStep(a.sam_rho) if a.sam_rho > 0 else None
    params = [p for p in model.parameters() if p.requires_grad]

    best, best_ep, bad, best_state = -1.0, 0, 0, None
    best_thr, best_yv, best_pv = 0.5, None, None
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for b in dtr:
            ids, am, lm = b["input_ids"].to(dev), b["attention_mask"].to(dev), b["line_mask"].to(dev)
            Ad, Ac, y = b["A_data"].to(dev), b["A_ctrl"].to(dev), b["labels"].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(ids, am, lm, Ad, Ac, a.micro)["vul_logits"], y)
            loss.backward()
            if sam is not None and sam.ascend(params):
                opt.zero_grad(set_to_none=True)
                lossf(model(ids, am, lm, Ad, Ac, a.micro)["vul_logits"], y).backward()
                sam.restore(params)
            torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
            opt.step(); sch.step(); tot += float(loss.item()) * y.size(0)
        yv, pv = infer(model, dva, dev, a.micro)
        thr, _ = find_best_threshold(yv.tolist(), pv.tolist())
        mv, mv_cal = metrics(yv, pv, 0.5), metrics(yv, pv, thr)
        score = mv["roc_auc"] if a.selection_metric == "roc_auc" else mv_cal["macro_f1"]
        print(f"[babel] epoch {ep:2d} | loss {tot/len(tr):.4f} | val ROC {mv['roc_auc']:.4f} "
              f"| val mF1@cal {mv_cal['macro_f1']:.4f} | {time.time()-t0:.0f}s", flush=True)
        if score > best:
            best, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr, best_yv, best_pv = thr, yv, pv
        else:
            bad += 1
            if ep >= a.min_epochs and bad >= a.patience:
                print(f"[babel] dung som o epoch {ep} (best {best_ep})", flush=True); break

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
        "source_checkpoint": a.init_ckpt, "target_checkpoint": None,
        "hyperparameters": vars(a) | {"architecture": f"babel_{a.node_mode}_twogcn"},
    }
    (out / f"fold{a.fold}.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    print(f"[babel] XONG fold {a.fold} | test mF1@0.5 {m05['macro_f1']:.4f} | ROC {m05['roc_auc']:.4f} "
          f"| PR {m05['pr_auc']:.4f} -> {out}/fold{a.fold}.json", flush=True)


if __name__ == "__main__":
    main()
