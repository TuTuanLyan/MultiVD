#!/usr/bin/env python3
"""Huong 2 (DE_XUAT_CAI_TIEN_TRANSFER.md): encoder dung chung + module tong hop NHIEU CUA SO.

Mot ham -> cac cua so lien tiep co overlap (W=510 token, stride S) -> encoder E (CodeBERT)
-> vector CLS moi cua so h_i -> module tong hop A (Transformer nho co embedding vi tri cua so,
hoac mean-pool) -> head nhi phan H o CAP HAM. Loss chi o cap ham. Khong dung diff/vi tri va.

Quy tac cua so chot truoc (muc 39): W=510, stride 384, tran K=8 -> bao cao DO PHU that
(ty le ham bi cat sau K cua so). Khong goi la "doc du ham".

Ba doi chung toi thieu (muc 2 cua de xuat):
  --init none          : target-only, E goc + A ngau nhien + H moi        (mw_b_*)
  --init encoder       : E tu checkpoint nguon, A ngau nhien, H moi        (mw_xE_*)
  --init encoder_agg   : E + A tu checkpoint nguon, H moi                  (mw_xEA_*)
Checkpoint nguon o day la checkpoint CUA CHINH script nay chay o --phase train tren nguon.

File RIENG, khong sua train.py / train_transfer.py / train_baseline.py dang duoc dung boi v2.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import classification_metrics, find_best_threshold      # noqa: E402
from model import build_backbone, pool_hidden_states                  # noqa: E402
from train_transfer import load_jsonl, limit_records, set_seed        # noqa: E402
from logging_utils import configure_logging, get_logger               # noqa: E402

logger = get_logger()


# ----------------------------------------------------------------------------- du lieu
class WindowDataset(Dataset):
    """Moi ham -> [K, max_length] input_ids/attention_mask + mat na cua so [K]."""

    def __init__(self, records, tokenizer, window, stride, max_windows, max_length=512):
        self.records = records
        self.tok = tokenizer
        self.window, self.stride, self.K, self.max_length = window, stride, max_windows, max_length
        self.pad = tokenizer.pad_token_id
        # tien tinh so cua so + do phu de bao cao (khong luu token de tiet kiem RAM)
        self.n_tokens, self.n_windows_total = [], []
        for r in records:
            n = len(tokenizer(r["code"], add_special_tokens=False, truncation=False,
                              return_attention_mask=False, verbose=False)["input_ids"])
            self.n_tokens.append(n)
            self.n_windows_total.append(self._count_windows(n))

    def _count_windows(self, n):
        if n <= self.window:
            return 1
        return 1 + math.ceil((n - self.window) / self.stride)

    def coverage(self):
        nt = np.asarray(self.n_tokens); nw = np.asarray(self.n_windows_total)
        cut = nw > self.K
        # token bi bo khi cat sau K cua so
        covered = np.minimum(nt, self.window + (self.K - 1) * self.stride)
        return {
            "n": int(len(nt)), "pct_gt_window": float(100 * (nt > self.window).mean()),
            "pct_truncated_at_K": float(100 * cut.mean()),
            "pct_tokens_dropped": float(100 * (1 - covered.sum() / max(1, nt.sum()))),
            "median_tokens": int(np.median(nt)), "p90_tokens": int(np.percentile(nt, 90)),
            "mean_windows_used": float(np.minimum(nw, self.K).mean()),
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        ids = self.tok(r["code"], add_special_tokens=False, truncation=False,
                       return_attention_mask=False, verbose=False)["input_ids"]
        starts = [0] if len(ids) <= self.window else list(range(0, len(ids) - self.window + self.stride, self.stride))
        starts = starts[: self.K]
        # dam bao cua so cuoi khong vuot qua chuoi
        wins = [ids[s: s + self.window] for s in starts]
        input_ids = torch.full((self.K, self.max_length), self.pad, dtype=torch.long)
        attn = torch.zeros((self.K, self.max_length), dtype=torch.long)
        wmask = torch.zeros(self.K, dtype=torch.bool)
        for k, w in enumerate(wins):
            seq = self.tok.build_inputs_with_special_tokens(w)
            input_ids[k, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[k, : len(seq)] = 1
            wmask[k] = True
        # vi tri tuong doi cua cua so trong ham (0..1), dung lam dac trung vi tri
        pos = torch.zeros(self.K, dtype=torch.float)
        for k, s in enumerate(starts):
            pos[k] = s / max(1, len(ids) - 1)
        return {"input_ids": input_ids, "attention_mask": attn, "window_mask": wmask,
                "window_pos": pos, "labels": torch.tensor(int(r["label"]), dtype=torch.long),
                "index": torch.tensor(i, dtype=torch.long)}


# ----------------------------------------------------------------------------- mo hinh
class MultiWindowModel(nn.Module):
    def __init__(self, backbone, max_windows, agg="transformer", agg_layers=2, agg_heads=8,
                 dropout=0.1, pooling="cls"):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.agg_kind = agg
        H = backbone.config.hidden_size
        self.win_pos_emb = nn.Embedding(max_windows, H)      # chi so cua so
        self.rel_pos = nn.Linear(1, H)                        # vi tri tuong doi 0..1
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
        """[B,K,L] -> [B,K,H]; chi dua cua so THAT qua backbone, theo microbatch."""
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
        h = self.encode_windows(input_ids, attention_mask, window_mask, micro)      # [B,K,H]
        B, K, _ = h.shape
        h = h + self.win_pos_emb(torch.arange(K, device=h.device)).unsqueeze(0) \
              + self.rel_pos(window_pos.unsqueeze(-1))
        if self.agg is not None:
            h = self.agg(h, src_key_padding_mask=~window_mask)
        m = window_mask.unsqueeze(-1).to(h.dtype)
        z = (h * m).sum(1) / m.sum(1).clamp(min=1.0)                                 # mean-pool co mat na
        return {"vul_logits": self.vul_head(self.dropout(z))}


# ----------------------------------------------------------------------------- vong lap
def run_eval(model, loader, device, micro):
    model.eval(); labels, probs, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for b in loader:
            out = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                        b["window_mask"].to(device), b["window_pos"].to(device), micro)
            y = b["labels"].to(device)
            loss_sum += F.cross_entropy(out["vul_logits"], y).item() * y.size(0); n += y.size(0)
            labels.extend(y.cpu().tolist())
            probs.extend(torch.softmax(out["vul_logits"], -1)[:, 1].cpu().tolist())
    res = classification_metrics(labels, probs, 0.5)
    res.update(loss=loss_sum / max(1, n), labels=labels, probabilities=probs)
    return res


def save_ckpt(path, model, epoch, score, args):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "best_epoch": epoch, "best_val_score": score,
                "architecture": "codebert_multiwindow", "model_name": args.model_name,
                "window": args.window, "stride": args.stride, "max_windows": args.max_windows,
                "agg": args.agg, "seed": args.seed, "fold": args.fold,
                "training_args": {k: v for k, v in vars(args).items() if not k.startswith("_")}}, path)


def init_from(model, path, parts, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    assert ck.get("architecture") == "codebert_multiwindow", f"{path}: khong phai checkpoint multiwindow"
    sd = ck["model_state_dict"]
    keep = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    if parts == "pos":      # CHI chuyen embedding vi tri cua so / vi tri tuong doi; encoder = CodeBERT goc (review 14/09 muc 4)
        keep = {k: v for k, v in sd.items() if k.startswith(("win_pos_emb.", "rel_pos."))}
    if parts == "encoder_agg":
        keep.update({k: v for k, v in sd.items()
                     if k.startswith(("agg.", "win_pos_emb.", "rel_pos."))})
    missing, unexpected = model.load_state_dict(keep, strict=False)
    logger.info("Khoi tao tu %s | parts=%s | nap %d tensor | head MOI (bo qua: %s)",
                path, parts, len(keep), [m for m in missing if not m.startswith("backbone.")][:6])
    assert not unexpected, unexpected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("train", "test"), required=True)
    p.add_argument("--run_name", required=True); p.add_argument("--fold", type=int, default=1)
    p.add_argument("--data_root", required=True); p.add_argument("--target_lang", default=None)
    p.add_argument("--model_name", required=True); p.add_argument("--seed", type=int, default=36)
    p.add_argument("--window", type=int, default=510); p.add_argument("--stride", type=int, default=384)
    p.add_argument("--max_windows", type=int, default=8); p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--agg", choices=("transformer", "mean"), default="transformer")
    p.add_argument("--agg_layers", type=int, default=2)
    p.add_argument("--init", choices=("none", "encoder", "encoder_agg", "pos"), default="none")
    p.add_argument("--init_ckpt", default=None)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--min_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4); p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--micro", type=int, default=16, help="so cua so moi luot qua backbone")
    p.add_argument("--learning_rate", type=float, default=2e-5); p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.10); p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--selection_metric", choices=("roc_auc", "macro_f1", "pr_auc"), default="roc_auc")
    p.add_argument("--grad_checkpoint", type=int, default=1)
    p.add_argument("--sam_rho", type=float, default=0.0, help="SAM (sam.py::SAMStep); 0 = tat, giu duong cu")
    p.add_argument("--recadam", type=int, default=0, help="1 = RecAdam neo ve trong so NGAY SAU init (encoder + vi tri tu pha 1; head/moi = khoi tao cua no)")
    p.add_argument("--pretrain_cof", type=float, default=5000.0); p.add_argument("--anneal_t0_ratio", type=float, default=0.05)
    p.add_argument("--anneal_k", type=float, default=0.05); p.add_argument("--anneal_fun", default="sigmoid")
    p.add_argument("--checkpoint_path", default=None); p.add_argument("--result_path", default=None)
    p.add_argument("--max_train_samples", type=int, default=None); p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0); p.add_argument("--pooling", default="cls")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    root = Path("model") / args.run_name / "multiwindow" / f"seed_{args.seed}"
    args.checkpoint_path = args.checkpoint_path or str(root / f"fold{args.fold}" / "best.pt")
    args.result_path = args.result_path or str(Path("results") / args.run_name / "multiwindow" / f"seed_{args.seed}" / f"fold{args.fold}.json")
    configure_logging(); set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    d = Path(args.data_root) / f"fold{args.fold}"
    load = lambda sp, cap: limit_records(load_jsonl(d / f"{sp}.jsonl", args.target_lang), cap, args.seed)
    mk = lambda recs, bs, shuf: DataLoader(WindowDataset(recs, tok, args.window, args.stride, args.max_windows, args.max_length),
                                           batch_size=bs, shuffle=shuf, num_workers=args.num_workers)

    backbone = build_backbone(args.model_name)
    if args.grad_checkpoint and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    model = MultiWindowModel(backbone, args.max_windows, args.agg, args.agg_layers, pooling=args.pooling).to(device)

    if args.phase == "train":
        tr = mk(load("train", args.max_train_samples), args.batch_size, True)
        va = mk(load("val", args.max_eval_samples), args.eval_batch_size, False)
        cov = {"train": tr.dataset.coverage(), "val": va.dataset.coverage()}
        logger.info("DO PHU cua so W=%d S=%d K=%d | %s", args.window, args.stride, args.max_windows, json.dumps(cov))
        if args.init != "none":
            init_from(model, args.init_ckpt, args.init, device)
        total = len(tr) * args.epochs
        if args.recadam:
            from RecAdam import RecAdam
            _params = [q for q in model.parameters() if q.requires_grad]
            _anchor = [q.detach().clone() for q in _params]          # neo = trong so ngay sau init_from (muc 84)
            _t0 = max(1, int(args.anneal_t0_ratio * total))
            opt = RecAdam(_params, lr=args.learning_rate, weight_decay=args.weight_decay, anneal_fun=args.anneal_fun,
                          anneal_k=args.anneal_k, anneal_t0=_t0, anneal_w=1.0, pretrain_cof=args.pretrain_cof, pretrain_params=_anchor)
            logger.info("RecAdam bat | cof=%.0f | anneal t0=%d/%d buoc | neo %d tensor (encoder+vi tri tu pha 1, head = khoi tao)", args.pretrain_cof, _t0, total, len(_anchor))
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        sam = None
        if args.sam_rho > 0:
            from sam import SAMStep
            sam = SAMStep(args.sam_rho, variant="sam", eta=0.01)
            logger.info("SAM bat | rho=%.4f | moi buoc 2 luot forward-backward", args.sam_rho)
        sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
        best, best_ep, wait = -math.inf, 0, 0; t0 = time.perf_counter(); n_fb = 0
        for ep in range(1, args.epochs + 1):
            model.train(); tl, tn = 0.0, 0; te = time.perf_counter()
            for b in tr:
                out = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                            b["window_mask"].to(device), b["window_pos"].to(device), args.micro)
                y = b["labels"].to(device); loss = F.cross_entropy(out["vul_logits"], y)
                opt.zero_grad(set_to_none=True); loss.backward()
                if sam is not None:
                    named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
                    trainable = [q for _, q in named]
                    if sam.ascend(trainable, names=[n for n, _ in named]):
                        opt.zero_grad(set_to_none=True)
                        out2 = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                     b["window_mask"].to(device), b["window_pos"].to(device), args.micro)
                        F.cross_entropy(out2["vul_logits"], y).backward()
                        sam.restore(trainable); n_fb += int(b["window_mask"].sum())
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm); opt.step(); sched.step()
                tl += loss.item() * y.size(0); tn += y.size(0); n_fb += int(b["window_mask"].sum())
            v = run_eval(model, va, device, args.micro)
            for _k in ("roc_auc", "macro_f1", "pr_auc"):
                if v.get(_k) is None: v[_k] = float("nan")     # val don lop (smoke) -> khong lam vo log
            score = v.get(args.selection_metric)
            if score is None or score != score: score = v["macro_f1"]
            if score > best:
                best, best_ep, wait = score, ep, 0; save_ckpt(args.checkpoint_path, model, ep, score, args)
                logger.info("Best checkpoint saved | Epoch: %d | Val %s: %.6f", ep, args.selection_metric, score)
            elif ep >= args.min_epochs:
                wait += 1
            logger.info("Epoch %d/%d | train loss %.4f | val loss %.4f | val roc_auc %.4f | val macro_f1 %.4f | "
                        "best ep %d | patience %d/%d | %.0fs | forward-cua-so luy ke %d",
                        ep, args.epochs, tl / max(1, tn), v["loss"], v["roc_auc"], v["macro_f1"],
                        best_ep, wait, args.patience, time.perf_counter() - te, n_fb)
            if ep >= args.min_epochs and wait >= args.patience:
                logger.info("Early stopping | Epoch: %d | Best epoch: %d", ep, best_ep); break
        logger.info("Train xong | best epoch %d | %.0fs | tong forward cua so %d", best_ep, time.perf_counter() - t0, n_fb)
    else:
        ck = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state_dict"])
        va = mk(load("val", args.max_eval_samples), args.eval_batch_size, False)
        te = mk(load("test", args.max_eval_samples), args.eval_batch_size, False)
        t0 = time.perf_counter(); v = run_eval(model, va, device, args.micro); thr, vf1 = find_best_threshold(v["labels"], v["probabilities"])
        t1 = time.perf_counter(); t = run_eval(model, te, device, args.micro); t2 = time.perf_counter()
        m05 = classification_metrics(t["labels"], t["probabilities"], 0.5)
        mvc = classification_metrics(t["labels"], t["probabilities"], thr)
        ds = te.dataset; nt = np.asarray(ds.n_tokens); lab = np.asarray(t["labels"]); pr = np.asarray(t["probabilities"])
        def sub(mask):
            from sklearn.metrics import roc_auc_score
            return (float(roc_auc_score(lab[mask], pr[mask])) if mask.sum() > 1 and len(set(lab[mask])) == 2 else None, int(mask.sum()))
        res = {"experiment_name": f"{args.run_name}/multiwindow", "phase": "test", "fold": args.fold, "seed": args.seed,
               "best_epoch": ck["best_epoch"], "best_val_score": ck["best_val_score"], "val_calibrated_threshold": thr,
               "val_macro_f1_at_valcal": vf1, "test_roc_auc": m05["roc_auc"], "test_pr_auc": m05["pr_auc"],
               "test_macro_f1_at_0.5": m05["macro_f1"], "test_macro_f1_at_valcal": mvc["macro_f1"],
               "test_inference_seconds": t2 - t1, "validation_and_threshold_seconds": t1 - t0,
               "subgroup_roc": {"le510": sub(nt <= 510), "gt510": sub(nt > 510),
                                "one_window": sub(np.asarray(ds.n_windows_total) == 1),
                                "truncated_at_K": sub(np.asarray(ds.n_windows_total) > args.max_windows)},
               "coverage": {"val": va.dataset.coverage(), "test": ds.coverage()},
               "hyperparameters": ck["training_args"]}
        Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.result_path, "w"), indent=2, sort_keys=True); print(json.dumps({k: res[k] for k in ("test_roc_auc", "best_epoch", "subgroup_roc")}))
        if os.environ.get("DUMP_PROBS", "1") == "1":
            np.savez_compressed(Path(args.result_path).with_suffix(".probs.npz"),
                                probabilities=pr.astype(np.float32), labels=lab.astype(np.int8), n_tokens=nt,
                                val_probabilities=np.asarray(v["probabilities"], np.float32), val_labels=np.asarray(v["labels"], np.int8),
                                threshold=np.float32(thr))
        logger.info("Ket qua: %s", args.result_path)


if __name__ == "__main__":
    main()
