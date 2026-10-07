#!/usr/bin/env python3
"""MW + BABEL (16/09): encoder CodeBERT nhieu cua so (train_mw.py, giu nguyen) + BIEU DIEN DONG cua BABEL
(https://github.com/gdufsnlp/BABEL, ICSME 2024) tren do thi dong heuristic, hop nhat giua cac ngon ngu.

  ma nguon -> cua so token (W=510, stride 384, K=8)  -> CodeBERT dung chung -> trang thai token [Nwin, 512, H]
        |                                                                          |
        |                                                   gop theo DONG (token -> dong qua offset) -> H0 [B, L, H]
        |                                                                          |
        +-> dong -> chuan hoa VARk/FUNk (clean_gadget) -> do thi [R, L, L]   -> R-GCN x2 (residual+LN) -> LSTM -> max-pool -> z_g
        |
        +-> CLS moi cua so -> vi tri cua so -> mean-pool -> z_mw   (nhanh MW goc, y het train_mw.py)

  head: [z_mw ; z_g] -> Linear -> 2 lop   (--fusion cat | sum | graph | mw)

Do thi (src/babel/graph.py, chep tu multibabel):  --graph typed: 4 quan he def_use / co_use / ctrl / next;
                                                  --graph babel: 2 quan he goc (co_use, ctrl).
Dong = dong KHONG trong cua ma nguon (giu nguyen text cho CodeBERT; dong chi comment / ngoac = dong "khong lenh").
Chuyen giao: --init all (encoder + vi tri + agg + nhanh do thi tu checkpoint nguon, head moi) + --recadam + --sam_rho,
giong train_mw.py. Muc dich: bieu dien do thi khong phu thuoc ngon ngu nen pha 1 (nguon C/JS/Java) -> pha 2 (Python) dong nhat hon.
"""
import argparse, json, math, os, re, sys, time
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
from babel import clean_gadget, build_relations                       # noqa: E402
from babel.graph import BRACKET_ONLY, VAR_RE, FUN_RE, construct_adjacency_matrix, get_depth_list, connect_elements, def_use_matrix, next_matrix  # noqa: E402

logger = get_logger()
def co_use_chain(code_symbolic):
    """co_use theo CHUOI: moi ky hieu chi noi hai lan xuat hien LIEN TIEP (i -> j ke tiep), doi xung.
    Khac ban goc noi MOI cap dong -> mat do tang theo binh phuong do dai ham, lech 2,8x giua cac ngon ngu (do 16/09)."""
    n = len(code_symbolic)
    m = np.eye(n, dtype="uint8")
    last = {}
    for i, line in enumerate(code_symbolic):
        for sym in set(VAR_RE.findall(line)) | {"f" + x for x in FUN_RE.findall(line)}:
            j = last.get(sym)
            if j is not None and j != i:
                m[i][j] = 1; m[j][i] = 1
            last[sym] = i
    return m


def build_rel_v2(lines, sym, typed, co_mode):
    """Nhu babel.build_relations nhung co_use co the la 'chain'."""
    if co_mode != "chain":
        return build_relations(lines, sym, typed)
    co = co_use_chain(sym)
    ctrl = connect_elements(get_depth_list(lines))
    if not typed:
        return np.stack([co, ctrl])
    return np.stack([def_use_matrix(sym).T, co, ctrl.T, next_matrix(lines).T])


LANG_MAP = {"ccpp": "c", "c": "c", "cpp": "c", "python": "python", "js": "js", "javascript": "js", "java": "java"}
COMMENT_LINE = re.compile(r"^\s*(#|//|/\*|\*|\"\"\"|''')")


# ----------------------------------------------------------------------------- du lieu
class GraphWindowDataset(Dataset):
    """Moi ham -> cua so token nhu WindowDataset + tok_line [K, max_length] (chi so dong, -1 = khong thuoc dong nao)
    + graphs [R, Lmax, Lmax] uint8 + line_mask [Lmax]."""

    def __init__(self, records, tokenizer, window, stride, max_windows, max_lines, typed, max_length=512, norm_text=False, max_words=60,
                 drop_bracket=False, co_mode="all"):
        self.records = records; self.tok = tokenizer; self.max_words = max_words
        self.window, self.stride, self.K, self.Lmax, self.max_length = window, stride, max_windows, max_lines, max_length
        self.typed, self.norm_text, self.pad = typed, norm_text, tokenizer.pad_token_id
        self.drop_bracket, self.co_mode = drop_bracket, co_mode
        pids = sorted({str(r["pair_id"]) for r in records if r.get("pair_id") is not None}); self.pid_index = {k: i for i, k in enumerate(pids)}
        self.pair_idx = [self.pid_index.get(str(r["pair_id"]), -1) if r.get("pair_id") is not None else -1 for r in records]
        self.R = 4 if typed else 2
        self.n_tokens, self.n_windows_total, self.n_lines = [], [], []
        self.items = [self._prep(r) for r in records]
        for it in self.items:
            self.n_tokens.append(it["n_tok"]); self.n_windows_total.append(self._count_windows(it["n_tok"])); self.n_lines.append(it["n_lines"])

    def _count_windows(self, n):
        return 1 if n <= self.window else 1 + math.ceil((n - self.window) / self.stride)

    def _prep(self, r):
        code = r["code"].replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
        lang = LANG_MAP.get(str(r.get("lang", "c")).lower(), "c")
        raw_lines = code.split("\n")
        # dong khong trong -> chi so dong do thi; dong comment/ngoac van la dong (graph.py tu bo qua qua BRACKET_ONLY)
        line_id, lines = [], []
        for ln in raw_lines:
            if ln.strip() and not (self.drop_bracket and BRACKET_ONLY.match(ln)):
                line_id.append(len(lines)); lines.append(ln)
            else:
                line_id.append(-1)
        if not lines:
            lines, line_id = [""], [0]
        n_lines = len(lines)
        L = min(n_lines, self.Lmax)
        sym = clean_gadget(lines[:L], lang)
        try:
            rel = build_rel_v2(lines[:L], sym[:L], self.typed, self.co_mode)
        except Exception:                          # phong khi heuristic loi tren mot ham la
            rel = np.zeros((self.R, L, L), dtype=np.uint8)
        text = "\n".join(sym) + ("\n" + "\n".join(lines[L:]) if n_lines > L else "") if self.norm_text else code
        if self.norm_text:                          # text da doi -> offset tinh lai tren text moi (dong giu nguyen)
            raw_lines = text.split("\n"); line_id = []
            k = 0
            for ln in raw_lines:
                ok = ln.strip() and not (self.drop_bracket and BRACKET_ONLY.match(ln))
                line_id.append(k if ok else -1); k += 1 if ok else 0
        enc = self.tok(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True,
                       return_attention_mask=False, verbose=False)
        ids = enc["input_ids"]; offs = enc["offset_mapping"]
        starts = np.cumsum([0] + [len(ln) + 1 for ln in raw_lines[:-1]])
        tl = np.full(len(ids), -1, dtype=np.int64)
        if ids:
            st = np.asarray([o[0] for o in offs]); en = np.asarray([o[1] for o in offs])
            pos = np.where(en > st, st, np.maximum(st - 1, 0))          # token rong (dau dong) -> ky tu truoc
            ri = np.searchsorted(starts, pos, side="right") - 1
            gl = np.asarray(line_id)[ri]
            tl = np.where((gl >= 0) & (gl < self.Lmax), gl, -1)
        # bo ma hoa dong goc BABEL: token id tung dong (text goc), toi da max_words/dong (BABEL max_wordnum 60)
        line_ids = np.full((self.Lmax, self.max_words), self.pad, dtype=np.int64)
        for j, ln in enumerate(lines[:L]):
            w = self.tok(ln, add_special_tokens=False, truncation=True, max_length=self.max_words, return_attention_mask=False, verbose=False)["input_ids"]
            line_ids[j, : len(w)] = w
        return {"ids": ids, "tl": tl, "rel": rel, "L": L, "n_tok": len(ids), "n_lines": n_lines, "line_ids": line_ids}

    def coverage(self):
        nt = np.asarray(self.n_tokens); nw = np.asarray(self.n_windows_total); nl = np.asarray(self.n_lines)
        covered = np.minimum(nt, self.window + (self.K - 1) * self.stride)
        return {"n": int(len(nt)), "pct_gt_window": float(100 * (nt > self.window).mean()),
                "pct_truncated_at_K": float(100 * (nw > self.K).mean()),
                "pct_tokens_dropped": float(100 * (1 - covered.sum() / max(1, nt.sum()))),
                "median_tokens": int(np.median(nt)), "p90_tokens": int(np.percentile(nt, 90)),
                "mean_windows_used": float(np.minimum(nw, self.K).mean()),
                "median_lines": int(np.median(nl)), "p90_lines": int(np.percentile(nl, 90)),
                "pct_lines_truncated": float(100 * (nl > self.Lmax).mean())}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        it = self.items[i]; ids, tl = it["ids"], it["tl"]
        starts = [0] if len(ids) <= self.window else list(range(0, len(ids) - self.window + self.stride, self.stride))
        starts = starts[: self.K]
        input_ids = torch.full((self.K, self.max_length), self.pad, dtype=torch.long)
        attn = torch.zeros((self.K, self.max_length), dtype=torch.long)
        tok_line = torch.full((self.K, self.max_length), -1, dtype=torch.long)
        wmask = torch.zeros(self.K, dtype=torch.bool); pos = torch.zeros(self.K, dtype=torch.float)
        for k, s in enumerate(starts):
            w = ids[s: s + self.window]
            seq = self.tok.build_inputs_with_special_tokens(w)
            input_ids[k, : len(seq)] = torch.tensor(seq, dtype=torch.long); attn[k, : len(seq)] = 1
            tok_line[k, 1: 1 + len(w)] = torch.from_numpy(tl[s: s + self.window])     # <s> o 0, </s> o cuoi -> -1
            wmask[k] = True; pos[k] = s / max(1, len(ids) - 1)
        g = torch.zeros((self.R, self.Lmax, self.Lmax), dtype=torch.uint8)
        L = it["L"]; g[:, :L, :L] = torch.from_numpy(it["rel"])
        lmask = torch.zeros(self.Lmax, dtype=torch.bool); lmask[:L] = True
        return {"input_ids": input_ids, "attention_mask": attn, "window_mask": wmask, "window_pos": pos,
                "tok_line": tok_line, "graphs": g, "line_mask": lmask, "line_ids": torch.from_numpy(it["line_ids"]),
                "labels": torch.tensor(int(self.records[i]["label"]), dtype=torch.long), "index": torch.tensor(i),
                "pair": torch.tensor(self.pair_idx[i], dtype=torch.long)}


class PairBatchSampler:
    """Pha 1 tren NGUON du cap: moi batch gom cac cap (loi, va) cung pair_id de tinh mat mat theo cap; mau le xep sau."""

    def __init__(self, dataset, batch_size, seed):
        self.bs, self.rng = batch_size, np.random.default_rng(seed)
        groups = {}
        for i, (pi, r) in enumerate(zip(dataset.pair_idx, dataset.records)):
            groups.setdefault(pi if pi >= 0 else f"s{i}", []).append(i)
        self.pairs = [g for g in groups.values() if len(g) == 2 and {dataset.records[g[0]]["label"], dataset.records[g[1]]["label"]} == {0, 1}]
        self.singles = [i for g in groups.values() for i in g if g not in self.pairs]
        self.n = len(dataset)

    def __iter__(self):
        pr = [list(p) for p in self.pairs]; self.rng.shuffle(pr); sg = list(self.singles); self.rng.shuffle(sg)
        flat = [i for p in pr for i in p] + sg
        for s in range(0, len(flat), self.bs):
            yield flat[s: s + self.bs]

    def __len__(self):
        return math.ceil(self.n / self.bs)


def pair_margin_loss(logits, y, pair, margin):
    """softplus(margin - (s_loi - s_va)) cho moi cap co mat du hai nhan trong batch; s = logit_1 - logit_0."""
    s = logits[:, 1] - logits[:, 0]; losses = []
    for pid in torch.unique(pair[pair >= 0]):
        m = pair == pid
        if m.sum() != 2 or y[m].sum() != 1: continue
        losses.append(F.softplus(margin - (s[m & (y == 1)] - s[m & (y == 0)])))
    return torch.cat(losses).mean() if losses else logits.sum() * 0.0


# ----------------------------------------------------------------------------- mo hinh
def _row_normalize(adj):
    rs = adj.sum(dim=-1, keepdim=True)
    return adj / rs.clamp(min=1.0)


class RGCNLayer(nn.Module):
    """R-GCN nhu layer/RGCN.py cua BABEL: sum_r  norm(A_r) H W_r  (+ b)."""

    def __init__(self, num_rel, d):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_rel, d, d)); nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, H, A):                                  # H [B,L,d], A [B,R,L,L]
        out = 0
        for r in range(A.size(1)):
            out = out + torch.bmm(_row_normalize(A[:, r]), H) @ self.weight[r]
        return out + self.bias


class BabelWordAttNet(nn.Module):
    """WordAttNet goc cua BABEL (model/word_att_model.py): bang embedding CodeBERT dong bang -> Dropout(0.5) -> Conv1d(k=5)
    -> attention theo tu (tanh W, v) -> vector dong. Khong di qua encoder CodeBERT."""

    def __init__(self, embed_weight, d, attn_hidden=256, pad_id=1):
        super().__init__()
        self.lookup = nn.Embedding.from_pretrained(embed_weight.detach().clone(), freeze=True, padding_idx=pad_id)
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv1d(embed_weight.size(1), d, kernel_size=5)
        self.attn_w = nn.Linear(d, attn_hidden); self.attn_v = nn.Linear(attn_hidden, 1, bias=False)
        self.pad_id = pad_id

    def forward(self, line_ids):                              # [B, L, Wd] -> [B, L, d]
        B, L, Wd = line_ids.shape
        x = self.lookup(line_ids.view(B * L, Wd))               # [BL, Wd, E]
        x = self.dropout(x).permute(0, 2, 1)                    # [BL, E, Wd]
        x = F.pad(x, (2, 2))                                    # giu du Wd vi tri (BABEL khong pad: dong < 5 token se rong)
        h = self.conv1(x).permute(0, 2, 1)                      # [BL, Wd, d]
        a = self.attn_v(torch.tanh(self.attn_w(h)))             # [BL, Wd, 1]
        a = torch.softmax(a, dim=1)                             # softmax theo tu, ke ca pad (nhu ma goc)
        return (a * h).sum(1).view(B, L, -1)


class LineGraphBranch(nn.Module):
    """H0 (dong, tu CodeBERT) -> RGCN x n (residual + LN) -> LSTM -> masked max-pool -> z_g [B, d]."""

    def __init__(self, d, num_rel, layers=2, dropout=0.1, use_lstm=True):
        super().__init__()
        self.ln0 = nn.LayerNorm(d)
        self.gcn = nn.ModuleList([RGCNLayer(num_rel, d) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(d, d, batch_first=True) if use_lstm else None

    def forward(self, H0, A, lmask):
        H = self.ln0(H0)
        for g, ln in zip(self.gcn, self.ln):
            H = ln(H + self.drop(F.relu(g(H, A))))
        if self.lstm is not None:
            H, _ = self.lstm(H)
        return H.masked_fill(~lmask.unsqueeze(-1), -1e4).max(dim=1).values


class MWGraphModel(nn.Module):
    def __init__(self, backbone, max_windows, agg="mean", agg_layers=2, agg_heads=8, dropout=0.1, pooling="cls",
                 num_rel=4, graph_layers=2, fusion="cat", graph_lstm=True, max_lines=150, line_enc="codebert"):
        super().__init__()
        self.line_enc = line_enc
        self.backbone, self.pooling, self.agg_kind, self.fusion, self.Lmax = backbone, pooling, agg, fusion, max_lines
        H = backbone.config.hidden_size; self.H = H
        self.win_pos_emb = nn.Embedding(max_windows, H); self.rel_pos = nn.Linear(1, H)
        if agg == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=H, nhead=agg_heads, dim_feedforward=4 * 256, dropout=dropout,
                                               batch_first=True, activation="gelu")
            self.agg = nn.TransformerEncoder(layer, num_layers=agg_layers)
        else:
            self.agg = None
        self.graph = LineGraphBranch(H, num_rel, graph_layers, dropout, graph_lstm)
        self.word_att = BabelWordAttNet(backbone.get_input_embeddings().weight, H, pad_id=backbone.config.pad_token_id) if line_enc == "babel" else None
        self.dropout = nn.Dropout(dropout)
        self.vul_head = nn.Linear(2 * H if fusion == "cat" else H, 2)

    def encode_windows(self, input_ids, attention_mask, window_mask, micro=16):
        """[B,K,L] -> pooled [B,K,H], token states [Nwin, L, H], flat_idx."""
        B, K, L = input_ids.shape
        flat_idx = window_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        ids = input_ids.view(B * K, L)[flat_idx]; am = attention_mask.view(B * K, L)[flat_idx]
        pooled, states = [], []
        for s in range(0, ids.size(0), micro):
            o = self.backbone(input_ids=ids[s: s + micro], attention_mask=am[s: s + micro]).last_hidden_state
            pooled.append(pool_hidden_states(o, am[s: s + micro], self.pooling)); states.append(o)
        h = torch.zeros(B * K, self.H, device=input_ids.device, dtype=pooled[0].dtype)
        h[flat_idx] = torch.cat(pooled, 0)
        return h.view(B, K, -1), torch.cat(states, 0), flat_idx

    def line_states(self, states, flat_idx, tok_line, B):
        """Gop token -> dong: mean cac trang thai token cua moi dong (qua moi cua so chua no)."""
        L = tok_line.size(-1)
        tl = tok_line.view(-1, L)[flat_idx]                                  # [Nwin, L]
        valid = tl >= 0
        b_of_win = (flat_idx // tok_line.size(1)).unsqueeze(1).expand_as(tl)   # chi so mau
        idx = (b_of_win * self.Lmax + tl)[valid]                               # [Ntok]
        acc = torch.zeros(B * self.Lmax, self.H, device=states.device, dtype=states.dtype)
        acc.index_add_(0, idx, states[valid])
        cnt = torch.zeros(B * self.Lmax, device=states.device, dtype=states.dtype)
        cnt.index_add_(0, idx, torch.ones_like(idx, dtype=states.dtype))
        return (acc / cnt.clamp(min=1.0).unsqueeze(1)).view(B, self.Lmax, self.H)

    def forward(self, b, micro=16):
        input_ids, wmask = b["input_ids"], b["window_mask"]
        B, K, _ = input_ids.shape
        h, states, flat_idx = self.encode_windows(input_ids, b["attention_mask"], wmask, micro)
        h = h + self.win_pos_emb(torch.arange(K, device=h.device)).unsqueeze(0) + self.rel_pos(b["window_pos"].unsqueeze(-1))
        if self.agg is not None:
            h = self.agg(h, src_key_padding_mask=~wmask)
        m = wmask.unsqueeze(-1).to(h.dtype)
        z_mw = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.fusion == "mw":
            return {"vul_logits": self.vul_head(self.dropout(z_mw))}
        H0 = self.word_att(b["line_ids"]) if self.word_att is not None else self.line_states(states, flat_idx, b["tok_line"], B)
        z_g = self.graph(H0, b["graphs"].to(states.dtype), b["line_mask"])
        z = {"cat": lambda: torch.cat([z_mw, z_g], -1), "sum": lambda: z_mw + z_g, "graph": lambda: z_g}[self.fusion]()
        return {"vul_logits": self.vul_head(self.dropout(z))}


# ----------------------------------------------------------------------------- vong lap
KEYS = ("input_ids", "attention_mask", "window_mask", "window_pos", "tok_line", "graphs", "line_mask", "line_ids", "pair")


def to_dev(b, device):
    return {k: b[k].to(device) for k in KEYS}


def run_eval(model, loader, device, micro):
    model.eval(); labels, probs, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for b in loader:
            out = model(to_dev(b, device), micro); y = b["labels"].to(device)
            loss_sum += F.cross_entropy(out["vul_logits"], y).item() * y.size(0); n += y.size(0)
            labels.extend(y.cpu().tolist()); probs.extend(torch.softmax(out["vul_logits"], -1)[:, 1].cpu().tolist())
    res = classification_metrics(labels, probs, 0.5); res.update(loss=loss_sum / max(1, n), labels=labels, probabilities=probs)
    return res


def save_ckpt(path, model, epoch, score, args):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "best_epoch": epoch, "best_val_score": score,
                "architecture": "codebert_mwgraph", "model_name": args.model_name, "window": args.window, "stride": args.stride,
                "max_windows": args.max_windows, "agg": args.agg, "seed": args.seed, "fold": args.fold,
                "training_args": {k: v for k, v in vars(args).items() if not k.startswith("_")}}, path)


def init_from(model, path, parts, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    assert ck.get("architecture") in ("codebert_mwgraph", "codebert_multiwindow"), f"{path}: khong phai checkpoint mw/mwgraph"
    sd = ck["model_state_dict"]
    pref = {"encoder": ("backbone.",), "encoder_agg": ("backbone.", "agg.", "win_pos_emb.", "rel_pos."),
            "graph": ("graph.", "word_att."), "graph_agg": ("graph.", "word_att.", "agg.", "win_pos_emb.", "rel_pos."),
            "all": ("backbone.", "agg.", "win_pos_emb.", "rel_pos.", "graph.", "word_att.")}[parts]
    keep = {k: v for k, v in sd.items() if k.startswith(pref)}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    logger.info("Khoi tao tu %s | parts=%s | nap %d tensor | head MOI | thieu (ngoai backbone): %s", path, parts, len(keep),
                [m for m in missing if not m.startswith("backbone.")][:8])
    assert not unexpected, unexpected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("train", "test"), required=True)
    p.add_argument("--run_name", required=True); p.add_argument("--fold", type=int, default=1)
    p.add_argument("--data_root", required=True); p.add_argument("--target_lang", default=None)
    p.add_argument("--model_name", required=True); p.add_argument("--seed", type=int, default=36)
    p.add_argument("--window", type=int, default=510); p.add_argument("--stride", type=int, default=384)
    p.add_argument("--max_windows", type=int, default=8); p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--agg", choices=("transformer", "mean"), default="mean"); p.add_argument("--agg_layers", type=int, default=2)
    # --- BABEL
    p.add_argument("--graph", choices=("typed", "babel"), default="typed", help="typed: 4 quan he; babel: 2 quan he goc")
    p.add_argument("--max_lines", type=int, default=150, help="BABEL max_sentnum")
    p.add_argument("--graph_layers", type=int, default=2); p.add_argument("--graph_lstm", type=int, default=1)
    p.add_argument("--fusion", choices=("cat", "sum", "graph", "mw"), default="cat")
    p.add_argument("--norm_text", type=int, default=0, help="1 = dua ca text da chuan hoa VARk/FUNk vao CodeBERT")
    p.add_argument("--line_enc", choices=("codebert", "babel"), default="codebert",
                   help="vector dong: codebert = gop trang thai token CodeBERT theo dong; babel = WordAttNet goc (embedding dong bang + Conv1d + attn)")
    p.add_argument("--drop_bracket", type=int, default=0, help="1 = bo dong chi co ngoac khoi tap dinh (Python 1,1 %% vs C/JS/Java 16-21 %% -> dong nhat hoa do thi)")
    p.add_argument("--co_mode", choices=("all", "chain"), default="all", help="chain = co_use chi noi hai lan xuat hien lien tiep cua moi ky hieu (mat do khong phu thuoc do dai ham)")
    p.add_argument("--max_words", type=int, default=60, help="BABEL max_wordnum (chi dung khi --line_enc babel)")
    p.add_argument("--graph_lr", type=float, default=1e-4, help="lr rieng cho nhanh do thi (moi, khong pretrain)")
    # --- chuyen giao / toi uu (giong train_mw.py)
    p.add_argument("--init", choices=("none", "encoder", "encoder_agg", "graph", "graph_agg", "all"), default="none",
                   help="graph = chi nap nhanh do thi tu checkpoint nguon (encoder = CodeBERT goc), dung voi pha 1 --freeze_backbone 1")
    p.add_argument("--pair_loss", type=float, default=0.0, help=">0: pha 1 tren nguon du cap, batch xep theo cap + lambda*softplus(margin - (s_loi - s_va))")
    p.add_argument("--pair_margin", type=float, default=1.0)
    p.add_argument("--freeze_backbone", type=int, default=0, help="1 = dong bang CodeBERT (pha 1 chi hoc nhanh do thi + vi tri + head)"); p.add_argument("--init_ckpt", default=None)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--min_epochs", type=int, default=3); p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4); p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5); p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.10); p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--selection_metric", choices=("roc_auc", "macro_f1", "pr_auc"), default="roc_auc")
    p.add_argument("--grad_checkpoint", type=int, default=1)
    p.add_argument("--sam_rho", type=float, default=0.0)
    p.add_argument("--sam_variant", default="sam", choices=["sam", "asam"], help="asam = Kwon et al. ICML 2021, eps = rho*T^2*g/||T*g||, T=|w|+eta (23/09)")
    p.add_argument("--sam_eta", type=float, default=0.01)
    # 04/10 (K15): ASAM bật MUỘN - tắt nhiễu cho tới khi train loss trung bình một epoch < ngưỡng, từ epoch sau bật và giữ bật. 0 = tắt cờ (hành vi cũ)
    p.add_argument("--sam_start_loss", type=float, default=0.0)
    # 04/10 (K15): luật dừng hai giai đoạn - patience chỉ đếm từ epoch đầu tiên có train loss < ngưỡng (kể cả epoch đó). 0 = tắt cờ (hành vi cũ)
    p.add_argument("--patience_start_loss", type=float, default=0.0)
    p.add_argument("--recadam", type=int, default=0); p.add_argument("--pretrain_cof", type=float, default=5000.0)
    p.add_argument("--anneal_t0_ratio", type=float, default=0.05); p.add_argument("--anneal_k", type=float, default=0.05); p.add_argument("--anneal_fun", default="sigmoid")
    p.add_argument("--checkpoint_path", default=None); p.add_argument("--result_path", default=None)
    p.add_argument("--max_train_samples", type=int, default=None); p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0); p.add_argument("--pooling", default="cls"); p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    root = Path("model") / args.run_name / "multiwindow" / f"seed_{args.seed}"
    args.checkpoint_path = args.checkpoint_path or str(root / f"fold{args.fold}" / "best.pt")
    args.result_path = args.result_path or str(Path("results") / args.run_name / "multiwindow" / f"seed_{args.seed}" / f"fold{args.fold}.json")
    configure_logging(); set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    d = Path(args.data_root) / f"fold{args.fold}"
    typed = args.graph == "typed"
    load = lambda sp, cap: limit_records(load_jsonl(d / f"{sp}.jsonl", args.target_lang), cap, args.seed)
    mk = lambda recs, bs, shuf: DataLoader(GraphWindowDataset(recs, tok, args.window, args.stride, args.max_windows, args.max_lines, typed,
                                                              args.max_length, bool(args.norm_text), args.max_words,
                                                              bool(args.drop_bracket), args.co_mode),
                                           batch_size=bs, shuffle=shuf, num_workers=args.num_workers)
    backbone = build_backbone(args.model_name)
    if args.freeze_backbone:
        for q in backbone.parameters(): q.requires_grad = False
        backbone.eval(); logger.info("CodeBERT DONG BANG (freeze_backbone=1): chi hoc nhanh do thi + vi tri cua so + head")
    elif args.grad_checkpoint and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    model = MWGraphModel(backbone, args.max_windows, args.agg, args.agg_layers, pooling=args.pooling, num_rel=4 if typed else 2,
                         graph_layers=args.graph_layers, fusion=args.fusion, graph_lstm=bool(args.graph_lstm), max_lines=args.max_lines, line_enc=args.line_enc).to(device)
    n_graph = sum(q.numel() for n, q in model.named_parameters() if n.startswith(("graph.", "word_att.")) and q.requires_grad)
    logger.info("MWGraph | graph=%s R=%d Lmax=%d layers=%d lstm=%d fusion=%s norm_text=%d line_enc=%s drop_bracket=%d co_mode=%s | tham so nhanh do thi %.2fM",
                args.graph, 4 if typed else 2, args.max_lines, args.graph_layers, args.graph_lstm, args.fusion, args.norm_text, args.line_enc,
                args.drop_bracket, args.co_mode, n_graph / 1e6)

    if args.phase == "train":
        t_prep = time.perf_counter()
        tr = mk(load("train", args.max_train_samples), args.batch_size, True); va = mk(load("val", args.max_eval_samples), args.eval_batch_size, False)
        if args.pair_loss > 0:
            ds_tr = tr.dataset; smp = PairBatchSampler(ds_tr, args.batch_size, args.seed)
            tr = DataLoader(ds_tr, batch_sampler=smp, num_workers=args.num_workers)
            logger.info("PAIR LOSS lambda=%.2f margin=%.2f | %d cap du hai nhan, %d mau le", args.pair_loss, args.pair_margin, len(smp.pairs), len(smp.singles))
        cov = {"train": tr.dataset.coverage(), "val": va.dataset.coverage()}
        logger.info("DO PHU W=%d S=%d K=%d Lmax=%d | tien xu ly %.0fs | %s", args.window, args.stride, args.max_windows, args.max_lines,
                    time.perf_counter() - t_prep, json.dumps(cov))
        if args.init != "none":
            init_from(model, args.init_ckpt, args.init, device)
        total = len(tr) * args.epochs
        GP = ("graph.", "word_att.") + (("vul_head.", "win_pos_emb.", "rel_pos.") if args.freeze_backbone else ())   # dong bang: moi thu con lai deu moi -> lr do thi
        g_params = [q for n, q in model.named_parameters() if n.startswith(GP) and q.requires_grad]
        o_params = [q for n, q in model.named_parameters() if not n.startswith(GP) and q.requires_grad]
        groups = [{"params": o_params, "lr": args.learning_rate}, {"params": g_params, "lr": args.graph_lr}]
        if args.recadam:
            from RecAdam import RecAdam
            _params = o_params + g_params
            _anchor = [q.detach().clone() for q in _params]          # neo = trong so ngay sau init_from
            _t0 = max(1, int(args.anneal_t0_ratio * total))
            opt = RecAdam([{"params": o_params, "lr": args.learning_rate, "pretrain_params": _anchor[: len(o_params)]},
                           {"params": g_params, "lr": args.graph_lr, "pretrain_params": _anchor[len(o_params):]}],
                          lr=args.learning_rate, weight_decay=args.weight_decay, anneal_fun=args.anneal_fun, anneal_k=args.anneal_k,
                          anneal_t0=_t0, anneal_w=1.0, pretrain_cof=args.pretrain_cof, pretrain_params=_anchor)
            logger.info("RecAdam bat | cof=%.0f | t0=%d/%d | neo %d tensor", args.pretrain_cof, _t0, total, len(_anchor))
        else:
            opt = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=args.weight_decay)
        sam = None
        if args.sam_rho > 0:
            from sam import SAMStep
            sam = SAMStep(args.sam_rho, variant=args.sam_variant, eta=args.sam_eta); logger.info("SAM bat | variant=%s rho=%.4f eta=%.4f", args.sam_variant, args.sam_rho, args.sam_eta)
        sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
        best, best_ep, wait = -math.inf, 0, 0; t0 = time.perf_counter()
        sam_on = args.sam_start_loss <= 0              # cờ tắt ⇒ ASAM chạy từ bước 1 như cũ
        pat_on = args.patience_start_loss <= 0         # cờ tắt ⇒ patience đếm như cũ
        if sam is not None and not sam_on:
            logger.info("ASAM tam tat toi khi train loss epoch < %.4f", args.sam_start_loss)
        if not pat_on:
            logger.info("Patience chi dem tu epoch dau tien co train loss < %.4f", args.patience_start_loss)
        for ep in range(1, args.epochs + 1):
            model.train(); tl, tn = 0.0, 0; te = time.perf_counter()
            if args.freeze_backbone: model.backbone.eval()
            for b in tr:
                bd = to_dev(b, device); y = b["labels"].to(device)
                lg = model(bd, args.micro)["vul_logits"]; loss = F.cross_entropy(lg, y)
                if args.pair_loss > 0: loss = loss + args.pair_loss * pair_margin_loss(lg, y, bd["pair"], args.pair_margin)
                opt.zero_grad(set_to_none=True); loss.backward()
                if sam is not None and sam_on:
                    named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]; trainable = [q for _, q in named]
                    if sam.ascend(trainable, names=[n for n, _ in named]):
                        opt.zero_grad(set_to_none=True)
                        lg2 = model(bd, args.micro)["vul_logits"]; l2 = F.cross_entropy(lg2, y)
                        if args.pair_loss > 0: l2 = l2 + args.pair_loss * pair_margin_loss(lg2, y, bd["pair"], args.pair_margin)
                        l2.backward(); sam.restore(trainable)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm); opt.step(); sched.step()
                tl += loss.item() * y.size(0); tn += y.size(0)
            v = run_eval(model, va, device, args.micro)
            for _k in ("roc_auc", "macro_f1", "pr_auc"):
                if v.get(_k) is None: v[_k] = float("nan")
            score = v.get(args.selection_metric)
            if score is None or score != score: score = v["macro_f1"]
            ep_tl = tl / max(1, tn)
            if not pat_on and ep_tl < args.patience_start_loss:
                pat_on = True; logger.info("Patience bat dau dem tu epoch %d (train loss %.4f < %.4f)", ep, ep_tl, args.patience_start_loss)
            if score > best:
                best, best_ep, wait = score, ep, 0; save_ckpt(args.checkpoint_path, model, ep, score, args)
                logger.info("Best checkpoint saved | Epoch: %d | Val %s: %.6f", ep, args.selection_metric, score)
            elif ep >= args.min_epochs and pat_on:
                wait += 1
            logger.info("Epoch %d/%d | train loss %.4f | val loss %.4f | val roc_auc %.4f | val macro_f1 %.4f | best ep %d | patience %d/%d | %.0fs",
                        ep, args.epochs, tl / max(1, tn), v["loss"], v["roc_auc"], v["macro_f1"], best_ep, wait, args.patience, time.perf_counter() - te)
            if ep >= args.min_epochs and wait >= args.patience:
                logger.info("Early stopping | Epoch: %d | Best epoch: %d", ep, best_ep); break
            if sam is not None and not sam_on and ep_tl < args.sam_start_loss:
                sam_on = True; logger.info("ASAM bat tu epoch %d (train loss epoch %d = %.4f < %.4f)", ep + 1, ep, ep_tl, args.sam_start_loss)
        logger.info("Train xong | best epoch %d | %.0fs", best_ep, time.perf_counter() - t0)
    else:
        ck = torch.load(args.checkpoint_path, map_location=device, weights_only=True); model.load_state_dict(ck["model_state_dict"])
        va = mk(load("val", args.max_eval_samples), args.eval_batch_size, False); te = mk(load("test", args.max_eval_samples), args.eval_batch_size, False)
        t0 = time.perf_counter(); v = run_eval(model, va, device, args.micro); thr, vf1 = find_best_threshold(v["labels"], v["probabilities"])
        t1 = time.perf_counter(); t = run_eval(model, te, device, args.micro); t2 = time.perf_counter()
        m05 = classification_metrics(t["labels"], t["probabilities"], 0.5); mvc = classification_metrics(t["labels"], t["probabilities"], thr)
        ds = te.dataset; nt = np.asarray(ds.n_tokens); lab = np.asarray(t["labels"]); pr = np.asarray(t["probabilities"])
        def sub(mask):
            from sklearn.metrics import roc_auc_score
            return (float(roc_auc_score(lab[mask], pr[mask])) if mask.sum() > 1 and len(set(lab[mask])) == 2 else None, int(mask.sum()))
        res = {"experiment_name": f"{args.run_name}/multiwindow", "architecture": "codebert_mwgraph", "phase": "test", "fold": args.fold, "seed": args.seed,
               "best_epoch": ck["best_epoch"], "best_val_score": ck["best_val_score"], "val_calibrated_threshold": thr, "val_macro_f1_at_valcal": vf1,
               "test_roc_auc": m05["roc_auc"], "test_pr_auc": m05["pr_auc"], "test_macro_f1_at_0.5": m05["macro_f1"], "test_macro_f1_at_valcal": mvc["macro_f1"],
               "test_inference_seconds": t2 - t1, "validation_and_threshold_seconds": t1 - t0,
               "subgroup_roc": {"le510": sub(nt <= 510), "gt510": sub(nt > 510), "one_window": sub(np.asarray(ds.n_windows_total) == 1),
                                "truncated_at_K": sub(np.asarray(ds.n_windows_total) > args.max_windows),
                                "lines_gt_Lmax": sub(np.asarray(ds.n_lines) > args.max_lines)},
               "coverage": {"val": va.dataset.coverage(), "test": ds.coverage()}, "hyperparameters": ck["training_args"]}
        Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.result_path, "w"), indent=2, sort_keys=True)
        print(json.dumps({k: res[k] for k in ("test_roc_auc", "test_macro_f1_at_valcal", "best_epoch", "subgroup_roc")}))
        if os.environ.get("DUMP_PROBS", "1") == "1":
            np.savez_compressed(Path(args.result_path).with_suffix(".probs.npz"), probabilities=pr.astype(np.float32), labels=lab.astype(np.int8),
                                n_tokens=nt, val_probabilities=np.asarray(v["probabilities"], np.float32), val_labels=np.asarray(v["labels"], np.int8),
                                threshold=np.float32(thr))
        logger.info("Ket qua: %s", args.result_path)


if __name__ == "__main__":
    main()
