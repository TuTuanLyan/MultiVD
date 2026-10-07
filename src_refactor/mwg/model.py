"""MWGraphModel: CodeBERT nhiều cửa sổ + nhánh đồ thị dòng.

  mã nguồn -> cửa sổ token (W=510, stride 384, K=8) -> CodeBERT dùng chung -> trạng thái token [Nwin, 512, H]
      |                                                      gộp theo DÒNG (token -> dòng) -> H0 [B, L, H]
      |                                                                                        |
      +-> dòng -> VARk/FUNk (mwg.symbols) -> đồ thị [R, L, L] -> R-GCN x2 (residual + LN) -> LSTM -> max-pool -> z_g
      +-> CLS mỗi cửa sổ + vị trí cửa sổ -> mean-pool -> z_mw
  head: fusion(z_mw, z_g) -> Linear -> 2 lớp   (--fusion cat | sum | graph | mw)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

ARCHITECTURE = "codebert_mwgraph"


def build_backbone(model_name):
    return AutoModel.from_pretrained(model_name)


def pool_hidden_states(hidden_states, attention_mask, strategy):
    """`cls`: vị trí 0; `mean`: trung bình theo attention mask."""
    if strategy == "cls":
        return hidden_states[:, 0, :]
    if strategy != "mean":
        raise ValueError(f"pooling không hỗ trợ: {strategy!r}")
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def _row_normalize(adj):
    return adj / adj.sum(dim=-1, keepdim=True).clamp(min=1.0)


class RGCNLayer(nn.Module):
    """R-GCN như layer/RGCN.py của BABEL: sum_r norm(A_r) H W_r (+ b)."""

    def __init__(self, num_rel, d):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_rel, d, d))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, H, A):                                   # H [B, L, d], A [B, R, L, L]
        out = 0
        for r in range(A.size(1)):
            out = out + torch.bmm(_row_normalize(A[:, r]), H) @ self.weight[r]
        return out + self.bias


class BabelWordAttNet(nn.Module):
    """Bộ mã hoá dòng gốc của BABEL (model/word_att_model.py), dùng khi --line_enc babel: embedding CodeBERT
    đóng băng -> Dropout(0.5) -> Conv1d(k=5) -> attention theo từ -> vector dòng. Không đi qua encoder CodeBERT."""

    def __init__(self, embed_weight, d, attn_hidden=256, pad_id=1):
        super().__init__()
        self.lookup = nn.Embedding.from_pretrained(embed_weight.detach().clone(), freeze=True, padding_idx=pad_id)
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv1d(embed_weight.size(1), d, kernel_size=5)
        self.attn_w = nn.Linear(d, attn_hidden)
        self.attn_v = nn.Linear(attn_hidden, 1, bias=False)
        self.pad_id = pad_id

    def forward(self, line_ids):                               # [B, L, Wd] -> [B, L, d]
        B, L, Wd = line_ids.shape
        x = self.lookup(line_ids.view(B * L, Wd))
        x = self.dropout(x).permute(0, 2, 1)
        x = F.pad(x, (2, 2))                                   # giữ đủ Wd vị trí
        h = self.conv1(x).permute(0, 2, 1)
        a = torch.softmax(self.attn_v(torch.tanh(self.attn_w(h))), dim=1)
        return (a * h).sum(1).view(B, L, -1)


class LineGraphBranch(nn.Module):
    """H0 (vector dòng) -> R-GCN x n (residual + LN) -> LSTM -> masked max-pool -> z_g [B, d]."""

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
        H = backbone.config.hidden_size
        self.H = H
        self.win_pos_emb = nn.Embedding(max_windows, H)
        self.rel_pos = nn.Linear(1, H)
        if agg == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=H, nhead=agg_heads, dim_feedforward=4 * 256, dropout=dropout,
                                               batch_first=True, activation="gelu")
            self.agg = nn.TransformerEncoder(layer, num_layers=agg_layers)
        else:
            self.agg = None
        self.graph = LineGraphBranch(H, num_rel, graph_layers, dropout, graph_lstm)
        self.word_att = (BabelWordAttNet(backbone.get_input_embeddings().weight, H, pad_id=backbone.config.pad_token_id)
                         if line_enc == "babel" else None)
        self.dropout = nn.Dropout(dropout)
        self.vul_head = nn.Linear(2 * H if fusion == "cat" else H, 2)

    def encode_windows(self, input_ids, attention_mask, window_mask, micro=16):
        """[B, K, L] -> CLS [B, K, H], trạng thái token [Nwin, L, H], chỉ số cửa sổ thật."""
        B, K, L = input_ids.shape
        flat_idx = window_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        ids = input_ids.view(B * K, L)[flat_idx]
        am = attention_mask.view(B * K, L)[flat_idx]
        pooled, states = [], []
        for s in range(0, ids.size(0), micro):
            o = self.backbone(input_ids=ids[s: s + micro], attention_mask=am[s: s + micro]).last_hidden_state
            pooled.append(pool_hidden_states(o, am[s: s + micro], self.pooling))
            states.append(o)
        h = torch.zeros(B * K, self.H, device=input_ids.device, dtype=pooled[0].dtype)
        h[flat_idx] = torch.cat(pooled, 0)
        return h.view(B, K, -1), torch.cat(states, 0), flat_idx

    def line_states(self, states, flat_idx, tok_line, B):
        """Token -> dòng: trung bình trạng thái các token của mỗi dòng (qua mọi cửa sổ chứa nó)."""
        L = tok_line.size(-1)
        tl = tok_line.view(-1, L)[flat_idx]                                  # [Nwin, L]
        valid = tl >= 0
        b_of_win = (flat_idx // tok_line.size(1)).unsqueeze(1).expand_as(tl)   # chỉ số mẫu
        idx = (b_of_win * self.Lmax + tl)[valid]
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


def pair_margin_loss(logits, y, pair, margin):
    """softplus(margin - (s_lỗi - s_vá)) cho mỗi cặp có đủ hai nhãn trong batch; s = logit_1 - logit_0."""
    s = logits[:, 1] - logits[:, 0]
    losses = []
    for pid in torch.unique(pair[pair >= 0]):
        m = pair == pid
        if m.sum() != 2 or y[m].sum() != 1:
            continue
        losses.append(F.softplus(margin - (s[m & (y == 1)] - s[m & (y == 0)])))
    return torch.cat(losses).mean() if losses else logits.sum() * 0.0
