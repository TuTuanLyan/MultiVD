import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

ARCHITECTURE = "codebert_multiwindow"


def build_backbone(model_name):
    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "is_encoder_decoder", False):
        from transformers import T5EncoderModel

        return T5EncoderModel.from_pretrained(model_name)
    return AutoModel.from_pretrained(model_name)


def pool_hidden_states(hidden_states, attention_mask, strategy):
    if strategy == "cls":
        return hidden_states[:, 0, :]
    if strategy != "mean":
        raise ValueError(f"unsupported pooling: {strategy!r}")
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def advance_rng_past_removed_graph_branch(hidden_size, num_relations=4, layers=2):
    for _ in range(layers):
        nn.init.xavier_uniform_(torch.empty(num_relations, hidden_size, hidden_size))
    nn.LSTM(hidden_size, hidden_size, batch_first=True)


class MultiWindowModel(nn.Module):
    def __init__(self, backbone, max_windows, agg="mean", agg_layers=2, agg_heads=8, dropout=0.1, pooling="cls"):
        super().__init__()
        self.backbone, self.pooling = backbone, pooling
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
        advance_rng_past_removed_graph_branch(H)
        self.dropout = nn.Dropout(dropout)
        self.vul_head = nn.Linear(H, 2)

    def encode_windows(self, input_ids, attention_mask, window_mask, micro=16):
        B, K, L = input_ids.shape
        flat_idx = window_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        ids = input_ids.view(B * K, L)[flat_idx]
        am = attention_mask.view(B * K, L)[flat_idx]
        pooled = []
        for s in range(0, ids.size(0), micro):
            o = self.backbone(input_ids=ids[s: s + micro], attention_mask=am[s: s + micro]).last_hidden_state
            pooled.append(pool_hidden_states(o, am[s: s + micro], self.pooling))
        h = torch.zeros(B * K, self.H, device=input_ids.device, dtype=pooled[0].dtype)
        h[flat_idx] = torch.cat(pooled, 0)
        return h.view(B, K, -1)

    def forward(self, b, micro=16):
        input_ids, wmask = b["input_ids"], b["window_mask"]
        B, K, _ = input_ids.shape
        h = self.encode_windows(input_ids, b["attention_mask"], wmask, micro)
        h = h + self.win_pos_emb(torch.arange(K, device=h.device)).unsqueeze(0) + self.rel_pos(b["window_pos"].unsqueeze(-1))
        if self.agg is not None:
            h = self.agg(h, src_key_padding_mask=~wmask)
        m = wmask.unsqueeze(-1).to(h.dtype)
        z = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return {"vul_logits": self.vul_head(self.dropout(z))}
