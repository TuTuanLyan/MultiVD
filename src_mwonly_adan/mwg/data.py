"""Dữ liệu MWG: mỗi hàm -> các cửa sổ token (multi-window) + đồ thị dòng + ánh xạ token -> dòng."""
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph import BRACKET_ONLY, build_relations
from .symbols import normalize_identifiers

LANG_MAP = {"ccpp": "c", "c": "c", "cpp": "c", "python": "python", "js": "js", "javascript": "js", "java": "java"}


class GraphWindowDataset(Dataset):
    """Mỗi hàm cho ra:
      input_ids / attention_mask [K, max_length]  cửa sổ token (window W, bước stride, tối đa K cửa sổ)
      tok_line [K, max_length]                   chỉ số dòng của từng token (-1 = không thuộc dòng đồ thị)
      graphs [R, Lmax, Lmax] uint8, line_mask [Lmax], line_ids [Lmax, max_words] (chỉ dùng khi line_enc=babel)

    Dòng đồ thị = dòng không rỗng của mã (bỏ thêm dòng chỉ có ngoặc nếu drop_bracket), tối đa Lmax dòng đầu.
    Văn bản đưa vào CodeBERT là mã GỐC (norm_text=False) hoặc mã đã chuẩn hoá VARk/FUNk (norm_text=True).
    Toàn bộ tiền xử lý làm một lần lúc khởi tạo."""

    def __init__(self, records, tokenizer, window, stride, max_windows, max_lines, typed, max_length=512, norm_text=False,
                 max_words=60, drop_bracket=False, co_mode="all"):
        self.records, self.tok, self.max_words = records, tokenizer, max_words
        self.window, self.stride, self.K, self.Lmax, self.max_length = window, stride, max_windows, max_lines, max_length
        self.typed, self.norm_text, self.pad = typed, norm_text, tokenizer.pad_token_id
        self.drop_bracket, self.co_mode = drop_bracket, co_mode
        pids = sorted({str(r["pair_id"]) for r in records if r.get("pair_id") is not None})
        self.pid_index = {k: i for i, k in enumerate(pids)}
        self.pair_idx = [self.pid_index.get(str(r["pair_id"]), -1) if r.get("pair_id") is not None else -1 for r in records]
        self.R = 4 if typed else 2
        self.items = [self._prep(r) for r in records]
        self.n_tokens = [it["n_tok"] for it in self.items]
        self.n_windows_total = [self._count_windows(it["n_tok"]) for it in self.items]
        self.n_lines = [it["n_lines"] for it in self.items]

    def _count_windows(self, n):
        return 1 if n <= self.window else 1 + math.ceil((n - self.window) / self.stride)

    def _is_graph_line(self, ln):
        return bool(ln.strip()) and not (self.drop_bracket and BRACKET_ONLY.match(ln))

    def _prep(self, r):
        code = r["code"].replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
        lang = LANG_MAP.get(str(r.get("lang", "c")).lower(), "c")
        raw_lines = code.split("\n")
        line_id, lines = [], []
        for ln in raw_lines:
            if self._is_graph_line(ln):
                line_id.append(len(lines))
                lines.append(ln)
            else:
                line_id.append(-1)
        if not lines:
            lines, line_id = [""], [0]
        n_lines = len(lines)
        L = min(n_lines, self.Lmax)
        sym = normalize_identifiers(lines[:L], lang)
        try:
            rel = build_relations(lines[:L], sym[:L], self.typed, self.co_mode)
        except Exception:                                   # phòng khi heuristic lỗi trên một hàm lạ
            rel = np.zeros((self.R, L, L), dtype=np.uint8)
        if self.norm_text:
            text = "\n".join(sym) + ("\n" + "\n".join(lines[L:]) if n_lines > L else "")
            raw_lines, line_id, k = text.split("\n"), [], 0           # chỉ số dòng tính lại trên văn bản mới
            for ln in raw_lines:
                ok = self._is_graph_line(ln)
                line_id.append(k if ok else -1)
                k += 1 if ok else 0
        else:
            text = code
        enc = self.tok(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True,
                       return_attention_mask=False, verbose=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        starts = np.cumsum([0] + [len(ln) + 1 for ln in raw_lines[:-1]])
        tl = np.full(len(ids), -1, dtype=np.int64)
        if ids:
            st = np.asarray([o[0] for o in offs])
            en = np.asarray([o[1] for o in offs])
            pos = np.where(en > st, st, np.maximum(st - 1, 0))          # token rỗng (đầu dòng) -> ký tự trước
            gl = np.asarray(line_id)[np.searchsorted(starts, pos, side="right") - 1]
            tl = np.where((gl >= 0) & (gl < self.Lmax), gl, -1)
        # token id từng dòng cho bộ mã hoá dòng gốc của BABEL (--line_enc babel), tối đa max_words / dòng
        line_ids = np.full((self.Lmax, self.max_words), self.pad, dtype=np.int64)
        for j, ln in enumerate(lines[:L]):
            w = self.tok(ln, add_special_tokens=False, truncation=True, max_length=self.max_words,
                         return_attention_mask=False, verbose=False)["input_ids"]
            line_ids[j, : len(w)] = w
        return {"ids": ids, "tl": tl, "rel": rel, "L": L, "n_tok": len(ids), "n_lines": n_lines, "line_ids": line_ids}

    def coverage(self):
        nt, nw, nl = np.asarray(self.n_tokens), np.asarray(self.n_windows_total), np.asarray(self.n_lines)
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
        it = self.items[i]
        ids, tl = it["ids"], it["tl"]
        starts = [0] if len(ids) <= self.window else list(range(0, len(ids) - self.window + self.stride, self.stride))
        starts = starts[: self.K]
        input_ids = torch.full((self.K, self.max_length), self.pad, dtype=torch.long)
        attn = torch.zeros((self.K, self.max_length), dtype=torch.long)
        tok_line = torch.full((self.K, self.max_length), -1, dtype=torch.long)
        wmask = torch.zeros(self.K, dtype=torch.bool)
        pos = torch.zeros(self.K, dtype=torch.float)
        for k, s in enumerate(starts):
            w = ids[s: s + self.window]
            seq = self.tok.build_inputs_with_special_tokens(w)
            input_ids[k, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[k, : len(seq)] = 1
            tok_line[k, 1: 1 + len(w)] = torch.from_numpy(tl[s: s + self.window])     # <s> ở 0, </s> ở cuối -> -1
            wmask[k] = True
            pos[k] = s / max(1, len(ids) - 1)
        g = torch.zeros((self.R, self.Lmax, self.Lmax), dtype=torch.uint8)
        L = it["L"]
        g[:, :L, :L] = torch.from_numpy(it["rel"])
        lmask = torch.zeros(self.Lmax, dtype=torch.bool)
        lmask[:L] = True
        return {"input_ids": input_ids, "attention_mask": attn, "window_mask": wmask, "window_pos": pos,
                "tok_line": tok_line, "graphs": g, "line_mask": lmask, "line_ids": torch.from_numpy(it["line_ids"]),
                "labels": torch.tensor(int(self.records[i]["label"]), dtype=torch.long), "index": torch.tensor(i),
                "pair": torch.tensor(self.pair_idx[i], dtype=torch.long)}


class PairBatchSampler:
    """Pha 1 trên nguồn đủ cặp: mỗi batch gom các cặp (bản lỗi, bản vá) cùng pair_id để tính pair loss;
    mẫu lẻ xếp sau."""

    def __init__(self, dataset, batch_size, seed):
        self.bs, self.rng = batch_size, np.random.default_rng(seed)
        groups = {}
        for i, pi in enumerate(dataset.pair_idx):
            groups.setdefault(pi if pi >= 0 else f"s{i}", []).append(i)
        self.pairs = [g for g in groups.values()
                      if len(g) == 2 and {dataset.records[g[0]]["label"], dataset.records[g[1]]["label"]} == {0, 1}]
        self.singles = [i for g in groups.values() for i in g if g not in self.pairs]
        self.n = len(dataset)

    def __iter__(self):
        pairs = [list(p) for p in self.pairs]
        self.rng.shuffle(pairs)
        singles = list(self.singles)
        self.rng.shuffle(singles)
        flat = [i for p in pairs for i in p] + singles
        for s in range(0, len(flat), self.bs):
            yield flat[s: s + self.bs]

    def __len__(self):
        return math.ceil(self.n / self.bs)
