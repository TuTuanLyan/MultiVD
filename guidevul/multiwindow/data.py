import math

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, records, tokenizer, window, stride, max_windows, max_length=512):
        self.records, self.tok = records, tokenizer
        self.window, self.stride, self.K, self.max_length = window, stride, max_windows, max_length
        self.pad = tokenizer.pad_token_id
        self.items = [self._prep(r) for r in records]
        self.n_tokens = [len(ids) for ids in self.items]
        self.n_windows_total = [self._count_windows(n) for n in self.n_tokens]

    def _count_windows(self, n):
        return 1 if n <= self.window else 1 + math.ceil((n - self.window) / self.stride)

    def _prep(self, r):
        code = r["code"].replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
        return self.tok(code, add_special_tokens=False, truncation=False, return_attention_mask=False, verbose=False)["input_ids"]

    def coverage(self):
        nt, nw = np.asarray(self.n_tokens), np.asarray(self.n_windows_total)
        covered = np.minimum(nt, self.window + (self.K - 1) * self.stride)
        return {"n": int(len(nt)), "pct_gt_window": float(100 * (nt > self.window).mean()),
                "pct_truncated_at_K": float(100 * (nw > self.K).mean()),
                "pct_tokens_dropped": float(100 * (1 - covered.sum() / max(1, nt.sum()))),
                "median_tokens": int(np.median(nt)), "p90_tokens": int(np.percentile(nt, 90)),
                "mean_windows_used": float(np.minimum(nw, self.K).mean())}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        ids = self.items[i]
        starts = [0] if len(ids) <= self.window else list(range(0, len(ids) - self.window + self.stride, self.stride))
        starts = starts[: self.K]
        input_ids = torch.full((self.K, self.max_length), self.pad, dtype=torch.long)
        attn = torch.zeros((self.K, self.max_length), dtype=torch.long)
        wmask = torch.zeros(self.K, dtype=torch.bool)
        pos = torch.zeros(self.K, dtype=torch.float)
        for k, s in enumerate(starts):
            seq = self.tok.build_inputs_with_special_tokens(ids[s: s + self.window])
            input_ids[k, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[k, : len(seq)] = 1
            wmask[k] = True
            pos[k] = s / max(1, len(ids) - 1)
        return {"input_ids": input_ids, "attention_mask": attn, "window_mask": wmask, "window_pos": pos,
                "labels": torch.tensor(int(self.records[i]["label"]), dtype=torch.long), "index": torch.tensor(i)}
