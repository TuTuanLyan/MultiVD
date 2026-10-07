#!/usr/bin/env python3
"""Do TAI NGUYEN cho moi o thi nghiem — tham so, MACs, gio, bo nho, token, dien.

Tap chi so lay theo ba nguon doi chieu duoc trong bai:

  [1] Ni et al. 2024, "Learning-based Models for Vulnerability Detection: An
      Extensive Study" (arXiv:2408.07526) Bang 12 — Pre-processing(s) | Training(s)
      | Inferring(s) | Parameter(M) | Cost($). Inferring do tren CA TAP TEST.
  [2] Steenhoek et al. ICSE'24 "DeepDFA" (arXiv:2212.08108) Bang 5 — Train time |
      GPU inference MOI VI DU (ms) | CPU inference moi vi du (ms) | MACs.
  [3] Schwartz et al. CACM 2020 "Green AI" — FPO (doc lap phan cung), gio thuc,
      so tham so, VA khai bao phan cung de doc duoc gio thuc.

KHONG do dien nang va tien (nguoi dung neu 21/09: chi can tai nguyen ma bai bao hoc
may thuong dung). Bo luon bo lay mau nvidia-smi chay nen — mot thu it di chay song
song voi lan huan luyen cung la mot nguon nhieu it di.

Cai [1] va [2] deu KHONG tach "chi phi do do dai dau vao". Voi bai nay do la truc
chinh (512 cat cut vs nhieu cua so vs do thi), nen them nhom `token_budget`:
so vi tri token thuc su chay qua model moi epoch, va TONG BINH PHUONG do dai cua
tung cua so — vi chi phi attention ti le voi L^2 con phan con lai ti le voi L.
Co hai so nay thi tra loi duoc "full/common/4cwe tiet kiem cai gi" ma khong phai
chay lai.

Dung:
    from resource_log import ResourceLog
    rl = ResourceLog(device, args)
    with rl.stage("preprocess"): ...          # tokenize / dung do thi
    rl.model_stats(model, components={"backbone": model.encoder, ...})
    rl.macs(model, sample_batch)              # tuy chon, can thop/ptflops
    with rl.stage("train"): ...
    rl.epoch(train_s, val_s, n_train)         # goi moi epoch
    with rl.stage("infer_test"): ...
    rl.tokens(lengths=[...], windows=[...])   # do dai token thuc cua tung mau
    rl.dump(path)
"""
import json, os, statistics, time


class ResourceLog:
    def __init__(self, device=None, args=None, **_ignored):
        self.device, self.args = device, args
        self.t0 = time.perf_counter()
        self.stages, self.epochs, self.tok = {}, [], {}
        self.model = {}
        self._reset_peak()

    # ---------------------------------------------------------------- bo nho
    def _reset_peak(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _peak_mb(self):
        out = {}
        try:
            import torch
            if torch.cuda.is_available():
                out["gpu_alloc_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1048576, 1)
                out["gpu_reserved_peak_mb"] = round(torch.cuda.max_memory_reserved() / 1048576, 1)
        except Exception:
            pass
        try:
            import resource
            out["cpu_rss_peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        except Exception:
            pass
        return out

    # ----------------------------------------------------------------- giai doan
    class _Stage:
        def __init__(self, parent, name):
            self.p, self.name = parent, name
        def __enter__(self):
            self.t = time.perf_counter(); return self
        def __exit__(self, *a):
            d = time.perf_counter() - self.t
            self.p.stages[self.name] = round(self.p.stages.get(self.name, 0.0) + d, 3)
            return False

    def stage(self, name):
        return ResourceLog._Stage(self, name)

    def epoch(self, train_seconds, val_seconds, n_train=None):
        self.epochs.append({"train_s": round(train_seconds, 3), "val_s": round(val_seconds, 3),
                            "n_train": n_train})

    # ------------------------------------------------------------------- model
    def model_stats(self, model, components=None):
        tot = sum(p.numel() for p in model.parameters())
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.model["params_total"] = tot
        self.model["params_trainable"] = tr
        self.model["params_total_M"] = round(tot / 1e6, 3)
        if components:
            self.model["params_by_component"] = {
                k: sum(p.numel() for p in m.parameters()) for k, m in components.items() if m is not None}
        return self.model

    def macs(self, model, sample_inputs):
        """MACs cho MOT lan forward — dung don vi cua DeepDFA Bang 5.

        `sample_inputs` la tuple tham so dua thang vao model. Thieu thu vien thi
        ghi None chu KHONG lam hong lan chay.
        """
        try:
            from thop import profile
            m, _ = profile(model, inputs=sample_inputs, verbose=False)
            self.model["macs_per_forward"] = int(m)
            self.model["macs_per_forward_G"] = round(m / 1e9, 4)
        except Exception as e:
            self.model["macs_per_forward"] = None
            self.model["macs_note"] = "khong do duoc: %s" % type(e).__name__
        return self.model.get("macs_per_forward")

    # ------------------------------------------------------------ ngan sach token
    def tokens(self, split, lengths, windows=None):
        """`lengths`: do dai token THUC cua tung cua so da chay qua model.
        `windows`: so cua so cua tung mau (K=1 cho moc cat 512).

        Ghi ca tong L va tong L^2: phan tuyen tinh cua transformer ti le voi L,
        phan attention ti le voi L^2 — co hai so thi suy ra duoc ca hai chieu.
        """
        L = [int(x) for x in lengths]
        d = {"n_windows": len(L), "sum_tokens": sum(L), "sum_tokens_sq": sum(x * x for x in L),
             "tok_mean": round(statistics.fmean(L), 1) if L else 0,
             "tok_max": max(L) if L else 0,
             "tok_p95": (sorted(L)[int(0.95 * (len(L) - 1))] if L else 0)}
        if windows:
            W = [int(x) for x in windows]
            d.update({"n_samples": len(W), "windows_per_sample_mean": round(statistics.fmean(W), 2),
                      "windows_per_sample_max": max(W)})
        self.tok[split] = d
        return d

    # -------------------------------------------------------------------- xuat
    def dump(self, path, extra=None):
        wall = time.perf_counter() - self.t0
        tr = [e["train_s"] for e in self.epochs]
        va = [e["val_s"] for e in self.epochs]
        rec = {
            "schema": "multivd.resource/1",
            "wall_seconds_total": round(wall, 2),
            "stages_seconds": self.stages,
            "epochs_run": len(self.epochs),
            "per_epoch_seconds": {
                "train_mean": round(statistics.fmean(tr), 3) if tr else None,
                "train_sd": round(statistics.pstdev(tr), 3) if len(tr) > 1 else 0.0,
                "val_mean": round(statistics.fmean(va), 3) if va else None,
                "list": self.epochs,
            },
            "model": self.model,
            "token_budget": self.tok,
            "memory": self._peak_mb(),
        }
        # dan xuat: thong luong (DeepDFA Bang 5 dung don vi ms/mau)
        for split, d in self.tok.items():
            s = self.stages.get("infer_" + split) or self.stages.get(split)
            if s and d.get("n_samples"):
                rec.setdefault("throughput", {})[split] = {
                    "seconds_whole_split": round(s, 3),
                    "ms_per_sample": round(1000.0 * s / d["n_samples"], 3),
                    "samples_per_s": round(d["n_samples"] / s, 2),
                }
        try:
            from runtime_env import hardware_fingerprint
            rec["hardware"] = hardware_fingerprint(self.device)
        except Exception:
            rec["hardware"] = {"note": "runtime_env khong nap duoc"}
        if extra:
            rec.update(extra)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)      # ghi nguyen tu, giong runtime_env.py
        return rec
