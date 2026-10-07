#!/usr/bin/env python3
"""Doc cac file <ket qua>.resource.json roi in BANG TAI NGUYEN de dua vao bai.

Cot theo Ni et al. 2024 Bang 12 (arXiv:2408.07526) + DeepDFA Bang 5 (arXiv:2212.08108)
+ Green AI (arXiv:1907.10597). KHONG co dien nang / tien.

Gio lay TRUNG VI qua cac fold, khong lay trung binh: mot fold dung som vi early
stopping keo lech trung binh nhung khong keo lech trung vi.

    python3 tools/resource_table.py clean/results/*/baseline/seed_*/
"""
import argparse, glob, json, os, statistics, sys


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def fmt_time(s):
    if s is None:
        return "-"
    if s < 90:
        return "%.1fs" % s
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return ("%dh%02dm" % (h, m)) if h else ("%dm%02ds" % (m, sec))


def big(n):
    if n is None:
        return "-"
    for u, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= d:
            return "%.2f %s" % (n / d, u)
    return str(int(n))


def collect(d):
    files = sorted(glob.glob(os.path.join(d, "*.resource.json")))
    if not files:
        return None
    R = [json.load(open(f)) for f in files]
    tr = [r.get("train_phase") or {} for r in R]
    def pick(rs, path, default=None):
        out = []
        for r in rs:
            cur = r
            for k in path:
                cur = (cur or {}).get(k) if isinstance(cur, dict) else None
            out.append(cur if cur is not None else default)
        return out

    def first(*lists):
        """Lay danh sach DAU TIEN co it nhat mot gia tri khong None.

        `pick` luon tra ve danh sach dung do dai, nen `a or b` KHONG bao gio roi
        sang `b` — danh sach toan None van la truthy. Da dinh dung loi nay.
        """
        for L in lists:
            if any(x is not None for x in L):
                return L
        return lists[-1]

    n_sample = pick(R, ("token_budget", "test", "n_samples"))
    row = {
        "n_fold": len(R),
        "params_total": med(pick(tr, ("model", "params_total")) or pick(R, ("model", "params_total"))),
        "params_by": (tr[0].get("model", {}) or {}).get("params_by_component")
                     or (R[0].get("model", {}) or {}).get("params_by_component"),
        "macs": med(pick(tr, ("model", "macs_per_forward"))),
        "preprocess": med(pick(tr, ("stages_seconds", "preprocess"))),
        "train": med(pick(tr, ("stages_seconds", "train"))),
        "epochs_run": med(pick(tr, ("epochs_run",))),
        "per_epoch": med(pick(tr, ("per_epoch_seconds", "train_mean"))),
        "infer_test": med(pick(R, ("stages_seconds", "infer_test"))),
        "ms_sample": med(pick(R, ("throughput", "test", "ms_per_sample"))),
        "vram_train": med(pick(tr, ("memory", "gpu_alloc_peak_mb"))),
        "vram_infer": med(pick(R, ("memory", "gpu_alloc_peak_mb"))),
        "sum_tok": med(first(pick(tr, ("token_budget", "train", "sum_tokens")),
                             pick(R, ("token_budget", "train", "sum_tokens")))),
        "sum_tok_sq": med(pick(tr, ("token_budget", "train", "sum_tokens_sq"))),
        "tok_raw_mean": med(pick(tr, ("token_budget", "train", "tok_raw_mean"))),
        "frac_trunc": med(pick(tr, ("token_budget", "train", "frac_truncated"))),
        "tok_disc": med(pick(tr, ("token_budget", "train", "tokens_discarded"))),
        "n_test": med(n_sample),
        "gpu": (R[0].get("hardware") or {}).get("gpu_name"),
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        d = d.rstrip("/")
        r = collect(d)
        if r:
            r["ten"] = "/".join(d.split("/")[-3:])
            rows.append(r)
    if not rows:
        print("khong tim thay file *.resource.json nao", file=sys.stderr)
        return 1

    print("BANG TAI NGUYEN — trung vi qua %d fold | GPU: %s" % (rows[0]["n_fold"], rows[0]["gpu"]))
    print("=" * 118)
    h = ("%-30s %10s %9s %9s %9s %7s %9s %8s %9s %9s" %
         ("nhanh", "#tham so", "MACs", "tien xu ly", "huan luyen", "epoch", "suy luan", "ms/mau", "VRAM tr", "VRAM inf"))
    print(h); print("-" * 118)
    for r in rows:
        print("%-30s %10s %9s %9s %9s %7s %9s %8s %9s %9s" % (
            r["ten"][:30], big(r["params_total"]), big(r["macs"]),
            fmt_time(r["preprocess"]), fmt_time(r["train"]),
            ("%g" % r["epochs_run"]) if r["epochs_run"] is not None else "-",
            fmt_time(r["infer_test"]),
            ("%.2f" % r["ms_sample"]) if r["ms_sample"] else "-",
            ("%.0f" % r["vram_train"]) if r["vram_train"] else "-",
            ("%.0f" % r["vram_infer"]) if r["vram_infer"] else "-"))
    print()
    print("NGAN SACH TOKEN (tap train) — cot tra loi 'cat cut 512 lam mat gi'")
    print("=" * 118)
    print("%-30s %12s %14s %11s %11s %12s" %
          ("nhanh", "sum_tokens", "sum_tokens^2", "tok TB goc", "%bi cat", "token mat"))
    print("-" * 118)
    for r in rows:
        print("%-30s %12s %14s %11s %11s %12s" % (
            r["ten"][:30], big(r["sum_tok"]), big(r["sum_tok_sq"]),
            ("%.0f" % r["tok_raw_mean"]) if r["tok_raw_mean"] else "-",
            ("%.1f%%" % (100 * r["frac_trunc"])) if r["frac_trunc"] is not None else "-",
            big(r["tok_disc"])))
    print()
    for r in rows:
        if r["params_by"]:
            print("  %s — tham so theo thanh phan: %s" % (r["ten"], r["params_by"]))
    print("\n  Gio thuc chi so duoc TRONG CUNG MOT MAY. So so sanh duoc xuyen may:")
    print("  #tham so, MACs, sum_tokens (Green AI: FPO doc lap phan cung, gio thuc thi khong).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
