#!/usr/bin/env python3
"""Bao cao khoi OPT1 — toi uu RecAdam (RESEARCH_2026-09-06_recadam.md §5.5).

    python3 tools/opt1_report.py [results/opt1_t5p ...]

Moi Δ ghep cap theo (cay ket qua, seed, fold): baseline cua CHINH may do, CHINH fold
do. Khong bao gio lay hieu hai trung binh (CLAUDE.md muc 2). Metric chinh: F1@0.5 —
khop tieu chi chon checkpoint (val F1@0.5). ROC-AUC va F1@nguong-val luon di kem.

Ba bang:
  1. Moi cau hinh vs baseline.
  2. Moi cau hinh vs hai doi chung cung phien: c5000_t0p05 (RecAdam mac dinh) va plain (AdamW).
  3. Lich su val: epoch tot nhat theo F1 (da chon) so voi epoch tot nhat theo ROC-AUC.
Mot o xuat hien o hai cay -> loi, khong gop.
"""
import json
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ARM_RE = re.compile(r"^transfer_latent_bottleneck_(?P<src>[a-z0-9]+)_l(?P<lam>[0-9p]+)_(?P<tag>.+?)(?P<adamw>_adamw)?$")
CONTROLS = (("c5000_t0p05", "recadam"), ("plain", "adamw"))


def load_tree(root):
    root = Path(root)
    base, arms = {}, defaultdict(dict)
    for f in sorted(root.glob("baseline/seed_*/fold*.json")):
        d = json.loads(f.read_text())
        base[(str(root), d["seed"], d["fold"])] = d
    for arm_dir in sorted(root.glob("transfer_latent_bottleneck_*")):
        m = ARM_RE.match(arm_dir.name)
        if not m:
            print(f"!! bo qua ten nhanh khong doc duoc: {arm_dir.name}", file=sys.stderr)
            continue
        opt = "adamw" if m.group("adamw") else "recadam"
        key = (m.group("src"), m.group("tag"), opt)
        for f in sorted(arm_dir.glob("seed_*/fold*.json")):
            d = json.loads(f.read_text())
            cell = (str(root), d["seed"], d["fold"])
            if cell in arms[key]:
                raise SystemExit(f"O trung: {key} {cell}")
            arms[key][cell] = d
    return base, arms


def sign_test_p(deltas):
    """p hai phia cua kiem dau (binomial), bo qua hieu = 0. Wilcoxon neu co scipy."""
    d = [x for x in deltas if x != 0]
    if not d:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        n, k = len(d), sum(1 for x in d if x > 0)
        tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
        return min(1.0, 2 * tail)


def summarize(pairs, key_a="test_macro_f1_at_0.5"):
    """pairs: list of (arm_json, ref_json). Tra ve dict cac thong ke ghep cap."""
    d_f1 = [a[key_a] - b[key_a] for a, b in pairs]
    d_auc = [(a["test_roc_auc"] or 0) - (b["test_roc_auc"] or 0) for a, b in pairs]
    d_vc = [a["test_macro_f1_at_valcal"] - b["test_macro_f1_at_valcal"] for a, b in pairs]
    n = len(pairs)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "n": n,
        "f1": mean(d_f1), "f1_pos": sum(x > 0 for x in d_f1), "f1_min": min(d_f1) if d_f1 else float("nan"),
        "f1_max": max(d_f1) if d_f1 else float("nan"), "p": sign_test_p(d_f1),
        "auc": mean(d_auc), "auc_pos": sum(x > 0 for x in d_auc),
        "vc": mean(d_vc), "vc_pos": sum(x > 0 for x in d_vc),
        "per_fold": {a["fold"]: round(a[key_a] - b[key_a], 4) for a, b in pairs},
    }


def fmt(s):
    return (f"{s['n']:>3} | {s['f1']:+.4f} {s['f1_pos']:>2}/{s['n']:<2} [{s['f1_min']:+.4f},{s['f1_max']:+.4f}] p={s['p']:.3f} "
            f"| AUC {s['auc']:+.4f} {s['auc_pos']:>2}/{s['n']:<2} | F1vc {s['vc']:+.4f} {s['vc_pos']:>2}/{s['n']:<2}")


def val_history_stats(d):
    """Epoch chon theo F1 vs epoch tot nhat theo AUC, tu runtime sidecar."""
    rt = (d.get("runtime") or {})
    rec = rt.get("phase2") or rt.get("baseline_train") or {}
    hist = rec.get("val_history") or []
    if not hist:
        return None
    by_f1 = max(hist, key=lambda h: (h["val_macro_f1"], h["epoch"]))
    aucs = [h for h in hist if h.get("val_roc_auc") is not None]
    if not aucs:
        return None
    by_auc = max(aucs, key=lambda h: (h["val_roc_auc"], -h["epoch"]))
    return {
        "epochs_run": len(hist), "ep_f1": by_f1["epoch"], "ep_auc": by_auc["epoch"],
        "auc_at_f1": by_f1["val_roc_auc"], "auc_max": by_auc["val_roc_auc"],
        "f1_at_auc": by_auc["val_macro_f1"], "f1_max": by_f1["val_macro_f1"],
    }


def main(roots):
    base, arms = {}, defaultdict(dict)
    for r in roots:
        b, a = load_tree(r)
        base.update(b)
        for k, v in a.items():
            for cell, d in v.items():
                if cell in arms[k]:
                    raise SystemExit(f"O trung giua hai cay: {k} {cell}")
                arms[k][cell] = d
    print(f"baseline: {len(base)} o | nhanh: {len(arms)} cau hinh x nguon | "
          f"tong o Pha 2: {sum(len(v) for v in arms.values())}")
    srcs = sorted({k[0] for k in arms})
    tags = sorted({(k[1], k[2]) for k in arms}, key=lambda t: (t[1] != "recadam", t[0]))

    print("\n=== 1. Δ vs baseline (cung cay/seed/fold). F1@0.5 chinh; AUC va F1@nguong-val kem ===")
    for src in srcs:
        print(f"\n[{src}]  tag (opt)                  n | ΔF1@0.5  +/n  [min,max]  p | ΔAUC | ΔF1vc")
        for tag, opt in tags:
            cells = arms.get((src, tag, opt), {})
            pairs = [(d, base[c]) for c, d in cells.items() if c in base]
            if not pairs:
                continue
            print(f"  {tag:<22}({opt:<7}) {fmt(summarize(pairs))}")
    # gop hai nguon
    print(f"\n[4cwe+com gop]")
    for tag, opt in tags:
        pairs = []
        for src in srcs:
            pairs += [(d, base[c]) for c, d in arms.get((src, tag, opt), {}).items() if c in base]
        if pairs:
            print(f"  {tag:<22}({opt:<7}) {fmt(summarize(pairs))}")

    print("\n=== 2. Δ vs doi chung CUNG PHIEN (cung cay/seed/fold/nguon) ===")
    for ctag, copt in CONTROLS:
        print(f"\n--- so voi {ctag} ({copt}) ---")
        for tag, opt in tags:
            if (tag, opt) == (ctag, copt):
                continue
            pairs = []
            for src in srcs:
                ref = arms.get((src, ctag, copt), {})
                pairs += [(d, ref[c]) for c, d in arms.get((src, tag, opt), {}).items() if c in ref]
            if pairs:
                print(f"  {tag:<22}({opt:<7}) {fmt(summarize(pairs))}")

    print("\n=== 3. Lich su val: epoch chon (F1@0.5) vs epoch tot nhat theo ROC-AUC ===")
    rows = []
    for (src, tag, opt), cells in arms.items():
        for c, d in cells.items():
            st = val_history_stats(d)
            if st:
                rows.append((tag, opt, src, c[2], st))
    for c, d in base.items():
        st = val_history_stats(d)
        if st:
            rows.append(("baseline", "adamw", "-", c[2], st))
    if not rows:
        print("  (chua co o nao mang val_history — ket qua chay truoc ban va 06/09)")
    else:
        same = sum(1 for r in rows if r[4]["ep_f1"] == r[4]["ep_auc"])
        gap = [abs(r[4]["ep_f1"] - r[4]["ep_auc"]) for r in rows]
        loss_auc = [r[4]["auc_max"] - r[4]["auc_at_f1"] for r in rows]
        loss_f1 = [r[4]["f1_max"] - r[4]["f1_at_auc"] for r in rows]
        print(f"  {len(rows)} lan chay | cung epoch: {same} | |Δepoch| trung binh {sum(gap)/len(gap):.2f}, "
              f"toi da {max(gap)} | val AUC bo lo khi chon theo F1: trung binh {sum(loss_auc)/len(loss_auc):.4f}, "
              f"toi da {max(loss_auc):.4f} | val F1 bo lo neu chon theo AUC: trung binh {sum(loss_f1)/len(loss_f1):.4f}")
        by = defaultdict(list)
        for tag, opt, src, fold, st in rows:
            by[(tag, opt)].append(st)
        print(f"  {'tag (opt)':<30} n  cung-epoch  |Δep|  AUC bo lo  F1 bo lo")
        for (tag, opt), sts in sorted(by.items()):
            n = len(sts)
            print(f"  {tag+' ('+opt+')':<30} {n:<2} {sum(s['ep_f1']==s['ep_auc'] for s in sts):>3}/{n:<3} "
                  f"{sum(abs(s['ep_f1']-s['ep_auc']) for s in sts)/n:>5.2f}  "
                  f"{sum(s['auc_max']-s['auc_at_f1'] for s in sts)/n:>8.4f}  "
                  f"{sum(s['f1_max']-s['f1_at_auc'] for s in sts)/n:>8.4f}")

    # Pha 1 val cua tung nguon (tu JSON moi), de nguoi doc gan vao moi o
    p1 = defaultdict(set)
    for (src, tag, opt), cells in arms.items():
        for d in cells.values():
            v = d.get("phase1_val_macro_f1")
            if v is not None:
                p1[(src, d["seed"])].add(round(v, 4))
    if p1:
        print("\nPha 1 val (nguon, seed): " + ", ".join(f"{k[0]}/s{k[1]}={sorted(v)}" for k, v in sorted(p1.items())))


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/opt1_t5p"])
