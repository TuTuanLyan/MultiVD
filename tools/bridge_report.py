#!/usr/bin/env python3
"""bridge_report.py — đọc khối `bridge3` (và mọi cây cùng bố cục): ghép cặp theo (cây, seed, fold).

    python3 tools/bridge_report.py results/bridge3_codebert [results/bridge3_t5p ...]
    python3 tools/bridge_report.py --b plain --a rpc,rp50,cwe05 results/bridge3_codebert

In LUÔN cả bốn chỉ số (F1@0.5, F1@val, ROC-AUC, PR-AUC) kèm số fold cùng dấu (CLAUDE.md 2b), rồi
per-CWE (F1@0.5 và ROC-AUC tính lại từ test_probabilities/test_labels/test_cwe_classes). Hoà
|Δ|<1e-12 bị loại khỏi phép đếm dấu và in `~k` (memory rank-metrics-tie-need-an-explicit-epsilon).
Mỗi nhánh được ghép với `--b` (mặc định `plain`) VÀ với `baseline` (không Pha 1) trong cùng ô.
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

EPS = 1e-12
MET = [("F1@0.5", "test_macro_f1_at_0.5"), ("F1@val", "test_macro_f1_at_valcal"),
       ("ROC-AUC", "test_roc_auc"), ("PR-AUC", "test_pr_auc")]
CWE = {0: "022", 1: "078", 2: "079", 3: "089"}
TAG_RE = re.compile(r"^transfer_[a-z_]+?_(?P<src>4cwe|com|full)_l\d+p\d+_(?P<tag>.+?)(?:_adamw|_spd)?$")


def load(roots):
    cells = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "*", "seed_*", "fold*.json")):
            arm = os.path.basename(os.path.dirname(os.path.dirname(f)))
            seed = int(os.path.basename(os.path.dirname(f)).split("_")[1])
            fold = int(re.search(r"fold(\d+)", os.path.basename(f)).group(1))
            if arm == "baseline":
                tag, src = "baseline", "-"
            else:
                m = TAG_RE.match(arm)
                if not m:
                    print(f"# bo qua (khong doc duoc ten): {arm}", file=sys.stderr); continue
                tag, src = m.group("tag"), m.group("src")
            try:
                d = json.load(open(f))
            except Exception as e:
                print(f"# hong: {f}: {e}", file=sys.stderr); continue
            cells[(root, seed, fold, src, tag)] = (src, d)
    return cells


def stat(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return None
    nz = v[np.abs(v) >= EPS]; pos = int((nz > 0).sum()); ties = len(v) - len(nz)
    p = binomtest(pos, len(nz), .5).pvalue if len(nz) else float("nan")
    return v.mean(), pos, len(v), p, ties


def fmt(s):
    if not s: return f"{'-':>26}"
    d, pos, n, p, ties = s
    t = f"~{ties}" if ties else "  "
    return f"{d:>+9.4f} {pos:>3d}/{n:<3d}{t} p={p:<6.3f}"


def per_cwe(d):
    """{cwe: (f1@0.5, roc or nan)} tinh lai tu xac suat tung mau."""
    p = np.asarray(d.get("test_probabilities") or [], float)
    y = np.asarray(d.get("test_labels") or [], int)
    c = np.asarray(d.get("test_cwe_classes") or [], int)
    out = {}
    if not len(p) or len(p) != len(y) or len(y) != len(c): return out
    for k in sorted(set(c.tolist())):
        m = c == k
        if m.sum() < 2: continue
        yk, pk = y[m], p[m]
        f1 = f1_score(yk, (pk >= 0.5).astype(int), average="macro", zero_division=0)
        roc = roc_auc_score(yk, pk) if len(set(yk.tolist())) == 2 else float("nan")
        out[k] = (f1, roc, int(m.sum()))
    return out


def ctl_pairs(cells, tag, ctl):
    """Cac cap (o cua `tag`, o doi chung) cung (cay, seed, fold); baseline khong co nguon nen bo qua src."""
    out = []
    for (r, s, f, src, tg), (_, d) in cells.items():
        if tg != tag: continue
        key = (r, s, f, "-", "baseline") if ctl == "baseline" else (r, s, f, src, ctl)
        if key in cells: out.append((d, cells[key][1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--b", default="plain", help="nhanh doi chung (tag)")
    ap.add_argument("--a", default=None, help="nhanh can do, cac tag cach nhau dau phay (vd rpc,rp50); mac dinh: moi tag khac --b/baseline")
    a = ap.parse_args()
    cells = load(a.roots)
    if not cells:
        print("khong co o nao"); return
    tags = sorted({k[4] for k in cells})
    print(f"{len(cells)} o. Tag: {tags}")
    # tuyet doi
    print(f"\n=== TUYET DOI (trung binh qua o co san) ===\n{'tag':<10}{'n':>3}  " + "".join(f"{m:>10}" for m, _ in MET) + f"{'best_ep':>9}")
    for t in tags:
        ds = [d for (r, s, f, sr, tg), (src, d) in cells.items() if tg == t]
        row = "".join(f"{np.mean([d.get(key) or np.nan for d in ds]):>10.4f}" for _, key in MET)
        be = np.mean([d.get("best_epoch") or np.nan for d in ds])
        print(f"{t:<10}{len(ds):>3}  {row}{be:>9.1f}")
        h = ds[0].get("hyperparameters") or {}
        if h.get("replay_mu") or h.get("phase2_lambda_cwe"):
            print(f"{'':13}replay_mu={h.get('replay_mu')} epochs={h.get('replay_epochs')} stratify={h.get('replay_stratify')} "
                  f"l_cwe_nguon={h.get('replay_lambda_cwe')} l_cwe_dich={h.get('phase2_lambda_cwe')} rows={h.get('replay_rows')} "
                  f"mu: {h.get('replay_mu_schedule')}")
    arms = a.a.split(",") if a.a else [t for t in tags if t not in (a.b, "baseline")]
    for ctl in (a.b, "baseline"):
        print(f"\n=== Δ so voi `{ctl}` — ghep cap theo (cay, seed, fold) ===")
        print(f"{'nhanh':<10}{'n':>3}  " + "".join(f"{m:>28}" for m, _ in MET))
        rows = list(arms) + ([a.b] if ctl == "baseline" and a.b not in arms else [])
        for t in rows:
            if t == ctl: continue
            pairs = ctl_pairs(cells, t, ctl)
            if not pairs:
                print(f"{t:<10}  0  (khong ghep duoc cap nao)"); continue
            cols = [stat([(x.get(key) or np.nan) - (y.get(key) or np.nan) for x, y in pairs]) for _, key in MET]
            print(f"{t:<10}{len(pairs):>3}  " + "".join(f"  {fmt(s)}" for s in cols))
        # per-CWE
        print(f"\n--- per-CWE Δ so voi `{ctl}` (F1@0.5 | ROC-AUC), hang test tb trong ngoac ---")
        for t in rows:
            if t == ctl: continue
            pairs = ctl_pairs(cells, t, ctl)
            if not pairs: continue
            dd = defaultdict(lambda: {"f1": [], "roc": [], "n": []})
            for x, y in pairs:
                px, py = per_cwe(x), per_cwe(y)
                for k in px:
                    if k in py:
                        dd[k]["f1"].append(px[k][0] - py[k][0]); dd[k]["roc"].append(px[k][1] - py[k][1]); dd[k]["n"].append(px[k][2])
            line = f"{t:<10}"
            for k in sorted(dd):
                sf, sr = stat(dd[k]["f1"]), stat(dd[k]["roc"])
                line += f" | {CWE.get(k, k)}({np.mean(dd[k]['n']):.0f}) F1 {sf[0]:+.3f} {sf[1]}/{sf[2]}  ROC {sr[0]:+.3f} {sr[1]}/{sr[2]}" if sf and sr else f" | {CWE.get(k,k)} -"
            print(line)
    print("\nBac 1 (n=3 fold, seed 42): chi SANG LOC. Phan chac la SO FOLD CUNG DAU; p o n=3 khong phan biet duoc gi.")


if __name__ == "__main__":
    main()
