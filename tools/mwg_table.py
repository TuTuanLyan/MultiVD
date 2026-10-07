#!/usr/bin/env python3
"""Bang so sanh cho dinh dang ket qua cua `train_mwg.py` / `train_baseline.py` (nhanh multibabel).

    python3 tools/mwg_table.py --a <tag nhanh> --b <tag doi chung> [--ref <json so cong bo>]

KHAC dinh dang cua repo nay: xac suat khong nam trong .json ma o file `<fold>.probs.npz`
kem ben (khoa `probabilities`, `labels`, `val_probabilities`, `val_labels`, `threshold`).
Nen bon chi so duoc tinh LAI tu .npz bang dung mot ham, giong het cach `mw_n5_table.py`
lam cho cac khoi cua ta — de hai ben dat canh nhau duoc.

Ghep cap LUON theo fold. Khong bao gio lay hieu hai trung binh.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion_gate import four, calib, sp          # noqa: E402

MET = ["F1@0.5", "F1@val", "ROC", "PR"]

# Ten hien thi phai lo thanh phan cua khoi (nguoi dung neu 18/09) — xem memory
# `use-mw-assemble-babel-block-name`. Tag chay that giu nguyen de khong hong duong dan.
ALIAS = {
    "b4_sven":                  "baseline",
    "mw_b_sven_mean":           "mw",
    "mwgU_b_sven":              "mw_babel",
    "mwg_b_sven":               "mw_babel(khong U)",
    "mwgU_xAsRl_jsCjv2sven":    "mw_assemble_babel",
    "mwgU_xAsRl_jsCjvR2sven":   "mw_assemble_babel(nguon rand)",
    "mwgU_xAsRl_jsCjvE16_2sven":     "mw_assemble_babel(P1 16ep)",
    "mwgU_xAsRl_jsCjvE16_2svenNC2":  "mw_assemble_babel(P1 16ep, dich XOA COMMENT)",
    "b4_svenNC2":                    "baseline(dich XOA COMMENT)",
    "mwgU_xAsRl_jsCpy2cc":           "mw_assemble_babel(nguon js+python -> dich C/C++)",
    "b4_cc":                         "baseline(dich C/C++)",
    "mwgU_xAsRl_ccppSven2js":        "mw_assemble_babel(nguon ccpp+python -> dich JS)",
    "b4_js":                         "baseline(dich JS)",
    "mwgU_mix_jsCjvSven":            "ft_babel_mix_lang (tron js+java+python, finetune MOT lan)",
    "mwabE16_K8":                    "mw_assemble_babel (158, K=8 do thi THAT)",
    "mwabE16_K1":                    "assemble_babel_KHONG_MW (K=1, tat da cua so)",
    "mwabE16_shufG":                 "mw_assemble_babel_DOTHI_XAO (hoan vi nhan dinh)",
    "b4_sven_t5pE60":                "baseline t5p (tran 60 epoch)",
    "mwgU_xAsRl_jsCjv_t5pE60_2sven": "mw_assemble_babel t5p (tran 60 epoch)",
    "mwgU_e12sven":                  "mw_assemble_babel (P1 val 0.597)",
    "mwgU_e22sven":                  "mw_assemble_babel (P1 val 0.676)",
    "mwgU_xAsRl_jsCjv_t5p2sven":     "mw_assemble_babel (t5p)",
    "b4_sven_t5p":                   "baseline (t5p)",
    "b4_sven42":                     "baseline (seed 42, may 161)",
}
def nice(tag):
    """Ten hien thi + tag that trong ngoac, de vua doc duoc vua tra duoc duong dan."""
    return f"{ALIAS[tag]} ({tag})" if tag in ALIAS else tag


def load(tag, seed=36, root="results"):
    """Tra ve {fold: {chi so}}. Uu tien tinh lai tu .npz; khong co thi doc so da luu."""
    out = {}
    for sub in ("multiwindow", "baseline"):
        for p in sorted(glob.glob(f"{root}/{tag}/{sub}/seed_{seed}/fold*.json")):
            f = int(Path(p).stem.replace("fold", ""))
            d = json.load(open(p))
            npz = p.replace(".json", ".probs.npz")
            if Path(npz).exists():
                z = np.load(npz)
                thr = float(z["threshold"]) if "threshold" in z else \
                      calib(z["val_labels"].tolist(), z["val_probabilities"].tolist())
                m = four(z["labels"].tolist(), z["probabilities"].tolist(), thr)
                m["_nguon"] = "npz"
            else:   # khong co xac suat: dung so da luu, GHI RO de khong tuong la cung mot phep tinh
                m = {"F1@0.5": d.get("test_macro_f1_at_0.5"), "F1@val": d.get("test_macro_f1_at_valcal"),
                     "ROC": d.get("test_roc_auc"), "PR": d.get("test_pr_auc"), "_nguon": "json"}
            m["_ep"] = d.get("best_epoch")
            out[f] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="tag nhanh chinh")
    ap.add_argument("--b", required=True, help="tag doi chung")
    ap.add_argument("--seed", type=int, default=36)
    ap.add_argument("--root", default="results")
    ap.add_argument("--ref_a", default="", help="so ho cong bo cho --a, vd '0.9304,0.9077,0.9503,0.9646,0.9174'")
    ap.add_argument("--ref_b", default="", help="so ho cong bo cho --b")
    a = ap.parse_args()

    A, B = load(a.a, a.seed, a.root), load(a.b, a.seed, a.root)
    fo = sorted(set(A) & set(B))
    if not fo:
        raise SystemExit(f"khong ghep duoc fold nao: {a.a} co {sorted(A)}, {a.b} co {sorted(B)}")
    src = {A[f]["_nguon"] for f in fo} | {B[f]["_nguon"] for f in fo}
    print(f"### {nice(a.a)}  vs  {nice(a.b)}   | seed {a.seed} | {len(fo)} fold ghep cap: {fo}")
    print(f"### nguon chi so: {'/'.join(sorted(src))}"
          + ("   (json = doc so da luu, khong tinh lai)" if "json" in src else ""))

    print(f"\n{'nhanh':42s} " + " ".join(f"{m:>9s}" for m in MET) + "   epoch tot nhat")
    for tag, D in ((a.a, A), (a.b, B)):
        print(f"{nice(tag):42s} " + " ".join(f"{np.mean([D[f][m] for f in fo]):9.4f}" for m in MET)
              + "   " + ",".join(str(D[f]["_ep"]) for f in fo))

    print(f"\n## Delta ghep cap tung fold: {nice(a.a)} − {nice(a.b)}")
    print(f"{'chi so':10s} {'Delta TB':>10s} {'dem dau':>9s} {'p':>7s}   tung fold")
    for m in MET:
        ds = [A[f][m] - B[f][m] for f in fo]
        k = sum(1 for x in ds if x > 0)
        print(f"{m:10s} {np.mean(ds):+10.4f} {f'{k}/{len(fo)}':>9s} {sp(k, len(fo)):7.3f}   "
              + " ".join(f"{x:+.4f}" for x in ds))

    for tag, D, ref in ((a.a, A, a.ref_a), (a.b, B, a.ref_b)):
        if not ref:
            continue
        r = [float(x) for x in ref.split(",")]
        print(f"\n## Tai lap {tag}: ta so voi so HO cong bo (ROC)")
        print(f"{'fold':>4s} {'ho':>9s} {'ta':>9s} {'lech':>9s}")
        dl = []
        for i, f in enumerate(sorted(D)):
            if i >= len(r):
                break
            dl.append(D[f]["ROC"] - r[i])
            print(f"{f:>4d} {r[i]:9.4f} {D[f]['ROC']:9.4f} {dl[-1]:+9.4f}")
        print(f"{'TB':>4s} {np.mean(r[:len(dl)]):9.4f} {np.mean([D[f]['ROC'] for f in sorted(D)][:len(dl)]):9.4f} "
              f"{np.mean(dl):+9.4f}   | dau: {''.join('+' if x > 0 else '-' for x in dl)}"
              f" | |lech| TB {np.mean(np.abs(dl)):.4f}")


if __name__ == "__main__":
    main()
