#!/usr/bin/env python3
"""Bang n=5 cho khoi MW_ASSEMBLE — bon chi so, dem dau, va khoang do CHON DOI CHUNG.

    python3 tools/mw_n5_table.py --run results/mw_assemble_codebert --backbone codebert

VI SAO CO CONG CU RIENG. Kho local co 20 cay baseline codebert va 26 cay t5p du 5 fold,
CUNG chu ky sieu tham so. Chon MOT cay la tu quyet ket luan (FACTS §61: doi doi chung
da lat mot ket qua tu "manh" sang "khong dat"). Luat da chot TRUOC khi nhin Delta, ghi o
CURRENT_RUN.md:
  1. loai cay `sz*` (baseline CO TAP NHO, sz76 sap ve ~0.54)
  2. khu trung lap theo bo 5 so (vai cay la ban sao byte cua nhau)
  3. doi chung chinh = TRUNG VI THEO TUNG FOLD cua so cay con lai
  4. kem khoang min-max cua Delta qua TAT CA cay doi chung
Ghep cap luon theo fold; KHONG BAO GIO lay hieu hai trung binh.
"""
import argparse, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from late_fusion_gate import four, fuse, sp   # noqa: E402

MET = ["F1@0.5", "F1@val", "ROC", "PR"]
SIG = dict(batch_size=16, max_length=512, truncation_strategy="head_middle_tail",
           patience=5, epochs=30, learning_rate=2e-5)


def cell_metrics(d):
    """Bon chi so tinh LAI tu xac suat, dung cung mot ham voi nhanh ghep."""
    from late_fusion_gate import calib
    thr = calib(d["val_labels"], d["val_probabilities"])
    return four(d["test_labels"], d["test_probabilities"], thr)


def load_arm(run, arm, seed):
    out = {}
    for p in sorted(glob.glob(f"{run}/{arm}/seed_{seed}/fold*.json")):
        f = int(Path(p).stem.replace("fold", ""))
        out[f] = json.load(open(p))
    return out


def baseline_pool(backbone, seed, exclude_run):
    """Cac cay baseline DU 5 FOLD, khop chu ky, khong phai `sz*`, da khu trung lap."""
    model = "microsoft/codebert-base" if backbone == "codebert" else "Salesforce/codet5p-220m-bimodal"
    trees = defaultdict(dict)
    for p in glob.glob("results*/**/baseline/seed_*/fold*.json", recursive=True):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        hp = d.get("hyperparameters", {})
        if hp.get("model_name") != model:                     continue
        if hp.get("data_root") != "data/sven_python_folds_norm": continue
        if d.get("seed") != seed:                              continue
        if any(hp.get(k) != v for k, v in SIG.items()):        continue
        # BAT BUOC co xac suat val+test: bon chi so phai tinh bang CUNG mot ham voi
        # nhanh ghep, khong duoc doc so da luu san (nguong calib co the khac ham).
        if not (d.get("val_probabilities") and d.get("val_labels")
                and d.get("test_probabilities")):                  continue
        t = p[:p.index("/baseline/")]
        if "/sz" in "/" + t or Path(t).name.startswith("sz"):  continue   # loai co tap nho
        if t == exclude_run:                                   continue   # de bao rieng
        trees[t][hp.get("fold")] = d
    part = {t: sorted(v) for t, v in trees.items() if set(v) != {1, 2, 3, 4, 5}}
    full = {t: v for t, v in trees.items() if set(v) == {1, 2, 3, 4, 5}}
    # khu trung lap theo bo 5 so ROC (ban sao byte cho ra bo giong het)
    seen, keep = {}, {}
    for t in sorted(full):
        key = tuple(round(full[t][f]["test_roc_auc"], 6) for f in range(1, 6))
        if key in seen:
            continue
        seen[key] = t
        keep[t] = full[t]
    return keep, len(full) - len(keep), len(part)


def signs(deltas, eps=1e-12):
    pos = sum(1 for d in deltas if d > eps)
    return pos, len(deltas), sp(pos, len(deltas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--backbone", required=True, choices=("codebert", "t5p"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="mwK8 mwTR mwK1")
    ap.add_argument("--fuse", default="", help="hai nhanh de ghep, vd 'babel babelTR'. "
                    "De trong = tu doan: hai nhanh dau tien trong --arms")
    a = ap.parse_args()

    arms = {n: load_arm(a.run, n, a.seed) for n in a.arms.split()}
    arms = {n: v for n, v in arms.items() if v}
    own = load_arm(a.run, "baseline", a.seed)

    # nhanh ghep: assemble tu HAI nhanh duoc chi dinh, tren CUNG fold
    fa, fb = (a.fuse.split() if a.fuse else a.arms.split()[:2] + ["", ""])[:2]
    if fa in arms and fb in arms:
        for mode in ("g05", "const", "logreg"):
            cells = {}
            for f in sorted(set(arms[fa]) & set(arms[fb])):
                A, B = arms[fa][f], arms[fb][f]
                if list(A["test_labels"]) != list(B["test_labels"]) or \
                   list(A["val_labels"]) != list(B["val_labels"]):
                    raise SystemExit(f"fold {f}: THU TU HANG hai nhanh khac nhau — khong ghep duoc")
                pt, thr = fuse(A, B, mode)
                cells[f] = {"test_labels": A["test_labels"], "test_probabilities": pt.tolist(),
                            "val_labels": A["val_labels"], "val_probabilities": A["val_probabilities"],
                            "_thr": thr}
            arms[f"asm_{mode}"] = cells

    # bang tuyet doi
    M = {}
    for n, v in arms.items():
        M[n] = {}
        for f, d in v.items():
            M[n][f] = four(d["test_labels"], d["test_probabilities"], d["_thr"]) if "_thr" in d \
                      else cell_metrics(d)

    pool, dup, part = baseline_pool(a.backbone, a.seed, a.run)
    Bm = {t: {f: cell_metrics(d) for f, d in v.items()} for t, v in pool.items()}
    folds_all = sorted({f for v in M.values() for f in v})
    med = {f: {m: float(np.median([Bm[t][f][m] for t in Bm])) for m in MET} for f in range(1, 6)}

    print(f"### {a.run} | backbone {a.backbone} | seed {a.seed}")
    print(f"### doi chung: TRUNG VI theo fold cua {len(Bm)} cay baseline local "
          f"(bo {dup} ban trung lap, {part} cay thieu fold; da loai cay sz* va cay "
          f"khong luu xac suat val)\n")

    hdr = f"{'nhanh':12s} {'n':>2s} " + " ".join(f"{m:>8s}" for m in MET)
    print("## Tuyet doi (trung binh qua fold)")
    print(hdr)
    print(f"{'baseline(TV)':12s} {5:>2d} " + " ".join(f"{np.mean([med[f][m] for f in range(1,6)]):8.4f}" for m in MET))
    if own:
        fo = sorted(own)
        ownM = {f: cell_metrics(own[f]) for f in fo}
        print(f"{'baseline(khoi)':12s} {len(fo):>2d} " + " ".join(f"{np.mean([ownM[f][m] for f in fo]):8.4f}" for m in MET))
    ORDER = [x for x in a.arms.split() if x in M] + [x for x in ("asm_g05", "asm_const", "asm_logreg") if x in M]
    for n in ORDER:
        fo = sorted(M[n])
        print(f"{n:12s} {len(fo):>2d} " + " ".join(f"{np.mean([M[n][f][m] for f in fo]):8.4f}" for m in MET))

    print("\n## Delta so voi baseline (trung vi theo fold) — ghep cap TUNG FOLD")
    print(f"{'nhanh':12s} " + " ".join(f"{m:>18s}" for m in MET))
    for n in ORDER:
        fo = sorted(M[n]); cols = []
        for m in MET:
            ds = [M[n][f][m] - med[f][m] for f in fo]
            k, nn, p = signs(ds)
            cols.append(f"{np.mean(ds):+7.4f} {k}/{nn} p{p:.3f}")
        print(f"{n:12s} " + " ".join(f"{c:>18s}" for c in cols))

    print("\n## Khoang Delta khi DOI cay doi chung (min..max trung binh Delta qua cac cay)")
    print(f"{'nhanh':12s} " + " ".join(f"{m:>20s}" for m in MET))
    for n in [x for x in (fa, fb, "asm_g05") if x in M]:
        fo = sorted(M[n]); cols = []
        for m in MET:
            per = [np.mean([M[n][f][m] - Bm[t][f][m] for f in fo]) for t in Bm]
            cols.append(f"{min(per):+7.4f} .. {max(per):+7.4f}")
        print(f"{n:12s} " + " ".join(f"{c:>20s}" for c in cols))

    print("\n## Delta GIUA cac nhanh — day moi la cau hoi cua de xuat")
    others = [x for x in a.arms.split() if x not in (fa, fb) and x in M]
    pairs = [(fb, fa), ("asm_g05", fa), ("asm_const", fa), ("asm_logreg", fa)] \
            + [(fa, o) for o in others]
    print(f"{'phep so':22s} " + " ".join(f"{m:>18s}" for m in MET))
    for x, y in pairs:
        if x not in M or y not in M:
            continue
        fo = sorted(set(M[x]) & set(M[y])); cols = []
        for m in MET:
            ds = [M[x][f][m] - M[y][f][m] for f in fo]
            k, nn, p = signs(ds)
            cols.append(f"{np.mean(ds):+7.4f} {k}/{nn} p{p:.3f}")
        print(f"{x+' - '+y:22s} " + " ".join(f"{c:>18s}" for c in cols))

    print("\n## Tung fold (ROC) — de thay fold sap")
    print(f"{'nhanh':12s} " + " ".join(f"{'f'+str(f):>8s}" for f in range(1, 6)))
    print(f"{'baseline(TV)':12s} " + " ".join(f"{med[f]['ROC']:8.4f}" for f in range(1, 6)))
    for n in [x for x in ORDER if x in M]:
        print(f"{n:12s} " + " ".join(f"{M[n][f]['ROC']:8.4f}" if f in M[n] else f"{'-':>8s}" for f in range(1, 6)))


if __name__ == "__main__":
    main()
