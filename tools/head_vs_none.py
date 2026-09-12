#!/usr/bin/env python3
"""head_vs_none.py — HEAD PHU co them gi ngoai "finetune HAI LAN thuan" khong?

    python3 tools/head_vs_none.py                 # quet toan bo results/
    python3 tools/head_vs_none.py --dump-cells    # in tung o cua tap "Pha 1 sap"

`aux_mode=none` CHINH LA doi chung "finetune 2 lan": Pha 1 fine-tune backbone tren
ccpp+js nhi phan (vul/non-vul) KHONG head phu, Pha 2 fine-tune tiep tren Python.
`latent_bottleneck` chi khac DUNG MOT thu: co them head phu o Pha 1.
Nen Δ(latent_bottleneck − none) la gia tri rieng cua HEAD, khong phai cua transfer.

BA CAI BAY FILE NAY CHAN (CLAUDE.md muc 2b):
  1. Doc mot chi so          -> in ca F1@0.5, F1@val, ROC-AUC, PR-AUC.
  2. Doc trung binh khong doc dem dau -> in +/n va p canh moi trung binh. O day no doi
     han ket luan: trung binh +0.0163 nhung 5 o "Pha 1 sap" gop gan het.
  3. Gop o cua nhieu khoi    -> ghep cap theo (cay, backbone, tag Pha 1, optimizer,
     rho, seed, fold). O le bi BO va bao ro.

BAY THU TU — DA MAC PHAI 12/09: cac khoi quet sieu tham so (bridge3, opt1) chay HANG CHUC
bien the Pha 2 tren CUNG mot tag Pha 1. Neu khoa ghep cap khong phan biet ten nhanh thi
chung de len nhau im lang (dung loi `tsize_report.py` 11/09). Nen o day chi giu nhanh
CAU HINH GOC: ten nhanh phai dung bang `transfer_<mode>_<tag>[_<opt>]`, va so va cham
khoa con lai duoc in ra de kiem — no PHAI bang 0.
"""
import argparse, json, glob, os, re, sys
from collections import defaultdict
import statistics as st
from math import comb

M = [("F1@0.5","f1"), ("F1@val","f1v"), ("ROC-AUC","roc"), ("PR-AUC","pr")]
BB = {"microsoft/codebert-base":"codebert", "Salesforce/codet5p-220m-bimodal":"t5p",
      "Salesforce/codet5p-110m-embedding":"t5pe", "microsoft/unixcoder-base":"unixcoder"}
SAP_ROC = 0.55   # none co ROC duoi muc nay = Pha 1 sap, doc RIENG

def sign_p(k, n):
    if n == 0: return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)

def tag_of(ck, mode):
    m = re.search(r'/([^/]+)/seed_[^/]+/best\.pt$', ck or "")
    if not m or "__" not in m.group(1): return "?"
    rest = m.group(1).split("__", 1)[1]
    if not rest.startswith(mode): return rest
    return rest[len(mode):].lstrip("_") or "-"

def load(roots):
    rows = []
    for root in roots:
        for p in glob.glob(os.path.join(root, "*", "*", "seed_*", "fold*.json")):
            try: d = json.load(open(p))
            except Exception: continue
            h = d.get("hyperparameters", {})
            arm = os.path.basename(os.path.dirname(os.path.dirname(p)))
            base = (arm == "baseline")
            rows.append(dict(
                tree=os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p)))),
                arm=arm, bb=BB.get(h.get("model_name"), h.get("model_name")),
                mode="BASELINE" if base else h.get("aux_mode"),
                tag="-" if base else tag_of(h.get("source_checkpoint"), h.get("aux_mode") or ""),
                opt="-" if base else h.get("phase2_optimizer"),
                rho=h.get("sam_rho"), var=h.get("sam_variant"),
                seed=d.get("seed"), fold=d.get("fold"),
                f1=d.get("test_macro_f1_at_0.5"), f1v=d.get("test_macro_f1_at_valcal"),
                roc=d.get("test_roc_auc"), pr=d.get("test_pr_auc"), path=p))
    return rows

def canonical(r):
    tag = "" if r["tag"] in ("-", None) else "_" + r["tag"]
    suf = "" if r["opt"] == "recadam" else f"_{r['opt']}"
    return r["arm"] == f"transfer_{r['mode']}{tag}{suf}"

def cells(v, ia, ib):
    out = []
    for _, key in M:
        dd = [x[ia][key] - x[ib][key] for x in v
              if x[ia].get(key) is not None and x[ib].get(key) is not None]
        if not dd: out.append(f"{'-':>20}"); continue
        pos = sum(1 for z in dd if z > 0)
        out.append(f"{st.mean(dd):+7.4f} {pos:>2}/{len(dd):<2} p={sign_p(pos,len(dd)):<4.2f}")
    return " | ".join(out)

def head(lbl, w=46):
    return f"{lbl:{w}} {'n':>3} | " + " | ".join(f"{n:>20}" for n, _ in M)

def sam(rho, var): return "SAM-tat" if not rho else f"{var}{rho}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["results"])
    ap.add_argument("--dump-cells", action="store_true")
    ap.add_argument("--tags", action="store_true",
                    help="chi liet ke tag Pha 1 co san cho tung nhanh roi thoat")
    a = ap.parse_args()
    rows = load(a.roots or ["results"])
    base = {(r["tree"], r["bb"], r["seed"], r["fold"]): r for r in rows if r["mode"] == "BASELINE"}
    KEY = lambda r: (r["tree"], r["bb"], r["tag"], r["opt"], r["rho"], r["var"], r["seed"], r["fold"])
    by, clash = {}, 0
    for r in rows:
        if r["mode"] not in ("none", "latent_bottleneck") or not canonical(r): continue
        k = (KEY(r), r["mode"])
        if k in by: clash += 1
        by[k] = r
    print(f"{len(rows)} o doc duoc; {len(by)} o cau hinh goc; {clash} va cham khoa con lai "
          f"({'OK' if clash == 0 else '!! PHAI SUA'})\n")
    if a.tags:
        for m in ("none", "latent_bottleneck"):
            t = sorted({k[2] for (k, mm) in by if mm == m})
            print(f"  tag Pha 1 co nhanh {m:18}: {t}")
        for m in ("none", "latent_bottleneck"):
            t = sorted({k[2] for (k, mm) in by if mm == m and "l0p05" in (k[2] or "")})
            print(f"  ... trong do tag l0p05 cua {m:14}: {t if t else 'KHONG CO'}")
        return 0

    pairs = [(r, by[(k, "none")], k) for (k, m), r in by.items()
             if m == "latent_bottleneck" and (k, "none") in by]
    sap_p = [p for p in pairs if p[1]["roc"] is not None and p[1]["roc"] < SAP_ROC]
    ok = [p for p in pairs if p not in sap_p]

    print("=" * 122)
    print("A. latent_bottleneck − none   = gia tri rieng cua HEAD PHU")
    print("=" * 122); print(head("tap"))
    print(f"{'TAT CA':46} {len(pairs):>3} | " + cells(pairs, 0, 1))
    print(f"{f'  (i) o none SAP (ROC < {SAP_ROC})':46} {len(sap_p):>3} | " + cells(sap_p, 0, 1))
    print(f"{'  (ii) o none BINH THUONG':46} {len(ok):>3} | " + cells(ok, 0, 1))
    print("\n  o sap thuoc:", sorted({(p[2][0], p[2][1], p[2][2], p[2][3]) for p in sap_p}))
    if a.dump_cells:
        for r, n, k in sorted(sap_p, key=lambda x: x[2][7]):
            print(f"    fold{k[7]}  none F1={n['f1']:.4f} ROC={n['roc']:.4f}"
                  f"   |  head F1={r['f1']:.4f} ROC={r['roc']:.4f}")
    for lbl, grp in (("B. tap BINH THUONG, theo backbone x optimizer x SAM",
                      lambda k: (k[1], k[3], sam(k[4], k[5]))),
                     ("C. tap BINH THUONG, gop backbone, theo optimizer x SAM",
                      lambda k: (k[3], sam(k[4], k[5])))):
        g = defaultdict(list)
        for r, n, k in ok: g[grp(k)].append((r, n))
        print("\n" + "=" * 122); print(lbl); print("=" * 122); print(head("tap"))
        for kk in sorted(g, key=lambda x: tuple(str(y) for y in x)):
            print(f"{' '.join(str(y) for y in kk):46} {len(g[kk]):>3} | " + cells(g[kk], 0, 1))

    # D) moi nhanh so voi BASELINE, chi tren o co CA HAI nhanh => so sanh cong bang
    g2 = defaultdict(lambda: defaultdict(list))
    for r, n, k in ok:
        b = base.get((k[0], k[1], k[6], k[7]))
        if b is None: continue
        g2[(k[3], sam(k[4], k[5]))]["none"].append((n, b))
        g2[(k[3], sam(k[4], k[5]))]["lb"].append((r, b))
    print("\n" + "=" * 122)
    print("D. Δ vs BASELINE (khong Pha 1) — CHI tren o co CA HAI nhanh, cung tap fold")
    print("=" * 122)
    print(f"{'optimizer / SAM':24} {'nhanh':22} {'n':>3} | " + " | ".join(f"{n:>20}" for n, _ in M))
    for kk in sorted(g2, key=lambda x: tuple(str(y) for y in x)):
        for m, lbl in (("none", "none = finetune 2 lan"), ("lb", "latent_bottleneck")):
            print(f"{' '.join(str(y) for y in kk):24} {lbl:22} {len(g2[kk][m]):>3} | "
                  + cells(g2[kk][m], 0, 1))
        print()
    print("CANH BAO: fold KHONG doc lap khi dung lai cung bo fold dich qua nhieu backbone/nguon")
    print("=> p lac quan. Phan chac la SO FOLD CUNG DAU. San nhieu cung-GPU 0.010.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
