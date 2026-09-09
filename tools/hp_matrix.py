#!/usr/bin/env python3
"""hp_matrix.py — ma tran (lambda Pha 1) x (rho SAM/ASAM) cho TUNG backbone.

Nguoi dung 09/09: "bang de theo doi cai thay doi cung backbone cua cac hyper param lambda
va p cua asam de xem cua tung backbone xem nhu nao" + "tong hop tu truoc luon cac phan chay".

Moi Delta ghep cap theo (CAY ket qua, backbone, seed, fold) voi arm `baseline` cung o do —
khong bao gio bac cau qua may hay qua khoi (CLAUDE.md muc 2 va 4). Xuat JSON cho trang.

--- lambda lay o dau (FACTS §29) ---
KHONG suy tu ten arm mot cach may moc: quy uoc ten DOI GIUA CAC DOT va con phu thuoc KHO.
Da doi chieu bang NOI DUNG checkpoint (`training_args.lambda_cwe`) tren 26 checkpoint con
song, ket qua:
  * `_l<so>`      -> so/100      (_l02=0.02, _l05=0.05, _l50=0.50, _l100=1.00)  [da do]
  * `_l<a>p<b>`   -> a.b         (_l0p05=0.05)                                   [da do]
  * khong hau to, kho `model/s42/` -> 0.05   [da do tren 20 checkpoint]
  * khong hau to, kho khac        -> `hyperparameters.lambda_cwe` (l1/m1/n1/r1 deu 0.2)  [da do tren 6]
Truong `phase1_lambda_cwe` (neu co) LUON thang, vi no ghi tu chinh checkpoint.
Moi dong deu kem `lamsrc` de biet con so den tu dau.

--- rho ---
`sam_rho`/`sam_variant` trong hyperparameters la cua PHA 2. Cac nhanh `_sam`, `_sam1`,
`_sam1r01` cua khoi m1/n1 la SAM o PHA 1 — truc khac han, danh dau `p1sam` de tach ra.
"""
import json, os, re, sys
from collections import defaultdict

MET = [("f1", "test_macro_f1_at_0.5"), ("f1v", "test_macro_f1_at_valcal"),
       ("roc", "test_roc_auc"), ("pr", "test_pr_auc")]
BB = {"microsoft/codebert-base": "codebert", "microsoft/unixcoder-base": "unixcoder",
      "Salesforce/codet5p-220m-bimodal": "t5p", "Salesforce/codet5-base": "codet5",
      "Salesforce/codet5p-110m-embedding": "t5pe"}

def lam_of(d, h, arm):
    """Tra ve (gia tri, nguon). Thu tu tin cay giam dan."""
    v = d.get("phase1_lambda_cwe")
    if v is not None: return float(v), "ghi trong ket qua"
    m = re.search(r"_l(\d+)p(\d+)(?:_|$)", arm)
    if m: return float(f"{m.group(1)}.{m.group(2)}"), "ten arm"
    m = re.search(r"_l(\d+)(?:_|$)", arm)
    if m: return int(m.group(1)) / 100.0, "ten arm"
    sc = h.get("source_checkpoint") or ""
    if sc.startswith("model/s42/"): return 0.05, "quy uoc kho s42"
    v = h.get("lambda_cwe")
    return (float(v), "mac dinh khoi") if v is not None else (None, "khong xac dinh")

def scan(roots):
    cells = defaultdict(dict)
    for root in roots:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if not re.fullmatch(r"fold\d+\.json", fn): continue
                path = os.path.join(dp, fn)
                try: d = json.load(open(path))
                except Exception: continue
                h = d.get("hyperparameters") or {}
                if not h: continue
                seed, fold = h.get("seed"), h.get("fold")
                if seed is None or fold is None: continue
                arm = os.path.basename(os.path.dirname(os.path.dirname(path)))
                bb = BB.get(h.get("model_name"), (h.get("model_name") or "?").split("/")[-1])
                aux = h.get("aux_mode")
                lam, lamsrc = (None, "-") if aux in (None, "none") else lam_of(d, h, arm)
                rec = dict(arm=arm, aux=aux or "baseline",
                           opt=h.get("phase2_optimizer") or "adamw",
                           rho=float(h.get("sam_rho") or 0.0),
                           var=(h.get("sam_variant") or "sam") if float(h.get("sam_rho") or 0) else "-",
                           lam=lam, lamsrc=lamsrc,
                           p1sam=bool(re.search(r"_sam1?(r\d+)?(_|$)", arm)),
                           p1val=d.get("phase1_val_macro_f1"),
                           src=next((s for s in ("4cwe","com","full") if f"_{s}_" in arm or arm.endswith("_"+s)), "-"),
                           mtime=os.path.getmtime(path))
                for m, f in MET: rec[m] = d.get(f)
                cells[(root, bb, seed, fold)]["baseline" if arm == "baseline" else arm] = rec
    return cells

def build(cells):
    g = defaultdict(lambda: {m: [] for m, _ in MET})
    meta = {}
    nskip = 0
    for (root, bb, seed, fold), arms in cells.items():
        base = arms.get("baseline")
        for arm, r in arms.items():
            if arm == "baseline" or r["aux"] == "baseline": continue
            if base is None: nskip += 1; continue
            k = (bb, r["lam"], r["rho"], r["var"], r["aux"], r["src"], r["opt"], r["p1sam"])
            for m, _ in MET:
                if r.get(m) is not None and base.get(m) is not None: g[k][m].append(r[m] - base[m])
            mt = meta.setdefault(k, dict(mtime=0, lamsrc=r["lamsrc"], p1val=r["p1val"], trees=set()))
            mt["mtime"] = max(mt["mtime"], r["mtime"]); mt["trees"].add(root)
    out = []
    for k, dd in g.items():
        bb, lam, rho, var, aux, src, opt, p1sam = k
        mt = meta[k]
        e = dict(bb=bb, lam=lam, rho=rho, var=var, aux=aux, src=src, opt=opt, p1sam=p1sam,
                 lamsrc=mt["lamsrc"], mtime=int(mt["mtime"]), ntree=len(mt["trees"]))
        for m, _ in MET:
            vs = dd[m]
            e[m] = round(sum(vs)/len(vs), 5) if vs else None
            e[m+"p"] = sum(1 for x in vs if x > 0)
            e[m+"n"] = len(vs)
        e["n"] = max(e[m+"n"] for m, _ in MET)
        out.append(e)
    out.sort(key=lambda e: (e["bb"], -1 if e["lam"] is None else e["lam"], e["rho"]))
    return out, nskip

if __name__ == "__main__":
    roots = sys.argv[1:] or sorted(
        {d for d in os.listdir(".") if d.startswith("results") and os.path.isdir(d)} |
        {os.path.join("results", d) for d in (os.listdir("results") if os.path.isdir("results") else [])
         if os.path.isdir(os.path.join("results", d))})
    cells = scan(roots)
    rows, nskip = build(cells)
    tot = sum(r["n"] for r in rows)
    sys.stderr.write(f"# {len(cells)} khoi | {len(rows)} dong | {tot} o ghep cap | {nskip} BO vi thieu baseline cung fold\n")
    json.dump({"rows": rows, "nroot": len(roots), "nskip": nskip,
               "gen": __import__("time").strftime("%Y-%m-%d %H:%M UTC", __import__("time").gmtime())},
              sys.stdout)
