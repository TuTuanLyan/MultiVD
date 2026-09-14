#!/usr/bin/env python3
"""Do co che TRUOC khi viet code huan luyen: huong-va co ton tai va co xuyen ngon ngu khong?

Gia thuyet can kiem (De xuat A/C trong NEXT_CONTRIBUTION.md):
    voi moi cap (ham truoc va, ham sau va) cung mot ham, `d = h(vul) - h(fixed)` la mot
    "huong lo hong"; voi CUNG MOT CWE huong do phai GIONG NHAU giua cac ngon ngu.

Neu sai thi ca huong nay chet o day, khong ton mot giay GPU huan luyen nao.

BA CAI BAY da tinh truoc:

1. **Thanh phan chung.** Moi `d` co the cung chia mot huong "vulnerable-ness" khong phu
   thuoc CWE, lam MOI cosine deu cao va bang nhau. Nen bao cao ca ban DA TRU trung binh
   toan cuc — phan con lai moi la phan RIENG cua CWE.
2. **Null that su.** So "cung CWE" voi "khac CWE" chua du: so cap khong deu giua cac CWE
   va giua hai ngon ngu. Nen chay hoan vi nhan CWE (giu nguyen moi thu khac) de lay phan
   bo null, va bao cao z cung voi hieu tho.
3. **Bac thang duoi.** Neu backbone GOC (chua Pha 1) da co san tinh chat nay thi Pha 1
   khong phai thu tao ra no — phai do ca hai.

Phep thu MANH NHAT o cuoi: lay `mu_c` uoc tu `com` (ccpp+js) roi cham voi dac trung cua
tap dich PYTHON. AUC > 0.5 co nghia huong-va tu ngon ngu khac phan loai duoc python
*khong huan luyen mot buoc nao*.
"""
import argparse, json, os, re, sys, collections
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.dataset import CodeDataset  # noqa: E402


def norm_cwe(x):
    m = re.search(r"(\d+)", str(x) if x is not None else "")
    return f"CWE-{int(m.group(1))}" if m else None


def load_backbone(model_name, ckpt, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name)
    tag = "pretrained"
    if ckpt:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = blob.get("model_state_dict", blob)
        sub = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        if not sub:
            raise SystemExit(f"khong co khoa `backbone.` nao trong {ckpt}")
        missing, unexpected = mod.load_state_dict(sub, strict=False)
        # strict=False o day la CO Y (checkpoint khong co pooler), nhung phai IN RA
        # cai gi thieu/thua — `strict=False` nuot ca khoa THIEU la loi da mac roi.
        print(f"  nap {ckpt}: thieu={len(missing)} thua={len(unexpected)} "
              f"| vd thieu={list(missing)[:3]} thua={list(unexpected)[:3]}")
        tag = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
    return tok, mod.to(device).eval(), tag


@torch.no_grad()
def embed(rows, tok, mod, device, pooling, max_length, batch_size):
    ds = CodeDataset(rows, tok, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for i, b in enumerate(dl):
        ids = b["input_ids"].to(device)
        am = b["attention_mask"].to(device)
        h = mod(input_ids=ids, attention_mask=am).last_hidden_state
        if pooling == "cls":
            p = h[:, 0, :]
        else:
            m = am.unsqueeze(-1).to(h.dtype)
            p = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
        out.append(p.float().cpu().numpy())
        if i % 20 == 0:
            print(f"    ...{(i+1)*batch_size}/{len(rows)}", flush=True)
    return np.concatenate(out, 0)


def unit(x, eps=1e-9):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cos_stats(D, cwes, langs, rng, n_perm=200):
    """Tra ve (cung CWE khac ngon ngu, cung CWE cung ngon ngu, khac CWE) + null hoan vi."""
    U = unit(D)
    n = len(U)
    iu = np.triu_indices(n, k=1)
    C = (U @ U.T)[iu]
    same_cwe = (np.array(cwes)[:, None] == np.array(cwes)[None, :])[iu]
    same_lang = (np.array(langs)[:, None] == np.array(langs)[None, :])[iu]

    def m(mask):
        return float(C[mask].mean()) if mask.sum() else float("nan"), int(mask.sum())

    res = {
        "cung_cwe_khac_ngonngu": m(same_cwe & ~same_lang),
        "cung_cwe_cung_ngonngu": m(same_cwe & same_lang),
        "khac_cwe_khac_ngonngu": m(~same_cwe & ~same_lang),
        "khac_cwe_cung_ngonngu": m(~same_cwe & same_lang),
        "tat_ca": m(np.ones_like(same_cwe, dtype=bool)),
    }
    # Hieu can quan tam: cung CWE khac ngon ngu  -  khac CWE khac ngon ngu
    obs = res["cung_cwe_khac_ngonngu"][0] - res["khac_cwe_khac_ngonngu"][0]
    null = []
    cw = np.array(cwes)
    for _ in range(n_perm):
        p = rng.permutation(cw)
        sc = (p[:, None] == p[None, :])[iu]
        a = C[sc & ~same_lang]
        b = C[~sc & ~same_lang]
        if len(a) and len(b):
            null.append(a.mean() - b.mean())
    null = np.array(null) if null else np.array([0.0])
    z = (obs - null.mean()) / (null.std() + 1e-12)
    p_emp = float((np.abs(null) >= abs(obs)).mean())
    res["_hieu_quan_tam"] = float(obs)
    res["_null_tb"] = float(null.mean())
    res["_null_sd"] = float(null.std())
    res["_z"] = float(z)
    res["_p_hoanvi"] = p_emp
    return res


def show(title, r):
    print(f"\n--- {title} ---")
    for k in ["cung_cwe_khac_ngonngu", "cung_cwe_cung_ngonngu",
              "khac_cwe_khac_ngonngu", "khac_cwe_cung_ngonngu", "tat_ca"]:
        v, n = r[k]
        print(f"   {k:24} cos = {v:+.4f}   (n cap = {n})")
    print(f"   HIEU (cung-khac, deu khac ngon ngu) = {r['_hieu_quan_tam']:+.4f}"
          f" | null {r['_null_tb']:+.4f}±{r['_null_sd']:.4f}"
          f" | z = {r['_z']:+.2f} | p_hoanvi = {r['_p_hoanvi']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="microsoft/codebert-base")
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    ap.add_argument("--checkpoint", default=None, help="bo trong = backbone GOC")
    ap.add_argument("--phase1", default="data/phase1_common.jsonl")
    ap.add_argument("--target", default="data/sven_python_folds_norm/data.jsonl")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help=">0 => torch.set_num_threads")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    print(f"== nap mo hinh | {args.model_name} | pooling={args.pooling} | device={args.device}")
    tok, mod, tag = load_backbone(args.model_name, args.checkpoint, args.device)

    src = [json.loads(l) for l in open(args.phase1)]
    # --- dung cap tu pair_id ---
    by = collections.defaultdict(list)
    for i, r in enumerate(src):
        if r.get("pair_id"):
            by[r["pair_id"]].append(i)
    pairs = []
    for pid, idxs in by.items():
        v = [i for i in idxs if src[i]["label"] == 1]
        f = [i for i in idxs if src[i]["label"] == 0]
        if len(v) == 1 and len(f) == 1:
            pairs.append((v[0], f[0]))
    print(f"== cap day du trong nguon: {len(pairs)} / {len(by)} pair_id")

    need = sorted({i for p in pairs for i in p})
    pos = {g: k for k, g in enumerate(need)}
    print(f"== nhung {len(need)} dong nguon")
    Hs = embed([src[i] for i in need], tok, mod, args.device, args.pooling,
               args.max_length, args.batch_size)

    D, cwes, langs = [], [], []
    for v, f in pairs:
        c = norm_cwe(src[v].get("cwe_id"))
        if c is None:
            continue
        D.append(Hs[pos[v]] - Hs[pos[f]])
        cwes.append(c)
        langs.append(src[v].get("lang"))
    D = np.stack(D)
    print(f"== {len(D)} vector huong-va | ngon ngu {collections.Counter(langs)}")
    print(f"   CWE hay gap: {collections.Counter(cwes).most_common(6)}")

    show("THO (chua tru trung binh toan cuc)", cos_stats(D, cwes, langs, rng))
    Dc = D - D.mean(0, keepdims=True)
    show("DA TRU trung binh toan cuc (phan RIENG cua CWE)", cos_stats(Dc, cwes, langs, rng))

    # do lon cua thanh phan chung
    mu = D.mean(0)
    frac = float(np.linalg.norm(mu) / (np.linalg.norm(D, axis=1).mean() + 1e-12))
    print(f"\n   |trung binh toan cuc| / |d| trung binh = {frac:.3f}"
          f"   (cao => moi cap dich cung mot huong, phan rieng CWE it)")

    # ---- PHEP THU CHINH: mu_c tu `com` co phan loai duoc PYTHON khong (0 buoc huan luyen) ----
    tgt = [json.loads(l) for l in open(args.target)]
    print(f"\n== nhung {len(tgt)} dong dich (python)")
    Ht = embed(tgt, tok, mod, args.device, args.pooling, args.max_length, args.batch_size)
    y = np.array([r["label"] for r in tgt])
    tc = np.array([norm_cwe(r.get("cwe_id")) for r in tgt])

    mus = {}
    for c in set(cwes):
        m = np.array([x == c for x in cwes])
        if m.sum() >= 2:
            mus[c] = D[m].mean(0)
    Hc = Ht - Ht.mean(0, keepdims=True)

    from sklearn.metrics import roc_auc_score
    print("\n--- mu_c uoc tu `com` (ccpp+js), cham voi dac trung PYTHON ---")
    print("    diem = <h - h_tb, mu_c>; AUC > 0.5 = huong-va tu ngon ngu KHAC phan loai duoc")
    rows = []
    for c in sorted(set(tc.tolist())):
        sel = tc == c
        if c not in mus or sel.sum() < 10 or len(set(y[sel])) < 2:
            print(f"   {c:8} bo qua (n={int(sel.sum())}, co mu={c in mus})")
            continue
        s = Hc[sel] @ unit(mus[c])
        auc = roc_auc_score(y[sel], s)
        # doi chung 1: mu cua CWE KHAC
        others = [k for k in mus if k != c]
        aucs_o = [roc_auc_score(y[sel], Hc[sel] @ unit(mus[k])) for k in others]
        # doi chung 2: huong ngau nhien cung chuan
        aucs_r = []
        for _ in range(50):
            g = rng.normal(size=mus[c].shape)
            aucs_r.append(roc_auc_score(y[sel], Hc[sel] @ unit(g)))
        print(f"   {c:8} n={int(sel.sum()):4} | AUC(mu dung CWE) = {auc:.4f}"
              f" | AUC(mu CWE khac) tb = {np.mean(aucs_o):.4f}"
              f" | AUC(huong ngau nhien) tb = {np.mean(aucs_r):.4f} ± {np.std(aucs_r):.4f}")
        rows.append({"cwe": c, "n": int(sel.sum()), "auc": auc,
                     "auc_cwe_khac": float(np.mean(aucs_o)),
                     "auc_ngau_nhien": float(np.mean(aucs_r)),
                     "sd_ngau_nhien": float(np.std(aucs_r))})
    # gop chung mot huong duy nhat (khong theo CWE) de tach "CWE co can thiet khong"
    mu_all = unit(D.mean(0))
    try:
        auc_all = roc_auc_score(y, Hc @ mu_all)
        print(f"\n   MOT huong duy nhat (trung binh moi CWE), toan bo python: AUC = {auc_all:.4f}")
    except Exception:
        auc_all = None

    if args.out:
        json.dump({"tag": tag, "model": args.model_name, "pooling": args.pooling,
                   "n_pair": len(D), "frac_chung": frac,
                   "per_cwe": rows, "auc_mot_huong": auc_all},
                  open(args.out, "w"), indent=1)
        print(f"\nda ghi {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
