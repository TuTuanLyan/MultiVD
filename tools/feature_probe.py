#!/usr/bin/env python3
"""feature_probe.py — CÁI GÌ chuyển giao: ĐẶC TRƯNG hay HÀM QUYẾT ĐỊNH? (0 GPU, chạy CPU)

RESEARCH_2026-09-06 §12.2 kết luận "hàm quyết định của Pha 1 không chuyển sang Python (zero-shot
0.52), chỉ có ĐẶC TRƯNG là chuyển được" — nhưng nửa sau chưa từng được đo trực tiếp. Phép đo:

  * trích đặc trưng pooled (CLS/mean, trước dropout) của (a) backbone PRETRAINED nguyên bản và
    (b) checkpoint PHA 1, trên toàn bộ 760 dòng Python — ĐÓNG BĂNG, không fine-tune;
  * mỗi fold: fit logistic regression trên train (chuẩn hoá theo train), chọn C trên val (ROC-AUC),
    chấm test: F1@0.5, ROC-AUC, PR-AUC, và per-CWE;
  * Δ = (b) − (a) ghép cặp theo fold. Dương ⇒ Pha 1 làm đặc trưng Python tách được tuyến tính hơn,
    tức đặc trưng THẬT SỰ chuyển giao. ≈0 ⇒ lợi ích của Pha 1 nằm ở quỹ đạo tối ưu (điểm khởi tạo),
    không ở đặc trưng tuyến tính.
  * kèm zero-shot của chính vul_head Pha 1 trên đặc trưng (b) — phải ≈0.5 như INT1 đã đo.

    OMP_NUM_THREADS=3 nice -n 19 python3 tools/feature_probe.py \
        --model_name microsoft/codebert-base --pooling cls \
        --ckpt model/n48/phase1/codebert__latent_bottleneck_4cwe_l0p05/seed_42/best.pt \
        --out results/probe/codebert_4cwe.json
"""
import argparse, json, os, sys, time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, "src")
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import train_transfer as T  # noqa: E402
from model import pool_hidden_states  # noqa: E402

CWE = {0: "022", 1: "078", 2: "079", 3: "089"}


def collect_rows(data_root, folds):
    """Mọi dòng duy nhất (theo code) + membership train/val/test của từng fold."""
    rows = OrderedDict()
    member = {}
    for k in folds:
        for split in ("train", "val", "test"):
            recs = T.load_jsonl(os.path.join(data_root, f"fold{k}", f"{split}.jsonl"), "python")
            for r in recs:
                key = r["code"]
                if key not in rows:
                    rows[key] = {"label": int(r["label"]), "cwe_class": int(r.get("cwe_class", -100))}
                member.setdefault(k, {}).setdefault(split, []).append(key)
    return rows, member


@torch.no_grad()
def extract(model, records, tok, max_length, batch, device):
    loader = T.build_dataloader(records, tok, max_length, batch, False, 0, 0, "head_middle_tail")
    feats, logits = [], []
    model.eval()
    t0 = time.time()
    for i, b in enumerate(loader):
        am = b["attention_mask"].to(device)
        out = model.backbone(input_ids=b["input_ids"].to(device), attention_mask=am)
        pooled = pool_hidden_states(out.last_hidden_state, am, model.pooling)
        feats.append(pooled.float().cpu().numpy())
        logits.append(model.vul_head(pooled).float().cpu().numpy())
        if i % 20 == 0:
            print(f"    batch {i}/{len(loader)}  {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(feats), np.concatenate(logits)


def metrics(y, p):
    out = {"f1": f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)}
    out["roc"] = roc_auc_score(y, p) if len(set(y.tolist())) == 2 else float("nan")
    out["pr"] = average_precision_score(y, p) if len(set(y.tolist())) == 2 else float("nan")
    return out


def probe_fold(F, idx, rows, member, k, control_seed=None):
    """`control_seed` != None => CONTROL TASK cua Hewitt & Liang (EMNLP 2019): thay nhan that
    bang nhan NGAU NHIEN co dinh theo hang (cung phan phoi). Probe nao cung fit duoc phan nao
    nhan ngau nhien; hieu (that - ngau nhien) moi la DO CHON LOC, va do moi la thu duoc phep
    goi la 'dac trung mang thong tin'. Khong co phep nay thi phan bien 'probe cua anh tu hoc
    tac vu chu khong do dac trung' khong tra loi duoc."""
    keys = list(rows)
    pos = {key: i for i, key in enumerate(keys)}
    y = np.array([rows[key]["label"] for key in keys])
    if control_seed is not None:
        y = np.random.default_rng(control_seed).permutation(y)
    c = np.array([rows[key]["cwe_class"] for key in keys])
    tr = np.array([pos[x] for x in member[k]["train"]]); va = np.array([pos[x] for x in member[k]["val"]])
    te = np.array([pos[x] for x in member[k]["test"]])
    sc = StandardScaler().fit(F[tr])
    Xtr, Xva, Xte = sc.transform(F[tr]), sc.transform(F[va]), sc.transform(F[te])
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y[tr])
        roc_va = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
        if best is None or roc_va > best[0]:
            best = (roc_va, C, clf)
    roc_va, C, clf = best
    p = clf.predict_proba(Xte)[:, 1]
    res = {"C": C, "val_roc": roc_va, "test": metrics(y[te], p), "per_cwe": {}}
    for cls in sorted(set(c[te].tolist())):
        m = c[te] == cls
        if m.sum() >= 2:
            res["per_cwe"][CWE.get(cls, str(cls))] = {**metrics(y[te][m], p[m]), "n": int(m.sum())}
    return res, (te, y[te], c[te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--pooling", default="cls")
    ap.add_argument("--ckpt", required=True, help="checkpoint Pha 1")
    ap.add_argument("--data_root", default="data/sven_python_folds_norm")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None,
                    help="file .npz de LUU/DUNG LAI dac trung da trich. Trich lai het 45 phut CPU "
                         "moi backbone, ma dac trung cua mo hinh DONG BANG thi khong bao gio doi")
    ap.add_argument("--control", action="store_true",
                    help="chay THEM control task (nhan xao tron co dinh) va in DO CHON LOC")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    rows, member = collect_rows(a.data_root, a.folds)
    records = [{"code": key, "label": v["label"], "cwe_class": v["cwe_class"], "lang": "python"} for key, v in rows.items()]
    print(f"# {len(records)} dong Python duy nhat | fold {a.folds} | {a.model_name} / {a.pooling}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_name)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck.get("training_args") or {}
    ns = argparse.Namespace(aux_mode=ck.get("aux_mode"), num_latent=ck.get("num_latent"),
                            num_cwes=ck.get("num_cwes"), pooling=ck.get("pooling") or a.pooling,
                            latent_temperature=ta.get("latent_temperature", 0.1), lora_rank=0,
                            freeze_backbone_layers=0, phase="probe")
    cached = None
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache)
        if int(z["n"]) == len(records):
            cached = z
            print(f"# DUNG LAI dac trung tu {a.cache} ({len(records)} hang) — khong trich lai")
        else:
            print(f"# bo dem {a.cache} co {int(z['n'])} hang, can {len(records)} — trich lai")
    if cached is not None:
        F_pre, F_p1, L_p1 = cached["F_pre"], cached["F_p1"], cached["L_p1"]
    else:
        print(f"# (a) PRETRAINED {a.model_name}")
        m_pre = T.make_model(a.model_name, a.device, ns)
        F_pre, _ = extract(m_pre, records, tok, a.max_length, a.batch, a.device)
        del m_pre
        print(f"# (b) PHA 1 {a.ckpt} | val {ck.get('best_val_macro_f1'):.4f} ep {ck.get('best_epoch')} | lambda {ta.get('lambda_cwe')}")
        m_p1 = T.make_model(a.model_name, a.device, ns)
        missing, unexpected = m_p1.load_state_dict(ck["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"# CANH BAO nap trong so: thieu {len(missing)}, thua {len(unexpected)}")
        F_p1, L_p1 = extract(m_p1, records, tok, a.max_length, a.batch, a.device)
        del m_p1
        if a.cache:
            os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
            np.savez_compressed(a.cache, F_pre=F_pre, F_p1=F_p1, L_p1=L_p1, n=len(records))
            print(f"# da luu dac trung -> {a.cache}")

    out = {"model_name": a.model_name, "ckpt": a.ckpt, "n_rows": len(records), "folds": {}}
    print(f"\n{'fold':<5}{'':<12}{'F1@0.5':>8}{'ROC':>8}{'PR':>8}   per-CWE ROC (022/078/079/089)")
    d_all = {"f1": [], "roc": [], "pr": []}; d_cwe = {}
    ctl = {"pre": [], "p1": []}
    for k in a.folds:
        r_pre, _ = probe_fold(F_pre, None, rows, member, k)
        r_p1, (te, yte, cte) = probe_fold(F_p1, None, rows, member, k)
        if a.control:
            c_pre, _ = probe_fold(F_pre, None, rows, member, k, control_seed=1000 + k)
            c_p1, _ = probe_fold(F_p1, None, rows, member, k, control_seed=1000 + k)
            ctl["pre"].append(c_pre["test"]); ctl["p1"].append(c_p1["test"])
        # zero-shot cua chinh vul_head Pha 1 (khong fit gi)
        pz = torch.softmax(torch.tensor(L_p1[te]), -1)[:, 1].numpy()
        r_zs = {"test": metrics(yte, pz)}
        out["folds"][k] = {"pretrained_LP": r_pre, "phase1_LP": r_p1, "phase1_zeroshot": r_zs}
        for name, r in (("pretrained LP", r_pre), ("Pha1 LP", r_p1), ("Pha1 zero-shot", r_zs)):
            t = r["test"]
            pc = "  ".join(f"{cw}:{r['per_cwe'][cw]['roc']:.3f}" for cw in ("022", "078", "079", "089") if cw in r.get("per_cwe", {}))
            print(f"{k:<5}{name:<12}{t['f1']:>8.4f}{t['roc']:>8.4f}{t['pr']:>8.4f}   {pc}")
        for m in d_all:
            d_all[m].append(r_p1["test"][m] - r_pre["test"][m])
        for cw in r_p1["per_cwe"]:
            if cw in r_pre["per_cwe"]:
                d_cwe.setdefault(cw, {"roc": [], "f1": []})
                d_cwe[cw]["roc"].append(r_p1["per_cwe"][cw]["roc"] - r_pre["per_cwe"][cw]["roc"])
                d_cwe[cw]["f1"].append(r_p1["per_cwe"][cw]["f1"] - r_pre["per_cwe"][cw]["f1"])

    def s(v):
        v = np.asarray(v, float); v = v[~np.isnan(v)]
        return f"{v.mean():+.4f} {int((v > 1e-12).sum())}/{len(v)}"
    print(f"\n=== Δ (Pha 1 LP − pretrained LP), ghep cap theo fold ===")
    print(f"F1@0.5 {s(d_all['f1'])} | ROC-AUC {s(d_all['roc'])} | PR-AUC {s(d_all['pr'])}")
    print("per-CWE: " + " | ".join(f"{cw} ROC {s(v['roc'])} F1 {s(v['f1'])}" for cw, v in sorted(d_cwe.items())))
    if a.control:
        def m(lst, key):
            v = np.asarray([x[key] for x in lst], float); v = v[~np.isnan(v)]
            return float(v.mean())
        print("\n=== CONTROL TASK (nhan xao tron) va DO CHON LOC ===")
        print(f"{'':16}{'that':>10}{'ngau nhien':>13}{'chon loc':>11}")
        for name, rr, cc in (("pretrained", [v['pretrained_LP']['test'] for v in out['folds'].values()], ctl["pre"]),
                             ("Pha 1", [v['phase1_LP']['test'] for v in out['folds'].values()], ctl["p1"])):
            for met in ("f1", "roc"):
                print(f"{name+' '+met:16}{m(rr,met):>10.4f}{m(cc,met):>13.4f}{m(rr,met)-m(cc,met):>11.4f}")
        out["control"] = {"pretrained": ctl["pre"], "phase1": ctl["p1"]}
    out["delta"] = {"all": {m: [float(x) for x in v] for m, v in d_all.items()},
                    "per_cwe": {cw: {m: [float(x) for x in vv] for m, vv in v.items()} for cw, v in d_cwe.items()}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"# ghi {a.out}")


if __name__ == "__main__":
    main()
