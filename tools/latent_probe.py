#!/usr/bin/env python3
"""latent_probe.py — NÚT THẮT 8 CHIỀU có mang được gì cho nhiệm vụ ĐÍCH không? (0 GPU)

Khai báo trước: `records/prediction_2026-09-11_nut_that_8_chieu.md`.

Cổng mảnh 1 (`tools/aux_head_probe.py`) hỏi "`cwe_head` đoán CWE có giỏi không" và đã TRƯỢT.
Nhưng mảnh 2 KHÔNG dùng `cwe_head`: nó đóng băng `latent_proj` (768→8) rồi đặt một head nhị phân
MỚI lên đầu ra 8 chiều, huấn luyện trên nhãn lỗ hổng của ĐÍCH. Câu hỏi quyết định vì thế là:

    ảnh 8 chiều P(f) có còn tách được LỖ HỔNG tuyến tính không,
    và có hơn một phép chiếu 8 chiều NGẪU NHIÊN của cùng đặc trưng đó không?

Vế sau là vế phải có. `latent_proj` không hơn ma trận ngẫu nhiên ⇒ "neo vào bảng phân loại của
nguồn" là câu chuyện rỗng, cái chạy được chỉ là "giảm chiều xuống 8". Đây là dạng đối chứng
Hewitt & Liang (EMNLP 2019) mà §36.1 đã dùng cho probe 768 chiều.

Bốn bộ đặc trưng, CÙNG một checkpoint, ghép cặp theo fold:
    p768  pooled 768 chiều của Pha 1           — trần trên (§36)
    lat8  latent_proj(pooled), 8 chiều, ĐÃ HỌC — cái mảnh 2 dùng thật
    rnd8  chiếu ngẫu nhiên 768→8 của CÙNG pooled — ĐỐI CHỨNG
    pca8  8 thành phần chính đầu, fit CHỈ TRÊN TRAIN — đối chứng thứ hai, mạnh hơn

    OMP_NUM_THREADS=3 nice -n 19 python3 tools/latent_probe.py \\
        --model_name microsoft/codebert-base --pooling cls \\
        --ckpt model/auxb/phase1/codebert__latent_bottleneck_4cwe_l0p05_bal/seed_42/best.pt \\
        --out results/probe/lat8_codebert_bal.json
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, "src")
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import train_transfer as T  # noqa: E402
from model import pool_hidden_states  # noqa: E402

CWE = {0: "022", 1: "078", 2: "079", 3: "089"}


def collect_rows(data_root, folds, lang):
    rows = OrderedDict()
    member = {}
    for k in folds:
        for split in ("train", "val", "test"):
            recs = T.load_jsonl(os.path.join(data_root, f"fold{k}", f"{split}.jsonl"), lang)
            for r in recs:
                key = r["code"]
                if key not in rows:
                    rows[key] = {"label": int(r["label"]),
                                 "cwe_class": int(r.get("cwe_class", -100))}
                member.setdefault(k, {}).setdefault(split, []).append(key)
    return rows, member


@torch.no_grad()
def extract(model, records, tok, max_length, batch, device):
    """Trả (pooled 768 chiều, latent 8 chiều). Một lượt duy nhất qua backbone cho cả hai."""
    loader = T.build_dataloader(records, tok, max_length, batch, False, 0, 0, "head_middle_tail")
    feats, lats = [], []
    model.eval()
    t0 = time.time()
    for i, b in enumerate(loader):
        am = b["attention_mask"].to(device)
        out = model.backbone(input_ids=b["input_ids"].to(device), attention_mask=am)
        pooled = pool_hidden_states(out.last_hidden_state, am, model.pooling)
        feats.append(pooled.float().cpu().numpy())
        # KHONG co dropout: model.eval() da tat, va day dung duong ma Pha 2 se di
        # (dropout chi bat luc huan luyen).
        lats.append(model.latent_proj(pooled).float().cpu().numpy())
        if i % 20 == 0:
            print(f"    batch {i}/{len(loader)}  {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(feats), np.concatenate(lats)


def metrics(y, p):
    out = {"f1": f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)}
    out["roc"] = roc_auc_score(y, p) if len(set(y.tolist())) == 2 else float("nan")
    out["pr"] = average_precision_score(y, p) if len(set(y.tolist())) == 2 else float("nan")
    return out


def probe(X, y, c, tr, va, te):
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
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
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--pooling", default="cls")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="data/sven_python_folds_norm")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42, help="seed cho chieu ngau nhien rnd8")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    rows, member = collect_rows(a.data_root, a.folds, a.lang)
    keys = list(rows)
    print(f"# {len(keys)} dong duy nhat | fold {a.folds} | {a.data_root}", flush=True)

    F = L = None
    if a.cache and os.path.exists(a.cache):
        # allow_pickle: khoa la chuoi ma nguon nen numpy luu dang object. File nay do chinh
        # script sinh ra trong thu muc ket qua cua du an, khong phai dau vao tu ngoai.
        z = np.load(a.cache, allow_pickle=True)
        if len(z["keys"]) == len(keys) and list(z["keys"]) == keys:
            F, L = z["F"], z["L"]
            print(f"# dung lai bo dem {a.cache}", flush=True)
        else:
            print("# bo dem LECH khoa -> tinh lai", flush=True)
    if F is None:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
        args_ck = ck.get("training_args", {})
        ns = argparse.Namespace(model_name=a.model_name, pooling=a.pooling,
                                aux_mode=ck.get("aux_mode", "latent_bottleneck"),
                                num_cwes=int(args_ck.get("num_cwes", 4)),
                                num_latent=int(args_ck.get("num_latent", 8)),
                                dropout_rate=float(args_ck.get("dropout_rate", 0.1)),
                                latent_temperature=float(args_ck.get("latent_temperature", 0.1)),
                                freeze_backbone_layers=0, freeze_prototypes_steps=0)
        model = T.make_model(a.model_name, torch.device(a.device), ns)
        model.load_state_dict(ck["model_state_dict"])
        model.to(a.device)
        if not hasattr(model, "latent_proj"):
            raise SystemExit("checkpoint khong co latent_proj — can aux_mode=latent_bottleneck")
        tok = T.AutoTokenizer.from_pretrained(a.model_name, trust_remote_code=True)
        recs = [{"code": k, "label": rows[k]["label"], "cwe_class": rows[k]["cwe_class"],
                 "cwe": "", "cwe_id": "", "lang": a.lang} for k in keys]
        F, L = extract(model, recs, tok, a.max_length, a.batch, torch.device(a.device))
        if a.cache:
            os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
            np.savez_compressed(a.cache, keys=np.array(keys, dtype=object), F=F, L=L)

    print(f"# pooled {F.shape} | latent {L.shape}", flush=True)
    pos = {k: i for i, k in enumerate(keys)}
    y = np.array([rows[k]["label"] for k in keys])
    c = np.array([rows[k]["cwe_class"] for k in keys])

    # DOI CHUNG rnd8: ma tran Gauss co dinh, CUNG cho moi fold. Chuan hoa cot de thang do
    # khong thanh mot bien lech — StandardScaler ben trong `probe` cung lo phan nay.
    rng = np.random.default_rng(a.seed)
    R = rng.normal(size=(F.shape[1], L.shape[1])) / np.sqrt(F.shape[1])
    RND = F @ R

    out = {"ckpt": a.ckpt, "model_name": a.model_name, "folds": a.folds, "per_fold": {}}
    for k in a.folds:
        tr = np.array([pos[x] for x in member[k]["train"]])
        va = np.array([pos[x] for x in member[k]["val"]])
        te = np.array([pos[x] for x in member[k]["test"]])
        # pca8 fit CHI TREN TRAIN cua chinh fold do — fit tren toan bo la ro ri.
        pca = PCA(n_components=L.shape[1], random_state=a.seed).fit(F[tr])
        PCA8 = pca.transform(F)
        out["per_fold"][str(k)] = {
            "p768": probe(F, y, c, tr, va, te),
            "lat8": probe(L, y, c, tr, va, te),
            "rnd8": probe(RND, y, c, tr, va, te),
            "pca8": probe(PCA8, y, c, tr, va, te),
        }
        r = out["per_fold"][str(k)]
        print(f"  fold {k}: p768 ROC {r['p768']['test']['roc']:.4f} | "
              f"lat8 {r['lat8']['test']['roc']:.4f} | rnd8 {r['rnd8']['test']['roc']:.4f} | "
              f"pca8 {r['pca8']['test']['roc']:.4f}", flush=True)

    # --- tong hop: GHEP CAP theo fold, in ca ba chi so kem so fold cung dau ---
    print("\n=== trung binh qua fold ===")
    print(f"{'bo dac trung':<8}{'F1@0.5':>10}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for name in ("p768", "lat8", "rnd8", "pca8"):
        v = [out["per_fold"][str(k)][name]["test"] for k in a.folds]
        print(f"{name:<8}{np.mean([x['f1'] for x in v]):>10.4f}"
              f"{np.mean([x['roc'] for x in v]):>10.4f}{np.mean([x['pr'] for x in v]):>10.4f}")

    print("\n=== Δ GHEP CAP theo fold (khai bao truoc: lat8 phai hon CA rnd8 lan pca8) ===")
    print(f"{'phep so':<16}{'ΔF1':>10}{'+/n':>7}{'ΔROC':>10}{'+/n':>7}{'ΔPR':>10}{'+/n':>7}")
    summary = {}
    for a_, b_ in (("lat8", "rnd8"), ("lat8", "pca8"), ("lat8", "p768"), ("rnd8", "p768")):
        row = f"{a_ + ' - ' + b_:<16}"
        summary[f"{a_}-{b_}"] = {}
        for key in ("f1", "roc", "pr"):
            d = np.array([out["per_fold"][str(k)][a_]["test"][key]
                          - out["per_fold"][str(k)][b_]["test"][key] for k in a.folds])
            nz = d[np.abs(d) >= 1e-12]
            row += f"{d.mean():>+10.4f}{f'{int((nz > 0).sum())}/{len(d)}':>7}"
            summary[f"{a_}-{b_}"][key] = {"mean": float(d.mean()),
                                          "pos": int((nz > 0).sum()), "n": int(len(d))}
        print(row)
    out["summary"] = summary

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as h:
        json.dump(out, h, indent=2, sort_keys=True)
    print(f"\n# da ghi {a.out}")
    print("Nguong khai bao truoc: lat8 - rnd8 >= +0.02 ROC VA >= 4/5 fold, "
          "lat8 - pca8 cung vay. lat8 ROC < 0.55 => bac thang.")


if __name__ == "__main__":
    main()
