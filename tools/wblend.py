#!/usr/bin/env python3
"""wblend.py — NOI SUY TRONG SO giua baseline (chi-dich) va mo hinh chuyen giao.

VI SAO: §25 do duoc phep TRON XAC SUAT baseline+chuyen giao vuot ca hai dau mut
(ROC +0.0165 so voi chuyen giao thuan, 59/83 khoi, p=0.0002). Nhung no ton HAI LAN
suy luan. Neu tron duoc trong KHONG GIAN TRONG SO thi chi phi ve lai MOT mo hinh —
do la khac biet giua "mot meo ensemble" va "mot phuong thuc trien khai duoc".

CO CHE DA KIEM TRUOC, 0 GPU (src/model.py): BaselineModel va TransferModel chia dung
`backbone.*` + `vul_head.{weight,bias}` — CUNG TEN, CUNG SHAPE. Cac tensor rieng cua
TransferModel (`latent_proj.*`, `cwe_head.*`) khong tham gia duong tinh `vul_logits`
nen giu nguyen, khong anh huong du doan. Vay phep noi suy XAC DINH duoc cho MOI tensor
co tac dong len ket qua.

KHAC `sweep_source_interpolation` da co trong train_transfer.py: cai do noi suy cap
(theta_Pha1, theta_Pha2). Day la cap (theta_baseline, theta_Pha2) — hai mo hinh CUNG
TAC VU DICH, va do moi la cap tra loi duoc "Pha 1 dong gop gi".

alpha chon tren VAL, bao tren TEST. In ca duong test day du de chan doan nhung KHONG
duoc dung no de chon. Bao kem phep tron XAC SUAT tren cung cap de so sanh truc tiep.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import torch
from transformers import AutoTokenizer

# `build_dataloader`/`load_jsonl` KHONG nam trong mot module data.py — chung o ngay
# trong train_transfer.py (dong 377 / 138). Ban dau toi import tu `data` va ast.parse
# van bao OK: cu phap dung khong chung minh import dung. Da kiem tung ky hieu bang grep.
from evaluate import classification_metrics, evaluate     # noqa: E402
from model import TransferModel, build_backbone           # noqa: E402
from train_transfer import build_dataloader, load_jsonl   # noqa: E402


def state_of(path, device):
    pay = torch.load(path, map_location=device, weights_only=False)
    st = pay.get("model_state_dict") or pay.get("model_state") or pay
    return {k: v.detach().cpu() for k, v in st.items()}, pay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--transfer_ckpt", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--data_root", default="data/sven_python_folds_norm")
    ap.add_argument("--model_name", default="Salesforce/codet5p-220m")
    ap.add_argument("--pooling", default="enc")
    ap.add_argument("--aux_mode", default="latent_bottleneck")
    ap.add_argument("--num_latent", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grid", default="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
    # --max_eval CHI de KIEM DUONG ONG cho nhanh (CPU chay het 152 hang mat >15 phut).
    # Chay that thi KHONG dat co nay — lay het hang.
    ap.add_argument("--max_eval", type=int, default=0, help="0 = lay het (chay that)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sb, _ = state_of(a.baseline_ckpt, dev)
    st, payt = state_of(a.transfer_ckpt, dev)
    ncwe = payt.get("training_args", {}).get("num_cwes") or \
        next((v.shape[0] for k, v in st.items() if k == "cwe_head.weight"), 4)

    tok = AutoTokenizer.from_pretrained(a.model_name)
    root = Path(a.data_root) / f"fold{a.fold}"
    loaders = {}
    for split in ("val", "test"):
        recs = load_jsonl(root / f"{split}.jsonl", "python")
        if a.max_eval:
            recs = recs[:a.max_eval]
            print(f"  !! CHE DO KIEM DUONG ONG: {split} cat con {len(recs)} hang — so KHONG dung de doc")
        # num_workers=0 CO CHU DICH: moi worker fork them ~2.3GB, ma tap eval chi 152
        # hang nen worker khong nhanh hon. 161 la may dung chung, da tung bao thieu RAM.
        loaders[split] = build_dataloader(recs, tok, a.max_length, a.batch_size,
                                          False, 42, 0, "head")

    backbone = build_backbone(a.model_name)
    model = TransferModel(backbone, num_cwes=ncwe, aux_mode=a.aux_mode,
                          num_latent=a.num_latent, pooling=a.pooling).to(dev)
    model.load_state_dict(st, strict=False)

    shared = [k for k, v in model.state_dict().items()
              if k in sb and k in st and sb[k].shape == v.shape == st[k].shape
              and v.is_floating_point()]
    only_t = [k for k in model.state_dict() if k not in sb]
    print(f"# tensor noi suy duoc: {len(shared)} | chi co o chuyen giao (giu nguyen): "
          f"{len(only_t)} -> {only_t[:6]}")
    if not shared:
        print("!! KHONG co tensor chung — phep noi suy vo nghia, dung"); return 2

    grid = [float(x) for x in a.grid.split()]
    rows, probs = [], {}
    for al in grid:
        with torch.no_grad():
            sd = model.state_dict()
            for k in shared:
                sd[k].copy_((1 - al) * sb[k].to(dev) + al * st[k].to(dev))
        r = {"alpha": al}
        for split in ("val", "test"):
            out = evaluate(model, loaders[split], dev)
            m = classification_metrics(out["labels"], out["probabilities"], 0.5)
            r[f"{split}_macro_f1"] = m["macro_f1"]
            r[f"{split}_roc_auc"] = m["roc_auc"]
            r[f"{split}_pr_auc"] = m["pr_auc"]
            if split == "test":
                probs[al] = [round(float(x), 6) for x in out["probabilities"]]
                r["test_labels_n"] = len(out["labels"])
                lab = out["labels"]
        rows.append(r)
        print(f"  alpha={al:.2f} | val F1 {r['val_macro_f1']:.4f} ROC {r['val_roc_auc']:.4f}"
              f" | test F1 {r['test_macro_f1']:.4f} ROC {r['test_roc_auc']:.4f}")

    # alpha chon tren VAL — hai tieu chi, bao ca hai vi chung co the khac nhau
    best_f1 = max(rows, key=lambda r: r["val_macro_f1"])
    best_roc = max(rows, key=lambda r: r["val_roc_auc"])
    # doi chung: TRON XAC SUAT cung cap, alpha=0.5 khai bao truoc
    pmix = 0.5 * np.asarray(probs[0.0]) + 0.5 * np.asarray(probs[1.0])
    mmix = classification_metrics(lab, pmix.tolist(), 0.5)
    res = {"grid": rows, "shared_tensors": len(shared), "transfer_only_tensors": only_t,
           "alpha_on_val_f1": best_f1["alpha"], "alpha_on_val_roc": best_roc["alpha"],
           "test_at_alpha_val_f1": {k: best_f1[k] for k in best_f1 if k.startswith("test")},
           "test_at_alpha_val_roc": {k: best_roc[k] for k in best_roc if k.startswith("test")},
           "prob_blend_0.5": {"macro_f1": mmix["macro_f1"], "roc_auc": mmix["roc_auc"],
                              "pr_auc": mmix["pr_auc"]},
           "baseline_ckpt": a.baseline_ckpt, "transfer_ckpt": a.transfer_ckpt, "fold": a.fold}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    b = next(r for r in rows if r["alpha"] == 0.0); t = next(r for r in rows if r["alpha"] == 1.0)
    print(f"\n  baseline (a=0)      test ROC {b['test_roc_auc']:.4f} F1 {b['test_macro_f1']:.4f}")
    print(f"  chuyen giao (a=1)   test ROC {t['test_roc_auc']:.4f} F1 {t['test_macro_f1']:.4f}")
    print(f"  TRON TRONG SO a={best_roc['alpha']:.2f} (chon tren val ROC)"
          f"  test ROC {best_roc['test_roc_auc']:.4f} F1 {best_roc['test_macro_f1']:.4f}")
    print(f"  TRON XAC SUAT a=0.50 (khai bao truoc)   test ROC {mmix['roc_auc']:.4f}"
          f" F1 {mmix['macro_f1']:.4f}")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
