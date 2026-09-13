#!/usr/bin/env python3
"""Do lop FUSION thuc su dung adapter NGUON bao nhieu.

    python tools/fusion_weights.py --checkpoint <phase2.pt> --model_name <hf> --pooling <cls|mean> \
        --data data/sven_python_folds_norm/fold1/test.jsonl

VI SAO. Khoi fus2 cho: adapter nguon NGAU NHIEN tai tao khoang mot nua loi ich cua adapter DA
HOC. Hai cach giai thich: (a) fusion gan trong so thap cho nguon nen noi dung nguon it quan
trong; (b) fusion dung nguon nhieu nhung noi dung nao cung duoc. Trong so attention phan biet
duoc hai cai do, va no chinh la hien vat dien giai duoc cua arXiv:2005.00247 — thu ma khoi
truoc da lam mat vi checkpoint Pha 2 bi xoa.

In trung binh trong so fusion cho tung adapter, theo TUNG LOP, tren tap test that.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import build_backbone, TransferModel          # noqa: E402
from adapters import (adapter_blocks, enable_fusion,      # noqa: E402
                      inject_adapters, spec_from_state_dict)
from transformers import AutoTokenizer                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--pooling", default="cls")
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--json_out", default="")
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"]
    spec = spec_from_state_dict(sd)
    if not spec["names"] or not spec["fusion"]:
        raise SystemExit(f"checkpoint khong co fusion: {spec}")
    h = ck.get("training_args", {})
    num_cwes = sd["cwe_head.weight"].shape[0] if "cwe_head.weight" in sd else 4
    model = TransferModel(build_backbone(a.model_name), num_cwes=num_cwes,
                          aux_mode="latent_bottleneck", num_latent=8, pooling=a.pooling)
    inject_adapters(model.backbone, names=tuple(spec["names"]), dim=int(spec["dim"]))
    enable_fusion(model.backbone)
    model.load_state_dict(sd)          # NGHIEM NGAT
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    tok = AutoTokenizer.from_pretrained(a.model_name)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    if a.max_samples:
        rows = rows[:a.max_samples]
    blocks = adapter_blocks(model.backbone)
    names = blocks[0].active
    # tong co trong so theo (lop, adapter) — trung binh tren MOI token khong bi pad
    tot = torch.zeros(len(blocks), len(names), dtype=torch.float64)
    ntok = 0
    with torch.no_grad():
        for i in range(0, len(rows), a.batch_size):
            b = tok([r["code"] for r in rows[i:i + a.batch_size]], return_tensors="pt",
                    truncation=True, max_length=a.max_length, padding=True)
            b = {k: v.to(dev) for k, v in b.items()}
            model(b["input_ids"], b["attention_mask"])
            m = b["attention_mask"].bool()
            for li, blk in enumerate(blocks):
                w = blk.fusion.last_weights          # (B, T, N)
                tot[li] += w[m].sum(0).double().cpu()
            ntok += int(m.sum())
    mean = (tot / max(ntok, 1)).tolist()
    print(f"checkpoint : {a.checkpoint}")
    print(f"adapter    : {names}   | {len(blocks)} lop | {len(rows)} mau, {ntok} token")
    print(f"\n{'lop':>4} | " + " | ".join(f"{n:>10}" for n in names))
    for li, row in enumerate(mean):
        print(f"{li:>4} | " + " | ".join(f"{x:10.4f}" for x in row))
    avg = [sum(r[j] for r in mean) / len(mean) for j in range(len(names))]
    print(f"{'TB':>4} | " + " | ".join(f"{x:10.4f}" for x in avg))
    print("\n0.5/0.5 = fusion khong phan biet hai adapter. Lech manh ve mot ben = no CO chon.")
    if a.json_out:
        json.dump({"names": names, "per_layer": mean, "mean": avg,
                   "checkpoint": a.checkpoint, "n_tokens": ntok}, open(a.json_out, "w"))
        print(f"da ghi {a.json_out}")


if __name__ == "__main__":
    main()
