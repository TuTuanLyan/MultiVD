#!/usr/bin/env python3
"""Do MACs cho MOT lan forward cua MWGraphModel — cot `MACs` cua DeepDFA Bang 5,
va FPO ma Green AI doi vi no DOC LAP PHAN CUNG (so duy nhat so duoc giua 161/158/vast).

Chay tren CPU, KHONG dung GPU — MACs la thuoc tinh TINH cua (kien truc, hinh dang dau
vao), khong doi theo fold/seed/may, nen do MOT lan la du cho moi o.

    CUDA_VISIBLE_DEVICES="" python3 tools/measure_macs.py --fusion cat
"""
import argparse, json, os, sys, types
sys.path.insert(0, "src_mwg")
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="/drive1/cuongtm/models/codebert-base")
    ap.add_argument("--data_root", default="data/sven_python_folds_norm")
    ap.add_argument("--fusion", default="cat")
    ap.add_argument("--max_windows", type=int, default=8)
    ap.add_argument("--out", default="clean_local/results/_macs.json")
    a = ap.parse_args()

    import train_mwg as T
    from transformers import AutoTokenizer
    from model import build_backbone            # dung DUNG ham ma trainer dung
    tok = AutoTokenizer.from_pretrained(a.model_name)
    backbone = build_backbone(a.model_name)

    rows = [json.loads(l) for l in open(os.path.join(a.data_root, "fold1", "test.jsonl"), encoding="utf-8")][:1]
    ds = T.GraphWindowDataset(rows, tok, window=510, stride=384, max_windows=a.max_windows,
                              max_lines=150, typed=True, drop_bracket=True, co_mode="chain")
    from torch.utils.data import DataLoader   # trainer khong co collate rieng -> mac dinh
    batch = next(iter(DataLoader(ds, batch_size=1)))

    model = T.MWGraphModel(backbone, a.max_windows, "mean", 2, pooling="cls", num_rel=4,
                           graph_layers=2, fusion=a.fusion, graph_lstm=True,
                           max_lines=150, line_enc="codebert").eval()

    from thop import profile
    with torch.no_grad():
        macs, params = profile(model, inputs=(batch,), verbose=False)

    res = {"fusion": a.fusion, "max_windows": a.max_windows,
           "macs_per_forward": int(macs), "macs_per_forward_G": round(macs / 1e9, 4),
           "params_thop": int(params),
           "params_total": sum(p.numel() for p in model.parameters()),
           "params_by_component": {n: sum(p.numel() for p in getattr(model, n).parameters())
                                   for n in ("backbone", "graph", "vul_head") if hasattr(model, n)},
           "input_shape": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
           "ghi_chu": ("thop dem MAC cua cac op GEMM/conv da dang ky; softmax/layernorm/GELU "
                       "KHONG duoc dem — day la CAN DUOI, phai ghi ro nhan nay khi dua vao bai.")}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("fusion", "macs_per_forward_G", "params_total", "params_by_component")},
                     ensure_ascii=False, indent=2))
    print("-> %s" % a.out)

if __name__ == "__main__":
    main()
