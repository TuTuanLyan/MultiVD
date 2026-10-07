"""Dien thoi gian TUNG EPOCH cho o baseline tu log, vi train_baseline.py khong goi rl.epoch().

Khong sua file resource goc: ghi file RIENG fold<k>.resource_epochs.json canh no, ghi ro nguon.
Dung: python3 tools/backfill_epoch_times.py <thu_muc_ket_qua> <thu_muc_log>
"""
import json, re, statistics as st, sys
from pathlib import Path

PAT = re.compile(r"Epoch (\d+)/(\d+) \| .*Train: ([0-9.]+)s \| Validation: ([0-9.]+)s")

def main(res_dir, log_dir):
    res_dir, log_dir = Path(res_dir), Path(log_dir)
    for fj in sorted(res_dir.glob("fold[0-9].json")):
        k = fj.stem[4:]
        log = log_dir / f"f{k}.log"
        rows = [m.groups() for m in map(PAT.search, log.read_text().splitlines()) if m]
        if not rows:
            print(f"  !! fold{k}: khong thay dong epoch trong {log}"); continue
        eps = [{"epoch": int(e), "train_s": float(t), "val_s": float(v)} for e, _, t, v in rows]
        tr = [e["train_s"] for e in eps]; va = [e["val_s"] for e in eps]
        out = {"schema": "multivd.resource_epochs/1", "source": str(log),
               "note": "trich tu log; train_baseline.py khong goi ResourceLog.epoch()",
               "epochs_run": len(eps), "best_epoch": json.load(open(fj)).get("best_epoch"),
               "train_mean": round(st.mean(tr), 3), "train_sd": round(st.pstdev(tr), 3),
               "val_mean": round(st.mean(va), 3), "list": eps}
        (res_dir / f"fold{k}.resource_epochs.json").write_text(json.dumps(out, indent=1))
        print(f"  fold{k}: {len(eps)} epoch, train {out['train_mean']}s ± {out['train_sd']}, val {out['val_mean']}s")

if __name__ == "__main__":
    main(*sys.argv[1:3])
