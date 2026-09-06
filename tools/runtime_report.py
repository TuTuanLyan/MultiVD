#!/usr/bin/env python3
"""Tong hop THOI GIAN CHAY va PHAN CUNG tu cac file ket qua, de khai bao trong bai.

    python3 tools/runtime_report.py results_night48 results results_night48_158
    python3 tools/runtime_report.py --group gpu,phase,source .

Doc truong `runtime` do src/runtime_env.py ghi vao moi ket qua. O nao chay truoc khi
co ban va do se khong co truong nay — chung duoc dem rieng va bao ro, KHONG bi bo im
lang: mot o thieu so do khac han mot o khong ton tai.
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path


def stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    n = len(xs); m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"n": n, "mean": m, "sd": sd, "min": min(xs), "max": max(xs), "total": sum(xs)}


def hhmm(sec):
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def arm_of(path):
    """Tach (nguon, rho, phuong phap) tu duong dan nhanh."""
    m = re.search(r'transfer_(\w+?)_(4cwe|com|full)_l([0-9p]+)_r([0-9p]+)', str(path))
    if m:
        return {"method": m.group(1), "source": m.group(2),
                "lambda": m.group(3).replace('p', '.'), "rho": m.group(4).replace('p', '.')}
    if "/baseline/" in str(path):
        return {"method": "baseline", "source": "-", "lambda": "-", "rho": "-"}
    return {"method": "?", "source": "?", "lambda": "?", "rho": "?"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--group", default="gpu,phase,source,rho",
                    help="truc gom nhom, ngan bang dau phay: gpu,phase,source,rho,method,lambda,host")
    a = ap.parse_args()
    keys = [k.strip() for k in a.group.split(",") if k.strip()]

    files = []
    for r in a.roots:
        files += sorted(Path(r).rglob("fold*.json"))
    if not files:
        print("khong tim thay file ket qua nao"); return 1

    # Mot lan huan luyen Pha 1 duoc DUNG CHUNG cho nhieu o (moi nhanh rho mot o), nen
    # neu dem thang tu ket qua thi mot lan chay bi tinh nhieu lan va "tong gio GPU"
    # phong len. Khu trung theo danh tinh cua chinh lan chay do.
    rows, missing, seen_train = [], [], {}
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rt = d.get("runtime")
        arm = arm_of(f)
        if not rt:
            missing.append((f, arm)); continue
        hw = rt.get("hardware_at_inference") or {}
        for phase_key, label in (("phase1", "Pha 1"), ("phase2", "Pha 2"),
                                 ("baseline_train", "baseline")):
            blk = rt.get(phase_key)
            if not blk:
                continue
            bhw = blk.get("hardware") or hw
            ident = (phase_key, arm["source"], blk.get("seed"), blk.get("fold"),
                     bhw.get("hostname"), blk.get("seconds", {}).get("total_training"))
            if phase_key in ("phase1", "baseline_train"):
                if ident in seen_train:
                    seen_train[ident] += 1      # o khac dung lai dung lan chay nay
                    continue
                seen_train[ident] = 1
            rows.append({
                "gpu": bhw.get("gpu_name") or "?", "host": bhw.get("hostname") or "?",
                "phase": label, "source": arm["source"], "rho": arm["rho"],
                "method": arm["method"], "lambda": arm["lambda"],
                "seconds": blk.get("seconds", {}).get("total_training"),
                "per_epoch": blk.get("seconds", {}).get("per_epoch", {}).get("mean"),
                "epochs": blk.get("epochs_run"),
                "n_train": (blk.get("samples") or {}).get("train"),
                "ms_per_sample": blk.get("ms_per_sample"),
                "torch": bhw.get("torch"), "transformers": bhw.get("transformers"),
                "file": str(f),
            })

    shared = sum(v - 1 for v in seen_train.values())
    print(f"Doc {len(files)} file ket qua | {len(rows)} lan chay huan luyen RIENG BIET"
          f" | {len(missing)} file KHONG co truong runtime")
    if shared:
        print(f"  ({shared} tham chieu den lan chay Pha 1/baseline dung chung — da khu trung,"
              f" khong tinh vao tong gio)")
    if missing:
        print(f"  (chay truoc khi co ban va ghi-gio; vi du: {missing[0][0]})")
    if not rows:
        return 0

    g = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k, "?") for k in keys)].append(r)

    hdr = " | ".join(f"{k:<10}" for k in keys)
    print("\n" + hdr + " |  n  | tong/lan chay        | giay/epoch     | epoch | mau  | ms/mau")
    print("-" * (len(hdr) + 74))
    for k in sorted(g):
        rs = g[k]
        s = stat([r["seconds"] for r in rs]); pe = stat([r["per_epoch"] for r in rs])
        ep = stat([r["epochs"] for r in rs]); nt = stat([r["n_train"] for r in rs])
        ms = stat([r["ms_per_sample"] for r in rs])
        print(" | ".join(f"{str(x):<10}" for x in k) +
              f" | {len(rs):>3} | {hhmm(s['mean']):>7} ± {hhmm(s['sd']):<7} "
              f"| {pe['mean']:>6.1f} ± {pe['sd']:<5.1f} | {ep['mean']:>5.1f} "
              f"| {int(nt['mean']) if nt else 0:>4} | {ms['mean'] if ms else 0:>6.1f}")

    print("\n--- Phan cung da dung (cho muc setup cua bai) ---")
    seen = {}
    for r in rows:
        seen.setdefault((r["gpu"], r["host"], r["torch"], r["transformers"]), 0)
        seen[(r["gpu"], r["host"], r["torch"], r["transformers"])] += 1
    for (gpu, host, tv, trv), n in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"  {gpu:<18} host={host:<12} torch={tv:<12} transformers={trv:<8} | {n} lan chay")

    tot = sum(r["seconds"] or 0 for r in rows)
    print(f"\nTong thoi gian GPU da do duoc: {hhmm(tot)} tren {len(rows)} lan chay huan luyen"
          f" rieng biet (da khu trung Pha 1/baseline dung chung)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
