#!/usr/bin/env python3
"""Vá ghi-nhat-ky-phan-cung vao src/train.py, src/train_transfer.py, src/train_baseline.py.

    python3 scripts/apply_runtime_logging.py --check   # chi thu, khong ghi
    python3 scripts/apply_runtime_logging.py --root .  # ap that

CHI CHAY KHI KHONG CON JOB NAO DANG CHAY. matrix.sh phong mot python moi cho moi o;
sua file giua chung thi o ke tiep co the vo phai file ghi do.
"""
import argparse, sys
from pathlib import Path

EDITS = []

# ---------- 1. train.py: gom gio moi epoch, ghi file phu khi xong ----------
EDITS.append(("src/train.py",
"""    best_score, best_epoch, patience_counter = -math.inf, 0, 0
    training_started = time.perf_counter()
    logger.info("Training started | Phase: %s | Epochs: %d", phase, args.epochs)""",
"""    best_score, best_epoch, patience_counter = -math.inf, 0, 0
    training_started = time.perf_counter()
    # Gom gio tung epoch de cuoi lan chay ghi ra <checkpoint>.runtime.json. Truoc day
    # nhung con so nay chi ra console roi troi mat cung may thue, trong khi muc setup
    # cua bai lai can chinh chung.
    _rt_epoch, _rt_train, _rt_val, _rt_epochs_run = [], [], [], 0
    logger.info("Training started | Phase: %s | Epochs: %d", phase, args.epochs)"""))

EDITS.append(("src/train.py",
"""        epoch_seconds = time.perf_counter() - epoch_started
        total_seconds = time.perf_counter() - training_started""",
"""        epoch_seconds = time.perf_counter() - epoch_started
        total_seconds = time.perf_counter() - training_started
        _rt_epoch.append(epoch_seconds)
        _rt_train.append(train_seconds)
        _rt_val.append(validation_seconds)
        _rt_epochs_run = epoch"""))

EDITS.append(("src/train.py",
"""    logger.info(
        "Training finished | Elapsed: %.2fs",
        time.perf_counter() - training_started,
    )""",
"""    _rt_total = time.perf_counter() - training_started
    logger.info("Training finished | Elapsed: %.2fs", _rt_total)
    try:
        from runtime_env import build_runtime_record, write_runtime_sidecar
        _rt = build_runtime_record(
            phase=phase, args=args, device=device,
            epochs_planned=args.epochs, epochs_run=_rt_epochs_run, best_epoch=best_epoch,
            epoch_seconds=_rt_epoch, train_seconds=_rt_train, val_seconds=_rt_val,
            total_seconds=_rt_total,
            n_train=len(getattr(train_loader, "dataset", []) or []),
            n_val=len(getattr(val_loader, "dataset", []) or []),
        )
        _p = write_runtime_sidecar(args.checkpoint_path, _rt)
        if _p:
            logger.info(
                "Runtime ghi lai | %s | %d epoch | %.1fs/epoch | %s",
                _p, _rt_epochs_run, _rt["seconds"]["per_epoch"].get("mean", 0.0),
                _rt["hardware"].get("gpu_name"),
            )
    except Exception as exc:
        # Ghi nhat ky hong khong duoc lam hong lan chay: mot o co so ma thieu gio van
        # dung hon mot o trong.
        logger.warning("Khong ghi duoc runtime sidecar: %s", exc)"""))

# ---------- 2. train_transfer.py: nhap gio vao ket qua JSON ----------
EDITS.append(("src/train_transfer.py",
"""        "per_cwe_at_valcal": per_cwe_at_valcal,
        "hyperparameters": checkpoint["training_args"],
    }""",
"""        "per_cwe_at_valcal": per_cwe_at_valcal,
        "hyperparameters": checkpoint["training_args"],
        "runtime": _collect_runtime(args, device, len(test["labels"]),
                                    validation_seconds, test_seconds),
    }"""))

EDITS.append(("src/train_transfer.py",
"""def run_test(args, device):""",
'''def _collect_runtime(args, device, n_test, validation_seconds, test_seconds):
    """Gom gio cua CA HAI pha vao ket qua, vi mot o Pha 2 khong the tach khoi Pha 1
    da sinh ra checkpoint cho no. Doc tu file phu canh checkpoint; thieu thi de None
    chu khong dung lan chay."""
    from runtime_env import hardware_fingerprint, read_runtime_sidecar
    out = {
        "phase2": read_runtime_sidecar(args.checkpoint_path),
        "phase1": read_runtime_sidecar(args.source_checkpoint) if args.source_checkpoint else None,
        "inference": {
            "samples_test": n_test,
            "validation_and_threshold_seconds": round(float(validation_seconds), 3),
            "test_seconds": round(float(test_seconds), 3),
            "ms_per_test_sample": round(1000.0 * float(test_seconds) / n_test, 3) if n_test else None,
        },
        "hardware_at_inference": hardware_fingerprint(device),
    }
    return out


def run_test(args, device):'''))

# ---------- 3. train_baseline.py: y het, baseline cung la mot o trong bang ----------
EDITS.append(("src/train_baseline.py",
"""        "phase": "test",""",
"""        "phase": "test",
        "runtime": _collect_runtime_baseline(args, device),"""))

EDITS.append(("src/train_baseline.py",
"""def run_test(args, device):""",
'''def _collect_runtime_baseline(args, device):
    from runtime_env import hardware_fingerprint, read_runtime_sidecar
    return {
        "baseline_train": read_runtime_sidecar(args.checkpoint_path),
        "hardware_at_inference": hardware_fingerprint(device),
    }


def run_test(args, device):'''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true", help="chi kiem tra, khong ghi")
    a = ap.parse_args()
    root = Path(a.root)
    texts, ok = {}, True
    for rel, old, new in EDITS:
        p = root / rel
        if rel not in texts:
            texts[rel] = p.read_text(encoding="utf-8")
        s = texts[rel]
        if new.strip().splitlines()[0] in s and old not in s:
            print(f"  [da co] {rel}: doan nay da duoc va truoc do")
            continue
        n = s.count(old)
        if n != 1:
            print(f"  !! {rel}: tim thay {n} lan doan can thay (phai dung 1) — DUNG")
            ok = False
            continue
        texts[rel] = s.replace(old, new, 1)
        print(f"  [ok] {rel}: thay 1 doan")
    if not ok:
        print("KHONG AP — co doan khong khop, xem lai code nguon.")
        return 2
    if a.check:
        print("CHECK: moi doan deu khop dung mot lan.")
        return 0
    for rel, s in texts.items():
        (root / rel).write_text(s, encoding="utf-8")
        print(f"  da ghi {rel}")
    print("XONG. Nho: src/runtime_env.py phai co mat cung thu muc src/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
