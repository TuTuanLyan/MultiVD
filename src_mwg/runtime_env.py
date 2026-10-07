"""Ghi nhan PHAN CUNG va THOI GIAN CHAY de khai bao trong bai.

Vi sao co file nay: cac con so "huan luyen mat bao lau, tren phan cung gi, voi bao
nhieu mau" la phan bat buoc cua muc setup trong bai bao, nhung truoc day chung chi
nam trong log console roi troi mat cung may thue. File nay dua chung vao mot ban ghi
co cau truc, luu canh checkpoint, roi nhap vao ket qua JSON.

Hai quyet dinh dang chu y:

1. Ghi ra FILE PHU `<checkpoint>.runtime.json` chu khong nhet vao checkpoint. Checkpoint
   nang ~418 MB; them truong vao no nghia la doc-sua-ghi lai ca 418 MB sau moi lan chay.
   Du an nay da mot lan mat 46 o vi dia day lam `torch.save` ghi cut checkpoint — cang it
   lan ghi de len file nang cang tot.

2. Ghi KIEU NGUYEN TU: ra file tam roi `os.replace`. Neu tien trinh chet giua chung thi
   file cu con nguyen, khong bao gio co file JSON cut doi ma van doc duoc mot nua.
"""

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _ram_total_gb():
    try:
        for line in Path("/proc/meminfo").read_text(errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1048576, 1)
    except Exception:
        pass
    return None


def _nvidia_driver():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


def hardware_fingerprint(device=None):
    """Moi thu can de khai bao "training setup" trong bai, doc mot lan moi lan chay."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": _ram_total_gb(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cudnn"] = getattr(torch.backends.cudnn, "version", lambda: None)()
        if torch.cuda.is_available():
            idx = device.index if (device is not None and getattr(device, "index", None) is not None) else 0
            p = torch.cuda.get_device_properties(idx)
            info["gpu_name"] = p.name
            info["gpu_total_mem_mb"] = round(p.total_memory / 1048576)
            info["gpu_capability"] = f"{p.major}.{p.minor}"
            info["gpu_count"] = torch.cuda.device_count()
            info["nvidia_driver"] = _nvidia_driver()
        else:
            info["gpu_name"] = None
    except Exception as exc:  # khong bao gio duoc lam hong lan chay chi vi ghi nhat ky
        info["torch_error"] = str(exc)
    for mod in ("transformers", "sklearn", "numpy", "scipy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def _stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mean, 3),
        "sd": round(var ** 0.5, 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "total": round(sum(values), 3),
    }


def build_runtime_record(*, phase, args, device, epochs_planned, epochs_run, best_epoch,
                         epoch_seconds, train_seconds, val_seconds, total_seconds,
                         n_train, n_val, extra=None):
    """Mot ban ghi cho MOT lan chay (mot phase, mot fold, mot seed, mot setting)."""
    steps_per_epoch = None
    if n_train and getattr(args, "batch_size", None):
        steps_per_epoch = -(-int(n_train) // int(args.batch_size))
    rec = {
        "phase": phase,
        "run_name": getattr(args, "run_name", None),
        "method_name": getattr(args, "method_name", None),
        "fold": getattr(args, "fold", None),
        "seed": getattr(args, "seed", None),
        "epochs_planned": epochs_planned,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "stopped_early": bool(epochs_run < epochs_planned),
        "samples": {"train": n_train, "val": n_val},
        "batch_size": getattr(args, "batch_size", None),
        "steps_per_epoch": steps_per_epoch,
        "max_length": getattr(args, "max_length", None),
        "seconds": {
            "total_training": round(float(total_seconds), 3),
            "per_epoch": _stats(epoch_seconds),
            "per_epoch_train_only": _stats(train_seconds),
            "per_epoch_validation": _stats(val_seconds),
        },
        # SAM/ASAM chay hai luot forward-backward moi buoc nen thoi gian ~2x; ghi ro de
        # nguoi doc bang khong so nham mot o co SAM voi mot o khong.
        "sam_rho": getattr(args, "sam_rho", 0.0),
        "sam_variant": getattr(args, "sam_variant", None) if getattr(args, "sam_rho", 0.0) else None,
        "hardware": hardware_fingerprint(device),
    }
    if n_train and epoch_seconds:
        per_epoch_mean = rec["seconds"]["per_epoch"].get("mean")
        if per_epoch_mean:
            rec["samples_per_second"] = round(float(n_train) / per_epoch_mean, 2)
            rec["ms_per_sample"] = round(1000.0 * per_epoch_mean / float(n_train), 3)
    if extra:
        rec.update(extra)
    return rec


def sidecar_path(checkpoint_path):
    return Path(str(checkpoint_path) + ".runtime.json")


def write_runtime_sidecar(checkpoint_path, record):
    """Ghi nguyen tu: file tam roi doi ten. Khong bao gio de lai JSON cut doi."""
    path = sidecar_path(checkpoint_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return path
    except Exception:
        return None      # ghi nhat ky hong thi thoi, khong duoc keo do ca lan chay


def read_runtime_sidecar(checkpoint_path):
    try:
        p = sidecar_path(checkpoint_path)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None
