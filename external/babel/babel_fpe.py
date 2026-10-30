import argparse
import ast
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
ROOT = HERE.parent.parent
sys.path.append(str(ROOT / "baseline"))
from evaluate import classification_metrics, find_best_threshold, per_cwe_metrics
sys.path.remove(str(ROOT / "baseline"))
assert Path(sys.modules["evaluate"].__file__).resolve().parent == ROOT / "baseline", "evaluate is not baseline/evaluate.py"

logger = logging.getLogger("babel_fpe")
CKPT_NAME = os.path.join("checkpoint-best-acc", "model.bin")
BABEL_FLAGS = (("max_sentnum", int), ("max_wordnum", int), ("patience_step", int), ("epoch", int), ("train_batch_size", int),
               ("eval_batch_size", int), ("learning_rate", float), ("max_grad_norm", float))
STATE = {"capture": None, "train_loss": [], "history": [], "best_acc": 0.0, "best_epoch": None, "t": None, "args": None,
         "eval_labels": None, "thr": None, "test": None}


def class_to_cwe():
    tree = ast.parse(open(ROOT / "baseline" / "train_transfer.py", encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CWE_MAPPING" for t in node.targets):
            return {v: k for k, v in ast.literal_eval(node.value).items()}
    sys.exit("babel_fpe: CWE_MAPPING not found in baseline/train_transfer.py")


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fold_records(args):
    lim = (lambda rows: rows[: args.limit_examples]) if args.limit_examples > 0 else (lambda rows: rows)
    js = {s: lim(load_jsonl(os.path.join(args.source_root, f"{s}.jsonl"))) for s in ("train", "val")}
    if args.mode == "source":
        if args.data_root:
            sys.exit("babel_fpe: --mode source (Phase 1, JavaScript only) does not take --data_root")
        return {"train": js["train"], "eval": js["val"], "test": js["val"], "python_target": False}
    if not args.data_root:
        sys.exit("babel_fpe: --mode %s needs --data_root (SVEN folds)" % args.mode)
    fold_dir = os.path.join(args.data_root, f"fold{args.fold}")
    py = {s: lim(load_jsonl(os.path.join(fold_dir, f"{s}.jsonl"))) for s in ("train", "val", "test")}
    train = js["train"] + py["train"] if args.mode == "mix" else py["train"]
    return {"train": train, "eval": py["val"], "test": py["test"], "python_target": True}


def write_babel_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as handle:
        for i, r in enumerate(records):
            handle.write(json.dumps({"func": str(r["code"]), "target": int(r["label"]), "idx": i}, ensure_ascii=False) + "\n")


def prepare_data(args, work):
    recs = fold_records(args)
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "eval", "test"):
        files[split] = data / f"{split}.jsonl"
        write_babel_jsonl(recs[split], files[split])
    logger.info("BABEL data | mode %s | fold %d | train %d (positive %d) | eval %d | test %d | %s", args.mode, args.fold, len(recs["train"]),
                sum(int(r["label"]) for r in recs["train"]), len(recs["eval"]), len(recs["test"]),
                "SVEN fold k" if recs["python_target"] else "JS pool (test = copy of val)")
    return recs, files


def scores(out):
    prob = out[1] if isinstance(out, tuple) else out
    return prob.detach()[:, 0].float().cpu()


def install_shims(args, parsed_init):
    import transformers
    from adamw_transformer import AdamW
    transformers.AdamW = AdamW
    sys.path.insert(0, str(UPSTREAM))
    import run
    for mod in ("run", "model.hierarchical_att_model", "model.word_att_model", "model.sent_han_attn_confusion", "evaluator.clean_gadget",
                "layer.GNNs", "layer.Attention", "layer.soft_attn"):
        where = Path(sys.modules[mod].__file__).resolve()
        assert UPSTREAM in where.parents, "%s loaded from %s, not from upstream/" % (mod, where)
    orig_model, orig_evaluate, orig_test = run.HierAttNet, run.evaluate, run.test

    def build_model(model_args, embed_table=None):
        model = orig_model(model_args, embed_table=embed_table)
        STATE["babel_args"] = {k: v for k, v in vars(model_args).items() if k != "device"}

        def hook(module, inputs, output):
            if module.training:
                STATE["train_loss"].append(float(output[0].detach()))
            elif STATE["capture"] is not None:
                STATE["capture"].append(scores(output))
        model.register_forward_hook(hook)
        if parsed_init:
            state = torch.load(parsed_init, map_location="cpu")
            model.load_state_dict(state, strict=True)
            logger.info("BABEL Phase 2 | loaded Phase 1 checkpoint %s (%d tensors, strict=True)", parsed_init, len(state))
        return model

    def evaluate(eval_args, eval_dataset, model, eval_when_training=False):
        STATE["capture"] = []
        result = orig_evaluate(eval_args, eval_dataset, model, eval_when_training=eval_when_training)
        probs = torch.cat(STATE["capture"]).numpy() if STATE["capture"] else np.zeros(0)
        STATE["capture"] = None
        labels = np.asarray([int(e.label) for e in eval_dataset.examples])
        assert len(probs) == len(labels), "hook captured %d probabilities, eval has %d functions" % (len(probs), len(labels))
        assert abs(float(np.mean(labels == (probs > 0.5))) - result["eval_acc"]) <= 5.1e-5, "recomputed accuracy differs from the original eval_acc"
        log_epoch(eval_args, result, probs, labels)
        return result

    def test(test_args, model, tokenizer):
        out = {}
        saved = test_args.test_data_file
        for name, path in (("thr", str(STATE["files"]["eval"])), ("test", saved)):
            test_args.test_data_file = path
            STATE["capture"] = []
            t0 = time.perf_counter()
            metrics = orig_test(test_args, model, tokenizer)
            probs = torch.cat(STATE["capture"]).numpy()
            STATE["capture"] = None
            out[name] = {"probs": probs, "metrics": metrics, "seconds": time.perf_counter() - t0}
        test_args.test_data_file = saved
        STATE["thr"], STATE["test"] = out["thr"], out["test"]
        return out["test"]["metrics"]

    run.HierAttNet, run.evaluate, run.test = build_model, evaluate, test
    return run


def log_epoch(eval_args, result, probs, labels):
    from sklearn.metrics import roc_auc_score, f1_score
    epoch = len(STATE["history"]) + 1
    if result["eval_acc"] > STATE["best_acc"]:
        STATE["best_acc"], STATE["best_epoch"] = result["eval_acc"], epoch
    losses, STATE["train_loss"] = STATE["train_loss"], []
    roc = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan")
    f1 = float(f1_score(labels, probs > 0.5, average="macro"))
    now = time.perf_counter()
    seconds, STATE["t"] = now - (STATE["t"] or now), now
    row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "val_loss": result["eval_loss"],
           "val_acc": result["eval_acc"], "val_roc_auc": roc, "val_macro_f1": f1, "seconds": round(seconds, 1)}
    STATE["history"].append(row)
    best = STATE["best_epoch"] or 0
    logger.info("Epoch %d/%d | train loss %.4f | val loss %.4f | val roc_auc %.4f | val macro_f1 %.4f | best ep %d | patience %d/%d | %ds",
                epoch, eval_args.epoch, row["train_loss"] if row["train_loss"] is not None else float("nan"), row["val_loss"], roc, f1,
                best, epoch - best, eval_args.patience_step, int(seconds))
    logger.info("BABEL val | acc %.4f (checkpoint selection by acc, original code) | best acc %.4f at epoch %d", result["eval_acc"], STATE["best_acc"], best)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def babel_argv(args, files, out_dir, phase):
    argv = ["run.py", "--output_dir", str(out_dir), "--cache_dir", args.model_name, "--seed", str(args.seed),
            "--train_data_file", str(files["train"]), "--eval_data_file", str(files["eval"]), "--test_data_file", str(files["test"])]
    argv += ["--do_train", "--do_eval", "--evaluate_during_training"] if phase == "train" else ["--do_test"]
    for name, _ in BABEL_FLAGS:
        argv += ["--" + name, str(getattr(args, name))]
    return argv


def run_train(args):
    ckpt = Path(args.checkpoint_path)
    work = ckpt.parent / "babel_work"
    meta_path = Path(str(ckpt) + ".babel.json")
    for stale in (work / CKPT_NAME, meta_path):
        if stale.exists():
            logger.warning("removing a leftover file from a previous run: %s", stale)
            stale.unlink()
    recs, files = prepare_data(args, work)
    STATE["files"] = files
    if args.mode == "target" and not args.init_ckpt:
        sys.exit("babel_fpe: --mode target needs --init_ckpt (Phase 1 checkpoint)")
    if args.mode != "target" and args.init_ckpt:
        sys.exit("babel_fpe: --init_ckpt is only for --mode target")
    if args.init_ckpt and not os.path.exists(args.init_ckpt):
        sys.exit("babel_fpe: Phase 1 checkpoint not found: %s" % args.init_ckpt)
    run = install_shims(args, args.init_ckpt)
    sys.argv = babel_argv(args, files, work, "train")
    logger.info("BABEL train | original run.main() with: %s", " ".join(sys.argv[1:]))
    STATE["t"] = time.perf_counter()
    t0 = time.perf_counter()
    run.main()
    seconds = time.perf_counter() - t0
    src = work / CKPT_NAME
    if not src.exists():
        sys.exit("babel_fpe: the original train() did not write %s (no epoch had eval_acc > 0)" % src)
    os.replace(src, ckpt)
    meta = {"mode": args.mode, "best_epoch": STATE["best_epoch"], "best_val_acc": STATE["best_acc"], "history": STATE["history"],
            "train_seconds": seconds, "init_ckpt": args.init_ckpt,
            "n_train": len(recs["train"]), "n_eval": len(recs["eval"]), "babel_argv": sys.argv[1:], "wrapper_args": vars(args),
            "babel_effective_args": STATE.get("babel_args"),
            "upstream_commit": "ac532522b3a0e43b148be5dcd29f7c72fac190f3",
            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None}
    with open(str(meta_path) + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(jsonable(meta), handle, indent=2, ensure_ascii=False)
    os.replace(str(meta_path) + ".tmp", meta_path)
    logger.info("Training done | best epoch %s | %ds | peak VRAM %s MiB | checkpoint %s", STATE["best_epoch"], int(seconds),
                "%.0f" % meta["peak_vram_allocated_mib"] if meta["peak_vram_allocated_mib"] is not None else "-", ckpt)


def run_test(args):
    if not args.result_path:
        sys.exit("babel_fpe: --phase test needs --result_path")
    ckpt = Path(args.checkpoint_path)
    meta_path = Path(str(ckpt) + ".babel.json")
    if not ckpt.exists() or not meta_path.exists():
        sys.exit("babel_fpe: missing %s or %s" % (ckpt, meta_path))
    meta = json.load(open(meta_path, encoding="utf-8"))
    work = ckpt.parent / "babel_work"
    recs, files = prepare_data(args, work)
    STATE["files"] = files
    link = work / CKPT_NAME
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    os.link(ckpt, link)
    run = install_shims(args, None)
    sys.argv = babel_argv(args, files, work, "test")
    logger.info("BABEL test | original run.main() with: %s", " ".join(sys.argv[1:]))
    try:
        run.main()
    finally:
        link.unlink()
    write_result(args, meta, ckpt, recs)


def write_result(args, meta, ckpt, recs):
    thr_p, test_p = STATE["thr"]["probs"].tolist(), STATE["test"]["probs"].tolist()
    thr_y, test_y = [int(r["label"]) for r in recs["eval"]], [int(r["label"]) for r in recs["test"]]
    assert len(thr_p) == len(thr_y) and len(test_p) == len(test_y), "the number of probabilities differs from the number of functions"
    for name, p, y in (("thr", thr_p, thr_y), ("test", test_p, test_y)):
        assert abs(float(np.mean(np.asarray(y) == (np.asarray(p) > 0.5))) - STATE[name]["metrics"]["test_acc"]) <= 5.1e-5, name
    threshold, calibrated_f1 = find_best_threshold(thr_y, thr_p)
    cwe = [r.get("cwe_class") for r in recs["test"]]
    at05, atval = classification_metrics(test_y, test_p, 0.5), classification_metrics(test_y, test_p, threshold)
    c2c = class_to_cwe()
    logger.info("Test | ROC %.4f | PR %.4f | macro-F1@0.5 %.4f | macro-F1@val %.4f (threshold %.2f)", at05["roc_auc"], at05["pr_auc"],
                at05["macro_f1"], atval["macro_f1"], threshold)
    pt = recs["python_target"]
    result = {
        "experiment_name": f"{args.run_name}/babel_{args.mode}", "phase": "test", "fold": args.fold, "seed": args.seed,
        "source_checkpoint": meta.get("init_ckpt"), "target_checkpoint": str(ckpt),
        "best_epoch": meta["best_epoch"], "best_val_acc_babel": meta["best_val_acc"], "babel_selection": "eval_acc > best_acc (original code)",
        "val_calibrated_threshold": threshold, "val_macro_f1_at_valcal": calibrated_f1,
        "validation_and_threshold_seconds": STATE["thr"]["seconds"], "test_inference_seconds": STATE["test"]["seconds"],
        "train_seconds": meta["train_seconds"], "peak_vram_allocated_mib": meta.get("peak_vram_allocated_mib"),
        "test_macro_f1_at_0.5": at05["macro_f1"], "test_macro_f1_at_valcal": atval["macro_f1"],
        "test_positive_f1_at_0.5": at05["positive_f1"], "test_positive_f1_at_valcal": atval["positive_f1"],
        "test_precision_at_0.5": at05["precision"], "test_precision_at_valcal": atval["precision"],
        "test_recall_at_0.5": at05["recall"], "test_recall_at_valcal": atval["recall"],
        "test_accuracy_at_0.5": at05["accuracy"], "test_accuracy_at_valcal": atval["accuracy"],
        "test_roc_auc": at05["roc_auc"], "test_pr_auc": at05["pr_auc"],
        "per_cwe": per_cwe_metrics(test_y, test_p, cwe, threshold, c2c) if pt else None,
        "per_cwe_at_0.5": per_cwe_metrics(test_y, test_p, cwe, 0.5, c2c) if pt else None,
        "per_cwe_at_valcal": per_cwe_metrics(test_y, test_p, cwe, threshold, c2c) if pt else None,
        "babel_mode": args.mode, "babel_history": meta["history"], "babel_original_test_metrics": jsonable(STATE["test"]["metrics"]),
        "babel_original_thr_metrics": jsonable(STATE["thr"]["metrics"]), "babel_upstream_commit": meta["upstream_commit"],
        "hyperparameters": jsonable(dict(meta["wrapper_args"], babel_argv=meta["babel_argv"])),
        "babel_effective_args": jsonable(meta.get("babel_effective_args")),
    }
    out = args.result_path
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(os.path.splitext(out)[0] + ".probs.npz", probabilities=np.asarray(test_p, dtype=np.float32),
                        labels=np.asarray(test_y, dtype=np.int8), cwe_classes=np.asarray(cwe),
                        val_probabilities=np.asarray(thr_p, dtype=np.float32), val_labels=np.asarray(thr_y, dtype=np.int8),
                        threshold=np.float32(threshold))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(jsonable(result), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    logger.info("BABEL result saved: %s", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("train", "test"), required=True)
    p.add_argument("--mode", choices=("mix", "source", "target"), required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--model_name", required=True, help="CodeBERT directory = original --cache_dir (tokenizer + embedding table)")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--result_path")
    p.add_argument("--init_ckpt", help="--mode target: Phase 1 checkpoint (best.pt of the --mode source run)")
    p.add_argument("--data_root", help="fold SVEN Python (data_root/fold<k>/{train,val,test}.jsonl)")
    p.add_argument("--source_root", required=True, help="pool JS ({train,val}.jsonl)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--limit_examples", type=int, default=0, help="smoke tests only; 0 = off")
    for name, kind in BABEL_FLAGS:
        p.add_argument("--" + name, type=kind, required=True, help="original run.py flag (README)")
    args = p.parse_args()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logger.info("BABEL | upstream @ac53252 | phase %s | mode %s | original flags: %s", args.phase, args.mode,
                " ".join("--%s %s" % (n, getattr(args, n)) for n, _ in BABEL_FLAGS))
    (run_train if args.phase == "train" else run_test)(args)


if __name__ == "__main__":
    main()
