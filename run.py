import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT, "experiments.json"), encoding="utf-8") as handle:
    EXPERIMENTS = json.load(handle)
MODELS = {"codebert": "codebert-base", "codet5": "codet5-base"}


def paths(name, seed, fold):
    tag = os.path.join(name, "seed_%d" % seed)
    return {
        "checkpoint": os.path.join("checkpoints", tag, "fold%d" % fold, "best.pt"),
        "result": os.path.join("results", tag, "fold%d.json" % fold),
        "log": os.path.join("logs", tag, "fold%d.log" % fold),
    }


def phase1_fold(exp, fold):
    return fold if exp.get("phase1_fold") == "same" else 1


def argv(flags):
    out = []
    for key, value in flags.items():
        out += [key, value]
    return out


def commands(name, seed, fold, model_dir="models"):
    exp = EXPERIMENTS[name]
    p = paths(name, seed, fold)
    flags = {key: str(value).format(seed=seed) for key, value in exp["flags"].items()}
    flags["--seed"] = str(seed)
    head = [sys.executable, "launch.py", exp["code"], exp["trainer"]]
    common = ["--run_name", name, "--fold", str(fold), "--model_name", os.path.join(model_dir, MODELS[exp["backbone"]]),
              "--checkpoint_path", p["checkpoint"]]
    train_flags = dict(flags)
    init = None
    if exp.get("phase1"):
        init = paths(exp["phase1"], seed, phase1_fold(exp, fold))["checkpoint"]
        train_flags["--init_ckpt"] = init
    test_flags = {key: value for key, value in flags.items() if key != "--init"}
    train = head + ["--phase", "train"] + common + argv(train_flags)
    test = head + ["--phase", "test"] + common + argv(test_flags) + ["--result_path", p["result"]]
    return train, test, p, init


def run_cell(name, seed, fold, args, force=False):
    exp = EXPERIMENTS[name]
    train, test, p, init = commands(name, seed, fold, args.model_dir)
    done = os.path.exists(os.path.join(ROOT, p["result"]))
    if exp["stage"] == "phase1":
        done = done and os.path.exists(os.path.join(ROOT, p["checkpoint"]))
    if done and not (force or args.force):
        print("skip %s seed %d fold %d (%s exists)" % (name, seed, fold, p["result"]), flush=True)
        return
    if init and not os.path.exists(os.path.join(ROOT, init)):
        run_cell(exp["phase1"], seed, phase1_fold(exp, fold), args, force=True)
    env = dict(os.environ, PYTHONHASHSEED="42", TOKENIZERS_PARALLELISM="false", **exp.get("env", {}))
    if args.dry_run:
        print("# %s seed %d fold %d" % (name, seed, fold))
        print(" ".join(train))
        print(" ".join(test), flush=True)
        return
    for path in (p["result"], p["result"][:-len(".json")] + ".probs.npz", p["checkpoint"]):
        if os.path.exists(os.path.join(ROOT, path)):
            os.remove(os.path.join(ROOT, path))
    for key in ("log", "result", "checkpoint"):
        os.makedirs(os.path.join(ROOT, os.path.dirname(p[key])), exist_ok=True)
    print("run %s seed %d fold %d -> %s" % (name, seed, fold, p["log"]), flush=True)
    with open(os.path.join(ROOT, p["log"]), "w", encoding="utf-8") as log:
        rc = subprocess.call(train, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if rc == 0:
            rc = subprocess.call(test, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        sys.exit("%s seed %d fold %d failed (exit %d), see %s" % (name, seed, fold, rc, p["log"]))
    if exp["stage"] == "target" and not args.keep_checkpoints:
        os.remove(os.path.join(ROOT, p["checkpoint"]))
    with open(os.path.join(ROOT, p["result"]), encoding="utf-8") as handle:
        result = json.load(handle)
    print("done %s seed %d fold %d test ROC-AUC %.4f" % (name, seed, fold, result["test_roc_auc"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep_checkpoints", action="store_true")
    args = parser.parse_args()
    if args.list or not args.experiments:
        for name, exp in EXPERIMENTS.items():
            print("%-44s %s" % (name, exp["paper"]))
        return
    names = [n for n in EXPERIMENTS if EXPERIMENTS[n]["stage"] == "target"] if args.experiments == ["all"] else args.experiments
    for name in names:
        if name not in EXPERIMENTS:
            sys.exit("unknown experiment %s (python run.py --list)" % name)
    for fold in sorted({f for n in names for f in EXPERIMENTS[n]["folds"]}):
        for seed in (42, 1234, 7):
            for name in names:
                exp = EXPERIMENTS[name]
                if fold not in exp["folds"] or seed not in exp["seeds"]:
                    continue
                if (args.folds and fold not in args.folds) or (args.seeds and seed not in args.seeds):
                    continue
                run_cell(name, seed, fold, args)


if __name__ == "__main__":
    main()
