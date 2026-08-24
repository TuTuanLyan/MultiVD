#!/usr/bin/env python3
"""Gộp mọi file kết quả rời rạc thành MỘT bảng tra cứu được.

Vì sao cần: 1049 file `fold*.json` nằm rải trong năm cây thư mục `results*`, và
để so hai nhánh thì phải biết trước đường dẫn của cả hai. Hệ quả đo được: §48
của RESULT.md sai 4/8 ô vì bảng được gõ tay từ các thư mục khác nhau thay vì
truy vấn từ một nguồn.

Mỗi dòng của file ra là một (run, arm, seed, fold) — đơn vị nhỏ nhất mà mọi
phép so trong dự án này thao tác trên đó. Giữ ĐỦ metric, giữ per-CWE, và giữ
đúng những hyperparameter phân biệt được hai nhánh với nhau.

Bộ fold KHÔNG được ghi trong file kết quả, nên suy từ cỡ tập test hàm ý bởi
`test_accuracy_at_0.5`: acc × n phải là số nguyên. Cùng cách audit_coverage.py
vẫn làm.
"""

import argparse
import glob
import json
import os

# Cỡ tập test từng fold của mỗi bộ. Dùng để suy bộ fold ngược từ accuracy.
FOLD_SIZES = {
    "goc":    {1: 152, 2: 152, 3: 152, 4: 152, 5: 152},
    "twin":   {1: 154, 2: 153, 3: 152, 4: 151, 5: 150},
    "random": {1: 146, 2: 153, 3: 157, 4: 153, 5: 151},
    "js":     {1: 230, 2: 229, 3: 227, 4: 226, 5: 226},
}

# Hyperparameter giữ lại: đúng những thứ có thể KHÁC nhau giữa hai nhánh và vì
# thế giải thích được chênh lệch. Phần còn lại (đường dẫn, cờ debug) bỏ đi.
KEEP_HP = [
    "model_name", "pooling", "aux_mode", "cwe_vocab", "num_latent",
    "lambda_cwe", "latent_temperature", "data_root", "target_lang",
    "phase2_optimizer", "pretrain_cof", "recadam_anchor", "source_interpolation",
    "sam_rho", "learning_rate", "batch_size", "epochs", "max_length",
    "truncation_strategy", "patience", "min_epochs", "freeze_backbone_layers",
    "lora_rank", "max_train_samples", "seed", "phase",
]

# Metric giữ nguyên tên. Mọi khoá bắt đầu bằng test_/val_/best_ đều được giữ,
# nên danh sách này chỉ để tài liệu hoá, không dùng để lọc.
SKIP_TOP = {"hyperparameters", "per_cwe", "per_cwe_at_0.5", "per_cwe_at_valcal",
            "experiment_name", "source_checkpoint", "target_checkpoint"}


def infer_fold_set(accuracy, fold):
    """Bộ fold nào cho ra một cỡ tập test khớp với accuracy này?

    Trả về tên bộ nếu CHỈ MỘT bộ khớp; None nếu không bộ nào khớp hoặc nhiều
    bộ cùng khớp (khi đó cỡ tập test không phân biệt được và đoán là sai lầm).
    """
    if accuracy is None or fold not in (1, 2, 3, 4, 5):
        return None
    hits = [name for name, sizes in FOLD_SIZES.items()
            if abs(accuracy * sizes[fold] - round(accuracy * sizes[fold])) < 1e-9]
    return hits[0] if len(hits) == 1 else None


def compact_per_cwe(block):
    """Giữ macro_f1 và cỡ mẫu cho từng lớp CWE, bỏ phần suy ra được từ chúng."""
    if not isinstance(block, dict):
        return None
    return {cwe: {"macro_f1": v.get("macro_f1"),
                  "positive_f1": v.get("positive_f1"),
                  "n": v.get("number_of_samples")}
            for cwe, v in block.items() if isinstance(v, dict)}


def run_and_arm(path):
    """Danh tính (run, nhánh) suy từ ĐƯỜNG DẪN, không từ nội dung file.

    Hai bố cục cùng tồn tại trong dự án:
        <root>/<run>/<arm>/seed_N/foldK.json   — bố cục hiện tại
        <root>/<run>/seed_N/foldK.json         — bố cục cũ, một nhánh duy nhất
    """
    parts = path.split(os.sep)
    if len(parts) >= 5 and parts[-2].startswith("seed_"):
        return parts[-4], parts[-3]
    if len(parts) >= 4 and parts[-2].startswith("seed_"):
        return parts[-3], "transfer"
    return parts[1], "transfer"


def resolve_fold_sets(records):
    """Cỡ tập test đôi khi khớp nhiều bộ fold cùng lúc (fold 3 có 152 mẫu ở cả
    bộ gốc lẫn bộ twin). Một fold đơn lẻ khi đó không phân biệt được, nhưng cả
    một run thì thường có: chỉ cần MỘT fold trong run đó khớp duy nhất.

    Nên gán theo run: nếu mọi fold phân biệt được trong cùng (run, seed) đều chỉ
    về một bộ, gán bộ đó cho cả nhóm. Nếu chúng bất đồng thì để None — bất đồng
    nghĩa là run đó trộn hai bộ fold, và đoán bừa còn tệ hơn không biết.
    """
    from collections import defaultdict
    votes = defaultdict(set)
    for row in records:
        if row["fold_set"]:
            votes[(row["run"], row["seed"])].add(row["fold_set"])
    for row in records:
        if row["fold_set"]:
            continue
        seen = votes.get((row["run"], row["seed"]), set())
        if len(seen) == 1:
            row["fold_set"] = next(iter(seen))
            row["fold_set_inferred_from_run"] = True
    return records


def build(roots):
    records, skipped = [], 0
    patterns = ["{r}/*/*/seed_*/fold*.json",   # bố cục hiện tại
                "{r}/*/seed_*/fold*.json",     # bố cục cũ, không có tầng nhánh
                "{r}/*/fold*_seed*.json",      # smoke run đời đầu
                "{r}/*/*/fold*_seed*.json"]
    for root in roots:
        seen_paths = set()
        for pattern in patterns:
            for path in sorted(glob.glob(pattern.format(r=root))):
                if path in seen_paths or os.path.basename(path).startswith("summary"):
                    continue
                seen_paths.add(path)
                try:
                    raw = json.load(open(path))
                except Exception:
                    skipped += 1
                    continue
                if raw.get("test_macro_f1_at_0.5") is None:
                    skipped += 1      # file của phase train, chưa có kết quả test
                    continue
                hp = raw.get("hyperparameters") or {}
                fold = int(raw.get("fold"))
                seed = raw.get("seed", hp.get("seed"))
                run, arm = run_and_arm(path)

                # `experiment_name` bên trong file KHÔNG đáng tin làm danh tính:
                # nhiều driver (norecadam-matrix.sh, sam-gate.sh, family-matrix.sh
                # lượt 2) sao chép nguyên file baseline sang run mới để khỏi chạy
                # lại, nên trường đó vẫn ghi run gốc. Danh tính lấy từ ĐƯỜNG DẪN;
                # trường trong file giữ lại riêng để thấy được file nào là bản sao.
                origin = raw.get("experiment_name")
                row = {
                    "root": root,
                    "run": run,
                    "arm": arm,
                    "seed": int(seed),
                    "fold": fold,
                    "fold_set": infer_fold_set(raw.get("test_accuracy_at_0.5"), fold),
                    "path": path,
                    "recorded_as": origin,
                    "is_copy": bool(origin and origin != f"{run}/{arm}"),
                }
                for key, value in raw.items():
                    if key not in SKIP_TOP and not isinstance(value, (dict, list)):
                        row[key] = value
                for key in KEEP_HP:
                    if key in hp:
                        row[f"hp_{key}"] = hp[key]
                for block_name in ("per_cwe_at_0.5", "per_cwe_at_valcal"):
                    block = compact_per_cwe(raw.get(block_name))
                    if block:
                        row[block_name] = block
                records.append(row)
    return resolve_fold_sets(records), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+",
                        default=["results", "results_vast", "results_vast2",
                                 "results_ntat", "results_ntat2"])
    parser.add_argument("--out", default="records/results_all.jsonl")
    args = parser.parse_args()

    roots = [r for r in args.roots if os.path.isdir(r)]
    records, skipped = build(roots)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    size_mb = os.path.getsize(args.out) / 1048576
    print(f"{len(records)} dòng -> {args.out}  ({size_mb:.1f} MB), bỏ qua {skipped} file")

    from collections import Counter
    print("\ntheo bộ fold:", dict(Counter(r["fold_set"] for r in records)))
    print("theo thư mục:", dict(Counter(r["root"] for r in records)))
    print("số (run, seed) khác nhau:", len({(r["run"], r["seed"]) for r in records}))


if __name__ == "__main__":
    main()
