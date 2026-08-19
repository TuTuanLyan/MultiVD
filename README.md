# MultiVD

MultiVD triển khai và so sánh hai phương pháp phát hiện lỗ hổng Python trên cùng năm folds:

- **Transfer**: multitask pretraining trên source JSONL, sau đó transfer sang Python bằng RecAdam.
- **Baseline**: fine-tune CodeBERT trực tiếp trên Python bằng AdamW và một binary classification head.

Chi tiết học thuật và công thức nằm trong [METHOD.md](METHOD.md).

## 0. Trạng thái phiên bản

Branch `v1-explicit-cwe4` đóng băng phiên bản dùng **CWE head tường minh 4 lớp**
(`CWE-022/078/079/089`), khớp đúng taxonomy của target Python. Công việc tổng quát hóa
sang auxiliary task dạng latent diễn ra trên branch khác; branch này giữ nguyên để
đối chiếu.

`src/RecAdam.py` là bản tham chiếu từ [Sanyuan-Chen/RecAdam](https://github.com/Sanyuan-Chen/RecAdam).
Phần toán học giữ nguyên. Khác biệt duy nhất so với upstream là port API in-place của
PyTorch (`addcdiv_(value, t1, t2)` dạng positional đã bị gỡ từ PyTorch 1.5), một dòng
`last_anneal_lambda` để log, và `raise ValueError` thay cho `ValueError` trần. Không sửa
file này khi mở rộng phương pháp.

### Kết quả đã chạy trên phiên bản này

Source `data/train_ccpp_js.jsonl`, target 5 fold Python, delta = transfer − baseline:

| Seed | Δ Macro-F1@0.5 | Δ Macro-F1@valcal | Thư mục |
| --- | --- | --- | --- |
| 7 | +0.0360 ± 0.0350 | +0.0482 ± 0.0590 | `results/seed7_ccppjs_py_compare_v1/` |
| 12 | +0.0596 ± 0.0425 | +0.0643 ± 0.0548 | `results/seed12_ccppjs_py_compare_v1/` |
| 36 | +0.0518 ± 0.0160 | +0.0591 ± 0.0077 | `results/seed18_ccppjs_py_compare_v1/` |
| 42 | +0.0530 ± 0.0248 | +0.0461 ± 0.0116 | `results/seed42_ccppjs_py_compare_v1/` |

### Hạn chế đã biết

Ba điểm cần xử lý trước khi mở rộng quy mô thí nghiệm:

1. **Rò rỉ cặp trong Python folds.** Năm fold được dựng sẵn bên ngoài repo và không
   group-aware. Đo trực tiếp: 61–67 trên 152 mẫu test mỗi fold có near-duplicate
   (SequenceMatcher ratio > 0.90) nằm trong train, và trên 90% số đó là partner
   đối nghịch label — tức bản vá/bản lỗi của chính nó. Điều này làm phồng chỉ số
   tuyệt đối của cả hai nhánh. Phép so sánh transfer-vs-baseline vẫn công bằng vì
   dùng chung folds, nhưng con số tuyệt đối không phản ánh khả năng khái quát hóa.
   Phase 1 ngược lại có bảo vệ group đầy đủ (`source_groups()`).
2. **Thiếu ablation tách nguồn lợi ích.** Khoảng cách transfer-vs-baseline hiện gộp
   ba yếu tố: được thấy dữ liệu source, có CWE auxiliary task, và dùng RecAdam thay
   AdamW. Cần thêm hai nhánh để tách: source binary-only → RecAdam, và
   source multitask → AdamW.
3. **Tên thư mục không khớp seed.** `results/seed18_ccppjs_py_compare_v1/` thực chất
   chứa `seed_36/`. `results/` đã được thêm vào `.gitignore`, nhưng 35 file kết quả cũ
   vẫn đang được track từ trước (bao gồm `seed18_ccpp_py_v1` thiếu fold5 và summary).

## 1. Cấu hình một run

Chỉnh [run/config.sh](run/config.sh). Hai biến quan trọng nhất:

```bash
RUN_NAME=ccpp_only_experiment
PHASE1_DATA_PATH=data/train_ccpp_filtered.jsonl
```

`RUN_NAME` là namespace cha của toàn bộ log, model và result. Hãy đổi nó cho mỗi thí nghiệm muốn giữ lại.

Ví dụ thử source khác mà không sửa Python:

```bash
# C/C++ only
RUN_NAME=source_ccpp \
PHASE1_DATA_PATH=data/train_ccpp_filtered.jsonl \
bash run/fullpipeline.sh 36

# JavaScript only
RUN_NAME=source_js \
PHASE1_DATA_PATH=data/train_js_filtered.jsonl \
bash run/fullpipeline.sh 36

# Mixed source
RUN_NAME=source_ccpp_js \
PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
bash run/fullpipeline.sh 36
```

Phase 1 không khóa cứng tên ngôn ngữ. Mọi JSONL hợp lệ đều được chấp nhận nếu mỗi dòng có:

```json
{
  "code": "...",
  "label": 0,
  "cwe": "CWE-089",
  "cwe_id": 89,
  "cwe_class": 3,
  "lang": "python_or_any_source_language"
}
```

`code`, `label` và ngôn ngữ là bắt buộc. Loader chấp nhận cả field chuẩn `lang` lẫn field legacy `language`, rồi normalize thành `lang` trong bộ nhớ mà không sửa dataset. Nếu cả hai cùng tồn tại nhưng khác giá trị, loader fail rõ ràng. Label phải thuộc `{0,1}`. CWE không hợp lệ được gán `-100`: mẫu vẫn dùng cho vulnerability loss nhưng không dùng cho CWE loss.

Các hyperparameter khác cũng nằm trong `run/config.sh`:

- Shared: `MODEL_NAME`, `MAX_LENGTH`, `TRUNCATION_STRATEGY`, batch size, weight decay, clipping, patience.
- Phase 1: data path, epochs, learning rate, `LAMBDA_CWE`.
- Phase 2: epochs, learning rate và các tham số RecAdam.
- Baseline: epochs và learning rate.
- Smoke test: số mẫu và sequence length nhỏ.

Giá trị truyền qua environment sẽ override config mà không cần sửa file.

## 2. Cấu trúc output

Với `RUN_NAME=my_run` và seed 36:

```text
log/my_run/
├── transfer/seed_36/
├── baseline/seed_36/
└── compare/seed_36/

model/my_run/
├── transfer/seed_36/
│   ├── source/best.pt
│   ├── fold1/best.pt
│   └── ...
└── baseline/seed_36/
    ├── fold1/best.pt
    └── ...

results/my_run/
├── transfer/seed_36/
├── baseline/seed_36/
└── compare/seed_36/
```

Transfer và baseline của `fullcompare.sh` luôn nằm trong cùng một `RUN_NAME` cha.

Log dùng append mode. Nếu file chưa có, `tee -a` tự tạo; nếu đã có, session mới được nối tiếp. Model/result vẫn được cập nhật khi chạy lại cùng `RUN_NAME` và seed, vì vậy hãy dùng `RUN_NAME` mới nếu muốn bảo toàn toàn bộ một lần chạy cũ.

## 3. Cách chạy

Scripts tự kích hoạt Conda environment `vdenv`.

### Transfer

```bash
bash run/train-phase1.sh 36
bash run/train-phase2.sh 1 36
bash run/test.sh 1 36
```

Chạy source một lần rồi cả năm Python folds:

```bash
bash run/fullpipeline.sh 36
```

### Baseline

Train một fold:

```bash
bash run/train-baseline.sh 1 36
```

Inference baseline trên validation và test, bao gồm threshold calibration và report từng CWE:

```bash
bash run/infer-baseline.sh 1 36
```

`run/test-baseline.sh` là tên tương thích và thực hiện cùng inference.

Chạy train + inference cả năm folds:

```bash
bash run/fullbaseline.sh 36
```

### So sánh đầy đủ

```bash
bash run/fullcompare.sh 36
```

Lệnh này chạy transfer và baseline trong cùng `RUN_NAME`, summarize riêng từng phương pháp rồi tạo:

```text
results/<RUN_NAME>/compare/seed_36/comparison.json
results/<RUN_NAME>/compare/seed_36/paired_folds.csv
```

Delta được định nghĩa là `transfer - baseline` trên cùng fold.

### Smoke test

```bash
bash run/smoke-test.sh 36
bash run/smoke-baseline.sh 36
```

Smoke artifacts nằm trong subtree `smoke`, không ghi đè full-run checkpoints.

## 4. Log và evaluation

Log có dạng:

```text
2026-04-29 04:24:30 - INFO - Using device: cuda
2026-04-29 04:25:26 - INFO - Epoch 1/20 | Train loss: ...
```

Không ghi logger name hoặc function name. Mỗi dòng classification report cũng có timestamp.

Mỗi epoch log:

- train/validation loss;
- sklearn classification report;
- confusion matrix và phân bố probability;
- train, validation, epoch và cumulative time;
- RecAdam target-task weight nếu là Phase 2;
- best epoch và early-stopping counter.

Inference log và lưu JSON cho:

- threshold 0.5;
- threshold tìm trên validation;
- test report tổng thể;
- report riêng CWE-022, CWE-078, CWE-079 và CWE-089 ở cả hai threshold;
- ROC-AUC, PR-AUC và thời gian inference.

Test chỉ được inference một lần. Hai threshold dùng lại cùng test probabilities.

## 5. Công dụng từng file

### Python

- `src/model.py`: kiến trúc transfer và baseline.
- `src/dataset.py`: tokenization và head-middle-tail truncation.
- `src/train_transfer.py`: CLI Phase 1, Phase 2 và transfer inference.
- `src/train_baseline.py`: CLI baseline train/inference.
- `src/train.py`: training epochs, early stopping, timing và RecAdam assertions.
- `src/evaluate.py`: metrics, threshold calibration, classification/per-CWE reports.
- `src/RecAdam.py`: optimizer RecAdam.
- `src/logging_utils.py`: format logger dùng chung.
- `src/summarize_results.py`: mean/std cho năm folds của một phương pháp.
- `src/compare_results.py`: paired comparison transfer và baseline.

### Shell

- `run/config.sh`: run namespace, data path và hyperparameter.
- `run/train-phase1.sh`: source multitask pretraining.
- `run/train-phase2.sh`: RecAdam transfer một fold.
- `run/test.sh`: transfer inference một fold.
- `run/fullpipeline.sh`: transfer đầy đủ năm folds.
- `run/train-baseline.sh`: baseline training một fold.
- `run/infer-baseline.sh`: baseline inference một fold.
- `run/test-baseline.sh`: compatibility alias target cho inference.
- `run/fullbaseline.sh`: baseline đầy đủ năm folds.
- `run/fullcompare.sh`: chạy và so sánh cả hai phương pháp.
- `run/smoke-test.sh`, `run/smoke-baseline.sh`: kiểm tra nhanh end-to-end.

## 6. CLI trực tiếp

```bash
python src/train_transfer.py --help
python src/train_baseline.py --help
python src/summarize_results.py --help
python src/compare_results.py --help
```

Pipeline hiện không dùng AMP, scheduler, PCGrad, adapter, MoE hoặc oracle CWE tại inference.
