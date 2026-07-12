# Method: Source Multitask Pretraining → Python RecAdam Transfer

## 1. Mục tiêu

Phương pháp hiện tại kiểm tra liệu kiến thức vulnerability/CWE học từ một source-code dataset có giúp phát hiện lỗ hổng Python hay không. Source dataset có thể chứa C/C++, JavaScript hoặc bất kỳ ngôn ngữ nào miễn đúng schema. Field ngôn ngữ chấp nhận `lang` hoặc legacy alias `language`. Target cố định là năm Python folds.

Phương pháp được so sánh với một baseline CodeBERT fine-tuning trực tiếp trên đúng Python folds và cùng seed.

## 2. Kiến trúc transfer

`TransferModel` gồm:

```text
source code
    ↓
CodeBERT backbone
    ↓ CLS representation
dropout
    ├── vulnerability head: Linear(hidden_size, 2)
    └── CWE head:           Linear(hidden_size, 4)
```

Bốn CWE classes:

```text
CWE-022 → 0
CWE-078 → 1
CWE-079 → 2
CWE-089 → 3
```

## 3. Phase 1: source multitask pretraining

Phase 1 chỉ đọc `PHASE1_DATA_PATH`. Python folds không được đọc trong phase này.

Backbone, vulnerability head và CWE head đều được update bằng AdamW. Với mẫu thứ `i`:

```text
L_phase1 = L_vulnerability + λ_cwe × L_CWE
```

Trong đó:

- `L_vulnerability`: cross-entropy nhị phân hai logits.
- `L_CWE`: 4-class cross-entropy.
- `λ_cwe`: `LAMBDA_CWE` trong config.
- CWE không hợp lệ có target `-100` và chỉ bị loại khỏi `L_CWE`.

Source được split 90/10 deterministic theo seed:

1. Ưu tiên group ID có thật như pair/commit/project/repo/CVE.
2. Nếu không có, bảo vệ các vulnerable/fixed rows liền nhau cùng language/CWE như một group.
3. Nếu không phát hiện group, fallback stratification theo label và CWE.

Split được thực hiện riêng cho từng giá trị `lang`, sau đó ghép lại. Dataset không có upstream group metadata vẫn có nguy cơ distant near-duplicate leakage; log đưa ra warning rõ ràng.

Best source checkpoint được chọn bằng validation Macro-F1 tại threshold 0.5.

## 4. Phase 2: Python sequential transfer

Mỗi Python fold thực hiện độc lập:

1. Load toàn bộ best source checkpoint.
2. Chỉ đọc Python train/validation của fold đó.
3. Freeze CWE head.
4. Giữ CodeBERT backbone và vulnerability head trainable.
5. Clone chính xác toàn bộ trainable source parameters trước optimizer step Python đầu tiên.
6. Optimize bằng RecAdam với vulnerability cross-entropy.

Không dùng CWE label trong Phase 2 forward hoặc inference.

## 5. RecAdam

RecAdam cân bằng hai mục tiêu:

- học target Python;
- không rời source solution quá nhanh.

Gọi:

- `θ` là trainable parameters hiện tại;
- `θ*` là bản clone bất biến từ source checkpoint;
- `L_T(θ)` là Python vulnerability loss;
- `λ(t)` là annealing weight tại optimizer step `t`;
- `γ` là `PRETRAIN_COF`.

Objective khái niệm:

```text
L_RecAdam(t) = λ(t) L_T(θ)
             + (1 - λ(t)) γ/2 Σᵢ ||θᵢ - θᵢ*||²
```

Với sigmoid annealing:

```text
λ(t) = ANNEAL_W / (1 + exp(-ANNEAL_K × (t - t0)))

t0 = max(1, int(ANNEAL_T0_RATIO × total_training_steps))
```

Đầu quá trình, source-anchor penalty mạnh hơn. Khi `t` tăng, target gradient dần chiếm ưu thế. Implementation nằm trực tiếp trong optimizer; không thay bằng cách cộng L2 penalty vào task loss.

Các kiểm tra ở optimizer step đầu tiên:

- current/source parameter lists cùng length, order và shape;
- source clones không `requires_grad` và không thay đổi;
- CWE head frozen và không có gradient;
- backbone hoặc vulnerability head có gradient;
- ít nhất một trainable parameter thay đổi sau step.

Best target checkpoint được chọn bằng Python validation Macro-F1 tại threshold 0.5. Exact ties giữ checkpoint mới nhất nhưng không reset early-stopping patience.

## 6. Baseline

Baseline không dùng Phase 1 hoặc RecAdam:

```text
Python source code
    ↓
fresh CodeBERT backbone
    ↓ CLS representation
dropout
    ↓
Linear(hidden_size, 2)
```

Mỗi fold khởi tạo lại bằng cùng seed và train bằng AdamW. Để so sánh công bằng, baseline dùng chung:

- Python folds;
- tokenizer và truncation strategy;
- batch/evaluation batch size;
- weight decay và gradient clipping;
- early stopping/checkpoint rule;
- validation threshold calibration;
- test protocol.

## 7. Tokenization

Code dài hơn `MAX_LENGTH` dùng `TRUNCATION_STRATEGY`:

- `head`: giữ prefix.
- `head_middle_tail`: giữ ba cửa sổ deterministic ở đầu, giữa và cuối.

Strategy được lưu trong checkpoint. Inference fail sớm nếu checkpoint và current strategy không khớp.

## 8. Evaluation protocol

Sau training:

1. Load best validation checkpoint.
2. Inference validation.
3. Tìm threshold từ 0.05 đến 0.95, bước 0.01, tối đa hóa validation Macro-F1.
4. Đóng băng threshold.
5. Inference test đúng một lần.
6. Dùng cùng probabilities để tính metric tại 0.5 và validation-calibrated threshold.

Metrics tổng thể:

- Macro-F1 và positive-class F1;
- precision, recall, accuracy;
- ROC-AUC và PR-AUC.

Test còn báo cáo riêng CWE-022/078/079/089 ở cả threshold 0.5 và val-calibrated threshold. CWE chỉ dùng để chia subgroup sau inference, không được đưa vào model.

## 9. Reproducibility và phạm vi

Seed được đặt cho Python `random`, NumPy, PyTorch, CUDA và DataLoader. cuDNN deterministic được bật. Kết quả vẫn có thể khác giữa GPU, CUDA, PyTorch hoặc Transformers versions.

Pipeline hiện không dùng AMP, scheduler, adapter, MoE, PCGrad, contrastive learning hoặc oracle CWE.
