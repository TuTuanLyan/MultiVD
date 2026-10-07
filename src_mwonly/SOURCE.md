# src_mwonly - mã nhánh `refactor` của maytinhdibo/GraphTransferVD (thư mục refactorMWG/) + tất định / TF32

- commit: `ab094cfb4a6f1925f4b7e5d4be69c7cd2a855b1c` (2026-10-04 20:28:10 +0700, "refactorMWG/run_p1.sh: Pha 1 mặc định 16 epoch, warmup 0,25; kiểm kẹt ở epoch 5")
- lấy ngày 2026-10-04 20:4x bằng `git archive HEAD refactorMWG`, BỎ `data/` (khối dùng `data/final_experiment_data/`).
- dùng cho khối MW-only n=5 (người dùng 04/10 20:4x): không đồ thị (`--fusion mw`), seed 42, TF32, tái lập từng bit.

## Sửa so với upstream (chỉ 2 file)

| file | sửa |
|---|---|
| `mwg/utils.py` | `set_seed(seed, deterministic=False, tf32=False)`: `deterministic` đặt `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `torch.use_deterministic_algorithms(True)` (không `warn_only`); `tf32` bật/tắt TF32 ở matmul + cuDNN. Lấy từ `set_seed(strict=True)` của mã latent bottleneck cũ (`src/train_transfer.py`) và `det_launch.py` của khối final. |
| `train.py` | cờ `--deterministic` (mặc định **1**) và `--tf32` (mặc định 0); `deterministic` mà `PYTHONHASHSEED` khác 42 thì tự chạy lại chính nó với `PYTHONHASHSEED=42`; log dòng `Tái lập | ...`; DataLoader trộn bằng `torch.Generator` riêng theo seed. Hai cờ tự vào `hyperparameters` của file kết quả. |

## Đã kiểm (161, RTX A4000, TF32, 04/10 20:5x)

- Pha 1 (pair loss, `--fusion mw`) chạy 2 lần: 218/218 tensor trùng từng bit. Lần không đặt `PYTHONHASHSEED` tự chạy lại với 42.
- Pha 2 (`--init all`, RecAdam cof 500 + ASAM ρ 0,5) chạy 2 lần + test: xác suất val và test trùng từng bit.

## Sửa thêm 04/10 22:1x (người dùng: "trong khi pha 1 chưa giảm train loss > 2% thì lấy check point theo train loss, sau đó mới lấy theo val ROC")

`train.py` thêm cờ `--select_after_drop` (mặc định 0 = tắt, hành vi cũ): chọn checkpoint theo train loss nhỏ nhất cho tới epoch đầu tiên có
train loss < (1 - giá trị) × train loss epoch 1, từ đó theo `--selection_metric` (điểm cũ bỏ đi, `best` đặt lại). Log dòng
`Chọn checkpoint theo <metric> từ epoch k (...)`. Kiểm GPU 161, 4CWE chỉ JS 96 hàm × 4 epoch: (a) cờ 0 trùng từng bit mã trước khi sửa
(log + checkpoint); (b) lr 1e-9 (loss không giảm đủ 2 %) chọn theo train loss suốt, lấy ep4; (c) lr 5,66e-5 chuyển sang val ROC ở ep2
(0,7535 < 98 % × 0,7929).

## Đẩy lên upstream 05/10 01:0x (người dùng: "push phần code mà dùng cách lấy check point mới lên nhánh refactor")

Chỉ phần `--select_after_drop` (train.py) + `P1_SELECT_AFTER_DROP` (scripts/run_p1.sh, mặc định 0.02) + 1 dòng README, đặt lên
`07fd99f` (commit của tác giả 04/10 20:49 thêm `--tf32` riêng: `torch.set_float32_matmul_precision("high")`, chỉ matmul) ⇒ commit
**`3ecc580`** trên `refactor`. KHÔNG đẩy phần tất định (`--deterministic`) và `--tf32` của bản này (trùng tên cờ với upstream - sẽ
lỗi argparse; upstream bật TF32 chỉ ở matmul, bản này bật cả matmul lẫn cuDNN). Nên `src_mwonly` (khối mwonly5) và upstream KHÔNG
trùng file: khác ở set_seed / --deterministic / --tf32 / Generator của DataLoader.

## Đẩy lên upstream lần 2, 05/10 01:3x (người dùng: "có đẩy thêm cờ chế độ tất định và --tf32 nhé")

Commit **`ae268e9`** (trên `3ecc580`): `--deterministic` (train.py mặc định 0; scripts `DETERMINISTIC=1` mặc định, theo đúng mẫu TF32
của tác giả), `set_seed(seed, deterministic)`, PYTHONHASHSEED tự chạy lại, DataLoader Generator riêng khi tất định, dòng log `Tái lập`.
`--tf32` GIỮ của upstream (07fd99f) - không thêm cờ trùng tên. Kiểm CPU: upstream `--deterministic 1 --tf32 1` trùng từng bit
`src_mwonly` `--deterministic 1 --tf32 1`; `--deterministic 0` trùng bit upstream trước khi sửa. Khác còn lại giữa hai bản: mặc định
của `--deterministic` (src_mwonly 1, upstream 0) và cách bật TF32 (src_mwonly đặt matmul + cuDNN theo cờ; upstream chỉ đặt matmul,
cuDNN để mặc định của PyTorch = bật) - khối mwonly5 luôn truyền `--deterministic 1 --tf32 1` nên chạy cùng phép tính.
