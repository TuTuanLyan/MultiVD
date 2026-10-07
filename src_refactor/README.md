# refactorMWG — MWG: dựng dữ liệu và tái lập thí nghiệm

MWG phát hiện hàm có lỗ hổng bằng **CodeBERT nhiều cửa sổ** cộng **nhánh đồ thị dòng** (lấy từ BABEL, dùng chung
cho mọi ngôn ngữ). Mô hình được chuyển giao hai pha: học trên nguồn Java + JavaScript rồi tinh chỉnh trên đích Python
(SVEN). Thư mục này chỉ chứa mã để dựng bộ dữ liệu và chạy lại thí nghiệm; lịch sử cũ nằm ở `../archive/`.

## Kiến trúc

```
hàm ──> token ──> cửa sổ 510 token, bước 384, tối đa K = 8 ──> CodeBERT (dùng chung)
                                                                  │
          ┌── CLS mỗi cửa sổ + vị trí cửa sổ ──> trung bình ─────┴──────────────────────────> z_mw ──┐
          │                                                                                          ├─ [z_mw; z_g] ─> Linear ─> {lành, lỗi}
          └── trạng thái token ──> trung bình theo dòng ──> H0 ──> R-GCN ×2 ──> LSTM ──> max-pool ──> z_g ──┘
                                                                     ▲
dòng mã ──> định danh thành VARk / FUNk ──> 4 quan hệ giữa các dòng ─┘
```

- **Nhánh multi-window** (`mwg/model.py`): mỗi cửa sổ qua CodeBERT, lấy CLS cộng embedding vị trí cửa sổ, rồi lấy trung bình → `z_mw`.
- **Nhánh đồ thị dòng** (`mwg/graph.py`, `mwg/data.py`):
  - Mỗi dòng không rỗng là một đỉnh; bỏ dòng chỉ có ngoặc; tối đa 150 dòng.
  - Vector đỉnh là trung bình trạng thái CodeBERT của các token thuộc dòng đó, nên không cần encoder thứ hai.
  - Cạnh tính trên mã đã chuẩn hoá định danh, không cần parser, gồm 4 quan hệ:
    - `def_use`: định nghĩa → lần dùng;
    - `co_use`: hai lần xuất hiện liên tiếp của cùng ký hiệu;
    - `ctrl`: cây thụt lề;
    - `next`: dòng kế tiếp.
  - Sau đó qua R-GCN 2 lớp (residual + LayerNorm), LSTM, max-pool → `z_g`.
- **Chuyển giao hai pha** (`train.py`):
  - **Pha 1** (nguồn, chạy 1 lần): CE + pair loss. Pair loss buộc điểm của hàm lỗi cao hơn bản vá của nó ít nhất margin 1.
  - **Pha 2** (đích, mỗi fold):
    - khởi tạo mọi phần trừ head từ Pha 1;
    - RecAdam kéo trọng số về điểm Pha 1 để chống quên;
    - ASAM tìm cực tiểu phẳng.
- **Target-only** (mốc so sánh): cùng kiến trúc, học thẳng trên đích bằng AdamW trần.

## Cách chạy

```bash
pip install -r requirements.txt            # torch 2.9.1, transformers 4.57.1 (đã kiểm với CUDA 12.8, RTX A4000 16 GB)
python tests/test_symbols.py               # 32/32: xoá comment + chuẩn hoá định danh
python tests/test_stuck.py                 # 20/20: luật phát hiện kẹt Pha 1 trên các lượt đã chạy

# 1. dựng dữ liệu (CPU), kiểm md5 69 file
PRIMEVUL_RAW=/path/PrimeVulRaw CLEANVUL_DIR=/path/cleanvul_csv bash scripts/build_data.sh

# 2. thí nghiệm (MODEL = thư mục microsoft/codebert-base)
MODEL=/path/codebert-base bash scripts/run_sota.sh          # Pha 1 (tự khởi động lại khi kẹt) + Pha 2 trên 5 fold
MODEL=/path/codebert-base bash scripts/run_p1.sh            # chỉ Pha 1
MODEL=/path/codebert-base bash scripts/run_target_only.sh   # mốc target-only
python scripts/summarize.py results/sota_p2/multiwindow/seed_36
```

| biến | mặc định | ý nghĩa |
|---|---|---|
| `SEED` / `SEED_P1` | 36 / `SEED` | seed Pha 2 / seed đầu tiên của Pha 1 |
| `TAG` | `sota` | tên lượt; kết quả ở `results/<TAG>_p2/`, log ở `log/<TAG>_p{1,2}/` |
| `FOLDS` | `1 2 3 4 5` | các fold Pha 2 |
| `P1_TRIES` | 3 | số lần chạy Pha 1 tối đa (seed `SEED_P1`, `SEED_P1+1`, …) |
| `STUCK_EPOCH` / `STUCK_MIN_DROP` | 3 / 0,02 | epoch quyết định kẹt / độ giảm train loss tối thiểu giữa hai epoch |
| `KEEP_STUCK` | 0 | 1 = giữ checkpoint lượt kẹt (`stuck.pt`) để kiểm bằng Pha 2 thay vì xoá |
| `P1_SELECT` / `P1_ALSO_SELECT` | `roc_auc` / — | metric chọn checkpoint Pha 1 (`train_loss` = train loss nhỏ nhất) / giữ thêm `best_<metric>.pt` để so hai cách chọn |
| `PY` | `python` | trình thông dịch |

Thời gian trên RTX A4000: Pha 1 khoảng 3,5 giờ, mỗi fold Pha 2 khoảng 50–80 phút. Các script bỏ qua phần đã có
kết quả nên chạy lại là chạy tiếp.

### Pha 1 tự khởi động lại khi kẹt

Pha 1 đôi khi kẹt ở nghiệm tầm thường: train loss đứng yên, val ROC khoảng 0,5. Tỉ lệ khoảng 1/3 số lượt trên nguồn v4.
Nguyên nhân là nhiễu GPU của attention backward, không phải do mã: cùng seed có lượt kẹt, có lượt không. Checkpoint Pha 1 kẹt có thể làm Pha 2 sụp.

`scripts/run_p1.sh` gọi `train.py --stuck_epoch 3 --stuck_min_drop 0.02`:

- Cuối epoch 3, so train loss với epoch 2. Nếu giảm chưa tới 2 % (tương đối) thì dừng với mã thoát 3.
- Khi đó script xoá checkpoint, đổi log thành `f1_seed<S>_stuck.log` và chạy lại với seed kế tiếp.
- Luật chỉ đo độ giảm giữa hai epoch, không so với một mức loss cố định.
- Checkpoint dùng được ghi vào `model/<TAG>_p1/P1_CKPT`.

Trên 20 lượt Pha 1 đã chạy (`tests/test_stuck.py`), độ giảm ở epoch 3:

| nhóm | độ giảm ở epoch 3 |
|---|---|
| lượt kẹt | từ −0,2 % đến 1,1 % |
| lượt học, cùng pool SOTA | từ 8,3 % trở lên |
| lượt học, mọi pool | từ 2,5 % trở lên |

### Hyperparameter SOTA

| nhóm | Pha 1 (nguồn) | Pha 2 (đích) |
|---|---|---|
| dữ liệu | `mwsrc_v4jsCjv_rand`: JS common 990 + Java 5 184 hàm, val 10 % ngẫu nhiên | `sven_python_folds_v4`, 5 fold |
| khởi tạo | CodeBERT-base | `--init all` từ Pha 1, head mới |
| mô hình | window 510, stride 384, K = 8, agg `mean`, đồ thị typed 4 quan hệ, `drop_bracket 1`, `co_mode chain`, fusion `cat` | như Pha 1 |
| lr | 2e-5, `graph_lr` 2e-5, warmup 0,10 | 2e-5, `graph_lr` 1e-4, warmup 0,10 |
| optimizer | AdamW, wd 0,01, grad clip 1,0 | RecAdam (`pretrain_cof` 500, `anneal_t0_ratio` 0,01) + ASAM ρ 0,5, η 0,01 |
| loss | CE + pair loss 1,0 (margin 1,0) | CE |
| epoch | 8, chọn theo val ROC | tối đa 30, patience 8, chọn theo val ROC |
| batch | 4 hàm, 16 cửa sổ mỗi lượt CodeBERT | như Pha 1 |

**Ablation:** dùng các cờ của `train.py`:

| cờ | tác dụng |
|---|---|
| `--fusion mw` | bỏ đồ thị |
| `--graph babel` | 2 quan hệ gốc |
| `--co_mode all` | co_use nối mọi cặp |
| `--line_enc babel` | WordAttNet gốc |
| `--norm_text 1` | CodeBERT đọc mã đã chuẩn hoá |
| `--sam_variant sam` | SAM thay ASAM |

**Đọc kết quả:**

- Cùng seed chạy hai lần vẫn lệch, do GPU không tất định.
- Giữa hai máy có thể lệch tới khoảng 2 điểm ROC.
- Vì vậy chỉ so các lượt chạy cùng máy, trên 5 fold, và kiểm bằng bootstrap theo commit.
- Tham khảo: các lượt SOTA trên RTX A4000 cho ROC trung bình 5 fold 0,933–0,949.

## Dữ liệu

| nguồn | lấy từ |
|---|---|
| PrimeVul v0.1 (C/C++) | bản phát hành PrimeVul, gồm `primevul_{train,valid,test}[_paired].jsonl` |
| CleanVul (Java, JS) | GitHub `yikun-li/CleanVul` @ `cbad711`, gồm `vulnerability_score_{3,4}.csv`. Không dùng bản HuggingFace |
| đặc tả CWE | MITRE `cwec_v4.20.xml` (script tự tải) |
| đích SVEN (Python) | có sẵn trong `data/sven_python_folds_norm` (760 hàm, 5 fold) |

`scripts/build_data.sh` chạy lần lượt:

1. **`dataset/build_sources.py`**:
   - xoá comment bằng `mwg/comments.py`, bộ xoá dùng chung với lúc dựng đồ thị;
   - lọc theo các luật:
     - **T0**: rỗng;
     - **T1**: trùng mã mà ngược nhãn;
     - **T2**: mã sinh hoặc nén (JS), không có thân;
     - **T3**: trùng nguyên văn;
     - **T4**: trùng sau khi trừu tượng hoá.
   - Bỏ một nửa cặp thì bỏ cả cặp. Lý do bị loại ghi ở `data/sources_v4/_dropped/`.
   - Tách bộ `common`: giữ hàm có mọi CWE đều thuộc nhóm có ở Python (luật R1–R6 trong `dataset/cwe.py`).
2. **`dataset/build_pools.py`**: dựng pool Pha 1, gồm pool SOTA `mwsrc_v4jsCjv_rand` (train 5 557 / val 617, chia theo hàm, seed 36).
3. **`dataset/make_target.py`**: xoá comment và docstring của SVEN, giữ nguyên thứ tự, nhãn và fold → `sven_python_folds_v4`.
4. **Kiểm md5**: so kết quả với `dataset/EXPECTED.md5`.

## Thư mục

```
mwg/        comments.py, symbols.py (VARk/FUNk), graph.py (4 quan hệ), data.py, model.py, optim.py (RecAdam, SAM/ASAM), metrics.py, utils.py
train.py    huấn luyện / kiểm tra một fold (cờ, checkpoint, json kết quả giữ như train_mwg.py cũ)
dataset/    dựng dữ liệu từ dữ liệu thô + EXPECTED.md5
scripts/    build_data.sh, run_p1.sh, run_sota.sh, run_target_only.sh, summarize.py
tests/      test_symbols.py, test_stuck.py
```

Đã kiểm tương đương với mã cũ `../archive/` (25/09/2026):

- Dữ liệu: 69/69 file khớp md5.
- Xoá comment, chuẩn hoá định danh, đồ thị và tensor dataset: trùng từng ký tự với mã cũ.
- `train.py`, chạy CPU cho cả Pha 1 và Pha 2: trọng số, xác suất và log trùng từng bit.
- Chạy GPU: lệch cũ–mới ngang lệch cũ–cũ.
- Cờ `--stuck_epoch` (thêm 26/09) mặc định tắt, không đổi phép tính; chỉ thêm hai khoá vào `hyperparameters`.
