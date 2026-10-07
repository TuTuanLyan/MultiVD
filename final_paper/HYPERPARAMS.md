# Siêu tham số — khối chạy cho bài (21/09/2026)

Máy **duy nhất**: vast `51144271` nhãn `ntat`, RTX 5060 Ti 16 GB, driver 570.153.02,
torch 2.9.1+cu128, transformers 4.57.1. **Mọi ô của mọi phép so đều nằm trên card này**
(cuBLAS chỉ đảm bảo lặp lại trên *cùng kiến trúc và cùng số SM*).

Bậc: **n=5 fold, seed 42**. Chưa có n=15 nào.

---

## 0. Ba thay đổi so với `qp9_n5.sh`, và vì sao

| # | đổi gì | vì sao |
|---|---|---|
| 1 | **Pha 1**: `--graph_lr` **1e-4 → 2e-5** VÀ `--warmup_ratio` **0.10 → 0.25** (Pha 2 giữ nguyên) | **Sửa lỗi Pha 1 kẹt.** §16.8b của đồng tác giả chạy 5 biến thể và chốt: nhánh đồ thị 9,45 M tham số khởi tạo ngẫu nhiên ở lr **gấp 5 lần encoder** kéo đầu phân loại về nghiệm hằng số trong warmup. `glr2e5` thoát, `glr5e5` **không** thoát. `qp9_n5.sh` không truyền cờ này nên ăn mặc định 1e-4 — và Pha 1 `v2common` của ta kẹt đúng như vậy |
| 2 | seed **36 → 42** | người dùng chốt; 36/37/38 có vấn đề văn hoá |
| 3 | pool nguồn → `mwsrc_v2{common,full,cwe4}` | bộ nguồn v2 dựng lại từ dữ liệu gốc, comment xoá có kiểm chứng |

**Bằng chứng ta đang ở đúng chế độ kẹt đó** — mức bình nguyên lý thuyết là
`CE ln2 (0,69) + softplus(margin=1) (1,31) × tỉ lệ hàng nằm trong cặp`:

```
pool v2common: 3 502 cặp / 8 792 hàng = 79,6 %
du bao:  0,69 + 1,31 × 0,796 = 1,73
do duoc: 1,757 → 1,741 → 1,748     (val ROC 0,501 / 0,486 / 0,510)
```

Trùng tới 0,02. Không phải "common khó hơn" — là lỗi tối ưu hoá đã biết.

> **Chưa áp dụng cả `warmup 0.25` lẫn `margin 0.5`** dù §16.8b nói chúng cũng thoát.
> Mỗi thứ đổi là một biến mới; chỉ đổi **một** cái (`graph_lr`), đúng cái nhắm thẳng
> nguyên nhân. Watchdog epoch 3 (ngưỡng val ROC 0,60) vẫn giữ làm lưới an toàn.

---

## 1. Pha 1 — huấn luyện trên nguồn (`train_mwg.py`)

```
--phase train --fold 1 --data_root data/mwsrc_<pool>
--seed 42 --model_name codebert-base
--window 510 --stride 384 --max_windows 8 --max_length 512
--agg mean --agg_layers 2
--batch_size 4 --eval_batch_size 8 --micro 16
--learning_rate 2e-5 --graph_lr 2e-5        ← ĐỔI (mặc định 1e-4)
--weight_decay 0.01 --warmup_ratio 0.25 --max_grad_norm 1.0   ← ĐỔI (họ dùng 0.10)
--epochs 8 --min_epochs 3 --patience 8 --selection_metric roc_auc
--pair_loss 1.0 --pair_margin 1.0
--sam_rho 0        (SAM TẮT ở Pha 1)
--recadam 0        (RecAdam TẮT ở Pha 1)
--grad_checkpoint 1 --num_workers 0 --pooling cls
```

Pha 1 chạy **một lần** mỗi pool (fold 1), ra một checkpoint dùng lại cho cả 5 fold đích.

## 2. Pha 2 — chuyển sang đích python (`train_mwg.py`)

```
--phase train --fold 1..5
--data_root data/sven_python_folds_norm --target_lang python
--init all --init_ckpt <checkpoint Pha 1>
--epochs 30 --min_epochs 3 --patience 8 --selection_metric roc_auc
--learning_rate 2e-5 --graph_lr 1e-4 --weight_decay 0.01 --warmup_ratio 0.10   (GIỮ NGUYÊN)
--sam_rho 0.02                                      (SAM BẬT ở Pha 2)
--recadam 1 --pretrain_cof 500 --anneal_t0_ratio 0.01   (RecAdam-light)
  (mọi cờ còn lại giống Pha 1)
```

`--epochs 30` là **trần**; patience 8 nên số epoch chạy thật thấp hơn, và `epochs_run`
được ghi vào bản ghi tài nguyên.

---

## 3. Heuristic của nhánh BABEL — phần anh hỏi riêng

| cờ | giá trị | nó làm gì |
|---|---|---|
| `--graph typed` | `typed` | đồ thị **4 quan hệ**: `def_use` · `co_use` · `ctrl` · `next`. (`typed` → R=4; nếu không thì R=2 chỉ còn `co_use` + `ctrl`) |
| `--co_mode` | **`chain`** | cạnh `co_use` nối các dòng dùng chung ký hiệu theo **chuỗi** (mỗi dòng nối dòng dùng lại gần nhất) thay vì `all` (nối **mọi cặp**). `all` làm đồ thị dày đặc bậc hai theo số dòng dùng chung |
| `--drop_bracket` | **`1`** | **bỏ dòng chỉ có ngoặc** khỏi tập đỉnh. Help của chính cờ này ghi lý do: *"Python 1,1 % vs C/JS/Java 16–21 % → đồng nhất hoá đồ thị"* — tức chuyển giao từ C/JS/Java sang Python mà không bỏ thì đồ thị nguồn có 16–21 % đỉnh rác còn đích chỉ 1,1 %, hai phân bố đồ thị lệch hẳn nhau |
| `--max_lines` (Lmax) | `150` | đồ thị cắt ở 150 dòng. Pool `v2common` có p90 = 130 dòng, `pct_lines_truncated` 8,0 % |
| `--max_words` | `60` | số từ tối đa mỗi dòng đưa vào bộ mã hoá dòng |
| `--line_enc` | `codebert` | biểu diễn mỗi dòng lấy từ chính CodeBERT (không dùng bộ mã hoá riêng của BABEL) |
| `--graph_layers` | `2` | 2 lớp R-GCN |
| `--graph_lstm` | `1` | có LSTM trên dãy dòng sau R-GCN |
| `--fusion` | `cat` | nối biểu diễn nhánh cửa sổ và nhánh đồ thị (thay vì `sum` / chỉ `graph` / chỉ `mw`) |
| `--norm_text` | `0` | **không** chuẩn hoá tên biến/hàm (không `clean_gadget`) |
| `--graph_lr` | **`2e-5`** | learning rate **riêng** cho nhánh đồ thị + `word_att`. Đây là cờ gây kẹt ở mặc định 1e-4 |

Nhánh đồ thị: **9 449 472 tham số** (đo được), backbone 124 645 632, head 3 074 —
tổng **134 105 858**.

Cắt cửa sổ: `W=510, S=384, K=8` → mỗi hàm tối đa 8 cửa sổ chồng lấn 126 token, gộp
`mean` có `win_pos_emb` học vị trí cửa sổ.

---

## 4. Pool nguồn Pha 1 (val 0.2 ngẫu nhiên theo hàng, seed 42)

| pool | ccpp | java | js | tổng | train | val |
|---|---:|---:|---:|---:|---:|---:|
| `mwsrc_v2common` | 4 620 | 5 508 | 1 500 | **11 628** | 9 302 | 2 326 |
| `mwsrc_v2full` | 9 408 | 5 508 | 1 836 | **16 752** | 13 402 | 3 350 |
| `mwsrc_v2cwe4` | 178 | 0 | 920 | **1 098** | 878 | 220 |

Tiêu chí `common` dùng **luật MITRE R1–R6** (`tools/cwe_rules.py`, vendored từ
`GraphTransferVD@79f0308`), không còn danh sách tay. Đã tái lập đúng con số của đồng
tác giả: ccpp **4 620/9 408**.

## 5. Mốc đối chứng (đã chạy xong)

`clean_baseline_codebert_py`: codebert thuần, `max_length 512`, **không warmup không
decay** (`warmup_ratio 0` ⇒ `train_baseline.py` không tạo scheduler ⇒ LR hằng số 2e-5),
lr 2e-5, epochs 30, patience 8, batch 4/16, `head_middle_tail`, pooling `cls`, seed 42.

> Mốc này **cố ý khác** nhánh phương pháp ở chỗ lịch LR (người dùng chốt: baseline chạy
> thuần theo yêu cầu, so chính là với SOTA). Khi đặt hai cột cạnh nhau phải ghi rõ
> baseline LR hằng số còn nhánh phương pháp có warmup 0,10.
