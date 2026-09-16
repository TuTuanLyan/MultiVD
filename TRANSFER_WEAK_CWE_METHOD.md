# Nhánh `transferweakcwe` — đọc file nào trong `src/` là đủ

`src/` có **34 file, 8 133 dòng**, phần lớn là hạ tầng cũ hoặc hướng đã bị loại. File này nói
**đọc cái gì** cho từng phương pháp, và **bỏ qua cái gì**.

> Số đo của mọi phát biểu nằm ở `FACTS.md` theo mục đánh số. Quy tắc chạy ở `CLAUDE.md`.
> Kết quả **không** nằm trong git (xem `.gitignore`) — chúng ở trên đĩa làm việc và máy chạy.

---

## 1. Đường chạy chính — đọc bốn file này trước

| file | dòng | vai trò |
|---|---|---|
| **`src/dataset.py`** | ~80 | `CodeDataset`. **Quan trọng nhất là `_truncate`**: mặc định `head_middle_tail` — giữ **170 đầu + 170 giữa + 170 cuối** chứ không cắt đầu. Hiểu sai chỗ này dẫn tới kết luận sai (FACTS §59 → §59.1) |
| **`src/model.py`** | ~420 | `build_backbone` (codebert / codet5p, `trust_remote_code`), `pool_hidden_states` (cls / mean), `BaselineModel`, `TransferModel` (backbone + `vul_head` + head phụ) |
| **`src/train_baseline.py`** | ~310 | **Model A** — chỉ học tập đích, không có Pha 1. Là đối chứng của mọi phép so |
| **`src/train_transfer.py`** | ~2000 | **Model B** — hai pha. `--phase phase1` học trên nguồn, `--phase phase2` fine-tune sang đích, `--phase test` chấm điểm |

`src/train.py` là vòng huấn luyện dùng chung cho cả hai (`train_one_epoch_phase1`, `train_loop`).
`src/evaluate.py` là chỗ tính chỉ số và ngưỡng hiệu chuẩn theo val.

## 2. Theo từng phương pháp

| phương pháp | đọc | trạng thái |
|---|---|---|
| **Transfer hai pha** (nền của mọi thứ) | `train_transfer.py` + `train.py` | đang dùng |
| **Head phụ `latent_bottleneck`** | `model.py::TransferModel`, cờ `--aux_mode` | **đã đóng** — FACTS §56: +0.0053, 5/10. Bỏ được |
| **Adapter + AdapterFusion** | `src/adapters.py` + `tests/test_adapters.py` (18 phép kiểm hai chiều) | **đã đóng** — §48/§50: một nửa lợi ích tái tạo được bằng adapter **nhiễu** |
| **Biên trong cặp** (Đề xuất B) | `src/pairloss.py` + `tests/test_pairloss.py` (13 phép kiểm) | **đã đóng** — §57: thua BCE |
| **Ghép muộn hai model** (Đề xuất 1) | `tools/late_fusion_gate.py` — **không nằm trong `src/`**, nó chỉ đọc xác suất đã lưu | đang chạy · §61, §63, §64 |
| **Nhiều cửa sổ (MW)** | `src/train_mw.py` | **mới, chưa có kết quả** |
| SAM / ASAM | `src/sam.py` | §2b: cải thiện **thứ hạng**, không cải thiện ngưỡng 0.5 |
| RecAdam | `src/RecAdam.py`, `src/recadam_fisher.py` | ổn định nhất giữa backbone nhưng không thắng |

## 3. Dựng dữ liệu

| file | làm gì |
|---|---|
| `src/build_sources.py` | dựng `data/phase1_{4cwe,common,full}.jsonl` từ PrimeVul + CleanVul |
| `src/build_folds.py` | chia `data/sven_python_folds_norm` — **chia ngẫu nhiên theo từng dòng là CHỦ Ý**, không phải lỗi |
| `src/build_shuffled_labels.py` | đối chứng xáo nhãn (§54), có kiểm hai chiều |

## 4. **Bỏ qua** — mã chết hoặc hướng đã bị bác

| | vì sao |
|---|---|
| `LoRALinear`, `inject_lora`, `--lora_rank` trong `train_transfer.py` | **mã chết**: `lora_rank=0` ở toàn bộ 1 452 ô có ghi hyperparameters. Dự án luôn fine-tune cả model |
| `latent_proto`, `--aux_mode latent_proto` | head riêng ≈ 0 mọi khối, tự sập ở `codebert × full` |
| `--aux_mode cwe` | cần nhãn CWE nên chỉ chạy được trên nguồn `4cwe`; độ tản giữa backbone 0.0443 |
| `src/replay.py`, `src/spd.py`, `src/fisher.py`, `src/uncertainty_weighting.py`, `src/anneal_adamw.py` | các nhánh quét đã thử và bỏ, giữ để tra cứu |
| `src/measure_*.py`, `src/inspect_backbone.py`, `src/audit_coverage.py` | công cụ phân tích một lần, không nằm trên đường chạy |

## 5. Công cụ đọc kết quả (`tools/`) — bắt buộc dùng, không tự tính tay

| file | vì sao bắt buộc |
|---|---|
| **`tools/report2.py`** | in **cả bốn** chỉ số kèm **đếm dấu**. Đọc một chỉ số đã giữ một kết luận sai suốt ba tuần (CLAUDE.md §2b) |
| `tools/leak_groups.py` · `leak_groups_pair.py` · `leak_groups_auc.py` | tách theo **nhóm rò rỉ** — phép kiểm reviewer yêu cầu |
| `tools/late_fusion_gate.py` | cổng ghép muộn: `g05` / `const` / `logreg` |
| `tools/gate_readout.py` | phân bố `g` theo CWE — đọc cơ chế |
| `tools/gate1_decide.py` · `gate2_analyse.py` | áp **đúng ngưỡng đã ghi trước**; thiếu dữ liệu thì trả mã 2 chứ không im lặng báo "đạt" |
| `tools/embed_target.py` | nhúng tập đích bằng backbone **gốc**, dùng cho router |

## 6. Ba cái bẫy đã trả giá — đọc trước khi sửa hạ tầng

1. **`MODES=""` không tắt được vòng lặp** — `run/matrix.sh:69` dùng `${MODES:-...}` nên chuỗi
   rỗng bị thay bằng mặc định 4 nhánh. Dùng `BASELINE_ONLY=1`.
2. **`pgrep -f` / `pkill -f` khớp chính dòng lệnh của mình.** Lọc theo `comm`, hoặc dùng `flock`.
3. **Đặt tham số mới ở CUỐI chữ ký hàm.** Chèn vào giữa làm mọi lời gọi theo vị trí lệch một
   bậc và lỗi nổ ở chỗ không liên quan.

## 7. Chạy

```bash
# baseline (model A)
PYTHON=<env> RUN_NAME=<tên> SEED=42 FOLDS="1 2 3" BACKBONES="codebert=microsoft/codebert-base:cls" \
  BASELINE_ONLY=1 SKIP_BASELINE=0 bash run/matrix.sh

# transfer hai pha (model B)
PYTHON=<env> RUN_NAME=<tên> SEED=42 FOLDS="1 2 3" MODES=none ARM_TAG=_com_real PHASE1_TAG=_com_real \
  PHASE1_STORE=model/shuf1/phase1 PHASE1_DATA_PATH=data/phase1_common.jsonl bash run/matrix.sh

# ghép muộn (0 GPU, đọc xác suất đã lưu)
python3 tools/late_fusion_gate.py --a baseline --b transfer_none_com_real results/<cây>
```
