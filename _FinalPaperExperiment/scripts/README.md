# Khối thí nghiệm final — cách chạy

Thư mục `scripts/` là **toàn bộ mã để chạy** khối final và là thư mục DUY NHẤT cần commit. Mọi định nghĩa run nằm trong
`runs.json`; script, bảng tham số và artifact đều sinh từ đó.

```
_FinalPaperExperiment/
  scripts/   runs.json · fpe.py · run.sh · launch.sh · sync_158.sh · det_launch.py · effective_args.py · orig_babel_clean_gadget.py
  meta/      runs_resolved.json · runs_effective.json (sinh tự động) · artifact/ (trang theo dõi)
  state/     lock + log của driver (không commit)
  logs/<run>/fold<F>.log · results/<run>/fold<F>.json (+ .probs.npz)   — MỘT bản mỗi run, chỉ trên 161 (không commit)
```

## Run

| run | loại | Pha 1 | đồ thị | tối ưu Pha 2 |
|---|---|---|---|---|
| `p1_mwg_common` | Pha 1 | — | BABEL đã sửa | — |
| `p1_mwg_origbabel_common` | Pha 1 | — | BABEL gốc | — |
| `p1_mw_common` | Pha 1 | — | không (`--fusion mw`) | — |
| `baseline` | đích | — | không | AdamW, LR hằng, wd 0 |
| `mwg_assemble` | đích | `p1_mwg_common` | BABEL đã sửa | RecAdam + ASAM ρ 0,5 |
| `mwg_assemble_noRAS` | đích | `p1_mwg_common` | BABEL đã sửa | AdamW |
| `mwg_assemble_nofixbracket` | đích | `p1_mwg_origbabel_common` | BABEL gốc | RecAdam + ASAM ρ 0,5 |
| `mw_assemble` | đích | `p1_mw_common` | không | RecAdam + ASAM ρ 0,5 |
| `mwg_mixsrc` | đích, một pha | — | BABEL đã sửa | AdamW, train = nguồn + SVEN |

Chung: seed 42, batch 8, lr 2e-5 (`graph_lr` 2e-5), 5 fold SVEN (`sven_python_folds_nocomment`), chọn checkpoint theo val ROC.
Tham số đầy đủ của từng run (kể cả mặc định, do chính trainer parse): `meta/runs_effective.json`.

## Lệnh

```bash
cd /drive1/cuongtm/ntat/MultiVD/_FinalPaperExperiment
python3 scripts/fpe.py order                                   # thứ tự: Pha 1 trước, rồi các run đích
PY_DRY=1 bash scripts/run.sh 161 "mwg_assemble" "1"             # chạy khô: in lệnh train/test, không chạy
bash scripts/launch.sh 158 "p1_mwg_common"                      # phóng trên 158 (đồng bộ scripts/ sang trước)
bash scripts/launch.sh 158 "mwg_assemble mwg_assemble_noRAS"    # Pha 2 phải chạy CÙNG máy với Pha 1 của nó
bash scripts/sync_158.sh                                        # chuyển fold đã xong của 158 về 161, xoá bản trên 158
python3 scripts/fpe.py dbrows                                   # kết quả -> meta/dbrows/ (để ghi vào artifact)
python3 scripts/pair_delta.py <run> <đối chứng> [...]           # Δ ghép cặp theo fold, 4 chỉ số, +/−/n, min-max, hp khác nhau
bash scripts/mon_fpe_v4.sh <161|158>                            # monitor nền (trạng thái ở state/monitor/)
```

- **Chạy lại xoá bản cũ**: `launch.sh` xoá log/kết quả của đúng run × fold sắp chạy trong `_FinalPaperExperiment`; `run.sh` xoá
  bản cũ ở máy chạy ngay trước mỗi fold.
- **Không bao giờ chạy trùng**: mỗi máy một lock (`state/fpe.lock`); `launch.sh` từ chối nếu lock đang bị giữ; `run.sh` dừng nếu
  thấy tiến trình `det_launch.py` khác; cổng VRAM (`VRAM_MIN`, mặc định 9000 MiB) đặt trước MỖI fold.
- **Tất định**: mọi trainer chạy qua `det_launch.py` — `use_deterministic_algorithms(True)` (gặp kernel không tất định là DỪNG),
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=42`, cudnn deterministic, TF32 tắt. Hai máy (161, 158) cùng Python 3.11.14,
  torch 2.9.1+cu128, transformers 4.57.1, A4000, CodeBERT md5 trùng; chỉ khác driver NVIDIA (535 / 575).
- **BABEL gốc**: `FPE_ORIG_CLEAN_GADGET=1` (đặt sẵn trong `runs.json`) thay `clean_gadget` bằng bản gốc gdufsnlp/BABEL @ac53252.
- Checkpoint nằm ở máy chạy (`model/final_paper/<run>/seed_42/fold<F>/best.pt`); Pha 2 xoá sau khi test (trừ `KEEP_CK=1`),
  Pha 1 luôn giữ.

Mã trainer: `src_final/` = GraphTransferVD `multibabel` @01670ba (`code_snapshot/src`) + `train.py` của `src_mwg_theirs`
(md5 e90630f5, chỉ được import); 17/17 file trùng md5 giữa 161 và 158.
Dữ liệu: `data/final_experiment_data/` (README trong đó), trùng md5 giữa hai máy.
