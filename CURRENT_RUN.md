# CURRENT_RUN — 10/09/2026, 00:20 VN

## ĐANG CHẠY — bổ sung n=15 cho khối `chot` (seed 7 + 1234)

Người dùng 10/09: *"chạy thêm 2 seed nữa cho đủ n=15 cho cả 3 source và setting... chạy thêm
n=10 là được, cái chạy rồi không cần chạy lại"* và *"nhớ chạy đúng khối đấy gồm cả có và không
recadam + asam theo đúng h-param tối ưu cho 2 backbone"*.

| máy | backbone | ρ | seed | cây kết quả | mốc |
|---|---|---|---|---|---|
| **161** A4000 | codebert | **0.1** | 7, 1234 | `results/chot161_codebert` | 70 ô |
| **158** A4000 | t5p | **2.0** | 7, 1234 | `results/chot158_t5p` | 70 ô |

Mỗi máy 70 ô = 2 seed × 5 fold × (1 baseline + 3 nguồn × 2 cấu hình).
**A** = `recadam` + ASAM ở ρ của chính backbone · **B** = `adamw`, `--sam_rho 0`.
Seed 42 (70 ô, FACTS §34) **giữ nguyên, không chạy lại**.

**Pha 1 huấn luyện lại cho từng seed**, không dùng lại bản seed 42. Phát biểu của khối là "lợi
ích đến từ Pha 1 + head"; giữ nguyên một Pha 1 mà chỉ đổi seed Pha 2 thì không phân biệt được
tính chất của **phương pháp** với tính chất của **đúng một checkpoint**. Seed còn quyết định cách
chia train/val của Pha 1 (`train_transfer.py:334`).

Chạy bằng `run/chot15.sh`. Watchdog cron 10 phút: `scripts/watch_chot15.sh`.

### Pha 1: cái nào đã có, cái nào phải chạy (quét bằng NỘI DUNG, 00:55 VN)

| | seed 7 | seed 1234 |
|---|---|---|
| **codebert** (161) | `4cwe` ✓ (train 00:13) — còn `com`, `full` | **chưa có gì**, cần cả 3 |
| **t5p** (158) | `4cwe` ✓ (train 00:12) — còn `com`, `full` | **đủ cả 3, có sẵn 05/09** ✓ |

t5p seed 1234 đã đọc nội dung xác nhận dùng lại được: `num_cwes` 4/10/10 đúng theo nguồn,
`cwe_vocab` `fixed4`/`precomputed`, λ=0.05, `training_args.seed=1234`. Tiết kiệm ~3 giờ cho 158.

> Chênh 192 byte giữa các file `4cwe` ở các seed **không phải** khác kiến trúc — chỉ là độ dài
> chuỗi trong `training_args`. `num_cwes=4` ở cả ba. Đã kiểm vì §29/§30 dạy không tin kích thước.

### CÂN LẠI KHI 161 XONG (việc phải làm, đừng quên)

Ước tính: 161 ~9 giờ (xong ~10:00 VN), 158 ~16,6 giờ (~17:30 VN). Tổng 26 giờ / 2 máy ⇒ nếu cân
thì cả khối xong ~**13:30 VN**.

Khi 161 đủ 70 ô codebert: nhìn xem 158 đã làm tới đâu, lấy **những fold cao nhất của seed 1234
chưa chạy** (khoảng 2 fold ≈ 14 ô ≈ 3 giờ) giao cho 161, và **khởi động lại 158 với danh sách
fold đã trừ đi phần đó** — nếu không hai máy làm trùng, và vì hai cây kết quả khác nhau nên
không bên nào bỏ qua bên nào. Vẫn chia theo **fold trọn vẹn**.

### Luật đêm 10/09 về việc thuê vast

| tình huống | xử lý |
|---|---|
| 158 mất kết nối / tự khởi động lại / reboot | **CHỜ**, hạn **20 phút**. Lên lại thì chạy tiếp. **Không thuê vast.** |
| Bị tranh mất GPU lúc chuyển pha/fold | `wait_vram` tự chờ. Quá **20 phút** không có tiến trình huấn luyện mà VRAM < 13 GB ⇒ coi là **VỠ**. |
| Một máy VỠ | được thuê **đúng MỘT** vast A4000 |
| **Cả hai** máy VỠ | vẫn chỉ **MỘT** vast, không hơn |
| Không máy nào vỡ | **không thuê** |

Vast phải canh kỹ — dùng bỏ phí là bị phạt. Watchdog **không tự thuê**: nó chỉ dựng cờ `VO` trong
`log/chot15_state`, monitor báo về để quyết bằng tay (phải kiểm giá và cài đặt).

Giá A4000 đã tra 00:15 VN: **$0.068–0.092/h**. Không có offer nào gần $0.008 — hiểu ngưỡng người
dùng nêu là **$0.08**. Đã chọn sẵn `id 39530408`, **$0.0756/h**, tin cậy **0.999**, 63 G RAM,
610 G đĩa, 6151 Mb/s — đắt hơn offer rẻ nhất $0.0075/h nhưng tin cậy 0.999 vs 0.965 và mạng nhanh
gấp 7.

### Khi xong

1. Kéo `results/chot158_t5p` từ 158 về, đối chiếu **số file + byte**.
2. Đọc: `python3 tools/chot_report.py results/chot_t5p results/chotv_t5p results/chot_codebert
   results/chot161_codebert results/chot158_t5p`
3. **Cập nhật artifact khối `chot`** (người dùng yêu cầu):
   <https://claude.ai/code/artifact/1ced3c61-bf8a-48b1-ab17-6ad575a0123b>
   Dựng lại dữ liệu bằng `tools/chot_export.py` rồi xuất bản lại **cùng đường dẫn file** để giữ URL.

---

## Đã xong trước đó

- **Khối `chot` seed 42** — 70 ô, FACTS §34/§34.1. Vast `ntat` id 50132360 đã huỷ 15:29 UTC 09/09.
- **§30.2** — đã đo head phụ: gần như không học được gì ở mọi nguồn và cả hai backbone.
