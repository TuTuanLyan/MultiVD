Hiện tại là server 161 các folder được dùng: /home/ntat/workspace toàn quyền.
/drive1/cuongtm/ntat: toàn quyền trừ Archive, ngoài ra không đụng đến. 

Server 158: ssh tranmanhcuong@112.137.129.158 key sử dụng ở ~/ntat/.ssh
các folder được dùng trên 158: /data/ntat/ toàn quyền, ngoài ra không đụng 

Vast: Theo vast rule.

---

## Chuyển chỗ làm việc sang /drive1 (chốt 03/09)

**Từ sau đợt quét λ×ρ đang chạy, nơi làm việc chính của MultiVD là:**

```
/drive1/cuongtm/ntat/MultiVD          <- chạy thí nghiệm mới tại đây
```

**Vì sao:** `/` (chứa `/home/ntat`) đã 98% đầy và từng làm hỏng checkpoint hai lần
vì `torch.save` ghi cụt khi hết chỗ. `/drive1` còn 383 GB.

**Trạng thái:**

| | |
|---|---|
| Bản sao đầu tiên sang drive1 | đã tạo 03/09 (ảnh chụp lúc job còn chạy) |
| Đồng bộ lại lần cuối | **phải làm sau khi đợt quét xong**, trước khi đổi chỗ |
| Kết quả `results/*.json` | nhỏ, để lại ở home cũng được |
| Checkpoint Phase 1 (`model/`) | chuyển sang drive1 |

**Lưu ý khi đang trong phiên:** phiên Claude CLI hiện tại mở ở
`/home/ntat/workspace/MultiVD`. Đừng chuyển thư mục ra khỏi dưới chân một phiên
đang chạy — đóng phiên rồi mới đổi, hoặc mở phiên mới tại đường dẫn drive1.

Nếu drive1 đã có `MultiVD` thì **lấy bản mới nhất ở home ghi đè** — hai bên được
giữ đồng bộ, home là nguồn đúng.

---

## Phân công máy cho đợt xác nhận (04/09)

| máy | nguồn | torch | ghi chú |
|---|---|---|---|
| vast `ntat` id 49840185, cổng 39702 | `common` | 2.11.0 | đang chạy |
| vast `ntat2` id 49840024, cổng 35245 | `4cwe` | 2.11.0 | **chuyển về local khi 161/158 rảnh, rồi huỷ máy này** |
| 161 hoặc 158 | `4cwe` | 2.9.1 | chờ `cuongtm` / `tranmanhcuong` xong |

**Chia theo NGUỒN, không theo fold.** Mỗi nguồn nằm trọn trong một máy nên Δ ghép
cặp của nó không dính chênh lệch phần cứng hay phiên bản. Phép **gộp hai nguồn**
thì có mang chênh lệch torch (2.11.0 với 2.9.1) — chấp nhận ở mức so sánh tương
đối, nhưng **phải ghi rõ khi báo cáo**.

`transformers` được ghim **4.57.1** trên mọi máy; đó là thư viện duy nhất bắt buộc
khớp, theo `requirements-pin.txt`.

Bộ dò trong `scripts/watch47.sh` (cron 10 phút/lần) ghi trạng thái GPU của 161 và
158; khi máy nào còn ≥13 GB trống thì log in dòng `>>> RANH` kèm việc cần làm.


---

## Đã chuyển sang /drive1 — 06/09/2026

`scripts/migrate_to_drive1.sh` chạy xong lúc 06/09. Kết quả đối chiếu của chính script:
**6 694 file cả hai bên, lệch 0 byte**. Kiểm độc lập sau đó: 11/11 file `.md`,
1 896/1 896 kết quả `fold*.json`, 75/75 checkpoint `.pt`, 18/18 memory.

| | đường dẫn | dùng khi nào |
|---|---|---|
| **chính** | `/drive1/cuongtm/ntat/MultiVD` | mọi thí nghiệm mới, mọi thứ nặng |
| tra cứu | `/home/ntat/workspace/MultiVD` | bản cũ, không chạy gì mới ở đây |
| 158 | `/data/ntat/MultiVD` | bản riêng của 158, 708 GB trống |

Đĩa sau khi chuyển: `/` còn 47 GB (98% đầy), `/drive1` còn 346 GB (81%).

**`model/` ở `/home` vẫn còn 33 GB** — bản sao đã có ở `/drive1` và đã đối chiếu, nhưng
chưa xoá. Xoá bằng:

```bash
CONFIRM_DELETE=1 bash scripts/migrate_to_drive1.sh
```

Nó đối chiếu lại số file `.pt` ở `/drive1` trước khi xoá, và chỉ xoá `.pt` — mã nguồn,
`data/` và `results/` giữ nguyên ở cả hai nơi.

**Ba chỗ trong script đã sửa cùng ngày**, vì bản viết đầu tháng đã lệch thực tế:
cổng canh gác kiểm lock của đợt `sweep46`/`day45` đã hết từ lâu (giờ kiểm lock
`night48`), phép đếm tiến trình bỏ sót `train_baseline`, và `results/` trước bị loại
khỏi rsync — bỏ lại thì khối NIGHT48 bị cắt đôi vì seed 7 fold 1–2 nằm trong
`results/n48_t5p`.

**Cron giám sát đã bị script gỡ** (bản lưu ở `/tmp/mvd_crontab.bak`). Hiện không có job
nào chạy nên chưa bật lại; khi bật phải sửa đường dẫn trong đó sang `/drive1`.
