- Nếu lần đầu push có thể viết file .sh ngay tại root project (tham khảo push2vast.sh và chỉnh sửa nếu cần)
- Chỉ dùng đúng label của instance mặc định nếu không yêu cầu gì thì đặt tên ntat cho label. Nếu được yêu cầu sẽ đặt tên khác
- Không đặt label khác, tải log về máy hiện tại để đọc lại trước khi hủy, chỉ tải model nếu tôi yêu cầu ở runname đó. 
- Rule sẽ tuân theo như trong yêu cầu sẽ được ưu tiên hơn khi chat. 
- Không kill hay đụng vào chỗ khác như label hoặc người dùng khác cùng team trên vast.
- Tại local nếu yêu cầu sửa code và run smoke test trước khi lên vast cũng tuyệt đối nghiêm cấm kill process khác ngoài user ntat trên máy, nếu kill của tôi cũng nên hỏi lại nneeus nó là tiến trình quan trọng mà không thuộc kiếm soát của ML4VD.
- Ưu tiên 5060ti và của Việt Nam - Châu Á. Hoặc 4080S (16GB)
- Không thuê GPU có giá lớn hơn 0.2 đô/h.
- Khi tìm thấy GPU hãy hỏi lại dù bất kì permission như nào để tôi quyết định thuê không về giá cả và thông tin về máy.
- Thay đổi code trên vast là được phép để tăng khả năng tận dụng GPU nếu được yêu cầu như batch size hay đổi tên model backbone, lr, runname, ... trong config.
- Tái tạo lại môi trường conda bằng conda create -f enenvironment.yml -n vdenv hoặc nếu trên vast dùng conda rồi hãy xử lý tải thêm thay vì tốn bộ nhớ cho conda env nữa.
- Luôn cài monitor nền khi có job chạy trên vast: theo dõi driver/GPU/tiến độ, báo khi xong fold hoặc khi hỏng, và kéo kết quả về + đối chiếu từng byte trước khi huỷ máy.


## Huỷ máy — DESTROY, không bao giờ STOP (người dùng nêu 08/09/2026)

## HẾT VIỆC + NGƯỜI DÙNG KHÔNG CÓ MẶT ⇒ HUỶ NGAY, KHÔNG CHỜ HỎI

Người dùng nêu **15/09/2026**, sau khi tôi để một 5060 Ti nằm không **6 giờ** ($0,49) trong đêm.

"**Chú ý vast**" nghĩa là **QUẢN LÝ**, không phải **quan sát**. Khi hàng đợi cạn và người dùng
đang ngủ hoặc vắng mặt:

1. Liệt kê **mọi** hiện vật trên máy — ô kết quả **và** checkpoint trung gian.
2. Đối chiếu **tên + byte** với local; số mục thiếu phải bằng **0**.
   *Cẩn thận*: khối kéo về dưới **tên thư mục khác** (`results/fus5060_codebert` →
   `results_fus5060_ntat`) sẽ bị báo "thiếu" giả nếu so theo đường dẫn thô.
3. `vastai destroy instance <id> -y`, **kiểm output** (có thể in `Aborted.` rồi thoát 0),
   rồi xác nhận instance biến mất khỏi `show instances`.
4. Báo lại sau. **Không hỏi trước.**

**Vì sao không chờ**: hoãn quyết định khi người dùng đang ngủ **không tạo ra thông tin**, chỉ tạo
ra lãng phí. Huỷ máy sau khi đã kéo hết dữ liệu thì **đảo ngược được** (thuê lại + script dựng
môi trường); tiền máy chạy không thì **không**.

**Ngoại lệ cho mục 9 CLAUDE.md — người dùng nêu 15/09:** khi chạy đêm mà **vast đang có sẵn**,
được **tự lấy một hướng MỚI ra chạy** để lấp chỗ trống, thay vì để máy nằm không rồi chờ hỏi.
Điều kiện: hướng đó phải đã có trong `NEXT_CONTRIBUTION.md` (đã bàn, chưa bị bác), **ghi dự
đoán trước khi đo**, và báo rõ trong lần trả lời kế tiếp. Không áp dụng cho việc **thuê máy
mới** — thuê thì vẫn phải hỏi.

> Cảnh báo "GPU nằm không" của monitor là **lệnh hành động**, không phải dòng ghi chú.

---

`stop` **không** an toàn: máy đã stop vẫn có thể bị thuê lại hoặc dính schedule, và người dùng
bị phạt vì để vast chạy không. Khi hết việc thì **huỷ hẳn**:

```
vastai destroy instance <id> -y
vastai show instances          # PHẢI xác nhận instance đã biến mất
```

`vastai` có khi in `Aborted.` mà **vẫn thoát 0**, nên đọc output chứ đừng tin mã thoát.

**Trình tự bắt buộc trước khi huỷ:**
1. Kéo hết kết quả về và **đối chiếu số file lẫn kích thước byte**.
2. Kéo log về để đọc lại sau.
3. Chỉ khi cả hai xong mới huỷ. **Mất log ⇒ không huỷ. Chưa đối chiếu ⇒ không huỷ.**

**Khi nào thì hết việc đáng chạy:** khối đáng tiền là khối trả lời được một câu **chưa biết**.
Chạy lại thứ đã rõ, hoặc thêm fold cho một hiệu ứng đã đo là null ở n đủ lớn, thì không đáng —
lúc đó chuyển phần còn lại về 161/158 và huỷ vast.

**Chỉ đụng nhãn `ntat` và `ntat2`.** Ngày 08/09 tài khoản còn có `dung` và `cuongtm4070s` của
người khác trong cùng team — không bao giờ đụng tới.

## Máy local ĐƯỢC PHÉP nằm không — chỉ vast mới cấm (người dùng nêu 09/09/2026)

> *"local thì để trống chờ kết quả rồi quyết định hướng chạy tiếp là được phép nhé. chỉ vast
> là đừng bỏ phí quá lâu."*

Nên **đừng bịa việc cho 161/158 chỉ để lấp chỗ**. Máy local không tốn tiền, và ép nó chạy khi
GPU đang chật thì tệ hơn để trống: ngày 08/09 mất **26 ô** vì đúng chuyện đó. Để trống chờ kết
quả rồi quyết hướng là lựa chọn hợp lệ và thường là đúng.

Vast thì ngược lại: mỗi giờ nằm không là tiền thật và người dùng bị phạt. Hết việc **đáng chạy**
thì huỷ ngay chứ không giữ lại "phòng khi cần".

## Máy local dùng chung — ai vào trước được trước

161 và 158 dùng chung với người khác (`cuongtm`, `ollama`). **Không bao giờ giết tiến trình của
họ**, kể cả khi nó chiếm GPU và làm job của mình OOM. Cách đúng là **chờ**: cổng VRAM trước từng
ô, và thử lại khi OOM. Ngày 08/09 mất 26 ô vì cổng chỉ kiểm một lần lúc khởi động.

