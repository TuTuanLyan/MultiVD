# src_final_eptest (01/10/2026)

Bản sao src_final, CHỈ khác train_mwg.py: biến môi trường FPE_EPOCH_TEST=1 chấm thêm tập test ở cuối MỖI epoch huấn luyện và lưu
xác suất vào <thư mục checkpoint>/ep_test/epNN.npz (kèm nhãn, n_tokens, xác suất val của epoch đó). Mọi trạng thái RNG (random, numpy,
torch CPU, torch CUDA) được lưu rồi khôi phục quanh phần chấm thêm, nên quỹ đạo huấn luyện phải trùng bit với run gốc (kiểm bằng val ROC
từng epoch và test ROC ở epoch được chọn). Dùng cho phép "kiểm checkpoint sớm" (người dùng 01/10). Không có biến ⇒ hành vi y hệt src_final.
