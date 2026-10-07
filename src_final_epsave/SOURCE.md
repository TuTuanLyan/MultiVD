# src_final_epsave (28/09/2026)

Bản sao src_final, CHỈ khác train_mwg.py: biến môi trường FPE_SAVE_EPOCH=k lưu <checkpoint>/ep<k>.pt cuối epoch k rồi dừng.
Dùng để lấy checkpoint Pha 1 ở một epoch cố định (vd. ep2 như bản có Java) mà không đổi --epochs (đổi --epochs là đổi lịch LR).
Không biến ⇒ hành vi y hệt src_final.
