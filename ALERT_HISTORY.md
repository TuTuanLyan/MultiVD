# Lịch sử cảnh báo đã xem - khối mwonly5

Cảnh báo sập / nghi sập / kẹt / lỗi mà người dùng đã bấm "Đã xem" trên artifact https://claude.ai/artifact/Pouzqz7Q1Kd9vwwrnQDm51 . Cảnh báo chưa xem nằm ở dải đỏ đầu trang và mục "⚠ CẢNH BÁO" của CURRENT_RUN.md.
Ghi bằng `_FinalPaperExperiment/scripts/alert_history.py`, cũ trước mới sau.

## 2026-10-05 01:40 · nghi sập · rasam_common_jscpp f2 · BÁO NHẦM
<!-- alert:a202610050140_rasam_common_jscpp_f2 -->

- máy: 158
- đã xem: 2026-10-05 01:53
- kết cục: báo nhầm

RecAdam + ASAM, common JS + C/C++, fold 2: train loss đứng ở bình nguyên ln2 (0,69-0,73) suốt ep2-12, val ROC 0,46-0,54; ep13 mới nhích (train loss 0,640, val ROC 0,608). Cùng kiểu sập của RA + ASAM ở tab cũ. Fold 1 cùng nguồn thoát ở ep6-7 (ROC 0,893). Job vẫn chạy tiếp (min_epochs 10, patience 8).

- 2026-10-05 01:4x: ep14 best val ROC 0,691 (từ 0,608 ở ep13) - đang thoát muộn
- 2026-10-05 01:49: KHÔNG sập: thoát muộn rồi học tới ep20 (dừng sớm ep28), test ROC 0,911 / PR 0,925 / F1@0.5 0,802 / F1@val 0,799 - ngang baseline cùng fold (ROC +0,001, F1@0.5 -0,014). Thời gian đứng ở ln2 vẫn làm mất lợi thế so với fold 1 (+0,033 ROC so với baseline).

## 2026-10-05 02:40 · nghi sập · rasam_common_jscpp f5 · BÁO NHẦM
<!-- alert:a202610050240_rasam_common_jscpp_f5 -->

- máy: 158
- đã xem: 2026-10-05 02:49
- kết cục: báo nhầm

RecAdam + ASAM, common JS + C/C++, fold 5: train loss ở bình nguyên ln2 (0,68-0,75) ep2-11, val ROC 0,52-0,55; ep12-13 train loss mới xuống 0,647 / 0,635 nhưng val ROC vẫn 0,56. Cùng kiểu fold 2 cùng nguồn (báo nhầm: thoát muộn ở ep13, test ROC 0,911). Job vẫn chạy tiếp.

- 2026-10-05 02:48: KHÔNG sập: thoát muộn rồi học tới ep24, test ROC 0,908 / PR 0,923 / F1@0.5 0,816 / F1@val 0,834 - hơn baseline cùng fold cả 4 chỉ số (ROC +0,028). Nguồn common JS + C/C++ xong 5/5: ROC TB 0,916.

## 2026-10-05 02:45 · nghi sập · nop1_asamonly f3 · BÁO NHẦM
<!-- alert:a202610050245_nop1_asamonly_f3 -->

- máy: paper_mw
- đã xem: 2026-10-05 03:00
- kết cục: báo nhầm

MW không Pha 1, chỉ ASAM, fold 3: train loss đứng ở ln2 (0,70-0,73) suốt 13 epoch, val ROC 0,37-0,45 (dưới 0,5), checkpoint chọn vẫn là ep2 (val ROC 0,455). Với min_epochs 10, patience 8 thì ô sẽ dừng khoảng ep17-18 nếu không thoát - nhiều khả năng sập thật. Khác f1 cùng cấu hình (thoát ở ep9-10). Job vẫn chạy tiếp.

- 2026-10-05 02:53: KHÔNG sập nhưng thoát RẤT MUỘN: đứng ở ln2 tới ep20, ep21-25 mới học (val ROC 0,78 -> 0,93), chọn ep27, chạy đủ 30 epoch. Test ROC 0,915 / PR 0,914 / F1@0.5 0,803 - hơn baseline cùng fold (ROC +0,025) và hơn không Pha 1 AdamW (+0,037). Sống sót nhờ val ROC nhích nhẹ ở ep15, ep18 làm patience đặt lại; nếu không sẽ dừng ở ep17 với checkpoint ep2.

## 2026-10-05 03:13 · nghi sập · nop1_asamonly f4 · SẬP THẬT
<!-- alert:a202610050313_nop1_asamonly_f4 -->

- máy: paper_mw
- đã xem: 2026-10-05 14:01
- kết cục: sập thật

MW không Pha 1, chỉ ASAM, fold 4: train loss ở ln2 (0,70-0,74) suốt 13 epoch, val ROC 0,48-0,55 (best 0,546 ở ep8). Cùng kiểu f3 cùng cấu hình (thoát rất muộn ở ep21, test ROC 0,915). Job vẫn chạy tiếp.

- 2026-10-05 03:15: ep13: luật train loss cũng báo (min 0,697, chưa xuống dưới 0,6). Cùng một ô, không phải cảnh báo mới. Job chạy tiếp; trần 30 epoch, patience 8.
- 2026-10-05 03:16: SẬP THẬT: dừng sớm ở ep17, train loss không rời ln2 (0,70-0,74) suốt 17 epoch, best val ROC 0,546 ở ep8. Test ROC 0,464, F1@0.5 0,374, PR-AUC 0,409. Khác f3 (thoát ở ep21): f4 hết patience trước khi kịp thoát.

## 2026-10-05 03:35 · nghi sập · nop1_asamonly f5 · BÁO NHẦM
<!-- alert:a202610050335_nop1_asamonly_f5 -->

- máy: paper_mw
- đã xem: 2026-10-05 13:36
- kết cục: báo nhầm

MW không Pha 1, chỉ ASAM, fold 5: train loss ở ln2 (0,69-0,75) suốt 13 epoch (min 0,691). Khác f4 (sập): val ROC vẫn nhích dần 0,50 → 0,73 ở ep12, nên patience liên tục đặt lại. Job vẫn chạy tiếp.

- 2026-10-05 03:36: BÁO NHẦM - thoát ở ep14: train loss 0,637, val ROC 0,746 (best mới). Thoát muộn giống f3; chờ kết quả test.
- 2026-10-05 03:50: Xong: chạy đủ 30 epoch, chọn ep26, test ROC 0,875 (nop1_adamw cùng fold 0,909).

## 2026-10-05 04:58 · nghi sập · rasam_full_jscpp f5 · BÁO NHẦM
<!-- alert:a202610050458_rasam_full_jscpp_f5 -->

- máy: 161
- đã xem: 2026-10-05 13:36
- kết cục: báo nhầm

CỘT CHÍNH (RecAdam + ASAM), nguồn full JS + C/C++, fold 5: train loss ở ln2 (0,69-0,78) suốt 12 epoch, val ROC 0,50-0,56, best 0,558 ở ep5, patience 3/8 ⇒ dừng ~ep17 nếu không thoát. f1-f4 cùng nguồn không sập (0,899 / 0,921 / 0,916 / 0,946). Job vẫn chạy tiếp.

- 2026-10-05 04:59: ep13: luật train loss cũng báo (min 0,691), val ROC 0,542, patience 4/8. Cùng một ô, không phải cảnh báo mới.
- 2026-10-05 05:11: BÁO NHẦM - thoát ở ep14-16 khi patience mới 4/8: val ROC 0,61 → 0,74 → 0,82, train loss 0,597 ở ep16. Best 0,908 ở ep24, đang ở ep29/30; chờ kết quả test.
- 2026-10-05 05:13: Xong: đủ 30 epoch, chọn ep24, test ROC 0,886 (thấp nhất trong 5 fold của nguồn này: 0,899 / 0,921 / 0,916 / 0,946).

## 2026-10-05 05:59 · nghi sập · rasam_common_ccpp f1 · BÁO NHẦM
<!-- alert:a202610050559_rasam_common_ccpp_f1 -->

- máy: paper_mw2
- đã xem: 2026-10-05 13:36
- kết cục: báo nhầm

Cột chính (RecAdam + ASAM), nguồn common chỉ C/C++, fold 1: train loss ở ln2 (0,70-0,74) ep2-11, val ROC 0,45-0,50 ep1-9; ep10-12 val ROC tăng 0,55 → 0,61 → 0,65, val loss giảm, patience 0/8 - đang có dấu hiệu thoát. Job vẫn chạy tiếp.

- 2026-10-05 06:00: ep13: luật train loss cũng báo (min 0,677 - đang giảm dần). Cùng một ô, không phải cảnh báo mới.
- 2026-10-05 06:09: BÁO NHẦM - thoát từ ep10, chọn ep16 (val ROC 0,938), test ROC 0,914, F1@0.5 0,838.

## 2026-10-05 06:59 · nghi sập · rasam_common_ccpp f4 · BÁO NHẦM
<!-- alert:a202610050659_rasam_common_ccpp_f4 -->

- máy: paper_mw2
- đã xem: 2026-10-05 13:36
- kết cục: báo nhầm

Cột chính (RecAdam + ASAM), nguồn common chỉ C/C++, fold 4: train loss ở ln2 (0,70-0,74) suốt 12 epoch, val ROC 0,43-0,53; ep11-12 nhích 0,485 → 0,527 nên patience đặt lại 0/8. Giống f1 cùng nguồn (thoát từ ep10, test 0,914). Job vẫn chạy tiếp.

- 2026-10-05 07:00: ep13: luật train loss cũng báo (min 0,689). Cùng một ô, không phải cảnh báo mới.
- 2026-10-05 07:14: BÁO NHẦM - val ROC nhích dần ep13-16 (0,54 → 0,69) rồi thoát ep17-19 (0,74 → 0,84 → 0,90; train loss 0,65 → 0,39). Chọn ep22 (val 0,921), test ROC 0,946, F1@0.5 0,862.

## 2026-10-05 08:50 · nghi sập · asamonly_common_jscpp f1 · BÁO NHẦM
<!-- alert:a202610050850_asamonly_common_jscpp_f1 -->

- máy: 158
- đã xem: 2026-10-05 13:36
- kết cục: báo nhầm

Chỉ ASAM, nguồn common JS + C/C++, fold 1: train loss ở ln2 (0,70-0,73) suốt 13 epoch, best val ROC 0,580 ở ep2, patience 4/8 ⇒ dừng ep17 nếu val không vượt 0,580 (đang nhích 0,52 → 0,56). Dự đoán T1 đã ghi trước: chỉ ASAM sẽ có ô đứng ở ln2 > 10 epoch. Job vẫn chạy tiếp.

- 2026-10-05 08:59: BÁO NHẦM - đã thoát: ep23 train loss 0,157, val ROC 0,932 (best mới). Chờ kết quả test.
- 2026-10-05 09:05: Xong: test ROC 0,903 (cột chính cùng fold 0,893).

## 2026-10-05 08:53 · nghi sập · asamonly_full_jscpp f4 · BÁO NHẦM
<!-- alert:a202610050853_asamonly_full_jscpp_f4 -->

- máy: 161
- đã xem: 2026-10-05 13:36
- kết cục: báo nhầm

Chỉ ASAM, nguồn full JS + C/C++, fold 4: train loss ở ln2 (0,70-0,73) suốt 12 epoch, val ROC 0,48-0,54; ep12 đạt 0,543 nên patience đặt lại 0/8. Kiểu dự đoán T1. Job vẫn chạy tiếp.

- 2026-10-05 08:54: ep13: luật train loss cũng báo (min 0,690). Cùng một ô, không phải cảnh báo mới.
- 2026-10-05 08:59: BÁO NHẦM - đã thoát: ep19 train loss 0,328, val ROC 0,889 (best mới). Chờ kết quả test.
- 2026-10-05 09:08: Xong: test ROC 0,942 (cột chính cùng fold 0,946).

## 2026-10-05 09:22 · nghi sập · asamonly_4cwe_jscpp f1 · SẬP THẬT
<!-- alert:a202610050922_asamonly_4cwe_jscpp_f1 -->

- máy: paper_mw
- đã xem: 2026-10-05 14:01
- kết cục: sập thật

Chỉ ASAM, nguồn 4cwe JS + C/C++, fold 1: train loss ở ln2 (0,71-0,74) suốt 12 epoch, best val ROC 0,626 chỉ ở ep1 rồi tụt về 0,46-0,48 (không nhích), patience 3/8 ⇒ dừng ep17 với checkpoint ep1 nếu không thoát. Khác hai ô chỉ ASAM lúc 08:5x (val vẫn nhích). Job vẫn chạy tiếp.

- 2026-10-05 09:23: ep13: luật train loss cũng báo (min 0,706). Cùng một ô, không phải cảnh báo mới.
- 2026-10-05 09:26: SẬP THẬT: dừng sớm, giữ checkpoint ep1 (val 0,626). Test ROC 0,607, F1@0.5 0,438, PR-AUC 0,639 (cột chính cùng fold 0,918). Không chạy lại khi chưa có quyết định của người dùng.

## 2026-10-05 09:31 · nghi sập · asamonly_full_jscpp f5 · BÁO NHẦM
<!-- alert:a202610050931_asamonly_full_jscpp_f5 -->

- máy: 161
- đã xem: 2026-10-05 13:36
- kết cục: báo nhầm

Chỉ ASAM, nguồn full JS + C/C++, fold 5: train loss ở ln2 (0,68-0,75) suốt 13 epoch (min 0,685). Nhưng val ROC tăng liên tục ep10-13 (0,63 → 0,67 → 0,68 → 0,77), val loss giảm, patience 0/8. Job vẫn chạy tiếp.

- 2026-10-05 09:31: BÁO NHẦM (đang thoát) - lúc cảnh báo val ROC đã lên 0,773 ở ep13. Chờ kết quả test.
- 2026-10-05 09:46: Xong: đủ 30 epoch, chọn ep27, test ROC 0,852 (cột chính cùng fold 0,886) - không sập nhưng thấp.

## 2026-10-05 09:57 · nghi sập · asamonly_common_jscpp f3 · SẬP THẬT
<!-- alert:a202610050957_asamonly_common_jscpp_f3 -->

- máy: 158
- đã xem: 2026-10-05 14:01
- kết cục: sập thật

Chỉ ASAM, nguồn common JS + C/C++, fold 3: train loss ở ln2 (0,69-0,73) suốt 12 epoch, best val ROC 0,521 ở ep9 rồi tụt về 0,40-0,43, patience 3/8 ⇒ dừng ep17 nếu val không vượt 0,521. Giống asamonly_4cwe_jscpp f1 (sập thật 09:26). Job vẫn chạy tiếp.

- 2026-10-05 09:59: ep13: luật train loss cũng báo (min 0,694). Cùng một ô, không phải cảnh báo mới.
- 2026-10-05 10:02: SẬP THẬT: dừng sớm ep17, giữ checkpoint ep9 (val 0,521). Test ROC 0,523, F1@0.5 0,444, PR-AUC 0,509 (cột chính cùng fold 0,916). Ô sập thứ hai của chỉ ASAM trên JS + C/C++ ⇒ dự đoán T1 (≤ 1/15 sập) SAI. Không chạy lại khi chưa có quyết định của người dùng.

## 2026-10-05 10:10 · nghi sập · asamonly_common_jscpp f4 · BÁO NHẦM
<!-- alert:a202610051010_asamonly_common_jscpp_f4 -->

- máy: 161
- đã xem: 2026-10-05 13:36
- kết cục: báo nhầm

Chỉ ASAM, nguồn common JS + C/C++, fold 4: train loss ở ln2 (0,69-0,74) suốt 12 epoch, val ROC đi ngang 0,50-0,52, best 0,526 ở ep2, patience 3/8 ⇒ dừng ep17 nếu val không vượt 0,526. Giống hai ô chỉ ASAM vừa sập thật (4cwe f1, common f3). Job vẫn chạy tiếp.

- 2026-10-05 10:12: ep14: luật train loss cũng báo (min 0,684). Cùng một ô, không phải cảnh báo mới.
- 2026-10-05 10:18: BÁO NHẦM - val ROC nhích ep13-14 (0,533 / 0,539 > best 0,526) đặt lại patience, rồi thoát ep16 (0,71 → 0,75 → 0,81 → 0,83 → 0,86 ở ep22). Sống sót nhờ hai lần nhích nhỏ đó. Chờ kết quả test.
- 2026-10-05 10:25: Xong: chọn ep24, test ROC 0,913 (cột chính cùng fold 0,955).

## 2026-10-05 10:43 · nghi sập · asamonly_common_jscpp f5 · BÁO NHẦM
<!-- alert:a202610051043_asamonly_common_jscpp_f5 -->

- máy: 161
- đã xem: 2026-10-05 13:36
- kết cục: báo nhầm

Chỉ ASAM, nguồn common JS + C/C++, fold 5: train loss ở ln2 (0,69-0,73) suốt 12 epoch, nhưng val ROC nhích dần 0,53 → 0,58 (best 0,583 ở ep11), patience 1/8 - khác hai ô sập thật (val tụt). Job vẫn chạy tiếp.

- 2026-10-05 10:44: ep13: luật train loss cũng báo (min 0,694). Cùng một ô, không phải cảnh báo mới.
- 2026-10-05 10:51: BÁO NHẦM - đã thoát: ep20 train loss 0,322, val ROC 0,850 (best mới). Chờ kết quả test.
- 2026-10-05 10:59: Xong: test ROC 0,893 (cột chính cùng fold 0,908).

## 2026-10-05 17:38 · nghi kẹt · rasam_sven_python_tojsfull f1 · BÁO NHẦM
<!-- alert:a202610051738_rasam_sven_python_tojsfull_f1 -->

- máy: paper_mw3
- đã xem: 2026-10-05 18:10
- kết cục: báo nhầm

SVEN Python → đích JS full, RecAdam + ASAM, fold 1: train loss ở ln2 (0,724 → 0,693) suốt 8 epoch, val ROC tụt 0,500 (ep1) → 0,458 (ep8), best vẫn ep1, patience 7/8 ⇒ dừng ở ep10 (min_epochs) với checkpoint ep1 nếu ep9-10 không vượt 0,500. Dấu hiệu giống các ô sập vì bình nguyên ASAM trên đích SVEN. Ngưỡng 0,75 không áp cho đích JS, cảnh báo này dựa vào train loss. Job vẫn chạy tiếp.

- 2026-10-05 17:45: SỬA LẠI: patience chỉ đếm từ ep10 (min_epochs; log ep8-9 ghi 'patience 0/8', ep10 1/8), nên KHÔNG dừng ở ep10 như đã ghi - sớm nhất ep18. Train loss đã rời ln2: ep9 0,671, ep10 0,680, ep11 0,650; val ROC nhích 0,451 → 0,473 → 0,489 (best vẫn 0,500 ở ep1). Có thể đang thoát bình nguyên; theo dõi tiếp.
- 2026-10-05 17:47: ep12: best val ROC lên 0,533 (vượt 0,500 của ep1, patience về 0). Luật 'học chậm' của monitor (best < 0,7 ở ep12) cũng báo, nhưng ngưỡng 0,7 hiệu chỉnh trên đích SVEN; JS full trong miền chỉ đạt 0,59 nên ngưỡng đó không áp được. Cùng một ô, không phải cảnh báo mới.
- 2026-10-05 18:02: KẾT CỤC: BÁO NHẦM - đã thoát bình nguyên (train loss 0,529 ở ep16), dừng sớm ep20, checkpoint ep12 (val 0,533). Test ROC 0,583, PR 0,617, F1@0.5 0,570 - ngang Pha 1 JS full trong miền (val 0,59) nên là mức của đích JS, không phải sập. Nằm trong dự đoán J2 [0,50; 0,68].
