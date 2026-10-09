#!/usr/bin/env bash
# Bản v8 (05/10 16:5x): như v7 nhưng KHÔNG gắn '⚠ SẬP' (test ROC < 0,75) cho run đích KHÁC SVEN (tên chứa "_tojs"): ngưỡng 0,75 đặt
# cho đích SVEN; JS full trong miền chỉ đạt val ROC 0,59 (p1_full_jsonly) nên mọi fold JS sẽ bị báo nhầm. Lỗi rc / thiếu ROC vẫn báo.
# Monitor khối final, bản v7 (04/10 22:5x): như v6 nhưng KHÔNG in từng dòng epoch (người dùng: cập nhật artifact 30 phút/lần);
# chỉ in bắt đầu / xong của ô, dừng sớm / train xong, cảnh báo, lỗi. Kết quả đẩy theo nhịp 30 phút (scripts/artifact_tick.sh).
# (v6:) như v5 nhưng BỎ nhãn '⚠ Pha 1 YẾU' - test Pha 1 là bản sao val chia ngẫu nhiên
# trên nguồn cặp nên ROC ~0,5-0,6 là bình thường (tab cũ 0,54-0,57, Pha 2 vẫn 0,94); sức khoẻ Pha 1 do luật kẹt train loss (fpe.py stall) lo.
# (v5:) như v4 + tự gắn nhãn khi một ô KẾT THÚC: '⚠ SẬP' (Pha 2/baseline test ROC < 0,75),
# '⚠ Pha 1 YẾU' (Pha 1 test ROC < 0,60), '⚠ LỖI' (rc ≠ 0 hoặc thiếu test_roc_auc). Người dùng 04/10: sập thì ping, không dừng chờ.
#   bash mon_fpe_v5.sh <host>
# (v4:) Monitor khối final, bản v4 dùng chung cho 161 / vast / 158
# Khác v3: dòng "bắt đầu / xong rc=" đọc từ state/driver_<host>.log — MỌI driver (kể cả driver do queue_after.sh exec) đều tee vào
# đó, nên fold xong của hàng đợi không còn bị lọt (v3 chỉ lọc 'phóng|!!' trong queue_<host>.out); dòng "XONG / !! / phóng / chờ VRAM"
# đọc từ MỌI file state/{driver,queue}_<host>*.out, mỗi file một bộ đếm riêng. Còn lại như v3: dòng quan trọng của log fold đang chạy,
# cảnh báo stall (fpe.py stall), GPU nhàn khi trainer còn sống, không đọc được trạng thái 5 lần liền; tự dừng khi không còn driver lẫn
# hàng đợi. Lần chạy ĐẦU (chưa có file trạng thái) chỉ ghi nhận số dòng hiện có, không in lại dòng cũ.
H=$1
SP=/drive1/cuongtm/ntat/MultiVD/_FinalPaperExperiment/state/monitor          # 30/09: chuyển khỏi scratchpad phiên cũ (scratchpad không còn dùng được)
case "$H" in
  161)  OUT=/drive1/cuongtm/ntat/MultiVD/_FinalPaperExperiment
        remote () { timeout 45 bash -c "$1"; } ;;
  vast) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 64226 root@162.200.81.25 "$1"; } ;;
  vast2) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging          # vast final_paper2 (53148947)
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 63971 root@162.200.81.25 "$1"; } ;;
  paper) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging          # vast nhãn paper (53831025, RTX 4070 SUPER), người dùng thuê 02/10
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 17691 root@137.175.22.196 "$1"; } ;;
  paper_mw) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging       # vast nhãn paper_mw (54059221, RTX A4000), người dùng thuê 04/10 cho tab MW-only
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 39308 root@202.122.49.242 "$1"; } ;;
  paper_mw2) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging      # vast nhãn paper_mw2 (54113977, RTX A4000), người dùng thuê 04/10 13:2x
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 42861 root@202.122.49.242 "$1"; } ;;
  paper_mw3) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging      # vast nhãn paper_mw3 MỚI (54437777, RTX A4000), người dùng thuê lại 06/10 ~13:0x
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 42920 root@202.122.49.242 "$1"; } ;;
  paper_mw4) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging      # vast nhãn paper_mw4 (54468293, RTX A4000), người dùng thuê 06/10 ~17:4x
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 40995 root@172.97.225.87 "$1"; } ;;
  paper_night) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging    # vast nhãn paper_night (54537908, RTX A4000), người dùng thuê 07/10 ~03:3x cho tab Kết quả paper
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 42446 root@202.122.49.242 "$1"; } ;;
  paper_night2) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging   # vast nhãn paper_night2 (54653423, RTX A4000), người dùng thuê 07/10 21:0x
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 36826 root@202.122.49.242 "$1"; } ;;
  paper3) OUT=/workspace/MultiVD/_FinalPaperExperiment_staging         # vast nhãn paper3 (54827622, RTX A4000), người dùng thuê 08/10 17:3x
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p 40980 root@172.97.225.87 "$1"; } ;;
  158)  OUT=/data/ntat/MultiVD/_FinalPaperExperiment_staging
        remote () { timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR tranmanhcuong@112.137.129.158 "$1"; } ;;
  *) echo "host lạ: $H"; exit 2 ;;
esac
mkdir -p "$SP"
ST=${MON_ST:-$SP/mon_fpe_${H}_v7.state}
PAT='Chọn checkpoint theo|Early stopping|Train xong|Traceback|Error|Killed|out of memory|CUDA error|NaN|nondeterministic|deterministic_algorithms'
PAT2='Early stopping|Train xong|Traceback|Error|Killed|out of memory|CUDA error|NaN|nondeterministic|deterministic_algorithms'
FIRST=0
[ -f "$ST" ] || { echo "0 0 0 0" > "$ST"; : > "$ST.files"; FIRST=1; }
read NDL NOUT IDLE FAIL < "$ST"
while true; do
  R=$(remote "
    cd $OUT || exit 1
    D=\$(ps -eo pid,args --no-headers | awk '\$2==\"bash\" && \$3 ~ /scripts\\/run\\.sh\$/ && \$4==\"$H\" {print \$1}' | head -1)
    echo \"DRV=\${D:-none}\"
    Q=\$(ps -eo pid,args --no-headers | awk '\$2==\"bash\" && \$3 ~ /scripts\\/queue_after\\.sh\$/ && \$4==\"$H\" {print \$1}' | head -1)
    echo \"QUE=\${Q:-none}\"
    echo \"TRN=\$(ps -eo args --no-headers | awk '\$2 ~ /det_launch\\.py\$/' | wc -l)\"
    echo \"GPU=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')\"
    CUR=\$(awk '/ bắt đầu\$/ {r=\$3; f=\$4} END {print r\"/\"substr(f,2)}' state/driver_$H.log 2>/dev/null)
    echo \"CUR=\$CUR\"
    L=logs/\${CUR%/*}/fold\${CUR#*/}.log
    echo \"NL=\$(wc -l < \$L 2>/dev/null || echo 0)\"
    echo \"NDL=\$(wc -l < state/driver_$H.log 2>/dev/null || echo 0)\"
    echo '---LOG'; cat \$L 2>/dev/null
    echo '---DLOG'; cat state/driver_$H.log 2>/dev/null
    for f in state/driver_$H*.out state/queue_$H*.out; do [ -f \"\$f\" ] || continue; echo \"@@FILE \${f#state/}\"; cat \"\$f\"; done
    echo '---END'
  " 2>/dev/null)
  if [ -z "$R" ] || ! grep -q '^---END$' <<<"$R"; then
    FAIL=$((FAIL+1)); [ "$FAIL" -eq 5 ] && echo "[mon $H $(date +%H:%M)] CẢNH BÁO: 5 lần liền không đọc được trạng thái $H"
    echo "$NDL $NOUT $IDLE $FAIL" > "$ST"; sleep 60; continue
  fi
  FAIL=0
  get () { sed -n "s/^$1=//p" <<<"$R" | head -1; }
  DRV=$(get DRV); QUE=$(get QUE); TRN=$(get TRN); GPU=$(get GPU); CUR=$(get CUR); NL=$(get NL); NDLN=$(get NDL)
  KEY=$(cat "$ST.cur" 2>/dev/null); [ "$KEY" != "$CUR" ] && { NOUT=0; echo "$CUR" > "$ST.cur"; }   # fold mới: đọc log từ đầu
  [ "${NL:-0}" -lt "$NOUT" ] && NOUT=0                          # log bị tạo lại (chạy lại cùng run/fold)
  [ "${NDLN:-0}" -lt "$NDL" ] && NDL=0                          # driver_<host>.log bị xoá/tạo lại
  case "$CUR" in p1_*|mwg_mixsrc/*) P="$PAT" ;; *) P="$PAT2" ;; esac  # Pha 2/baseline: chỉ epoch 1, dừng sớm, lỗi, kết quả test
  if [ "$FIRST" = 1 ]; then NOUT=${NL:-0}; NDL=${NDLN:-0}; fi
  awk '/^---LOG$/{m=1;next} /^---DLOG$/{m=0} m' <<<"$R" | tail -n +$((NOUT+1)) | grep -E "$P" | grep -v '^# ' | sed "s|^|[$H $CUR] |" | cut -c1-260
  awk '/^---DLOG$/{m=1;next} /^@@FILE |^---END$/{m=0} m' <<<"$R" | tail -n +$((NDL+1)) | grep -E 'bắt đầu|xong rc=' \
    | awk -v H="$H" '{ line=$0; sub(/^ */, "", line); print "[" H " driver] " substr(line, 1, 200)
        if (line ~ /xong rc=/) { run=$3; f=$4; match(line, /xong rc=[0-9]+/); rc=substr(line, RSTART+8, RLENGTH-8); roc=""
          if (match(line, /"test_roc_auc": [0-9.]+/)) roc=substr(line, RSTART+16, RLENGTH-16)
          if (rc != 0) print "[" H " driver] ⚠ LỖI " run " " f " rc=" rc
          else if (roc == "") print "[" H " driver] ⚠ LỖI " run " " f " xong nhưng không có test_roc_auc"
          else if (run !~ /^p1_/ && run !~ /_tojs/ && roc+0 < 0.75) print "[" H " driver] ⚠ SẬP " run " " f " test ROC " roc }
        fflush() }'
  # từng file .out: bộ đếm riêng trong $ST.files ("tên số_dòng")
  awk '/^@@FILE /{f=$2; n[f]=0; next} /^---END$/{f=""} f!="" {n[f]++; print f "\t" $0} END {for (k in n) print "@@COUNT\t" k "\t" n[k]}' <<<"$R" > "$ST.outs"
  NEWF=$(mktemp "$ST.files.XXXX")
  while IFS=$'\t' read -r tag name cnt; do
    [ "$tag" = "@@COUNT" ] || continue
    old=$(awk -v k="$name" '$1==k {print $2}' "$ST.files"); old=${old:-0}
    [ "$FIRST" = 1 ] && old=$cnt
    [ "$cnt" -lt "$old" ] && old=0                               # file bị ghi đè khi phóng driver mới
    awk -F'\t' -v k="$name" '$1==k {print substr($0, length(k)+2)}' "$ST.outs" | tail -n +$((old+1)) \
      | grep -E 'XONG|!!|phóng|chờ VRAM|đã thoát' | sed "s|^|[$H $name] |" | cut -c1-220
    echo "$name $cnt" >> "$NEWF"
  done < <(grep '^@@COUNT' "$ST.outs")
  mv "$NEWF" "$ST.files"
  RUNID=${CUR%/*}
  awk '/^---LOG$/{m=1;next} /^---DLOG$/{m=0} m' <<<"$R" > "$ST.log"
  AL=$(python3 /drive1/cuongtm/ntat/MultiVD/_FinalPaperExperiment/scripts/fpe.py stall "$RUNID" "$ST.log" 2>/dev/null)
  # so "loại" cảnh báo, bỏ chữ số: cùng một cảnh báo chỉ đổi số epoch thì KHÔNG báo lại mỗi epoch (28/09: lặp ep13, 14, 15, 16…)
  if [ -n "$AL" ] && [ "$CUR|$(tr -d '0-9.,' <<<"$AL")" != "$(cat "$ST.alertkind" 2>/dev/null)" ]; then
    echo "[$H $CUR] ⚠ CẢNH BÁO $AL"; echo "$AL" > "$ST.alert"; echo "$CUR|$(tr -d '0-9.,' <<<"$AL")" > "$ST.alertkind"; fi
  NOUT=${NL:-$NOUT}; NDL=${NDLN:-$NDL}; FIRST=0
  if [ "${TRN:-0}" -ge 1 ] && [ "${GPU%%,*}" -le 2 ] 2>/dev/null; then IDLE=$((IDLE+1)); [ "$IDLE" -eq 8 ] && echo "[mon $H $(date +%H:%M)] CẢNH BÁO: trainer còn sống nhưng GPU ~0% suốt 8 phút (GPU=$GPU)"; else IDLE=0; fi
  echo "$NDL $NOUT $IDLE $FAIL" > "$ST"
  if [ "$DRV" = none ] && [ "$QUE" = none ]; then echo "[mon $H $(date +%H:%M)] không còn driver lẫn hàng đợi trên $H (trainer=$TRN) — monitor dừng"; exit 0; fi
  [ -n "${ONCE:-}" ] && exit 0
  sleep 60
done
