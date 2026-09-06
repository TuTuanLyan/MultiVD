#!/usr/bin/env bash
# watch_opt1.sh — giam sat + XEP HANG HAI GIAI DOAN cho khoi OPT1 tren 161 (local) va 158 (ssh),
# cron 10 phut/lan.
#
#   giai doan 1: SOURCES="4cwe com"   (driver dang chay tu 11:19 UTC 06/09)
#   giai doan 2: SOURCES="full"       (nguoi dung 06/09: "co them nhung chay cuoi cung xem anh huong")
#
# Quyet dinh "chay tiep hay khong" bang DEM HIEN VAT (so fold*.json), khong doc dong log
# (memory one-driver-one-lock-and-count-artifacts): thieu o nao thi vong sau matrix.sh chay
# dung o do. Moi giai doan phong lai toi da MAXPASS lan de khong lap vo han khi mot o luon hong.
#
# CHI ba viec: (1) ghi mot dong trang thai moi may, (2) phong driver giai doan ke tiep khi
# khong con driver nao song, (3) keo ket qua 158 ve results_opt1_158/.
# KHONG BAO GIO giet tien trinh, KHONG xoa gi. 161 dung chung GPU: chi phong khi VRAM trong
# >= 13 GB va khong con job train nao cua khoi nay dang chay.
#
# Cron khong co moi truong (memory cron-has-no-environment): tu dat HOME/USER/PATH.
# Thu: env -i /bin/bash -c 'DRY=1 bash /drive1/cuongtm/ntat/MultiVD/scripts/watch_opt1.sh'
# Thu cong xep hang: bash scripts/watch_opt1.sh --test-stage
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
set -u
ROOT=/drive1/cuongtm/ntat/MultiVD
cd "$ROOT" || exit 1
LOG=log/watch_opt1.log
PY161=/home/ntat/miniconda3/envs/vdenv/bin/python
R158="tranmanhcuong@112.137.129.158"; ROOT158=/data/ntat/MultiVD; PY158=/data/ntat/envs/vdenv/bin/python
LOCK161="${LOCK161:-/tmp/multivd_opt1.lock}"
FOLDS161="1 2"; FOLDS158="3 4 5"
NCONF=10
MAXPASS="${MAXPASS:-3}"
DRY="${DRY:-0}"
ts(){ date -u '+%F %T'; }
say(){ echo "$(ts) | $*" >> "$LOG"; }

# next_stage <n_4cwe> <n_com> <n_full> <so_fold>  ->  in "main" | "full" | "done"
next_stage(){
  local need=$(( $4 * NCONF ))
  if (( $1 + $2 < 2*need )); then echo main
  elif (( $3 < need )); then echo full
  else echo done; fi
}
if [[ "${1:-}" == "--test-stage" ]]; then
  # hai chieu: thieu giai doan 1 -> main; du 1 thieu 2 -> full; du ca -> done
  [[ $(next_stage 10 20 0 2) == main ]] && [[ $(next_stage 20 20 0 2) == full ]] \
    && [[ $(next_stage 20 20 20 2) == done ]] && [[ $(next_stage 30 30 29 3) == full ]] \
    && echo "next_stage OK" || { echo "next_stage SAI"; exit 1; }
  exit 0
fi

count(){ # $1=thu muc ket qua $2=nguon
  ls "$1"/transfer_latent_bottleneck_"$2"_l0p05_*/seed_42/fold*.json 2>/dev/null | wc -l; }

# ---------------- 161 ----------------
alive161=$(flock -n "$LOCK161" -c true 2>/dev/null && echo no || echo yes)
a4=$(count results/opt1_t5p 4cwe); ac=$(count results/opt1_t5p com); af=$(count results/opt1_t5p full)
b161=$(ls results/opt1_t5p/baseline/seed_42/fold*.json 2>/dev/null | wc -l)
NF161=$(echo $FOLDS161 | wc -w)
stage161=$(next_stage "$a4" "$ac" "$af" "$NF161")
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); used="${used:-0}"
free=$(( 16376 - used ))
# job train cua chinh khoi nay con chay? (loc theo cot user + chuoi; dong lenh cua script nay la `bash scripts/watch_opt1.sh`, khong tu khop)
orphan=$(ps -eo user,args --no-headers | awk '$1=="ntat" && /src\/train_(transfer|baseline)\.py/ && /opt1/' | wc -l)
last161=$(grep -E "^=====|^=== .*\| fold|THAT BAI|xong" log/opt1_161.log 2>/dev/null | tail -1 | cut -c1-120)
say "161 | driver=$alive161 | 4cwe=$a4 com=$ac full=$af /$(( NF161*NCONF )) moi nguon, base=$b161/$NF161 | ke tiep=$stage161 | vram_used=${used}MiB | train_procs=$orphan | $last161"
if [[ "$alive161" == no && "$stage161" != done ]]; then
  pass=$(cat "log/opt1_161_passes_$stage161" 2>/dev/null || echo 0)
  if (( pass >= MAXPASS )); then say "161 | giai doan $stage161 da phong $pass lan ma chua du — DUNG, can nguoi xem"
  elif (( orphan > 0 )); then say "161 | con $orphan job train dang chay — cho"
  elif (( free < 13000 )); then say "161 | VRAM trong ${free}MiB < 13000 — NHUONG, chua phong"
  else
    SRCS="4cwe com"; [[ "$stage161" == full ]] && SRCS="full"
    if [[ "$DRY" == 1 ]]; then say "161 | [DRY] se phong giai doan $stage161 (SOURCES=$SRCS) lan $((pass+1))"
    else
      echo $((pass+1)) > "log/opt1_161_passes_$stage161"
      SOURCES="$SRCS" FOLD_LIST="$FOLDS161" PYTHON=$PY161 setsid nohup bash run/opt1.sh >> log/opt1_161.log 2>&1 < /dev/null &
      say "161 | PHONG giai doan $stage161 (SOURCES=$SRCS) lan $((pass+1))"
    fi
  fi
fi

# ---------------- 158 ----------------
out=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$R158" "cd $ROOT158 || exit 1
{ flock -n /tmp/multivd_opt1.lock -c true && echo alive=no || echo alive=yes; }
for S in 4cwe com full; do echo n_\$S=\$(ls results/opt1_t5p/transfer_latent_bottleneck_\${S}_l0p05_*/seed_42/fold*.json 2>/dev/null | wc -l); done
echo b=\$(ls results/opt1_t5p/baseline/seed_42/fold*.json 2>/dev/null | wc -l)
echo used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo orphan=\$(ps -eo args --no-headers | grep -c 'src/train_[a-z]*\.py.*opt1')
echo ckfull=\$(stat -c %s model/n48/phase1/t5p__latent_bottleneck_full_l0p05/seed_42/best.pt 2>/dev/null)
echo last=\$(grep -E '^=====|THAT BAI|xong' log/opt1_158.log 2>/dev/null | tail -1 | cut -c1-120)" 2>/dev/null)
if [[ -z "$out" ]]; then
  say "158 | KHONG SSH DUOC"
else
  g(){ sed -n "s/^$1=//p" <<<"$out" | head -1; }
  alive158=$(g alive); c4=$(g n_4cwe); cc=$(g n_com); cf=$(g n_full); b158=$(g b)
  used158=$(g used); orphan158=$(g orphan); orphan158="${orphan158:-0}"; ckfull=$(g ckfull); last158=$(g last)
  NF158=$(echo $FOLDS158 | wc -w)
  stage158=$(next_stage "${c4:-0}" "${cc:-0}" "${cf:-0}" "$NF158")
  say "158 | driver=$alive158 | 4cwe=${c4:-?} com=${cc:-?} full=${cf:-?} /$(( NF158*NCONF )) moi nguon, base=${b158:-?}/$NF158 | ke tiep=$stage158 | vram_used=${used158:-?}MiB | train_procs=$orphan158 | $last158"
  if [[ "$alive158" == no && "$stage158" != done ]]; then
    pass=$(cat "log/opt1_158_passes_$stage158" 2>/dev/null || echo 0)
    if (( pass >= MAXPASS )); then say "158 | giai doan $stage158 da phong $pass lan ma chua du — DUNG, can nguoi xem"
    elif (( orphan158 > 0 )); then say "158 | con $orphan158 job train — cho"
    elif [[ "$stage158" == full && "${ckfull:-0}" != 438519277 ]]; then say "158 | THIEU/lech checkpoint Pha 1 full seed 42 (${ckfull:-0} B, can 438519277) — chua phong full"
    else
      SRCS="4cwe com"; [[ "$stage158" == full ]] && SRCS="full"
      if [[ "$DRY" == 1 ]]; then say "158 | [DRY] se phong giai doan $stage158 (SOURCES=$SRCS) lan $((pass+1))"
      else
        echo $((pass+1)) > "log/opt1_158_passes_$stage158"
        timeout 60 ssh -o BatchMode=yes "$R158" "cd $ROOT158 && (SOURCES='$SRCS' FOLD_LIST='$FOLDS158' PYTHON=$PY158 setsid nohup bash run/opt1.sh >> log/opt1_158.log 2>&1 < /dev/null &)" 2>/dev/null
        say "158 | PHONG giai doan $stage158 (SOURCES=$SRCS) lan $((pass+1))"
      fi
    fi
  fi
  # keo ket qua ve (chi them, khong xoa, khong ghi de)
  if (( ${c4:-0} + ${cc:-0} + ${cf:-0} + ${b158:-0} > 0 )); then
    mkdir -p results_opt1_158
    if rsync -az --ignore-existing "$R158:$ROOT158/results/opt1_t5p/" results_opt1_158/ 2>/dev/null; then
      say "158 | rsync -> results_opt1_158/ : $(ls results_opt1_158/*/seed_42/*.json 2>/dev/null | wc -l) file"
    fi
  fi
fi
