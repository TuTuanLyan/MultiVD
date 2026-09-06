#!/usr/bin/env bash
# Canh dot quet lambda x rho chay TRON VEN TREN 158. Khong chep gi ve 161.
#
#   nohup bash scripts/sweep46_watch.sh &
#
# VI SAO DON HET VE MOT MAY
# Nguoi dung khong muon chep file giua hai server (ca hai deu la may chinh, truy
# cap duoc nhu nhau). Ma moi o Phase 2 deu phai NAP mot checkpoint Phase 1, nen
# neu khong chep checkpoint sang 161 thi 161 khong the lam phan Phase 2 nao cua
# luoi nay. Vi vay ca Phase 1 lan Phase 2 nam tren 158.
#
# Loi keo theo: lambda va rho deu duoc so tren cung mot may, cung mot GPU, nen
# phep so trong luoi nay khong dinh chenh lech phan cung — sach hon ca ban chia
# hai may truoc do.
#
# BAI HOC DA AP DUNG
#   - Kiem driver song/chet bang `flock`, KHONG dung `ps|grep` (mau tu khop chinh
#     dong lenh cua no, driver chet van dem ra 1).
#   - Dieu kien "xong" la dong ket thuc do chinh driver in ra, khong phai dem o.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

H158=tranmanhcuong@112.137.129.158
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i /home/ntat/.ssh/id_ed25519"
R158=/data/ntat/MultiVD
PY158=/data/ntat/envs/vdenv/bin/python
EVERY="${EVERY:-600}"; MAX="${MAX:-300}"
FOLDS_ALL="${FOLDS_ALL:-1 2 3}"
NEED=36          # 3 lambda x 4 rho x 3 fold
LOG=log/sweep46_watch.log; mkdir -p log
say(){ echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
rsh(){ timeout 60 $SSH $H158 "$1" 2>/dev/null; }

p1done(){ rsh "grep -c 'SWEEP46 PHASE1 xong' $R158/log/sweep46_p1.log 2>/dev/null || true" | tail -1; }
p1alive(){ rsh "pgrep -f 'bash run/sweep46_p1.sh' >/dev/null && echo 1 || echo 0" | tail -1; }
p2alive(){ rsh "flock -n /tmp/multivd_sweep46.lock -c true 2>/dev/null && echo 0 || echo 1" | tail -1; }
ncell(){ rsh "ls $R158/results/sw_t5p/transfer_latent_bottleneck_com_*/seed_42/fold*.json 2>/dev/null | wc -l" | tail -1; }
nck(){ rsh "ls $R158/model/sw/phase1/*/seed_42/best.pt 2>/dev/null | wc -l" | tail -1; }

P1FAIL=0; P2FAIL=0
say "== sweep46_watch bat dau | tat ca tren 158 | can $NEED o (3 lambda x 4 rho x 3 fold) =="

for ((i=0;i<MAX;i++)); do
  D=$(p1done); D="${D:-0}"

  if (( D == 0 )); then
    A=$(p1alive); A="${A:-0}"
    say "[Pha 1] chua xong | checkpoint da co: $(nck)/3 | driver=$A"
    if (( A == 0 )); then
      P1FAIL=$((P1FAIL+1))
      say "[Pha 1] driver CHET ma chua xong — phong lai (lan $P1FAIL)"
      if (( P1FAIL > 3 )); then say "[Pha 1] that bai $P1FAIL lan, dung de nguoi xem."; exit 1; fi
      rsh "cd $R158 && PYTHON=$PY158 setsid nohup bash run/sweep46_p1.sh >> log/sweep46_p1.log 2>&1 </dev/null &"
      sleep 25
    fi
    sleep "$EVERY"; continue
  fi

  N=$(ncell); N="${N:-0}"
  A=$(p2alive); A="${A:-0}"
  say "[Pha 2] $N/$NEED o | driver=$A | checkpoint: $(nck)/3"

  if (( N >= NEED )); then
    say "== DOT QUET XONG: $N/$NEED o, tat ca nam tren 158 tai $R158/results/sw_t5p =="
    exit 0
  fi
  if (( A == 0 )); then
    P2FAIL=$((P2FAIL+1))
    if (( P2FAIL > 6 )); then say "[Pha 2] phong lai $P2FAIL lan van khong xong — dung de nguoi xem."; exit 1; fi
    say "[Pha 2] phong driver tren 158 — fold: $FOLDS_ALL (lan $P2FAIL)"
    rsh "cd $R158 && FOLD_LIST='$FOLDS_ALL' PYTHON=$PY158 setsid nohup bash run/sweep46_p2.sh >> log/sweep46_p2.log 2>&1 </dev/null &"
    sleep 25
  fi
  sleep "$EVERY"
done
say "== het $MAX vong =="
