#!/usr/bin/env bash
# Giam sat dot quet lambda x rho tren CA HAI may, chiu duoc reboot.
#
# Goi tu cron moi 10 phut:
#   */10 * * * * flock -n /tmp/mvd_super.lock bash /home/ntat/workspace/MultiVD/scripts/sweep46_super.sh
#
# Cron la thu song sot qua reboot, khong phai tien trinh nen. Moi lan chay, script
# nay chi NHIN trang thai roi phong lai thu gi con thieu — chay lai bao nhieu lan
# cung khong hai (matrix.sh bo qua o da co ket qua). Vi vay:
#   - 161 khoi dong lai  -> cron chay lai script nay -> phong lai driver 4cwe
#   - 158 khoi dong lai  -> script nay thay driver chet -> phong lai qua SSH
#
# KHONG chep file giua hai may. Moi may tu chua Phase 1, Phase 2 va baseline cua
# rieng no, nen moi luoi tu no da so sanh duoc, khong can ghep cheo.
#
#   158 -> nguon `common`     161 -> nguon `4cwe`
#   ca hai: latent_bottleneck, t5p-bimodal, lambda {0.01,0.05,0.2}
#           x rho {0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0},
#           RecAdam, fold 1-3, seed 42  => 63 o moi may
#
# CACH KIEM TIEN TRINH — khong duoc tu khop
# `pgrep -f X` va `ps|grep X` deu bat luon chinh dong lenh cua no, nen driver chet
# van dem ra 1 va khong bao gio duoc phong lai. Da mac loi nay ba lan trong du an.
# O day dung `flock` (khong the tu khop) va thu thuat ngoac `[s]weep` cho ps.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# CRON KHONG DAT BIEN MOI TRUONG NAO CA.
# Ban dau script dung "$USER"; duoi cron bien do khong ton tai, gap `set -u` thanh
# loi chi mang: script chet ngay tai do, nhanh 161 khong bao gio chay toi, va may
# nam khong 7 tieng ma log giam sat im lang. Khong duoc dua vao bien moi truong nao
# ma tu shell dang nhap; thu gi can thi tu dat.
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

H158=tranmanhcuong@112.137.129.158
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i /home/ntat/.ssh/id_ed25519"
R158=/data/ntat/MultiVD
PY158=/data/ntat/envs/vdenv/bin/python
PY161=/home/ntat/miniconda3/envs/vdenv/bin/python
NEED=63   # 7 rho x 3 lambda x 3 fold; 36 o dau da chay xong
LOG=log/sweep46_super.log; mkdir -p log
say(){ echo "$(date -u '+%F %T') $*" >> "$LOG"; }
rsh(){ timeout 60 $SSH $H158 "$1" 2>/dev/null; }

# ---------- 158: nguon common ----------
nck8=$(rsh "ls $R158/model/sw/phase1/*com*/seed_42/best.pt 2>/dev/null | wc -l" | tail -1)
nck8="${nck8:-0}"
p1done8=$(( nck8 >= 3 ? 1 : 0 ))
busy8=$(rsh "ps -eo comm,args --no-headers | awk '\$1==\"python\" && /src\/train_transfer/ {n++} \$1==\"bash\" && /[s]weep46_p[12]\.sh/ {n++} END{print n+0}'" | tail -1)
busy8="${busy8:-1}"
n8=$(rsh "ls $R158/results/sw_t5p/transfer_latent_bottleneck_com_*/seed_42/fold*.json 2>/dev/null | wc -l" | tail -1)
n8="${n8:-0}"

if (( p1done8 == 0 )); then
  if (( busy8 == 0 )); then
    say "[158] Pha 1 chua xong va khong co tien trinh nao — phong lai Pha 1"
    rsh "cd $R158 && PYTHON=$PY158 setsid nohup bash run/sweep46_p1.sh >> log/sweep46_p1.log 2>&1 </dev/null &"
  else
    say "[158] Pha 1 dang chay (tien trinh=$busy8, checkpoint $nck8/3)"
  fi
elif (( n8 < NEED )); then
  if (( busy8 == 0 )); then
    say "[158] Pha 2: $n8/$NEED o, khong co driver — phong lai (fold 1 2 3, nguon com)"
    rsh "cd $R158 && SRC=com RHOS='0 0.05 0.1 0.2 0.5 1.0 2.0' FOLD_LIST='1 2 3' PYTHON=$PY158 setsid nohup bash run/sweep46_p2.sh >> log/sweep46_p2.log 2>&1 </dev/null &"
  else
    say "[158] Pha 2 dang chay: $n8/$NEED o"
  fi
else
  say "[158] XONG: $n8/$NEED o"
fi

# ---------- 161: nguon 4cwe ----------
nck1=$(ls model/sw/phase1/*4cwe*/seed_42/best.pt 2>/dev/null | wc -l)
p1done1=$(( nck1 >= 3 ? 1 : 0 ))
busy1=$(ps -eo comm,args --no-headers -u "$(id -un)" | awk '$1=="python" && /src\/train_transfer/ {n++} $1=="bash" && /[s]weep46_p[12]\.sh/ {n++} END{print n+0}')
n1=$(ls results/sw_t5p/transfer_latent_bottleneck_4cwe_*/seed_42/fold*.json 2>/dev/null | wc -l)

if (( p1done1 == 0 )); then
  if (( busy1 == 0 )); then
    say "[161] Pha 1 (4cwe) chua xong va khong co tien trinh — phong lai"
    SRC=4cwe PYTHON=$PY161 setsid nohup nice -n 10 bash run/sweep46_p1.sh \
      >> log/sweep46_p1_4cwe.log 2>&1 </dev/null &
  else
    say "[161] Pha 1 (4cwe) dang chay (tien trinh=$busy1, checkpoint $nck1/3)"
  fi
elif (( n1 < NEED )); then
  if (( busy1 == 0 )); then
    say "[161] Pha 2 (4cwe): $n1/$NEED o, khong co driver — phong lai (fold 1 2 3)"
    SRC=4cwe RHOS='0 0.05 0.1 0.2 0.5 1.0 2.0' FOLD_LIST='1 2 3' PYTHON=$PY161 setsid nohup nice -n 10 bash run/sweep46_p2.sh \
      >> log/sweep46_p2_4cwe.log 2>&1 </dev/null &
  else
    say "[161] Pha 2 (4cwe) dang chay: $n1/$NEED o"
  fi
else
  say "[161] XONG: $n1/$NEED o"
fi

say "  tom tat: 158 com $n8/$NEED (ckpt $nck8/3) | 161 4cwe $n1/$NEED (ckpt $nck1/3)"
