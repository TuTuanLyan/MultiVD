#!/usr/bin/env bash
# Tong hop dot quet lambda x rho khi da chay xong. Doc du lieu TAI CHO tren tung
# may (khong keo file ve), in ra bang de dan vao FACTS.md.
#
#   bash scripts/finalize_sweep46.sh
#
# In ba bang cho MOI nguon:
#   [A] anh huong cua rho, ghep cap voi chinh cau hinh rho=0 (cung lambda, cung fold)
#   [B] gop ca ba lambda
#   [C] lambda nao cho ket qua cao nhat, kem Delta so voi baseline
# Cong them mot bang gop theo tung lambda — do la truc dang co tin hieu.
#
# Ghep cap la bat buoc: cung checkpoint Phase 1, cung lambda, cung fold, cung may.
# Khong bao gio lay hieu cua hai trung binh (muc 2 CLAUDE.md).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

H158=tranmanhcuong@112.137.129.158
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i /home/ntat/.ssh/id_ed25519"
R158=/data/ntat/MultiVD
PY158=/data/ntat/envs/vdenv/bin/python
PY161=/home/ntat/miniconda3/envs/vdenv/bin/python
SC=/tmp/claude-1045/-home-ntat-workspace-MultiVD/a84c9fd3-bdf4-4ecf-8166-dc8953f347d2/scratchpad

# Script phan tich duoc giu trong repo de phien sau con dung lai duoc
mkdir -p tools
for f in asam_effect.py bylam.py; do
  [[ -f "tools/$f" ]] || { [[ -f "$SC/$f" ]] && cp "$SC/$f" "tools/$f"; }
done

echo "############ TIEN DO ############"
N1=$(ls results/sw_t5p/transfer_latent_bottleneck_4cwe_*/seed_42/fold*.json 2>/dev/null | wc -l)
N8=$($SSH $H158 "ls $R158/results/sw_t5p/transfer_latent_bottleneck_com_*/seed_42/fold*.json 2>/dev/null | wc -l" 2>/dev/null | tail -1)
echo "  161 nguon 4cwe : $N1/63 o"
echo "  158 nguon com  : ${N8:-?}/63 o"
if (( N1 < 63 )) || (( ${N8:-0} < 63 )); then
  echo "  !! CHUA DU 63 o ca hai ben — so duoi day la ket qua giua chung."
fi

echo
echo "############ 161 · nguon 4cwe ############"
"$PY161" tools/asam_effect.py 4cwe .
echo
echo "  --- gop theo tung lambda ---"
"$PY161" tools/bylam.py 4cwe .

echo
echo "############ 158 · nguon common ############"
$SSH $H158 "test -f $R158/tools/asam_effect.py" 2>/dev/null || \
  scp -q -o BatchMode=yes -i /home/ntat/.ssh/id_ed25519 tools/asam_effect.py tools/bylam.py "$H158:$R158/tools/" 2>/dev/null || \
  { $SSH $H158 "mkdir -p $R158/tools" 2>/dev/null
    scp -q -o BatchMode=yes -i /home/ntat/.ssh/id_ed25519 tools/asam_effect.py tools/bylam.py "$H158:$R158/tools/" 2>/dev/null; }
$SSH $H158 "cd $R158 && $PY158 tools/asam_effect.py com $R158" 2>/dev/null
echo
echo "  --- gop theo tung lambda ---"
$SSH $H158 "cd $R158 && $PY158 tools/bylam.py com $R158" 2>/dev/null

cat <<'TXT'

############ NHAC ############
  - n=3 fold moi o: san kiem dau la p=0.25, khong o don le nao dat duoc p<0.05.
  - San nhieu do duoc cua du an: 0.010 (chay lai cung cau hinh, may khac).
    Hieu ung nho hon nguong nay khong phan biet duoc voi viec chay lai.
  - Neu doc so nay de viet vao FACTS.md thi ghi ca n, so fold cung dau va bien do
    min-max, khong chi ghi trung binh va p.
TXT
