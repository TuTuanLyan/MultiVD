#!/usr/bin/env bash
# P1SEED — Pha 1 KHONG PHA 2: bo head `latent_bottleneck` khi CO adapter thi Pha 1 co LUON sap khong?
#
# Boi canh (FACTS §55): trong khoi `fusnone`, nhanh `none` + adapter sap hoan toan o seed 42 —
# train loss dung im o ln(2)=0.6931 suot 13 epoch, val macro-F1 chot 0.4430 (DUOI muc doan bua).
# Nhanh CO head cung giao thuc dat 0.5758. Va `none` tren `com` KHONG co adapter thi binh thuong
# (checkpoint cu s42: val 0.5518) — nen nghi van la chinh ADAPTER lam mat on dinh khi khong co head.
#
# §48.2 da do duoc: hai lan rut cung cau hinh co the lech rat xa. Nen mot lan sap KHONG du
# de goi la tinh chat. Phep kiem re nhat va quyet dinh nhat la chay lai DUNG Pha 1 do o
# seed khac, khong ton mot o Pha 2 nao.
#
#   sap lai o ca hai seed  => "adapter khong co head phu thi Pha 1 khong on dinh" la that
#   khong sap             => lan truoc chi la xui, va cau hoi goc (fusion co can head?) chua duoc tra loi
#
# Doi chung o cung seed: nhanh CO head, de biet seed do co "xau" cho ca hai hay chi cho `none`.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
SEEDS="${SEEDS:-7 1234}"
OUT="${OUT:-model/p1seed}"

exec 4>/tmp/mvd_p1seed.lock || exit 1
flock -n 4 || { echo "DA CO p1seed dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/workspace/p1seed.pid}"
echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

[ -f data/phase1_common.jsonl ] || { echo "!! THIEU data/phase1_common.jsonl"; exit 5; }
"$PY" -c "import torch,transformers,sklearn" || exit 5
mkdir -p "$OUT" log/p1seed

# Giao thuc GIONG HET khoi fusnone (doc tu training_args cua checkpoint co head).
run_p1() {  # $1=aux_mode  $2=seed  $3=ten
  local extra=""
  [ "$1" = latent_bottleneck ] && extra="--num_latent 8 --latent_temperature 0.1"
  echo "===== $(ts) | aux_mode=$1 seed=$2 ====="
  "$PY" -u src/train_transfer.py --phase phase1 \
    --run_name p1seed --method_name "p1_$3" \
    --data_path data/phase1_common.jsonl \
    --aux_mode "$1" --cwe_vocab precomputed $extra \
    --epochs 15 --min_epochs 3 --patience 10 \
    --learning_rate 2e-5 --lambda_cwe 0.05 \
    --checkpoint_path "$OUT/$3.pt" \
    --sam_rho 0 --adapter_dim 48 --adapter_lr 1e-4 \
    --seed "$2" --batch_size 16 --eval_batch_size 16 --max_length 512 \
    --truncation_strategy head_middle_tail --weight_decay 0.01 --max_grad_norm 1.0 \
    --num_workers 0 --model_name microsoft/codebert-base --pooling cls \
    > "log/p1seed/$3.log" 2>&1
  local rc=$?
  "$PY" - "$OUT/$3.pt" "$3" "log/p1seed/$3.log" <<'PYEOF'
import torch,sys,re
p,name,logf=sys.argv[1],sys.argv[2],sys.argv[3]
try:
    b=torch.load(p,map_location="cpu",weights_only=False)
    v,e=b["best_val_macro_f1"],b["best_epoch"]
except Exception as ex:
    print("  %-22s KHONG CO checkpoint (%s)" % (name,type(ex).__name__)); raise SystemExit
txt=open(logf,encoding="utf-8",errors="replace").read()
ne=txt.count("split=validation")
tl=re.findall(r"Train loss: ([0-9.]+)", txt)
sap = "SAP" if v < 0.50 else ("gan sap" if v < 0.53 else "binh thuong")
print("  %-22s val=%.4f epoch=%-3s epoch_da_chay=%-3s train_loss dau=%s cuoi=%s  => %s" % (
    name, v, e, ne, tl[0] if tl else "?", tl[-1] if tl else "?", sap))
PYEOF
  return $rc
}

echo "########## P1SEED bat dau $(ts) | seed: $SEEDS ##########"
for S in $SEEDS; do
  run_p1 none              "$S" "none_ad48_s$S"
  run_p1 latent_bottleneck "$S" "head_ad48_s$S"
done
echo "########## P1SEED xong $(ts) ##########"
echo "--- tong hop (nho so voi seed 42: none 0.4430 SAP / head 0.5758) ---"
"$PY" - "$OUT" <<'PYEOF'
import torch,glob,os,sys
for p in sorted(glob.glob(os.path.join(sys.argv[1],"*.pt"))):
    b=torch.load(p,map_location="cpu",weights_only=False)
    v=b["best_val_macro_f1"]
    print("  %-22s val=%.4f  %s" % (os.path.basename(p)[:-3], v, "SAP" if v<0.50 else ""))
PYEOF
