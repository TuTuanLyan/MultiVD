#!/usr/bin/env bash
# THU KHOI adapter-fusion — CPU, 32 mau, 1 epoch. Muc dich KHONG phai do chat luong ma la
# chung minh ba dieu TRUOC khi tieu GPU:
#   1. Pha 1 ghi duoc adapter vao checkpoint, va adapter DA DICH khoi 0
#      (adapter khoi tao up=0; neu lr sai thi no nam im va ca khoi Pha 2 do tren tien de sai)
#   2. Pha 2 nap lai duoc, them adapter dich + fusion, va adapter NGUON dong bang that
#   3. Ca hai bien the `ft` / `frozen` khac nhau DUNG o backbone
# Da tra gia cho bai hoc nay o cong 8 chieu: no ket o g=0.5000 vi lr backbone qua nho.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/opt/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""          # CPU — GPU dang chay khoi doi chung, khong dung vao
PY="${PYTHON:-python}"
M=microsoft/codebert-base
OUT=/tmp/fusion_smoke; rm -rf $OUT; mkdir -p $OUT
COMMON="--seed 42 --batch_size 4 --eval_batch_size 4 --max_length 128 \
  --truncation_strategy head_middle_tail --weight_decay 0.01 --patience 1 --min_epochs 1 \
  --max_grad_norm 1.0 --num_workers 0 --data_root data/sven_python_folds_norm \
  --target_lang python --model_name $M --pooling cls"

echo "########## 1) PHA 1 voi adapter ##########"
$PY -u src/train_transfer.py --phase phase1 \
  --run_name smoke --method_name p1 --data_path data/phase1_common.jsonl \
  --aux_mode latent_bottleneck --cwe_vocab precomputed --num_latent 8 \
  --epochs 1 --learning_rate 2e-5 --lambda_cwe 0.05 --sam_rho 0 \
  --adapter_dim 48 --adapter_lr 1e-4 \
  --max_train_samples 32 --max_eval_samples 16 \
  --checkpoint_path $OUT/p1.pt $COMMON > $OUT/p1.log 2>&1
rc=$?; echo "  ma thoat $rc"; [ $rc -eq 0 ] || { tail -20 $OUT/p1.log; exit 1; }
grep -E "Adapter BAT|nhom ADAPTER" $OUT/p1.log | sed 's/^/  /'

$PY - <<'PYEOF'
import torch, sys
sd = torch.load("/tmp/fusion_smoke/p1.pt", map_location="cpu", weights_only=True)["model_state_dict"]
ad = {k: v for k, v in sd.items() if ".adapters." in k}
print(f"  khoa adapter trong checkpoint: {len(ad)}")
assert ad, "!! Pha 1 KHONG ghi adapter vao checkpoint"
ups = [v for k, v in ad.items() if k.endswith("up.weight")]
mx = max(float(v.abs().max()) for v in ups)
print(f"  |up.weight| lon nhat sau 1 epoch = {mx:.3e}  (khoi tao = 0)")
assert mx > 1e-6, "!! adapter NAM IM o 0 — lr sai, y het bay cong 8 chieu"
print("  => Pha 1 OK: adapter co trong checkpoint VA da hoc")
PYEOF
[ $? -eq 0 ] || exit 1

for MODE in ft frozen; do
  echo "########## 2) PHA 2 bien the $MODE ##########"
  $PY -u src/train_transfer.py --phase phase2 \
    --run_name smoke --method_name "p2_$MODE" --fold 1 \
    --aux_mode latent_bottleneck --cwe_vocab precomputed --num_latent 8 \
    --epochs 1 --learning_rate 2e-5 --phase2_optimizer adamw --sam_rho 0 \
    --phase2_fusion $MODE --adapter_lr 1e-4 \
    --max_train_samples 32 --max_eval_samples 16 \
    --source_checkpoint $OUT/p1.pt --checkpoint_path $OUT/p2_$MODE.pt \
    --output_dir $OUT/res_$MODE $COMMON > $OUT/p2_$MODE.log 2>&1
  rc=$?; echo "  ma thoat $rc"; [ $rc -eq 0 ] || { tail -25 $OUT/p2_$MODE.log; exit 1; }
  grep -E "Dung lai adapter|ADAPTER FUSION BAT|nhom ADAPTER rieng" $OUT/p2_$MODE.log | sed 's/^/  /'
done

echo "########## 3) DOI CHIEU hai bien the ##########"
$PY - <<'PYEOF'
import torch
L = lambda p: torch.load(p, map_location="cpu", weights_only=True)["model_state_dict"]
p1 = L("/tmp/fusion_smoke/p1.pt")
ok = True
for mode in ("ft", "frozen"):
    p2 = L(f"/tmp/fusion_smoke/p2_{mode}.pt")
    src = [k for k in p2 if ".adapters.src." in k]
    tgt = [k for k in p2 if ".adapters.tgt." in k]
    fus = [k for k in p2 if ".fusion." in k]
    dsrc = max(float((p2[k] - p1[k]).abs().max()) for k in src)
    dtgt = max(float(p2[k].abs().max()) for k in tgt if k.endswith("up.weight"))
    bbk = [k for k in p2 if k.startswith("backbone.") and ".adapters." not in k
           and ".fusion." not in k and k in p1]
    dbb = max(float((p2[k] - p1[k]).abs().max()) for k in bbk)
    print(f"  [{mode:6}] adapter nguon {len(src)} khoa, |doi| = {dsrc:.3e}   <- PHAI = 0")
    print(f"           adapter dich  {len(tgt)} khoa, |up| = {dtgt:.3e}   <- PHAI > 0")
    print(f"           fusion        {len(fus)} khoa")
    print(f"           backbone      |doi| = {dbb:.3e}   <- ft PHAI > 0, frozen PHAI = 0")
    if dsrc > 1e-9: print(f"  !! [{mode}] adapter NGUON bi doi — sai"); ok = False
    if dtgt <= 1e-9: print(f"  !! [{mode}] adapter DICH nam im — sai"); ok = False
    if mode == "ft"     and dbb <= 1e-9: print("  !! [ft] backbone KHONG doi — phai fine-tune"); ok = False
    if mode == "frozen" and dbb >  1e-9: print("  !! [frozen] backbone BI DOI — phai dong bang"); ok = False
print("\n  " + ("TAT CA DAT — duong chay dung nhu mo ta" if ok else "CO PHEP HONG"))
raise SystemExit(0 if ok else 1)
PYEOF
