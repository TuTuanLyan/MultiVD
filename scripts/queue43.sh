#!/usr/bin/env bash
# Hang doi viec cho ntat. Moi dong mot job:
#     <ten>|<backbone-spec>|<bien moi truong>|<script, mac dinh run/day43_machine.sh>
# Dong bat dau bang # la ghi chu. Job da xong duoc ghi vao log/queue43.done.
#
# Muc dich: KHONG DE VAST TRONG. May ranh ma van tinh tien la lang phi duy nhat
# khong bao chua duoc — con chay them mot thi nghiem thi tệ nhat cung ra mot
# ket qua null co ich.
#
#   bash scripts/queue43.sh next        # in job ke tiep chua chay (rong = het)
#   bash scripts/queue43.sh mark <ten>  # danh dau da xong
#   bash scripts/queue43.sh list        # xem trang thai
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q=log/queue43.txt; D=log/queue43.done; mkdir -p log; touch "$D"

if [[ ! -f "$Q" ]]; then
cat > "$Q" <<'EOF'
# THU TU UU TIEN — ly do o cuoi moi dong.
#
# 1) rho=0.5 cho t5p. Cau hoi quan trong nhat sau ket qua rong o rho=0.1:
#    ASAM khong giup, hay ta chua bao gio bat no du manh? Bai ASAM quet
#    {5e-5..2.0} va chon 0.5 (CIFAR-10) / 1.0 (CIFAR-100, ImageNet); thi nghiem
#    transformer duy nhat cua ho dung 0.2. Minh dang o 0.1 — thap hon MOI tien
#    le. Checkpoint t5p da nam san tren may sau job truoc nen khong ton truyen.
#    Nhanh _ctl dung lai duoc tu vong rho=0.1 -> chi chay 50 o thay vi 100.
t5p_asam_r05|t5p=Salesforce/codet5p-220m-bimodal:mean|ASAM_RHO=0.5 MODES_SKIP_CTL=1
#
# 2) rho=1.0. Neu 0.5 van rong thi 1.0 khep lai dai ma bai bao dung. Neu 0.5 da
#    co hieu ung thi job nay van co ich de biet chieu bien thien.
t5p_asam_r10|t5p=Salesforce/codet5p-220m-bimodal:mean|ASAM_RHO=1.0 MODES_SKIP_CTL=1
#
# 3) Vong AdamW cho t5p. Nguoi dung da hen "AdamW de sau", va no do dung thu
#    TRAM (ICLR 2024) canh bao: cong mot regularizer NEO vao loss ma SAM dang
#    nhieu lam ket qua te hon Adam tron (M2D2: Adam 27.4, ASAM 26.8,
#    ASAM+TRPO 30.2). RecAdam thuoc dung ho do. Chay AdamW la phep tach tuong
#    tac ASAM x neo — khong the suy ra tu vong RecAdam.
t5p_adamw|t5p=Salesforce/codet5p-220m-bimodal:mean|ASAM_RHO=0.1 OPTIMIZERS=adamw
EOF
fi

case "${1:-list}" in
  next)
    while IFS= read -r line; do
      [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
      n="${line%%|*}"
      grep -qx "$n" "$D" && continue
      echo "$line"; exit 0
    done < "$Q"
    exit 0 ;;
  mark) echo "${2:?ten job}" >> "$D" ;;
  list)
    while IFS= read -r line; do
      [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
      n="${line%%|*}"
      if grep -qx "$n" "$D"; then echo "  [xong] $n"; else echo "  [cho ] $n"; fi
    done < "$Q" ;;
esac
