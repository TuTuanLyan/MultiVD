#!/usr/bin/env bash
# Giet MOT chuoi driver cua CHINH MINH cho sach, khong de lai mo coi.
#
#   bash scripts/kill_my_chain.sh 'run/chot2bb.sh'          # that
#   DRY=1 bash scripts/kill_my_chain.sh 'run/chot2bb.sh'    # chi liet ke
#
# --- Vi sao co file nay (FACTS §31) ---
# 09/09 mot ban tu viet da sot: no duyet con truoc roi giet cha, nhung trong luc no giet o
# duoi sau thi CHA VAN SONG va kip sinh mot `bash run/matrix.sh` moi. Buoc quet mo coi chi
# khop `train_*.py` — CHI CAI LA — nen bo sot dung cai VO trung gian, va vo do de ra mot
# python moi SAU khi quet chay xong. 12 GB VRAM bi giu, suyt thanh hai chuoi mot GPU.
#
# --- Vi sao co CAC CHAN o duoi (FACTS §31.1) ---
# Ban dau cua file nay KHONG chan gi. Mot lenh thu goi nham voi PAT rong, va no khop MOI
# tien trinh cua user: systemd --user, code-server, tmux, daemon claude, ssh, VA ca driver
# thi nghiem dang chay. Chi DRY=1 cuu. Mot cong cu giet ma khong tu tu choi khi mau khop
# qua nhieu thi khong duoc phep ton tai.
#
# Bon quy tac van hanh:
#   1. STOP cha TRUOC  -> no khong sinh them duoc gi nua khi minh dang don.
#   2. Quet phu MOI TANG (vo trung gian lan la), khong chi tang duoi cung.
#   3. LAP den khi khong con gi; mot luot khong du.
#   4. Kiem chu so huu voi BA ket cuc truoc MOI lan giet. Khong bao gio `pkill -f`.
set -uo pipefail
ME=$(id -un)
DRY="${DRY:-0}"
MAXROOTS="${MAXROOTS:-4}"
# Moi tang trong chuoi. Them tang moi vao day khi co runner moi.
TIERS="${TIERS:-train_transfer\.py|train_baseline\.py|train\.py|run/matrix\.sh|run/opt1\.sh}"
# TUYET DOI khong giet, du mau co khop: ha tang phien lam viec cua chinh nguoi dung.
NEVER='systemd|\(sd-pam\)|pipewire|dbus-daemon|code-server|\.vscode-server|tmux|/bin/sshd|[^a-z]ssh |claude|node .*server-main|shellIntegration-bash|bg-pty-host|bg-spare|daemon run'

say(){ echo "$(date -u '+%F %T') | $*"; }
die(){ say "!! $*"; exit 2; }

PAT="${1-}"
# --- CHAN 1: mau phai co that va phai du dac trung ---
[ -n "$PAT" ]      || die "thieu mau driver. Vi du: bash scripts/kill_my_chain.sh 'run/chot2bb.sh'"
[ ${#PAT} -ge 6 ]  || die "mau '$PAT' qua ngan (<6 ky tu) — de khop nham. Dat duong dan day du, vd 'run/chot2bb.sh'"
case "$PAT" in *[.]\**|\*|.\*) die "mau '$PAT' la mau bat tat ca — tu choi";; esac

AWKPAT="${PAT//\\./.}"
roots=$(ps -u "$ME" -o pid=,args= | awk -v p="$AWKPAT" -v self=$$ '$0 ~ p && $0 !~ /kill_my_chain/ && $1 != self {print $1}')
nr=$(printf '%s\n' $roots | grep -c . || true)

# --- CHAN 2: mau khop qua nhieu => gan nhu chac chan la mau sai ---
if [ "$nr" -gt "$MAXROOTS" ]; then
  say "mau '$PAT' khop $nr tien trinh (toi da $MAXROOTS). Gan nhu chac chan la MAU SAI."
  ps -u "$ME" -o pid=,args= | awk -v p="$AWKPAT" '$0 ~ p {print "    " substr($0,1,110)}' | head -20
  die "tu choi chay. Neu that su muon, dat MAXROOTS cao hon MOT cach co y thuc."
fi
[ "$nr" = 0 ] && { say "khong co tien trinh nao khop /$PAT/ — khong co gi de don"; }

# --- BA ket cuc, khong phai hai ---
# `ps -o user= -p <pid>` tra RONG ca khi tien trinh DA THOAT lan khi KHONG DOC DUOC. Gop hai
# cai do vao "khong phai cua toi" thi mot tien trinh vua thoat bi bao nham thanh "cua user
# khac", va mot tien trinh cua chinh minh doc hut se bi BO QUA dung luc can giet. Cung ho voi
# loi da xoa mat hai checkpoint 08/09 (CLAUDE.md muc 3).
#   0 = cua toi | 1 = cua user khac | 2 = da thoat | 3 = con song nhung khong doc duoc
owner(){ local u; u=$(ps -o user= -p "$1" 2>/dev/null | tr -d ' ')
  if [ -z "$u" ]; then [ -d "/proc/$1" ] && { echo "?"; return 3; } || { echo "-"; return 2; }; fi
  [ "$u" = "$ME" ] && { echo "$u"; return 0; } || { echo "$u"; return 1; }; }

protected(){ ps -o args= -p "$1" 2>/dev/null | grep -qE "$NEVER"; }

zap(){ local p=$1 what=$2 u rc
  u=$(owner "$p"); rc=$?
  case $rc in
    2) say "  bo qua $p — da tu thoat";                                                       return ;;
    1) say "  BO QUA $p — cua user khac ($u)";                                                return ;;
    3) say "  !! $p con song nhung khong doc duoc chu so huu — KHONG giet, can nguoi xem";     return ;;
  esac
  # --- CHAN 3: danh sach cam, du la cua minh ---
  if protected "$p"; then say "  BAO VE $p ($(ps -o comm= -p $p|tr -d ' ')) — trong danh sach cam, KHONG giet"; return; fi
  if [ "$DRY" = 1 ]; then say "  [DRY] se giet $p ($what)"; else say "  giet $p ($what)"; kill -9 "$p" 2>/dev/null; fi; }

say "mau '$PAT' khop $nr tien trinh goc${DRY:+  (DRY=$DRY)}"
for r in $roots; do
  owner "$r" >/dev/null; [ $? = 0 ] || continue
  protected "$r" && { say "BAO VE $r — trong danh sach cam, bo qua ca cay"; continue; }
  [ "$DRY" = 1 ] && say "[DRY] se STOP $r" || { say "STOP $r (de no khong sinh them)"; kill -STOP "$r" 2>/dev/null; }
done

kt(){ local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do kt "$c"; done
      zap "$p" "$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ')"; }
for r in $roots; do protected "$r" || kt "$r"; done

for i in 1 2 3 4 5; do
  sleep 3
  left=$(ps -u "$ME" -o pid=,ppid=,args= | awk -v t="${TIERS//\\./.}" '$2==1 && $0 ~ t {print $1}')
  [ -z "$left" ] && { say "luot $i: sach"; break; }
  for p in $left; do zap "$p" "mo coi tang giua/la"; done
  [ "$i" = 5 ] && say "!! VAN CON sau 5 luot: $(echo $left) — CAN NGUOI XEM"
done

say "chuoi con lai cua toi:"
o=$(ps -u "$ME" -o pid=,ppid=,etimes=,args= | grep -E "${TIERS//\\./.}|$AWKPAT" | grep -vE 'grep|kill_my_chain')
[ -n "$o" ] && cut -c1-110 <<<"$o" | sed 's/^/    /' || echo "    (khong con)"
if command -v nvidia-smi >/dev/null; then
  g=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
  say "GPU:"; [ -n "$g" ] && sed 's/^/    /' <<<"$g" || echo "    (khong tien trinh nao dang dung GPU)"
fi
