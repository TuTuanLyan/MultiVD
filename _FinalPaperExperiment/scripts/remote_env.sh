# Dùng chung cho các script điều khiển máy từ xa (158, vast): nạp SSH/OUT/cổng từ runs.json.
#   . scripts/remote_env.sh <host>   ->  $SSH $OUT $PORT, mảng SSHO (tuỳ chọn ssh), $RSH (cho rsync -e)
_H=$1; _MAN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs.json
SSH=$(python3 -c "import json;print(json.load(open('$_MAN'))['hosts']['$_H']['ssh'])")
OUT=$(python3 -c "import json;print(json.load(open('$_MAN'))['hosts']['$_H']['out'])")
PORT=$(python3 -c "import json;print(json.load(open('$_MAN'))['hosts']['$_H'].get('port',22))")
SSHO=(-o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR -p "$PORT")
RSH="ssh -o BatchMode=yes -o LogLevel=ERROR -p $PORT"
