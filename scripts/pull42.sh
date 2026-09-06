#!/usr/bin/env bash
# Keo ket qua fold ve may nay va in bang so sanh. Chay bat cu luc nao.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
mkdir -p results
for L in ntat ntat2; do
  read -r H P <<< "$(vast_endpoint "$L")"
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $P" \
    --include='s42_*/' --include='s42_*/**' --exclude='*' \
    "root@$H:/workspace/MultiVD/results/" results/ 2>/dev/null
done
python3 - <<'PY'
import json,glob,collections
rows=[]
for f in glob.glob("results/s42_*/*/seed_42/fold*.json"):
    p=f.split("/"); bb=p[1].replace("s42_",""); arm=p[2]; fold=int(p[4][4:-5])
    try: d=json.load(open(f))
    except Exception: continue
    rows.append((bb,arm,fold,d.get("test_macro_f1_at_0.5"),d.get("test_roc_auc")))
if not rows: print("chua co ket qua nao"); raise SystemExit
by=collections.defaultdict(dict)
for bb,arm,fold,f1,auc in rows: by[(bb,arm)][fold]=(f1,auc)
for bb in sorted({r[0] for r in rows}):
    base={fo:v for (b,a),d in by.items() if b==bb and a=="baseline" for fo,v in d.items()}
    print("\n=== %s ===  (baseline fold da co: %s)"%(bb, sorted(base) or "chua co"))
    print("  %-40s %-7s %-8s %-8s %s"%("nhanh","fold","MacroF1","AUC","Δ MacroF1 vs baseline"))
    for (b,a),d in sorted(by.items()):
        if b!=bb: continue
        for fo in sorted(d):
            f1,auc=d[fo]
            delta = "" if a=="baseline" or fo not in base or f1 is None or base[fo][0] is None \
                    else "%+.4f"%(f1-base[fo][0])
            print("  %-40s %-7d %-8s %-8s %s"%(a,fo,
                  "%.4f"%f1 if f1 is not None else "-",
                  "%.4f"%auc if auc is not None else "-", delta))
print("\ntong: %d ket qua fold"%len(rows))

# --- Delta vs `none`: phep so DUNG cho head phu ---
# Phai khop NGUON, OPTIMIZER va FOLD. So sanh hai trung binh la sai; mot fold
# ngoai le tung ganh ca mot ket luan bi rut lai (memory: pair-per-fold-when-baseline-shared).
import re
def parse(arm):
    m=re.match(r"transfer_(none|cwe|latent_bottleneck|latent_proto)_(4cwe|full|com)(_adamw)?$",arm)
    if not m: return None
    return m.group(1), m.group(2), ("adamw" if m.group(3) else "recadam")
cell={}
for bb,arm,fold,f1,auc in rows:
    q=parse(arm)
    if q and f1 is not None: cell[(bb,q[1],q[2],fold,q[0])]=f1
print("\n=== D vs `none` (cung backbone, cung nguon, cung optimizer, cung fold) ===")
print("  %-10s %-5s %-8s %-4s %-20s %8s %8s %9s"%("backbone","nguon","opt","fold","nhanh","none","nhanh","D"))
seen=False
for (bb,src,opt,fold,mode),v in sorted(cell.items()):
    if mode=="none": continue
    base=cell.get((bb,src,opt,fold,"none"))
    if base is None: continue
    seen=True
    print("  %-10s %-5s %-8s %-4d %-20s %8.4f %8.4f %+9.4f"%(bb,src,opt,fold,mode,base,v,v-base))
if not seen: print("  (chua du cap de so)")

# --- Gop qua cac fold: chi in o co n>=2 ---
agg=collections.defaultdict(list)
for (bb,src,opt,fold,mode),v in cell.items():
    if mode=="none": continue
    b=cell.get((bb,src,opt,fold,"none"))
    if b is not None: agg[(bb,src,opt,mode)].append(v-b)
print("\n=== D vs `none` GOP QUA FOLD (chi o co n>=2) ===")
print("  %-10s %-5s %-8s %-20s %3s %9s %9s %8s"%("backbone","nguon","opt","nhanh","n","D tb","min..max","dau"))
any2=False
for k,vals in sorted(agg.items()):
    if len(vals)<2: continue
    any2=True
    bb,src,opt,mode=k
    pos=sum(1 for x in vals if x>0)
    print("  %-10s %-5s %-8s %-20s %3d %+9.4f %+.4f..%+.4f %4d/%d"%(
        bb,src,opt,mode,len(vals),sum(vals)/len(vals),min(vals),max(vals),pos,len(vals)))
if not any2: print("  (chua o nao co >=2 fold)")

print("\n=== RecAdam - AdamW (cung nhanh, cung nguon, cung fold) ===")
print("  %-10s %-5s %-20s %-4s %8s %8s %9s"%("backbone","nguon","nhanh","fold","recadam","adamw","hieu"))
for (bb,src,opt,fold,mode),v in sorted(cell.items()):
    if opt!="recadam": continue
    a=cell.get((bb,src,"adamw",fold,mode))
    if a is None: continue
    print("  %-10s %-5s %-20s %-4d %8.4f %8.4f %+9.4f"%(bb,src,mode,fold,v,a,v-a))
PY
