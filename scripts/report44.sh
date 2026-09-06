#!/usr/bin/env bash
# So hai tap dich: `sven_python_folds_norm` (ro ri ~40%) vs `sven_python_twin`
# (chia theo cum, sach). Cung Phase-1 checkpoint, cung seed, cung optimizer,
# cung fold — doi DUNG MOT BIEN la cach chia fold.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - <<'PY'
import json,glob,re,collections,math
def load(f):
    try: return json.load(open(f)).get("test_macro_f1_at_0.5")
    except Exception: return None
# cell[(tapdich, backbone, nhanh, nguon, opt, fold)] = macroF1
cell={}
for f in glob.glob("results/s42*_*/*/seed_42/fold*.json"):
    p=f.split("/"); run=p[1]
    ds = "twin" if run.startswith("s42tw_") else "norm"
    bb = run.replace("s42tw_","").replace("s42_","")
    fo=int(p[4][4:-5]); arm=p[2]
    m=re.match(r"transfer_(none|cwe|latent_bottleneck|latent_proto)_(4cwe|full|com)(_adamw)?$",arm)
    if not m: continue
    v=load(f)
    if v is not None: cell[(ds,bb,m.group(1),m.group(2),"adamw" if m.group(3) else "recadam",fo)]=v
def sp(pos,n): return min(1.0,2*sum(math.comb(n,k) for k in range(pos,n+1))/2**n)

for ds in ("norm","twin"):
    ks=[k for k in cell if k[0]==ds]
    if not ks: continue
    g=collections.defaultdict(list)
    for k in ks:
        if k[2]=="none": continue
        b=cell.get((ds,k[1],"none",k[3],k[4],k[5]))
        if b is not None: g[(k[2],k[3],k[4])].append(cell[k]-b)
    print("\n===== TAP DICH: %s  (%d o) ====="%("SACH (twin)" if ds=="twin" else "ro ri (norm)", len(ks)))
    print("  %-20s %-5s %-8s %3s %9s %8s %8s"%("nhanh","nguon","opt","n","D tb","dau","sign p"))
    rows=[]
    for (m,s,o),vs in g.items():
        if len(vs)<3: continue
        pos=sum(1 for x in vs if x>0)
        rows.append((sum(vs)/len(vs),m,s,o,len(vs),pos,sp(pos,len(vs))))
    for mean,m,s,o,n,pos,p in sorted(rows,reverse=True)[:8]:
        star=" <<<" if p<0.05 else ""
        print("  %-20s %-5s %-8s %3d %+9.4f %4d/%-3d %8.4f%s"%(m,s,o,n,mean,pos,n,p,star))

# So truc tiep o thang tren hai tap dich
print("\n===== O THANG (latent_bottleneck + AdamW + 4cwe) tren hai tap dich =====")
print("  %-6s %-10s %4s %9s %9s %9s"%("tapdich","backbone","fold","none","bottleneck","D"))
for ds in ("norm","twin"):
    ds_all=[]
    for bb in ("codebert","t5p","unixcoder"):
        for fo in (1,2,3,4,5):
            b=cell.get((ds,bb,"none","4cwe","adamw",fo)); v=cell.get((ds,bb,"latent_bottleneck","4cwe","adamw",fo))
            if b is None or v is None: continue
            ds_all.append(v-b)
            print("  %-6s %-10s %4d %9.4f %9.4f %+9.4f"%(ds,bb,fo,b,v,v-b))
    if ds_all:
        pos=sum(1 for x in ds_all if x>0)
        print("  -> %s: n=%d  D tb %+.4f  %d/%d duong  sign p=%.4f\n"%(ds,len(ds_all),sum(ds_all)/len(ds_all),pos,len(ds_all),sp(pos,len(ds_all))))
PY
