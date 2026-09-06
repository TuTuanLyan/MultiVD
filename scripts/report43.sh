#!/usr/bin/env bash
# So ASAM@Phase2 voi nhanh khong-ASAM: cung Phase-1 checkpoint, cung fold, cung
# RecAdam, cung seed. Khac dung mot bien.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - <<'PY'
import json,glob,re,math,collections
def load(f):
    try: return json.load(open(f)).get("test_macro_f1_at_0.5")
    except Exception: return None
asam, base = {}, {}
for f in glob.glob("results/s42_*/*/seed_42/fold*.json"):
    p=f.split("/"); bb=p[1].replace("s42_",""); arm=p[2]; fo=int(p[4][4:-5])
    m=re.match(r"transfer_(none|cwe|latent_bottleneck|latent_proto)_(4cwe|full|com)_asam$",arm)
    if m: asam[(bb,m.group(1),m.group(2),fo)]=load(f); continue
    m=re.match(r"transfer_(none|cwe|latent_bottleneck|latent_proto)_(4cwe|full|com)_ctl$",arm)
    if m: base[(bb,m.group(1),m.group(2),fo)]=load(f)
rows=[(k,base.get(k),v) for k,v in asam.items() if base.get(k) is not None and v is not None]
if not rows:
    print("  chua co o nao du cap de so"); raise SystemExit
print("  So voi DOI CHUNG cung may cung phien (nhanh _ctl), khong phai ket qua 27-28/08.\n")
print("  %-10s %-20s %-5s %4s %9s %9s %9s"%("backbone","nhanh","nguon","fold","doi chung","ASAM","D"))
for (bb,md,src,fo),b,v in sorted(rows):
    print("  %-10s %-20s %-5s %4d %9.4f %9.4f %+9.4f"%(bb,md,src,fo,b,v,v-b))
d=[v-b for _,b,v in rows]; pos=sum(1 for x in d if x>0)
p=min(1.0,2*sum(math.comb(len(d),k) for k in range(pos,len(d)+1))/2**len(d))
print("\n  GOP: n=%d  D tb %+.4f  %d/%d duong  sign p=%.4f"%(len(d),sum(d)/len(d),pos,len(d),p))
g=collections.defaultdict(list)
for (bb,md,src,fo),b,v in rows: g[bb].append(v-b)
for bb,vs in sorted(g.items()):
    print("    %-10s n=%-3d D tb %+.4f  %d/%d duong"%(bb,len(vs),sum(vs)/len(vs),sum(1 for x in vs if x>0),len(vs)))
PY
