import json, glob, sys, statistics, os
from math import comb
src, root = sys.argv[1], sys.argv[2]
LT={"l0p01":0.01,"l0p05":0.05,"l0p2":0.2}
RT={"r0":0.0,"r0p05":0.05,"r0p1":0.1,"r0p2":0.2,"r0p5":0.5,"r1p0":1.0,"r2p0":2.0}
cells={}
for p in glob.glob(os.path.join(root,f"results/sw_t5p/transfer_latent_bottleneck_{src}_*/seed_42/fold*.json")):
    tag=p.split("/")[-3].replace(f"transfer_latent_bottleneck_{src}_","")
    lt,rt=tag.rsplit("_",1)
    if lt in LT and rt in RT:
        d=json.load(open(p)); cells[(LT[lt],RT[rt],d["fold"])]=d["test_macro_f1_at_valcal"]
def sp(pos,n):
    if n==0: return 1.0
    k=min(pos,n-pos); return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)
print(f"  {src}: goc rho>0 so voi rho=0, GOP theo tung lambda")
print("  %-8s %4s %10s %9s %9s %s"%("lambda","n","D trung binh","cung dau","p","bien do"))
for lam in sorted({k[0] for k in cells}):
    ds=[cells[(lam,r,f)]-cells[(lam,0.0,f)] for (l,r,f) in cells
        if l==lam and r>0 and (lam,0.0,f) in cells]
    if not ds: continue
    pos=sum(1 for x in ds if x>0)
    print("  %-8s %4d %+10.4f %9s %9.4f  %+.4f … %+.4f"%(
        lam,len(ds),statistics.mean(ds),f"{pos}/{len(ds)}",sp(pos,len(ds)),min(ds),max(ds)))
