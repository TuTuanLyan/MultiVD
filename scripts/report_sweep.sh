#!/usr/bin/env bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - <<'PY'
import json,glob,re
rows=[]
for f in glob.glob("results/s42_*/transfer_none_4cwe_*/seed_42/fold1.json"):
    arm=f.split("/")[2]; bb=f.split("/")[1].replace("s42_","")
    try: d=json.load(open(f))
    except Exception: continue
    if arm.endswith("_ctl"): r=0.0
    else:
        m=re.search(r"_asam_r(\d+)$",arm)
        if not m: continue
        t=m.group(1); r=float(t[0]+"."+t[1:]) if len(t)>1 else float(t)
    rows.append((bb,r,d.get("test_macro_f1_at_0.5"),d.get("test_roc_auc"),d.get("best_epoch")))
if not rows: print("  chua co ket qua quet"); raise SystemExit
print("  %-10s %7s %9s %9s %6s %9s"%("backbone","rho","MacroF1","AUC","ep","D vs rho=0"))
for bb in sorted({r[0] for r in rows}):
    base=next((x[2] for x in rows if x[0]==bb and x[1]==0.0), None)
    for _,r,f1,auc,ep in sorted([x for x in rows if x[0]==bb], key=lambda z:z[1]):
        d = "" if base is None or r==0.0 else "%+.4f"%(f1-base)
        print("  %-10s %7.2f %9.4f %9.4f %6s %9s"%(bb,r,f1,auc,ep,d))
PY
