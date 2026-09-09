#!/usr/bin/env python3
"""pick_vast_offer.py — chon MOT offer A4000 tai THOI DIEM CAN, khong chon truoc.

Nguoi dung 10/09: "Luc nao can thi tim nhe, pre truoc thi no cung chua chac con khi can."

Tieu chi, theo dung thu tu:
  1. RTX A4000, 1 GPU, dang cho thue, gia <= tran (mac dinh $0.080/h — nguoi dung dat).
  2. Do tin cay >= 0.95 (may chap chon lam hong ca dem thi re cung thanh dat).
  3. Xep hang: tin cay giam dan -> XUNG NHIP MOT NHAN giam dan -> mang nhanh -> gia re.

Vi sao uu tien xung nhip chu khong phai so nhan: moi lenh cua khoi nay chay `--num_workers 0`,
tuc dataloader DON LUONG. Cai quyet dinh la toc do mot nhan. Mot i7 5.2GHz nhanh hon EPYC
2.9GHz cho viec tokenize/nap du lieu, du EPYC nhieu nhan hon.

In ra dong dau tien la ID de script vo dung; moi thu khac ra stderr.
"""
import json, subprocess, sys, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dph", type=float, default=0.080)
    ap.add_argument("--min-rel", type=float, default=0.95)
    ap.add_argument("--gpu", default="RTX_A4000")
    a = ap.parse_args()
    q = f"gpu_name={a.gpu} num_gpus=1 rentable=true dph<{a.max_dph+0.0005:.4f}"
    try:
        out = subprocess.run(["vastai","search","offers",q,"-o","dph+","--raw"],
                             capture_output=True, text=True, timeout=120)
        offers = json.loads(out.stdout)
    except Exception as e:
        print(f"!! khong tra duoc offer: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(2)
    ok = [o for o in offers
          if o.get("dph_total", 9) <= a.max_dph and o.get("reliability2", 0) >= a.min_rel]
    print(f"# {len(offers)} offer <= ${a.max_dph:.3f}/h, {len(ok)} dat do tin cay >= {a.min_rel}",
          file=sys.stderr)
    if not ok:
        print(f"!! KHONG CO offer nao dat tieu chi (tran ${a.max_dph:.3f}/h, tin cay >= {a.min_rel}).",
              file=sys.stderr)
        print("!! KHONG THUE. Bao nguoi dung de quyet: noi tran gia hay doi loai GPU.", file=sys.stderr)
        sys.exit(1)
    ok.sort(key=lambda o: (-o.get("reliability2",0), -o.get("cpu_ghz",0),
                           -o.get("inet_down",0), o.get("dph_total",9)))
    print(f"# {'id':>10}{'$/h':>8}{'GHz':>6}{'vCPU':>6}{'RAM':>6}{'down':>7}{'tin cay':>9}  CPU",
          file=sys.stderr)
    for o in ok[:5]:
        print(f"# {o['id']:>10}{o.get('dph_total',0):>8.4f}{o.get('cpu_ghz',0):>6.1f}"
              f"{o.get('cpu_cores_effective',0):>6.0f}{o.get('cpu_ram',0)/1024:>5.0f}G"
              f"{o.get('inet_down',0):>6.0f}M{o.get('reliability2',0):>9.3f}  "
              f"{(o.get('cpu_name') or '?')[:32]}", file=sys.stderr)
    b = ok[0]
    print(f"# => CHON {b['id']} (${b.get('dph_total',0):.4f}/h, tin cay {b.get('reliability2',0):.3f}, "
          f"{b.get('cpu_ghz',0):.1f}GHz)", file=sys.stderr)
    print(b["id"])

if __name__ == "__main__": main()
