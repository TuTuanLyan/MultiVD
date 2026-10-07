"""In TOÀN BỘ tham số hiệu lực (kể cả mặc định) của một trainer cho đúng dòng lệnh, rồi thoát: chặn ở parse_args
bên trong chính main() của trainer — không dựng model, không đụng GPU.
    CUDA_VISIBLE_DEVICES= python effective_args.py <src_dir> <trainer.py> [cờ...]"""
import argparse, json, os, runpy, sys

src_dir, trainer = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.abspath(src_dir))
_parse = argparse.ArgumentParser.parse_args


def _dump(self, *a, **k):
    ns = _parse(self, *a, **k)
    print("EFFECTIVE_ARGS " + json.dumps(vars(ns), ensure_ascii=False, sort_keys=True, default=str))
    sys.exit(0)


argparse.ArgumentParser.parse_args = _dump
sys.argv = [trainer] + sys.argv[3:]
runpy.run_path(os.path.join(src_dir, trainer), run_name="__main__")
