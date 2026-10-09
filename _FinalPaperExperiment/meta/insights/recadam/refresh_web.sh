#!/bin/bash
# Dựng lại hình/bảng/báo cáo RecAdam rồi tạo bản SVG cho web (bỏ DOCTYPE - artifact không nhận DTD) và doc cho artifact (thêm fig_base).
# Sau đó phiên điều phối đăng lại artifact (files = web/fig_*.svg) và set insights/recadam_contribution từ _artifact_doc.json.
set -e
cd "$(dirname "$0")"
/home/ntat/miniconda3/envs/vdenv/bin/python make_recadam_figures.py
mkdir -p web
for f in fig_*.svg; do python3 -c "
import re, sys; s = open(sys.argv[1], encoding='utf-8').read(); s = re.sub(r'<!DOCTYPE[^>]*>\s*', '', s, flags=re.S)
assert '<!DOCTYPE' not in s and '<!ENTITY' not in s; open('web/' + sys.argv[1], 'w', encoding='utf-8').write(s)" "$f"; done
python3 -c "
import json; d = json.load(open('recadam_contribution.json', encoding='utf-8')); d['fig_base'] = 'insights/recadam/'
json.dump(d, open('_artifact_doc.json', 'w', encoding='utf-8'), ensure_ascii=False); print('doc ok', d['updated'])"
