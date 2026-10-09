#!/usr/bin/env bash
# Dựng toàn bộ dữ liệu từ dữ liệu thô rồi kiểm md5 (chỉ cần CPU). Chạy từ thư mục refactorMWG/:
#   PRIMEVUL_RAW=<thư mục primevul_*.jsonl> CLEANVUL_DIR=<thư mục vulnerability_score_{3,4}.csv> bash scripts/build_data.sh
set -euo pipefail
: "${PRIMEVUL_RAW:?đặt PRIMEVUL_RAW}" "${CLEANVUL_DIR:?đặt CLEANVUL_DIR}"
PY=${PY:-python}
WORKERS=${WORKERS:-4}
CWE_XML=data/cwe_spec/cwec_v4.20.xml

if [ ! -f "$CWE_XML" ]; then
  mkdir -p data/cwe_spec
  (cd data/cwe_spec && curl -sSfLO https://cwe.mitre.org/data/xml/cwec_v4.20.xml.zip && unzip -o -q cwec_v4.20.xml.zip)
fi
echo "efd2581cfe58dd678965c8e582945c48  $CWE_XML" | md5sum -c --quiet

$PY dataset/build_sources.py --primevul_raw "$PRIMEVUL_RAW" --cleanvul_dir "$CLEANVUL_DIR" --cwe_xml "$CWE_XML" \
    --out data/sources_v4 --workers "$WORKERS"                       # nguồn: xoá comment + luật T0–T4
$PY dataset/build_pools.py --src_dir data/sources_v4 --out_root data   # pool Pha 1 (có pool SOTA mwsrc_v4jsCjv_rand)
$PY dataset/make_target.py data/sven_python_folds_norm data/sven_python_folds_v4   # đích SVEN đã xoá comment
rm -rf data/clean_sources_v4
$PY dataset/export_clean_sources.py data/clean_sources_v4             # nguồn chuẩn để chia sẻ (by_source / merged)

(cd data && md5sum -c --quiet ../dataset/EXPECTED.md5) && echo "md5: $(wc -l < dataset/EXPECTED.md5) file khớp"
