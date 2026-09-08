#!/usr/bin/env bash
# Vo boc de worklist goi duoc: worklist chi truyen FOLD_LIST/SOURCES_LIST, khong truyen BB.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BB=t5p exec bash run/e60b.sh
