#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BB=codebert exec bash run/e60b.sh
