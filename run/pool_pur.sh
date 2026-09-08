#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POOLS="pur100_n930 pur75_n930 pur50_n930 pur25_n930 pur12_n930 pur100_n465 pur100_n232" exec bash run/pool1.sh
