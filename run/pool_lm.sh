#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POOLS="lm100_n930 lm75_n930 lm50_n930 lm25_n930 lm12_n930" exec bash run/pool1.sh
