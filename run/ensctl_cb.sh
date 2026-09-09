#!/usr/bin/env bash
# ENSCTL_CB — ban codebert cua run/ensctl.sh. Xem chu thich o file do.
# BB lay tu hyperparameters cua baseline seed 42 trong cay poolcb_codebert:
#   model_name = microsoft/codebert-base , pooling = cls
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=poolcb BB='codebert=microsoft/codebert-base:cls' exec bash run/ensctl.sh
