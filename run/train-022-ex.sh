#!/bin/bash

python src/train_ex.py \
  --train_files /drive1/cuongtm/ntat/MulVulMoe/dataset/MulVulEx/train_022.jsonl \
  --val_files /drive1/cuongtm/ntat/MulVulMoe/dataset/MulVulEx/val_022.jsonl \
  --cwe_id 022 \
  --epochs 30 \
  --batch_size 16 \
  --learning_rate 1e-5\
  --aux_weight 0.05 \
  --output_dir "/drive1/cuongtm/ntat/MulVulMoe/model/experts"