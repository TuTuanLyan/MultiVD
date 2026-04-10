#!/bin/bash

python src/train_ex.py \
  --train_files /drive1/cuongtm/ntat/MulVulMoe/dataset/MulVulMoeER/train_078.jsonl \
  --val_files /drive1/cuongtm/ntat/MulVulMoe/dataset/MulVulMoeER/val_078.jsonl \
  --cwe_id 078 \
  --epochs 30 \
  --batch_size 16 \
  --learning_rate 2e-5\
  --aux_weight 0.05 \
  --output_dir "/drive1/cuongtm/ntat/MulVulMoe/model/experts/"