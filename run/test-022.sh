#!/bin/bash

python src/evaluate.py \
  --test_files /drive1/cuongtm/ntat/MulVulMoe/dataset/MulVulEx/test_022.jsonl \
  --model_path /drive1/cuongtm/ntat/MulVulMoe/model/experts/mulvul_expert_022_sl_best.pth \
  --cwe_id 022 \
  --batch_size 32 \
  --pretrained_model microsoft/codebert-base