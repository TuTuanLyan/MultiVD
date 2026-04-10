#!/bin/bash

python src/evaluate.py \
  --test_files /drive1/cuongtm/ntat/MulVulMoe/dataset/sven0802/cwe-079_test.jsonl \
  --model_path /drive1/cuongtm/ntat/MulVulMoe/model/experts/mulvul_expert079.pth \
  --cwe_id 079 \
  --batch_size 16 \
  --pretrained_model microsoft/codebert-base