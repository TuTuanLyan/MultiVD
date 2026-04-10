#!/bin/bash

python src/train_router.py \
    --train_files /drive1/cuongtm/ntat/MulVulMoe/dataset/MulVulMoeER/train_vul_mapped.jsonl \
    --num_experts 4 \
    --batch_size 16\
    --epochs 20 \
    --output_dir /drive1/cuongtm/ntat/MulVulMoe/model/routers