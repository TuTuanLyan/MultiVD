#!/bin/bash

# Default CWE_ID if not provided
CWE_ID=${1:-089}

echo "Running test for CWE $CWE_ID"

python src/test_mulvulmoe.py \
    --router_pth /drive1/cuongtm/ntat/MulVulMoe/model/routers/mulvulmoe_router_best.pth \
    --expert_pths /drive1/cuongtm/ntat/MulVulMoe/model/experts/mulvul_expert022.pth \
     /drive1/cuongtm/ntat/MulVulMoe/model/experts/mulvul_expert078.pth \
     /drive1/cuongtm/ntat/MulVulMoe/model/experts/mulvul_expert079.pth \
     /drive1/cuongtm/ntat/MulVulMoe/model/experts/mulvul_expert089.pth \
    --test_file /drive1/cuongtm/ntat/MulVulMoe/dataset/sven0802/cwe-${CWE_ID}_test.jsonl \
    --CWE_ID $CWE_ID \
    --top_k 1 \
    --threshold 0.5