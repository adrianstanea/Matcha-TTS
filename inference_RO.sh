#!/bin/bash

# matcha-base-1000.ckpt
bas_files=(
    # matcha-bas-10_15.ckpt
    # matcha-bas-10_50.ckpt
    # matcha-bas-10_100.ckpt

    matcha-bas-950_15.ckpt
    matcha-bas-950_50.ckpt
    matcha-bas-950_100.ckpt
)
for pt_file in ${bas_files[@]}; do
    echo pt_file: $pt_file
    python3 inference_RO.py \
        --file /workspace/local/evaluation/eval_bas_42.txt \
        --checkpoint $pt_file
done

# sgs_files=(
#     matcha-sgs-10_15.ckpt
#     matcha-sgs-10_50.ckpt
#     matcha-sgs-10_100.ckpt

#     matcha-sgs-950_15.ckpt
#     matcha-sgs-950_50.ckpt
#     matcha-sgs-950_100.ckpt
# )
# for pt_file in ${sgs_files[@]}; do
#     echo pt_file: $pt_file
#     python3 inference_RO.py \
#         --file /workspace/local/evaluation/eval_sgs_42.txt \
#         --checkpoint $pt_file
# done

# Run baseline
# pt_file="/workspace/local/checkpoints/matcha-base-1000.ckpt"
# python3 inference_RO.py \
#     --file /workspace/local/evaluation/eval_bas_42.txt \
#     --checkpoint matcha-base-1000.ckpt