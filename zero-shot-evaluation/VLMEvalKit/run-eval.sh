#!/bin/sh

output_path="eval_outputs/"

datasets="MMVet MathVista_MINI MMStar AI2D_TEST HallusionBench MMBench_DEV_EN_V11"


# model="Llama-3.2-11B-Vision-Instruct"
# model="LLaVA-CoT-Efficient"
model="LLaVA-CoT-Efficient-LORA"

torchrun --nproc-per-node=1 \
          run.py \
          --data $datasets \
          --model $model \
          --work-dir $output_path \
          --reuse

