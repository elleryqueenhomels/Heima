
data_path="/mnt/localssd/llava-cot-dataset/train.jsonl"
image_path="/mnt/localssd/llava-cot-dataset/image_files/"

# compression_ratio_thinking_tokens_summary=0.9
# compression_ratio_thinking_tokens_caption=0.9
# compression_ratio_thinking_tokens_reasoning=0.9

# compression_ratio_thinking_tokens_summary=0.8
# compression_ratio_thinking_tokens_caption=0.8
# compression_ratio_thinking_tokens_reasoning=0.8

# compression_ratio_thinking_tokens_summary=0.7
# compression_ratio_thinking_tokens_caption=0.7
# compression_ratio_thinking_tokens_reasoning=0.7

# compression_ratio_thinking_tokens_summary=0.6
# compression_ratio_thinking_tokens_caption=0.6
# compression_ratio_thinking_tokens_reasoning=0.6

# compression_ratio_thinking_tokens_summary=0.5
# compression_ratio_thinking_tokens_caption=0.5
# compression_ratio_thinking_tokens_reasoning=0.5

# compression_ratio_thinking_tokens_summary=0.4
# compression_ratio_thinking_tokens_caption=0.4
# compression_ratio_thinking_tokens_reasoning=0.4

#compression_ratio_thinking_tokens_summary=0.3
#compression_ratio_thinking_tokens_caption=0.3
#compression_ratio_thinking_tokens_reasoning=0.3

# compression_ratio_thinking_tokens_summary=0.2
# compression_ratio_thinking_tokens_caption=0.2
# compression_ratio_thinking_tokens_reasoning=0.2

 compression_ratio_thinking_tokens_summary=0.1
 compression_ratio_thinking_tokens_caption=0.1
 compression_ratio_thinking_tokens_reasoning=0.1





train_split_ratio=1.0

python3 -u ../main_python/1_3-organize_dataset-num_thinking_tokens-adaptive-sequence_special_token.py \
        --compression-ratio-thinking-tokens-summary $compression_ratio_thinking_tokens_summary \
        --compression-ratio-thinking-tokens-caption $compression_ratio_thinking_tokens_caption \
        --compression-ratio-thinking-tokens-reasoning $compression_ratio_thinking_tokens_reasoning \
        --train-split-ratio $train_split_ratio \
        --data-path $data_path \
        --image-path $image_path \
