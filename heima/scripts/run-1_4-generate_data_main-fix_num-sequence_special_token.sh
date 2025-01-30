
data_path="/mnt/localssd/llava-cot-dataset/train.jsonl"
image_path="/mnt/localssd/llava-cot-dataset/image_files/"


# compression_ratio_thinking_tokens_summary=2
# compression_ratio_thinking_tokens_caption=2
# compression_ratio_thinking_tokens_reasoning=2

# compression_ratio_thinking_tokens_summary=4
# compression_ratio_thinking_tokens_caption=4
# compression_ratio_thinking_tokens_reasoning=4

# compression_ratio_thinking_tokens_summary=8
# compression_ratio_thinking_tokens_caption=8
# compression_ratio_thinking_tokens_reasoning=8

#compression_ratio_thinking_tokens_summary=16
#compression_ratio_thinking_tokens_caption=16
#compression_ratio_thinking_tokens_reasoning=16

 compression_ratio_thinking_tokens_summary=32
 compression_ratio_thinking_tokens_caption=32
 compression_ratio_thinking_tokens_reasoning=32


train_split_ratio=1.0

python3 -u ../main_python/1_4-organize_dataset-num_thinking_tokens-fix_num-sequence_speical_token.py \
        --compression-ratio-thinking-tokens-summary $compression_ratio_thinking_tokens_summary \
        --compression-ratio-thinking-tokens-caption $compression_ratio_thinking_tokens_caption \
        --compression-ratio-thinking-tokens-reasoning $compression_ratio_thinking_tokens_reasoning \
        --train-split-ratio $train_split_ratio \
        --data-path $data_path \
        --image-path $image_path \
