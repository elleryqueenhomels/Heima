
data_path="/mnt/localssd/llava-cot-dataset/train.jsonl"
image_path="/mnt/localssd/llava-cot-dataset/image_files/"


compression_ratio_thinking_tokens_summary=1
compression_ratio_thinking_tokens_caption=1
compression_ratio_thinking_tokens_reasoning=1



# train_split_ratio=0.98
train_split_ratio=1

python3 -u ../main_python/1_2-organize_dataset-pure_llm_decoder-num_thinking_tokens-fix_num.py \
        --compression-ratio-thinking-tokens-summary $compression_ratio_thinking_tokens_summary \
        --compression-ratio-thinking-tokens-caption $compression_ratio_thinking_tokens_caption \
        --compression-ratio-thinking-tokens-reasoning $compression_ratio_thinking_tokens_reasoning \
        --train-split-ratio $train_split_ratio \
        --data-path $data_path \
        --image-path $image_path \
