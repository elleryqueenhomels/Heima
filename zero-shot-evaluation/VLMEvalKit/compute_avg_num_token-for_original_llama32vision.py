import pandas as pd
import re

from torchtune import config
from omegaconf import OmegaConf


def read_prediction_column(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Extract the 'prediction' column as a list
    prediction_list = df['prediction'].tolist()
    
    return prediction_list

tokenizer_config_dict = {
    "_component_": "torchtune.models.llama3_2_vision.llama3_2_vision_transform",
    "path": "/mnt/localssd/llava-cot-checkpoints/llava-cot-pretrained/Llama-3.2V-11B-cot/original/tokenizer.model",
    "image_size": 560,
    "max_seq_len": 2048
}
tokenizer_config = OmegaConf.create(tokenizer_config_dict)
model_transform = config.instantiate(tokenizer_config)


# todo
model_type = "Llama-3.2-11B-Vision-Instruct"

# todo
output_file = "outputs_temp_8"

# todo
dataset_all = ["MMStar", "MMBench_DEV_EN_V11", "MMVet", "MathVista_MINI", "AI2D_TEST", "HallusionBench"]

match_error_count = 0
for dataset in dataset_all:

    file_path = "/home/xuans/sensei-fs-link/code/efficient-reasoning/efficient-reasoning/zero-shot-evaluation/VLMEvalKit/eval_outputs/" + \
                "{}/{}/{}_{}.xlsx".format(output_file, model_type, model_type, dataset)

    predictions = read_prediction_column(file_path)

    res_num_token = []
    for ele in predictions:
        try:
            current_tokens = model_transform.encode(ele)
            res_num_token.append(len(current_tokens)-2)  # Xuan: remove start and end token
        except:
            # print("Match error in: ", ele)
            match_error_count += 1
            continue

    print("#" * 10 + " Dataset: " + dataset + " " + "#" * 10)
    # print("Number of generarted tokens: ", res_num_token)
    print("   Total Avg:   ", sum(res_num_token) / len(res_num_token))
    print("#" * 10+ " Dataset: " + dataset + " " + "#" * 10)
    print()

print("Match error count: ", match_error_count)