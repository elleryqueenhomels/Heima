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


# model_type = "LLaVA-CoT-Efficient"
model_type = "LLaVA-CoT-Efficient-LORA"

output_file = "outputs_1"

dataset_all = ["MMStar", "MMBench_DEV_EN_V11", "MMVet", "MathVista_MINI", "AI2D_TEST", "HallusionBench"]

match_error_count = 0
for dataset in dataset_all:

    file_path = "/home/xuans/sensei-fs-link/code/efficient-reasoning/efficient-reasoning/zero-shot-evaluation/VLMEvalKit/eval_outputs/" + \
                "{}/{}/{}_{}.xlsx".format(output_file, model_type, model_type, dataset)

    pattern_num_token = r"<num_tokens>(.*?)<num_tokens>"
    pattern_conclusion = r"<CONCLUSION>[\s\S]*?</CONCLUSION>"

    predictions = read_prediction_column(file_path)

    res_num_token = []
    res_before_conclusion = []
    res_conclusion = []
    for ele in predictions:
        try:
            match_num_token = re.search(pattern_num_token, ele)
            match_conclusion = re.search(pattern_conclusion, ele)
        except:
            # print("Match error in: ", ele)
            match_error_count += 1
            continue

        if match_num_token and match_conclusion:
            current_num_token = int(match_num_token.group(1))
            res_num_token.append(current_num_token)

            conclusion_text = match_conclusion.group()
            conclusion_token = model_transform.encode(conclusion_text)
            res_before_conclusion.append(current_num_token - (len(conclusion_token)-2))  # Xuan: remove start and end token
            res_conclusion.append(len(conclusion_token)-2)
        else:
            # print("No match found for: ", ele)
            res_num_token.append(0)
            res_before_conclusion.append(0)
            res_conclusion.append(0)

    print("#" * 10 + " Dataset: " + dataset + " " + "#" * 10)
    # print("Number of generarted tokens: ", res_num_token)
    print("   Total Avg:   ", sum(res_num_token) / len(res_num_token))
    # print("Number of generarted tokens before conclusion: ", res_before_conclusion)
    print("3 Stages Avg:   ", sum(res_before_conclusion) / len(res_before_conclusion))
    # print("Number of generarted tokens for conclusion: ", res_conclusion)
    print("Conclusion Avg: ", sum(res_conclusion) / len(res_conclusion))
    print("#" * 10+ " Dataset: " + dataset + " " + "#" * 10)
    print()

print("Match error count: ", match_error_count)