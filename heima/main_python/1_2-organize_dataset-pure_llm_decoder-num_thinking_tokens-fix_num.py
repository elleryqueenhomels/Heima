import json
import os
import re
import copy
import random
random.seed(42)
import argparse
from torchtune import config
from omegaconf import OmegaConf


def load_from_json(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ignore empty lines
                data.append(json.loads(line))
    return data


def extract_sections(text):
    # Define a pattern to match the sections
    pattern = r"<(SUMMARY|CAPTION|REASONING|CONCLUSION)>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)

    # Convert matches into a dictionary
    extracted_content = {section: content.strip() for section, content in matches}
    return extracted_content


def get_middle_sublist(input_list, start_sublist, end_sublist):
        # Find the indices of the start and end sublists
        try:
            start_index = next(i for i in range(len(input_list)) if input_list[i:i+len(start_sublist)] == start_sublist)
            end_index = next(i for i in range(start_index + len(start_sublist), len(input_list)) if input_list[i:i+len(end_sublist)] == end_sublist)
            
            # Indices for the middle sublist
            middle_start_index = start_index + len(start_sublist)
            middle_end_index = end_index
            
            return middle_start_index, middle_end_index
        except StopIteration:
            return None, None
        
        
def get_args():
    parser = argparse.ArgumentParser(description="Script for configuring dataset paths and training parameters.")

    # Argument for number of thinking tokens
    parser.add_argument(
        "--compression-ratio-thinking-tokens-summary",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--compression-ratio-thinking-tokens-caption",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--compression-ratio-thinking-tokens-reasoning",
        type=int,
        default=10,
    )

    # Argument for train split ratio
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=0.9,
        help="Train split ratio for dataset."
    )

    # Argument for data path
    parser.add_argument(
        "--data-path",
        type=str,
        default="/mnt/localssd/llava-cot-dataset/train.jsonl",
        help="Path to the training data file."
    )

    # Argument for image path
    parser.add_argument(
        "--image-path",
        type=str,
        default="/mnt/localssd/llava-cot-dataset/image_files/",
        help="Path to the image files."
    )

    args = parser.parse_args()
    return args


args = get_args()
compression_ratio_thinking_tokens_summary = args.compression_ratio_thinking_tokens_summary
compression_ratio_thinking_tokens_caption = args.compression_ratio_thinking_tokens_caption
compression_ratio_thinking_tokens_reasoning = args.compression_ratio_thinking_tokens_reasoning
train_split_ratio = args.train_split_ratio
data_path = args.data_path
image_path = args.image_path


decode_summary_question_template = [
    '{question}\nCan you provide the details of thinking progress ' + '<THINKING_OF_SUMMARY>' + ' for summarizing the given question?',
    'For the question "{question}", how does the thinking progress ' + '<THINKING_OF_SUMMARY>' + ' unfold during the summarization?',
    'Explain the detailed thinking process ' + '<THINKING_OF_SUMMARY>' + ' to summarize this question: {question}',
    'What is the step-by-step thinking progress ' + '<THINKING_OF_SUMMARY>' + ' for summarizing the provided question: {question}?',
    'Could you elaborate on the thinking progress ' + '<THINKING_OF_SUMMARY>' + ' used to generate the summary of the question: {question}?',
    'How is the thinking progress ' + '<THINKING_OF_SUMMARY>' + ' applied to summarize this question: {question}?',
    '{question}\nDescribe the reasoning behind the summarization process using ' + '<THINKING_OF_SUMMARY>' + '.',
    'Can you outline the thought process ' + '<THINKING_OF_SUMMARY>' + ' for crafting the summary of this question: {question}?',
    'Could you provide the logical flow ' + '<THINKING_OF_SUMMARY>' + ' for summarizing the question: {question}?',
    'What thinking steps ' + '<THINKING_OF_SUMMARY>' + ' were followed to create the summary of this question: {question}?',
]
decode_summary_answer_template = [
    'The thinking progress for the summary of the given question is: {summary}',
    'Here is the thinking progress for summarizing the question: {summary}',
    'The reasoning steps for summarization are as follows: {summary}',
    'To summarize the question, the thinking progress can be described as: {summary}',
    'The process of summarization is as follows: {summary}',
    'The logical flow for summarization includes: {summary}',
    'Below is the sequence of thought used for the summary: {summary}',
    'The cognitive steps for creating the summary are: {summary}',
    'Here is a breakdown of the thinking process: {summary}',
    'The thinking sequence leading to the summary is: {summary}',
]

decode_caption_question_template = [
    '{question}\nCan you provide the thinking progress ' + '<THINKING_OF_CAPTION>' + ' for the caption of the given question?',
    'What is the thinking progress ' + '<THINKING_OF_CAPTION>' + ' involved in crafting the caption for the question: {question}?',
    'For the question "{question}", explain the step-by-step thinking process ' + '<THINKING_OF_CAPTION>' + ' that leads to the caption.',
    'How does the thinking progress ' + '<THINKING_OF_CAPTION>' + ' assist in generating the caption for this question: {question}?',
    'Could you detail the thinking progress ' + '<THINKING_OF_CAPTION>' + ' used to explain the caption for: {question}?',
    '{question}\nCan you outline the reasoning steps ' + '<THINKING_OF_CAPTION>' + ' for constructing the caption?',
    'Describe the thought process ' + '<THINKING_OF_CAPTION>' + ' that defines the caption for this question: {question}.',
    'How is the logical flow ' + '<THINKING_OF_CAPTION>' + ' employed to form the caption of this question: {question}?',
    'What insights ' + '<THINKING_OF_CAPTION>' + ' were considered for captioning this question: {question}?',
    'Provide a step-by-step explanation of the thinking progress ' + '<THINKING_OF_CAPTION>' + ' for the caption of this question: {question}.',
]
decode_caption_answer_template = [
    'The thinking progress for the caption of the given question can be explained as follows: {caption}',
    'Here is the thinking progress for generating the caption: {caption}',
    'The step-by-step thinking process for the caption can be described as: {caption}',
    'The caption is derived from the following thinking progress: {caption}',
    'The thinking sequence for the caption of the question is: {caption}',
    'Below is the logical reasoning applied to generate the caption: {caption}',
    'The following steps were used to explain the caption: {caption}',
    'The reasoning process leading to the caption is: {caption}',
    'The thinking outline for forming the caption is as follows: {caption}',
    'This is the detailed process for caption generation: {caption}',
]

decode_reasoning_question_template = [
    '{question}\nCan you provide the thinking progress ' + '<THINKING_OF_REASONING>' + ' for the reasoning of the given question?',
    'Explain how the thinking progress ' + '<THINKING_OF_REASONING>' + ' unfolds during reasoning for the question: {question}?',
    'What steps are involved in the thinking progress ' + '<THINKING_OF_REASONING>' + ' for reasoning about this question: {question}?',
    'Could you describe the reasoning thinking process ' + '<THINKING_OF_REASONING>' + ' used to answer the question: {question}?',
    'For the question "{question}", detail the thinking progress ' + '<THINKING_OF_REASONING>' + ' used in reasoning.',
    '{question}\nProvide the logical breakdown ' + '<THINKING_OF_REASONING>' + ' for reasoning about this question.',
    'How is the thinking progress ' + '<THINKING_OF_REASONING>' + ' utilized to derive reasoning for this question: {question}?',
    'What reasoning flow ' + '<THINKING_OF_REASONING>' + ' was employed to explain this question: {question}?',
    'Explain the sequence ' + '<THINKING_OF_REASONING>' + ' of cognitive steps for reasoning about this question: {question}.',
    'What process ' + '<THINKING_OF_REASONING>' + ' explains the reasoning for the provided question: {question}?',
]
decode_reasoning_answer_template = [
    'The thinking progress for the reasoning of the given question is illustrated as follows: {reasoning}',
    'Here is how the thinking progress applies to reasoning about the question: {reasoning}',
    'The step-by-step reasoning process can be detailed as follows: {reasoning}',
    'The reasoning process, is as follows: {reasoning}',
    'The thinking sequence for reasoning about the question is: {reasoning}',
    'Here’s the logical flow explaining the reasoning: {reasoning}',
    'The following cognitive process was used for reasoning: {reasoning}',
    'Below is the detailed reasoning flow for the question: {reasoning}',
    'The thinking outline for reasoning can be described as: {reasoning}',
    'The thought process leading to the reasoning is: {reasoning}',
]

tokenizer_config_dict = {
    "_component_": "torchtune.models.llama3_2_vision.llama3_2_vision_transform",
    "path": "/mnt/localssd/llava-cot-checkpoints/llava-cot-pretrained/Llama-3.2V-11B-cot/original/tokenizer.model",
    "image_size": 560,
    "max_seq_len": 8192
}
tokenizer_config = OmegaConf.create(tokenizer_config_dict)
model_transform = config.instantiate(tokenizer_config)

total_data = load_from_json(data_path)
thinking_data = []
decode_thinking_data = []
both_data = []
for idx, whole_sample in enumerate(total_data):
    if (idx+1) % 100 == 0:
        print(f"Processing {idx+1} samples")
    
    # check if image exists
    if not os.path.exists(image_path + whole_sample['image']):
        continue

    for i in range(0, len(whole_sample['conversations']), 2):
        sample = {
            "id": whole_sample['id']+f"-{i+1}",
            # "image": whole_sample['image'],
            "conversations": whole_sample['conversations'][i:i+2],
        }
        current_question = sample['conversations'][0]['value']

        current_answer = sample['conversations'][1]['value']
        current_sections = extract_sections(current_answer)
        try:
            current_summary = current_sections['SUMMARY']
            current_caption = current_sections['CAPTION']
            current_reasoning = current_sections['REASONING']
        except:
            # there is no summary, caption, reasoning in the answer
            continue
        
        thinking_sample = copy.deepcopy(sample)

        length_current_num_summary_tokens = compression_ratio_thinking_tokens_summary
        length_current_num_caption_tokens = compression_ratio_thinking_tokens_caption
        length_current_num_reasoning_tokens = compression_ratio_thinking_tokens_reasoning
        
        thinking_answer = current_answer.replace(
            current_summary, "<THINKING_OF_SUMMARY>"*length_current_num_summary_tokens
        ).replace(
            current_caption, "<THINKING_OF_CAPTION>"*length_current_num_caption_tokens
        ).replace(
            current_reasoning, "<THINKING_OF_REASONING>"*length_current_num_reasoning_tokens
        )
        thinking_sample['conversations'][1]['value'] = thinking_answer

        # we find there is data contains more than one <THINKING_OF_SUMMARY>, <THINKING_OF_CAPTION>, <THINKING_OF_REASONING>
        #  thus, we remove those data
        count_summary = thinking_answer.count("<THINKING_OF_SUMMARY>")
        count_caption = thinking_answer.count("<THINKING_OF_CAPTION>")
        count_reasoning = thinking_answer.count("<THINKING_OF_REASONING>")
        if count_summary != length_current_num_summary_tokens \
            or count_caption != length_current_num_caption_tokens \
            or count_reasoning != length_current_num_reasoning_tokens:
            continue

        # if idx % 2 == 0:    # learn from llava
        #     thinking_question = "<image>" + current_question
        # else:
        #     thinking_question = current_question + "<image>"
        # thinking_question = thinking_question + " "
        # thinking_sample['conversations'][0]['value'] = thinking_question
        # thinking_data.append(thinking_sample)

        both_sample = copy.deepcopy(thinking_sample)

        both_sample['conversations_for_eval'] = copy.deepcopy(thinking_sample['conversations'])
        both_sample['conversations_for_eval'][1]['value'] = ""
        
        decode_summary_answer = random.choice(decode_summary_answer_template).format(summary=current_summary)
        decode_summary_question = random.choice(decode_summary_question_template).format(question=current_question)

        decode_summary_question = decode_summary_question.replace("<THINKING_OF_SUMMARY>", "<THINKING_OF_SUMMARY>"*length_current_num_summary_tokens)
        # if idx % 2 == 0:    # learn from llava
        #     decode_summary_question = "<image>" + decode_summary_question
        # else:
        #     decode_summary_question = decode_summary_question + "<image>"
        # decode_summary_question = decode_summary_question + " "
        decode_sample_summary = copy.deepcopy(sample)
        decode_sample_summary['conversations'][1]['value'] = decode_summary_answer
        decode_sample_summary['conversations'][0]['value'] = decode_summary_question
        decode_sample_summary['id'] = sample['id'] + "-summary"
        # decode_thinking_data.append(decode_sample_summary)

        both_sample['conversations_summary'] = decode_sample_summary['conversations']
        both_sample['conversations_summary_for_eval'] = copy.deepcopy(decode_sample_summary['conversations'])
        both_sample['conversations_summary_for_eval'][1]['value'] = ""
        both_sample['conversations_summary_for_pure_llm'] = copy.deepcopy(decode_sample_summary['conversations'])
        # both_sample['conversations_summary_for_pure_llm'][0]['value'] = both_sample['conversations_summary_for_pure_llm'][0]['value'].replace("<image>", "")

        decode_caption_answer = random.choice(decode_caption_answer_template).format(caption=current_caption)
        decode_caption_question = random.choice(decode_caption_question_template).format(question=current_question)

        decode_caption_question = decode_caption_question.replace("<THINKING_OF_CAPTION>", "<THINKING_OF_CAPTION>"*length_current_num_caption_tokens)
        # if idx % 2 == 0:    # learn from llava
        #     decode_caption_question = "<image>" + decode_caption_question
        # else:
        #     decode_caption_question = decode_caption_question + "<image>"
        # decode_caption_question = decode_caption_question + " "
        decode_sample_caption = copy.deepcopy(sample)
        decode_sample_caption['conversations'][1]['value'] = decode_caption_answer
        decode_sample_caption['conversations'][0]['value'] = decode_caption_question
        decode_sample_caption['id'] = sample['id'] + "-caption"
        # decode_thinking_data.append(decode_sample_caption)

        both_sample['conversations_caption'] = decode_sample_caption['conversations']
        both_sample['conversations_caption_for_eval'] = copy.deepcopy(decode_sample_caption['conversations'])
        both_sample['conversations_caption_for_eval'][1]['value'] = ""
        both_sample['conversations_caption_for_pure_llm'] = copy.deepcopy(decode_sample_caption['conversations'])
        # both_sample['conversations_caption_for_pure_llm'][0]['value'] = both_sample['conversations_caption_for_pure_llm'][0]['value'].replace("<image>", "")

        decode_reasoning_answer = random.choice(decode_reasoning_answer_template).format(reasoning=current_reasoning)
        decode_reasoning_question = random.choice(decode_reasoning_question_template).format(question=current_question)

        decode_reasoning_question = decode_reasoning_question.replace("<THINKING_OF_REASONING>", "<THINKING_OF_REASONING>"*length_current_num_reasoning_tokens)
        # if idx % 2 == 0:    # learn from llava
        #     decode_reasoning_question = "<image>" + decode_reasoning_question
        # else:
        #     decode_reasoning_question = decode_reasoning_question + "<image>"
        # decode_reasoning_question = decode_reasoning_question + " "
        decode_sample_reasoning = copy.deepcopy(sample)
        decode_sample_reasoning['conversations'][1]['value'] = decode_reasoning_answer
        decode_sample_reasoning['conversations'][0]['value'] = decode_reasoning_question
        decode_sample_reasoning['id'] = sample['id'] + "-reasoning"
        # decode_thinking_data.append(decode_sample_reasoning)

        both_sample['conversations_reasoning'] = decode_sample_reasoning['conversations']
        both_sample['conversations_reasoning_for_eval'] = copy.deepcopy(decode_sample_reasoning['conversations'])
        both_sample['conversations_reasoning_for_eval'][1]['value'] = ""
        both_sample['conversations_reasoning_for_pure_llm'] = copy.deepcopy(decode_sample_reasoning['conversations'])
        # both_sample['conversations_reasoning_for_pure_llm'][0]['value'] = both_sample['conversations_reasoning_for_pure_llm'][0]['value'].replace("<image>", "")
        
        both_sample['conversations_original'] = copy.deepcopy(sample['conversations'])

        both_sample['conversations_original_with_image'] = copy.deepcopy(sample['conversations'])
        if idx % 2 == 0:
            both_sample['conversations_original_with_image'][0]['value'] = "<image>" + both_sample['conversations_original_with_image'][0]['value']
        else:
            both_sample['conversations_original_with_image'][0]['value'] = both_sample['conversations_original_with_image'][0]['value'] + "<image>"
        original_answer = model_transform.encode(both_sample['conversations_original_with_image'][1]['value'])
        summary_start_1, summary_end_1 = get_middle_sublist(
            original_answer,
            [19389, 2864, 49970, 29], 
            [524, 28477, 49970, 1363]
        )
        summary_start_2, summary_end_2 = get_middle_sublist(
            original_answer,
            [19389, 2864, 49970, 29], 
            [694, 28477, 49970, 1363]
        )
        caption_start_1, caption_end_1 = get_middle_sublist(
            original_answer,
            [20996, 2599, 60459, 29], 
            [524, 32500, 60459, 1363]
        )
        caption_start_2, caption_end_2 = get_middle_sublist(
            original_answer,
            [20996, 2599, 60459, 29], 
            [694, 32500, 60459, 1363]
        )
        reasoning_start_1, reasoning_end_1 = get_middle_sublist(
            original_answer,
            [27, 793, 36404, 1753, 29], 
            [524, 793, 36404, 1753, 1363]
        )
        reasoning_start_2, reasoning_end_2 = get_middle_sublist(
            original_answer,
            [27, 793, 36404, 1753, 29], 
            [694, 793, 36404, 1753, 1363]
        )
        if (summary_start_1 is None and summary_start_2 is None) \
            or (caption_start_1 is None and caption_start_2 is None) \
            or (reasoning_start_1 is None and reasoning_start_2 is None):
            continue    # Xuan: skip this data  # total: 215095
        
        both_data.append(both_sample)
    

# print(f"Total thinking data: {len(thinking_data)}")     # 254866
# print(f"Total decode thinking data: {len(decode_thinking_data)}")   # 764598 = 254866 * 3

random.shuffle(both_data)
split_index = int(train_split_ratio * len(both_data))
both_data_train = both_data[:split_index]
both_data_test = both_data[split_index:]
print(f"Total both data: {len(both_data)}")     # total: 254823
print(f"Total both data train: {len(both_data_train)}")     # 80%: 203858
print(f"Total both data test: {len(both_data_test)}")     # 20%: 50965

# random.shuffle(thinking_data)
# split_index = int(0.8 * len(thinking_data))
# thinking_data_train = thinking_data[:split_index]
# thinking_data_test = thinking_data[split_index:]
# thinking_data_train_ids = [sample['id'] for sample in thinking_data_train]
# thinking_data_test_ids = [sample['id'] for sample in thinking_data_test]

# decode_thinking_data_dict = {sample['id']: sample for sample in decode_thinking_data}
# decode_thinking_data_train = []
# for id in thinking_data_train_ids:
#     decode_thinking_data_train.append(decode_thinking_data_dict[id+"-summary"])
#     decode_thinking_data_train.append(decode_thinking_data_dict[id+"-caption"])
#     decode_thinking_data_train.append(decode_thinking_data_dict[id+"-reasoning"])
# decode_thinking_data_test = []
# for id in thinking_data_test_ids:
#     decode_thinking_data_test.append(decode_thinking_data_dict[id+"-summary"])
#     decode_thinking_data_test.append(decode_thinking_data_dict[id+"-caption"])
#     decode_thinking_data_test.append(decode_thinking_data_dict[id+"-reasoning"])

# print("len(thinking_data_train):", len(thinking_data_train))    # 203892
# print("len(thinking_data_test):", len(thinking_data_test))      # 50974
# print("len(decode_thinking_data_train):", len(decode_thinking_data_train)) # 611676
# print("len(decode_thinking_data_test):", len(decode_thinking_data_test))  # 152922

# # Check if the ids are correctly matched
# for idx, sample in enumerate(thinking_data_train):
#     current_id = sample['id']
#     if current_id+"-summary" != decode_thinking_data_train[3*idx]['id']:
#         print(current_id+"-summary")
#         print(decode_thinking_data_train[3*idx]['id'])
#         print(f"Error at index summary {idx}")
#         exit()
#     if current_id+"-caption" != decode_thinking_data_train[3*idx+1]['id']:
#         print(current_id+"-caption")
#         print(decode_thinking_data_train[3*idx+1]['id'])
#         print(f"Error at index caption {idx}")
#         exit()
#     if current_id+"-reasoning" != decode_thinking_data_train[3*idx+2]['id']:
#         print(f"Error at index reasoning {idx}")
#         exit()
# for idx, sample in enumerate(thinking_data_test):
#     current_id = sample['id']
#     if current_id+"-summary" != decode_thinking_data_test[3*idx]['id']:
#         print(f"Error at index summary test {idx}")
#         exit()
#     if current_id+"-caption" != decode_thinking_data_test[3*idx+1]['id']:
#         print(f"Error at index caption test {idx}")
#         exit()
#     if current_id+"-reasoning" != decode_thinking_data_test[3*idx+2]['id']:
#         print(f"Error at index reasoning test {idx}")


# # Save the thinking data into json file
# with open("/mnt/localssd/llava-cot-dataset/json_files/thinking_data-train.json", "w") as f:
#     json.dump(thinking_data_train, f, indent=4)

# with open("/mnt/localssd/llava-cot-dataset/json_files/thinking_data-test.json", "w") as f:
#     json.dump(thinking_data_test, f, indent=4)

# with open("/mnt/localssd/llava-cot-dataset/json_files/decode_thinking_data-train.json", "w") as f:
#     json.dump(decode_thinking_data_train, f, indent=4)

# with open("/mnt/localssd/llava-cot-dataset/json_files/decode_thinking_data-test.json", "w") as f:
#     json.dump(decode_thinking_data_test, f, indent=4)

# with open("/mnt/localssd/llava-cot-dataset/json_files/both_data-train.json", "w") as f:
with open("/mnt/localssd/llava-cot-dataset/json_files/data_train-pure_llm-num_thinking_token_summary{}_caption{}_reasoning{}.json".format(
    compression_ratio_thinking_tokens_summary,
    compression_ratio_thinking_tokens_caption,
    compression_ratio_thinking_tokens_reasoning
), "w") as f:
    json.dump(both_data_train, f, indent=4)

# with open("/mnt/localssd/llava-cot-dataset/json_files/both_data-test.json", "w") as f:
with open("/mnt/localssd/llava-cot-dataset/json_files/data_test-pure_llm-num_thinking_token_summary{}_caption{}_reasoning{}.json".format(
    compression_ratio_thinking_tokens_summary,
    compression_ratio_thinking_tokens_caption,
    compression_ratio_thinking_tokens_reasoning
), "w") as f:
    json.dump(both_data_test, f, indent=4)
