# Heima: Efficient Reasoning with Hidden Thinking

This repository provides an overview of all resources for the paper ["Efficient Reasoning with Hidden Thinking"](https://arxiv.org/abs/2501.19201).

## Demo
![M3](asserts/1987-BMW-E30-M3.png)

#### Input Question: 
```text
Which automotive brand does this car belong to, and what visual cues or badges indicate that?
```

#### Heima Encoder (MLLMs):
```text
<SUMMARY> <THINKING_OF_SUMMARY> </SUMMARY>

<CAPTION> <THINKING_OF_CAPTION> </CAPTION>

<REASONING> <THINKING_OF_REASONING> </REASONING>

<CONCLUSION> The image shows a black BMW M3 driving down a road. </CONCLUSION>
```

#### Heima Decoder (LLMs):
```text
Summary:
Below is the sequence of thought used for the summary: 
I will identify the car brand by examining visual cues such as logos, 
color schemes, and design elements present in the image.

Caption:
The step-by-step thinking process for the caption can be described as: 
The image shows a sleek, modern sports car with a black exterior. 
It has a distinct logo on the side, which resembles a cross with a circle.

Reasoning:
The thinking progress for the reasoning of the given question is illustrated as follows: 
The key to identifying the brand lies in the visible badge. 
The badge on the front grille of the car is crucial for determining the brand. 
In this image, the badge on the car is "BMW," which is a common symbol for the BMW brand. 
BMW is known for its distinctive badge, and the presence of this badge confirms the brand.
```


## Quick Start

### Install torchtune and vlmevalkit
1. Go to `torchtune_pkg/torchtune` and install by `pip install -e .`.
2. Go to `zero-shot-evaluation/VLMEvalKit` and install by `pip install -e .`.


### Prepare dataset
1. Download the [LLaVA-CoT-100k](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k) dataset.
2. Go to `heima/scripts/`.
3. Set the data path in `run-1_1-... sh` and `run-1_2-... .sh`.
4. Run by `sh .sh` to generate the data.


### Prepare LoRA checkpoints
1. We provide the checkpoints on HuggingFace: [shawnricecake/Heima](https://huggingface.co/shawnricecake/Heima/tree/main).
2. There are both Heima Encoder and 3 Heima Decoders for summary, caption, and reasoning, separately.


### Train Heima
1. We also provide the training code.
2. Set the right checkpoint path and data path for `LLaVA-CoT` and `Llama3.1-8B-Instruct` in `heima/configs` from `2_1... .yaml` to `2_5... .yaml`.
3. Go to `heima/scripts/` and run with `sh run-2-... .sh`.
4. You will get the final Heima Encoder after step 4 and 3 decoders after step 5.


### Evaluation for Heima Encoder
1. Set the checkpoint path in `zero-shot-evaluation/VLMEvalKit/configs/3-...-lora.yaml`.
2. Go to `zero-shot-evaluation/VLMEvalKit/` and run `sh run-eval.sh`.


### Evaluation for Heima Decoder
1. Set the right checkpoint path and data path for `LLaVA-CoT` and `Llama3.1-8B-Instruct` in `heima/configs` in `4_1... .yaml`.
2. Generate CoT reconstruction results by: go to `heima/scripts` and run with `sh run-4_1-... .sh`.
3. You can split into 8 GPUs for parallel running by revise:
```yaml
GPU_split_num: 0  # 0,1,2,3,4,5,6,7
GPU_total_split_num: 8
```
4. Compute the evaluation metrics by go to `heima/scripts` and run `sh run-4_2-... .sh`.


### Compute number of generated tokens
1. Go to `zero-shot-evaluation/VLMEvalKit/vlmeval/inference.py`.
2. Uncomment 139 and run the evaluation.
3. Evaluate Heima Encoder again.
4. `python3 compute_avg_num_token.py`


### Demo
1. Set the checkpoint path, your question, and your image in `heima/configs/5-... .yaml`.
2. Go to `heima/scripts/` and run with `sh run-5-... .sh`.

