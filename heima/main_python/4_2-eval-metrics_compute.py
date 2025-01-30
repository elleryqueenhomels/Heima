import evaluate
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Experiment Configuration")

    parser.add_argument("--main_exp_name", type=str, default="exp-step_3", help="Main experiment name")
    parser.add_argument("--decoder_exp_name", type=str, default="exp-step_4", help="Decoder experiment name")
    parser.add_argument("--GPU_total_split_num", type=int, default=8, help="Total GPU split number")

    args = parser.parse_args()

    return args

args = get_args()

data_path = f"/mnt/localssd/llava-cot-checkpoints/output-generation_record/main_{args.main_exp_name}--decoder_{args.decoder_exp_name}/"

save_path = data_path + "/evaluation_metrics_{metric}.json"

total_data = []
for i in range(GPU_total_split_num):
    with open(data_path + "/generation_records-split_{}_of_{}.json".format(i+1, GPU_total_split_num), "r") as f:
        data = json.load(f)
        total_data.extend(data)


bleu = evaluate.load("bleu")
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')
bertscore = evaluate.load("bertscore")

summary_bleu_score = []
summary_meteor_res = []
summary_rouge_L = []
summary_bertscore_res = []
caption_bleu_score = []
caption_meteor_res = []
caption_rouge_L = []
caption_bertscore_res = []
reasoning_bleu_score = []
reasoning_meteor_res = []
reasoning_rouge_L = []
reasoning_bertscore_res = []
conclusion_bleu_score = []
conclusion_meteor_res = []
conclusion_rouge_L = []
conclusion_bertscore_res = []


for idx, data in enumerate(total_data):

    if (idx + 1) % 100 == 0:
        print("Processing: ", idx + 1)
        print("Current evaluation metrics:")
        print()
        print("Summary BLEU: ", sum(summary_bleu_score) / len(summary_bleu_score))
        print("Summary METEOR: ", sum(summary_meteor_res) / len(summary_meteor_res))
        print("Summary ROUGE-L: ", sum(summary_rouge_L) / len(summary_rouge_L))
        print("Summary BERTSCORE: ", sum(summary_bertscore_res) / len(summary_bertscore_res))
        print()
        print("Caption BLEU: ", sum(caption_bleu_score) / len(caption_bleu_score))
        print("Caption METEOR: ", sum(caption_meteor_res) / len(caption_meteor_res))
        print("Caption ROUGE-L: ", sum(caption_rouge_L) / len(caption_rouge_L))
        print("Caption BERTSCORE: ", sum(caption_bertscore_res) / len(caption_bertscore_res))
        print()
        print("Reasoning BLEU: ", sum(reasoning_bleu_score) / len(reasoning_bleu_score))
        print("Reasoning METEOR: ", sum(reasoning_meteor_res) / len(reasoning_meteor_res))
        print("Reasoning ROUGE-L: ", sum(reasoning_rouge_L) / len(reasoning_rouge_L))
        print("Reasoning BERTSCORE: ", sum(reasoning_bertscore_res) / len(reasoning_bertscore_res))
        print()
        print("Conclusion BLEU: ", sum(conclusion_bleu_score) / len(conclusion_bleu_score))
        print("Conclusion METEOR: ", sum(conclusion_meteor_res) / len(conclusion_meteor_res))
        print("Conclusion ROUGE-L: ", sum(conclusion_rouge_L) / len(conclusion_rouge_L))
        print("Conclusion BERTSCORE: ", sum(conclusion_bertscore_res) / len(conclusion_bertscore_res))
        print()
        print()
        print()

    decoded_text_summary = data["generated_summary"]
    current_answer_summary = data["ground_truth_summary"]

    decoded_text_caption = data["generated_caption"]
    current_answer_caption = data["ground_truth_caption"]

    decoded_text_reasoning = data["generated_reasoning"]
    current_answer_reasoning = data["ground_truth_reasoning"]

    generated_conclusion = data["generated_conclusion"]
    current_conclusion = data["ground_truth_conclusion"]


    # Xuan: compute evaluation metrics
    summary_predictions = [decoded_text_summary]
    summary_references = [[current_answer_summary]]

    caption_predictions = [decoded_text_caption]
    caption_references = [[current_answer_caption]]

    reasoning_predictions = [decoded_text_reasoning]
    reasoning_references = [[current_answer_reasoning]]

    conclusion_predictions = [generated_conclusion]
    conclusion_references = [[current_conclusion]]

    try:
        results_bleu = bleu.compute(predictions=summary_predictions, references=summary_references)
        results_meteor = meteor.compute(predictions=summary_predictions, references=summary_references)
        results_rouge = rouge.compute(predictions=summary_predictions, references=summary_references)
        results_bertscore = bertscore.compute(predictions=summary_predictions, references=summary_references[0], lang="en", model_type="microsoft/deberta-xlarge-mnli")["f1"][0]

        summary_bleu_score.append(results_bleu['bleu'])
        summary_meteor_res.append(results_meteor['meteor'])
        summary_rouge_L.append(results_rouge['rougeL'])
        summary_bertscore_res.append(results_bertscore)

        results_bleu = bleu.compute(predictions=caption_predictions, references=caption_references)
        results_meteor = meteor.compute(predictions=caption_predictions, references=caption_references)
        results_rouge = rouge.compute(predictions=caption_predictions, references=caption_references)
        results_bertscore = bertscore.compute(predictions=caption_predictions, references=caption_references[0], lang="en", model_type="microsoft/deberta-xlarge-mnli")["f1"][0]

        caption_bleu_score.append(results_bleu['bleu'])
        caption_meteor_res.append(results_meteor['meteor'])
        caption_rouge_L.append(results_rouge['rougeL'])
        caption_bertscore_res.append(results_bertscore)

        results_bleu = bleu.compute(predictions=reasoning_predictions, references=reasoning_references)
        results_meteor = meteor.compute(predictions=reasoning_predictions, references=reasoning_references)
        results_rouge = rouge.compute(predictions=reasoning_predictions, references=reasoning_references)
        results_bertscore = bertscore.compute(predictions=reasoning_predictions, references=reasoning_references[0], lang="en", model_type="microsoft/deberta-xlarge-mnli")["f1"][0]

        reasoning_bleu_score.append(results_bleu['bleu'])
        reasoning_meteor_res.append(results_meteor['meteor'])
        reasoning_rouge_L.append(results_rouge['rougeL'])
        reasoning_bertscore_res.append(results_bertscore)

        results_bleu = bleu.compute(predictions=conclusion_predictions, references=conclusion_references)
        results_meteor = meteor.compute(predictions=conclusion_predictions, references=conclusion_references)
        results_rouge = rouge.compute(predictions=conclusion_predictions, references=conclusion_references)
        results_bertscore = bertscore.compute(predictions=conclusion_predictions, references=conclusion_references[0], lang="en", model_type="microsoft/deberta-xlarge-mnli")["f1"][0]

        conclusion_bleu_score.append(results_bleu['bleu'])
        conclusion_meteor_res.append(results_meteor['meteor'])
        conclusion_rouge_L.append(results_rouge['rougeL'])
        conclusion_bertscore_res.append(results_bertscore)

    except:
        continue


print("Final evaluation metrics:")
print()
print("Summary BLEU: ", sum(summary_bleu_score) / len(summary_bleu_score))
print("Summary METEOR: ", sum(summary_meteor_res) / len(summary_meteor_res))
print("Summary ROUGE-L: ", sum(summary_rouge_L) / len(summary_rouge_L))
print("Summary BERTSCORE: ", sum(summary_bertscore_res) / len(summary_bertscore_res))
print()
print("Caption BLEU: ", sum(caption_bleu_score) / len(caption_bleu_score))
print("Caption METEOR: ", sum(caption_meteor_res) / len(caption_meteor_res))
print("Caption ROUGE-L: ", sum(caption_rouge_L) / len(caption_rouge_L))
print("Caption BERTSCORE: ", sum(caption_bertscore_res) / len(caption_bertscore_res))
print()
print("Reasoning BLEU: ", sum(reasoning_bleu_score) / len(reasoning_bleu_score))
print("Reasoning METEOR: ", sum(reasoning_meteor_res) / len(reasoning_meteor_res))
print("Reasoning ROUGE-L: ", sum(reasoning_rouge_L) / len(reasoning_rouge_L))
print("Reasoning BERTSCORE: ", sum(reasoning_bertscore_res) / len(reasoning_bertscore_res))
print()
print("Conclusion BLEU: ", sum(conclusion_bleu_score) / len(conclusion_bleu_score))
print("Conclusion METEOR: ", sum(conclusion_meteor_res) / len(conclusion_meteor_res))
print("Conclusion ROUGE-L: ", sum(conclusion_rouge_L) / len(conclusion_rouge_L))
print("Conclusion BERTSCORE: ", sum(conclusion_bertscore_res) / len(conclusion_bertscore_res))
print()



# Save evaluation metrics
with open(save_path.format(metric="summary_bleu"), "w") as f:
    json.dump(summary_bleu_score, f, indent=4)

with open(save_path.format(metric="summary_meteor"), "w") as f:
    json.dump(summary_meteor_res, f, indent=4)

with open(save_path.format(metric="summary_rouge"), "w") as f:
    json.dump(summary_rouge_L, f, indent=4)

with open(save_path.format(metric="summary_bertscore"), "w") as f:
    json.dump(summary_bertscore_res, f, indent=4)

with open(save_path.format(metric="caption_bleu"), "w") as f:
    json.dump(caption_bleu_score, f, indent=4)

with open(save_path.format(metric="caption_meteor"), "w") as f:
    json.dump(caption_meteor_res, f, indent=4)

with open(save_path.format(metric="caption_rouge"), "w") as f:
    json.dump(caption_rouge_L, f, indent=4)

with open(save_path.format(metric="caption_bertscore"), "w") as f:
    json.dump(caption_bertscore_res, f, indent=4)

with open(save_path.format(metric="reasoning_bleu"), "w") as f:
    json.dump(reasoning_bleu_score, f, indent=4)

with open(save_path.format(metric="reasoning_meteor"), "w") as f:
    json.dump(reasoning_meteor_res, f, indent=4)

with open(save_path.format(metric="reasoning_rouge"), "w") as f:
    json.dump(reasoning_rouge_L, f, indent=4)

with open(save_path.format(metric="reasoning_bertscore"), "w") as f:
    json.dump(reasoning_bertscore_res, f, indent=4)

with open(save_path.format(metric="conclusion_bleu"), "w") as f:
    json.dump(conclusion_bleu_score, f, indent=4)

with open(save_path.format(metric="conclusion_meteor"), "w") as f:
    json.dump(conclusion_meteor_res, f, indent=4)

with open(save_path.format(metric="conclusion_rouge"), "w") as f:
    json.dump(conclusion_rouge_L, f, indent=4)

with open(save_path.format(metric="conclusion_bertscore"), "w") as f:
    json.dump(conclusion_bertscore_res, f, indent=4)

