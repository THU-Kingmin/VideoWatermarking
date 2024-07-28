from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge 
from pycocoevalcap.cider.cider import Cider
import argparse
import sys

sys.setrecursionlimit(2000)

parser = argparse.ArgumentParser()
parser.add_argument("--pred_text_path", default='/apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir/attack_w0_m16/clean.txt', type=str)
parser.add_argument("--tgt_text_path",  default='/apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir/attack_w0_m16/answer.txt', type=str)
args = parser.parse_args()

# 假设你有两个文本文件，一个包含预测的文本，一个包含真实的文本
with open(args.pred_text_path, 'r') as f:
    predicted_sentences = f.readlines()

with open(args.tgt_text_path, 'r') as f:
    true_sentences = f.readlines()

# 计算 BLEU 分数
bleu_score = 0
smoothie = SmoothingFunction().method4
for predicted, true in zip(predicted_sentences, true_sentences):
    bleu_score += sentence_bleu([true.split()], predicted.split(), smoothing_function=smoothie)
bleu_score /= len(predicted_sentences)

# 计算 ROUGE-L 分数
rouge = Rouge()
rouge_l_score = 0
for predicted, true in zip(predicted_sentences, true_sentences):
    scores = rouge.get_scores(predicted, true)
    rouge_l_score += scores[0]['rouge-l']['f']
rouge_l_score /= len(predicted_sentences)

# 计算 CIDEr 分数
def list_to_dict(sentences):
    sentence_dict = {}
    for i, sentence in enumerate(sentences):
        sentence_dict[i] = [sentence.strip()]
    return sentence_dict

true_sentences_dict = list_to_dict(true_sentences)
predicted_sentences_dict = list_to_dict(predicted_sentences)
cider_scorer = Cider()
cider_score, _ = cider_scorer.compute_score(true_sentences_dict, predicted_sentences_dict)

print(f"BLEU: {bleu_score}, ROUGE-L: {rouge_l_score}, CIDEr: {cider_score}")