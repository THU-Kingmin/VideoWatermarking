import openai
import ast
import argparse
import time

openai.api_key = "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"

def annotate(question, answer, pred):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": 
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                }
            ]
        )
        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        response_dict = ast.literal_eval(response_message)
    except Exception as e:
        print(f"An error occurred: {e}")
        response_dict = {}
        response_dict['score'] = 0
        response_dict['pred'] = 'no'
    return response_dict

parser = argparse.ArgumentParser()
parser.add_argument("--pred_text_path", default='/apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir/baseline_m16/clean.txt', type=str)
parser.add_argument("--tgt_text_path",  default='/apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir/baseline_m16/answer.txt', type=str)
parser.add_argument("--q_text_path",  default='/apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir/baseline_m16/question.txt', type=str)
args = parser.parse_args()

with open(args.pred_text_path, 'r') as f:
    predicted_sentences = f.readlines()

with open(args.tgt_text_path, 'r') as f:
    true_sentences = f.readlines()

with open(args.q_text_path, 'r') as f:
    q_sentences = f.readlines()

print(args.pred_text_path.split('/')[-2], args.pred_text_path.split('/')[-1])
score = 0
acc = 0
total = 0
for predicted, true, q in zip(predicted_sentences, true_sentences, q_sentences):
    # print('predicted', predicted)
    # print('true', true)
    # print('q', q)
    response_dict = annotate(q, true, predicted)
    # print(response_dict)
    total += 1
    score += response_dict['score']
    if response_dict['pred'] == 'yes':
        acc += 1

print(f'acc: {acc/total}, score: {score/total}')
# time.sleep(10)
    