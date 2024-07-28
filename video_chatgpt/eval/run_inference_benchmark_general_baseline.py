import os
from random import random
import sys
sys.path.append('.')
import argparse
import json
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer, video_chatgpt_random, video_chatgpt_black, video_chatgpt_white
import torch
import cv2
import numpy as np

def save_normalized_tensor_as_video(image_tensor, filename, fps=5):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    # 反归一化图像张量
    denormalized_tensor = (image_tensor * std) + mean
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)
    # 将张量转换为 NumPy 数组，并将值范围从 [0, 1] 转换为 [0, 255]
    denormalized_array = (denormalized_tensor.numpy() * 255).astype(np.uint8)
    # 获取图像的宽度和高度
    height, width = denormalized_array.shape[2], denormalized_array.shape[3]
    # 创建一个 VideoWriter 对象，用于将 NumPy 数组转换为 MP4 视频
    video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # 将每个帧写入视频
    for i in range(denormalized_array.shape[0]):
        frame = denormalized_array[i].transpose(1, 2, 0)  # 将通道维度从 [C, H, W] 转换为 [H, W, C]
        video_writer.write(frame)
    video_writer.release()

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)

    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_iter", type=int, default=300)
    parser.add_argument("--weight_loss2", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--max_modify", type=int, default=16)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--type", type=str, default='type1')

    return parser.parse_args()


def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # Load the ground truth file
    with open(args.gt_file) as file:
        gt_contents = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}')):
        os.makedirs(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}'))

    output_list = []  # List to store the output results
    output_list_txt = []
    answer_list_txt = []
    question_list_txt = []
    random_list_txt = []
    black_list_txt = []
    white_list_txt = []
    conv_mode = args.conv_mode
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    i = 0
    delta_randoms = 0
    delta_blacks = 0
    delta_whites = 0
    for sample in tqdm(gt_contents):
        video_name = sample['video_name']
        sample_set = sample
        question = sample['Q']
        answer = sample['A']
        answer_list_txt.append(answer)

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if video_path is not None:  # Modified this line
            video_frames = load_video(video_path, num_frm=args.seq_len)

        output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                            tokenizer, image_processor, video_token_len)
        sample_set['pred'] = output
        output_list.append(sample_set)
        output_list_txt.append(output)

        output_random,image_clean, image_random, delta_random = video_chatgpt_random(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len, args)
        output_black,image_clean, image_black, delta_black = video_chatgpt_black(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len, args)
        output_white,image_clean, image_white, delta_white = video_chatgpt_white(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len, args)
        output_random = output_random.replace('\n', '')
        output_black = output_black.replace('\n', '')
        output_white = output_white.replace('\n', '')
        random_list_txt.append(output_random)
        black_list_txt.append(output_black)
        white_list_txt.append(output_white)
        question_list_txt.append(question)
        print('question: ', question)
        print('answer: ', answer)
        print('clean: ', output)
        print('output_random: ', output_random)
        print('output_black: ', output_black)
        print('output_white: ', output_white)

        save_normalized_tensor_as_video(image_clean.cpu().detach(), os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"{i}_{video_name}_clean_video.mp4"))
        save_normalized_tensor_as_video(image_random.cpu().detach(), os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"{i}_{video_name}_random_video.mp4"))
        save_normalized_tensor_as_video(image_black.cpu().detach(), os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"{i}_{video_name}_black_video.mp4"))
        save_normalized_tensor_as_video(image_white.cpu().detach(), os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"{i}_{video_name}_white_video.mp4"))
        delta_randoms += delta_random
        delta_blacks += delta_black
        delta_whites += delta_white
        i += 1
    
    print('delta_randoms:', delta_randoms / len(gt_contents))
    print('delta_blacks:', delta_blacks / len(gt_contents))
    print('delta_whites:', delta_whites / len(gt_contents))
    # Save
    with open(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"clean.txt"), 'w', encoding='utf-8') as f1:
        for text in output_list_txt:
            f1.write(text + '\n')
    with open(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"answer.txt"), 'w', encoding='utf-8') as f2:
        for text in answer_list_txt:
            f2.write(text + '\n')
    with open(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"question.txt"), 'w', encoding='utf-8') as f3:
        for text in question_list_txt:
            f3.write(text + '\n')
    with open(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"random.txt"), 'w', encoding='utf-8') as f4:
        for text in random_list_txt:
            f4.write(text + '\n')
    with open(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"black.txt"), 'w', encoding='utf-8') as f5:
        for text in black_list_txt:
            f5.write(text + '\n')
    with open(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"white.txt"), 'w', encoding='utf-8') as f6:
        for text in white_list_txt:
            f6.write(text + '\n')


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
