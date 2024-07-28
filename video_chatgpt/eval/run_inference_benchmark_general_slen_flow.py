import os
import sys
sys.path.append('.')
import argparse
import json
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference_seq_mask import video_chatgpt_infer, video_chatgpt_attack_seq
from video_chatgpt.inference_random_mask import video_chatgpt_infer, video_chatgpt_attack_random
from video_chatgpt.inference_flow_mask import video_chatgpt_infer, video_chatgpt_attack_flow
import torch
import cv2
import numpy as np

from flow_util import gen_flow, save_tensor_images

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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 将图像从RGB转换为BGR
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
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)

    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_iter", type=int, default=300)
    parser.add_argument("--weight_clip", type=int, default=1)
    parser.add_argument("--weight_llm", type=int, default=3)
    parser.add_argument("--weight_spa", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--max_modify", type=int, default=16)
    parser.add_argument("--mask_type", type=str, default='seq')
    parser.add_argument("--mask", type=int, default=20)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--type", type=str, default='type1')

    return parser.parse_args()

import string

def count_unique_words(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punctuation = text.translate(translator)

    words = text_without_punctuation.split()
    words = [word.lower() for word in words]
    unique_words = set(words)
    return len(unique_words), len(words)

def isGarbled(texts):
    # print('texts', texts)
    garble = []
    count = 0
    total = len(texts)
    for text in texts:
        new_len, total_len = count_unique_words(text)
        if new_len < total_len//3 or total_len < 5:
            count += 1
            garble.append(1)
        else:
            garble.append(0)
    print('Garbled: ', count / total)
    return garble

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def vis_feature(data, garble, path):
    data_np = data.numpy()
    garble_np = garble.numpy()
    pca = PCA(n_components=10)
    data_pca = pca.fit_transform(data_np)
    shapes = ['o', 's']
    colors = ['r', 'b']  

    for j in range(data_pca.shape[1]-1):
        plt.figure()
        for i in range(data_pca.shape[0]):
            shape = shapes[garble[i]]
            color = colors[garble[i]]  # 选择颜色
            plt.scatter(data_pca[i, j], data_pca[i, j+1], marker=shape, color=color)

        merged_array = np.concatenate((data_pca[:, j:j+2], garble_np), axis=1)
        # print('merged_array:', merged_array.shape)
        np.savetxt(f"{path}_pca{j}_vs_pca{j+1}.txt", merged_array, fmt="%d", delimiter="\t")
        plt.savefig(f"{path}_pca{j}_vs_pca{j+1}.png")

def vis_feature_v0(data, garble, path):
    data_np = data.numpy()

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_np)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_pca)
    shapes = ['o', 's']
    plt.figure()
    labels = kmeans.labels_
    colors = ['r', 'g', 'b']  # 定义三种颜色

    for i in range(data_pca.shape[0]):
        shape = shapes[garble[i]]
        color = colors[labels[i]]  # 选择颜色
        plt.scatter(data_pca[i, 0], data_pca[i, 1], marker=shape, color=color)

    plt.savefig(path)

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

    save_path = f'{args.version}_{args.type}_clip{args.weight_clip}_llm{args.weight_llm}_spa{args.weight_spa}_m{args.max_modify}_s{args.seq_len}_{args.mask_type}{args.mask}'
    # Create the output directory if it doesn't exist
    if not os.path.exists(os.path.join(args.output_dir, save_path)):
        os.makedirs(os.path.join(args.output_dir, save_path))

    output_list = []  # List to store the output results
    output_list_txt = []
    attack_list_txt = []
    answer_list_txt = []
    question_list_txt = []
    output_vis_list = []
    output_clip_list = []
    conv_mode = args.conv_mode
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    i = 0
    delta_attacks = 0
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

        if args.mask_type == 'flow':
            flow_sort = gen_flow(video_frames, os.path.join(args.output_dir, save_path, f"{i}_{video_name}_flow"))
            i += 1
            # continue
            
        output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                            tokenizer, image_processor, video_token_len)
        sample_set['pred'] = output
        output_list.append(sample_set)
        output_list_txt.append(output)

        if args.mask_type == 'seq':
            output_attack,image_clean, image_attack, delta_attack = video_chatgpt_attack_seq(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len, args)
        elif args.mask_type == 'random':
            output_attack,image_clean, image_attack, delta_attack = video_chatgpt_attack_random(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len, args)
        elif args.mask_type == 'flow':
            output_attack,image_clean, image_attack, delta_attack, modifier, output_vis, output_clip = video_chatgpt_attack_flow(video_frames, flow_sort, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len, args)                       
        output_attack = output_attack.replace('\n', '')
        attack_list_txt.append(output_attack)
        question_list_txt.append(question)
        output_vis_list.append(output_vis) # torch.Size([1, 1, 4096])
        output_clip_list.append(output_clip) # torch.Size([356, 1024])
        print('question: ', question)
        print('answer: ', answer)
        print('clean: ', output)
        print('attack: ', output_attack)
        print('output_vis: ', output_vis.shape)
        print('output_clip: ', output_clip.shape)

        if args.mask_type == 'flow':
            save_tensor_images(modifier, os.path.join(args.output_dir, save_path, f"{i}_{video_name}_modi"))
        save_normalized_tensor_as_video(image_clean.cpu().detach(), os.path.join(args.output_dir, save_path, f"{i}_{video_name}_clean_video.mp4"))
        save_normalized_tensor_as_video(image_attack.cpu().detach(), os.path.join(args.output_dir, save_path, f"{i}_{video_name}_attack_video.mp4"))
        i += 1
        delta_attacks += delta_attack
    
    garble = isGarbled(attack_list_txt)
    data_tensor = torch.cat([x.view(1, -1) for x in output_vis_list], dim=0)
    clip_tensor = torch.cat([x.view(1, -1) for x in output_clip_list], dim=0)
    garble_tensor = torch.tensor(garble)
    garble_tensor = garble_tensor.unsqueeze(1)
    print('data_tensor', data_tensor.shape)
    print('clip_tensor', clip_tensor.shape)
    print('garble_tensor', garble_tensor.shape, garble_tensor)
    vis_feature(data_tensor, garble_tensor, os.path.join(args.output_dir, save_path, f"vis_feature_v3"))
    vis_feature(clip_tensor, garble_tensor, os.path.join(args.output_dir, save_path, f"clip_feature_v3"))
    # Save
    with open(os.path.join(args.output_dir, save_path, f"clean.txt"), 'w', encoding='utf-8') as f1:
        for text in output_list_txt:
            f1.write(text + '\n')
    with open(os.path.join(args.output_dir, save_path, f"answer.txt"), 'w', encoding='utf-8') as f2:
        for text in answer_list_txt:
            f2.write(text + '\n')
    with open(os.path.join(args.output_dir, save_path, f"question.txt"), 'w', encoding='utf-8') as f3:
        for text in question_list_txt:
            f3.write(text + '\n')
    with open(os.path.join(args.output_dir, save_path, f"attack.txt"), 'w', encoding='utf-8') as f4:
        for text in attack_list_txt:
            f4.write(text + '\n')
    print('delta_attacks:', delta_attacks / len(gt_contents))

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
