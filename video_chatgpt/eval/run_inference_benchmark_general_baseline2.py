import os
from random import random
import sys
sys.path.append('.')
import argparse
import json
from tqdm import tqdm

from video_chat.utils.config import Config
config_file = "video_chat/configs/config_7b.json"
cfg = Config.from_file(config_file)
from video_chat.models.videochat import VideoChat
from video_chat.inference_api import load_video, ask, model_answer
from video_chat.utils.easydict import EasyDict
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

    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_iter", type=int, default=300)
    parser.add_argument("--weight_loss2", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--max_modify", type=int, default=16)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--type", type=str, default='type1')

    return parser.parse_args()

def video_chat_infer(video, question, msg, model):
    img_list = []
    image_emb, _ = model.encode_img(video)
    img_list.append(image_emb)
    chat = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
    ask(question, chat)
    output = model_answer(conv=chat, model=model, img_list=img_list, max_new_tokens=1000)[0]
    return output

def video_chat_random(video, question, msg, model, args):
    print('video_chatgpt_random')
    modif_max = args.max_modify / 255 / 0.26130258
    modifier = (torch.rand(1, args.seq_len, 3, 224, 224) * modif_max).half().cuda()
    min_in = video.min().detach() #-1.7923
    max_in = video.max().detach() #2.1459
    image_random = torch.clamp((modifier + video), min_in, max_in)

    print('Delta: ', abs(image_random-video).mean() * 0.26130258 * 255)

    img_list = []
    image_emb, _ = model.encode_img(video)
    img_list.append(image_emb)
    chat = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
    ask(question, chat)
    outputs_random = model_answer(conv=chat, model=model, img_list=img_list, max_new_tokens=1000)[0]
    
    image_random = torch.squeeze(image_random, 0)
    video = torch.squeeze(video, 0)
    return outputs_random, video, image_random, abs(image_random-video).mean() * 0.26130258 * 255

def video_chat_black(video, question, msg, model, args):
    print('video_chatgpt_black')
    image_black = ((torch.zeros(1, args.seq_len, 3, 224, 224)-0.4814)/0.2686).half().cuda()
    print('Delta: ', abs(image_black-video).mean() * 0.26130258 * 255)

    img_list = []
    image_emb, _ = model.encode_img(image_black)
    img_list.append(image_emb)
    chat = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
    ask(question, chat)
    outputs_black = model_answer(conv=chat, model=model, img_list=img_list, max_new_tokens=1000)[0]
    
    image_black = torch.squeeze(image_black, 0)
    video = torch.squeeze(video, 0)
    return outputs_black, video, image_black, abs(image_black-video).mean() * 0.26130258 * 255

def video_chat_white(video, question, msg, model, args):
    print('video_chatgpt_white')
    image_white = ((torch.ones(1, args.seq_len, 3, 224, 224)-0.40821073)/0.2757).half().cuda()
    print('Delta: ', abs(image_white-video).mean() * 0.26130258 * 255)

    img_list = []
    image_emb, _ = model.encode_img(image_white)
    img_list.append(image_emb)
    chat = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
    ask(question, chat)
    outputs_white = model_answer(conv=chat, model=model, img_list=img_list, max_new_tokens=1000)[0]
    
    image_white = torch.squeeze(image_white, 0)
    video = torch.squeeze(video, 0)
    return outputs_white, video, image_white, abs(image_white-video).mean() * 0.26130258 * 255

def run_inference(args):
    model = VideoChat(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.eval()

    with open(args.gt_file) as file:
        gt_contents = json.load(file)
    if not os.path.exists(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}')):
        os.makedirs(os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}'))

    output_list_txt = []
    answer_list_txt = []
    question_list_txt = []
    random_list_txt = []
    black_list_txt = []
    white_list_txt = []
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    i = 0
    delta_randoms = 0
    delta_blacks = 0
    delta_whites = 0
    for sample in tqdm(gt_contents):
        video_name = sample['video_name']
        question = sample['Q']
        answer = sample['A']
        answer_list_txt.append(answer)

        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        if video_path is not None:  # Modified this line
            vid, msg = load_video(video_path, num_segments=args.seq_len, return_msg=True)
            # print('vid: ', vid.shape) #torch.Size([48, 224, 224])
            # print('msg: ', msg) #The video contains 16 frames sampled at 6.3..., 194.6 seconds.

        TC, H, W = vid.shape
        video = vid.reshape(1, TC//3, 3, H, W).half().to("cuda") # The model expects inputs of shape: 1 x T x C x H x W
        output = video_chat_infer(video, question, msg, model)
        output = output.replace('\n', '')
        output_list_txt.append(output)

        output_random, image_clean, image_random, delta_random = video_chat_random(video, question, msg, model, args)
        output_black, image_clean, image_black, delta_black = video_chat_black(video, question, msg, model, args)
        output_white, image_clean, image_white, delta_white = video_chat_white(video, question, msg, model, args)
        output_random = output_random.replace('\n', '')
        output_black = output_black.replace('\n', '')
        output_white = output_white.replace('\n', '')
        random_list_txt.append(output_random)
        black_list_txt.append(output_black)
        white_list_txt.append(output_white)
        question_list_txt.append(question)
        print('1. question: ', question)
        print('2. answer: ', answer)
        print('3. clean: ', output)
        print('4. output_random: ', output_random)
        print('5. output_black: ', output_black)
        print('6. output_white: ', output_white)

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
