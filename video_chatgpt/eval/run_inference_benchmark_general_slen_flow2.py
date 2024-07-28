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
from video_chat.inference_api import load_video, ask, model_answer, model_attack
from video_chat.utils.easydict import EasyDict
import torch
import cv2
import numpy as np
from flow_util import gen_flow
import torch.nn as nn
from torch.autograd import Variable
import time
from datetime import datetime
import pytz
shanghai_tz = pytz.timezone('Asia/Shanghai')

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

def video_chat_attack_flow(image_clean, flow_sort, question, msg, model, args):
    with torch.no_grad():
        image_emb, _ = model.encode_img(image_clean)
        print('Video features: ', image_emb.shape) #torch.Size([1, 96, 4096])
    chat = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
    ask(question, chat)
    with torch.no_grad():
        output_clean = model_attack(conv=chat, model=model, img_list=[image_emb])
        print('LLM features: ', output_clean.shape)
    #Define the modifier and the optimizer
    modif_max = args.max_modify / 255 / 0.26130258
    step = args.step / 255 / 0.26130258
    print('modif_max', modif_max)
    print('step', step)
    modif = torch.Tensor(1, args.seq_len, 3, 224, 224).fill_(step).half().to('cuda')
    modifier = torch.nn.Parameter(modif, requires_grad=True)
    prev_loss = 1e-5
    mse_loss = nn.MSELoss()

    # model.llama_model.model.requires_grad_(False)
    # model.llama_model.lm_head.requires_grad_(False)
    model.requires_grad_(False)
    image_clean = Variable(image_clean, requires_grad=False)
    min_in = image_clean.min().detach() #-1.7923
    max_in = image_clean.max().detach() #2.1459

    mask_len = int(args.mask / 100 * args.seq_len)
    mask = torch.zeros(1, args.seq_len, 3, 224, 224)
    flow_indices = torch.tensor(flow_sort)[:args.seq_len-mask_len]
    mask[0,flow_indices,...] = 1
    print('mask_len:', mask_len, ', seq_len', args.seq_len, flow_indices)
    mask = mask.half().to('cuda')

    # train
    start_time = time.time()
    for iter in range(args.num_iter):
        image_attack = torch.clamp((modifier * mask + image_clean), min_in, max_in)
        image_emb_attack, _ = model.encode_img(image_attack)
        chat = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
        ask(question, chat)
        output_attack = model_attack(conv=chat, model=model, img_list=[image_emb_attack])
        
        loss_clip = - mse_loss(image_emb, image_emb_attack)
        
        loss_llm = - mse_loss(output_clean, output_attack)
        
        loss_spa = torch.sum(torch.sqrt(torch.mean(torch.pow((torch.unsqueeze(modifier, 0)), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        
        loss = args.weight_clip * loss_clip + args.weight_llm * loss_llm + args.weight_spa * loss_spa

        loss.backward()
        grad_sign = torch.sign(modifier.grad)
        modifier.data -= grad_sign * step
        modifier.grad.zero_()
        modifier.data = torch.clamp(modifier.data, -modif_max, modif_max)

        if (iter+1) % 50 == 0: 
            if prev_loss < loss:
                print(f'Iteration: [{iter+1}/{args.num_iter}], Loss: {loss}(\u25b2), loss_clip: {loss_clip}, loss_llm: {loss_llm}, loss_spa: {loss_spa}')
            elif prev_loss > loss: 
                print(f'Iteration: [{iter+1}/{args.num_iter}], Loss: {loss}(\u25bc), loss_clip: {loss_clip}, loss_llm: {loss_llm}, loss_spa: {loss_spa}')
            else: 
                print(f'Iteration: [{iter+1}/{args.num_iter}], Loss: {loss}, loss_clip: {loss_clip}, loss_llm: {loss_llm}, loss_spa: {loss_spa}')
        prev_loss = loss

    print(datetime.now(shanghai_tz).strftime("%Y-%m-%d %H:%M:%S"), "Training time: {:.2f} seconds".format(time.time() - start_time))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_clean = Variable(image_clean, requires_grad=False)
    image_attack = torch.clamp((modifier * mask + image_clean), min_in, max_in)
    print('Delta: ', abs(image_attack-image_clean).mean() * 0.26130258 * 255)

    with torch.no_grad():
        image_emb_attack, _ = model.encode_img(image_attack)
        chat = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
        ask(question, chat)
        outputs_attack = model_answer(conv=chat, model=model, img_list=[image_emb_attack], max_new_tokens=1000)[0]

    return outputs_attack, image_clean, image_attack, abs(image_attack-image_clean).mean() * 0.26130258 * 255

def run_inference(args):
    model = VideoChat(config=cfg.model)
    model = model.to(torch.device(cfg.device))
    model = model.eval()

    with open(args.gt_file) as file:
        gt_contents = json.load(file)
    save_path = f'{args.version}_{args.type}_clip{args.weight_clip}_llm{args.weight_llm}_spa{args.weight_spa}_m{args.max_modify}_s{args.seq_len}_{args.mask_type}{args.mask}'
    if not os.path.exists(os.path.join(args.output_dir, save_path)):
        os.makedirs(os.path.join(args.output_dir, save_path))

    output_list_txt = []
    attack_list_txt = []
    answer_list_txt = []
    question_list_txt = []
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    i = 0
    delta_attacks = 0
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
        if args.mask_type == 'flow':
            flow_sort = gen_flow(video.cpu().squeeze(0).permute(0,2,3,1), os.path.join(args.output_dir, save_path, f"{i}_{video_name}_flow"))
            i += 1
        
        output = video_chat_infer(video, question, msg, model)
        output = output.replace('\n', '')
        output_list_txt.append(output)

        if args.mask_type == 'flow':
            output_attack,image_clean, image_attack, delta_attack = video_chat_attack_flow(video, flow_sort, question, msg, model, args)  
        
        output_attack = output_attack.replace('\n', '')
    
        attack_list_txt.append(output_attack)
        question_list_txt.append(question)
        print('1. question: ', question)
        print('2. answer: ', answer)
        print('3. clean: ', output)
        print('4. output_attack: ', output_attack)

        save_normalized_tensor_as_video(image_clean.cpu().detach().squeeze(0), os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"{i}_{video_name}_clean_video.mp4"))
        save_normalized_tensor_as_video(image_attack.cpu().detach().squeeze(0), os.path.join(args.output_dir, f'{args.version}_m{args.max_modify}_s{args.seq_len}', f"{i}_{video_name}_attack_video.mp4"))
        
        delta_attacks += delta_attack
        i += 1
    
    print('delta_attacks:', delta_attacks / len(gt_contents))
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


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
