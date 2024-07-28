import torch
import cv2
import numpy as np
from PIL import Image
from AMT.flow_generation.liteflownet.run import estimate
import os

def flow_to_color(flow, max_flow=None):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hue = angle * 180 / np.pi / 2
    if max_flow is not None:
        magnitude = np.clip(magnitude / max_flow, 0, 1)
    else:
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # magnitude[magnitude < 0.2] = 0
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = hue
    hsv[..., 1] = 1
    hsv[..., 2] = magnitude

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), magnitude

def pred_flow(img1, img2):
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1) / 255.0
    with torch.no_grad():
        flow = estimate(img1, img2)

    flow = flow.permute(1, 2, 0).cpu().numpy()
    return flow

def gen_flow(video_frames, path):
    print('video_frames: ', video_frames.shape)
    flow_mean = []
    for i in range(1, len(video_frames)):
        im1 = video_frames[i-1]
        im2 = video_frames[i]
        im1 = np.array(im1)[:, :, ::-1]
        im2 = np.array(im2)[:, :, ::-1]
        img1_copy = im1.copy()
        img2_copy = im2.copy()
        flow = pred_flow(img1_copy, img2_copy)
        flow_color, magnitude = flow_to_color(flow)
        flow_color_uint8 = (flow_color * 255).astype(np.uint8)
        flow_mean.append(magnitude.mean())
              
    flow_mean.append(0)
    flow_mean_np = np.array(flow_mean)
    flow_sort = np.argsort(-flow_mean_np).tolist()
    print(f'flow_mean: {flow_mean}, flow_sort: {flow_sort}')
    return flow_sort

def gen_flow2(video_frames, path):
    if not os.path.exists(os.path.join(path, 'image')):
        os.makedirs(os.path.join(path, 'image'))
    interval = 30
    flow_mean = []
    j = 0
    for i in range(1, len(video_frames)):
        im1 = video_frames[i-1]
        im2 = video_frames[i]
        im1 = np.array(im1)[:, :, ::-1]
        im2 = np.array(im2)[:, :, ::-1]
        img1_copy = im1.copy()
        img2_copy = im2.copy()
        flow = pred_flow(img1_copy, img2_copy)
        flow_color, magnitude = flow_to_color(flow)
        if i % interval == 0 and i <= 1500:
            j += 1
            
            video_frames[i].save(path + f'/image/{j}.png', 'PNG')

            if not os.path.exists(os.path.join(path, f'{j}/ori')):
                os.makedirs(os.path.join(path, f'{j}/ori'))
            if not os.path.exists(os.path.join(path, f'{j}/n')):
                os.makedirs(os.path.join(path, f'{j}/n'))

            video_frames[i].save(path + f'/{j}/ori/1.png', 'PNG')
            video_frames[i].save(path + f'/{j}/ori/2.png', 'PNG')
            video_frames[i].save(path + f'/{j}/ori/3.png', 'PNG')
            video_frames[i].save(path + f'/{j}/ori/4.png', 'PNG')
            video_frames[i].save(path + f'/{j}/ori/5.png', 'PNG')
            video_frames[i].save(path + f'/{j}/ori/6.png', 'PNG')
            flow_mean.append(magnitude.mean())
            if i > 25 and i < 1500+25:
                video_frames[i-9].save(path + f'/{j}/n/n1.png', 'PNG')
                video_frames[i-6].save(path + f'/{j}/n/n2.png', 'PNG')
                video_frames[i-3].save(path + f'/{j}/n/n3.png', 'PNG')
                video_frames[i+3].save(path + f'/{j}/n/n4.png', 'PNG')
                video_frames[i+6].save(path + f'/{j}/n/n5.png', 'PNG')
                video_frames[i+9].save(path + f'/{j}/n/n6.png', 'PNG')
                
    flow_mean_np = np.array(flow_mean)
    print(f'flow_mean: {flow_mean_np}', flow_mean_np.shape)
    np.savetxt(os.path.join(path, f"flow_mean.txt"), flow_mean_np, fmt='%.3f')
    return j

def save_tensor_images(tensor, path):
    """
    将形状为 (N, C, H, W) 的 PyTorch 张量保存为多个图像文件。

    参数:
        tensor (torch.Tensor): 形状为 (N, C, H, W) 的 PyTorch 张量。
        mean (list): 用于反归一化的均值列表。
        std (list): 用于反归一化的标准差列表。
        file_prefix (str): 生成的图像文件名的前缀。
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    tensor_np = tensor.numpy()
    # 将数组的形状从 (N, C, H, W) 转换为 (N, H, W, C)
    tensor_np = np.transpose(tensor_np, (0, 2, 3, 1))
    # 反归一化操作
    tensor_np = abs(tensor_np * std)
    min_value = np.amin(tensor_np)
    max_value = np.amax(tensor_np)
    tensor_np = (tensor_np - min_value) / (max_value - min_value)
    tensor_np = tensor_np * 255 
    # 将数据类型转换为 uint8（0-255 的整数值）
    tensor_np = tensor_np.astype(np.uint8)
    # 遍历所有图像并将其保存为文件
    for i, image_np in enumerate(tensor_np):
        # 将 NumPy 数组转换为 PIL 图像对象
        image_pil = Image.fromarray(image_np)
        # 保存图像到文件
        image_pil.save(f'{path}_{i}.png')
