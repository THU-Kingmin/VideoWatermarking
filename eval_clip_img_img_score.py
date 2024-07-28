import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import clip
from PIL import Image

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

# 在加载模型后，定义预处理函数
preprocess = Compose([
    Resize((224, 224), interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
    return images

def compute_img_sim(pred_image_folder, tgt_image_folder, clip_model):
    # Load images from folders
    pred_images = load_images_from_folder(pred_image_folder)
    tgt_images = load_images_from_folder(tgt_image_folder)
    
    # Compute image features
    pred_image_features = []
    tgt_image_features = []

    with torch.no_grad():
        for i in range(0, len(pred_images), args.batch_size):
            pred_images_batch = pred_images[i:i + args.batch_size]
            tgt_images_batch = tgt_images[i:i + args.batch_size]

            pred_images_tensor = torch.stack([preprocess(img) for img in pred_images_batch]).to(device)
            tgt_images_tensor = torch.stack([preprocess(img) for img in tgt_images_batch]).to(device)

            pred_features = clip_model.encode_image(pred_images_tensor)
            tgt_features = clip_model.encode_image(tgt_images_tensor)

            pred_image_features.append(pred_features)
            tgt_image_features.append(tgt_features)

    pred_image_features = torch.cat(pred_image_features, dim=0)
    tgt_image_features = torch.cat(tgt_image_features, dim=0)

    pred_image_features = (pred_image_features / pred_image_features.norm(dim=1, keepdim=True)).detach()
    tgt_image_features = (tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)).detach()

    clip_score = torch.mean(torch.sum(pred_image_features * tgt_image_features, dim=1)).item()

    # print(f"Average CLIP Score between pred_images and tgt_images: {np.mean(clip_score):.5f}", '\n')
    return np.mean(clip_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--pred_image_folder", default='path/to/pred/image/folder', type=str)
    parser.add_argument("--tgt_image_folder", default='path/to/tgt/image/folder', type=str)
    parser.add_argument("--clip_encoder", default='ViT-B/32', type=str)
    args = parser.parse_args()

    # Load CLIP model
    clip_model, _ = clip.load(args.clip_encoder, device=device)
    clip_model.eval()

    clip_score_list = []
    for i in range(1, 51):
        pred_image_folder = args.pred_image_folder + f'{i}/n'
        tgt_image_folder = args.tgt_image_folder + f'{i}/ori'
        # Compute image▍
        clip_score = compute_img_sim(pred_image_folder, tgt_image_folder, clip_model)
        clip_score_list.append(clip_score)
        print(i, clip_score)
    clip_score_list = np.array(clip_score_list)
    print(f'clip_score_list: {clip_score_list}', clip_score_list.shape)
    np.savetxt(os.path.join(args.pred_image_folder, f"clip_score_image.txt"), clip_score_list, fmt='%.3f')
    print(clip_score_list)
    