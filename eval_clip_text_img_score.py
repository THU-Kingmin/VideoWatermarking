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

def compute_clip_score(image_folder, text_file, clip_model):
    # Load images from folders
    images = load_images_from_folder(image_folder)
    
    # Load text lines from the text file
    with open(text_file, 'r') as f:
        text_lines = f.readlines()

    # Compute image and text features
    image_features = []
    text_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(images), args.batch_size), desc="Compute image and text features"):
            images_batch = images[i:i + args.batch_size]
            text_batch = text_lines[i:i + args.batch_size]

            images_tensor = torch.stack([preprocess(img) for img in images_batch]).to(device)
            text_tensor = clip.tokenize(text_batch).to(device)

            image_features_batch = clip_model.encode_image(images_tensor)
            text_features_batch = clip_model.encode_text(text_tensor)

            image_features.append(image_features_batch)
            text_features.append(text_features_batch)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    image_features = (image_features / image_features.norm(dim=1, keepdim=True)).detach()
    text_features = (text_features / text_features.norm(dim=1, keepdim=True)).detach()

    clip_scores = torch.sum(image_features * text_features, dim=1).cpu().numpy()

    return clip_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--root_folder", default='path/to/your/root/folder', type=str)
    parser.add_argument("--text_file", default='path/to/your/text/file.txt', type=str)
    parser.add_argument("--clip_encoder", default='ViT-B/32', type=str)
    args = parser.parse_args()

    # Load CLIP model
    clip_model, _ = clip.load(args.clip_encoder, device=device)
    clip_model.eval()

    image_folder = os.path.join(args.root_folder, 'image')
    clip_scores = compute_clip_score(image_folder, args.text_file, clip_model)

    print(f'clip_scores: {clip_scores}', clip_scores.shape)
    np.savetxt(os.path.join(args.root_folder, f"clip_score_mix.txt"), clip_scores, fmt='%.3f')
    print(clip_scores)