import argparse
import os
import random
import clip
import numpy as np
import torch
from tqdm import tqdm

DEFAULT_RANDOM_SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

def compute_txt_sim(args, clip_model):
    # load predicted text
    with open(os.path.join(args.pred_text_path), 'r') as f:
        pred_text = f.readlines()[:args.num_samples]
        f.close()
    # compute predicted text features
    pred_text_features = []
    with torch.no_grad():
        pred_text_token = clip.tokenize(pred_text, truncate=True).to(device)
        for i in tqdm(range(args.num_samples//args.batch_size), desc="compute pred text features"):
            pred_text_token_idx = pred_text_token[args.batch_size * i : args.batch_size * (i+1)]
            pred_text_token_idx_features = clip_model.encode_text(pred_text_token_idx)
            pred_text_features.append(pred_text_token_idx_features)
    pred_text_features = torch.concat(pred_text_features, dim=0)
    pred_text_features = (pred_text_features / pred_text_features.norm(dim=1, keepdim=True)).detach()
    
    # load target text
    with open(os.path.join(args.tgt_text_path), 'r') as f:
        tgt_text = f.readlines()[:args.num_samples]
        f.close()  
    
    max_char_length = 76
    truncated_texts = [text[:max_char_length] for text in tgt_text]

    # compute target text features
    tgt_text_features = []
    with torch.no_grad():
        tgt_text_token = clip.tokenize(truncated_texts).to(device)
        for i in tqdm(range(args.num_samples//args.batch_size), desc="compute tgt text features"):
            tgt_text_token_idx = tgt_text_token[args.batch_size * i : args.batch_size * (i+1)]
            tgt_text_token_idx_features = clip_model.encode_text(tgt_text_token_idx)
            tgt_text_features.append(tgt_text_token_idx_features)
    tgt_text_features = torch.concat(tgt_text_features, dim=0)
    tgt_text_features = (tgt_text_features / tgt_text_features.norm(dim=1, keepdim=True)).detach()
    
    clip_score = torch.mean(torch.sum(pred_text_features * tgt_text_features, dim=1)).item()

    print(f"Average CLIP Score between pred_text and tgt_text: {np.mean(clip_score):.5f}", '\n')
    return (np.mean(clip_score))
    
if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--pred_text_path", default='/apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir/attack_w0_m16/clean.txt', type=str)
    parser.add_argument("--tgt_text_path",  default='/apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir/attack_w0_m16/answer.txt', type=str)
    parser.add_argument("--clip_encoder", default=None, type=str)
    args = parser.parse_args()

    # load clip
    cos_sim = []
    clip_encoder_list = ['RN50', 'RN101', 'ViT-B/16', 'ViT-B/32', 'ViT-L/14']
    
    if not args.clip_encoder:
        for model_type in clip_encoder_list:
            print(f"Texts encoded by CLIP Encoder {model_type}.")
            clip_model, _ = clip.load(model_type, device=device, download_root='/apdcephfs/share_1275055/lijinmin/taiji_log/AttackVLM/clip_download')
            clip_model.eval()
            clip_score = compute_txt_sim(args, clip_model)
            
            cos_sim.append(clip_score)
        print("Ensemble of CLIP text encoders:", np.mean(cos_sim))
    else:
        print(f"Texts encoded by CLIP Encoder {args.clip_encoder}.")
        clip_model, _ = clip.load(args.clip_encoder, device=device, download_root='/apdcephfs/share_1275055/lijinmin/taiji_log/AttackVLM/clip_download')
        clip_model.eval()
        clip_score = compute_txt_sim(args, clip_model)