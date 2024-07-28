import os
import sys
sys.path.append('.')
import argparse
import json
from video_chatgpt.eval.model_utils import load_video
from flow_util import gen_flow2

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--seq_len", type=int, default=16)
   

    return parser.parse_args()

def run_inference(args):
    with open(args.gt_file) as file:
        gt_contents = json.load(file)
    if not os.path.exists(os.path.join(args.output_dir)):
        os.makedirs(os.path.join(args.output_dir))

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    for sample in gt_contents:
        video_name = sample['video_name']
        if video_name != 'v_KlgrI3Ngwv0':
            continue
        answer = sample['A']
        answer_list_txt = []
        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        video_frames = load_video(video_path, num_frm=args.seq_len)
        length = gen_flow2(video_frames, args.output_dir)
        for i in range(length):
            answer_list_txt.append(answer)
        with open(os.path.join(args.output_dir, f"answer.txt"), 'w', encoding='utf-8') as f1:
            for text in answer_list_txt:
                f1.write(text + '\n')

        print('video_name', video_name)
        print('A', answer)

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
