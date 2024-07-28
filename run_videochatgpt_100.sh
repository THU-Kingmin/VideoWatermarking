python video_chatgpt/eval/run_inference_benchmark_general_baseline.py \
    --video_dir /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/Test_Videos_ori \
    --gt_file /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/Benchmarking_QA/generic_qa_100.json \
    --output_dir /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir_100 \
    --version baseline \
    --model-name /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/ckpt/LLaVA \
    --projection_path /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/ckpt/VideoChatGPT/video_chatgpt-7B.bin \
    --seq_len 16 \
    --max_modify 16