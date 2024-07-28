python video_chatgpt/eval/run_inference_benchmark_general_slen_flow.py \
    --video_dir /apdcephfs_cq10/share_1275055/lijinmin/datasets/Video/ChatGPT/Test_Videos_ori \
    --gt_file /apdcephfs_cq10/share_1275055/lijinmin/datasets/Video/ChatGPT/Benchmarking_QA/generic_qa_100.json \
    --output_dir /apdcephfs_cq10/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir_100 \
    --version attack_v5 \
    --model-name /apdcephfs_cq10/share_1275055/lijinmin/datasets/Video/ChatGPT/ckpt/LLaVA \
    --projection_path /apdcephfs_cq10/share_1275055/lijinmin/datasets/Video/ChatGPT/ckpt/VideoChatGPT/video_chatgpt-7B.bin \
    --weight_clip 1 \
    --weight_llm 3 \
    --weight_spa 0 \
    --max_modify 16 \
    --num_iter 1000 \
    --seq_len 16 \
    --mask 0 \
    --mask_type flow \
    --type type3