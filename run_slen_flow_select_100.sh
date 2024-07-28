python video_chatgpt/eval/run_inference_benchmark_general_slen_flow_select.py \
    --video_dir /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/Test_Videos_ori \
    --gt_file /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/Benchmarking_QA/generic_qa_100.json \
    --output_dir /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/output_dir_100 \
    --version attack \
    --model-name /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/ckpt/LLaVA \
    --projection_path /apdcephfs/share_1275055/lijinmin/datasets/Video/ChatGPT/ckpt/VideoChatGPT/video_chatgpt-7B.bin \
    --weight_clip 1 \
    --weight_llm 4 \
    --weight_spa 2 \
    --max_modify 16 \
    --num_iter 1000 \
    --seq_len 32 \
    --mask 80 \
    --mask_type flow_select \
    --type type3