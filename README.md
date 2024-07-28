### Video Watermarking: Safeguarding Your Video from (Unauthorized) Annotations by Video-based LLMs [ICML Workshop 2024 🔥]
[Paper](https://arxiv.org/pdf/2407.02411) Jinmin Li*, Kuofeng Gao*, Yang Bai, Jingyun Zhang, Shu-Tao Xia

\* Equally contributing first authors
## Installation :wrench:

Following [VideoChatGPT]( https://github.com/mbzuai-oryx/Video-ChatGPT): we recommend setting up a conda environment for the project:
```shell
conda create --name=video_chatgpt python=3.10
conda activate video_chatgpt

git clone https://github.com/mbzuai-oryx/Video-ChatGPT.git
cd Video-ChatGPT
pip install -r requirements.txt

export PYTHONPATH="./:$PYTHONPATH"
```
Additionally, install [FlashAttention](https://github.com/HazyResearch/flash-attention) for training,
```shell
pip install ninja

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout v1.0.7
python setup.py install
```

---
## Training :train:
### for videochatgpt:
- bash run_slen_flow_videochatgpt_100.sh
- bash run_videochatgpt_100.sh
### for videochat:
- bash run_slen_flow_videochat_100.sh
- bash run_videochat_100.sh
---
## Evaluation :bar_chart:
- bash run_eval_caption.sh
- bash run_eval_clip_text.sh
- run_eval_gpt4_slen_100.sh
- run_eval_gpt_slen_100.sh

---
## Dataset:
To do

## Acknowledgements :pray:

+ [VideoChatGPT]( https://github.com/mbzuai-oryx/Video-ChatGPT)

## License :scroll:
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.