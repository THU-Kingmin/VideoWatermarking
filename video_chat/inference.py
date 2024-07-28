import os
from utils.config import Config
config_file = "configs/config_7b.json"
cfg = Config.from_file(config_file)

from models.videochat import VideoChat
from utils.easydict import EasyDict
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from models.video_transformers import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms

model = VideoChat(config=cfg.model)
model = model.to(torch.device(cfg.device))
model = model.eval()

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret

def get_context_emb(conv, model, img_list):
    prompt = get_prompt(conv)
    # print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda").input_ids
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])
        
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
      
def answer(conv, model, img_list, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0):
    stop_words_ids = [
        torch.tensor([835]).to("cuda"),
        torch.tensor([2277, 29937]).to("cuda")]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], None])
    embs = get_context_emb(conv, model, img_list)
    outputs = model.llama_model.generate(
        inputs_embeds=embs,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        num_beams=num_beams,
        do_sample=True,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        temperature=temperature,
    )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text, output_token.cpu().numpy()

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=8, return_msg=False):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = 224
    scale_size = 224
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs

vid_path = "./example/yoga.mp4"
# vid_path = "./example/jesse_dance.mp4"
vid, msg = load_video(vid_path, num_segments=8, return_msg=True)
print('msg: ', msg)
# The model expects inputs of shape: T x C x H x W
TC, H, W = vid.shape
video = vid.reshape(1, TC//3, 3, H, W).to("cuda")
img_list = []
image_emb, _ = model.encode_img(video)
img_list.append(image_emb)

chat = EasyDict({
#     "system": "You are an AI assistant. A human gives an image or a video and asks some questions. You should give helpful, detailed, and polite answers.\n",
    "system": "",
    "roles": ("Human", "Assistant"),
    "messages": [],
    "sep": "###"
})
chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
# ask("Describe the video in detail.", chat)
# ask("Is she safe to doing something in the video?", chat)
# ask("Where is she?", chat)
# ask("Who are you?", chat)
# ask("Are you an assistant?", chat)
# ask("How can I learn to do as the video?", chat)
# ask("Tell me what she did at the fourth second", chat)
# ask("Can you provide me some urls to learn to do as the video?", chat)
# ask("List the president of America.", chat)
# ask("What do you feel from the video?", chat)
# ask("Do you think it is funny? Why?", chat)
ask("Explain why it is funny?", chat)
# ask("Explain why the video is ridiculous?", chat)
llm_message = answer(conv=chat, model=model, img_list=img_list, max_new_tokens=1000)[0]
print('llm_message: ', llm_message)