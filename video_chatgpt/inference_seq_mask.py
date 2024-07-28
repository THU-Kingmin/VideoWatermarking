from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
from datetime import datetime
import pytz
shanghai_tz = pytz.timezone('Asia/Shanghai')
from PIL import Image

# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens


def video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len):
    """
    Run inference using the Video-ChatGPT model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Video-ChatGPT model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens.

    Returns:
    dict: Dictionary containing the model's output.
    """

    # Prepare question string for the model
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    inputs = tokenizer([prompt])
    # Preprocess video frames and get image tensor
    image_clean = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

    # Move image tensor to GPU and reduce precision to half
    image_clean = image_clean.half().cuda()
    # print('2. image_clean', image_clean.shape) #[100, 3, 224, 224]
    # Generate video spatio-temporal features
    with torch.no_grad():
        image_forward_outs = vision_tower(image_clean, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
    video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    # print('3. video_spatio_temporal_features', video_spatio_temporal_features.shape) #[356, 1024]
    # print('4. input_ids', input_ids.shape) #[1, N1]
    # Define stopping criteria for generation
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    # print('stop_str', stop_str, stopping_criteria.keywords, stopping_criteria.keyword_ids)
    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    # print('5. output_ids', output_ids.shape) #torch.Size([1, 554]) # [1, N2]
    # Check if output is the same as input
    n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    # print('6. output_ids[:, input_ids.shape[1]:]', output_ids[:, input_ids.shape[1]:]) # [1, N2-N1]
    # Decode output tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    # Clean output string
    outputs = outputs.strip().rstrip(stop_str).strip()

    return outputs

def video_chatgpt_attack_seq(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, args):
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    #video_frames 0～255
    print('video_frames size: ', video_frames[0].size)
    image_clean = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
    image_clean = image_clean.half().cuda() #[100, 3, 224, 224]

    with torch.no_grad():
        image_forward_outs = vision_tower(image_clean, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
        video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    with torch.no_grad():
        output_clean = model.attack(input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0)).logits
   
    #Define the modifier and the optimizer
    modif_max = args.max_modify / 255 / 0.26130258
    step = args.step / 255 / 0.26130258
    print('modif_max', modif_max)
    print('step', step)
    modif = torch.Tensor(args.seq_len, 3, 224, 224).fill_(step).half().to('cuda')
    modifier = torch.nn.Parameter(modif, requires_grad=True)
    prev_loss = 1e-5
    mse_loss = nn.MSELoss()

    model.model.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    model.requires_grad_(False)
    image_clean = Variable(image_clean, requires_grad=False)
    min_in = image_clean.min().detach() #-1.7923
    max_in = image_clean.max().detach() #2.1459

    mask_len = int(args.mask / 100 * args.seq_len)
    mask = torch.zeros(args.seq_len, 3, 224, 224)
    mask[:args.seq_len-mask_len] = 1
    print('mask_len:', mask_len, ', seq_len', args.seq_len)
    mask = mask.half().to('cuda')
    # indicator = [1 for _ in range(args.seq_len)] #Frames to be perturbed
    
    # train
    start_time = time.time()
    for iter in range(args.num_iter):
        image_attack = torch.clamp((modifier * mask + image_clean), min_in, max_in)
        
        image_forward_outs_attack = vision_tower(image_attack, output_hidden_states=True)
        frame_features_attack = image_forward_outs_attack.hidden_states[-2][:, 1:]
        video_spatio_temporal_features_attack = get_spatio_temporal_features_torch(frame_features_attack)
        output_attack = model.attack(input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features_attack.unsqueeze(0)).logits
        
        # if args.type == 'type1':
        loss_clip = - mse_loss(video_spatio_temporal_features, video_spatio_temporal_features_attack)
        # elif args.type == 'type2':
        loss_llm = - mse_loss(output_clean, output_attack)
        # else:
        loss_spa = torch.sum(torch.sqrt(torch.mean(torch.pow((torch.unsqueeze(modifier, 0)), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        loss = args.weight_clip * loss_clip + args.weight_llm * loss_llm + args.weight_spa * loss_spa

        loss.backward()
        grad_sign = torch.sign(modifier.grad)
        modifier.data -= grad_sign * step
        modifier.grad.zero_()
        modifier.data = torch.clamp(modifier.data, -modif_max, modif_max)

        if (iter+1) % 50 == 0: 
            if prev_loss < loss:
                print(f'Iteration: [{iter+1}/{args.num_iter}], Loss: {loss}(\u25b2), loss_clip: {loss_clip}, loss_llm: {loss_llm}, loss_spa: {loss_spa}')
            elif prev_loss > loss: 
                print(f'Iteration: [{iter+1}/{args.num_iter}], Loss: {loss}(\u25bc), loss_clip: {loss_clip}, loss_llm: {loss_llm}, loss_spa: {loss_spa}')
            else: 
                print(f'Iteration: [{iter+1}/{args.num_iter}], Loss: {loss}, loss_clip: {loss_clip}, loss_llm: {loss_llm}, loss_spa: {loss_spa}')
        prev_loss = loss

    print(datetime.now(shanghai_tz).strftime("%Y-%m-%d %H:%M:%S"), "Training time: {:.2f} seconds".format(time.time() - start_time))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_clean = Variable(image_clean, requires_grad=False)
    image_attack = torch.clamp((modifier * mask + image_clean), min_in, max_in)
    print('Delta: ', abs(image_attack-image_clean).mean() * 0.26130258 * 255)
    image_forward_outs_attack = vision_tower(image_attack, output_hidden_states=True)
    frame_features_attack = image_forward_outs_attack.hidden_states[-2][:, 1:]
    video_spatio_temporal_features_attack = get_spatio_temporal_features_torch(frame_features_attack)
    with torch.inference_mode():
        output_ids_attack = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features_attack.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])
    n_diff_input_output = (input_ids != output_ids_attack[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs_attack = tokenizer.batch_decode(output_ids_attack[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    outputs_attack = outputs_attack.strip().rstrip(stop_str).strip()

    return outputs_attack, image_clean, image_attack, abs(image_attack-image_clean).mean() * 0.26130258 * 255

def video_chatgpt_random(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, args):
    print('video_chatgpt_random')
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    #video_frames 0～255
    print('video_frames size: ', video_frames[0].size)
    image_clean = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
    image_clean = image_clean.half().cuda() #[100, 3, 224, 224]

    modif_max = args.max_modify / 255 / 0.26130258
    modifier = (torch.rand(args.seq_len, 3, 224, 224) * modif_max).half().cuda()

    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    min_in = image_clean.min().detach() #-1.7923
    max_in = image_clean.max().detach() #2.1459
    image_random = torch.clamp((modifier + image_clean), min_in, max_in)

    print('Delta: ', abs(image_random-image_clean).mean() * 0.26130258 * 255)
    image_forward_outs_random = vision_tower(image_random, output_hidden_states=True)
    frame_features_random = image_forward_outs_random.hidden_states[-2][:, 1:]
    video_spatio_temporal_features_random = get_spatio_temporal_features_torch(frame_features_random)
    with torch.inference_mode():
        output_ids_random = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features_random.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])
    n_diff_input_output = (input_ids != output_ids_random[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs_random = tokenizer.batch_decode(output_ids_random[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    outputs_random = outputs_random.strip().rstrip(stop_str).strip()

    return outputs_random, image_clean, image_random, abs(image_random-image_clean).mean() * 0.26130258 * 255

def video_chatgpt_black(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, args):
    print('video_chatgpt_black')
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    #video_frames 0～255
    print('video_frames size: ', video_frames[0].size)

    image_clean = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
    image_clean = image_clean.half().cuda() #[100, 3, 224, 224]

    width, height = video_frames[0].size
    black_image = Image.new("RGB", (width, height))
    video_frames_black = [black_image.copy() for _ in range(len(video_frames))]
    image_black = image_processor.preprocess(video_frames_black, return_tensors='pt')['pixel_values']
    image_black = image_black.half().cuda() #[100, 3, 224, 224]

    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    

    print('Delta: ', abs(image_black-image_clean).mean() * 0.26130258 * 255)
    image_forward_outs_black = vision_tower(image_black, output_hidden_states=True)
    frame_features_black = image_forward_outs_black.hidden_states[-2][:, 1:]
    video_spatio_temporal_features_black = get_spatio_temporal_features_torch(frame_features_black)
    with torch.inference_mode():
        output_ids_black = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features_black.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])
    n_diff_input_output = (input_ids != output_ids_black[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs_black = tokenizer.batch_decode(output_ids_black[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    outputs_black = outputs_black.strip().rstrip(stop_str).strip()

    return outputs_black, image_clean, image_black, abs(image_black-image_clean).mean() * 0.26130258 * 255

def video_chatgpt_white(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len, args):
    print('video_chatgpt_white')
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    #video_frames 0～255
    print('video_frames size: ', video_frames[0].size)

    image_clean = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
    image_clean = image_clean.half().cuda() #[100, 3, 224, 224]

    width, height = video_frames[0].size
    white_image = Image.new("RGB", (width, height), "white")
    video_frames_white = [white_image.copy() for _ in range(len(video_frames))]
    image_white = image_processor.preprocess(video_frames_white, return_tensors='pt')['pixel_values']
    image_white = image_white.half().cuda() #[100, 3, 224, 224]

    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    

    print('Delta: ', abs(image_white-image_clean).mean() * 0.26130258 * 255)
    image_forward_outs_white = vision_tower(image_white, output_hidden_states=True)
    frame_features_white = image_forward_outs_white.hidden_states[-2][:, 1:]
    video_spatio_temporal_features_white = get_spatio_temporal_features_torch(frame_features_white)
    with torch.inference_mode():
        output_ids_white = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features_white.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])
    n_diff_input_output = (input_ids != output_ids_white[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs_white = tokenizer.batch_decode(output_ids_white[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    outputs_white = outputs_white.strip().rstrip(stop_str).strip()

    return outputs_white, image_clean, image_white, abs(image_white-image_clean).mean() * 0.26130258 * 255