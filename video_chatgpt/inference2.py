from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
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

def video_chatgpt_attack(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len):
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
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0)).logits # [1, N1, 4096] 
    # print('1. attack output_clean', output_clean.shape)
    ### attack
    learning_rate = 0.00001
    num_iter = 100
    weight_loss2 = 1
    seq_len = 10
    #Define the modifier and the optimizer
    modif = torch.Tensor(seq_len, 3, 224, 224).fill_(1/255).half().to('cuda')
    modifier = torch.nn.Parameter(modif, requires_grad=True)
    optimizer = torch.optim.Adam([modifier], lr=learning_rate)
    min_loss = 1e-5
    prev_loss = 1e-5

    model.model.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    model.requires_grad_(False)
    image_clean = Variable(image_clean, requires_grad=False)
    min_in = image_clean.min().detach()
    max_in = image_clean.max().detach()
    print('2. min_in, max_in', min_in, max_in)
    #Frames to be perturbed
    indicator = [1 for _ in range(seq_len)]
    
    # train
    for iter in range(num_iter):
        print('0. modifier', modifier.mean().item(), modifier.max().item(), modifier.min().item())
        image_clean = Variable(image_clean, requires_grad=False)
        image_attack = torch.clamp((modifier[0,:,:,:]+image_clean[0,:,:,:]), min_in, max_in)
        image_attack = torch.unsqueeze(image_attack, 0)
        for ll in range(seq_len-1):
            if indicator[ll+1] == 1:
                mask_temp = torch.clamp((modifier[ll+1,:,:,:]+image_clean[ll+1,:,:,:]), min_in, max_in)
            else:
                mask_temp = image_clean[ll+1,:,:,:]
            mask_temp = torch.unsqueeze(mask_temp,0)
            image_attack = torch.cat((image_attack, mask_temp),0)
        image_forward_outs_attack = vision_tower(image_attack, output_hidden_states=True)
        frame_features_attack = image_forward_outs_attack.hidden_states[-2][:, 1:]
        video_spatio_temporal_features_attack = get_spatio_temporal_features_torch(frame_features_attack)
        output_attack = model.attack(input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features_attack.unsqueeze(0)).logits
        
        # output_clean = torch.softmax(output_clean, dim=-1)
        # output_attack = torch.softmax(output_attack, dim=-1)
        mse_loss = nn.MSELoss()
        loss1 = - mse_loss(output_clean, output_attack)
        # loss2 = 0
        loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow((torch.unsqueeze(image_attack-image_clean, 0)), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        loss = loss1 + weight_loss2 * loss2

        optimizer.zero_grad()
        loss.backward()
        print('0. grad', modifier.grad.mean().item(), modifier.grad.max().item(), modifier.grad.min().item())
        torch.nn.utils.clip_grad_norm_(modifier, 0.1)
        print('0. grad', modifier.grad.mean().item(), modifier.grad.max().item(), modifier.grad.min().item())
        optimizer.step()

        if (iter+1) % 1 == 0: 
            if prev_loss < loss : 
                print(f'1. Iteration: [{iter+1}/{num_iter}], Loss: {loss}(\u25b2), Loss1: {loss1}, Loss2: {loss2}')
            elif prev_loss > loss: 
                print(f'2. Iteration: [{iter+1}/{num_iter}], Loss: {loss}(\u25bc), Loss1: {loss1}, Loss2: {loss2}')
            else: 
                print(f'3. Iteration: [{iter+1}/{num_iter}], Loss: {loss}, Loss1: {loss1}, Loss2: {loss2}')
        prev_loss = loss

        if loss < min_loss:
            if torch.abs(loss-min_loss) < 0.0001:
                print ('Aborting early!')
            min_loss = loss

    print('Train Finished !!!')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_clean = Variable(image_clean, requires_grad=False)
    image_attack = torch.clamp((modifier[0,:,:,:]+image_clean[0,:,:,:]), min_in, max_in)
    image_attack = torch.unsqueeze(image_attack, 0)
    for ll in range(seq_len-1):
        if indicator[ll+1] == 1:
            mask_temp = torch.clamp((modifier[ll+1,:,:,:]+image_clean[ll+1,:,:,:]), min_in, max_in)
        else:
            mask_temp = image_clean[ll+1,:,:,:]
        mask_temp = torch.unsqueeze(mask_temp,0)
        image_attack = torch.cat((image_attack, mask_temp),0)
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

    return outputs_attack
