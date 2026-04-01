
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os

from PIL import Image
import math
from qwen_vl_utils import process_vision_info
# import kornia
from transformers import set_seed
from torchvision import transforms
import random
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from .image_utils import process_image, get_prompts_llava_hf,get_prompts_qwen_hf
from .icv_utils import  mask_patches
from transformers import AutoProcessor, LlavaForConditionalGeneration


def get_demos_llava(args, model, processor): 
    # Initialize a list to store the JSON objects
    data = []
    # Open the file and read line by line
    file_path = args.hallu_demo_file
    patch_size = args.patch_size
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a complete JSON object
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.num_demos)

    inputs_images = []
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image(processor.image_processor, image_raw)
        image_tensor = image_tensor.to(model.model.device)
        image_tensor_cd_all_trials = []

        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
                
            image_tensor_cd_all_trials.append(image_tensor_cd)

        inputs_images.append([image_tensor_cd_all_trials, image_tensor])

    input_ids = get_prompts_llava_hf(args, model, processor, data_demos, question)
    return inputs_images, input_ids
from tqdm import tqdm
def get_demos_qwen(args, model, processor): 
    # Initialize a list to store the JSON objects
    data = []
    # Open the file and read line by line
    file_path = args.hallu_demo_file
    patch_size = args.patch_size
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a complete JSON object
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.num_demos)

    inputs_images = []
    temp_image_path = os.path.join(args.data_file, 'train2014', data_demos[0]['image'])
    for i in tqdm(range(len(data_demos))):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "resized_height": 336,
                        "resized_width": 336,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        image_tensor = transforms.ToTensor()(image_inputs[0]).unsqueeze(0)
        image_tensor = image_tensor.to("cuda").to(torch.float16)[0]
        image_tensor_cd_all_trials = []
        for t in range(args.num_trials):
            
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
            
            image_tensor_cd_all_trials.append(processor.image_processor(image_tensor_cd,do_rescale=False))

        inputs_images.append([image_tensor_cd_all_trials, [processor.image_processor(image_tensor,do_rescale=False)]])
    input_ids = get_prompts_qwen_hf(args, model, processor, data_demos, question,temp_image_path)
    return inputs_images, input_ids


if __name__ == "__main__":
    # VTI config
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--mask_ratio", type=float, default=0.99)
    parser.add_argument("--alpha_image", type=float, default=0.4)
    parser.add_argument("--alpha_text", type=float, default=0.0)
    parser.add_argument("--file_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # VTI processing
    input_images, input_ids = get_demos(processor.image_processor, processor,args.file_path, args.num_demos, args.data_file, device, args.num_trials)
