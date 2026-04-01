

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
import math
# import kornia
from transformers import set_seed
from qwen_vl_utils import process_vision_info
import random
import torch.nn.functional as F
import numpy as np
#from torchvision import transforms


# from lavis.common.gradcam import getAttMap
# from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
# from lavis.models import load_model_and_preprocess
def get_prompts_qwen_hf(args, model, processor, data_demos, question,image_path):
    rs_inputs = []
    qs_pos = question
    qs_neg = question
    for k in data_demos:
        #image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        pos_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        'resized_height': 336,
                        'resized_width': 336,
                    },
                    {"type": "text", "text": qs_pos},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": k['value']}],
            }
        ]
        neg_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        'resized_height': 336,
                        'resized_width': 336,
                    },
                    {"type": "text", "text": qs_neg},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": k['h_value']}],
            }
        ]
        pos_text = processor.apply_chat_template(
            pos_messages, tokenize=False, add_generation_prompt=True
        )
        pos_image_inputs, pos_video_inputs = process_vision_info(pos_messages)
        pos_inputs = processor(
            text=[pos_text],
            images=pos_image_inputs,
            videos=pos_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        pos_inputs = pos_inputs.to("cuda")
        input_ids_positive = pos_inputs['input_ids']
        neg_text = processor.apply_chat_template(
            neg_messages, tokenize=False, add_generation_prompt=True
        )
        neg_image_inputs, neg_video_inputs = process_vision_info(neg_messages)
        neg_inputs = processor(
            text=[neg_text],
            images=neg_image_inputs,
            videos=neg_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        neg_inputs = neg_inputs.to("cuda")
        input_ids_negative = neg_inputs['input_ids']
        rs_inputs.append((input_ids_negative, input_ids_positive))
    inputs = tuple(rs_inputs)
    return inputs

def get_prompts_llava_hf(args, model, processor, data_demos, question):
    rs_inputs = []
    qs_pos = question
    qs_neg = question
    for k in data_demos:
        conversation_pos = "USER:\n <image>"+qs_pos+"\nASSISTANT:\n"+k['value']
        conversation_neg = "USER:\n <image>"+qs_neg+"\nASSISTANT:\n"+k['h_value']
        prompts_positive  = processor.tokenizer(conversation_pos, return_tensors='pt').to(model.device)
        prompts_negative  = processor.tokenizer(conversation_neg, return_tensors='pt').to(model.device)
        input_ids_positive = prompts_positive['input_ids']
        input_ids_negative = prompts_negative['input_ids']
        rs_inputs.append((input_ids_negative, input_ids_positive))
    inputs = tuple(rs_inputs)
    return inputs


def get_prompts_llava_hf_pos(model, processor, data_demos, question):
    rs_inputs = []
    qs_pos = question
    qs_ori = question
    for k in data_demos:
        conversation_pos = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": qs_pos},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": k['value']},
                ],
            },
        ]
        conversation_ori = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": qs_ori},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": k['h_value']},
                ],
            },
        ]
        conversation_pos = processor.apply_chat_template(conversation_pos, add_generation_prompt=True)
        conversation_ori = processor.apply_chat_template(conversation_ori, add_generation_prompt=True)
        prompts_positive  = processor.tokenizer(conversation_pos, return_tensors='pt').to(model.device)
        prompts_negative  = processor.tokenizer(conversation_ori, return_tensors='pt').to(model.device)
        input_ids_positive = prompts_positive['input_ids']
        input_ids_negative = prompts_negative['input_ids']
        rs_inputs.append((input_ids_negative, input_ids_positive))
        inputs = tuple(rs_inputs)
    return inputs



def process_image(image_processor, image_raw):
    answer = image_processor(image_raw)

    # Check if the result is a dictionary and contains 'pixel_values' key
    if 'pixel_values' in answer:
        answer = answer['pixel_values'][0]
    
    # Convert numpy array to torch tensor if necessary
    if isinstance(answer, np.ndarray):
        answer = torch.from_numpy(answer)
    
    # If it's already a tensor, return it directly
    elif isinstance(answer, torch.Tensor):
        return answer
    
    else:
        raise ValueError("Unexpected output format from image_processor.")
    
    return answer


if __name__ == "__main__":
    pass
   