
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
from PIL import Image
import math

# import kornia
from transformers import set_seed

import random
from .pca import PCA
import torch.nn.functional as F
import numpy as np
#from torchvision import transforms
from typing import List, Tuple

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

def mask_patches(tensor, indices, patch_size=14):
    """
    Creates a new tensor where specified patches are set to the mean of the original tensor.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (C, H, W)
    indices (list of int): Indices of the patches to modify
    patch_size (int): Size of one side of the square patch
    
    Returns:
    torch.Tensor: New tensor with modified patches
    """
    # Clone the original tensor to avoid modifying it
    new_tensor = tensor.clone()

    # Calculate the mean across the spatial dimensions
    mean_values = tensor.mean(dim=(1, 2), keepdim=True)
    
    # Number of patches along the width
    patches_per_row = tensor.shape[2] // patch_size
    total_patches = (tensor.shape[1] // patch_size) * (tensor.shape[2] // patch_size)


    for index in indices:
        # Calculate row and column position of the patch
        row = index // patches_per_row
        col = index % patches_per_row

        # Calculate the starting pixel positions
        start_x = col * patch_size
        start_y = row * patch_size

        # Replace the patch with the mean values
        new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = mean_values.expand(-1, patch_size, patch_size)#new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size].mean(dim=(1, 2), keepdim=True).expand(-1, patch_size, patch_size)# mean_values.expand(-1, patch_size, patch_size)

    return new_tensor
def get_hiddenstates_qwen(model, inputs, image_tensor):
        h_all = []
        with torch.no_grad():
            for example_id in range(len(inputs)):
                embeddings_for_all_styles= []
                for style_id in range(len(inputs[example_id])):
                    if image_tensor is None:
                        h = model(
                                **inputs[example_id][style_id],
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                    else:
                        input_ids = inputs[example_id][style_id]
                        text_len = input_ids.shape[-1]
                        attention_mask = torch.ones(1, text_len, dtype=torch.long).to(model.device)
                        
                        pixel_values = torch.tensor(image_tensor[example_id][-1].pixel_values).to(model.device, dtype=torch.float16)
                        image_grid_thw = torch.tensor(image_tensor[example_id][-1].image_grid_thw).to(model.device)
                        #image_grid_thw = image_tensor[example_id][-1].image_grid_thw
                        h = model(
                                pixel_values =pixel_values,
                                image_grid_thw = image_grid_thw,
                                input_ids = input_ids,
                                attention_mask=attention_mask,
                                #images=image_tensor[example_id][-1].unsqueeze(0).half(),
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                    embedding_token = []
                    for layer in range(len(h)):
                        embedding_token.append(h[layer][:,-1].detach().cpu())
                    
                    embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                    embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all


def get_hiddenstates(model, inputs, image_tensor):
        h_all = []
        with torch.no_grad():
            for example_id in range(len(inputs)):
                embeddings_for_all_styles= []
                for style_id in range(len(inputs[example_id])):
                    if image_tensor is None:
                        h = model(
                                **inputs[example_id][style_id],
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                    else:
                        input_ids = inputs[example_id][style_id]
                        text_len = input_ids.shape[-1]
                        vision_len = 577
                        attention_mask = torch.ones(1, text_len + vision_len, dtype=torch.long).to(model.device)
                        h = model(
                                input_ids = input_ids,
                                pixel_values =image_tensor[example_id][-1].unsqueeze(0).half().cuda(),
                                attention_mask=attention_mask,
                                #images=image_tensor[example_id][-1].unsqueeze(0).half(),
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True).hidden_states

                    embedding_token = []
                    for layer in range(len(h)):
                        embedding_token.append(h[layer][:,-1].detach().cpu())
                    
                    embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                    embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all

def obtain_textual_vti(model, inputs, image_tensor, rank=1):
    if model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        hidden_states = get_hiddenstates_qwen(model, inputs, image_tensor) 
    elif model.__class__.__name__ == "LlavaForConditionalGeneration" or model.__class__.__name__ == "LlavaForConditionalGeneration":
        hidden_states = get_hiddenstates(model, inputs, image_tensor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data) 

    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))#h_pca.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction

def average_tuples(tuples: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    # Check that the input list is not empty
    # tuple [(),(),()]
    if not tuples:
        raise ValueError("The input list of tuples is empty.")

    # Check that all tuples have the same length
    n = len(tuples[0])
    if not all(len(t) == n for t in tuples):
        raise ValueError("All tuples must have the same length.")

    # Initialize a list to store the averaged tensors
    averaged_tensors = []
    
    # Iterate over the indices of the tuples
    for i in range(n):
        # Stack the tensors at the current index and compute the average
        tensors_at_i = torch.stack([t[i].detach().cpu() for t in tuples])
        averaged_tensor = tensors_at_i.mean(dim=0)
        averaged_tensors.append(averaged_tensor)
    #
    # 32  * 576 * 1280
    # Convert the list of averaged tensors to a tuple
    averaged_tuple = tuple(averaged_tensors)
    # [(),(),()] with 32 items.
    return averaged_tuple

def get_visual_hiddenstates(model, image_tensor):
    h_all = []
    with torch.no_grad():
        
        vision_model = model.vision_tower   
        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles= []
            for style_id in range(len(image_tensor[example_id])):
                h = []
                for image_tensor_ in image_tensor[example_id][style_id]:
                  
                    if image_tensor_.shape[0] == 336:
                        image_tensor_ = image_tensor_.unsqueeze(0).repeat(3,1,1)
                    
                    h_ = vision_model(
                        image_tensor_.unsqueeze(0).half().cuda(),
                        output_hidden_states=True,
                        return_dict=True).hidden_states
                    
                        # _, h_ = vision_model(
                        #     image_tensor_.unsqueeze(0).cuda())
                    h.append(h_)
                h = average_tuples(h)
                
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,:].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0)
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))

    del h, embedding_token

    return h_all


def get_visual_hiddenstates_qwen(model, image_tensor):
    h_all = []
    with torch.no_grad():
        
        vision_model = model.visual 
        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles= []
            for style_id in range(len(image_tensor[example_id])):
                h = []
                for image_tensor_ in image_tensor[example_id][style_id]:
                   
                    pixel_values = torch.tensor(image_tensor_.pixel_values).to(model.device, dtype=torch.float16)
                    
                    image_grid_thw = torch.tensor(image_tensor_.image_grid_thw).to(model.device)
 
                    h_,hidden_states_ls = vision_model(
                        hidden_states = pixel_values,
                        grid_thw = image_grid_thw,
                        )
                    
                        # _, h_ = vision_model(
                        #     image_tensor_.unsqueeze(0).cuda())
                    h.append(tuple(hidden_states_ls))
                h = average_tuples(h)
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,:].detach().cpu())
                #embedding_token = torch.cat(embedding_token, dim=0)
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))

    del h, embedding_token

    return h_all

def obtain_visual_vti(model, image_tensor, rank=1):
    if model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        hidden_states = get_visual_hiddenstates_qwen(model, image_tensor)
    elif model.__class__.__name__ == "LlavaForConditionalGeneration":
        hidden_states = get_visual_hiddenstates(model, image_tensor)
    n_layers = len(hidden_states[0][0])
    n_tokens, feat_dim = hidden_states[0][0][0].shape
    num_demonstration = len(hidden_states)
    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        if model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
            top_hidden_states = torch.stack(hidden_states[demonstration_id][0])
            bot_hidden_states = torch.stack(hidden_states[demonstration_id][1])
            h = top_hidden_states.reshape(n_tokens,-1) - bot_hidden_states.reshape(n_tokens,-1)
        elif model.__class__.__name__ == "LlavaForConditionalGeneration":
            h = hidden_states[demonstration_id][0].reshape(n_tokens,-1) - hidden_states[demonstration_id][1].reshape(n_tokens,-1)
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all,dim=1)[:] # n_token (no CLS token) x n_demos x D
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)
    return direction, reading_direction
