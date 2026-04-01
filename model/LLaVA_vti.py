import sys
import os
import json
import cv2
import torch
from torch import nn
import numpy as np
from io import BytesIO
from transformers import TextStreamer
from transformers.generation import BeamSearchDecoderOnlyOutput
from model.base import LargeMultimodalModel

from transformers import AutoProcessor,LlavaForConditionalGeneration

from PIL import Image

class LLaVA(LargeMultimodalModel):
    def __init__(self, args):
        super(LLaVA, self).__init__()
        load_8bit = args.load_8bit 
        load_4bit = args.load_4bit
        
        # Load Model
        #disable_torch_init()

        #model_name = get_model_name_from_path(args.model_path)
        model_name = args.model_name
        self.args = args
        self.model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            #low_cpu_mem_usage=True,
            load_in_8bit=load_8bit,
            load_in_4bit=load_4bit,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        processor.patch_size = 14
        processor.vision_feature_select_strategy = "default"
        self.processor = processor
    def refresh_chat(self):
        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = self.conv.roles
    
    @torch.no_grad()
    def _basic_forward(self, image_path, prompt, answer=None, return_dict=False):
        #self.refresh_chat()
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        
        outputs = self.model(
            **inputs,
            return_dict=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict)
        return outputs
    
    @torch.no_grad()
    def chat(self, image_path, prompt, answer=None, return_dict=False):
        #self.refresh_chat()
        raw_image = Image.open(image_path).convert("RGB")
        prompt = "USER:\n <image>"+prompt + "\nASSISTANT:"
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_length,
                use_cache=True,
                return_dict_in_generate=return_dict,
                output_attentions=return_dict,
                output_hidden_states=return_dict,
                output_scores=return_dict,
                )
        if not return_dict:
            output_texts = self.processor.batch_decode(
                outputs, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            output_text = output_texts[0].split("ASSISTANT:")[-1]
        else:
            output_texts = processor.batch_decode(
                    outputs['sequences'], 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
            output_text = output_texts.split("ASSISTANT:")[-1]
        return output_text
    
   