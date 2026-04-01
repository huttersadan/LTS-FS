from typing import Optional
import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
from qwen_vl_chat.modeling_qwen import make_context
from model.base import LargeMultimodalModel, create_hook
from qwen_vl_utils import process_vision_info


class Qwen_VL_Chat(LargeMultimodalModel):

    def __init__(self, args):
        self.args = args
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    @torch.no_grad()
    def _basic_forward(self, image_path, prompt, answer=None, return_dict=False):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
       
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        outputs = self.model(
            **inputs,
            return_dict=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict
        )
        return outputs
    
    @torch.no_grad()
    def chat(self, image_path, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

  

        generated_ids = self.model.generate(
            **inputs, 
            do_sample=True if self.args.temperature > 0 else False,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            num_beams=self.args.num_beams,
            max_new_tokens=self.args.max_length
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_texts[0].strip()
        return output_text
    def register_hooks(self):
        self.model.attn_heads, self.model.attn_residual, self.model.mlp_residual, self.model.vit_satt = [], [], [], []
        attn_head_hook = create_hook(self.model.attn_heads, loc='input')
        attn_residual_hook = create_hook(self.model.attn_residual)
        mlp_residual_hook = create_hook(self.model.mlp_residual)
        vit_forward_hook = create_hook(self.model.vit_satt, loc='input')
        self.hooks = []
        
        #for layer in self.model.transformer.h:
        for layer in self.model.model.layers:
            #self.hooks.append(layer.attn.c_proj.register_forward_hook(attn_head_hook))
            self.hooks.append(layer.self_attn.o_proj.register_forward_hook(attn_head_hook))
            #self.hooks.append(layer.attn.register_forward_hook(attn_residual_hook))
            self.hooks.append(layer.self_attn.register_forward_hook(attn_residual_hook))
            #self.hooks.append(layer.mlp.register_forward_hook(mlp_residual_hook))
            self.hooks.append(layer.mlp.register_forward_hook(mlp_residual_hook))

      

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_activations(self, image_path, prompt, answer=None):
        self.register_hooks()
        outputs = self._basic_forward(image_path, prompt, answer, return_dict=True)
        #attn_heads = torch.cat(self.model.attn_heads).reshape(32, -1, 32, 128)   # [32, seq_len, 4096] -> [32, seq_len, 32, 128]
        attn_heads_original = torch.cat(self.model.attn_heads)   # [28, seq_len, 3584]
        attn_heads  = attn_heads_original.reshape(attn_heads_original.shape[0],attn_heads_original.shape[1],attn_heads_original.shape[0],-1)# [28,seq_len,3584]-> [28,seq_len,28,128]
        
        #attn_residual = torch.cat(self.model.attn_residual)   # [32, seq_len, 4096]
        attn_residual = torch.cat(self.model.attn_residual)   # [28, seq_len, 3584]
        
        #mlp_residual = torch.stack(self.model.mlp_residual)   # [32, seq_len, 4096]
        mlp_residual = torch.stack(self.model.mlp_residual)   # [28, seq_len, 3584]

        #hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
        hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [28, seq_len, 3584]
        
        #vit_attn_heads = torch.cat(self.model.vit_satt).reshape(48, -1, 16, 104)   # [48, 1024, 1, 1664] -> [48, 1024, 16, 104]

        self.remove_hooks()
        return hidden_states, mlp_residual, attn_residual, attn_heads #, vit_attn_heads
