import os
import torch
import json
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import seaborn as sns
import numpy as np
import argparse
from model import build_model
from utils.utils import get_demos_llava, get_demos_qwen
from utils.llm_layers import add_vti_layers
from utils.icv_utils import obtain_visual_vti, obtain_textual_vti
from dataset import build_dataset

def get_model_answer_llava_bench(args, data, model, answer_file):

    with open(answer_file, 'w') as ans_file:
        for ins in tqdm(data):
            image_id = ins['image_id']
            image_path = ins['image_path']
            prompt = ins['question']
            response = model.chat(image_path, prompt)

            out = {
                "image_id": image_id,
                "model_name": args.model_name,
                "question": prompt,
                "caption": response,
            }

            ans_file.write(json.dumps(out) + "\n")

    print(f'----llava_bench----\nSaved responses to {answer_file}')

def get_model_answer_chair(args, data, model, answer_file):

    with open(answer_file, 'w') as ans_file:
        for ins in tqdm(data):
            image_id = ins['image_id']
            image_path = ins['image_path']
            prompt = ins['question']
            response = model.chat(image_path, prompt)

            out = {
                "image_id": image_id,
                "model_name": args.model_name,
                "question": prompt,
                "caption": response,
            }

            ans_file.write(json.dumps(out) + "\n")

    print(f'----CHAIR----\nSaved responses to {answer_file}')


def get_model_answer_pope(args, data, model, answer_file):

    for strategy, sub_data in data.items():

        chat_save_file = answer_file.replace('_chat.jsonl', f'_{strategy}_chat.jsonl')
        result_save_file = answer_file.replace('_chat.jsonl', f'_{strategy}_result.json')
        
        label_list, pred_list = [], []
        with open(chat_save_file, 'w') as ans_file:
            for ins in tqdm(sub_data):
                response = model.chat(ins['image_path'], ins['question']).strip()

                ins['image_path'] = os.path.basename(ins['image_path'])
                ins['response'] = response
                ins['answer'] = 'no' if any(kw in response.lower() for kw in ["no", "not", "false", f"n't"]) else 'yes'

                ans_file.write(json.dumps(ins) + '\n')

    print(f'----POPE----\nSaved responses to {answer_file}')

def save_direction(args):
    device = torch.device("cuda")
    args.use_vti_impl = True
    model = build_model(args)
    if args.model_name == "LLaVA-7B" or args.model_name == "LLaVA-13B":
        input_images, input_ids = get_demos_llava(args, model, model.processor)
    elif args.model_name == "Qwen_VL_Chat":
        input_images, input_ids = get_demos_qwen(args, model, model.processor)
    args.save_dir = os.path.join(args.save_path, "{}_alpha_text{}_alpha_image{}".format(args.model_name, args.alpha_text, args.alpha_image))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    vti_vision, _ = obtain_visual_vti(
                model.model, input_images, rank=1
                )
    visual_direction = vti_vision[1:]

    torch.save(visual_direction, os.path.join(args.save_dir, "vti_visual_direction.pt"))
    

def excute_vti_to_model(args,model):
    device = torch.device("cuda")
    args.save_dir = os.path.join(args.save_path, "{}_alpha_text{}_alpha_image{}".format(args.model_name, args.alpha_text, args.alpha_image))
    visual_direction = torch.load(os.path.join(args.save_dir, "vti_visual_direction.pt")).to(device)
    textual_direction = torch.load(os.path.join(args.save_dir, "vti_textual_direction.pt")).to(device)
    add_vti_layers(model.model, torch.stack([textual_direction], dim=1).cuda(), alpha=[args.alpha_text], sparse_layers_list=args.sparse_layers_list)
    add_vti_layers(model.model.vision_tower, torch.stack([visual_direction], dim=1).cuda(), alpha=[args.alpha_image])
    return model



def main(args):
    args.use_vti_impl = True
    model = build_model(args)
    device = torch.device("cuda")
    args.save_dir = os.path.join(args.save_path, "{}_alpha_text{}_alpha_image{}".format(args.model_name, args.alpha_text, args.alpha_image))
    visual_direction = torch.load(os.path.join(args.save_dir, "vti_visual_direction.pt")).to(device)
    textual_direction = torch.load(os.path.join(args.save_dir, "vti_textual_direction.pt")).to(device)
    if args.specific_tag != "original":
        add_vti_layers(model.model, torch.stack([textual_direction], dim=1).cuda(), alpha=[args.alpha_text], sparse_layers_list=args.sparse_layers_list)
        if args.model_name == "Qwen_VL_Chat":
            add_vti_layers(model.model.visual, torch.stack([visual_direction], dim=1).cuda(), alpha=[args.alpha_image])
        elif args.model_name == "LLaVA-7B" or args.model_name == "LLaVA-13B":
            add_vti_layers(model.model.vision_tower, torch.stack([visual_direction], dim=1).cuda(), alpha=[args.alpha_image])
    data = build_dataset(args.dataset, args.split, args.sampling, args.num_samples, args.st_idx, args.ed_idx)  
    file_name = args.model_path.split('/')[-1] +  "-" + args.specific_tag
    save_dir = f"./eval/{args.dataset}/{file_name}/"
    os.makedirs(save_dir, exist_ok=True)

    model_tag = (
        f"_t={args.temperature}_" if args.temperature != 0.0 else ""
    ) + f"_beam{args.num_beams}_num{args.max_length}_st{args.st_idx}_ed{args.ed_idx}"
    sampling_tag = f"_{args.sampling}{args.num_samples}" if args.num_samples else ""
    save_tag = f"_{args.save}" if args.save else ""

    save_file = os.path.join(
        save_dir,
        f"{args.split}{sampling_tag}{model_tag}{save_tag}_{args.seed}_chat.jsonl"
    )

    if args.dataset == "chair":
        get_model_answer_chair(args, data, model, save_file)
    elif args.dataset == "pope":
        get_model_answer_pope(args, data, model, save_file)

        from calculate_pope import pope_calculation
        pope_calculation(save_dir)
    elif args.dataset == "llava_bench":
        get_model_answer_llava_bench(args, data, model, save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'mPLUG_Owl2','Qwen_VL_Chat',"LLaVA-13B"], default="LLaVA-13B") 
    parser.add_argument("--model_path", default="") 
    parser.add_argument("--dataset", choices=['chair', 'pope', 'opope','mme','llava_bench'], default="chair")
    parser.add_argument("--split", default="val")
    

    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--sampling", choices=['first', 'random'], default='random')

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")

    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=64)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--st_idx", type=int, default=0)
    parser.add_argument("--ed_idx", type=int, default=3)

    parser.add_argument("--num_demos", type=int, default=50)
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--mask_ratio", type=float, default=0.99)
    parser.add_argument("--alpha_image", type=float, default=0.4)
    parser.add_argument("--alpha_text", type=float, default=0.4)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--data_file",type=str,default = "")
    parser.add_argument("--hallu_demo_file", type=str, default="")
    parser.add_argument("--save_path",default="")
    parser.add_argument("--sparse_layers", type=str, default="27, 26, 30, 22, 28, 29, 31, 20, 25, 24, 21, 19, 16, 17, 18, 14")
    parser.add_argument("--specific_tag", type=str, default="")
    args = parser.parse_args()
    args.sparse_layers_list = [int(x) for x in args.sparse_layers.split(",")]
    main(args)
