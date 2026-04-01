import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time

import cv2
import json
import numpy as np
from tqdm import tqdm

from model import build_model
from dataset import build_dataset

import random, torch
import torch.backends.cudnn as cudnn
def parse_model_answer(text: str) -> str:
    text = text.strip()
    for ch in ["A", "B", "C", "D"]:
        if ch in text:
            return ch
    if len(text) > 0 and text[0] in "ABCD":
        return text[0]
    return "N"   

def get_model_answer_mmmu(args, data, model, answer_file):
    correct = 0
    idx = 0
    answer_file = answer_file.split(args.split)[0] + answer_file.split(args.split)[1]
    with open(answer_file, 'w') as ans_file:
        for ins in tqdm(data):
            image = ins['image']
            #mage_path = ins['image_path']
            prompt = ins['prompt']
            response = model.chat(image, prompt)

            out = {
                "image": idx,
                "model_name": args.model_name,
                "question": prompt,
                "caption": response,
            }
            idx += 1
            pred = parse_model_answer(response)
            gt = ins['answer']
            if pred == gt:
                correct += 1
            
            print('prompt:{}'.format(prompt))
            print("pred:{}".format(pred))
            print("ground truth:{}".format(gt))
            ans_file.write(json.dumps(out) + "\n")

    print(f'----MMMU----\nSaved responses to {answer_file}')
    print(f'Accuracy: {correct/len(data):.4f}')
    print("Total samples:", len(data))
    result_save_file = answer_file.replace('_chat.jsonl', f'_result.txt')
    with open(result_save_file,'w') as f:
        f.write(f'Accuracy: {correct/len(data):.4f}\n')
        f.write("Total samples:"+ str(len(data)) + '\n')

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
        
def get_model_answer_mme(args, data, model, answer_file):

    for strategy, sub_data in data.items():

        chat_save_file = answer_file.replace('_chat.jsonl', f'_{strategy}.txt')
        
        label_list, pred_list = [], []
        with open(chat_save_file, 'w') as ans_file:
            for ins in tqdm(sub_data):
                response = model.chat(ins['image_path'], ins['question']).strip()

                ins['image_path'] = os.path.basename(ins['image_path'])
                ins['response'] = response
                ins['answer'] = 'no' if any(kw in response.lower() for kw in ["no", "not", "false", f"n't"]) else 'yes'
                text_inst = f"{os.path.basename(ins['image_path'])}\t{ins['question']}\t{ins['label']}\t{response}"
                print(text_inst)
                #result_save_file.write(text_inst+'\n')
                ans_file.write(text_inst + '\n')

    #print(f'----MME----\nSaved responses to {answer_file}')
    print(f'----MME----\nSaved results to {chat_save_file}')
def get_model_answer_opope(args, data, model, answer_file):
    with open(answer_file, "w") as ans_file:
        for idx, image_path in tqdm(enumerate(data), total=len(data)):
            prompt = f"Please describe this image in detail."
            response = model.chat(image_path, prompt).strip()

            img_save = {}
            img_save["image_id"] = image_path.split('/')[-1]
            img_save["caption"] = response

            ans_file.write(json.dumps(img_save) + "\n")

    print(f'----OPOPE----\nSaved responses to {answer_file}')




def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True



def main(args):

    setup_seeds(args.seed)

    model = build_model(args)
    if args.dataset == "mme":
        args.split = args.split.split(',')
        if type(args.split) is str:
            args.split = [args.split]
    data = build_dataset(args.dataset, args.split, args.sampling, args.num_samples, args.st_idx, args.ed_idx)    
    save_dir = f"./eval/{args.dataset}/{args.model_path.split('/')[-1]}/"
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
    save_name = f"{args.split}{sampling_tag}{model_tag}{save_tag}_{args.seed}_chat.jsonl"

    if args.dataset == "chair":
        get_model_answer_chair(args, data, model, save_file)

        from calculate_chair import chair_calculation
        chair_calculation(save_file)

    elif args.dataset == "pope":
        get_model_answer_pope(args, data, model, save_file)

        from calculate_pope import pope_calculation
        pope_calculation(save_dir)

    elif args.dataset == "opope":
        get_model_answer_opope(args, data, model, save_file)
        
        from calculate_opope import opope_calculation
        opope_calculation(save_file)
    elif args.dataset == "mme":
        
        get_model_answer_mme(args, data, model, save_file)

        from mme_calculation import calculate_metrics
        cal = calculate_metrics()
        cal.process_result(save_dir,save_name,args.split)
    elif args.dataset == "llava_bench":
        get_model_answer_llava_bench(args, data, model, save_file)
    elif args.dataset == "mmmu":
        get_model_answer_mmmu(args, data, model, save_file)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'mPLUG_Owl2','Qwen_VL_Chat','LLaVA-13B'], default="MiniGPT4") 
    parser.add_argument("--model_path", default="") 
    parser.add_argument("--dataset", choices=['chair', 'pope', 'opope','mme','llava_bench','mmmu'], default="chair")
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
    # MME
    parser.add_argument("--reference_dir", default="")
    parser.add_argument("--base_dir", default="")
    
    main(parser.parse_args())