import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import argparse
import cv2
import json
import numpy as np
import random
import pickle
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model import build_model
from utils.halluedit import HalluEdit

def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# pos: halluciation, neg: non_halluciation
def load_embedding_data(pkl_path, loc):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    pos_data, neg_data = [], []
    for entry in data:
        if entry['label'] == 0:
            pos_data.append(entry[loc])
        else:
            neg_data.append(entry[loc])

    if not pos_data:
        raise ValueError("No positive data found.")
    if not neg_data:
        raise ValueError("No negative data found.")

    pos_data = torch.stack(pos_data).float()
    neg_data = torch.stack(neg_data).float()

    if pos_data.size(0) != neg_data.size(0):
        raise ValueError("Positive and negative data sizes do not match.")

    return pos_data, neg_data


def save_model_and_config(tokenizer, edited_model, save_path, model_name, config_paths):

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    
    if model_name == 'MiniGPT4':
        edited_model.llama_model.save_pretrained(save_path)
    else:
        edited_model.save_pretrained(save_path)

    for config_name, config_path in config_paths.items():
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(save_path, config_name))
            print(f"Copied {config_path} to {save_path}")

    print(f'Saved edited model to {save_path}')


def parse_layer_list(s: str):
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_alpha_map(s: str):
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(":")
        out[int(k.strip())] = float(v.strip())
    return out

def generate_alpha_schedule(schedule: str, default_low: int, default_high: int):

    if not schedule:
        return {}
    try:
        mode, rest = schedule.split(":", 1)
        rng, span = rest.split(":", 1)
        low_s, high_s = rng.split("-")
        low, high = int(low_s), int(high_s)
        start_s, end_s = span.split("->")
        a0, a1 = float(start_s), float(end_s)
    except Exception as e:
        raise ValueError(f"Invalid --alpha_schedule format: {schedule}") from e

    if high <= low:
        raise ValueError("alpha_schedule range must have high > low")

    layers = list(range(low, high))
    n = len(layers)
    alphas = {}

    if mode.strip().lower() == "linear":
        for i, L in enumerate(layers):
            t = i/(n-1) if n > 1 else 0.0
            alphas[L] = a0 + (a1 - a0)*t
    elif mode.strip().lower() == "exp":
        ratio = (a1 / a0) if a0 != 0 else 0.0
        for i, L in enumerate(layers):
            t = i/(n-1) if n > 1 else 0.0
            alphas[L] = a0 * (ratio ** t) if a0 != 0 else 0.0
    else:
        raise ValueError(f"Unsupported schedule mode: {mode}")

    return alphas

def build_layer_plan(lowest: int, highest: int, sparse_layers, alpha_default: float,
                     alpha_schedule: str, alpha_per_layer: str):
   
    base_layers = list(range(lowest, highest)) if (lowest != -1 and highest != -1) else []
    if sparse_layers:
        selected_layers = [L for L in parse_layer_list(sparse_layers)]
    else:
        selected_layers = base_layers

    alpha_map = {L: alpha_default for L in selected_layers}
    sched_map = generate_alpha_schedule(alpha_schedule, lowest, highest) if alpha_schedule else {}
    for L, a in sched_map.items():
        if L in alpha_map:
            alpha_map[L] = a
    custom_map = parse_alpha_map(alpha_per_layer) if alpha_per_layer else {}
    for L, a in custom_map.items():
        if L in alpha_map or not selected_layers:
            if L not in alpha_map and not selected_layers:
                selected_layers.append(L)
            alpha_map[L] = a

    selected_layers = sorted(set(selected_layers))
    alpha_map = {L: alpha_map.get(L, alpha_default) for L in selected_layers}

    return selected_layers, alpha_map


def main(args):

    setup_seeds()

    model = build_model(args)
    
    if args.emb_path is not None:
        loc = {
            'mean': 'hidden_states_mean',
            'last': 'hidden_states',
            'mlp_residual': 'mlp_residual'
        }.get(args.ebd)

        pos_data, neg_data = load_embedding_data(args.emb_path, loc=loc)
        print(f'Loading offline embeddings from {args.emb_path}')

    output_dir = os.path.join("./output", args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.lowest_layer == -1 or args.highest_layer == -1:
        selected_layers, alpha_map = build_layer_plan(
            lowest=-1, highest=-1,
            sparse_layers=args.sparse_layers,
            alpha_default=args.alpha,
            alpha_schedule=args.alpha_schedule,
            alpha_per_layer=args.alpha_per_layer
        )
    else:
        selected_layers, alpha_map = build_layer_plan(
            lowest=args.lowest_layer, highest=args.highest_layer,
            sparse_layers=args.sparse_layers,
            alpha_default=args.alpha,
            alpha_schedule=args.alpha_schedule,
            alpha_per_layer=args.alpha_per_layer
        )

    if not selected_layers and not (args.lowest_layer == -1 or args.highest_layer == -1):
        selected_layers = list(range(args.lowest_layer, args.highest_layer))
        alpha_map = {L: args.alpha for L in selected_layers}

    editor = HalluEdit(model=model, ebd=args.ebd, centering=args.centering, alpha=args.alpha,
                       top_k_ranks=args.top_k_ranks, edit_layer_range=None, random_dps=True)

    edited_model = model
    if selected_layers:
        print(f"[Sparse Edit] Selected layers: {selected_layers}")
        for L in selected_layers:
            a = alpha_map.get(L, args.alpha)
            if hasattr(editor, "alpha"):
                editor.alpha = a
            else:
                print("[Warn] editor has no attribute 'alpha'; per-layer alpha may be ignored.")

            layer_range = np.arange(L, L+1)
            print(f"  - Editing layer {L} with alpha={a}")
            edited_model = editor.apply_edit_end_to_end(
                pos_data, neg_data,
                edit_keys=args.edit_keys,
                edit_values=args.edit_values,
                layer_range=layer_range
            )
    else:
        if args.lowest_layer == -1 or args.highest_layer == -1:
            layer_range = None
        else:
            layer_range = np.arange(args.lowest_layer, args.highest_layer)
        print(f"[Dense Edit] Editing layer range: {layer_range}, alpha={args.alpha}")
        editor = HalluEdit(model=model, ebd=args.ebd, centering=args.centering, alpha=args.alpha,
                           top_k_ranks=args.top_k_ranks, edit_layer_range=layer_range, random_dps=True)
        edited_model = editor.apply_edit_end_to_end(
            pos_data, neg_data,
            edit_keys=args.edit_keys, edit_values=args.edit_values,
            layer_range=layer_range
        )
    
    save_dir = args.save_model_dir
    os.makedirs(save_dir, exist_ok=True)

    save_tag = f"-{args.save}" if args.save is not None else ""

    layer_tag = f"sparse-{','.join(map(str, selected_layers))}" if selected_layers else f"range-{args.lowest_layer}-{args.highest_layer}"
    save_name = f"{args.model_name}-top{args.top_k_ranks}-{layer_tag}{save_tag}"
    save_path = os.path.join(args.save_model_dir, save_name)
    
    config_paths = {
        'preprocessor_config.json': os.path.join(args.model_path, 'preprocessor_config.json'),
        'configuration.json': os.path.join(args.model_path, 'configuration.json')
    }

    save_model_and_config(editor.tokenizer, edited_model, save_path, args.model_name, config_paths)

    # Save the layer plan metadata.
    meta = {
        "model_name": args.model_name,
        "top_k_ranks": args.top_k_ranks,
        "lowest_layer": args.lowest_layer,
        "highest_layer": args.highest_layer,
        "sparse_layers": selected_layers,
        "alpha_default": args.alpha,
        "alpha_map": alpha_map,
        "alpha_schedule": args.alpha_schedule,
        "alpha_per_layer": args.alpha_per_layer
    }
    with open(os.path.join(save_path, "edit_plan.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved edit plan to {os.path.join(save_path, 'edit_plan.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Run a model')
    parser.add_argument("--model_name", choices=['LLaVA-7B', 'MiniGPT4', 'mPLUG_Owl2','Qwen_VL_Chat','LLaVA-13B'], default="MiniGPT4") 
    parser.add_argument("--model_path", default="")
    parser.add_argument(
        "--emb_path", type=str, 
        default=""
    ) 

    parser.add_argument("--centering", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=1.0)          
    parser.add_argument("--ebd", choices=['mean', 'last', 'mlp_residual'], default='mean')
    parser.add_argument("--edit_keys", action="store_true", default=False)
    parser.add_argument("--edit_values", action="store_true", default=True)

    parser.add_argument("--top_k_ranks", type=int, default=8)
    parser.add_argument("--lowest_layer", type=int, default=16)
    parser.add_argument("--highest_layer", type=int, default=32)

    parser.add_argument("--save_model_dir", type=str, default="./output/edited_model")
    parser.add_argument("--save", type=str, default="test")

    parser.add_argument("--sparse_layers", type=str, default="")
                        
    parser.add_argument("--alpha_schedule", type=str, default="")
    parser.add_argument("--alpha_per_layer", type=str, default="")

    main(parser.parse_args())
