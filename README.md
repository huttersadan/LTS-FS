# LTS-FS

This repository contains a unified codebase that combines the previous `nullu` and `vti` branches.

## Project Layout

```text
.
├── dataset/
├── model/
├── scripts/
├── utils/
├── README.md
└── 14033_appendix.pdf
```

`14033_appendix.pdf` is supplementary material and is not required for running the code.

## Environment

Recommended: Python 3.10+

```bash
conda create -n ltsfs python=3.10 -y
conda activate ltsfs

pip install torch torchvision torchaudio
pip install transformers pillow opencv-python tqdm numpy seaborn shortuuid
pip install pycocotools
```

If you run CHAIR evaluation, install `pycocoevalcap` and its dependencies as well.

## Data Path Configuration

Before running, update dataset roots in:

- `dataset/__init__.py`

Some scripts also expect annotation files under `./data/...`. Please adjust paths based on your local setup.

## Usage

### 1) Nullu-style pipeline

1. Extract activations:

```bash
python scripts/model_run_new.py \
  --model_name Qwen_VL_Chat \
  --model_path /path/to/model \
  --dataset lure \
  --split train \
  --sampling random \
  --num_samples 500 \
  --seed 0
```

2. Apply sparse editing:

```bash
python scripts/model_edit_sparse.py \
  --model_name Qwen_VL_Chat \
  --model_path /path/to/model \
  --emb_path /path/to/activations.pkl \
  --top_k_ranks 8 \
  --lowest_layer 16 \
  --highest_layer 32 \
  --alpha 1.0 \
  --save_model_dir ./output/edited_model
```

3. Run response generation and evaluation (example: POPE):

```bash
python scripts/model_response.py \
  --model_name Qwen_VL_Chat \
  --model_path /path/to/edited_or_base_model \
  --dataset pope \
  --split val \
  --sampling random \
  --num_samples 500 \
  --st_idx 0 \
  --ed_idx 3
```

### 2) VTI pipeline

```bash
python scripts/excute_vti.py \
  --model_name Qwen_VL_Chat \
  --model_path /path/to/model \
  --dataset chair \
  --split val \
  --sampling random \
  --num_samples 500 \
  --save_path /path/to/vti_direction_dir \
  --specific_tag vti
```

The VTI script expects these files to exist:

- `vti_visual_direction.pt`
- `vti_textual_direction.pt`

Path format:

```text
{save_path}/{model_name}_alpha_text{alpha_text}_alpha_image{alpha_image}/
```

## Notes

- The unified `model/build_model` supports both default and VTI-specific model implementations.
- `scripts/excute_vti.py` automatically selects the VTI model implementation.
