# Model Preparation

This document covers model preparation. For the complete setup and inference
pipeline, see `README.md`.

The runtime expects Qwen2.5-7B-Instruct weights in two stages:

1. Split Hugging Face weights into one `.pt` file per component.
2. Convert each `.pt` file into a flat float16 `.bin` plus `.json` metadata.

Install preparation dependencies after installing a CUDA-enabled PyTorch build:

```bash
pip install -r requirements.txt
```

Prepare Qwen2.5-7B-Instruct:

```bash
python download_split.py \
  --repo-id Qwen/Qwen2.5-7B-Instruct \
  --output-dir qwen_7b_split_weights \
  --dtype float16 \
  --overwrite

python convert.py \
  --weights-dir qwen_7b_split_weights \
  --out-dir qwen_7b_bin \
  --check-values
```

If the Hugging Face snapshot has already been downloaded:

```bash
python download_split.py \
  --model-dir /path/to/Qwen2.5-7B-Instruct \
  --output-dir qwen_7b_split_weights \
  --dtype float16 \
  --overwrite
```

The expected Qwen2.5-7B layout is 28 decoder layers plus `embed_tokens`,
`norm`, and `lm_head`.
