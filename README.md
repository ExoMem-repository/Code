# ExoMem

ExoMem is a memory-management framework for local LLM inference on unified-memory edge devices. It separates model execution into a reusable **shell** and weight **payloads**, keeping operators in PyTorch while managing weights through OS-backed file mappings.

Paper: [ExoMem: OS-Governed Memory for Local Large-LLM Inference on Mobile Edge Devices](2026-MobiCom-ExoMem.pdf).

## 1. Set Up the Environment

- NVIDIA Jetson Orin or AGX Orin running Linux and JetPack.
- Python 3.10.
- CUDA toolkit and C++ build tools.
- CUDA-enabled PyTorch compatible with the installed JetPack and CUDA versions.
- Jetson `tegrastats` for memory profiling.

Install PyTorch from the NVIDIA packages for your JetPack version, then install the Python dependencies. Run the following commands from this repository directory:

```bash
pip install -r requirements.txt
```

The requirements include `transformers==4.57.6`, `accelerate`, `huggingface_hub`, `safetensors`, and `numpy`.

Build the native extension:

```bash
python setup.py build_ext --inplace
```

This produces the `qwen_managed_loader` extension in the repository directory.

## 2. Prepare the Model

Download Qwen2.5-7B-Instruct and split its weights into decoder layers, embeddings, final normalization, and the language-model head:

```bash
python download_split.py \
  --repo-id Qwen/Qwen2.5-7B-Instruct \
  --output-dir qwen_7b_split_weights \
  --dtype float16 \
  --overwrite
```

The output directory contains 28 decoder-layer weight files, the remaining model components, and the configuration and tokenizer files used during inference.

To prepare an existing local checkpoint, use:

```bash
python download_split.py \
  --model-dir /path/to/Qwen2.5-7B-Instruct \
  --output-dir qwen_7b_split_weights \
  --dtype float16 \
  --overwrite
```

## 3. Build the Weight Payloads

Convert the split weights into contiguous FP16 binary payloads and JSON metadata describing each tensor's name, offset, length, and shape:

```bash
python convert.py \
  --weights-dir qwen_7b_split_weights \
  --out-dir qwen_7b_bin \
  --check-values
```

`--check-values` verifies the converted FP16 values against the split weights. Inference reads the model configuration and tokenizer from `qwen_7b_split_weights`, and the payloads and metadata from `qwen_7b_bin`.

## 4. Run Inference

Before a cold-cache run, stop any previous inference process, then flush pending writes and clear the reclaimable Linux page, directory-entry, and inode caches:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

Start inference:

```bash
python exomem.py \
  --model-dir qwen_7b_split_weights \
  --bin-dir qwen_7b_bin \
  --prompt "Explain unified memory in one paragraph." 
```

## 5. Inspect the Profiling Results

Memory profiling runs alongside inference using Jetson `tegrastats`. The report includes:

- **Peak memory:** the largest sampled system RAM increase relative to the initial baseline.
- **Average memory:** the mean sampled system RAM increase relative to the same baseline.
- **Throughput efficiency:** generated tokens per minute divided by the measured peak memory in GB, reported as `tok/min/GB`.

Use `--profile-interval-ms` to configure the sampling interval.

