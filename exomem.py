from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension.*",
    category=UserWarning,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)

from profiling import JetsonMemoryProfiler

try:
    import qwen_managed_loader
except ImportError:
    qwen_managed_loader = None


DEFAULT_MODEL_DIR = "./qwen_7b_split_weights"
DEFAULT_BIN_DIR = "./qwen_7b_bin"
DEFAULT_PROMPT_PATH = "./prompt.txt"


def require_managed_loader():
    if qwen_managed_loader is None:
        raise RuntimeError(
            "Missing qwen_managed_loader extension. "
            "Build it first with: python setup.py build_ext --inplace"
        )
    return qwen_managed_loader


def clear_module_parameters(module: nn.Module) -> None:
    for sub_module in module.modules():
        for name, param in list(sub_module._parameters.items()):
            if param is None or name == "rotary_emb":
                continue
            if hasattr(sub_module, name):
                try:
                    delattr(sub_module, name)
                except AttributeError:
                    pass
            sub_module._parameters.pop(name, None)


def read_payload_meta(bin_dir: str, payload_name: str) -> dict:
    path = os.path.join(bin_dir, f"{payload_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metadata: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_payload(bin_dir: str, payload_name: str) -> torch.Tensor:
    loader = require_managed_loader()
    meta = read_payload_meta(bin_dir, payload_name)
    bin_path = os.path.join(bin_dir, f"{payload_name}.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Missing payload: {bin_path}")
    return loader.load_bin_to_managed_1d_mmap(
        bin_path, int(meta["total_length"])
    )


def mount_payload_to_module(
    module: nn.Module,
    payload: torch.Tensor,
    bin_dir: str,
    payload_name: str,
) -> None:
    meta = read_payload_meta(bin_dir, payload_name)
    clear_module_parameters(module)

    for info in meta["tensors"]:
        name = info["name"]
        if "rotary_emb" in name:
            continue

        offset = int(info["offset"])
        length = int(info["length"])
        shape = tuple(info["shape"])
        view = payload[offset : offset + length].view(*shape)
        param = nn.Parameter(view, requires_grad=False)

        sub_module = module
        levels = name.split(".")
        for level in levels[:-1]:
            sub_module = getattr(sub_module, level)
        sub_module.register_parameter(levels[-1], param)


def load_and_mount_module(
    module: nn.Module,
    bin_dir: str,
    payload_name: str,
) -> torch.Tensor:
    payload = load_payload(bin_dir, payload_name)
    mount_payload_to_module(module, payload, bin_dir, payload_name)
    return payload


def load_single_weight(bin_dir: str, weight_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = load_payload(bin_dir, weight_name)
    meta = read_payload_meta(bin_dir, weight_name)
    if len(meta["tensors"]) != 1 or meta["tensors"][0]["name"] != "weight":
        raise ValueError(f"{weight_name} must contain exactly one tensor named 'weight'")

    info = meta["tensors"][0]
    offset = int(info["offset"])
    length = int(info["length"])
    shape = tuple(info["shape"])
    return payload, payload[offset : offset + length].view(*shape)


def release_payload(payload: Optional[torch.Tensor], sync: bool = False) -> None:
    if payload is None:
        return
    loader = require_managed_loader()
    if sync or not hasattr(loader, "release_mmap_async_stream"):
        loader.release_mmap_sync(payload)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return

    stream_ptr = torch.cuda.current_stream().cuda_stream
    loader.release_mmap_async_stream(payload, stream_ptr)


def belady_evict_candidate(
    cached_layers: List[int],
    current_step: int,
    current_layer: int,
    num_layers: int,
    total_steps: int,
) -> int:
    if not cached_layers:
        raise ValueError("cached_layers must not be empty")

    trace_len = total_steps * num_layers

    def next_use(layer: int) -> int:
        if layer > current_layer:
            pos = current_step * num_layers + layer
        else:
            pos = (current_step + 1) * num_layers + layer
        if pos >= trace_len:
            return trace_len + 1
        return pos

    return max(cached_layers, key=next_use)


class ExoMemQwen7B:
    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        bin_dir: str = DEFAULT_BIN_DIR,
        cache_size: int = 15,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to run ExoMem")

        self.model_dir = model_dir
        self.bin_dir = bin_dir
        self.cache_size = max(1, int(cache_size))
        self.dtype = dtype
        self.device = torch.device("cuda")

        self.config = AutoConfig.from_pretrained(model_dir)
        self.config._attn_implementation = "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        with init_empty_weights():
            self.shared_block = Qwen2DecoderLayer(self.config, layer_idx=0)
            self.norm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            self.lm_head = nn.Linear(
                self.config.hidden_size, self.config.vocab_size, bias=False
            )
            self.rotary_emb = Qwen2RotaryEmbedding(self.config, device=self.device)

        self.embed_payload: Optional[torch.Tensor] = None
        self.embed_weight: Optional[torch.Tensor] = None
        self.norm_payload: Optional[torch.Tensor] = None
        self.head_payload: Optional[torch.Tensor] = None
        self.layer_cache: Dict[int, torch.Tensor] = {}
        self.last_elapsed_seconds = 0.0
        self.last_generated_tokens = 0

    def load_static_weights(self) -> None:
        if self.embed_payload is None:
            self.embed_payload, self.embed_weight = load_single_weight(
                self.bin_dir, "embed_tokens"
            )

        if self.norm_payload is None:
            self.norm_payload = load_and_mount_module(self.norm, self.bin_dir, "norm")
            self.norm.to(self.device, dtype=self.dtype)

        if self.head_payload is None:
            self.head_payload = load_and_mount_module(self.lm_head, self.bin_dir, "lm_head")
            self.lm_head.to(self.device, dtype=self.dtype)

    def ensure_layer(self, layer_idx: int, step: int, total_steps: int) -> Tuple[bool, float]:
        name = f"layer_{layer_idx}"
        start = time.perf_counter()

        if layer_idx in self.layer_cache:
            mount_payload_to_module(
                self.shared_block, self.layer_cache[layer_idx], self.bin_dir, name
            )
            self.shared_block.to(self.device, dtype=self.dtype)
            return True, (time.perf_counter() - start) * 1000

        if len(self.layer_cache) >= self.cache_size:
            victim = belady_evict_candidate(
                list(self.layer_cache.keys()),
                current_step=step,
                current_layer=layer_idx,
                num_layers=self.config.num_hidden_layers,
                total_steps=total_steps,
            )
            victim_payload = self.layer_cache.pop(victim)
            clear_module_parameters(self.shared_block)
            release_payload(victim_payload, sync=True)
            del victim_payload
            gc.collect()

        payload = load_and_mount_module(self.shared_block, self.bin_dir, name)
        self.shared_block.to(self.device, dtype=self.dtype)
        self.layer_cache[layer_idx] = payload
        gc.collect()
        return False, (time.perf_counter() - start) * 1000

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 5,
        temperature: float = 1.0,
        top_p: float = 1.0,
        verbose: bool = True,
    ) -> str:
        if verbose:
            print("Running....", flush=True)

        self.load_static_weights()
        if self.embed_weight is None:
            raise RuntimeError("Embedding weights were not loaded")

        self.shared_block.eval()
        self.norm.eval()
        self.lm_head.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        generated_ids = input_ids.clone()
        num_layers = self.config.num_hidden_layers
        past_key_values: List[Optional[DynamicCache]] = [None] * num_layers

        start = time.perf_counter()
        for step in range(max_new_tokens):
            is_first_step = step == 0
            cur_input_ids = generated_ids if is_first_step else generated_ids[:, -1:]
            hidden_states = F.embedding(cur_input_ids, self.embed_weight)

            current_seq_len = generated_ids.shape[1]
            if is_first_step:
                position_ids = torch.arange(
                    current_seq_len, dtype=torch.long, device=self.device
                ).unsqueeze(0)
                cache_position = torch.arange(
                    current_seq_len, dtype=torch.long, device=self.device
                )
            else:
                position_ids = torch.tensor(
                    [[current_seq_len - 1]], dtype=torch.long, device=self.device
                )
                cache_position = torch.tensor(
                    [current_seq_len - 1], dtype=torch.long, device=self.device
                )

            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for layer_idx in range(num_layers):
                self.ensure_layer(layer_idx, step, max_new_tokens)
                if past_key_values[layer_idx] is None:
                    past_key_values[layer_idx] = DynamicCache()

                outputs = self.shared_block(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=True,
                    past_key_value=past_key_values[layer_idx],
                    cache_position=cache_position,
                )
                torch.cuda.synchronize()
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states[:, -1, :]) / max(temperature, 1e-6)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs_sorted = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
                remove_sorted = cumulative_probs > top_p
                remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
                remove_sorted[..., 0] = False
                remove = remove_sorted.scatter(1, sorted_indices, remove_sorted)
                logits = logits.masked_fill(remove, float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            generated_ids = torch.cat([generated_ids, next_token_id.to(self.device)], dim=1)

        elapsed = time.perf_counter() - start
        new_tokens = max(1, generated_ids.shape[1] - input_ids.shape[1])
        self.last_elapsed_seconds = elapsed
        self.last_generated_tokens = new_tokens
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def decode_new_tokens(self, prompt: str, output: str) -> str:
        if output.startswith(prompt):
            return output[len(prompt) :]
        return output

    def close(self) -> None:
        clear_module_parameters(self.shared_block)
        clear_module_parameters(self.norm)
        clear_module_parameters(self.lm_head)

        for payload in list(self.layer_cache.values()):
            release_payload(payload, sync=True)
        self.layer_cache.clear()

        release_payload(self.embed_payload, sync=True)
        release_payload(self.norm_payload, sync=True)
        release_payload(self.head_payload, sync=True)
        self.embed_payload = None
        self.embed_weight = None
        self.norm_payload = None
        self.head_payload = None
        gc.collect()


def read_default_prompt(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Explain unified memory in one paragraph."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExoMem Qwen2.5-7B inference runtime")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--bin-dir", default=DEFAULT_BIN_DIR)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument(
        "--cache-size",
        type=int,
        default=15,
        help="Number of decoder-layer payloads retained in the cache. Use 1 for a one-layer decoder cache.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true", help="Hide the running status line.")
    parser.add_argument("--no-profiler", action="store_true", help="Disable Jetson memory profiling.")
    parser.add_argument("--profile-interval-ms", type=int, default=100)
    parser.add_argument("--profile-type", default="exomem_qwen2.5_7b")
    parser.add_argument(
        "--print-full-output",
        action="store_true",
        help="Print the prompt plus completion. By default only the completion suffix is printed.",
    )
    parser.add_argument(
        "--keep-loaded",
        action="store_true",
        help="Keep mmap payloads alive after generation, useful for repeated interactive calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt if args.prompt is not None else read_default_prompt(args.prompt_file)

    profiler = None
    if not args.no_profiler:
        profiler = JetsonMemoryProfiler(
            profiling_type=args.profile_type,
            interval=args.profile_interval_ms,
            verbose=False,
        )
        profiler.start_profiling()

    engine = ExoMemQwen7B(
        model_dir=args.model_dir,
        bin_dir=args.bin_dir,
        cache_size=args.cache_size,
    )
    try:
        output = engine.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=not args.quiet,
        )
        if args.print_full_output:
            print(output)
        else:
            print(engine.decode_new_tokens(prompt, output))
    finally:
        if not args.keep_loaded:
            engine.close()
        if profiler is not None:
            profiler.stop_profiling()
            profiler.print_statistics(
                generated_tokens=engine.last_generated_tokens,
                elapsed_seconds=engine.last_elapsed_seconds,
            )


if __name__ == "__main__":
    main()
