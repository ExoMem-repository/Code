import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch


TOKENIZER_FILES = [
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
]

DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

QWEN_LAYER_KEY_ORDER = {
    "self_attn.q_proj.weight": 0,
    "self_attn.q_proj.bias": 1,
    "self_attn.k_proj.weight": 2,
    "self_attn.k_proj.bias": 3,
    "self_attn.v_proj.weight": 4,
    "self_attn.v_proj.bias": 5,
    "self_attn.o_proj.weight": 6,
    "mlp.gate_proj.weight": 7,
    "mlp.up_proj.weight": 8,
    "mlp.down_proj.weight": 9,
    "input_layernorm.weight": 10,
    "post_attention_layernorm.weight": 11,
}


def _import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with "
            "`pip install huggingface_hub`."
        ) from exc
    return snapshot_download


def _import_safetensors():
    try:
        from safetensors.torch import safe_open
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: safetensors. Install with `pip install safetensors`."
        ) from exc
    return safe_open


def parse_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key not in DTYPES:
        raise ValueError(f"Unsupported dtype {name!r}. Choose from: {sorted(DTYPES)}")
    return DTYPES[key]


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    return str(dtype).replace("torch.", "")


def download_model(repo_id: str, local_dir: Path, revision: str | None) -> Path:
    snapshot_download = _import_snapshot_download()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    return local_dir


def _copy_tokenizer_and_config_fallback(model_dir: Path, output_dir: Path, dtype: torch.dtype) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.pop("_name_or_path", None)
    config.pop("torch_dtype", None)
    config["dtype"] = dtype_name(dtype)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    for name in TOKENIZER_FILES:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    return config


def copy_tokenizer_and_config(
    model_dir: Path,
    output_dir: Path,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> dict:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError:
        print("[WARN] transformers is not installed; copying raw tokenizer/config files.")
        return _copy_tokenizer_and_config_fallback(model_dir, output_dir, dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(output_dir)

    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(config, "dtype"):
        config.dtype = dtype
    else:
        config.torch_dtype = dtype
    config.save_pretrained(output_dir)
    return config.to_dict()


def split_meta(config: dict, repo_id: str | None, dtype: torch.dtype, method: str) -> dict:
    return {
        "repo_id": repo_id,
        "num_layers": int(config.get("num_hidden_layers", 0)),
        "hidden_size": config.get("hidden_size"),
        "vocab_size": config.get("vocab_size"),
        "torch_dtype": dtype_name(dtype),
        "split_method": method,
    }


def _route_weight(full_name: str) -> tuple[str, str] | None:
    if "rotary_emb" in full_name:
        return None

    if full_name == "model.embed_tokens.weight":
        return "embed_tokens", "weight"

    if full_name == "lm_head.weight":
        return "lm_head", "weight"

    if full_name == "model.norm.weight":
        return "norm", "weight"

    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", full_name)
    if match:
        return f"layer_{int(match.group(1))}", match.group(2)

    return None


def _find_safetensor_files(model_dir: Path) -> dict[str, list[str]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"{index_path} does not contain a weight_map")
        result: dict[str, list[str]] = defaultdict(list)
        for tensor_name, shard_name in weight_map.items():
            result[shard_name].append(tensor_name)
        return dict(result)

    single = model_dir / "model.safetensors"
    if single.exists():
        safe_open = _import_safetensors()
        with safe_open(single, framework="pt", device="cpu") as f:
            return {single.name: list(f.keys())}

    return {}


def _group_safetensors_by_output(shards: dict[str, list[str]]) -> dict[str, list[tuple[str, str, str]]]:
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    unknown = []
    for shard_name, tensor_names in shards.items():
        for full_name in tensor_names:
            routed = _route_weight(full_name)
            if routed is None:
                if "rotary_emb" not in full_name:
                    unknown.append(full_name)
                continue
            out_name, short_name = routed
            grouped[out_name].append((shard_name, full_name, short_name))

    if unknown:
        print("[WARN] Unused tensors:")
        for name in unknown[:20]:
            print(f"  {name}")
        if len(unknown) > 20:
            print(f"  ... {len(unknown) - 20} more")

    return dict(grouped)


def _ordered_entries(out_name: str, entries: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    if out_name.startswith("layer_"):
        return sorted(
            entries,
            key=lambda item: (QWEN_LAYER_KEY_ORDER.get(item[2], 10_000), item[2]),
        )
    return entries


def _expected_outputs(num_layers: int) -> list[str]:
    return ["embed_tokens", "norm", "lm_head"] + [f"layer_{i}" for i in range(num_layers)]


def split_from_safetensors(
    model_dir: Path,
    output_dir: Path,
    dtype: torch.dtype,
    num_layers: int,
    tie_lm_head_if_missing: bool,
    only: set[str] | None = None,
) -> None:
    shards = _find_safetensor_files(model_dir)
    if not shards:
        raise FileNotFoundError(
            f"No safetensors checkpoint found in {model_dir}. "
            "Use --method model for PyTorch checkpoints."
        )

    safe_open = _import_safetensors()
    grouped = _group_safetensors_by_output(shards)

    output_order = _expected_outputs(num_layers)
    if only is not None:
        unknown = sorted(only - set(output_order))
        if unknown:
            raise ValueError(f"Unknown component names in --only: {unknown}")
        output_order = [name for name in output_order if name in only]

    missing = [name for name in output_order if name not in grouped]
    if missing == ["lm_head"] and tie_lm_head_if_missing:
        missing = []
    if missing:
        raise ValueError(f"Missing expected weight groups: {missing}")

    for out_name in output_order:
        entries = grouped.get(out_name)
        if not entries:
            continue

        state = {}
        handles = {}
        with ExitStack() as stack:
            for shard_name, full_name, short_name in _ordered_entries(out_name, entries):
                if shard_name not in handles:
                    shard_path = model_dir / shard_name
                    handles[shard_name] = stack.enter_context(
                        safe_open(shard_path, framework="pt", device="cpu")
                    )
                state[short_name] = handles[shard_name].get_tensor(full_name).to(dtype).contiguous()

        torch.save(state, output_dir / f"{out_name}.pt")
        print(f"[SAVE] {out_name}.pt ({len(state)} tensors)")
        del state

    if "lm_head" not in grouped and tie_lm_head_if_missing and (only is None or "lm_head" in only):
        embed_state = torch.load(output_dir / "embed_tokens.pt", map_location="cpu", weights_only=False)
        torch.save({"weight": embed_state["weight"].clone()}, output_dir / "lm_head.pt")
        print("[SAVE] lm_head.pt tied from embed_tokens.pt")


def _convert_state_dict_dtype(state_dict: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().to(dtype).contiguous() for name, tensor in state_dict.items()}


def split_from_model(
    model_dir: Path,
    output_dir: Path,
    dtype: torch.dtype,
    trust_remote_code: bool,
    only: set[str] | None = None,
) -> None:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install with `pip install transformers`."
        ) from exc

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    core = getattr(model, "model", None)
    if core is None or not hasattr(core, "layers"):
        raise ValueError("Could not locate model.model.layers on this checkpoint.")

    if only is None or "embed_tokens" in only:
        torch.save(
            _convert_state_dict_dtype(core.embed_tokens.state_dict(), dtype),
            output_dir / "embed_tokens.pt",
        )
        print("[SAVE] embed_tokens.pt")

    for idx, layer in enumerate(core.layers):
        name = f"layer_{idx}"
        if only is not None and name not in only:
            continue
        torch.save(
            _convert_state_dict_dtype(layer.state_dict(), dtype),
            output_dir / f"{name}.pt",
        )
        print(f"[SAVE] {name}.pt")

    if only is None or "norm" in only:
        torch.save(_convert_state_dict_dtype(core.norm.state_dict(), dtype), output_dir / "norm.pt")
        print("[SAVE] norm.pt")

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("Could not locate lm_head on this checkpoint.")
    if only is None or "lm_head" in only:
        torch.save(_convert_state_dict_dtype(lm_head.state_dict(), dtype), output_dir / "lm_head.pt")
        print("[SAVE] lm_head.pt")


def clean_generated_files(output_dir: Path) -> None:
    for path in output_dir.glob("*.pt"):
        path.unlink()
    meta = output_dir / "split_meta.json"
    if meta.exists():
        meta.unlink()


def list_existing_generated_files(output_dir: Path) -> Iterable[Path]:
    yield from output_dir.glob("*.pt")
    meta = output_dir / "split_meta.json"
    if meta.exists():
        yield meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Qwen causal LM and split it into per-layer .pt files."
    )
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-dir", default=None, help="Use an existing HF snapshot directory.")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Where to store the downloaded snapshot. Defaults to <output-dir>/raw.",
    )
    parser.add_argument("--output-dir", default="qwen_7b_split_weights")
    parser.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "safetensors", "model"],
        help="safetensors streams shards; model instantiates AutoModelForCausalLM.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Split only selected components, e.g. --only embed_tokens layer_0 norm lm_head.",
    )
    parser.add_argument(
        "--tie-lm-head-if-missing",
        action="store_true",
        help="For tied-embedding checkpoints without a separate lm_head.weight.",
    )
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output_dir / "raw"

    model_dir = Path(args.model_dir) if args.model_dir else raw_dir
    if args.model_dir is None:
        print(f"[DOWNLOAD] {args.repo_id} -> {raw_dir}")
        download_model(args.repo_id, raw_dir, args.revision)

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(list_existing_generated_files(output_dir))
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing[:6])
        raise SystemExit(
            f"{output_dir} already contains generated files ({names}). "
            "Pass --overwrite to replace them."
        )
    if args.overwrite:
        clean_generated_files(output_dir)

    config = copy_tokenizer_and_config(model_dir, output_dir, dtype, args.trust_remote_code)
    num_layers = int(config.get("num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError("config.json does not contain a valid num_hidden_layers value.")
    only = set(args.only) if args.only else None

    method = args.method
    if method == "auto":
        method = "safetensors" if _find_safetensor_files(model_dir) else "model"

    print(f"[SPLIT] method={method}, layers={num_layers}, dtype={dtype_name(dtype)}")
    if method == "safetensors":
        split_from_safetensors(
            model_dir,
            output_dir,
            dtype,
            num_layers,
            args.tie_lm_head_if_missing,
            only,
        )
    else:
        split_from_model(model_dir, output_dir, dtype, args.trust_remote_code, only)

    meta = split_meta(config, args.repo_id if args.model_dir is None else None, dtype, method)
    with open(output_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Split weights saved in {output_dir}")


if __name__ == "__main__":
    main()
