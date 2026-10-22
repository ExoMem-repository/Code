import argparse
import filecmp
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


def _load_num_layers(weights_dir: Path, override_layers: int | None) -> int:
    if override_layers is not None:
        return override_layers

    meta_path = weights_dir / "split_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for key in ["num_layers", "num_hidden_layers", "n_layer", "n_layers"]:
            value = meta.get(key)
            if isinstance(value, int) and value > 0:
                return value

    config_path = weights_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        value = config.get("num_hidden_layers")
        if isinstance(value, int) and value > 0:
            return value

    layer_ids = []
    for path in weights_dir.glob("layer_*.pt"):
        stem = path.stem[len("layer_") :]
        if stem.isdigit():
            layer_ids.append(int(stem))

    if layer_ids:
        return max(layer_ids) + 1

    raise ValueError("Unable to infer number of layers. Pass --num-layers.")


def _iter_state_items(state: object) -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(state, torch.Tensor):
        items = [("weight", state)]
    elif isinstance(state, dict):
        items = [(k, v) for k, v in state.items() if isinstance(v, torch.Tensor)]
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(state)!r}")

    items = [(k, v) for k, v in items if "rotary_emb" not in k]
    items.sort(key=lambda item: item[0])
    return items


def convert_one(weights_dir: Path, out_dir: Path, name: str) -> bool:
    src_path = weights_dir / f"{name}.pt"
    if not src_path.exists():
        print(f"[SKIP] {src_path} not found")
        return False

    print(f"[CONVERT] {src_path}")
    state = torch.load(src_path, map_location="cpu", weights_only=False)
    items = list(_iter_state_items(state))

    meta_tensors = []
    flat_list = []
    offset = 0

    for tensor_name, tensor in items:
        flat = tensor.to(torch.float16).contiguous().view(-1)
        length = flat.numel()
        meta_tensors.append(
            {
                "name": tensor_name,
                "shape": list(tensor.shape),
                "dtype": "float16",
                "offset": int(offset),
                "length": int(length),
            }
        )
        flat_list.append(flat)
        offset += length

    if not flat_list:
        print(f"[WARN] no tensors in {src_path}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"

    torch.cat(flat_list, dim=0).numpy().tofile(bin_path)

    meta = {
        "total_length": int(offset),
        "dtype": "float16",
        "tensors": meta_tensors,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  -> {bin_path}")
    print(f"  -> {json_path}")
    return True


def validate_one(weights_dir: Path, out_dir: Path, name: str, check_values: bool) -> bool:
    src_path = weights_dir / f"{name}.pt"
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"

    if not (src_path.exists() and bin_path.exists() and json_path.exists()):
        print(f"[SKIP] missing files for {name}")
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("dtype") != "float16":
        raise ValueError(f"{name}: json dtype mismatch: {meta.get('dtype')}")

    expected_total = int(meta.get("total_length", -1))
    expected_bytes = expected_total * np.dtype(np.float16).itemsize
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{name}: file size mismatch: {actual_bytes} vs {expected_bytes}")

    if not check_values:
        print(f"[CHECK] {name}: size OK")
        return True

    print(f"[CHECK] {name}: metadata and values")
    state = torch.load(src_path, map_location="cpu", weights_only=False)
    items = list(_iter_state_items(state))
    tensors_meta = meta.get("tensors", [])

    if len(tensors_meta) != len(items):
        raise ValueError(f"{name}: tensor count mismatch: {len(tensors_meta)} vs {len(items)}")

    bin_data = np.memmap(bin_path, dtype=np.float16, mode="r")
    for (tensor_name, tensor), tensor_meta in zip(items, tensors_meta):
        if tensor_meta["name"] != tensor_name:
            raise ValueError(f"{name}: tensor name mismatch: {tensor_meta['name']} vs {tensor_name}")
        if list(tensor_meta["shape"]) != list(tensor.shape):
            raise ValueError(f"{name}: shape mismatch for {tensor_name}")

        offset = int(tensor_meta["offset"])
        length = int(tensor_meta["length"])
        actual = bin_data[offset : offset + length]
        expected = tensor.to(torch.float16).contiguous().view(-1).cpu().numpy()
        if actual.shape[0] != expected.shape[0]:
            raise ValueError(f"{name}: length mismatch for {tensor_name}")
        if not np.array_equal(actual, expected):
            diff = np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
            raise ValueError(f"{name}: value mismatch for {tensor_name}, max_abs_diff={diff}")

    print(f"  OK ({len(items)} tensors)")
    return True


def compare_one(out_dir: Path, compare_dir: Path, name: str) -> bool:
    ok = True
    for ext in [".json", ".bin"]:
        actual = out_dir / f"{name}{ext}"
        expected = compare_dir / f"{name}{ext}"
        if not expected.exists():
            print(f"[COMPARE] {name}{ext}: reference missing")
            ok = False
            continue
        if not actual.exists():
            print(f"[COMPARE] {name}{ext}: output missing")
            ok = False
            continue
        if not filecmp.cmp(actual, expected, shallow=False):
            print(f"[COMPARE] {name}{ext}: differs")
            ok = False
        else:
            print(f"[COMPARE] {name}{ext}: OK")
    return ok


def build_names(num_layers: int, only: list[str] | None) -> list[str]:
    if only:
        return only

    names = ["embed_tokens", "norm", "lm_head"]
    names.extend(f"layer_{idx}" for idx in range(num_layers))

    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert split .pt weights to flat float16 .bin files plus .json metadata."
    )
    parser.add_argument("--weights-dir", "--weights_dir", default="./qwen_7b_split_weights")
    parser.add_argument("--out-dir", "--out_dir", default="./qwen_7b_bin")
    parser.add_argument("--num-layers", "--num_layers", type=int, default=None)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Convert only these component names, e.g. --only layer_0 norm.",
    )
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument(
        "--check-values",
        action="store_true",
        help="During validation, compare every value in the bin against the source .pt.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not rewrite .bin/.json; only validate existing output files.",
    )
    parser.add_argument(
        "--compare-dir",
        default=None,
        help="Optional reference bin directory for byte-for-byte comparison.",
    )
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)
    num_layers = _load_num_layers(weights_dir, args.num_layers)
    names = build_names(num_layers, args.only)

    converted = []
    if args.validate_only:
        converted = [name for name in names if (out_dir / f"{name}.bin").exists()]
    else:
        for name in names:
            if convert_one(weights_dir, out_dir, name):
                converted.append(name)

    if not args.skip_validate:
        for name in converted:
            validate_one(weights_dir, out_dir, name, args.check_values)

    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
        failed = [name for name in converted if not compare_one(out_dir, compare_dir, name)]
        if failed:
            raise SystemExit(f"Reference comparison failed for: {failed}")

    print(f"[DONE] {len(converted)} components processed")


if __name__ == "__main__":
    main()
