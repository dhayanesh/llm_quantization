import os
import json
import glob
import argparse
from collections import Counter

import torch


def load_json_if_exists(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def inspect_checkpoint(model_dir: str, result: dict):
    print("\n######## CHECKPOINT INSPECTION ########")
    print(f"Model dir: {model_dir}")

    ckpt = {
        "model_dir": model_dir,
        "metadata_files": {},
        "safetensors": {},
        "heuristics": {},
    }

    for fname in [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "quantization_config.json",
        "compression_config.json",
        "compressed_tensors_config.json",
        "model.safetensors.index.json",
    ]:
        path = os.path.join(model_dir, fname)
        data = load_json_if_exists(path)
        if data is not None:
            ckpt["metadata_files"][fname] = data
            print(f"\n== {fname} ==")
            print(json.dumps(data, indent=2)[:4000])

    from safetensors.torch import safe_open

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        print("\nNo *.safetensors shards found.")
        result["checkpoint"] = ckpt
        return

    dtype_counter = Counter()
    sample_tensors = []

    with safe_open(shards[0], framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            dtype_counter[str(t.dtype)] += 1
            if (
                ("layers.0" in k or "model.layers.0" in k)
                and k.endswith(".weight")
                and len(sample_tensors) < 10
            ):
                sample_tensors.append(
                    {
                        "name": k,
                        "dtype": str(t.dtype),
                        "shape": list(t.shape),
                    }
                )

    ckpt["safetensors"] = {
        "num_shards": len(shards),
        "dtype_distribution": dict(dtype_counter),
        "sample_layer0_weights": sample_tensors,
    }

    dts = set(dtype_counter.keys())
    ckpt["heuristics"] = {
        "has_fp8": any("float8" in dt for dt in dts),
        "has_int8": "torch.int8" in dts or "torch.uint8" in dts,
        "has_int32": "torch.int32" in dts,
    }

    print("\nDtype distribution (first shard):")
    for dt, c in dtype_counter.most_common():
        print(f"  {dt:>18} : {c}")

    result["checkpoint"] = ckpt


def inspect_runtime(model_dir: str, result: dict):
    print("\n######## RUNTIME INSPECTION (vLLM) ########")

    runtime = {}

    if torch.cuda.is_available():
        runtime["gpu_name"] = torch.cuda.get_device_name(0)
        runtime["compute_capability"] = list(torch.cuda.get_device_capability(0))
    else:
        runtime["gpu_name"] = "NO CUDA"
        runtime["compute_capability"] = [0, 0]

    runtime["supports_fp8_activation"] = tuple(runtime["compute_capability"]) >= (8, 9)

    print(f"GPU: {runtime['gpu_name']}")
    print(f"Compute capability: {runtime['compute_capability']}")
    print(f"Supports true FP8 activations: {runtime['supports_fp8_activation']}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=1,
        dtype="float16",
    )

    engine_info = {}
    engine = getattr(llm, "llm_engine", None)

    if engine:
        for attr in ["model_config", "vllm_config"]:
            obj = getattr(engine, attr, None)
            if not obj:
                continue
            engine_info[attr] = {}
            for key in ["quantization", "dtype", "kv_cache_dtype", "load_format"]:
                if hasattr(obj, key):
                    engine_info[attr][key] = str(getattr(obj, key))
                    print(f"  {attr}.{key} = {getattr(obj, key)}")

    runtime["engine_config"] = engine_info

    sp = SamplingParams(max_tokens=8, temperature=0.0)
    out = llm.generate(["Quantization check."], sp, use_tqdm=False)

    runtime["test_generation"] = out[0].outputs[0].text
    print("\nTiny generation output:")
    print(runtime["test_generation"])

    result["runtime"] = runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument(
        "--out_dir",
        default="inspect_results",
        help="Directory to store JSON results",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for model_dir in args.models:
        model_name = os.path.basename(model_dir.rstrip("/"))
        print("\n" + "=" * 80)
        print(f"INSPECTING MODEL: {model_name}")
        print("=" * 80)

        result = {
            "model": model_name,
        }

        inspect_checkpoint(model_dir, result)
        inspect_runtime(model_dir, result)

        out_path = os.path.join(
            args.out_dir, f"inspect_{model_name}.json"
        )
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nSaved results to: {out_path}")

    print("\nTIP:")
    print("  VLLM_LOG_LEVEL=DEBUG python confirm_quant.py <models>")
    print("  Look for FP8 / Marlin / fallback messages")


if __name__ == "__main__":
    main()
