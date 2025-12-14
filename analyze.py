import os
import json
import glob
from collections import Counter

import torch

MODEL_DIR = "llama3_8b-W8A8-quant"

def print_json_if_exists(path: str):
    if os.path.exists(path):
        print(f"\n== {os.path.basename(path)} ==")
        with open(path, "r") as f:
            print(json.dumps(json.load(f), indent=2)[:4000])

def inspect_checkpoint(model_dir: str):
    print("\n######## CHECKPOINT INSPECTION ########")

    # Common metadata/config file names used by HF + compressors
    for fname in [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "quantization_config.json",
        "compression_config.json",
        "compressed_tensors_config.json",
        "model.safetensors.index.json",
    ]:
        print_json_if_exists(os.path.join(model_dir, fname))

    # Inspect safetensors dtypes
    try:
        from safetensors.torch import safe_open
    except Exception as e:
        print("\nCould not import safetensors.torch. Install with: pip install safetensors")
        raise

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        print("\nNo *.safetensors shards found in:", model_dir)
        return

    print(f"\nFound {len(shards)} shard(s). Inspecting first shard for dtype summary...")
    dtype_counter = Counter()
    sample_keys = []

    with safe_open(shards[0], framework="pt", device="cpu") as f:
        keys = f.keys()
        for k in keys:
            t = f.get_tensor(k)
            dtype_counter[str(t.dtype)] += 1
            # pick a few common transformer weight tensors to show explicitly
            if (("layers.0" in k or "model.layers.0" in k) and k.endswith(".weight")):
                sample_keys.append(k)
            if len(sample_keys) >= 10:
                break

    print("\nDtype distribution (first shard):")
    for dt, c in dtype_counter.most_common():
        print(f"  {dt:>18} : {c}")

    # Print a few representative layer weights
    if sample_keys:
        print("\nSample layer-0 weight tensor dtypes:")
        with safe_open(shards[0], framework="pt", device="cpu") as f:
            for k in sample_keys[:10]:
                t = f.get_tensor(k)
                print(f"  {k}  ->  {t.dtype}  {tuple(t.shape)}")

    # Heuristic hints
    print("\nHeuristic hints:")
    dts = set(dtype_counter.keys())
    if any("float8" in dt for dt in dts):
        print("  - float8 tensors present: checkpoint likely contains FP8 weights.")
    if "torch.int8" in dts or "torch.uint8" in dts:
        print("  - int8/uint8 tensors present: could be INT8 weights or packed low-bit weights + scales.")
    if "torch.int32" in dts:
        print("  - int32 tensors present: often used for packed 4-bit/3-bit formats (GPTQ/AWQ-style).")
    print("  - To be certain, also verify runtime backend selection below.")

def inspect_runtime(model_dir: str):
    print("\n######## RUNTIME INSPECTION (vLLM) ########")

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA"
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    print(f"GPU: {name}")
    print(f"Compute capability: {cc}")
    supports_true_fp8_act = cc >= (8, 9)  # rule-of-thumb used by vLLM docs (Ada/Hopper+)
    print(f"Supports true FP8 activations (W8A8 FP8 kernels): {supports_true_fp8_act}")

    #   VLLM_LOG_LEVEL=DEBUG python analyze.py
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=1,
        dtype="float16", 
    )

    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        print("Could not access llm.llm_engine (API changed). Use DEBUG logs instead.")
    else:
        # Try common locations for quantization info
        possible = []
        for attr in ["model_config", "vllm_config"]:
            obj = getattr(engine, attr, None)
            if obj is not None:
                possible.append((attr, obj))

        print("\nEngine config (best-effort):")
        printed_any = False
        for name, obj in possible:
            for key in ["quantization", "dtype", "kv_cache_dtype", "load_format"]:
                if hasattr(obj, key):
                    print(f"  {name}.{key} = {getattr(obj, key)}")
                    printed_any = True
        if not printed_any:
            print("  (No known quantization fields found via introspection; rely on DEBUG logs.)")

    sp = SamplingParams(max_tokens=8, temperature=0.0)
    out = llm.generate(["Quantization check."], sp, use_tqdm=False)
    print("\nRan a tiny generation. Output:")
    print(out[0].outputs[0].text)

if __name__ == "__main__":
    inspect_checkpoint(MODEL_DIR)
    inspect_runtime(MODEL_DIR)

    print("\nTIP: Run with verbose backend logs:")
    print("  VLLM_LOG_LEVEL=DEBUG python confirm_quant.py")
    print("Look for lines indicating FP8 backend / Marlin / fallback behavior.")
