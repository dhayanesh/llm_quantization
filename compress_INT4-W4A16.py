# # prepare calibration data
# from datasets import load_dataset
# NUM_CALIBRATION_SAMPLES=512
# MAX_SEQUENCE_LENGTH=2048

# ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
# ds = ds.shuffle(seed=42)

# def preprocess(example):
#     return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False,)}
# ds = ds.map(preprocess)

# def tokenize(sample):
#     return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
# ds = ds.map(tokenize, remove_columns=ds.column_names)


#load model and quantize
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "llama3_8b"
SAVE_DIR = MODEL_ID + "-W8A8-quant"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    recipe=recipe,
    # dataset=ds,
    # max_seq_length=MAX_SEQUENCE_LENGTH,
    # num_calibration_samples=NUM_CALIBRATION_SAMPLES, 
    output_dir=SAVE_DIR,
)

print("Quantized model saved at:", SAVE_DIR)
