"""Model loading and signal generation for the fine-tuned Qwen hidden-RSI-
divergence model.

Shared by scripts/infer_stream.py and scripts/run_paper_trading.py so both
load/prompt the model identically.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from rsid.prompt import build_messages, parse_completion


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.LORA_MERGED_DIR.exists() and any(config.LORA_MERGED_DIR.iterdir()):
        print(f"loading merged fine-tuned model from {config.LORA_MERGED_DIR}")
        path = str(config.LORA_MERGED_DIR)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
        return tokenizer, model

    if config.LORA_ADAPTER_DIR.exists() and any(config.LORA_ADAPTER_DIR.iterdir()):
        from peft import PeftModel

        print(f"loading base model + LoRA adapter from {config.LORA_ADAPTER_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(config.LORA_ADAPTER_DIR))
        base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=torch.bfloat16, device_map=device)
        model = PeftModel.from_pretrained(base, str(config.LORA_ADAPTER_DIR))
        return tokenizer, model

    print("no fine-tuned model found -- falling back to base model (untrained on this task)")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=torch.bfloat16, device_map=device)
    return tokenizer, model


def generate_signal(tokenizer, model, bars: list[dict], event: dict) -> dict:
    messages = build_messages(bars, event)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return parse_completion(text)
