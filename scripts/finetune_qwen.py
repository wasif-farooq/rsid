#!/usr/bin/env python3
"""QLoRA fine-tune of Qwen2.5-0.5B-Instruct on the hidden-RSI-divergence SFT dataset.

Sized for a 4GB GPU: 4-bit NF4 base weights, LoRA adapters, gradient
checkpointing, small per-device batch size with gradient accumulation.

Usage:
    python scripts/finetune_qwen.py --train-path data/dataset/train.jsonl --val-path data/dataset/val.jsonl
    python scripts/finetune_qwen.py --merge   # after training, merge LoRA into base weights
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_dataset("json", data_files={"train": args.train_path, "validation": args.val_path})
    has_val = len(dataset["validation"]) > 0 and not args.no_eval

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if has_val else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=True,
        max_length=args.max_seq_length,
        assistant_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_val else None,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint == "auto":
        resume_from_checkpoint = get_last_checkpoint(args.output_dir)
        if resume_from_checkpoint is None:
            print(f"--resume-from-checkpoint given but no checkpoint found in {args.output_dir}, starting fresh")
        else:
            print(f"resuming from latest checkpoint: {resume_from_checkpoint}")
    elif resume_from_checkpoint:
        print(f"resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved -> {args.output_dir}")


def merge(args):
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    base_model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base_model, args.output_dir)
    merged = merged.merge_and_unload()
    config.LORA_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(config.LORA_MERGED_DIR))
    tokenizer.save_pretrained(str(config.LORA_MERGED_DIR))
    print(f"Merged model saved -> {config.LORA_MERGED_DIR}")
    print(
        "To serve via Ollama: convert to GGUF with llama.cpp's convert_hf_to_gguf.py, "
        "then `ollama create <name> -f Modelfile` pointing FROM at the .gguf file."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(config.DATASET_DIR / "train.jsonl"))
    parser.add_argument("--val-path", default=str(config.DATASET_DIR / "val.jsonl"))
    parser.add_argument(
        "--output-dir",
        default=str(config.LORA_ADAPTER_DIR),
        help="Where checkpoints and the final adapter are saved. Point this at a mounted "
        "Google Drive path (e.g. /content/drive/MyDrive/rsid-lora) on Colab so checkpoints "
        "survive a runtime disconnect.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=500, help="Save a checkpoint every N steps (in addition to the final save).")
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Keep at most this many checkpoints under the output dir, deleting older ones.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training. Pass with no value to auto-resume from the latest checkpoint "
        "under the LoRA output dir, or give an explicit checkpoint directory path.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip validation entirely. Note: HF Trainer runs one eval pass at the very end of "
        "training regardless of --eval-steps unless eval is fully disabled -- use this flag for "
        "quick timing/smoke runs instead of a huge --eval-steps value.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (trades VRAM for speed -- there's headroom on the 4GB GPU).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge a previously trained LoRA adapter into the base model instead of training.",
    )
    args = parser.parse_args()

    if args.merge:
        merge(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
