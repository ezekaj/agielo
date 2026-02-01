#!/usr/bin/env python3
"""
CLOUD TRAINING SCRIPT
=====================
For use on Vast.ai, RunPod, or Lambda Labs with A100 GPUs.

This script uses PyTorch + Transformers + TRL for full GPU training.
Much faster than local MLX on cloud A100s.

Usage:
    1. Rent an A100 instance on Vast.ai/RunPod
    2. Upload your training_data.jsonl
    3. Run: python train_cloud.py

Requirements (install on cloud):
    pip install torch transformers trl peft datasets accelerate bitsandbytes
"""

import os
import json
import torch
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Full precision on cloud
TRAINING_DATA = "./training_data.jsonl"   # Upload to cloud
OUTPUT_DIR = "./neuro-trained"

# Training hyperparameters (optimized for A100 80GB)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048

# LoRA config (higher rank for cloud = more capacity)
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def load_training_data() -> Dataset:
    """Load and format training data."""
    log("Loading training data...")

    if not os.path.exists(TRAINING_DATA):
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA}")

    # Load JSONL
    examples = []
    with open(TRAINING_DATA) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = data.get("prompt", "").strip()
                completion = data.get("completion", "").strip()

                if prompt and completion and len(completion) >= 30:
                    # Format as chat for Qwen
                    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"
                    examples.append({"text": text})
            except:
                continue

    log(f"Loaded {len(examples)} training examples")

    # Create dataset
    dataset = Dataset.from_list(examples)

    # Split 95/5 train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    log(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    return split


def setup_model():
    """Load model with 4-bit quantization for memory efficiency."""
    log("Loading model...")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    log("Model ready")
    return model, tokenizer


def train(model, tokenizer, dataset):
    """Run training."""
    log("Starting training...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        group_by_length=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,  # Efficient packing for faster training
    )

    # Train
    trainer.train()

    # Save
    log("Saving model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    log("Training complete!")
    return trainer


def merge_and_save(model, tokenizer):
    """Merge LoRA weights and save full model."""
    log("Merging LoRA weights...")

    merged_model = model.merge_and_unload()

    save_path = os.path.join(OUTPUT_DIR, "merged")
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    log(f"Merged model saved to: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("NEURO CLOUD TRAINER")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Training data: {TRAINING_DATA}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    print("=" * 60)
    print()

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        log("WARNING: No GPU detected! Training will be very slow.")

    # Load data
    dataset = load_training_data()

    # Setup model
    model, tokenizer = setup_model()

    # Train
    trainer = train(model, tokenizer, dataset)

    # Merge (optional - adapters can be used directly too)
    try:
        merge_and_save(model, tokenizer)
    except Exception as e:
        log(f"Merge failed (not critical): {e}")
        log("You can use the adapters directly from the 'final' directory")

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"""
Next steps:
1. Download the trained model:
   scp -r {OUTPUT_DIR}/final user@your-mac:~/Desktop/neuro-trained/

2. Convert to MLX format (on Mac):
   mlx_lm.convert --hf-path ~/Desktop/neuro-trained/final -q 4bit

3. Or use directly with transformers:
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("{OUTPUT_DIR}/merged")
""")


if __name__ == "__main__":
    main()
