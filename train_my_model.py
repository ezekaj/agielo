#!/usr/bin/env python3
"""
NEURO Model Trainer - Simple MLX Fine-tuning
=============================================

This script fine-tunes a base LLM on your collected knowledge.
No ML experience needed - just run it!

What it does:
1. Loads your training data (Q&A pairs you collected)
2. Downloads a base model (Qwen 7B)
3. Fine-tunes it on YOUR data
4. Saves YOUR custom model

Usage:
    python3 train_my_model.py

Requirements:
    pip install mlx-lm
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
BASE_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"  # Good balance of size/quality
TRAINING_DATA = os.path.expanduser("~/.cognitive_ai_knowledge/training_data.jsonl")
OUTPUT_DIR = os.path.expanduser("~/Desktop/neuro-model-trained")
LORA_LAYERS = 16  # How many layers to train (more = better but slower)
ITERATIONS = 1000  # Training steps (more = better but slower)
BATCH_SIZE = 4  # How many examples per step (lower if out of memory)


def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def check_requirements():
    """Check if MLX-LM is installed."""
    print_header("Step 1: Checking Requirements")

    try:
        import mlx_lm
        print("✓ MLX-LM is installed")
        return True
    except ImportError:
        print("✗ MLX-LM not installed")
        print("\nInstalling MLX-LM...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mlx-lm"], check=True)
        print("✓ MLX-LM installed!")
        return True


def prepare_training_data():
    """Convert training data to MLX format."""
    print_header("Step 2: Preparing Training Data")

    if not os.path.exists(TRAINING_DATA):
        print(f"✗ Training data not found at {TRAINING_DATA}")
        print("  Run the learning system first to collect data!")
        return None

    # Count entries
    with open(TRAINING_DATA) as f:
        lines = f.readlines()

    print(f"✓ Found {len(lines)} training examples")

    # MLX expects data in a specific format
    # Convert to chat format
    train_file = "/tmp/neuro_train.jsonl"
    valid_file = "/tmp/neuro_valid.jsonl"

    # Split 90% train, 10% validation
    split_idx = int(len(lines) * 0.9)
    train_lines = lines[:split_idx]
    valid_lines = lines[split_idx:]

    def convert_line(line):
        """Convert Q&A to chat format."""
        try:
            data = json.loads(line)
            prompt = data.get("prompt", "")
            completion = data.get("completion", "")

            if not prompt or not completion:
                return None

            # Chat format for instruction tuning
            return json.dumps({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            })
        except:
            return None

    # Write training file
    with open(train_file, 'w') as f:
        for line in train_lines:
            converted = convert_line(line)
            if converted:
                f.write(converted + '\n')

    # Write validation file
    with open(valid_file, 'w') as f:
        for line in valid_lines:
            converted = convert_line(line)
            if converted:
                f.write(converted + '\n')

    # Count actual examples
    with open(train_file) as f:
        train_count = len(f.readlines())
    with open(valid_file) as f:
        valid_count = len(f.readlines())

    print(f"✓ Training examples: {train_count}")
    print(f"✓ Validation examples: {valid_count}")

    return train_file, valid_file


def train_model(train_file, valid_file):
    """Run MLX fine-tuning."""
    print_header("Step 3: Fine-tuning Model")

    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"This will take a while... (30 min - 2 hours)")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build MLX training command
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", os.path.dirname(train_file),
        "--batch-size", str(BATCH_SIZE),
        "--lora-layers", str(LORA_LAYERS),
        "--iters", str(ITERATIONS),
        "--adapter-path", os.path.join(OUTPUT_DIR, "adapters"),
    ]

    print("Running:", " ".join(cmd))
    print()
    print("-" * 60)

    # Run training
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output
    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode == 0:
        print("-" * 60)
        print("✓ Training complete!")
        return True
    else:
        print("✗ Training failed!")
        return False


def merge_model():
    """Merge LoRA adapters into base model."""
    print_header("Step 4: Creating Final Model")

    adapter_path = os.path.join(OUTPUT_DIR, "adapters")
    final_path = os.path.join(OUTPUT_DIR, "neuro-7b")

    if not os.path.exists(adapter_path):
        print("✗ Adapters not found. Training may have failed.")
        return False

    print("Merging adapters into base model...")
    print("(This creates a standalone model you can use anywhere)")

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", BASE_MODEL,
        "--adapter-path", adapter_path,
        "--save-path", final_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Model saved to: {final_path}")
        return final_path
    else:
        print("Note: Fuse step optional. You can use adapters directly.")
        return adapter_path


def test_model(model_path):
    """Test the trained model."""
    print_header("Step 5: Testing Your Model")

    print("Asking your model a question...")
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", BASE_MODEL,
        "--adapter-path", os.path.join(OUTPUT_DIR, "adapters"),
        "--prompt", "What is machine learning?",
        "--max-tokens", "100",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("Your model says:")
    print("-" * 40)
    print(result.stdout)
    print("-" * 40)


def main():
    print_header("NEURO MODEL TRAINER")
    print("""
Welcome! This script will train YOUR custom AI model.

What's happening:
1. We take a base AI model (Qwen 7B)
2. We teach it everything you've collected
3. You get YOUR own custom model!

No ML experience needed. Just sit back and watch.
""")

    input("Press Enter to start training...")

    # Step 1: Check requirements
    if not check_requirements():
        return

    # Step 2: Prepare data
    result = prepare_training_data()
    if not result:
        return
    train_file, valid_file = result

    # Step 3: Train
    if not train_model(train_file, valid_file):
        return

    # Step 4: Merge
    model_path = merge_model()

    # Step 5: Test
    if model_path:
        test_model(model_path)

    print_header("DONE!")
    print(f"""
Your custom NEURO model is ready!

Location: {OUTPUT_DIR}

To use it:
    # With MLX
    python -m mlx_lm.generate --model {OUTPUT_DIR}/neuro-7b --prompt "Your question"

    # Or load in LM Studio
    Copy {OUTPUT_DIR}/neuro-7b to ~/.cache/lm-studio/models/

To keep improving:
    1. Keep the learning system running (more data = better model)
    2. Re-run this script periodically to train on new data

Congratulations! You just trained an AI! 🎉
""")


if __name__ == "__main__":
    main()
