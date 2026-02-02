"""
Self-Evolution System
=====================

The AI evolves itself through:
1. Learning (no duplicates)
2. Benchmarking (measure progress)
3. MLX Fine-tuning (when improved)
4. Architecture modification (add layers/functions)
5. Reflection (what worked)

This creates TRUE self-improvement, not just RAG.
"""

import os
import sys
import json
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import (
    KNOWLEDGE_DIR, EVOLUTION_DIR, TRAINING_DATA_FILE,
    LEARNED_HASHES_FILE, BENCHMARK_HISTORY_FILE, EVOLUTION_STATE_FILE,
    ADAPTERS_DIR, LLAMA_FACTORY_OUTPUT_DIR, MLX_MODEL_PATH, HF_MODEL_PATH
)


class SelfEvolution:
    """
    Self-evolving AI system.

    Cycle:
    1. Learn 100 unique facts
    2. Re-benchmark
    3. If improved 1%+ → MLX fine-tune
    4. Add new functions/capabilities
    5. Reflect and continue
    """

    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else EVOLUTION_DIR
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Model paths - using centralized config
        self.mlx_model_path = str(MLX_MODEL_PATH)
        self.hf_model_path = HF_MODEL_PATH
        self.adapters_path = ADAPTERS_DIR if not storage_path else self.storage_path / "adapters"
        self.adapters_path.mkdir(parents=True, exist_ok=True)
        self.llama_factory_output = LLAMA_FACTORY_OUTPUT_DIR if not storage_path else self.storage_path / "llama_factory_output"
        self.llama_factory_output.mkdir(parents=True, exist_ok=True)

        # Track learned content hashes (no duplicates)
        self.learned_hashes: Set[str] = set()
        self.learned_hashes_file = LEARNED_HASHES_FILE if not storage_path else self.storage_path / "learned_hashes.json"
        self._load_hashes()

        # Benchmark history
        self.benchmark_history: List[Dict] = []
        self.benchmark_file = BENCHMARK_HISTORY_FILE if not storage_path else self.storage_path / "benchmark_history.json"
        self._load_benchmark_history()

        # Evolution state
        self.state = {
            'current_cycle': 0,
            'facts_this_cycle': 0,
            'facts_per_cycle': 100,
            'baseline_score': None,
            'current_score': None,
            'total_trainings': 0,
            'improvements': [],
            'added_functions': [],
            'train_every_cycle': True  # NEW: train after every cycle
        }
        self.state_file = EVOLUTION_STATE_FILE if not storage_path else self.storage_path / "evolution_state.json"
        self._load_state()

        # Training data - using centralized config
        self.training_data_file = str(TRAINING_DATA_FILE) if not storage_path else os.path.join(str(self.storage_path), "training_data.jsonl")

    def _hash_content(self, content: str) -> str:
        """Create hash of content to detect duplicates."""
        # Normalize: lowercase, remove extra spaces
        normalized = ' '.join(content.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def is_duplicate(self, content: str) -> bool:
        """Check if content was already learned."""
        h = self._hash_content(content)
        return h in self.learned_hashes

    def mark_learned(self, content: str) -> bool:
        """
        Mark content as learned.

        Returns:
            True if content was learned successfully
            False if duplicate or invalid (empty/whitespace-only)
        """
        # Validate content is not empty or whitespace-only
        if not content or not content.strip():
            return False

        h = self._hash_content(content)
        if h in self.learned_hashes:
            return False
        self.learned_hashes.add(h)
        self.state['facts_this_cycle'] += 1
        self._save_hashes()
        return True

    def should_benchmark(self) -> bool:
        """Check if we should run benchmark (every 100 facts)."""
        return self.state['facts_this_cycle'] >= self.state['facts_per_cycle']

    def record_benchmark(self, score: float, details: Dict = None):
        """Record benchmark result."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'cycle': self.state['current_cycle'],
            'facts_learned': len(self.learned_hashes),
            'details': details or {}
        }
        self.benchmark_history.append(result)

        # Update state
        if self.state['baseline_score'] is None:
            self.state['baseline_score'] = score
        self.state['current_score'] = score

        self._save_benchmark_history()
        self._save_state()

        return result

    def get_improvement(self) -> float:
        """Get improvement since baseline (percentage points)."""
        if self.state['baseline_score'] is None or self.state['current_score'] is None:
            return 0.0
        return self.state['current_score'] - self.state['baseline_score']

    def should_train(self, min_improvement: float = 0.01) -> Tuple[bool, str]:
        """
        Check if we should do MLX fine-tuning.

        This method never throws - it returns safe defaults on any error.

        Returns:
            (should_train, reason)
        """
        try:
            improvement = self.get_improvement()

            # Count training pairs
            training_count = self._count_training_pairs()

            if training_count < 10:
                return False, f"Not enough training data ({training_count} pairs, need 10+)"

            # Train if we have enough data (500+) even without improvement
            # This bootstraps the model to USE the knowledge
            total_trainings = self.state.get('total_trainings', 0)
            if training_count >= 500 and total_trainings == 0:
                return True, f"First training with {training_count} pairs - bootstrap learning!"

            if improvement >= min_improvement:
                return True, f"Improved {improvement:.1%} - ready to train!"

            if improvement < 0:
                return False, f"Score decreased by {abs(improvement):.1%} - need more learning"

            return False, f"Only {improvement:.1%} improvement - need {min_improvement:.1%}+"

        except Exception as e:
            # Fail safe - don't train if we can't determine state
            return False, f"Error checking training readiness: {e}"

    def _count_training_pairs(self) -> int:
        """
        Count valid training data pairs.

        Handles corrupted JSONL by counting only valid JSON lines.
        Returns 0 on any file access error.
        """
        if not os.path.exists(self.training_data_file):
            return 0
        try:
            valid_count = 0
            with open(self.training_data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Validate required fields exist and are non-empty
                        if (isinstance(data, dict) and
                            data.get('prompt', '').strip() and
                            data.get('completion', '').strip()):
                            valid_count += 1
                    except json.JSONDecodeError:
                        # Skip corrupted lines silently
                        continue
            return valid_count
        except (IOError, OSError, UnicodeDecodeError) as e:
            print(f"[Evolution] Warning: Could not read training data: {e}")
            return 0

    def start_new_cycle(self):
        """Start a new learning cycle."""
        self.state['current_cycle'] += 1
        self.state['facts_this_cycle'] = 0
        self._save_state()

    def reset_cycle(self, preserve_learned: bool = True) -> Dict:
        """
        Reset the evolution cycle to recover from stuck states.

        Args:
            preserve_learned: If True, keeps learned facts but resets cycle counters.
                            If False, does a full reset including learned content.

        Returns:
            Dict with reset details and previous state info.
        """
        previous_state = {
            'cycle': self.state['current_cycle'],
            'facts_this_cycle': self.state['facts_this_cycle'],
            'total_facts': len(self.learned_hashes),
            'baseline_score': self.state['baseline_score'],
            'current_score': self.state['current_score']
        }

        # Reset cycle-related state
        self.state['current_cycle'] = 0
        self.state['facts_this_cycle'] = 0
        self.state['baseline_score'] = None
        self.state['current_score'] = None

        if not preserve_learned:
            # Full reset - clear all learned content
            self.learned_hashes.clear()
            self.benchmark_history.clear()
            self.state['total_trainings'] = 0
            self.state['improvements'] = []
            self.state['added_functions'] = []
            self._save_hashes()
            self._save_benchmark_history()

        self._save_state()

        return {
            'success': True,
            'preserve_learned': preserve_learned,
            'previous_state': previous_state,
            'message': 'Cycle reset. ' + (
                'Learned facts preserved.' if preserve_learned
                else 'Full reset completed.'
            )
        }

    def run_mlx_training(self) -> Dict:
        """
        Run fine-tuning using MLX (best for Mac + MLX models).

        Your model is MLX format, so we use MLX training directly.
        """
        # Use MLX directly for MLX models (faster on Mac)
        print(f"[Training] Using MLX for your MLX model...")
        result = self._run_mlx_training_fallback()

        # Only try LLaMA Factory if MLX fails AND you have a HuggingFace model
        if not result['success'] and 'mlx' not in self.mlx_model_path.lower():
            print(f"[Training] MLX failed, trying LLaMA Factory...")
            result = self.run_llama_factory_training()

        return result

    def run_llama_factory_training(self) -> Dict:
        """
        Run QLoRA fine-tuning using LLaMA Factory.

        Better than MLX: more methods, better memory handling, web UI.
        """
        result = {
            'success': False,
            'message': '',
            'method': 'llama_factory',
            'timestamp': datetime.now().isoformat()
        }

        # Check training data
        training_count = self._count_training_pairs()
        if training_count < 10:
            result['message'] = f"Not enough training data: {training_count} pairs"
            return result

        # Prepare training data in LLaMA Factory format (ShareGPT)
        train_file = self.storage_path / "train.jsonl"
        self._prepare_llama_factory_data(train_file)

        # Copy dataset_info.json to storage path
        dataset_info = {
            "cognitive_training": {
                "file_name": "train.jsonl",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "messages"
                }
            }
        }
        with open(self.storage_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

        try:
            print(f"\n[LLaMA Factory] Fine-tuning with {training_count} examples...")
            print(f"[LLaMA Factory] Method: QLoRA (4-bit)")
            print(f"[LLaMA Factory] Output: {self.llama_factory_output}")

            cmd = [
                "python3", "-m", "llamafactory.cli", "train",
                "--model_name_or_path", self.hf_model_path,
                "--stage", "sft",
                "--do_train", "true",
                "--finetuning_type", "lora",
                "--quantization_bit", "4",
                "--dataset_dir", str(self.storage_path),
                "--dataset", "cognitive_training",
                "--output_dir", str(self.llama_factory_output),
                "--per_device_train_batch_size", "1",
                "--gradient_accumulation_steps", "4",
                "--learning_rate", "2e-4",
                "--num_train_epochs", "1",
                "--lora_rank", "16",
                "--lora_alpha", "32",
                "--lora_target", "all",
                "--max_length", "1024",
                "--logging_steps", "10",
                "--save_steps", "100",
                "--bf16", "true",
                "--gradient_checkpointing", "true",
                "--overwrite_output_dir", "true"
            ]

            print(f"[LLaMA Factory] Running training...")

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour max
            )

            if process.returncode == 0:
                result['success'] = True
                result['message'] = f"LLaMA Factory training completed! LoRA saved to {self.llama_factory_output}"
                self.state['total_trainings'] += 1
                self.state['improvements'].append({
                    'cycle': self.state['current_cycle'],
                    'training_pairs': training_count,
                    'method': 'llama_factory',
                    'timestamp': datetime.now().isoformat()
                })
                self._save_state()
                print(f"[LLaMA Factory] Success! Training #{self.state['total_trainings']}")
            else:
                result['message'] = f"LLaMA Factory failed: {process.stderr[:500]}"
                print(f"[LLaMA Factory] Error: {process.stderr[:500]}")

        except subprocess.TimeoutExpired:
            result['message'] = "LLaMA Factory training timed out (>1 hour)"
        except FileNotFoundError:
            result['message'] = "llamafactory-cli not found. Run: pip install llamafactory"
        except Exception as e:
            result['message'] = f"LLaMA Factory error: {str(e)}"

        return result

    def _prepare_llama_factory_data(self, output_file: Path):
        """Convert training data to LLaMA Factory ShareGPT format."""
        with open(self.training_data_file, 'r') as f_in:
            with open(output_file, 'w') as f_out:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        # ShareGPT format
                        entry = {
                            "messages": [
                                {"role": "user", "content": data['prompt']},
                                {"role": "assistant", "content": data['completion']}
                            ]
                        }
                        f_out.write(json.dumps(entry) + '\n')
                    except:
                        continue
        print(f"[LLaMA Factory] Prepared data at {output_file}")

    def _run_mlx_training_fallback(self) -> Dict:
        """Fallback to MLX LoRA if LLaMA Factory fails."""
        result = {
            'success': False,
            'message': '',
            'method': 'mlx',
            'timestamp': datetime.now().isoformat()
        }

        training_count = self._count_training_pairs()
        if training_count < 10:
            result['message'] = f"Not enough training data: {training_count} pairs"
            return result

        train_file = self.storage_path / "train.jsonl"
        valid_file = self.storage_path / "valid.jsonl"
        self._prepare_mlx_data(train_file, valid_file)

        try:
            print(f"\n[MLX] Fine-tuning Qwen3 Coder 30B with {training_count} examples...")

            cmd = [
                "python3", "-m", "mlx_lm", "lora",
                "--model", self.mlx_model_path,
                "--train",
                "--data", str(self.storage_path),
                "--adapter-path", str(self.adapters_path),
                "--batch-size", "1",
                "--num-layers", "8",
                "--iters", "100",
                "--learning-rate", "1e-5"
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800
            )

            if process.returncode == 0:
                result['success'] = True
                result['message'] = f"MLX fine-tuning completed! Adapters: {self.adapters_path}"
                self.state['total_trainings'] += 1
                self.state['improvements'].append({
                    'cycle': self.state['current_cycle'],
                    'training_pairs': training_count,
                    'method': 'mlx',
                    'timestamp': datetime.now().isoformat()
                })
                self._save_state()
            else:
                result['message'] = f"MLX failed: {process.stderr[:300]}"

        except Exception as e:
            result['message'] = f"MLX error: {str(e)}"

        return result

    def _prepare_mlx_data(self, train_file: Path, valid_file: Path):
        """Convert training data to MLX format (train.jsonl + valid.jsonl)."""
        all_data = []

        with open(self.training_data_file, 'r') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    # MLX-LM format for Qwen: {"text": "prompt\nresponse"}
                    mlx_entry = {
                        "text": f"<|im_start|>user\n{data['prompt']}<|im_end|>\n<|im_start|>assistant\n{data['completion']}<|im_end|>"
                    }
                    all_data.append(mlx_entry)
                except:
                    continue

        # Split 90/10 for train/valid
        split_idx = int(len(all_data) * 0.9)
        train_data = all_data[:split_idx]
        valid_data = all_data[split_idx:] if split_idx < len(all_data) else all_data[-10:]

        with open(train_file, 'w') as f:
            for entry in train_data:
                f.write(json.dumps(entry) + '\n')

        with open(valid_file, 'w') as f:
            for entry in valid_data:
                f.write(json.dumps(entry) + '\n')

        print(f"[MLX] Prepared {len(train_data)} train, {len(valid_data)} valid examples")

    def add_function(self, name: str, code: str, description: str) -> bool:
        """
        Add a new function to the AI's capabilities.

        This is self-modification - the AI adds new code to itself.
        """
        functions_file = self.storage_path / "added_functions.py"

        try:
            # Append new function
            with open(functions_file, 'a') as f:
                f.write(f"\n\n# Added: {datetime.now().isoformat()}\n")
                f.write(f"# Description: {description}\n")
                f.write(code)
                f.write("\n")

            # Record
            self.state['added_functions'].append({
                'name': name,
                'description': description,
                'timestamp': datetime.now().isoformat()
            })
            self._save_state()

            return True
        except Exception as e:
            print(f"[Evolution] Failed to add function: {e}")
            return False

    def reflect(self) -> str:
        """Generate reflection on current evolution state."""
        improvement = self.get_improvement()
        cycles = self.state['current_cycle']
        facts = len(self.learned_hashes)
        trainings = self.state['total_trainings']

        baseline_str = f"{self.state['baseline_score']:.1%}" if self.state['baseline_score'] is not None else 'N/A'
        current_str = f"{self.state['current_score']:.1%}" if self.state['current_score'] is not None else 'N/A'

        reflection = f"""
=== EVOLUTION REFLECTION ===
Cycle: {cycles}
Facts learned: {facts} (unique)
Baseline score: {baseline_str}
Current score: {current_str}
Improvement: {improvement:+.1%}
MLX trainings: {trainings}
Functions added: {len(self.state['added_functions'])}

"""

        if improvement > 0:
            reflection += f"✓ IMPROVING: Gained {improvement:.1%} since start\n"
        elif improvement < 0:
            reflection += f"✗ REGRESSING: Lost {abs(improvement):.1%} - need different approach\n"
        else:
            reflection += "→ STABLE: No change yet - keep learning\n"

        if trainings > 0:
            reflection += f"✓ TRAINED: Model fine-tuned {trainings} times\n"

        return reflection

    def get_stats(self) -> Dict:
        """Get evolution statistics."""
        return {
            'cycle': self.state['current_cycle'],
            'facts_this_cycle': self.state['facts_this_cycle'],
            'total_facts': len(self.learned_hashes),
            'baseline_score': self.state['baseline_score'],
            'current_score': self.state['current_score'],
            'improvement': self.get_improvement(),
            'trainings': self.state['total_trainings'],
            'functions_added': len(self.state['added_functions'])
        }

    # Persistence methods

    def _load_hashes(self):
        if self.learned_hashes_file.exists():
            try:
                with open(self.learned_hashes_file, 'r') as f:
                    self.learned_hashes = set(json.load(f))
            except:
                self.learned_hashes = set()

    def _save_hashes(self):
        with open(self.learned_hashes_file, 'w') as f:
            json.dump(list(self.learned_hashes), f)

    def _load_benchmark_history(self):
        if self.benchmark_file.exists():
            try:
                with open(self.benchmark_file, 'r') as f:
                    self.benchmark_history = json.load(f)
            except:
                self.benchmark_history = []

    def _save_benchmark_history(self):
        with open(self.benchmark_file, 'w') as f:
            json.dump(self.benchmark_history, f, indent=2)

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    loaded = json.load(f)
                    self.state.update(loaded)
            except:
                pass

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)


# Global instance
_evolution: Optional[SelfEvolution] = None


def get_evolution() -> SelfEvolution:
    """Get global evolution instance."""
    global _evolution
    if _evolution is None:
        _evolution = SelfEvolution()
    return _evolution


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("SELF-EVOLUTION TEST")
    print("=" * 60)

    evo = SelfEvolution("/tmp/test_evolution")

    # Test duplicate detection
    print("\n1. Testing duplicate detection:")
    print(f"   Learn 'AI is great': {evo.mark_learned('AI is great')}")  # True
    print(f"   Learn 'AI is great' again: {evo.mark_learned('AI is great')}")  # False
    print(f"   Learn 'ML is cool': {evo.mark_learned('ML is cool')}")  # True

    # Test benchmark
    print("\n2. Testing benchmark tracking:")
    evo.record_benchmark(0.23, {'math': 0.0, 'logic': 0.1})
    print(f"   Baseline: {evo.state['baseline_score']}")

    # Simulate improvement
    evo.record_benchmark(0.25)
    print(f"   Current: {evo.state['current_score']}")
    print(f"   Improvement: {evo.get_improvement():.1%}")

    # Check if should train
    print("\n3. Should train?")
    should, reason = evo.should_train()
    print(f"   {should}: {reason}")

    # Reflection
    print("\n4. Reflection:")
    print(evo.reflect())

    print("=" * 60)
