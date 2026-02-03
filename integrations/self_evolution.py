"""
Self-Evolution System
=====================

The AI evolves itself through:
1. Learning (no duplicates)
2. Benchmarking (measure progress)
3. MLX Fine-tuning (when improved)
4. Architecture modification (add layers/functions)
5. Code self-modification (generate, validate, deploy code)
6. Reflection (what worked)

This creates TRUE self-improvement, not just RAG.
"""

import os
import sys
import json
import hashlib
import subprocess
import time
import atexit
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import (
    KNOWLEDGE_DIR, EVOLUTION_DIR, TRAINING_DATA_FILE,
    LEARNED_HASHES_FILE, BENCHMARK_HISTORY_FILE, EVOLUTION_STATE_FILE,
    ADAPTERS_DIR, LLAMA_FACTORY_OUTPUT_DIR, MLX_MODEL_PATH, HF_MODEL_PATH
)

# Import code evolution system for true self-modification
try:
    from integrations.code_evolution import (
        CodeEvolution, CodeChangeType, ValidationResult,
        get_code_evolution
    )
    CODE_EVOLUTION_AVAILABLE = True
except ImportError:
    CODE_EVOLUTION_AVAILABLE = False
    print("[SelfEvolution] Warning: code_evolution not available, self-modification disabled")


# Track instances for atexit cleanup using weak references
_self_evolution_instances: List[weakref.ref] = []


def _cleanup_all_instances():
    """Cleanup function called at program exit to save all SelfEvolution state."""
    for ref in _self_evolution_instances:
        instance = ref()
        if instance is not None:
            instance.cleanup()


# Register the cleanup function with atexit
atexit.register(_cleanup_all_instances)


def _register_instance(instance: 'SelfEvolution'):
    """Register a SelfEvolution instance for cleanup on exit."""
    _self_evolution_instances.append(weakref.ref(instance))


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
            'code_modifications': [],  # Track code self-modifications
            'train_every_cycle': True  # NEW: train after every cycle
        }
        self.state_file = EVOLUTION_STATE_FILE if not storage_path else self.storage_path / "evolution_state.json"
        self._load_state()

        # Training data - using centralized config
        self.training_data_file = str(TRAINING_DATA_FILE) if not storage_path else os.path.join(str(self.storage_path), "training_data.jsonl")

        # Initialize code evolution system for true self-modification
        self.code_evolution: Optional[CodeEvolution] = None
        if CODE_EVOLUTION_AVAILABLE:
            try:
                self.code_evolution = CodeEvolution(self.storage_path / "code_evolution")
                print("[SelfEvolution] Code evolution system initialized")
            except Exception as e:
                print(f"[SelfEvolution] Warning: Could not initialize code evolution: {e}")

        # Register for cleanup on program exit
        _register_instance(self)

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
        """Check if enough facts learned for cycle completion (every 100 facts)."""
        return self.state['facts_this_cycle'] >= self.state['facts_per_cycle']

    def cycle_complete(self) -> bool:
        """Alias for should_benchmark - check if cycle is complete."""
        return self.should_benchmark()

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
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        # Skip malformed entries silently - expected for corrupted training data
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
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    # Skip malformed entries silently - expected for corrupted training data
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
        Uses the CodeEvolution system for validation and safe deployment.
        """
        # Use CodeEvolution if available (preferred - validates and sandboxes)
        if self.code_evolution:
            return self._add_function_safe(name, code, description)

        # Fallback to legacy method (no validation)
        return self._add_function_legacy(name, code, description)

    def _add_function_safe(self, name: str, code: str, description: str) -> bool:
        """Add function using CodeEvolution with validation and testing."""
        try:
            # Propose the change (validates and tests)
            change = self.code_evolution.propose_change(
                change_type=CodeChangeType.NEW_FUNCTION,
                new_code=code,
                description=description
            )

            # Check validation result
            if change.validation_result != ValidationResult.VALID:
                print(f"[Evolution] Function '{name}' failed validation: {change.validation_result}")
                return False

            # Deploy the validated code
            success = self.code_evolution.deploy_change(change)

            if success:
                # Record in state
                self.state['added_functions'].append({
                    'name': name,
                    'description': description,
                    'timestamp': datetime.now().isoformat(),
                    'change_id': change.id,
                    'validated': True
                })
                self.state['code_modifications'].append({
                    'type': 'new_function',
                    'name': name,
                    'change_id': change.id,
                    'timestamp': datetime.now().isoformat()
                })
                self._save_state()
                print(f"[Evolution] Successfully added validated function: {name}")

            return success

        except Exception as e:
            print(f"[Evolution] Error adding function safely: {e}")
            return False

    def _add_function_legacy(self, name: str, code: str, description: str) -> bool:
        """Legacy method - adds function without validation (not recommended)."""
        functions_file = self.storage_path / "added_functions.py"

        try:
            # Append new function
            with open(functions_file, 'a') as f:
                f.write(f"\n\n# Added: {datetime.now().isoformat()}\n")
                f.write(f"# Description: {description}\n")
                f.write(f"# WARNING: Not validated (legacy method)\n")
                f.write(code)
                f.write("\n")

            # Record
            self.state['added_functions'].append({
                'name': name,
                'description': description,
                'timestamp': datetime.now().isoformat(),
                'validated': False
            })
            self._save_state()

            return True
        except Exception as e:
            print(f"[Evolution] Failed to add function: {e}")
            return False

    def modify_code(self, target_file: str, old_code: str, new_code: str, description: str) -> bool:
        """
        Modify existing code in the repository.

        This is TRUE self-modification - the AI changes its own source code.
        Requires CodeEvolution for safety validation.
        """
        if not self.code_evolution:
            print("[Evolution] Code modification requires CodeEvolution system")
            return False

        try:
            # Propose the modification
            change = self.code_evolution.propose_change(
                change_type=CodeChangeType.MODIFY_FUNCTION,
                new_code=new_code,
                description=description,
                target_file=target_file,
                original_code=old_code
            )

            if change.validation_result != ValidationResult.VALID:
                print(f"[Evolution] Modification failed validation: {change.validation_result}")
                return False

            # Deploy the change
            success = self.code_evolution.deploy_change(change)

            if success:
                self.state['code_modifications'].append({
                    'type': 'modify',
                    'target_file': target_file,
                    'change_id': change.id,
                    'description': description,
                    'timestamp': datetime.now().isoformat()
                })
                self._save_state()
                print(f"[Evolution] Successfully modified code in {target_file}")

            return success

        except Exception as e:
            print(f"[Evolution] Error modifying code: {e}")
            return False

    def optimize_function(self, func_name: str, optimized_code: str, description: str) -> bool:
        """
        Optimize an existing function with improved code.

        Used when the AI learns a better implementation.
        """
        if not self.code_evolution:
            print("[Evolution] Optimization requires CodeEvolution system")
            return False

        try:
            change = self.code_evolution.propose_change(
                change_type=CodeChangeType.OPTIMIZATION,
                new_code=optimized_code,
                description=f"Optimize {func_name}: {description}"
            )

            if change.validation_result != ValidationResult.VALID:
                print(f"[Evolution] Optimization failed validation")
                return False

            success = self.code_evolution.deploy_change(change)

            if success:
                self.state['code_modifications'].append({
                    'type': 'optimization',
                    'function': func_name,
                    'change_id': change.id,
                    'description': description,
                    'timestamp': datetime.now().isoformat()
                })
                self._save_state()

            return success

        except Exception as e:
            print(f"[Evolution] Error optimizing function: {e}")
            return False

    def rollback_last_modification(self) -> bool:
        """
        Rollback the last code modification.

        Safety mechanism if a change causes problems.
        """
        if not self.code_evolution:
            print("[Evolution] Rollback requires CodeEvolution system")
            return False

        success = self.code_evolution.rollback_last()

        if success and self.state['code_modifications']:
            rolled_back = self.state['code_modifications'].pop()
            rolled_back['rolled_back'] = True
            rolled_back['rollback_time'] = datetime.now().isoformat()
            self._save_state()
            print(f"[Evolution] Rolled back modification: {rolled_back.get('description', 'unknown')}")

        return success

    def call_evolved_function(self, name: str, *args, **kwargs) -> Any:
        """
        Call a function that was added through evolution.

        This allows using capabilities the AI has created for itself.
        """
        if not self.code_evolution:
            print("[Evolution] Evolved functions require CodeEvolution system")
            return None

        try:
            return self.code_evolution.call_evolved_function(name, *args, **kwargs)
        except Exception as e:
            print(f"[Evolution] Error calling evolved function '{name}': {e}")
            return None

    def get_evolved_functions(self) -> Dict[str, Any]:
        """Get all functions the AI has created through evolution."""
        if not self.code_evolution:
            return {}
        return self.code_evolution.get_active_functions()

    def get_code_evolution_stats(self) -> Dict:
        """Get statistics about code evolution."""
        if not self.code_evolution:
            return {'available': False}

        stats = self.code_evolution.get_stats()
        stats['available'] = True
        stats['total_modifications'] = len(self.state.get('code_modifications', []))
        return stats

    # === SELF-INTROSPECTION ===
    # The AI can read and understand its own code

    def read_own_code(self, file_path: str) -> Optional[str]:
        """
        Read the AI's own source code.

        Args:
            file_path: Relative path like "integrations/self_evolution.py"

        Example:
            code = evo.read_own_code("integrations/benchmark.py")
        """
        if not self.code_evolution:
            print("[Evolution] Introspection requires CodeEvolution system")
            return None
        return self.code_evolution.read_own_code(file_path)

    def read_own_function(self, file_path: str, function_name: str) -> Optional[str]:
        """
        Read a specific function from the AI's source.

        Example:
            func_code = evo.read_own_function("integrations/self_evolution.py", "should_train")
        """
        if not self.code_evolution:
            return None
        return self.code_evolution.read_own_function(file_path, function_name)

    def list_own_functions(self, file_path: str) -> List[Dict]:
        """
        List all functions in one of the AI's source files.

        Returns list of {name, args, docstring, lineno}.
        """
        if not self.code_evolution:
            return []
        return self.code_evolution.list_own_functions(file_path)

    def search_own_code(self, pattern: str) -> List[Dict]:
        """
        Search for a pattern across all source code.

        Example:
            matches = evo.search_own_code("def.*benchmark")
        """
        if not self.code_evolution:
            return []
        return self.code_evolution.search_own_code(pattern)

    def analyze_own_code(self, file_path: str) -> Dict:
        """
        Analyze complexity of a source file.

        Helps identify functions that need optimization.
        """
        if not self.code_evolution:
            return {}
        return self.code_evolution.analyze_own_code(file_path)

    def find_own_files(self) -> List[str]:
        """
        List all Python files in the project.

        Returns list of relative paths.
        """
        if not self.code_evolution:
            return []
        return self.code_evolution.find_own_files()

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
Code modifications: {len(self.state.get('code_modifications', []))}

"""

        if improvement > 0:
            reflection += f"✓ IMPROVING: Gained {improvement:.1%} since start\n"
        elif improvement < 0:
            reflection += f"✗ REGRESSING: Lost {abs(improvement):.1%} - need different approach\n"
        else:
            reflection += "→ STABLE: No change yet - keep learning\n"

        if trainings > 0:
            reflection += f"✓ TRAINED: Model fine-tuned {trainings} times\n"

        # Add code evolution status
        if self.code_evolution:
            code_stats = self.get_code_evolution_stats()
            reflection += f"\n=== CODE SELF-MODIFICATION ===\n"
            reflection += f"Active evolved functions: {code_stats.get('active_functions', 0)}\n"
            reflection += f"Deployed changes: {code_stats.get('deployed_changes', 0)}\n"
            reflection += f"Rollbacks: {code_stats.get('rollbacks', 0)}\n"

            # Show recent modifications
            mods = self.state.get('code_modifications', [])[-3:]
            if mods:
                reflection += "\nRecent modifications:\n"
                for mod in mods:
                    status = "↩" if mod.get('rolled_back') else "✓"
                    reflection += f"  {status} {mod.get('type', 'unknown')}: {mod.get('description', 'no desc')[:50]}\n"
        else:
            reflection += "\n⚠ Code self-modification: DISABLED (CodeEvolution not available)\n"

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
            except (json.JSONDecodeError, IOError, OSError, TypeError) as e:
                # Start fresh if hashes file is corrupted or unreadable
                self.learned_hashes = set()

    def _save_hashes(self):
        with open(self.learned_hashes_file, 'w') as f:
            json.dump(list(self.learned_hashes), f)

    def _load_benchmark_history(self):
        if self.benchmark_file.exists():
            try:
                with open(self.benchmark_file, 'r') as f:
                    self.benchmark_history = json.load(f)
            except (json.JSONDecodeError, IOError, OSError, TypeError) as e:
                # Start fresh if benchmark file is corrupted or unreadable
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
            except (json.JSONDecodeError, IOError, OSError, KeyError, TypeError) as e:
                # Use default state if file is corrupted or unreadable
                pass

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def cleanup(self):
        """
        Cleanup resources and save state on exit.

        This method is registered with atexit to ensure state is saved
        when the program terminates. It saves:
        - Learned content hashes
        - Benchmark history
        - Evolution state (cycles, improvements, code modifications)
        """
        try:
            # Save all state
            self._save_hashes()
            self._save_benchmark_history()
            self._save_state()

            facts_count = len(self.learned_hashes)
            benchmarks_count = len(self.benchmark_history)
            mods_count = len(self.state.get('code_modifications', []))

            print(f"[SelfEvolution] Cleanup complete: saved {facts_count} hashes, "
                  f"{benchmarks_count} benchmarks, {mods_count} code modifications")
        except Exception as e:
            print(f"[SelfEvolution] Cleanup error: {e}")


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
