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
import json
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime


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
        self.storage_path = Path(storage_path or os.path.expanduser("~/.cognitive_ai_knowledge/evolution"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Track learned content hashes (no duplicates)
        self.learned_hashes: Set[str] = set()
        self.learned_hashes_file = self.storage_path / "learned_hashes.json"
        self._load_hashes()

        # Benchmark history
        self.benchmark_history: List[Dict] = []
        self.benchmark_file = self.storage_path / "benchmark_history.json"
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
            'added_functions': []
        }
        self.state_file = self.storage_path / "evolution_state.json"
        self._load_state()

        # Training data for MLX
        self.training_data_file = os.path.expanduser("~/.cognitive_ai_knowledge/training_data.jsonl")

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
        """Mark content as learned. Returns False if duplicate."""
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

        Returns:
            (should_train, reason)
        """
        improvement = self.get_improvement()

        # Count training pairs
        training_count = self._count_training_pairs()

        if training_count < 10:
            return False, f"Not enough training data ({training_count} pairs, need 10+)"

        # Train if we have enough data (500+) even without improvement
        # This bootstraps the model to USE the knowledge
        if training_count >= 500 and self.state['total_trainings'] == 0:
            return True, f"First training with {training_count} pairs - bootstrap learning!"

        if improvement >= min_improvement:
            return True, f"Improved {improvement:.1%} - ready to train!"

        if improvement < 0:
            return False, f"Score decreased by {abs(improvement):.1%} - need more learning"

        return False, f"Only {improvement:.1%} improvement - need {min_improvement:.1%}+"

    def _count_training_pairs(self) -> int:
        """Count training data pairs."""
        if not os.path.exists(self.training_data_file):
            return 0
        try:
            with open(self.training_data_file, 'r') as f:
                return sum(1 for _ in f)
        except:
            return 0

    def start_new_cycle(self):
        """Start a new learning cycle."""
        self.state['current_cycle'] += 1
        self.state['facts_this_cycle'] = 0
        self._save_state()

    def run_mlx_training(self, model_name: str = "ministral-3:8b") -> Dict:
        """
        Run actual MLX fine-tuning on MacBook.

        This is REAL training that modifies the model.
        """
        result = {
            'success': False,
            'message': '',
            'timestamp': datetime.now().isoformat()
        }

        # Check if MLX is available
        try:
            import mlx
            import mlx.core as mx
        except ImportError:
            result['message'] = "MLX not installed. Run: pip install mlx mlx-lm"
            return result

        # Check training data
        training_count = self._count_training_pairs()
        if training_count < 10:
            result['message'] = f"Not enough training data: {training_count} pairs"
            return result

        # Prepare training data in MLX format
        mlx_data_file = self.storage_path / "mlx_training_data.jsonl"
        self._prepare_mlx_data(mlx_data_file)

        # Run MLX fine-tuning
        try:
            print(f"\n[MLX] Starting fine-tuning with {training_count} examples...")
            print(f"[MLX] This may take a few minutes on your M-series MacBook...")

            # MLX-LM fine-tuning command
            # Using LoRA for efficient training
            cmd = [
                "python", "-m", "mlx_lm.lora",
                "--model", model_name,
                "--data", str(mlx_data_file),
                "--train",
                "--batch-size", "1",
                "--lora-layers", "4",
                "--iters", "100"
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )

            if process.returncode == 0:
                result['success'] = True
                result['message'] = "MLX fine-tuning completed successfully!"
                self.state['total_trainings'] += 1
                self.state['improvements'].append({
                    'cycle': self.state['current_cycle'],
                    'improvement': self.get_improvement(),
                    'timestamp': datetime.now().isoformat()
                })
                self._save_state()
            else:
                result['message'] = f"MLX training failed: {process.stderr[:200]}"

        except subprocess.TimeoutExpired:
            result['message'] = "MLX training timed out (>10 minutes)"
        except FileNotFoundError:
            result['message'] = "mlx_lm not found. Run: pip install mlx-lm"
        except Exception as e:
            result['message'] = f"MLX training error: {str(e)}"

        return result

    def _prepare_mlx_data(self, output_file: Path):
        """Convert training data to MLX format."""
        with open(self.training_data_file, 'r') as f_in:
            with open(output_file, 'w') as f_out:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        # MLX format: {"text": "<prompt>Response</prompt>"}
                        mlx_entry = {
                            "text": f"<s>[INST] {data['prompt']} [/INST] {data['completion']}</s>"
                        }
                        f_out.write(json.dumps(mlx_entry) + '\n')
                    except:
                        continue

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
