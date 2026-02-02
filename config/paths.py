"""
Centralized Path Configuration
==============================

All file paths used across the system should be defined here to ensure
consistency between components (self_evolution, super_agent, etc.).

Usage:
    from config.paths import TRAINING_DATA_FILE, KNOWLEDGE_DIR
"""

import os
from pathlib import Path


# Base directories
KNOWLEDGE_DIR = Path(os.path.expanduser("~/.cognitive_ai_knowledge"))
EVOLUTION_DIR = KNOWLEDGE_DIR / "evolution"

# Training data - single source of truth
TRAINING_DATA_FILE = KNOWLEDGE_DIR / "training_data.jsonl"

# Evolution state files
LEARNED_HASHES_FILE = EVOLUTION_DIR / "learned_hashes.json"
BENCHMARK_HISTORY_FILE = EVOLUTION_DIR / "benchmark_history.json"
EVOLUTION_STATE_FILE = EVOLUTION_DIR / "evolution_state.json"
ADAPTERS_DIR = EVOLUTION_DIR / "adapters"
LLAMA_FACTORY_OUTPUT_DIR = EVOLUTION_DIR / "llama_factory_output"

# Model paths
MLX_MODEL_PATH = Path(os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit"
))
HF_MODEL_PATH = "Qwen/Qwen2.5-Coder-32B-Instruct"

# LM Studio
LM_STUDIO_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen/qwen3-vl-30b"


def ensure_directories():
    """Create all required directories if they don't exist."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    LLAMA_FACTORY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on import
ensure_directories()
