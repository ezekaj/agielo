"""
Configuration module for human-cognition-ai.

This module provides centralized configuration including file paths,
model settings, and other shared constants.
"""

from .paths import (
    KNOWLEDGE_DIR,
    EVOLUTION_DIR,
    TRAINING_DATA_FILE,
    LEARNED_HASHES_FILE,
    BENCHMARK_HISTORY_FILE,
    EVOLUTION_STATE_FILE,
    ADAPTERS_DIR,
    LLAMA_FACTORY_OUTPUT_DIR,
    MLX_MODEL_PATH,
    HF_MODEL_PATH,
    LM_STUDIO_URL,
    DEFAULT_MODEL,
    ensure_directories,
)
