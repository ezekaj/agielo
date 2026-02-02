"""
Tests for config/paths.py

Verifies:
1. All paths are properly defined
2. Directories are created on import
3. Paths are consistent across modules
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.paths import (
    KNOWLEDGE_DIR, EVOLUTION_DIR, TRAINING_DATA_FILE,
    LEARNED_HASHES_FILE, BENCHMARK_HISTORY_FILE, EVOLUTION_STATE_FILE,
    ADAPTERS_DIR, LLAMA_FACTORY_OUTPUT_DIR, MLX_MODEL_PATH, HF_MODEL_PATH,
    LM_STUDIO_URL, DEFAULT_MODEL, ensure_directories
)


class TestPathDefinitions:
    """Test that all paths are properly defined."""

    def test_knowledge_dir_is_path(self):
        """KNOWLEDGE_DIR should be a Path object."""
        assert isinstance(KNOWLEDGE_DIR, Path)

    def test_evolution_dir_is_path(self):
        """EVOLUTION_DIR should be a Path object."""
        assert isinstance(EVOLUTION_DIR, Path)

    def test_training_data_file_is_path(self):
        """TRAINING_DATA_FILE should be a Path object."""
        assert isinstance(TRAINING_DATA_FILE, Path)

    def test_training_data_file_extension(self):
        """TRAINING_DATA_FILE should have .jsonl extension."""
        assert str(TRAINING_DATA_FILE).endswith('.jsonl')

    def test_evolution_dir_under_knowledge_dir(self):
        """EVOLUTION_DIR should be under KNOWLEDGE_DIR."""
        assert str(EVOLUTION_DIR).startswith(str(KNOWLEDGE_DIR))

    def test_lm_studio_url_format(self):
        """LM_STUDIO_URL should be a valid URL format."""
        assert LM_STUDIO_URL.startswith('http')
        assert ':' in LM_STUDIO_URL

    def test_default_model_not_empty(self):
        """DEFAULT_MODEL should not be empty."""
        assert DEFAULT_MODEL and len(DEFAULT_MODEL) > 0


class TestPathConsistency:
    """Test that paths are consistent between modules."""

    def test_self_evolution_uses_shared_training_path(self):
        """self_evolution.py should use TRAINING_DATA_FILE from config."""
        from integrations.self_evolution import SelfEvolution

        # Default instance (no custom storage_path)
        evo = SelfEvolution()
        assert evo.training_data_file == str(TRAINING_DATA_FILE)

    def test_self_evolution_custom_path_override(self):
        """self_evolution.py should allow custom path override."""
        import tempfile
        from integrations.self_evolution import SelfEvolution

        with tempfile.TemporaryDirectory() as tmpdir:
            evo = SelfEvolution(storage_path=tmpdir)
            # Should use custom path, not default
            assert tmpdir in evo.training_data_file

    def test_super_agent_imports_config(self):
        """super_agent.py should import from config.paths."""
        # This test verifies the import works without error
        from integrations.super_agent import TRAINING_DATA_FILE as SA_TRAINING_DATA_FILE
        assert SA_TRAINING_DATA_FILE == TRAINING_DATA_FILE


class TestDirectoryCreation:
    """Test that directories are created properly."""

    def test_ensure_directories_creates_knowledge_dir(self):
        """ensure_directories should create KNOWLEDGE_DIR."""
        ensure_directories()
        assert KNOWLEDGE_DIR.exists()

    def test_ensure_directories_creates_evolution_dir(self):
        """ensure_directories should create EVOLUTION_DIR."""
        ensure_directories()
        assert EVOLUTION_DIR.exists()

    def test_ensure_directories_creates_adapters_dir(self):
        """ensure_directories should create ADAPTERS_DIR."""
        ensure_directories()
        assert ADAPTERS_DIR.exists()

    def test_ensure_directories_creates_llama_factory_dir(self):
        """ensure_directories should create LLAMA_FACTORY_OUTPUT_DIR."""
        ensure_directories()
        assert LLAMA_FACTORY_OUTPUT_DIR.exists()


class TestPathExpansion:
    """Test that paths with ~ are properly expanded."""

    def test_knowledge_dir_expanded(self):
        """KNOWLEDGE_DIR should have ~ expanded."""
        assert '~' not in str(KNOWLEDGE_DIR)

    def test_mlx_model_path_expanded(self):
        """MLX_MODEL_PATH should have ~ expanded."""
        assert '~' not in str(MLX_MODEL_PATH)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
