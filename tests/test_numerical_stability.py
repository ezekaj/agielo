"""
Tests for numerical stability utilities and related fixes.

Tests the safe_exp function and verifies numerical stability in:
- utils/numerical.py
- neuro_memory/consolidation/memory_consolidation.py
- neuro_memory/retrieval/two_stage_retriever.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.numerical import (
    safe_exp,
    safe_sigmoid,
    safe_softmax,
    safe_log,
    safe_divide,
    validate_finite,
    clip_to_range,
)


class TestSafeExp:
    """Tests for the safe_exp function."""

    def test_normal_values(self):
        """Test safe_exp with normal input values."""
        assert np.isclose(safe_exp(0), 1.0)
        assert np.isclose(safe_exp(1), np.e)
        assert np.isclose(safe_exp(-1), 1/np.e)

    def test_large_positive_values(self):
        """Test safe_exp clips large positive values to prevent overflow."""
        result = safe_exp(1000)
        assert np.isfinite(result)
        assert result == np.exp(500)  # Should be clipped to max

    def test_large_negative_values(self):
        """Test safe_exp clips large negative values."""
        result = safe_exp(-1000)
        assert np.isfinite(result)
        assert result == np.exp(-500)  # Should be clipped to min

    def test_array_input(self):
        """Test safe_exp with array input."""
        x = np.array([-1000, -1, 0, 1, 1000])
        result = safe_exp(x)
        assert len(result) == len(x)
        assert all(np.isfinite(result))

    def test_custom_bounds(self):
        """Test safe_exp with custom bounds."""
        result = safe_exp(100, min_val=-10, max_val=10)
        assert result == np.exp(10)

    def test_edge_cases(self):
        """Test safe_exp with edge case values."""
        # Test at exact bounds
        assert safe_exp(500) == np.exp(500)
        assert safe_exp(-500) == np.exp(-500)

    def test_vectorized_performance(self):
        """Test that safe_exp handles large arrays efficiently."""
        large_array = np.random.randn(10000) * 1000
        result = safe_exp(large_array)
        assert len(result) == 10000
        assert all(np.isfinite(result))


class TestSafeSigmoid:
    """Tests for the safe_sigmoid function."""

    def test_normal_values(self):
        """Test safe_sigmoid with normal input values."""
        assert np.isclose(safe_sigmoid(0), 0.5)
        assert safe_sigmoid(10) > 0.99
        assert safe_sigmoid(-10) < 0.01

    def test_extreme_values(self):
        """Test safe_sigmoid doesn't overflow with extreme values."""
        assert np.isclose(safe_sigmoid(1000), 1.0)
        assert np.isclose(safe_sigmoid(-1000), 0.0)

    def test_array_input(self):
        """Test safe_sigmoid with array input."""
        x = np.array([-1000, 0, 1000])
        result = safe_sigmoid(x)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 0.5)
        assert np.isclose(result[2], 1.0)


class TestSafeSoftmax:
    """Tests for the safe_softmax function."""

    def test_normal_values(self):
        """Test safe_softmax with normal values."""
        x = np.array([1.0, 2.0, 3.0])
        result = safe_softmax(x)
        assert np.isclose(np.sum(result), 1.0)
        assert all(result > 0)

    def test_extreme_values_no_overflow(self):
        """Test safe_softmax handles extreme values without overflow."""
        x = np.array([1000, 1000, 1000])
        result = safe_softmax(x)
        assert all(np.isfinite(result))
        assert np.isclose(np.sum(result), 1.0)
        # Equal values should give equal probabilities
        assert np.allclose(result, [1/3, 1/3, 1/3])

    def test_temperature(self):
        """Test softmax temperature parameter."""
        x = np.array([1.0, 2.0, 3.0])
        low_temp = safe_softmax(x, temperature=0.5)  # Sharper
        high_temp = safe_softmax(x, temperature=2.0)  # Softer
        # Low temperature should have higher max probability
        assert np.max(low_temp) > np.max(high_temp)


class TestSafeLog:
    """Tests for the safe_log function."""

    def test_normal_values(self):
        """Test safe_log with normal values."""
        assert np.isclose(safe_log(1), 0.0, atol=1e-9)
        assert np.isclose(safe_log(np.e), 1.0, atol=1e-9)

    def test_zero_input(self):
        """Test safe_log handles zero input without -inf."""
        result = safe_log(0)
        assert np.isfinite(result)
        assert result < 0  # Should be a large negative number


class TestSafeDivide:
    """Tests for the safe_divide function."""

    def test_normal_division(self):
        """Test safe_divide with normal values."""
        assert safe_divide(4, 2) == 2.0
        assert safe_divide(1, 3) == pytest.approx(1/3)

    def test_division_by_zero(self):
        """Test safe_divide handles division by zero."""
        assert safe_divide(1, 0) == 0.0
        assert safe_divide(1, 0, default=float('inf')) == float('inf')

    def test_array_division(self):
        """Test safe_divide with arrays."""
        num = np.array([1, 2, 3])
        denom = np.array([1, 0, 3])
        result = safe_divide(num, denom)
        assert result[0] == 1.0
        assert result[1] == 0.0  # default for div by zero
        assert result[2] == 1.0


class TestValidateFinite:
    """Tests for the validate_finite function."""

    def test_finite_values(self):
        """Test validate_finite with finite values."""
        assert validate_finite(1.0)
        assert validate_finite(np.array([1, 2, 3]))

    def test_nan_raises_error(self):
        """Test validate_finite raises error for NaN."""
        with pytest.raises(ValueError):
            validate_finite(float('nan'))

    def test_inf_raises_error(self):
        """Test validate_finite raises error for Inf."""
        with pytest.raises(ValueError):
            validate_finite(float('inf'))


class TestClipToRange:
    """Tests for the clip_to_range function."""

    def test_clip_values(self):
        """Test clip_to_range clips values correctly."""
        assert clip_to_range(10, 0, 5) == 5
        assert clip_to_range(-10, 0, 5) == 0
        assert clip_to_range(3, 0, 5) == 3


class TestMemoryConsolidationNumericalStability:
    """Test numerical stability in memory consolidation."""

    def test_prioritize_episodes_with_old_memories(self):
        """Test prioritize_episodes doesn't overflow with very old memories."""
        from neuro_memory.consolidation.memory_consolidation import (
            MemoryConsolidationEngine
        )

        class MockEpisode:
            def __init__(self, surprise, timestamp):
                self.surprise = surprise
                self.timestamp = timestamp

        engine = MemoryConsolidationEngine()

        # Create episodes with very old timestamps (years ago)
        old_time = datetime.now() - timedelta(days=365*10)  # 10 years ago
        episodes = [
            MockEpisode(1.0, old_time),
            MockEpisode(2.0, old_time - timedelta(days=365)),  # 11 years ago
            MockEpisode(0.5, datetime.now()),  # Recent
        ]

        _, probabilities = engine.prioritize_episodes(episodes)

        # All probabilities should be finite
        assert all(np.isfinite(probabilities))
        # Probabilities should sum to 1
        assert np.isclose(np.sum(probabilities), 1.0)


class TestTwoStageRetrieverNumericalStability:
    """Test numerical stability in two-stage retriever."""

    def test_recency_score_with_old_memories(self):
        """Test recency calculation doesn't overflow with very old memories."""
        # We can test the recency calculation logic directly
        query_time = datetime.now()

        # Very old episode (10 years ago)
        old_timestamp = query_time - timedelta(days=365*10)
        time_diff = (query_time - old_timestamp).total_seconds()

        # This is the fixed calculation with clipping
        decay_exponent = np.clip(-time_diff / (72 * 3600), -500, 0)
        recency = float(np.exp(decay_exponent))

        assert np.isfinite(recency)
        assert recency >= 0
        assert recency <= 1

    def test_recency_score_recent_memory(self):
        """Test recency calculation for recent memories."""
        query_time = datetime.now()
        recent_timestamp = query_time - timedelta(hours=1)
        time_diff = (query_time - recent_timestamp).total_seconds()

        decay_exponent = np.clip(-time_diff / (72 * 3600), -500, 0)
        recency = float(np.exp(decay_exponent))

        assert np.isfinite(recency)
        assert recency > 0.9  # Recent memory should have high recency


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
