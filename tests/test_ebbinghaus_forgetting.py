"""
Tests for EbbinghausForgetting class in neuro_memory/memory/forgetting.py

Tests the Ebbinghaus forgetting curve implementation including:
- Memory registration and state tracking
- Retention computation using R = e^(-t/S) formula
- Stability increase with successful retrievals (spaced repetition)
- should_forget threshold checking
- State persistence
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_memory.memory.forgetting import (
    EbbinghausForgetting,
    EbbinghausConfig,
    MemoryState,
    ForgettingEngine,
    ForgettingConfig
)


class TestMemoryState:
    """Tests for MemoryState dataclass."""

    def test_memory_state_creation(self):
        """Test creating a MemoryState."""
        now = datetime.now().timestamp()
        state = MemoryState(
            memory_id="test_123",
            created_at=now,
            last_access=now,
            access_count=0,
            stability_score=1.0,
            initial_retention=1.0
        )

        assert state.memory_id == "test_123"
        assert state.created_at == now
        assert state.last_access == now
        assert state.access_count == 0
        assert state.stability_score == 1.0
        assert state.initial_retention == 1.0

    def test_memory_state_to_dict(self):
        """Test serialization to dict."""
        now = datetime.now().timestamp()
        state = MemoryState(
            memory_id="test_123",
            created_at=now,
            last_access=now,
            access_count=5,
            stability_score=2.5,
            initial_retention=0.9
        )

        data = state.to_dict()

        assert data["memory_id"] == "test_123"
        assert data["created_at"] == now
        assert data["access_count"] == 5
        assert data["stability_score"] == 2.5
        assert data["initial_retention"] == 0.9

    def test_memory_state_from_dict(self):
        """Test deserialization from dict."""
        now = datetime.now().timestamp()
        data = {
            "memory_id": "restored_memory",
            "created_at": now,
            "last_access": now,
            "access_count": 3,
            "stability_score": 4.5,
            "initial_retention": 0.8
        }

        state = MemoryState.from_dict(data)

        assert state.memory_id == "restored_memory"
        assert state.access_count == 3
        assert state.stability_score == 4.5
        assert state.initial_retention == 0.8


class TestEbbinghausConfig:
    """Tests for EbbinghausConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EbbinghausConfig()

        assert config.base_stability == 1.0
        assert config.stability_multiplier == 1.5
        assert config.forget_threshold == 0.3
        assert config.min_stability == 0.5
        assert config.max_stability == 720.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = EbbinghausConfig(
            base_stability=2.0,
            stability_multiplier=2.0,
            forget_threshold=0.5
        )

        assert config.base_stability == 2.0
        assert config.stability_multiplier == 2.0
        assert config.forget_threshold == 0.5


class TestEbbinghausForgetting:
    """Tests for EbbinghausForgetting class."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        ebbinghaus = EbbinghausForgetting()

        assert ebbinghaus.config is not None
        assert ebbinghaus._memory_states == {}

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = EbbinghausConfig(base_stability=5.0)
        ebbinghaus = EbbinghausForgetting(config=config)

        assert ebbinghaus.config.base_stability == 5.0

    def test_register_memory(self):
        """Test registering a new memory."""
        ebbinghaus = EbbinghausForgetting()

        state = ebbinghaus.register_memory("memory_1")

        assert state.memory_id == "memory_1"
        assert state.access_count == 0
        assert state.stability_score == ebbinghaus.config.base_stability
        assert "memory_1" in ebbinghaus._memory_states

    def test_register_memory_with_timestamp(self):
        """Test registering memory with specific timestamp."""
        ebbinghaus = EbbinghausForgetting()
        custom_time = 1700000000.0

        state = ebbinghaus.register_memory("memory_1", timestamp=custom_time)

        assert state.created_at == custom_time
        assert state.last_access == custom_time

    def test_get_memory_state(self):
        """Test retrieving memory state."""
        ebbinghaus = EbbinghausForgetting()
        ebbinghaus.register_memory("memory_1")

        state = ebbinghaus.get_memory_state("memory_1")
        assert state is not None
        assert state.memory_id == "memory_1"

        # Non-existent memory
        assert ebbinghaus.get_memory_state("nonexistent") is None


class TestRetentionComputation:
    """Tests for compute_retention method."""

    def test_retention_at_creation(self):
        """Test retention is 1.0 at time of creation."""
        ebbinghaus = EbbinghausForgetting()
        now = datetime.now().timestamp()

        ebbinghaus.register_memory("memory_1", timestamp=now)
        retention = ebbinghaus.compute_retention("memory_1", current_time=now)

        assert retention == pytest.approx(1.0, abs=0.01)

    def test_retention_decays_over_time(self):
        """Test retention decreases over time following R = e^(-t/S)."""
        config = EbbinghausConfig(base_stability=1.0)  # 1 hour stability
        ebbinghaus = EbbinghausForgetting(config=config)

        base_time = datetime.now().timestamp()
        ebbinghaus.register_memory("memory_1", timestamp=base_time)

        # After 1 hour with S=1, R = e^(-1) ≈ 0.368
        one_hour_later = base_time + 3600
        retention = ebbinghaus.compute_retention("memory_1", current_time=one_hour_later)
        expected = 0.368  # e^(-1)

        assert retention == pytest.approx(expected, abs=0.01)

    def test_retention_formula_correctness(self):
        """Test the exact formula R = e^(-t/S)."""
        import numpy as np

        config = EbbinghausConfig(base_stability=2.0)  # 2 hour stability
        ebbinghaus = EbbinghausForgetting(config=config)

        base_time = 1000000.0
        ebbinghaus.register_memory("memory_1", timestamp=base_time)

        # Test at various time intervals
        test_cases = [
            (0, 1.0),           # t=0: R = e^0 = 1.0
            (2, np.exp(-1)),    # t=2h, S=2: R = e^(-2/2) = e^(-1)
            (4, np.exp(-2)),    # t=4h, S=2: R = e^(-4/2) = e^(-2)
            (6, np.exp(-3)),    # t=6h, S=2: R = e^(-6/2) = e^(-3)
        ]

        for hours, expected in test_cases:
            time = base_time + hours * 3600
            retention = ebbinghaus.compute_retention("memory_1", current_time=time)
            assert retention == pytest.approx(expected, abs=0.001), f"Failed at {hours}h"

    def test_retention_nonexistent_memory(self):
        """Test retention returns 0 for non-existent memory."""
        ebbinghaus = EbbinghausForgetting()

        retention = ebbinghaus.compute_retention("nonexistent")

        assert retention == 0.0


class TestShouldForget:
    """Tests for should_forget method."""

    def test_should_not_forget_fresh_memory(self):
        """Test fresh memory should not be forgotten."""
        ebbinghaus = EbbinghausForgetting()
        now = datetime.now().timestamp()

        ebbinghaus.register_memory("memory_1", timestamp=now)

        assert ebbinghaus.should_forget("memory_1", current_time=now) is False

    def test_should_forget_after_threshold(self):
        """Test memory should be forgotten when retention below threshold."""
        config = EbbinghausConfig(
            base_stability=1.0,
            forget_threshold=0.3
        )
        ebbinghaus = EbbinghausForgetting(config=config)

        base_time = datetime.now().timestamp()
        ebbinghaus.register_memory("memory_1", timestamp=base_time)

        # After 2 hours with S=1, R = e^(-2) ≈ 0.135 < 0.3
        two_hours_later = base_time + 2 * 3600

        assert ebbinghaus.should_forget("memory_1", current_time=two_hours_later) is True

    def test_should_forget_custom_threshold(self):
        """Test should_forget with custom threshold."""
        config = EbbinghausConfig(base_stability=1.0)
        ebbinghaus = EbbinghausForgetting(config=config)

        base_time = datetime.now().timestamp()
        ebbinghaus.register_memory("memory_1", timestamp=base_time)

        one_hour_later = base_time + 3600

        # With threshold=0.5, retention ~0.368 should be forgotten
        assert ebbinghaus.should_forget("memory_1", threshold=0.5, current_time=one_hour_later) is True

        # With threshold=0.2, retention ~0.368 should NOT be forgotten
        assert ebbinghaus.should_forget("memory_1", threshold=0.2, current_time=one_hour_later) is False


class TestSpacedRepetition:
    """Tests for spaced repetition (stability increase on retrieval)."""

    def test_stability_increases_on_successful_retrieval(self):
        """Test stability increases after successful retrieval."""
        config = EbbinghausConfig(
            base_stability=1.0,
            stability_multiplier=1.5
        )
        ebbinghaus = EbbinghausForgetting(config=config)

        ebbinghaus.register_memory("memory_1")
        initial_stability = ebbinghaus.get_memory_state("memory_1").stability_score

        ebbinghaus.record_retrieval("memory_1", success=True)
        new_stability = ebbinghaus.get_memory_state("memory_1").stability_score

        assert new_stability == pytest.approx(initial_stability * 1.5)

    def test_multiple_retrievals_compound_stability(self):
        """Test multiple successful retrievals compound stability."""
        config = EbbinghausConfig(
            base_stability=1.0,
            stability_multiplier=2.0,
            max_stability=1000.0
        )
        ebbinghaus = EbbinghausForgetting(config=config)

        ebbinghaus.register_memory("memory_1")

        # 5 successful retrievals: 1 -> 2 -> 4 -> 8 -> 16 -> 32
        for _ in range(5):
            ebbinghaus.record_retrieval("memory_1", success=True)

        final_stability = ebbinghaus.get_memory_state("memory_1").stability_score
        expected = 1.0 * (2.0 ** 5)  # 32

        assert final_stability == pytest.approx(expected)

    def test_failed_retrieval_resets_stability(self):
        """Test failed retrieval resets stability to minimum."""
        config = EbbinghausConfig(
            base_stability=1.0,
            stability_multiplier=2.0,
            min_stability=0.5
        )
        ebbinghaus = EbbinghausForgetting(config=config)

        ebbinghaus.register_memory("memory_1")

        # Build up stability
        for _ in range(3):
            ebbinghaus.record_retrieval("memory_1", success=True)

        # Stability should be 8.0 now
        assert ebbinghaus.get_memory_state("memory_1").stability_score == 8.0

        # Failed retrieval
        ebbinghaus.record_retrieval("memory_1", success=False)

        # Should be reset to min_stability
        assert ebbinghaus.get_memory_state("memory_1").stability_score == 0.5

    def test_stability_capped_at_max(self):
        """Test stability is capped at max_stability."""
        config = EbbinghausConfig(
            base_stability=100.0,
            stability_multiplier=10.0,
            max_stability=500.0
        )
        ebbinghaus = EbbinghausForgetting(config=config)

        ebbinghaus.register_memory("memory_1")

        # Multiple retrievals should hit the cap
        for _ in range(10):
            ebbinghaus.record_retrieval("memory_1", success=True)

        stability = ebbinghaus.get_memory_state("memory_1").stability_score

        assert stability == 500.0

    def test_access_count_increments(self):
        """Test access_count is incremented on retrieval."""
        ebbinghaus = EbbinghausForgetting()

        ebbinghaus.register_memory("memory_1")
        assert ebbinghaus.get_memory_state("memory_1").access_count == 0

        ebbinghaus.record_retrieval("memory_1", success=True)
        assert ebbinghaus.get_memory_state("memory_1").access_count == 1

        ebbinghaus.record_retrieval("memory_1", success=False)
        assert ebbinghaus.get_memory_state("memory_1").access_count == 2


class TestBulkOperations:
    """Tests for bulk operations and statistics."""

    def test_get_all_retentions(self):
        """Test getting retention for all memories."""
        ebbinghaus = EbbinghausForgetting()
        now = datetime.now().timestamp()

        ebbinghaus.register_memory("mem_1", timestamp=now)
        ebbinghaus.register_memory("mem_2", timestamp=now)
        ebbinghaus.register_memory("mem_3", timestamp=now)

        retentions = ebbinghaus.get_all_retentions(current_time=now)

        assert len(retentions) == 3
        assert all(r == pytest.approx(1.0, abs=0.01) for r in retentions.values())

    def test_get_memories_below_threshold(self):
        """Test getting memories below retention threshold."""
        config = EbbinghausConfig(base_stability=1.0, forget_threshold=0.5)
        ebbinghaus = EbbinghausForgetting(config=config)

        base_time = 1000000.0

        # Fresh memory
        ebbinghaus.register_memory("fresh", timestamp=base_time)

        # Old memory (2 hours old, retention ~ 0.135)
        ebbinghaus.register_memory("old", timestamp=base_time - 2 * 3600)

        below = ebbinghaus.get_memories_below_threshold(current_time=base_time)

        assert len(below) == 1
        assert below[0][0] == "old"
        assert below[0][1] < 0.5

    def test_get_statistics(self):
        """Test statistics computation."""
        ebbinghaus = EbbinghausForgetting()
        now = datetime.now().timestamp()

        for i in range(5):
            ebbinghaus.register_memory(f"mem_{i}", timestamp=now)
            if i > 0:
                ebbinghaus.record_retrieval(f"mem_{i}", success=True)

        stats = ebbinghaus.get_statistics()

        assert stats["total_memories"] == 5
        assert stats["avg_stability"] > 0
        assert "stability_distribution" in stats

    def test_statistics_empty(self):
        """Test statistics with no memories."""
        ebbinghaus = EbbinghausForgetting()

        stats = ebbinghaus.get_statistics()

        assert stats["total_memories"] == 0
        assert stats["avg_stability"] == 0.0

    def test_remove_memory(self):
        """Test removing a memory."""
        ebbinghaus = EbbinghausForgetting()

        ebbinghaus.register_memory("memory_1")
        assert "memory_1" in ebbinghaus._memory_states

        result = ebbinghaus.remove_memory("memory_1")

        assert result is True
        assert "memory_1" not in ebbinghaus._memory_states

    def test_remove_nonexistent_memory(self):
        """Test removing non-existent memory returns False."""
        ebbinghaus = EbbinghausForgetting()

        result = ebbinghaus.remove_memory("nonexistent")

        assert result is False


class TestPersistence:
    """Tests for state persistence."""

    def test_save_and_load_state(self):
        """Test saving and loading state to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "forgetting_state.json"

            # Create and populate
            ebbinghaus1 = EbbinghausForgetting(state_path=state_path)
            ebbinghaus1.register_memory("mem_1")
            ebbinghaus1.record_retrieval("mem_1", success=True)
            ebbinghaus1.register_memory("mem_2")

            # State should be saved
            assert state_path.exists()

            # Load in new instance
            ebbinghaus2 = EbbinghausForgetting(state_path=state_path)

            assert len(ebbinghaus2._memory_states) == 2
            assert ebbinghaus2.get_memory_state("mem_1") is not None
            assert ebbinghaus2.get_memory_state("mem_1").access_count == 1

    def test_persistence_preserves_stability(self):
        """Test that stability scores are preserved across loads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"

            # Create with high stability
            ebbinghaus1 = EbbinghausForgetting(state_path=state_path)
            ebbinghaus1.register_memory("mem_1")
            for _ in range(5):
                ebbinghaus1.record_retrieval("mem_1", success=True)

            original_stability = ebbinghaus1.get_memory_state("mem_1").stability_score

            # Load in new instance
            ebbinghaus2 = EbbinghausForgetting(state_path=state_path)
            loaded_stability = ebbinghaus2.get_memory_state("mem_1").stability_score

            assert loaded_stability == pytest.approx(original_stability)


class TestIntegrationWithExistingForgettingEngine:
    """Tests to ensure new class works alongside existing ForgettingEngine."""

    def test_both_classes_coexist(self):
        """Test both forgetting implementations can be used together."""
        # Existing engine
        old_engine = ForgettingEngine()

        # New Ebbinghaus engine
        new_engine = EbbinghausForgetting()

        # Both should work independently
        activation = old_engine.compute_activation(
            initial_activation=2.0,
            timestamp=datetime.now()
        )
        assert activation > 0

        new_engine.register_memory("test")
        retention = new_engine.compute_retention("test")
        assert retention > 0

    def test_similar_decay_behavior(self):
        """Test both engines exhibit similar decay patterns."""
        # Both should show decreasing values over time

        old_engine = ForgettingEngine(ForgettingConfig(
            decay_rate=1.0,
            use_power_law=False  # Use exponential for comparison
        ))

        new_engine = EbbinghausForgetting(EbbinghausConfig(
            base_stability=1.0
        ))

        # Old engine at t=1 hour
        timestamp = datetime.now()
        old_activation = old_engine.compute_activation(
            initial_activation=1.0,
            timestamp=timestamp - timedelta(hours=1),
            current_time=timestamp
        )

        # New engine at t=1 hour
        base_time = timestamp.timestamp()
        new_engine.register_memory("test", timestamp=base_time - 3600)
        new_retention = new_engine.compute_retention("test", current_time=base_time)

        # Both should be around e^(-1) ≈ 0.368
        # (though old engine's formula might differ slightly)
        assert old_activation < 1.0  # Decay happened
        assert new_retention < 1.0   # Decay happened
        assert new_retention == pytest.approx(0.368, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
