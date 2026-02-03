"""
Tests for Online Continual Learning Module
==========================================

Tests cover:
- Basic replay buffer operations
- Division by zero protection in priority sampling
- Adaptive threshold updates
- EWC loss computation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuro_memory.online_learning import OnlineLearner, OnlineLearningConfig


class TestOnlineLearnerReplayBuffer:
    """Tests for replay buffer operations."""

    def test_add_to_replay_buffer(self):
        """Test basic add to replay buffer."""
        learner = OnlineLearner()
        obs = np.random.randn(64)

        learner.add_to_replay_buffer(obs, surprise=0.5)

        assert len(learner.replay_buffer) == 1
        assert learner.replay_buffer[0]['surprise'] == 0.5

    def test_replay_buffer_max_size(self):
        """Test replay buffer respects max size."""
        config = OnlineLearningConfig(replay_buffer_size=10)
        learner = OnlineLearner(config)

        # Add more than max size
        for i in range(20):
            obs = np.random.randn(64)
            learner.add_to_replay_buffer(obs, surprise=float(i))

        assert len(learner.replay_buffer) == 10

    def test_high_priority_replaces_low_priority(self):
        """Test that high priority items replace low priority ones."""
        config = OnlineLearningConfig(replay_buffer_size=5)
        learner = OnlineLearner(config)

        # Fill buffer with low priority items
        for i in range(5):
            obs = np.random.randn(64)
            learner.add_to_replay_buffer(obs, surprise=0.1)

        # Add high priority item
        high_obs = np.ones(64)
        learner.add_to_replay_buffer(high_obs, surprise=10.0)

        # High priority should be in buffer
        priorities = [exp['priority'] for exp in learner.replay_buffer]
        assert 10.0 in priorities


class TestSampleReplayBatch:
    """Tests for replay batch sampling with priority."""

    def test_sample_empty_buffer(self):
        """Test sampling from empty buffer returns empty list."""
        learner = OnlineLearner()

        batch = learner.sample_replay_batch(batch_size=10)

        assert batch == []

    def test_sample_normal_case(self):
        """Test sampling with normal priorities."""
        learner = OnlineLearner()

        # Add items with different priorities
        for i in range(10):
            obs = np.random.randn(64)
            learner.add_to_replay_buffer(obs, surprise=float(i + 1))

        batch = learner.sample_replay_batch(batch_size=5)

        assert len(batch) == 5
        for exp in batch:
            assert 'observation' in exp
            assert 'surprise' in exp
            assert 'priority' in exp

    def test_sample_with_zero_priorities_division_by_zero_fix(self):
        """Test that sampling handles all-zero priorities without division by zero."""
        learner = OnlineLearner()

        # Add items with zero priority (zero surprise)
        for i in range(10):
            obs = np.random.randn(64)
            learner.add_to_replay_buffer(obs, surprise=0.0)

        # This should NOT raise division by zero error
        batch = learner.sample_replay_batch(batch_size=5)

        assert len(batch) == 5
        # All items should have had equal probability of selection
        for exp in batch:
            assert exp['surprise'] == 0.0

    def test_sample_with_mixed_zero_and_nonzero_priorities(self):
        """Test sampling with mix of zero and non-zero priorities."""
        learner = OnlineLearner()

        # Add mix of zero and non-zero priorities
        for i in range(5):
            obs = np.random.randn(64)
            learner.add_to_replay_buffer(obs, surprise=0.0)
        for i in range(5):
            obs = np.random.randn(64)
            learner.add_to_replay_buffer(obs, surprise=float(i + 1))

        batch = learner.sample_replay_batch(batch_size=5)

        assert len(batch) == 5

    def test_sample_batch_size_larger_than_buffer(self):
        """Test sampling when batch size > buffer size."""
        learner = OnlineLearner()

        # Add fewer items than batch size
        for i in range(3):
            obs = np.random.randn(64)
            learner.add_to_replay_buffer(obs, surprise=1.0)

        batch = learner.sample_replay_batch(batch_size=10)

        # Should return all available items
        assert len(batch) == 3


class TestAdaptiveThresholds:
    """Tests for adaptive threshold updates."""

    def test_surprise_threshold_update(self):
        """Test surprise threshold exponential moving average."""
        learner = OnlineLearner()
        initial_threshold = learner.surprise_threshold

        learner.update_adaptive_threshold(5.0, 'surprise')

        # Should move towards 5.0
        assert learner.surprise_threshold > initial_threshold
        assert learner.surprise_threshold < 5.0

    def test_novelty_threshold_update(self):
        """Test novelty threshold exponential moving average."""
        learner = OnlineLearner()
        initial_threshold = learner.novelty_threshold

        learner.update_adaptive_threshold(0.1, 'novelty')

        # Should move towards 0.1
        assert learner.novelty_threshold < initial_threshold
        assert learner.novelty_threshold > 0.1

    def test_adaptive_threshold_disabled(self):
        """Test that thresholds don't update when disabled."""
        config = OnlineLearningConfig(adaptive_threshold=False)
        learner = OnlineLearner(config)
        initial_surprise = learner.surprise_threshold
        initial_novelty = learner.novelty_threshold

        learner.update_adaptive_threshold(10.0, 'surprise')
        learner.update_adaptive_threshold(0.0, 'novelty')

        assert learner.surprise_threshold == initial_surprise
        assert learner.novelty_threshold == initial_novelty


class TestEWCLoss:
    """Tests for Elastic Weight Consolidation loss."""

    def test_ewc_loss_no_fisher_info(self):
        """Test EWC loss with no Fisher information."""
        learner = OnlineLearner()

        current_params = {'layer1': np.ones(10)}
        old_params = {'layer1': np.zeros(10)}

        loss = learner.compute_ewc_loss(current_params, old_params)

        # No Fisher info means no loss
        assert loss == 0.0

    def test_ewc_loss_with_fisher_info(self):
        """Test EWC loss computation with Fisher information."""
        learner = OnlineLearner()

        # Set up Fisher information
        learner.fisher_information['layer1'] = np.ones(10)

        current_params = {'layer1': np.ones(10) * 2}
        old_params = {'layer1': np.ones(10)}

        loss = learner.compute_ewc_loss(current_params, old_params)

        # Loss should be positive
        assert loss > 0

    def test_ewc_loss_no_change(self):
        """Test EWC loss is zero when params unchanged."""
        learner = OnlineLearner()

        # Set up Fisher information
        learner.fisher_information['layer1'] = np.ones(10)

        params = {'layer1': np.ones(10)}

        loss = learner.compute_ewc_loss(params, params.copy())

        assert loss == 0.0


class TestFisherInformation:
    """Tests for Fisher information updates."""

    def test_update_fisher_new_param(self):
        """Test Fisher update for new parameter."""
        learner = OnlineLearner()

        gradient = np.ones(10) * 2.0
        learner.update_fisher_information('layer1', gradient)

        assert 'layer1' in learner.fisher_information
        # Should be 0.1 * gradient^2 = 0.1 * 4 = 0.4
        np.testing.assert_allclose(
            learner.fisher_information['layer1'],
            np.ones(10) * 0.4
        )

    def test_update_fisher_existing_param(self):
        """Test Fisher update for existing parameter."""
        learner = OnlineLearner()

        # First update
        gradient1 = np.ones(10) * 2.0
        learner.update_fisher_information('layer1', gradient1)

        # Second update
        gradient2 = np.ones(10) * 3.0
        learner.update_fisher_information('layer1', gradient2)

        # Running average: 0.9 * 0.4 + 0.1 * 9 = 0.36 + 0.9 = 1.26
        np.testing.assert_allclose(
            learner.fisher_information['layer1'],
            np.ones(10) * 1.26
        )


class TestOnlineUpdate:
    """Tests for the complete online update flow."""

    def test_online_update_basic(self):
        """Test basic online update flow."""
        learner = OnlineLearner()

        obs = np.random.randn(64)
        learner.online_update(obs, surprise=0.5)

        assert learner.total_updates == 1
        assert len(learner.replay_buffer) == 1

    def test_online_update_triggers_replay(self):
        """Test that online update triggers replay when buffer is full enough."""
        config = OnlineLearningConfig(replay_batch_size=5)
        learner = OnlineLearner(config)

        # Add enough items to trigger replay
        for i in range(10):
            obs = np.random.randn(64)
            learner.online_update(obs, surprise=1.0)

        # Replay should have occurred
        assert learner.replay_count > 0

    def test_online_update_with_update_fn(self):
        """Test online update with custom update function."""
        learner = OnlineLearner()
        update_calls = []

        def mock_update(obs):
            update_calls.append(obs)

        obs = np.random.randn(64)
        learner.online_update(obs, surprise=0.5, update_fn=mock_update)

        # Update function should have been called
        assert len(update_calls) >= 1


class TestGetStatistics:
    """Tests for statistics reporting."""

    def test_get_statistics(self):
        """Test that statistics are reported correctly."""
        learner = OnlineLearner()

        # Perform some updates
        for i in range(5):
            obs = np.random.randn(64)
            learner.online_update(obs, surprise=float(i))

        stats = learner.get_statistics()

        assert 'total_updates' in stats
        assert 'replay_count' in stats
        assert 'replay_buffer_size' in stats
        assert 'surprise_threshold' in stats
        assert 'novelty_threshold' in stats
        assert 'fisher_params' in stats

        assert stats['total_updates'] == 5
        assert stats['replay_buffer_size'] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
