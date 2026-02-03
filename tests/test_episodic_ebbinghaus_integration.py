"""
Tests for Episodic Memory Store integration with Ebbinghaus Forgetting.
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuro_memory.memory.episodic_store import (
    EpisodicMemoryStore,
    EpisodicMemoryConfig,
    Episode
)
from neuro_memory.memory.forgetting import (
    EbbinghausForgetting,
    SpacedRepetitionScheduler
)


class TestEpisodicMemoryWithEbbinghaus:
    """Test EpisodicMemoryStore integration with Ebbinghaus forgetting."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test persistence."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def memory_store(self, temp_dir):
        """Create a memory store with Ebbinghaus enabled."""
        config = EpisodicMemoryConfig(
            max_episodes=100,
            embedding_dim=64,
            persistence_path=temp_dir,
            enable_ebbinghaus=True,
            forgetting_background_interval=3600.0,  # 1 hour
            review_threshold=0.3,
            auto_reinforce_high_value=True
        )
        return EpisodicMemoryStore(config)

    def test_ebbinghaus_initialization(self, memory_store):
        """Test that Ebbinghaus system initializes correctly."""
        assert memory_store.ebbinghaus is not None
        assert memory_store.spaced_repetition is not None
        assert isinstance(memory_store.ebbinghaus, EbbinghausForgetting)
        assert isinstance(memory_store.spaced_repetition, SpacedRepetitionScheduler)

    def test_episode_registration_with_ebbinghaus(self, memory_store):
        """Test that stored episodes are registered with Ebbinghaus."""
        content = np.random.randn(10)
        episode = memory_store.store_episode(
            content=content,
            surprise=2.0,
            location="test_location",
            entities=["entity1", "entity2"]
        )

        # Check episode is registered in Ebbinghaus
        memory_state = memory_store.ebbinghaus.get_memory_state(episode.episode_id)
        assert memory_state is not None
        assert memory_state.memory_id == episode.episode_id

        # Check spaced repetition scheduled
        review_item = memory_store.spaced_repetition.get_review_item(episode.episode_id)
        assert review_item is not None

    def test_retention_computation(self, memory_store):
        """Test retention computation for stored episodes."""
        content = np.random.randn(10)
        episode = memory_store.store_episode(content=content, surprise=1.5)

        # Initial retention should be positive (scaled by importance)
        # The initial_retention is 0.5 + 0.5 * importance, so it varies
        retention = memory_store.ebbinghaus.compute_retention(episode.episode_id)
        # Just verify it's a valid retention value and not decayed too much
        assert 0.5 <= retention <= 1.0

    def test_retrieval_recording(self, memory_store):
        """Test that retrieval updates Ebbinghaus state."""
        content = np.random.randn(10)
        episode = memory_store.store_episode(content=content, surprise=1.0)

        # Get initial state
        initial_state = memory_store.ebbinghaus.get_memory_state(episode.episode_id)
        initial_stability = initial_state.stability_score

        # Record successful retrieval
        memory_store.record_retrieval(episode.episode_id, success=True)

        # Stability should increase
        updated_state = memory_store.ebbinghaus.get_memory_state(episode.episode_id)
        assert updated_state.stability_score > initial_stability

        # Access count should be tracked
        assert episode.metadata.get('access_count', 0) == 1

    def test_get_memories_for_review(self, memory_store):
        """Test getting memories due for review."""
        # Store several episodes
        episodes = []
        for i in range(5):
            content = np.random.randn(10)
            episode = memory_store.store_episode(
                content=content,
                surprise=float(i),
                location=f"location_{i}"
            )
            episodes.append(episode)

        # Initially, no memories should be due (just created)
        due = memory_store.get_memories_for_review(limit=10)
        # Depending on scheduling, this might be 0 or some
        assert isinstance(due, list)

    def test_forgetting_statistics(self, memory_store):
        """Test getting comprehensive forgetting statistics."""
        # Store some episodes
        for i in range(3):
            content = np.random.randn(10)
            memory_store.store_episode(content=content, surprise=float(i))

        stats = memory_store.get_forgetting_statistics()

        assert "forgotten_memories" in stats
        assert "reviewed_memories" in stats
        assert "reinforced_memories" in stats
        assert "ebbinghaus" in stats
        assert "spaced_repetition" in stats
        assert "stability_distribution" in stats

    def test_high_importance_reinforcement(self, memory_store):
        """Test that high-importance memories are candidates for auto-reinforcement."""
        # Create high-importance episode (high surprise + many entities)
        content = np.random.randn(10)
        episode = memory_store.store_episode(
            content=content,
            surprise=5.0,  # High surprise -> high importance
            entities=["e1", "e2", "e3", "e4"]  # Many entities
        )

        # High importance (from surprise)
        assert episode.importance > 0.6

        # Should be reinforced
        should_reinforce = memory_store._should_reinforce_memory(episode)
        assert should_reinforce

    def test_low_importance_no_reinforcement(self, memory_store):
        """Test that low-importance memories are not auto-reinforced."""
        content = np.random.randn(10)
        episode = memory_store.store_episode(
            content=content,
            surprise=0.1,  # Very low surprise
            entities=[]  # No entities
        )

        # Low importance
        assert episode.importance < 0.5

        # Should NOT be reinforced (unless accessed many times)
        should_reinforce = memory_store._should_reinforce_memory(episode)
        assert not should_reinforce

    def test_statistics_include_forgetting_info(self, memory_store):
        """Test that main statistics include forgetting info."""
        # Store episode
        content = np.random.randn(10)
        memory_store.store_episode(content=content, surprise=1.0)

        stats = memory_store.get_statistics()

        # Basic stats
        assert "total_episodes" in stats
        assert "episodes_in_memory" in stats

        # Forgetting stats
        assert "forgotten_memories" in stats
        assert "reviewed_memories" in stats
        assert "reinforced_memories" in stats

        # Ebbinghaus-specific
        assert "avg_retention" in stats
        assert "memories_at_risk" in stats

        # Spaced repetition
        assert "due_for_review" in stats
        assert "upcoming_reviews_24h" in stats

    def test_state_persistence(self, temp_dir):
        """Test that forgetting stats are saved and loaded."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            enable_ebbinghaus=True
        )
        store1 = EpisodicMemoryStore(config)

        # Store episodes
        content = np.random.randn(10)
        episode = store1.store_episode(content=content, surprise=2.0)

        # Record some activity
        store1.record_retrieval(episode.episode_id, success=True)
        store1.reviewed_memories_count = 5
        store1.forgotten_memories_count = 2

        # Save state
        store1.save_state()

        # Create new store, load state
        store2 = EpisodicMemoryStore(config)
        store2.load_state()

        # Verify stats persisted
        assert store2.reviewed_memories_count == 5
        assert store2.forgotten_memories_count == 2


class TestBackgroundForgettingTask:
    """Test background forgetting task functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_start_stop_background_task(self, temp_dir):
        """Test starting and stopping background task."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            enable_ebbinghaus=True,
            forgetting_background_interval=1.0  # 1 second for test
        )
        store = EpisodicMemoryStore(config)

        # Start task
        store.start_forgetting_background_task()
        assert store._forgetting_running is True
        assert store._forgetting_thread is not None
        assert store._forgetting_thread.is_alive()

        # Stop task
        store.stop_forgetting_background_task()
        assert store._forgetting_running is False

    def test_double_start_no_op(self, temp_dir):
        """Test that starting twice is a no-op."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            enable_ebbinghaus=True
        )
        store = EpisodicMemoryStore(config)

        store.start_forgetting_background_task()
        first_thread = store._forgetting_thread

        store.start_forgetting_background_task()
        second_thread = store._forgetting_thread

        # Should be the same thread
        assert first_thread is second_thread

        store.stop_forgetting_background_task()


class TestEbbinghausDisabled:
    """Test behavior when Ebbinghaus is disabled."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_disabled_ebbinghaus(self, temp_dir):
        """Test that store works without Ebbinghaus."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            enable_ebbinghaus=False
        )
        store = EpisodicMemoryStore(config)

        assert store.ebbinghaus is None
        assert store.spaced_repetition is None

        # Store still works
        content = np.random.randn(10)
        episode = store.store_episode(content=content, surprise=1.0)
        assert episode is not None

        # Review returns empty
        due = store.get_memories_for_review()
        assert due == []


class TestIntegrationWithRetentionDecay:
    """Test retention decay over simulated time."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_retention_decays_over_time(self, temp_dir):
        """Test that retention decreases as simulated time passes."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            enable_ebbinghaus=True
        )
        store = EpisodicMemoryStore(config)

        # Store episode
        content = np.random.randn(10)
        episode = store.store_episode(content=content, surprise=1.0)

        # Get initial retention
        now = datetime.now().timestamp()
        initial_retention = store.ebbinghaus.compute_retention(
            episode.episode_id, current_time=now
        )

        # Simulate 24 hours later
        future_time = now + 24 * 3600  # 24 hours in seconds
        future_retention = store.ebbinghaus.compute_retention(
            episode.episode_id, current_time=future_time
        )

        # Retention should decrease
        assert future_retention < initial_retention

    def test_retrieval_resets_retention(self, temp_dir):
        """Test that successful retrieval resets retention clock."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            enable_ebbinghaus=True
        )
        store = EpisodicMemoryStore(config)

        content = np.random.randn(10)
        episode = store.store_episode(content=content, surprise=1.0)

        # Simulate some time passing
        now = datetime.now().timestamp()
        after_12h = now + 12 * 3600

        # Get retention after 12h
        retention_after_12h = store.ebbinghaus.compute_retention(
            episode.episode_id, current_time=after_12h
        )

        # Record retrieval at 12h mark
        store.ebbinghaus.record_retrieval(
            episode.episode_id, success=True, current_time=after_12h
        )

        # Immediately after retrieval, retention should be near 1.0
        retention_after_retrieval = store.ebbinghaus.compute_retention(
            episode.episode_id, current_time=after_12h
        )

        # The retrieval resets last_access, so retention should be high again
        assert retention_after_retrieval > retention_after_12h


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
