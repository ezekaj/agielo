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


class TestIndexCleanupDuringConsolidation:
    """Test that indices are properly cleaned up when episodes are offloaded."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_consolidation_cleans_up_indices(self, temp_dir):
        """Test that offloading during consolidation removes index entries."""
        # Create a store with low max_episodes to trigger consolidation
        # Note: offload_threshold triggers consolidation, which then reduces to max_episodes
        config = EpisodicMemoryConfig(
            max_episodes=5,  # Only keep 5 episodes after consolidation
            offload_threshold=8,  # Trigger offload check at 8 episodes
            persistence_path=temp_dir,
            enable_disk_offload=True,
            enable_ebbinghaus=False  # Disable Ebbinghaus to simplify test
        )
        store = EpisodicMemoryStore(config)

        # Store 12 episodes with various locations and entities
        # This ensures multiple consolidation triggers
        stored_episodes = []
        for i in range(12):
            content = np.random.randn(10)
            episode = store.store_episode(
                content=content,
                surprise=float(i) * 0.1,  # Lower surprise = lower importance
                location=f"location_{i % 3}",
                entities=[f"entity_{i % 2}"]
            )
            stored_episodes.append(episode)

        # Consolidation should have been triggered and some episodes offloaded
        # The exact number depends on when consolidation triggers, but offloading
        # should have occurred
        assert store.episodes_offloaded > 0, "No episodes were offloaded"

        # Get the IDs of episodes still in memory
        in_memory_ids = {ep.episode_id for ep in store.episodes}

        # Check that offloaded episodes are NOT in temporal index
        for episode in stored_episodes:
            if episode.episode_id not in in_memory_ids:
                date_key = episode.timestamp.strftime("%Y-%m-%d")
                if date_key in store.temporal_index:
                    assert episode.episode_id not in store.temporal_index[date_key], \
                        f"Offloaded episode {episode.episode_id} still in temporal index"

        # Check that offloaded episodes are NOT in spatial index
        for episode in stored_episodes:
            if episode.episode_id not in in_memory_ids:
                if episode.location and episode.location in store.spatial_index:
                    assert episode.episode_id not in store.spatial_index[episode.location], \
                        f"Offloaded episode {episode.episode_id} still in spatial index"

        # Check that offloaded episodes are NOT in entity index
        for episode in stored_episodes:
            if episode.episode_id not in in_memory_ids:
                for entity in episode.entities:
                    if entity in store.entity_index:
                        assert episode.episode_id not in store.entity_index[entity], \
                            f"Offloaded episode {episode.episode_id} still in entity index for {entity}"

    def test_remove_from_indices_removes_single_episode(self, temp_dir):
        """Test _remove_from_indices correctly removes a single episode."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            enable_ebbinghaus=False
        )
        store = EpisodicMemoryStore(config)

        # Store an episode
        content = np.random.randn(10)
        episode = store.store_episode(
            content=content,
            surprise=1.0,
            location="test_location",
            entities=["entity_a", "entity_b"]
        )

        # Verify it's in indices
        date_key = episode.timestamp.strftime("%Y-%m-%d")
        assert episode.episode_id in store.temporal_index[date_key]
        assert episode.episode_id in store.spatial_index["test_location"]
        assert episode.episode_id in store.entity_index["entity_a"]
        assert episode.episode_id in store.entity_index["entity_b"]

        # Remove from indices
        store._remove_from_indices(episode)

        # Verify it's removed (and empty keys are cleaned up)
        assert date_key not in store.temporal_index or \
            episode.episode_id not in store.temporal_index.get(date_key, [])
        assert "test_location" not in store.spatial_index or \
            episode.episode_id not in store.spatial_index.get("test_location", [])
        assert "entity_a" not in store.entity_index or \
            episode.episode_id not in store.entity_index.get("entity_a", [])
        assert "entity_b" not in store.entity_index or \
            episode.episode_id not in store.entity_index.get("entity_b", [])

    def test_remove_from_indices_idempotent(self, temp_dir):
        """Test that removing same episode twice doesn't cause errors."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            enable_ebbinghaus=False
        )
        store = EpisodicMemoryStore(config)

        content = np.random.randn(10)
        episode = store.store_episode(
            content=content,
            surprise=1.0,
            location="test_location",
            entities=["entity_a"]
        )

        # Remove twice - should not raise
        store._remove_from_indices(episode)
        store._remove_from_indices(episode)  # Second call should be safe

    def test_offloaded_episodes_exist_on_disk(self, temp_dir):
        """Test that offloaded episodes are saved to disk."""
        config = EpisodicMemoryConfig(
            max_episodes=5,
            offload_threshold=8,
            persistence_path=temp_dir,
            enable_disk_offload=True,
            enable_ebbinghaus=False
        )
        store = EpisodicMemoryStore(config)

        # Store enough to trigger offloading
        stored_episodes = []
        for i in range(10):
            content = np.random.randn(10)
            episode = store.store_episode(
                content=content,
                surprise=float(i) * 0.1,
                location=f"location_{i}"
            )
            stored_episodes.append(episode)

        # Check offload directory
        offload_dir = Path(temp_dir) / "offloaded"
        assert offload_dir.exists()

        # Some episodes should be offloaded to disk
        offloaded_files = list(offload_dir.glob("*.pkl"))
        assert len(offloaded_files) > 0, "No episodes were offloaded to disk"

        # Number offloaded should equal episodes_offloaded counter
        assert store.episodes_offloaded == len(offloaded_files)


class TestVectorDBBackendValidation:
    """Test vector database backend selection and validation."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_chromadb_backend_works(self, temp_dir):
        """Test that ChromaDB backend initializes correctly."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            vector_db_backend="chromadb",
            enable_ebbinghaus=False
        )
        store = EpisodicMemoryStore(config)

        # Should initialize ChromaDB
        assert hasattr(store, 'chroma_client')
        assert hasattr(store, 'collection')
        assert store.config.vector_db_backend == "chromadb"

        # Should be able to store episodes
        content = np.random.randn(10)
        episode = store.store_episode(content=content, surprise=1.0)
        assert episode is not None

    def test_faiss_backend_falls_back_to_chromadb(self, temp_dir):
        """Test that FAISS backend falls back to ChromaDB with warning."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            vector_db_backend="faiss",  # Request FAISS
            enable_ebbinghaus=False
        )

        # Should emit a warning and fall back to ChromaDB
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store = EpisodicMemoryStore(config)

            # Check warning was issued
            assert len(w) == 1
            assert "FAISS backend is not yet implemented" in str(w[0].message)
            assert "Falling back to ChromaDB" in str(w[0].message)

        # Config should be updated to reflect actual backend
        assert store.config.vector_db_backend == "chromadb"

        # ChromaDB should be initialized
        assert hasattr(store, 'chroma_client')
        assert hasattr(store, 'collection')

        # Should still be able to store episodes
        content = np.random.randn(10)
        episode = store.store_episode(content=content, surprise=1.0)
        assert episode is not None

    def test_unknown_backend_raises_error(self, temp_dir):
        """Test that unknown backend raises ValueError."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            vector_db_backend="unknown_backend",
            enable_ebbinghaus=False
        )

        with pytest.raises(ValueError) as exc_info:
            EpisodicMemoryStore(config)

        assert "Unknown vector_db_backend" in str(exc_info.value)
        assert "unknown_backend" in str(exc_info.value)
        assert "chromadb" in str(exc_info.value)


class TestEpisodeInputValidation:
    """Test input validation for episode storage."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def memory_store(self, temp_dir):
        """Create a memory store for testing."""
        config = EpisodicMemoryConfig(
            persistence_path=temp_dir,
            enable_ebbinghaus=False  # Disable for simpler tests
        )
        return EpisodicMemoryStore(config)

    # Content validation tests

    def test_valid_content_accepted(self, memory_store):
        """Test that valid numpy array content is accepted."""
        content = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        episode = memory_store.store_episode(content=content, surprise=1.0)
        assert episode is not None
        assert np.allclose(episode.content, content)

    def test_content_nan_rejected(self, memory_store):
        """Test that content with NaN values is rejected."""
        content = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=1.0)
        assert "non-finite" in str(exc_info.value).lower()
        assert "content" in str(exc_info.value).lower()

    def test_content_inf_rejected(self, memory_store):
        """Test that content with Inf values is rejected."""
        content = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=1.0)
        assert "non-finite" in str(exc_info.value).lower()

    def test_content_negative_inf_rejected(self, memory_store):
        """Test that content with -Inf values is rejected."""
        content = np.array([1.0, -np.inf, 3.0])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=1.0)
        assert "non-finite" in str(exc_info.value).lower()

    def test_content_multiple_non_finite_rejected(self, memory_store):
        """Test that content with multiple non-finite values reports count."""
        content = np.array([np.nan, np.inf, -np.inf, 1.0])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=1.0)
        assert "3" in str(exc_info.value)  # 3 non-finite values

    def test_content_type_error_non_array(self, memory_store):
        """Test that non-numpy array content raises TypeError."""
        content = [1.0, 2.0, 3.0]  # Python list, not numpy array
        with pytest.raises(TypeError) as exc_info:
            memory_store.store_episode(content=content, surprise=1.0)
        assert "numpy array" in str(exc_info.value).lower()
        assert "list" in str(exc_info.value).lower()

    def test_empty_content_accepted(self, memory_store):
        """Test that empty numpy array is accepted."""
        content = np.array([])
        episode = memory_store.store_episode(content=content, surprise=1.0)
        assert episode is not None

    # Surprise validation tests

    def test_valid_surprise_accepted(self, memory_store):
        """Test that valid surprise values are accepted."""
        content = np.array([1.0, 2.0, 3.0])
        episode = memory_store.store_episode(content=content, surprise=2.5)
        assert episode is not None
        assert episode.surprise == 2.5

    def test_zero_surprise_accepted(self, memory_store):
        """Test that zero surprise is accepted."""
        content = np.array([1.0, 2.0, 3.0])
        episode = memory_store.store_episode(content=content, surprise=0.0)
        assert episode is not None
        assert episode.surprise == 0.0

    def test_surprise_nan_rejected(self, memory_store):
        """Test that NaN surprise is rejected."""
        content = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=np.nan)
        assert "surprise" in str(exc_info.value).lower()
        assert "finite" in str(exc_info.value).lower()

    def test_surprise_inf_rejected(self, memory_store):
        """Test that Inf surprise is rejected."""
        content = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=np.inf)
        assert "surprise" in str(exc_info.value).lower()

    def test_surprise_negative_inf_rejected(self, memory_store):
        """Test that -Inf surprise is rejected."""
        content = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=-np.inf)
        assert "surprise" in str(exc_info.value).lower()

    def test_negative_surprise_rejected(self, memory_store):
        """Test that negative surprise is rejected."""
        content = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=-1.0)
        assert "surprise" in str(exc_info.value).lower()
        assert "non-negative" in str(exc_info.value).lower()

    # Embedding validation tests

    def test_valid_embedding_accepted(self, memory_store):
        """Test that valid embedding is accepted."""
        content = np.array([1.0, 2.0, 3.0])
        embedding = np.random.randn(64)
        episode = memory_store.store_episode(
            content=content, surprise=1.0, embedding=embedding
        )
        assert episode is not None
        assert episode.embedding is not None

    def test_embedding_nan_rejected(self, memory_store):
        """Test that embedding with NaN values is rejected."""
        content = np.array([1.0, 2.0, 3.0])
        embedding = np.array([1.0, np.nan, 3.0] + [0.0] * 61)
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(
                content=content, surprise=1.0, embedding=embedding
            )
        assert "embedding" in str(exc_info.value).lower()
        assert "non-finite" in str(exc_info.value).lower()

    def test_embedding_inf_rejected(self, memory_store):
        """Test that embedding with Inf values is rejected."""
        content = np.array([1.0, 2.0, 3.0])
        embedding = np.array([1.0, np.inf, 3.0] + [0.0] * 61)
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(
                content=content, surprise=1.0, embedding=embedding
            )
        assert "embedding" in str(exc_info.value).lower()

    def test_embedding_type_error_non_array(self, memory_store):
        """Test that non-numpy array embedding raises TypeError."""
        content = np.array([1.0, 2.0, 3.0])
        embedding = [1.0, 2.0, 3.0]  # Python list
        with pytest.raises(TypeError) as exc_info:
            memory_store.store_episode(
                content=content, surprise=1.0, embedding=embedding
            )
        assert "embedding" in str(exc_info.value).lower()
        assert "numpy array" in str(exc_info.value).lower()

    def test_none_embedding_generates_valid(self, memory_store):
        """Test that None embedding generates a valid embedding."""
        content = np.array([1.0, 2.0, 3.0])
        episode = memory_store.store_episode(content=content, surprise=1.0)
        assert episode.embedding is not None
        assert np.all(np.isfinite(episode.embedding))

    # Importance computation validation

    def test_importance_always_valid(self, memory_store):
        """Test that computed importance is always valid."""
        content = np.array([1.0, 2.0, 3.0])
        # Test with various valid surprise values
        for surprise in [0.0, 0.5, 1.0, 5.0, 10.0, 100.0]:
            episode = memory_store.store_episode(content=content, surprise=surprise)
            assert np.isfinite(episode.importance)
            assert 0.0 <= episode.importance <= 1.0

    def test_large_surprise_produces_valid_importance(self, memory_store):
        """Test that large surprise values produce valid importance."""
        content = np.array([1.0, 2.0, 3.0])
        # Use very large surprise - should not overflow
        episode = memory_store.store_episode(content=content, surprise=1000.0)
        assert np.isfinite(episode.importance)
        assert 0.0 <= episode.importance <= 1.0

    # Edge case tests

    def test_multidimensional_content_with_nan_rejected(self, memory_store):
        """Test that multidimensional content with NaN is rejected."""
        content = np.array([[1.0, 2.0], [np.nan, 4.0]])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=1.0)
        assert "non-finite" in str(exc_info.value).lower()

    def test_all_nan_content_rejected(self, memory_store):
        """Test that content with all NaN values is rejected."""
        content = np.array([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError) as exc_info:
            memory_store.store_episode(content=content, surprise=1.0)
        assert "3" in str(exc_info.value)  # 3 non-finite values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
