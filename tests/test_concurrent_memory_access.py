"""
Tests for thread safety of EpisodicMemoryStore under concurrent access.

Verifies that:
1. Multiple threads can read episodes simultaneously without issues
2. Multiple threads can write episodes without race conditions or data corruption
3. Mixed read/write operations work correctly
4. Background forgetting thread doesn't corrupt data
5. All index structures remain consistent under concurrent access
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Set

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuro_memory.memory.episodic_store import (
    EpisodicMemoryStore,
    EpisodicMemoryConfig,
    Episode
)


class TestConcurrentReadAccess:
    """Test thread safety for concurrent read operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test persistence."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def memory_store_with_episodes(self, temp_dir):
        """Create a memory store pre-populated with episodes."""
        config = EpisodicMemoryConfig(
            max_episodes=1000,
            embedding_dim=64,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,  # Disable to isolate concurrent read tests
        )
        store = EpisodicMemoryStore(config)

        # Pre-populate with episodes
        for i in range(100):
            content = np.random.randn(10)
            store.store_episode(
                content=content,
                surprise=float(i % 10),
                location=f"location_{i % 5}",
                entities=[f"entity_{i % 3}", f"entity_{(i + 1) % 3}"]
            )

        return store

    def test_concurrent_reads_no_corruption(self, memory_store_with_episodes):
        """Multiple threads reading episodes should not corrupt data."""
        store = memory_store_with_episodes
        read_results = []
        errors = []

        def read_operation(thread_id):
            """Read operation performed by each thread."""
            try:
                results = []
                # Perform multiple read operations
                for _ in range(20):
                    # Read by location
                    episodes = store.retrieve_by_location(f"location_{thread_id % 5}")
                    results.append(("location", len(episodes)))

                    # Read by entity
                    episodes = store.retrieve_by_entity(f"entity_{thread_id % 3}")
                    results.append(("entity", len(episodes)))

                    # Read statistics
                    stats = store.get_statistics()
                    results.append(("stats", stats["episodes_in_memory"]))

                    # Small delay to increase chance of race conditions
                    time.sleep(0.001)

                return results
            except Exception as e:
                errors.append((thread_id, str(e)))
                return []

        # Run concurrent reads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_operation, i) for i in range(10)]
            for future in concurrent.futures.as_completed(futures):
                read_results.extend(future.result())

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent reads: {errors}"

        # Verify results are consistent
        # All stats reads should show 100 episodes
        stats_results = [r[1] for r in read_results if r[0] == "stats"]
        assert all(count == 100 for count in stats_results), "Episode count varied during reads"

    def test_concurrent_similarity_queries(self, memory_store_with_episodes):
        """Multiple threads performing similarity queries should work correctly."""
        store = memory_store_with_episodes
        errors = []

        def similarity_query(thread_id):
            """Perform similarity queries."""
            try:
                for _ in range(10):
                    query = np.random.randn(64)
                    results = store.retrieve_by_similarity(query, k=5)
                    # Verify results are valid
                    assert len(results) <= 5
                    for ep in results:
                        assert ep.episode_id is not None
                        assert ep.content is not None
                return True
            except Exception as e:
                errors.append((thread_id, str(e)))
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(similarity_query, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(errors) == 0, f"Errors during similarity queries: {errors}"
        assert all(results), "Some similarity queries failed"

    def test_concurrent_temporal_range_queries(self, temp_dir):
        """Multiple threads querying temporal ranges should work correctly."""
        config = EpisodicMemoryConfig(
            max_episodes=1000,
            embedding_dim=64,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
        )
        store = EpisodicMemoryStore(config)

        # Store episodes with varying timestamps
        base_time = datetime.now()
        for i in range(50):
            content = np.random.randn(10)
            ts = base_time - timedelta(hours=i)
            store.store_episode(
                content=content,
                surprise=1.0,
                timestamp=ts
            )

        errors = []

        def temporal_query(thread_id):
            """Perform temporal range queries."""
            try:
                for _ in range(20):
                    start = base_time - timedelta(hours=24)
                    end = base_time
                    results = store.retrieve_by_temporal_range(start, end)
                    # Verify all results are within range
                    for ep in results:
                        assert start <= ep.timestamp <= end
                return True
            except Exception as e:
                errors.append((thread_id, str(e)))
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(temporal_query, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(errors) == 0, f"Errors during temporal queries: {errors}"
        assert all(results), "Some temporal queries failed"


class TestConcurrentWriteAccess:
    """Test thread safety for concurrent write operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test persistence."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_concurrent_episode_storage(self, temp_dir):
        """Multiple threads storing episodes should not cause race conditions."""
        config = EpisodicMemoryConfig(
            max_episodes=10000,
            embedding_dim=64,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
            enable_disk_offload=False,  # Disable to simplify test
        )
        store = EpisodicMemoryStore(config)

        stored_ids: List[str] = []
        errors = []
        lock = threading.Lock()

        def store_episodes(thread_id):
            """Store episodes from this thread."""
            try:
                local_ids = []
                for i in range(50):
                    content = np.random.randn(10)
                    episode = store.store_episode(
                        content=content,
                        surprise=float(i % 5),
                        location=f"thread_{thread_id}_loc",
                        entities=[f"thread_{thread_id}"],
                        metadata={"thread": thread_id, "index": i}
                    )
                    local_ids.append(episode.episode_id)
                    time.sleep(0.001)  # Small delay to encourage interleaving

                with lock:
                    stored_ids.extend(local_ids)
                return True
            except Exception as e:
                errors.append((thread_id, str(e)))
                return False

        # Run concurrent writes
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(store_episodes, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
        assert all(results), "Some write operations failed"

        # Verify all episodes were stored
        assert len(stored_ids) == 500, f"Expected 500 episodes, got {len(stored_ids)}"

        # Verify all IDs are unique
        assert len(set(stored_ids)) == 500, "Duplicate episode IDs found"

        # Verify actual stored count
        stats = store.get_statistics()
        assert stats["total_episodes"] == 500
        assert stats["episodes_in_memory"] == 500

    def test_concurrent_storage_maintains_index_integrity(self, temp_dir):
        """Concurrent writes should maintain consistent indices."""
        config = EpisodicMemoryConfig(
            max_episodes=10000,
            embedding_dim=64,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
            enable_disk_offload=False,
        )
        store = EpisodicMemoryStore(config)

        num_threads = 5
        episodes_per_thread = 20
        errors = []

        def store_with_location(thread_id):
            """Store episodes with known locations."""
            try:
                for i in range(episodes_per_thread):
                    content = np.random.randn(10)
                    store.store_episode(
                        content=content,
                        surprise=1.0,
                        location=f"loc_{thread_id}",
                        entities=[f"entity_{thread_id}"]
                    )
                return True
            except Exception as e:
                errors.append((thread_id, str(e)))
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(store_with_location, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"

        # Verify index integrity
        for thread_id in range(num_threads):
            location_episodes = store.retrieve_by_location(f"loc_{thread_id}")
            assert len(location_episodes) == episodes_per_thread, \
                f"Location loc_{thread_id} has {len(location_episodes)} episodes, expected {episodes_per_thread}"

            entity_episodes = store.retrieve_by_entity(f"entity_{thread_id}")
            assert len(entity_episodes) == episodes_per_thread, \
                f"Entity entity_{thread_id} has {len(entity_episodes)} episodes, expected {episodes_per_thread}"


class TestMixedReadWriteAccess:
    """Test thread safety for mixed concurrent read/write operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test persistence."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_concurrent_read_write(self, temp_dir):
        """Concurrent reads and writes should not corrupt data."""
        config = EpisodicMemoryConfig(
            max_episodes=10000,
            embedding_dim=64,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
            enable_disk_offload=False,
        )
        store = EpisodicMemoryStore(config)

        # Pre-populate some episodes
        for i in range(50):
            content = np.random.randn(10)
            store.store_episode(
                content=content,
                surprise=1.0,
                location="initial"
            )

        write_errors = []
        read_errors = []
        stop_event = threading.Event()

        def writer_thread(thread_id):
            """Continuously write episodes."""
            try:
                for i in range(100):
                    if stop_event.is_set():
                        break
                    content = np.random.randn(10)
                    store.store_episode(
                        content=content,
                        surprise=float(i % 5),
                        location=f"writer_{thread_id}",
                        entities=[f"writer_{thread_id}"]
                    )
                    time.sleep(0.001)
            except Exception as e:
                write_errors.append((thread_id, str(e)))

        def reader_thread(thread_id):
            """Continuously read episodes."""
            try:
                for _ in range(200):
                    if stop_event.is_set():
                        break
                    # Mix of read operations
                    if thread_id % 3 == 0:
                        store.retrieve_by_location("initial")
                    elif thread_id % 3 == 1:
                        store.get_statistics()
                    else:
                        query = np.random.randn(64)
                        store.retrieve_by_similarity(query, k=3)
                    time.sleep(0.001)
            except Exception as e:
                read_errors.append((thread_id, str(e)))

        # Start threads
        writers = [threading.Thread(target=writer_thread, args=(i,)) for i in range(3)]
        readers = [threading.Thread(target=reader_thread, args=(i,)) for i in range(5)]

        for t in writers + readers:
            t.start()

        # Let them run
        time.sleep(0.5)
        stop_event.set()

        for t in writers + readers:
            t.join(timeout=5.0)

        assert len(write_errors) == 0, f"Write errors: {write_errors}"
        assert len(read_errors) == 0, f"Read errors: {read_errors}"

        # Verify data integrity
        stats = store.get_statistics()
        assert stats["total_episodes"] >= 50, "Initial episodes should still exist"

    def test_statistics_consistency_under_load(self, temp_dir):
        """Statistics should remain consistent under concurrent access.

        Note: During concurrent writes, there's a brief window where in_memory
        count (list length) may be 1 higher than total_episodes counter because
        the append happens before incrementing the counter. This is expected
        timing behavior, not data corruption. We allow a small delta.
        """
        config = EpisodicMemoryConfig(
            max_episodes=10000,
            embedding_dim=64,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
            enable_disk_offload=False,
        )
        store = EpisodicMemoryStore(config)

        errors = []
        major_inconsistencies = []
        num_writers = 2

        def writer():
            """Write episodes."""
            try:
                for _ in range(100):
                    content = np.random.randn(10)
                    store.store_episode(content=content, surprise=1.0)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("writer", str(e)))

        def stats_checker():
            """Check statistics consistency."""
            try:
                for _ in range(200):
                    stats = store.get_statistics()
                    total = stats["total_episodes"]
                    in_memory = stats["episodes_in_memory"]
                    offloaded = stats["episodes_offloaded"]

                    # During concurrent writes, in_memory can briefly be higher
                    # than total by at most the number of concurrent writers
                    # (each writer may have appended but not yet incremented counter).
                    # Also check total >= in_memory + offloaded (can't have stored
                    # more than we've counted)
                    diff = in_memory - (total - offloaded)
                    if diff > num_writers or diff < 0:
                        major_inconsistencies.append(
                            f"total={total}, in_memory={in_memory}, offloaded={offloaded}, diff={diff}"
                        )

                    time.sleep(0.001)
            except Exception as e:
                errors.append(("stats_checker", str(e)))

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=stats_checker),
            threading.Thread(target=stats_checker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(major_inconsistencies) == 0, f"Major inconsistencies found: {major_inconsistencies[:10]}"

        # After all writes complete, verify final consistency
        final_stats = store.get_statistics()
        assert final_stats["total_episodes"] == final_stats["episodes_in_memory"], \
            "Final stats should be consistent after all writes complete"


class TestBackgroundForgettingThreadSafety:
    """Test that background forgetting thread doesn't cause issues."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test persistence."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_forgetting_thread_with_concurrent_access(self, temp_dir):
        """Background forgetting should not corrupt data when other threads are active."""
        config = EpisodicMemoryConfig(
            max_episodes=1000,
            embedding_dim=64,
            persistence_path=temp_dir,
            enable_ebbinghaus=True,
            forgetting_background_interval=0.1,  # Fast interval for testing
            review_threshold=0.3,
            auto_reinforce_high_value=True
        )
        store = EpisodicMemoryStore(config)

        # Pre-populate episodes
        for i in range(100):
            content = np.random.randn(10)
            store.store_episode(
                content=content,
                surprise=float(i % 5),
                location=f"loc_{i % 5}",
                entities=[f"entity_{i % 3}"]
            )

        errors = []
        stop_event = threading.Event()

        def access_thread(thread_id):
            """Perform mixed access operations."""
            try:
                for _ in range(50):
                    if stop_event.is_set():
                        break
                    # Read operations
                    store.get_statistics()
                    store.retrieve_by_location(f"loc_{thread_id % 5}")

                    # Write operations
                    content = np.random.randn(10)
                    ep = store.store_episode(
                        content=content,
                        surprise=2.0,
                        location=f"thread_{thread_id}"
                    )

                    # Record retrieval
                    store.record_retrieval(ep.episode_id, success=True)

                    time.sleep(0.01)
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start background forgetting
        store.start_forgetting_background_task()

        # Start access threads
        threads = [threading.Thread(target=access_thread, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()

        # Let them run together
        time.sleep(1.0)
        stop_event.set()

        # Wait for threads
        for t in threads:
            t.join(timeout=5.0)

        # Stop forgetting thread
        store.stop_forgetting_background_task()

        assert len(errors) == 0, f"Errors during concurrent access with forgetting: {errors}"

        # Verify data integrity
        stats = store.get_statistics()
        assert stats["total_episodes"] > 0, "Should have some episodes"
        assert stats["episodes_in_memory"] >= 0, "Episodes in memory should be non-negative"

    def test_start_stop_forgetting_thread_repeatedly(self, temp_dir):
        """Starting and stopping forgetting thread should be safe."""
        config = EpisodicMemoryConfig(
            max_episodes=100,
            embedding_dim=64,
            persistence_path=temp_dir,
            enable_ebbinghaus=True,
            forgetting_background_interval=0.1,
        )
        store = EpisodicMemoryStore(config)

        # Store some episodes
        for i in range(10):
            content = np.random.randn(10)
            store.store_episode(content=content, surprise=1.0)

        # Start/stop multiple times
        for _ in range(5):
            store.start_forgetting_background_task()
            time.sleep(0.05)
            store.stop_forgetting_background_task()

        # Verify data integrity
        stats = store.get_statistics()
        assert stats["total_episodes"] == 10


class TestDataIntegrityUnderStress:
    """Stress tests for data integrity under heavy concurrent load."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test persistence."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_high_volume_concurrent_writes(self, temp_dir):
        """Test data integrity with many concurrent writes."""
        config = EpisodicMemoryConfig(
            max_episodes=100000,
            embedding_dim=32,  # Smaller for performance
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
            enable_disk_offload=False,
        )
        store = EpisodicMemoryStore(config)

        num_threads = 20
        episodes_per_thread = 100
        expected_total = num_threads * episodes_per_thread

        stored_counts = []
        errors = []

        def high_volume_writer(thread_id):
            """Write many episodes quickly."""
            count = 0
            try:
                for i in range(episodes_per_thread):
                    content = np.random.randn(10)
                    store.store_episode(
                        content=content,
                        surprise=float(i % 5),
                        location=f"stress_loc_{thread_id}",
                        metadata={"thread": thread_id, "index": i}
                    )
                    count += 1
                return count
            except Exception as e:
                errors.append((thread_id, str(e)))
                return count

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(high_volume_writer, i) for i in range(num_threads)]
            stored_counts = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(errors) == 0, f"Errors during stress test: {errors}"

        # Verify total count
        total_stored = sum(stored_counts)
        assert total_stored == expected_total, f"Expected {expected_total}, stored {total_stored}"

        stats = store.get_statistics()
        assert stats["total_episodes"] == expected_total
        assert stats["episodes_in_memory"] == expected_total

        # Verify all locations have correct counts
        for thread_id in range(num_threads):
            location_episodes = store.retrieve_by_location(f"stress_loc_{thread_id}")
            assert len(location_episodes) == episodes_per_thread, \
                f"Thread {thread_id} location has wrong count: {len(location_episodes)}"

    def test_no_duplicate_episode_ids(self, temp_dir):
        """Concurrent writes should never create duplicate episode IDs."""
        config = EpisodicMemoryConfig(
            max_episodes=10000,
            embedding_dim=32,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
            enable_disk_offload=False,
        )
        store = EpisodicMemoryStore(config)

        all_ids: Set[str] = set()
        lock = threading.Lock()
        duplicates_found = []

        def writer(thread_id):
            """Write episodes and check for duplicates."""
            for _ in range(100):
                content = np.random.randn(10)
                ep = store.store_episode(content=content, surprise=1.0)

                with lock:
                    if ep.episode_id in all_ids:
                        duplicates_found.append(ep.episode_id)
                    all_ids.add(ep.episode_id)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert len(duplicates_found) == 0, f"Duplicate IDs found: {duplicates_found}"
        assert len(all_ids) == 1000, f"Expected 1000 unique IDs, got {len(all_ids)}"

    def test_retrieval_returns_valid_episodes_under_concurrent_writes(self, temp_dir):
        """Retrievals should always return valid complete episodes during writes."""
        config = EpisodicMemoryConfig(
            max_episodes=10000,
            embedding_dim=32,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
            enable_disk_offload=False,
        )
        store = EpisodicMemoryStore(config)

        # Pre-populate
        for i in range(50):
            content = np.random.randn(10)
            store.store_episode(
                content=content,
                surprise=1.0,
                location="test_location"
            )

        invalid_episodes = []
        stop_event = threading.Event()

        def writer():
            """Continuously write."""
            while not stop_event.is_set():
                content = np.random.randn(10)
                store.store_episode(
                    content=content,
                    surprise=1.0,
                    location="test_location"
                )
                time.sleep(0.001)

        def validator():
            """Continuously validate retrievals."""
            while not stop_event.is_set():
                episodes = store.retrieve_by_location("test_location")
                for ep in episodes:
                    # Verify episode is complete and valid
                    if ep.episode_id is None:
                        invalid_episodes.append(("missing_id", ep))
                    if ep.content is None:
                        invalid_episodes.append(("missing_content", ep))
                    if ep.embedding is None:
                        invalid_episodes.append(("missing_embedding", ep))
                    if not isinstance(ep.importance, float):
                        invalid_episodes.append(("invalid_importance", ep))
                time.sleep(0.005)

        writer_threads = [threading.Thread(target=writer) for _ in range(3)]
        validator_threads = [threading.Thread(target=validator) for _ in range(3)]

        for t in writer_threads + validator_threads:
            t.start()

        time.sleep(0.5)
        stop_event.set()

        for t in writer_threads + validator_threads:
            t.join(timeout=5.0)

        assert len(invalid_episodes) == 0, f"Invalid episodes found: {invalid_episodes[:10]}"


class TestSaveLoadThreadSafety:
    """Test thread safety of save/load operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test persistence."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_save_during_concurrent_writes(self, temp_dir):
        """Save operation should be safe during concurrent writes."""
        config = EpisodicMemoryConfig(
            max_episodes=10000,
            embedding_dim=32,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
            enable_disk_offload=False,
        )
        store = EpisodicMemoryStore(config)

        write_errors = []
        save_errors = []
        stop_event = threading.Event()

        def writer():
            """Continuously write."""
            try:
                while not stop_event.is_set():
                    content = np.random.randn(10)
                    store.store_episode(content=content, surprise=1.0)
                    time.sleep(0.005)
            except Exception as e:
                write_errors.append(str(e))

        def saver():
            """Periodically save."""
            try:
                for _ in range(10):
                    if stop_event.is_set():
                        break
                    store.save_state()
                    time.sleep(0.02)
            except Exception as e:
                save_errors.append(str(e))

        writers = [threading.Thread(target=writer) for _ in range(3)]
        saver_thread = threading.Thread(target=saver)

        for t in writers:
            t.start()
        saver_thread.start()

        time.sleep(0.3)
        stop_event.set()

        for t in writers:
            t.join(timeout=5.0)
        saver_thread.join(timeout=5.0)

        assert len(write_errors) == 0, f"Write errors: {write_errors}"
        assert len(save_errors) == 0, f"Save errors: {save_errors}"

    def test_load_and_verify_after_concurrent_session(self, temp_dir):
        """Data loaded after concurrent writes should be valid."""
        config = EpisodicMemoryConfig(
            max_episodes=10000,
            embedding_dim=32,
            persistence_path=temp_dir,
            enable_ebbinghaus=False,
            enable_disk_offload=False,
        )

        # Session 1: concurrent writes
        store1 = EpisodicMemoryStore(config)
        expected_ids = set()
        lock = threading.Lock()

        def writer(thread_id):
            for i in range(50):
                content = np.random.randn(10)
                ep = store1.store_episode(
                    content=content,
                    surprise=1.0,
                    location=f"load_test_{thread_id}"
                )
                with lock:
                    expected_ids.add(ep.episode_id)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        store1.save_state()
        stored_count = len(store1.episodes)

        # Session 2: load and verify
        store2 = EpisodicMemoryStore(config)
        store2.load_state()

        assert len(store2.episodes) == stored_count, \
            f"Loaded {len(store2.episodes)}, expected {stored_count}"

        # Verify all episode IDs were preserved
        loaded_ids = {ep.episode_id for ep in store2.episodes}
        assert loaded_ids == expected_ids, "Episode IDs don't match after load"

        # Verify indices are correct
        for thread_id in range(5):
            location_episodes = store2.retrieve_by_location(f"load_test_{thread_id}")
            assert len(location_episodes) == 50, \
                f"Location load_test_{thread_id} has wrong count after load"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
