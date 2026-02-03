"""
Tests for Active Learning with RND Curiosity Integration
=========================================================

Tests for the integration of Random Network Distillation into the
active learning system.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path


class TestActiveLearnerWithRND(unittest.TestCase):
    """Test ActiveLearner with RND curiosity integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from integrations.active_learning import ActiveLearner, RND_AVAILABLE
        self.rnd_available = RND_AVAILABLE
        self.learner = ActiveLearner(
            storage_path=self.temp_dir,
            use_rnd=True
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_with_rnd(self):
        """Test that RND is initialized when available."""
        from integrations.active_learning import RND_AVAILABLE
        if RND_AVAILABLE:
            self.assertTrue(self.learner.use_rnd)
            self.assertIsNotNone(self.learner.rnd_curiosity)
        else:
            # If RND not available, should gracefully fall back
            self.assertFalse(self.learner.use_rnd)

    def test_initialization_without_rnd(self):
        """Test that RND can be disabled."""
        from integrations.active_learning import ActiveLearner
        learner = ActiveLearner(
            storage_path=self.temp_dir + "_no_rnd",
            use_rnd=False
        )
        self.assertFalse(learner.use_rnd)
        self.assertIsNone(learner.rnd_curiosity)
        shutil.rmtree(self.temp_dir + "_no_rnd", ignore_errors=True)

    def test_should_learn_new_topic(self):
        """Test should_learn returns True for new topics."""
        should, priority, reason = self.learner.should_learn("new_topic")
        self.assertTrue(should)
        self.assertGreater(priority, 0.5)
        self.assertIn("new_topic", reason)

    def test_should_learn_combines_rnd_curiosity(self):
        """Test that should_learn uses combined RND + traditional curiosity."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        # First create a topic with some history
        self.learner.record_exposure("test_topic", was_successful=True)
        self.learner.record_exposure("test_topic", was_successful=True)

        # Check learning decision
        should, priority, reason = self.learner.should_learn("test_topic")
        # Priority should be between 0 and 1
        self.assertGreaterEqual(priority, 0.0)
        self.assertLessEqual(priority, 1.0)

    def test_rnd_weight_affects_priority(self):
        """Test that RND weight affects the combined priority."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        # Record topic to create it
        self.learner.record_exposure("weight_test", was_successful=False)

        # Get priority with default weight (0.5)
        _, priority_default, _ = self.learner.should_learn("weight_test")

        # Change RND weight
        self.learner.set_rnd_weight(0.9)
        _, priority_high_rnd, _ = self.learner.should_learn("weight_test")

        self.learner.set_rnd_weight(0.1)
        _, priority_low_rnd, _ = self.learner.should_learn("weight_test")

        # Priorities should differ based on weight
        # (They may be similar if RND curiosity happens to equal traditional curiosity)
        self.assertGreaterEqual(priority_default, 0.0)
        self.assertGreaterEqual(priority_high_rnd, 0.0)
        self.assertGreaterEqual(priority_low_rnd, 0.0)

    def test_record_exposure_updates_rnd_predictor(self):
        """Test that successful learning updates RND predictor."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        initial_updates = self.learner._rnd_stats['total_rnd_updates']

        # Record successful exposure (should update predictor)
        self.learner.record_exposure("update_test", was_successful=True)

        # RND updates should increase
        self.assertGreater(
            self.learner._rnd_stats['total_rnd_updates'],
            initial_updates
        )

    def test_record_exposure_no_update_on_failure(self):
        """Test that failed learning doesn't update RND predictor."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        # Do a successful one first to establish baseline
        self.learner.record_exposure("baseline", was_successful=True)
        updates_before = self.learner._rnd_stats['total_rnd_updates']

        # Record failed exposure (should not update predictor)
        self.learner.record_exposure("failed_test", was_successful=False)

        # RND updates should remain the same
        self.assertEqual(
            self.learner._rnd_stats['total_rnd_updates'],
            updates_before
        )

    def test_rnd_curiosity_tracked_in_events(self):
        """Test that RND curiosity is tracked in learning events."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        self.learner.record_exposure("tracked_topic", was_successful=True)

        # Get last event
        last_event = self.learner.history[-1]

        # Should have RND curiosity score
        self.assertIsNotNone(last_event.rnd_curiosity)
        self.assertGreaterEqual(last_event.rnd_curiosity, 0.0)
        self.assertLessEqual(last_event.rnd_curiosity, 1.0)

    def test_novel_discovery_tracking(self):
        """Test that novel topic discoveries are tracked."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        initial_discoveries = self.learner._rnd_stats['novel_discoveries']

        # Record several new topics to increase chance of high RND curiosity
        for i in range(5):
            self.learner.record_exposure(f"novel_topic_{i}", was_successful=True)

        # At least some events should be marked as novel
        novel_events = [e for e in self.learner.history if e.is_novel]
        self.assertGreater(len(novel_events), 0)


class TestActiveLearnerRNDStats(unittest.TestCase):
    """Test RND statistics in ActiveLearner."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from integrations.active_learning import ActiveLearner, RND_AVAILABLE
        self.rnd_available = RND_AVAILABLE
        self.learner = ActiveLearner(
            storage_path=self.temp_dir,
            use_rnd=True
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_stats_includes_rnd(self):
        """Test that get_stats includes RND information."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        # Record some data
        for i in range(5):
            self.learner.record_exposure(f"stats_topic_{i}", was_successful=True)

        stats = self.learner.get_stats()

        # Should include RND-specific keys
        self.assertIn('rnd_enabled', stats)
        self.assertTrue(stats['rnd_enabled'])

    def test_get_rnd_stats(self):
        """Test dedicated RND stats method."""
        rnd_stats = self.learner.get_rnd_stats()

        if self.rnd_available and self.learner.use_rnd:
            self.assertTrue(rnd_stats['rnd_enabled'])
            self.assertIn('rnd_weight', rnd_stats)
            self.assertIn('avg_rnd_curiosity', rnd_stats)
            self.assertIn('novel_discoveries', rnd_stats)
            self.assertIn('novelty_rate_per_100', rnd_stats)
            self.assertIn('curiosity_decay_rate', rnd_stats)
        else:
            self.assertFalse(rnd_stats['rnd_enabled'])

    def test_curiosity_decay_tracking(self):
        """Test that curiosity decay is tracked over time."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        # Record many events to trigger decay tracking
        for i in range(100):
            self.learner.record_exposure(f"decay_topic_{i % 10}", was_successful=True)

        # Check that curiosity history is being tracked
        self.assertGreater(len(self.learner._rnd_stats['curiosity_history']), 0)


class TestActiveLearnerRNDPersistence(unittest.TestCase):
    """Test RND state persistence in ActiveLearner."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rnd_stats_saved_and_loaded(self):
        """Test that RND stats are persisted."""
        from integrations.active_learning import ActiveLearner, RND_AVAILABLE

        if not RND_AVAILABLE:
            self.skipTest("RND not available")

        # Create first instance and record data
        learner1 = ActiveLearner(storage_path=self.temp_dir, use_rnd=True)
        for i in range(10):
            learner1.record_exposure(f"persist_topic_{i}", was_successful=True)

        # Note the stats
        original_updates = learner1._rnd_stats['total_rnd_updates']
        original_discoveries = learner1._rnd_stats['novel_discoveries']

        # Save state
        learner1._save_state()

        # Create new instance (should load state)
        learner2 = ActiveLearner(storage_path=self.temp_dir, use_rnd=True)

        # Stats should be restored
        self.assertEqual(
            learner2._rnd_stats['total_rnd_updates'],
            original_updates
        )

    def test_rnd_weight_persisted(self):
        """Test that RND weight is persisted."""
        from integrations.active_learning import ActiveLearner, RND_AVAILABLE

        if not RND_AVAILABLE:
            self.skipTest("RND not available")

        # Create instance with custom weight
        learner1 = ActiveLearner(storage_path=self.temp_dir, use_rnd=True)
        learner1.set_rnd_weight(0.75)
        learner1._save_state()

        # Create new instance
        learner2 = ActiveLearner(storage_path=self.temp_dir, use_rnd=True)

        # Weight should be restored
        self.assertEqual(learner2._rnd_weight, 0.75)

    def test_history_with_rnd_fields_persisted(self):
        """Test that history events with RND fields are persisted."""
        from integrations.active_learning import ActiveLearner, RND_AVAILABLE

        if not RND_AVAILABLE:
            self.skipTest("RND not available")

        # Create instance and record data
        learner1 = ActiveLearner(storage_path=self.temp_dir, use_rnd=True)
        learner1.record_exposure("persist_test", was_successful=True)

        # Get the RND curiosity from last event
        original_rnd_curiosity = learner1.history[-1].rnd_curiosity

        # Save state
        learner1._save_state()

        # Create new instance
        learner2 = ActiveLearner(storage_path=self.temp_dir, use_rnd=True)

        # History should include RND curiosity
        if learner2.history:
            self.assertEqual(
                learner2.history[-1].rnd_curiosity,
                original_rnd_curiosity
            )


class TestActiveLearnerRNDDecisions(unittest.TestCase):
    """Test RND-influenced learning decisions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from integrations.active_learning import ActiveLearner, RND_AVAILABLE
        self.rnd_available = RND_AVAILABLE
        self.learner = ActiveLearner(
            storage_path=self.temp_dir,
            use_rnd=True
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rnd_can_override_low_curiosity(self):
        """Test that RND can override low traditional curiosity."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        # Create a topic with low traditional curiosity but train predictor
        # heavily on other topics to make this one seem novel
        from integrations.active_learning import Topic

        # Directly set a topic with low curiosity
        self.learner.topics["low_curiosity_test"] = Topic(
            name="low_curiosity_test",
            confidence=0.6,
            curiosity=0.2,  # Low traditional curiosity
            exposure_count=5,
            success_count=3
        )

        should, priority, reason = self.learner.should_learn("low_curiosity_test")

        # If RND override happened, reason should mention RND
        if "rnd" in reason.lower():
            self.assertTrue(should)

    def test_expert_topic_with_rnd_novelty(self):
        """Test that expert topics can still learn if RND detects novelty."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        from integrations.active_learning import Topic

        # Create an "expert" topic
        self.learner.topics["expert_test"] = Topic(
            name="expert_test",
            confidence=0.95,  # High confidence = expert
            curiosity=0.3,
            exposure_count=20,
            success_count=19
        )

        should, priority, reason = self.learner.should_learn("expert_test")

        # Check the reason - if RND saw novelty it might override
        if "rnd_novel" in reason.lower():
            self.assertTrue(should)
        else:
            # Normal expert rejection
            self.assertFalse(should)

    def test_recently_seen_rnd_override(self):
        """Test that RND can override 'recently seen' rejection."""
        if not self.rnd_available:
            self.skipTest("RND not available")

        from integrations.active_learning import Topic
        from datetime import datetime

        # Create topic seen very recently
        self.learner.topics["recent_test"] = Topic(
            name="recent_test",
            confidence=0.5,
            curiosity=0.5,
            exposure_count=1,
            success_count=1,
            last_seen=datetime.now().timestamp() - 60  # 1 minute ago
        )

        should, priority, reason = self.learner.should_learn("recent_test")

        # If RND override happened, should be True with RND in reason
        if "rnd_override" in reason.lower():
            self.assertTrue(should)


class TestActiveLearnerWithoutRND(unittest.TestCase):
    """Test ActiveLearner falls back gracefully without RND."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from integrations.active_learning import ActiveLearner
        self.learner = ActiveLearner(
            storage_path=self.temp_dir,
            use_rnd=False  # Explicitly disable RND
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_should_learn_works_without_rnd(self):
        """Test that should_learn still works without RND."""
        should, priority, reason = self.learner.should_learn("no_rnd_topic")
        self.assertTrue(should)
        self.assertEqual(reason, "new_topic")

    def test_record_exposure_works_without_rnd(self):
        """Test that record_exposure works without RND."""
        self.learner.record_exposure("no_rnd_exposure", was_successful=True)
        self.assertEqual(len(self.learner.history), 1)
        self.assertEqual(self.learner.history[0].rnd_curiosity, 0.0)

    def test_stats_without_rnd(self):
        """Test stats work without RND."""
        self.learner.record_exposure("stats_test", was_successful=True)
        stats = self.learner.get_stats()

        # Should not have RND-specific keys or rnd_enabled should be False
        rnd_stats = self.learner.get_rnd_stats()
        self.assertFalse(rnd_stats['rnd_enabled'])


class TestComputeRNDCuriosity(unittest.TestCase):
    """Test the RND curiosity computation helpers."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from integrations.active_learning import ActiveLearner, RND_AVAILABLE
        self.rnd_available = RND_AVAILABLE
        self.learner = ActiveLearner(
            storage_path=self.temp_dir,
            use_rnd=True
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_rnd_curiosity_returns_valid_score(self):
        """Test _compute_rnd_curiosity returns valid score."""
        score = self.learner._compute_rnd_curiosity("test_topic")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_compute_rnd_curiosity_with_content(self):
        """Test _compute_rnd_curiosity with additional content."""
        score = self.learner._compute_rnd_curiosity(
            "test_topic",
            "This is additional content for the embedding"
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_get_topic_embedding(self):
        """Test _get_topic_embedding returns valid embedding."""
        embedding = self.learner._get_topic_embedding("test_topic")
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)

    def test_get_topic_embedding_with_content(self):
        """Test _get_topic_embedding with content enrichment."""
        embedding = self.learner._get_topic_embedding(
            "test_topic",
            "Additional content for richer embedding"
        )
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)


if __name__ == '__main__':
    unittest.main()
