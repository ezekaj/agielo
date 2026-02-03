"""
Tests for /curiosity command functionality.

Tests the integration of RND curiosity visualization into chat.py.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCuriosityCommandIntegration(unittest.TestCase):
    """Test /curiosity command integration components."""

    def test_rnd_curiosity_import_in_chat(self):
        """Test that RND curiosity can be imported in chat.py style."""
        try:
            from integrations.rnd_curiosity import RNDCuriosity, get_rnd_curiosity
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import RND curiosity module")

    def test_get_rnd_curiosity_singleton(self):
        """Test get_rnd_curiosity returns a singleton instance."""
        from integrations.rnd_curiosity import get_rnd_curiosity
        rnd1 = get_rnd_curiosity()
        rnd2 = get_rnd_curiosity()
        self.assertIs(rnd1, rnd2)

    def test_get_exploration_stats_format(self):
        """Test exploration stats returns expected keys."""
        from integrations.rnd_curiosity import RNDCuriosity
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            rnd = RNDCuriosity(storage_path=tmpdir)
            stats = rnd.get_exploration_stats()

            # Check expected keys
            expected_keys = [
                'total_explored',
                'avg_curiosity',
                'curiosity_trend',
                'high_curiosity_count',
                'low_curiosity_count',
                'running_mean',
                'running_std'
            ]
            for key in expected_keys:
                self.assertIn(key, stats)

    def test_get_curiosity_map_format(self):
        """Test curiosity map returns dict with float values."""
        from integrations.rnd_curiosity import RNDCuriosity
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            rnd = RNDCuriosity(storage_path=tmpdir)
            topics = ['machine learning', 'cooking', 'python']
            curiosity_map = rnd.get_curiosity_map(topics)

            self.assertIsInstance(curiosity_map, dict)
            self.assertEqual(set(curiosity_map.keys()), set(topics))
            for topic, score in curiosity_map.items():
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_get_most_curious_topics(self):
        """Test get_most_curious_topics returns correct format."""
        from integrations.rnd_curiosity import RNDCuriosity
        import tempfile
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            rnd = RNDCuriosity(storage_path=tmpdir)
            rnd.reset()

            # Record some curiosity with topics
            for i, topic in enumerate(['topic_a', 'topic_b', 'topic_c']):
                emb = np.random.randn(384).astype(np.float32)
                emb /= np.linalg.norm(emb)
                rnd.record_curiosity(emb, topic=topic)

            most_curious = rnd.get_most_curious_topics(k=3)
            self.assertIsInstance(most_curious, list)
            self.assertLessEqual(len(most_curious), 3)

            for item in most_curious:
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 2)
                topic, score = item
                self.assertIsInstance(topic, str)
                self.assertIsInstance(score, float)

    def test_get_novelty_hotspots(self):
        """Test get_novelty_hotspots returns sorted topics."""
        from integrations.rnd_curiosity import RNDCuriosity
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            rnd = RNDCuriosity(storage_path=tmpdir)
            topics = ['AI', 'blockchain', 'quantum computing']
            hotspots = rnd.get_novelty_hotspots(topics, k=3)

            self.assertIsInstance(hotspots, list)
            self.assertLessEqual(len(hotspots), 3)

            # Verify sorted by curiosity (descending)
            if len(hotspots) > 1:
                for i in range(len(hotspots) - 1):
                    self.assertGreaterEqual(hotspots[i][1], hotspots[i+1][1])

    def test_active_learner_rnd_stats_format(self):
        """Test active learner RND stats format."""
        from integrations.active_learning import ActiveLearner
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            learner = ActiveLearner(storage_path=tmpdir, use_rnd=True)
            stats = learner.get_rnd_stats()

            self.assertIsInstance(stats, dict)
            self.assertIn('rnd_enabled', stats)


class TestCuriosityVisualization(unittest.TestCase):
    """Test curiosity visualization helpers."""

    def test_curiosity_marker_logic(self):
        """Test the curiosity marker selection logic from /curiosity command."""
        def get_marker(curiosity):
            if curiosity > 0.7:
                return "🔥"
            elif curiosity > 0.5:
                return "🟢"
            else:
                return "🔵"

        self.assertEqual(get_marker(0.9), "🔥")
        self.assertEqual(get_marker(0.71), "🔥")
        self.assertEqual(get_marker(0.7), "🟢")
        self.assertEqual(get_marker(0.6), "🟢")
        self.assertEqual(get_marker(0.51), "🟢")
        self.assertEqual(get_marker(0.5), "🔵")
        self.assertEqual(get_marker(0.3), "🔵")
        self.assertEqual(get_marker(0.1), "🔵")


if __name__ == '__main__':
    unittest.main()
