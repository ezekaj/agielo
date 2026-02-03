"""
Tests for RND Curiosity Module
==============================

Tests for the Random Network Distillation curiosity system.
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSimpleMLPNetwork(unittest.TestCase):
    """Test the SimpleMLPNetwork class."""

    def setUp(self):
        """Set up test fixtures."""
        from integrations.rnd_curiosity import SimpleMLPNetwork
        self.network = SimpleMLPNetwork(
            input_dim=32,
            hidden_dim=16,
            output_dim=16,
            seed=42
        )

    def test_initialization(self):
        """Test network initializes correctly."""
        self.assertEqual(self.network.input_dim, 32)
        self.assertEqual(self.network.hidden_dim, 16)
        self.assertEqual(self.network.output_dim, 16)
        self.assertEqual(self.network.W1.shape, (32, 16))
        self.assertEqual(self.network.W2.shape, (16, 16))
        self.assertEqual(self.network.b1.shape, (16,))
        self.assertEqual(self.network.b2.shape, (16,))

    def test_forward_single(self):
        """Test forward pass with single input."""
        x = np.random.randn(32).astype(np.float32)
        out = self.network.forward(x)
        self.assertEqual(out.shape, (16,))

    def test_forward_batch(self):
        """Test forward pass with batch input."""
        x = np.random.randn(5, 32).astype(np.float32)
        out = self.network.forward(x)
        self.assertEqual(out.shape, (5, 16))

    def test_deterministic_output(self):
        """Test that same input gives same output."""
        x = np.random.randn(32).astype(np.float32)
        out1 = self.network.forward(x)
        out2 = self.network.forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_seeded_reproducibility(self):
        """Test that seeded networks are reproducible."""
        from integrations.rnd_curiosity import SimpleMLPNetwork
        net1 = SimpleMLPNetwork(input_dim=32, hidden_dim=16, output_dim=16, seed=123)
        net2 = SimpleMLPNetwork(input_dim=32, hidden_dim=16, output_dim=16, seed=123)

        x = np.random.randn(32).astype(np.float32)
        out1 = net1.forward(x)
        out2 = net2.forward(x)
        np.testing.assert_array_almost_equal(out1, out2)

    def test_forward_with_cache(self):
        """Test forward pass with cache for backprop."""
        x = np.random.randn(32).astype(np.float32)
        out, cache = self.network.forward_with_cache(x)
        self.assertEqual(out.shape, (16,))
        self.assertIn('x', cache)
        self.assertIn('z1', cache)
        self.assertIn('h', cache)

    def test_backward_updates_weights(self):
        """Test that backward pass updates weights."""
        x = np.random.randn(32).astype(np.float32)
        target = np.random.randn(16).astype(np.float32)

        old_W1 = self.network.W1.copy()
        old_W2 = self.network.W2.copy()

        out, cache = self.network.forward_with_cache(x)
        grad = 2 * (out - target) / 16
        self.network.backward(grad, cache, lr=0.1)

        # Weights should have changed
        self.assertFalse(np.allclose(old_W1, self.network.W1))
        self.assertFalse(np.allclose(old_W2, self.network.W2))

    def test_get_set_weights(self):
        """Test weight get/set operations."""
        weights = self.network.get_weights()
        self.assertIn('W1', weights)
        self.assertIn('W2', weights)
        self.assertIn('b1', weights)
        self.assertIn('b2', weights)

        # Modify and restore
        original_W1 = weights['W1'].copy()
        weights['W1'] = np.zeros_like(weights['W1'])
        self.network.set_weights(weights)
        np.testing.assert_array_equal(self.network.W1, np.zeros_like(original_W1))


class TestRNDCuriosity(unittest.TestCase):
    """Test the RNDCuriosity class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from integrations.rnd_curiosity import RNDCuriosity
        self.rnd = RNDCuriosity(
            input_dim=64,
            hidden_dim=32,
            storage_path=self.temp_dir,
            target_seed=42,
            learning_rate=0.01
        )
        self.rnd.reset()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test RND initializes correctly."""
        self.assertEqual(self.rnd.input_dim, 64)
        self.assertEqual(self.rnd.hidden_dim, 32)
        self.assertIsNotNone(self.rnd.target_network)
        self.assertIsNotNone(self.rnd.predictor_network)

    def test_compute_curiosity_returns_valid_score(self):
        """Test curiosity returns score in valid range."""
        embedding = np.random.randn(64).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        curiosity = self.rnd.compute_curiosity(embedding)
        self.assertGreaterEqual(curiosity, 0.0)
        self.assertLessEqual(curiosity, 1.0)

    def test_curiosity_decreases_after_training(self):
        """Test that curiosity decreases for seen embeddings."""
        embedding = np.random.randn(64).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        # Measure initial curiosity
        initial_curiosity = self.rnd.compute_curiosity(embedding)

        # Train predictor on this embedding
        for _ in range(100):
            self.rnd.update_predictor(embedding, n_steps=5)

        # Curiosity should decrease
        final_curiosity = self.rnd.compute_curiosity(embedding)
        self.assertLess(final_curiosity, initial_curiosity)

    def test_novel_embeddings_stay_curious(self):
        """Test that novel embeddings maintain high curiosity."""
        embedding1 = np.random.randn(64).astype(np.float32)
        embedding1 /= np.linalg.norm(embedding1)

        # Train on embedding1
        for _ in range(50):
            self.rnd.update_predictor(embedding1, n_steps=5)

        # embedding2 should still be curious
        embedding2 = np.random.randn(64).astype(np.float32)
        embedding2 /= np.linalg.norm(embedding2)

        curiosity1 = self.rnd.compute_curiosity(embedding1)
        curiosity2 = self.rnd.compute_curiosity(embedding2)

        # Novel embedding should have higher curiosity
        self.assertGreater(curiosity2, curiosity1)

    def test_embedding_dimension_mismatch_handling(self):
        """Test handling of mismatched embedding dimensions."""
        # Smaller embedding - should be padded
        small_emb = np.random.randn(32).astype(np.float32)
        curiosity1 = self.rnd.compute_curiosity(small_emb)
        self.assertGreaterEqual(curiosity1, 0.0)

        # Larger embedding - should be truncated
        large_emb = np.random.randn(128).astype(np.float32)
        curiosity2 = self.rnd.compute_curiosity(large_emb)
        self.assertGreaterEqual(curiosity2, 0.0)

    def test_compute_curiosity_batch(self):
        """Test batch curiosity computation."""
        embeddings = np.random.randn(5, 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        scores = self.rnd.compute_curiosity_batch(embeddings)
        self.assertEqual(len(scores), 5)
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_update_predictor_returns_loss(self):
        """Test that update_predictor returns loss value."""
        embedding = np.random.randn(64).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        loss = self.rnd.update_predictor(embedding, n_steps=1)
        self.assertGreaterEqual(loss, 0.0)

    def test_update_predictor_batch(self):
        """Test batch predictor update."""
        embeddings = np.random.randn(5, 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        avg_loss = self.rnd.update_predictor_batch(embeddings, n_steps=1)
        self.assertGreaterEqual(avg_loss, 0.0)

    def test_record_curiosity(self):
        """Test recording curiosity with optional topic."""
        embedding = np.random.randn(64).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        curiosity = self.rnd.record_curiosity(embedding, topic="test_topic")
        self.assertGreaterEqual(curiosity, 0.0)
        self.assertLessEqual(curiosity, 1.0)

        # Check history was updated
        self.assertGreater(len(self.rnd.curiosity_history), 0)
        self.assertEqual(self.rnd.curiosity_history[-1].topic, "test_topic")

    def test_record_curiosity_updates_predictor(self):
        """Test that record_curiosity updates predictor by default."""
        embedding = np.random.randn(64).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        initial_curiosity = self.rnd.compute_curiosity(embedding)

        # Record multiple times
        for _ in range(20):
            self.rnd.record_curiosity(embedding, topic="test")

        final_curiosity = self.rnd.compute_curiosity(embedding)

        # Curiosity should have decreased
        self.assertLess(final_curiosity, initial_curiosity)


class TestRNDCuriosityStats(unittest.TestCase):
    """Test RNDCuriosity statistics and analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from integrations.rnd_curiosity import RNDCuriosity
        self.rnd = RNDCuriosity(
            input_dim=64,
            hidden_dim=32,
            storage_path=self.temp_dir
        )
        self.rnd.reset()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_exploration_stats_empty(self):
        """Test stats with no history."""
        stats = self.rnd.get_exploration_stats()
        self.assertEqual(stats['total_explored'], 0)
        self.assertEqual(stats['avg_curiosity'], 0.5)

    def test_get_exploration_stats_with_data(self):
        """Test stats with recorded data."""
        # Record some curiosity measurements
        for i in range(20):
            emb = np.random.randn(64).astype(np.float32)
            emb /= np.linalg.norm(emb)
            self.rnd.record_curiosity(emb, topic=f"topic_{i % 5}")

        stats = self.rnd.get_exploration_stats()
        self.assertEqual(stats['total_explored'], 20)
        self.assertEqual(stats['unique_topics'], 5)
        self.assertGreater(stats['avg_curiosity'], 0.0)

    def test_get_most_curious_topics(self):
        """Test getting most curious topics."""
        # Record with different topics
        topics = ["novel_topic", "known_topic", "medium_topic"]

        # Train predictor heavily on "known_topic"
        for topic in topics:
            emb = np.random.randn(64).astype(np.float32)
            emb /= np.linalg.norm(emb)
            self.rnd.record_curiosity(emb, topic=topic, update_predictor=False)

        most_curious = self.rnd.get_most_curious_topics(k=3)
        self.assertLessEqual(len(most_curious), 3)

    def test_get_novelty_hotspots(self):
        """Test finding novelty hotspots."""
        topics = ["topic_a", "topic_b", "topic_c", "topic_d"]
        hotspots = self.rnd.get_novelty_hotspots(topics, k=2)

        self.assertEqual(len(hotspots), 2)
        for topic, score in hotspots:
            self.assertIn(topic, topics)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestRNDCuriosityPersistence(unittest.TestCase):
    """Test RNDCuriosity state persistence."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_state(self):
        """Test saving and loading state."""
        from integrations.rnd_curiosity import RNDCuriosity

        # Create and train
        rnd1 = RNDCuriosity(
            input_dim=64,
            hidden_dim=32,
            storage_path=self.temp_dir
        )
        rnd1.reset()

        # Record some curiosity
        embedding = np.random.randn(64).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        for _ in range(10):
            rnd1.record_curiosity(embedding, topic="test")

        # Save state
        rnd1._save_state()

        # Get curiosity before
        curiosity1 = rnd1.compute_curiosity(embedding)

        # Create new instance (should load state)
        rnd2 = RNDCuriosity(
            input_dim=64,
            hidden_dim=32,
            storage_path=self.temp_dir
        )

        # Curiosity should be similar (predictor was restored)
        curiosity2 = rnd2.compute_curiosity(embedding)

        # They should be close (not exact due to running stats)
        self.assertAlmostEqual(curiosity1, curiosity2, delta=0.1)

        # History should be restored
        self.assertGreater(len(rnd2.curiosity_history), 0)

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        from integrations.rnd_curiosity import RNDCuriosity

        rnd = RNDCuriosity(
            input_dim=64,
            hidden_dim=32,
            storage_path=self.temp_dir
        )

        # Record some data
        embedding = np.random.randn(64).astype(np.float32)
        for _ in range(5):
            rnd.record_curiosity(embedding, topic="test")

        self.assertGreater(len(rnd.curiosity_history), 0)

        # Reset
        rnd.reset()

        # Should be cleared
        self.assertEqual(len(rnd.curiosity_history), 0)
        self.assertEqual(rnd.update_count, 0)
        self.assertEqual(rnd.running_mean, 0.0)


class TestRNDCuriosityGlobalInstance(unittest.TestCase):
    """Test global instance functions."""

    def test_get_rnd_curiosity(self):
        """Test global instance getter."""
        from integrations.rnd_curiosity import get_rnd_curiosity

        rnd1 = get_rnd_curiosity()
        rnd2 = get_rnd_curiosity()

        # Should return same instance
        self.assertIs(rnd1, rnd2)

    def test_convenience_functions(self):
        """Test convenience functions."""
        from integrations.rnd_curiosity import compute_curiosity, record_curiosity

        embedding = np.random.randn(384).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        curiosity = compute_curiosity(embedding)
        self.assertGreaterEqual(curiosity, 0.0)
        self.assertLessEqual(curiosity, 1.0)

        recorded = record_curiosity(embedding, topic="convenience_test")
        self.assertGreaterEqual(recorded, 0.0)
        self.assertLessEqual(recorded, 1.0)


class TestRNDCuriosityNumericalStability(unittest.TestCase):
    """Test numerical stability of RND curiosity calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from integrations.rnd_curiosity import RNDCuriosity
        self.rnd = RNDCuriosity(
            input_dim=64,
            hidden_dim=32,
            storage_path=self.temp_dir
        )
        self.rnd.reset()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extreme_mse_values_no_overflow(self):
        """Test that extreme MSE values don't cause overflow in sigmoid."""
        # Create embeddings that produce very different outputs
        # to test the clipping mechanism at line 323

        # Test with embedding that might produce extreme normalized_curiosity
        embedding = np.ones(64, dtype=np.float32) * 1000  # Large values
        curiosity = self.rnd.compute_curiosity(embedding)

        # Should return valid score without overflow
        self.assertFalse(np.isnan(curiosity))
        self.assertFalse(np.isinf(curiosity))
        self.assertGreaterEqual(curiosity, 0.0)
        self.assertLessEqual(curiosity, 1.0)

    def test_very_small_embedding_no_overflow(self):
        """Test with very small embedding values."""
        embedding = np.ones(64, dtype=np.float32) * 1e-10
        curiosity = self.rnd.compute_curiosity(embedding)

        self.assertFalse(np.isnan(curiosity))
        self.assertFalse(np.isinf(curiosity))
        self.assertGreaterEqual(curiosity, 0.0)
        self.assertLessEqual(curiosity, 1.0)

    def test_zero_embedding_no_overflow(self):
        """Test with zero embedding."""
        embedding = np.zeros(64, dtype=np.float32)
        curiosity = self.rnd.compute_curiosity(embedding)

        self.assertFalse(np.isnan(curiosity))
        self.assertFalse(np.isinf(curiosity))
        self.assertGreaterEqual(curiosity, 0.0)
        self.assertLessEqual(curiosity, 1.0)

    def test_mixed_extreme_values(self):
        """Test with mixed extreme positive and negative values."""
        embedding = np.zeros(64, dtype=np.float32)
        embedding[:32] = 1e6
        embedding[32:] = -1e6

        curiosity = self.rnd.compute_curiosity(embedding)

        self.assertFalse(np.isnan(curiosity))
        self.assertFalse(np.isinf(curiosity))
        self.assertGreaterEqual(curiosity, 0.0)
        self.assertLessEqual(curiosity, 1.0)

    def test_high_curiosity_scale_no_overflow(self):
        """Test with high curiosity_scale parameter."""
        from integrations.rnd_curiosity import RNDCuriosity

        # Create with high scale that would cause overflow without clipping
        rnd_high_scale = RNDCuriosity(
            input_dim=64,
            hidden_dim=32,
            storage_path=self.temp_dir + "_high",
            curiosity_scale=1000.0  # Very high scale
        )

        embedding = np.random.randn(64).astype(np.float32)
        curiosity = rnd_high_scale.compute_curiosity(embedding)

        self.assertFalse(np.isnan(curiosity))
        self.assertFalse(np.isinf(curiosity))
        self.assertGreaterEqual(curiosity, 0.0)
        self.assertLessEqual(curiosity, 1.0)


class TestRNDCuriosityCuriosityMap(unittest.TestCase):
    """Test curiosity map functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        from integrations.rnd_curiosity import RNDCuriosity
        self.rnd = RNDCuriosity(
            input_dim=384,  # Match default embedding dim
            hidden_dim=128,
            storage_path=self.temp_dir
        )
        self.rnd.reset()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_curiosity_map(self):
        """Test getting curiosity map for topics."""
        topics = ["machine learning", "cooking", "quantum physics"]
        curiosity_map = self.rnd.get_curiosity_map(topics)

        self.assertEqual(len(curiosity_map), 3)
        for topic in topics:
            self.assertIn(topic, curiosity_map)
            self.assertGreaterEqual(curiosity_map[topic], 0.0)
            self.assertLessEqual(curiosity_map[topic], 1.0)

    def test_hash_curiosity_fallback(self):
        """Test hash-based fallback for curiosity."""
        curiosity = self.rnd._hash_curiosity("test topic")
        self.assertGreaterEqual(curiosity, 0.0)
        self.assertLessEqual(curiosity, 1.0)


if __name__ == '__main__':
    unittest.main()
