"""
RND (Random Network Distillation) Curiosity Module
===================================================

Implements curiosity-driven exploration using Random Network Distillation.
Based on "Exploration by Random Network Distillation" (Burda et al., 2018).

Key Idea:
- Target network: Fixed random neural network
- Predictor network: Trained to match target outputs
- Curiosity = prediction error (MSE between target and predictor)
- High curiosity = novel/unknown area (predictor can't match target)

This provides intrinsic motivation for exploring unfamiliar topics,
complementing the active learning system's existing curiosity measures.
"""

import numpy as np
import json
import os
import atexit
import weakref
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import threading


# Default embedding dimension (matches semantic_embeddings.py)
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_HIDDEN_DIM = 128


@dataclass
class CuriosityRecord:
    """Record of a curiosity measurement."""
    embedding_hash: str
    curiosity_score: float
    timestamp: float
    topic: Optional[str] = None


# Track instances for atexit cleanup using weak references
_rnd_curiosity_instances: List[weakref.ref] = []


def _cleanup_all_instances():
    """Cleanup function called at program exit to save all RNDCuriosity state."""
    for ref in _rnd_curiosity_instances:
        instance = ref()
        if instance is not None:
            instance.cleanup()


# Register the cleanup function with atexit
atexit.register(_cleanup_all_instances)


def _register_instance(instance: 'RNDCuriosity'):
    """Register an RNDCuriosity instance for cleanup on exit."""
    _rnd_curiosity_instances.append(weakref.ref(instance))


class SimpleMLPNetwork:
    """
    Simple 2-layer MLP using numpy.

    Architecture:
        input -> Linear -> ReLU -> Linear -> output

    No backprop needed for target network (fixed weights).
    Predictor network is trained with simple gradient descent.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int = DEFAULT_HIDDEN_DIM,
        seed: Optional[int] = None
    ):
        """
        Initialize MLP with random weights.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Initialize weights using Xavier initialization
        # Layer 1: input -> hidden
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32)
        self.W1 *= np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        # Layer 2: hidden -> output
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32)
        self.W2 *= np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input embedding (shape: [input_dim] or [batch, input_dim])

        Returns:
            Output vector (shape: [output_dim] or [batch, output_dim])
        """
        # Ensure input is 2D
        single_input = x.ndim == 1
        if single_input:
            x = x.reshape(1, -1)

        # Layer 1 + ReLU
        h = np.maximum(0, x @ self.W1 + self.b1)

        # Layer 2 (no activation for output)
        out = h @ self.W2 + self.b2

        if single_input:
            out = out.squeeze(0)

        return out

    def forward_with_cache(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass with intermediate values cached for backprop.

        Args:
            x: Input embedding

        Returns:
            (output, cache) where cache contains intermediate values
        """
        single_input = x.ndim == 1
        if single_input:
            x = x.reshape(1, -1)

        # Layer 1 + ReLU
        z1 = x @ self.W1 + self.b1
        h = np.maximum(0, z1)

        # Layer 2
        out = h @ self.W2 + self.b2

        cache = {
            'x': x,
            'z1': z1,
            'h': h,
            'single_input': single_input
        }

        if single_input:
            out = out.squeeze(0)

        return out, cache

    def backward(self, grad_output: np.ndarray, cache: Dict, lr: float = 0.001) -> None:
        """
        Backward pass and weight update using simple SGD.

        Args:
            grad_output: Gradient of loss w.r.t. output
            cache: Cache from forward_with_cache
            lr: Learning rate
        """
        x = cache['x']
        z1 = cache['z1']
        h = cache['h']

        if cache['single_input']:
            grad_output = grad_output.reshape(1, -1)

        batch_size = x.shape[0]

        # Layer 2 gradients
        grad_W2 = h.T @ grad_output / batch_size
        grad_b2 = np.mean(grad_output, axis=0)

        # Backprop through layer 2
        grad_h = grad_output @ self.W2.T

        # ReLU backward
        grad_z1 = grad_h * (z1 > 0)

        # Layer 1 gradients
        grad_W1 = x.T @ grad_z1 / batch_size
        grad_b1 = np.mean(grad_z1, axis=0)

        # Update weights
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get network weights as dictionary."""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set network weights from dictionary."""
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W2 = weights['W2'].copy()
        self.b2 = weights['b2'].copy()


class RNDCuriosity:
    """
    Random Network Distillation for curiosity-driven exploration.

    The key insight: a fixed random network produces consistent outputs
    for any input. If a predictor network can match the target's output
    for an input, that input has been "seen" before. If the predictor
    fails to match, the input is novel.

    Curiosity = ||target(x) - predictor(x)||^2
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        storage_path: Optional[str] = None,
        target_seed: int = 42,
        learning_rate: float = 0.001,
        curiosity_scale: float = 1.0
    ):
        """
        Initialize RND curiosity system.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension for networks
            storage_path: Path to store state
            target_seed: Seed for target network (must be fixed)
            learning_rate: Learning rate for predictor training
            curiosity_scale: Scale factor for curiosity scores
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.curiosity_scale = curiosity_scale

        # Storage path
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(os.path.expanduser(
                "~/.cognitive_ai_knowledge/rnd_curiosity"
            ))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create networks
        # Target network: fixed random weights (never updated)
        self.target_network = SimpleMLPNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            seed=target_seed  # Fixed seed for reproducibility
        )

        # Predictor network: trained to match target
        self.predictor_network = SimpleMLPNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            seed=None  # Random initialization
        )

        # Statistics tracking
        self.curiosity_history: List[CuriosityRecord] = []
        self.running_mean = 0.0
        self.running_var = 1.0
        self.update_count = 0

        # Thread safety
        self._lock = threading.Lock()

        # Load saved state if exists
        self._load_state()

        # Register for cleanup on program exit
        _register_instance(self)

    def compute_curiosity(self, embedding: np.ndarray) -> float:
        """
        Compute curiosity score for an embedding.

        High curiosity = the predictor can't match the target = novel area

        Args:
            embedding: Input embedding (should be normalized)

        Returns:
            Curiosity score (0-1, normalized by running statistics)
        """
        with self._lock:
            # Ensure embedding is the right shape
            embedding = np.asarray(embedding, dtype=np.float32)
            if embedding.ndim == 1:
                if len(embedding) != self.input_dim:
                    # Resize if needed
                    if len(embedding) < self.input_dim:
                        padded = np.zeros(self.input_dim, dtype=np.float32)
                        padded[:len(embedding)] = embedding
                        embedding = padded
                    else:
                        embedding = embedding[:self.input_dim]

            # Get outputs from both networks
            target_output = self.target_network.forward(embedding)
            predictor_output = self.predictor_network.forward(embedding)

            # Compute MSE
            mse = np.mean((target_output - predictor_output) ** 2)

            # Update running statistics for normalization
            self.update_count += 1
            delta = mse - self.running_mean
            self.running_mean += delta / self.update_count
            if self.update_count > 1:
                delta2 = mse - self.running_mean
                self.running_var += (delta * delta2 - self.running_var) / self.update_count

            # Normalize using running statistics
            std = np.sqrt(self.running_var + 1e-8)
            normalized_curiosity = (mse - self.running_mean) / std

            # Scale to [0, 1] using sigmoid-like function
            # Clip to prevent overflow in exp() - exp(500) overflows, exp(-500) -> 0
            sigmoid_input = np.clip(-normalized_curiosity * self.curiosity_scale, -500, 500)
            curiosity_score = 1.0 / (1.0 + np.exp(sigmoid_input))

            return float(curiosity_score)

    def compute_curiosity_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute curiosity scores for multiple embeddings.

        Args:
            embeddings: Input embeddings (shape: [batch, input_dim])

        Returns:
            Array of curiosity scores
        """
        scores = []
        for emb in embeddings:
            scores.append(self.compute_curiosity(emb))
        return np.array(scores)

    def update_predictor(
        self,
        embedding: np.ndarray,
        n_steps: int = 1
    ) -> float:
        """
        Train the predictor network on a seen embedding.

        After seeing an embedding, the predictor should learn to match
        the target's output for it, reducing curiosity for similar inputs.

        Args:
            embedding: Input embedding that was "seen"
            n_steps: Number of gradient steps

        Returns:
            Final loss value
        """
        with self._lock:
            embedding = np.asarray(embedding, dtype=np.float32)
            if embedding.ndim == 1:
                if len(embedding) != self.input_dim:
                    if len(embedding) < self.input_dim:
                        padded = np.zeros(self.input_dim, dtype=np.float32)
                        padded[:len(embedding)] = embedding
                        embedding = padded
                    else:
                        embedding = embedding[:self.input_dim]

            # Get target output (fixed)
            target_output = self.target_network.forward(embedding)

            final_loss = 0.0
            for _ in range(n_steps):
                # Forward pass with cache
                predictor_output, cache = self.predictor_network.forward_with_cache(embedding)

                # Compute MSE loss
                loss = np.mean((target_output - predictor_output) ** 2)
                final_loss = loss

                # Compute gradient of MSE: 2 * (pred - target) / n
                grad = 2 * (predictor_output - target_output) / len(target_output)

                # Backward pass and update
                self.predictor_network.backward(grad, cache, lr=self.learning_rate)

            return float(final_loss)

    def update_predictor_batch(
        self,
        embeddings: np.ndarray,
        n_steps: int = 5
    ) -> float:
        """
        Train predictor on a batch of embeddings.

        Args:
            embeddings: Batch of embeddings (shape: [batch, input_dim])
            n_steps: Number of gradient steps

        Returns:
            Average final loss
        """
        total_loss = 0.0
        for emb in embeddings:
            total_loss += self.update_predictor(emb, n_steps=n_steps)
        return total_loss / len(embeddings)

    def record_curiosity(
        self,
        embedding: np.ndarray,
        topic: Optional[str] = None,
        update_predictor: bool = True
    ) -> float:
        """
        Record curiosity for an embedding and optionally learn from it.

        Args:
            embedding: Input embedding
            topic: Optional topic name for tracking
            update_predictor: Whether to train predictor on this embedding

        Returns:
            Curiosity score
        """
        # Compute curiosity before updating
        curiosity = self.compute_curiosity(embedding)

        # Record the measurement
        with self._lock:
            import hashlib
            emb_bytes = embedding.astype(np.float32).tobytes()
            emb_hash = hashlib.md5(emb_bytes).hexdigest()[:16]

            record = CuriosityRecord(
                embedding_hash=emb_hash,
                curiosity_score=curiosity,
                timestamp=datetime.now().timestamp(),
                topic=topic
            )
            self.curiosity_history.append(record)

            # Keep history bounded
            if len(self.curiosity_history) > 1000:
                self.curiosity_history = self.curiosity_history[-500:]

        # Update predictor (reduces curiosity for similar inputs)
        if update_predictor:
            self.update_predictor(embedding, n_steps=3)

        # Periodically save state
        if len(self.curiosity_history) % 50 == 0:
            self._save_state()

        return curiosity

    def get_curiosity_map(self, topics: List[str]) -> Dict[str, float]:
        """
        Get curiosity scores for a list of topics.

        Requires embedder to convert topics to embeddings.

        Args:
            topics: List of topic names

        Returns:
            Dictionary mapping topic -> curiosity score
        """
        # Import embedder lazily to avoid circular imports
        try:
            from integrations.semantic_embeddings import get_embedder
            embedder = get_embedder()
        except ImportError:
            # Fallback: use simple hash-based embeddings
            return {topic: self._hash_curiosity(topic) for topic in topics}

        curiosity_map = {}
        for topic in topics:
            embedding = embedder.embed(topic)
            curiosity_map[topic] = self.compute_curiosity(embedding)

        return curiosity_map

    def _hash_curiosity(self, text: str) -> float:
        """Fallback curiosity computation using hash-based embedding."""
        import hashlib
        # Create simple hash-based embedding
        embedding = np.zeros(self.input_dim, dtype=np.float32)
        text_hash = hashlib.sha256(text.encode()).digest()
        for i, byte in enumerate(text_hash):
            pos = i % self.input_dim
            embedding[pos] = byte / 255.0 - 0.5

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return self.compute_curiosity(embedding)

    def get_exploration_stats(self) -> Dict:
        """
        Get statistics about exploration.

        Returns:
            Dictionary with exploration statistics
        """
        with self._lock:
            if not self.curiosity_history:
                return {
                    'total_explored': 0,
                    'avg_curiosity': 0.5,
                    'curiosity_trend': 0.0,
                    'high_curiosity_count': 0,
                    'low_curiosity_count': 0,
                    'running_mean': self.running_mean,
                    'running_std': np.sqrt(self.running_var)
                }

            # Calculate statistics
            curiosities = [r.curiosity_score for r in self.curiosity_history]
            recent = curiosities[-50:] if len(curiosities) >= 50 else curiosities
            older = curiosities[:-50] if len(curiosities) > 50 else []

            # Trend: are we exploring more or less novel areas?
            trend = 0.0
            if older:
                trend = np.mean(recent) - np.mean(older)

            # Topic statistics
            topics = [r.topic for r in self.curiosity_history if r.topic]
            unique_topics = set(topics)

            # Time distribution
            timestamps = [r.timestamp for r in self.curiosity_history]
            if len(timestamps) >= 2:
                time_span = timestamps[-1] - timestamps[0]
                exploration_rate = len(timestamps) / max(1, time_span / 3600)  # per hour
            else:
                exploration_rate = 0.0

            return {
                'total_explored': len(self.curiosity_history),
                'unique_topics': len(unique_topics),
                'avg_curiosity': float(np.mean(curiosities)),
                'recent_avg_curiosity': float(np.mean(recent)),
                'curiosity_trend': float(trend),
                'high_curiosity_count': sum(1 for c in recent if c > 0.7),
                'low_curiosity_count': sum(1 for c in recent if c < 0.3),
                'running_mean': float(self.running_mean),
                'running_std': float(np.sqrt(self.running_var)),
                'exploration_rate_per_hour': float(exploration_rate)
            }

    def get_most_curious_topics(self, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get the topics with highest recorded curiosity.

        Args:
            k: Number of topics to return

        Returns:
            List of (topic, curiosity) tuples
        """
        with self._lock:
            topic_curiosities = {}
            for record in self.curiosity_history:
                if record.topic:
                    if record.topic not in topic_curiosities:
                        topic_curiosities[record.topic] = []
                    topic_curiosities[record.topic].append(record.curiosity_score)

            # Get average curiosity per topic
            avg_curiosities = [
                (topic, np.mean(scores))
                for topic, scores in topic_curiosities.items()
            ]

            # Sort by curiosity (descending)
            avg_curiosities.sort(key=lambda x: x[1], reverse=True)

            return avg_curiosities[:k]

    def get_novelty_hotspots(self, topics: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most novel (highest curiosity) topics from a list.

        Args:
            topics: List of candidate topics
            k: Number to return

        Returns:
            List of (topic, curiosity) tuples, sorted by curiosity
        """
        curiosity_map = self.get_curiosity_map(topics)
        sorted_topics = sorted(
            curiosity_map.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_topics[:k]

    def _save_state(self):
        """Save state to disk."""
        try:
            state = {
                'predictor_weights': {
                    k: v.tolist() for k, v in
                    self.predictor_network.get_weights().items()
                },
                'running_mean': float(self.running_mean),
                'running_var': float(self.running_var),
                'update_count': int(self.update_count),
                'history': [
                    {
                        'embedding_hash': r.embedding_hash,
                        'curiosity_score': float(r.curiosity_score),
                        'timestamp': float(r.timestamp),
                        'topic': r.topic
                    }
                    for r in self.curiosity_history[-200:]  # Keep last 200
                ],
                'input_dim': int(self.input_dim),
                'hidden_dim': int(self.hidden_dim),
                'learning_rate': float(self.learning_rate),
                'curiosity_scale': float(self.curiosity_scale)
            }

            with open(self.storage_path / 'state.json', 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            print(f"[RNDCuriosity] Save error: {e}")

    def _load_state(self):
        """Load state from disk."""
        state_file = self.storage_path / 'state.json'
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Load predictor weights
            if 'predictor_weights' in state:
                weights = {
                    k: np.array(v, dtype=np.float32)
                    for k, v in state['predictor_weights'].items()
                }
                self.predictor_network.set_weights(weights)

            # Load statistics
            self.running_mean = state.get('running_mean', 0.0)
            self.running_var = state.get('running_var', 1.0)
            self.update_count = state.get('update_count', 0)

            # Load history
            for data in state.get('history', []):
                self.curiosity_history.append(CuriosityRecord(
                    embedding_hash=data['embedding_hash'],
                    curiosity_score=data['curiosity_score'],
                    timestamp=data['timestamp'],
                    topic=data.get('topic')
                ))

        except Exception as e:
            print(f"[RNDCuriosity] Load error: {e}")

    def cleanup(self):
        """
        Cleanup resources and save state on exit.

        This method is registered with atexit to ensure state is saved
        when the program terminates. It saves:
        - Predictor network weights
        - Running statistics (mean, variance, update count)
        - Curiosity history
        """
        try:
            with self._lock:
                self._save_state()
                print(f"[RNDCuriosity] Cleanup complete: saved {len(self.curiosity_history)} history records, "
                      f"{self.update_count} updates tracked")
        except Exception as e:
            print(f"[RNDCuriosity] Cleanup error: {e}")

    def reset(self):
        """Reset the predictor network and statistics (for testing)."""
        with self._lock:
            self.predictor_network = SimpleMLPNetwork(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                seed=None
            )
            self.curiosity_history = []
            self.running_mean = 0.0
            self.running_var = 1.0
            self.update_count = 0


# Global instance
_rnd_curiosity: Optional[RNDCuriosity] = None


def get_rnd_curiosity(input_dim: int = DEFAULT_EMBEDDING_DIM) -> RNDCuriosity:
    """Get the global RND curiosity instance."""
    global _rnd_curiosity
    if _rnd_curiosity is None:
        _rnd_curiosity = RNDCuriosity(input_dim=input_dim)
    return _rnd_curiosity


# Convenience functions
def compute_curiosity(embedding: np.ndarray) -> float:
    """Compute curiosity for an embedding."""
    return get_rnd_curiosity().compute_curiosity(embedding)


def record_curiosity(embedding: np.ndarray, topic: str = None) -> float:
    """Record curiosity and update predictor."""
    return get_rnd_curiosity().record_curiosity(embedding, topic)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("RND CURIOSITY TEST")
    print("=" * 60)

    # Create a fresh instance for testing
    rnd = RNDCuriosity(
        input_dim=384,
        hidden_dim=128,
        storage_path="/tmp/test_rnd_curiosity"
    )
    rnd.reset()  # Start fresh

    print(f"\nNetwork dims: {rnd.input_dim} -> {rnd.hidden_dim}")

    # Test 1: Novel embeddings should have high curiosity
    print("\n--- Test 1: Novel Embeddings ---")
    novel_embedding = np.random.randn(384).astype(np.float32)
    novel_embedding /= np.linalg.norm(novel_embedding)

    curiosity1 = rnd.compute_curiosity(novel_embedding)
    print(f"First exposure curiosity: {curiosity1:.3f}")

    # Test 2: After training, curiosity should decrease
    print("\n--- Test 2: Learning Reduces Curiosity ---")
    rnd.update_predictor(novel_embedding, n_steps=50)
    curiosity2 = rnd.compute_curiosity(novel_embedding)
    print(f"After training curiosity: {curiosity2:.3f}")
    print(f"Curiosity decreased: {curiosity1 > curiosity2}")

    # Test 3: Different embeddings should still have high curiosity
    print("\n--- Test 3: Different Embeddings Stay Curious ---")
    different_embedding = np.random.randn(384).astype(np.float32)
    different_embedding /= np.linalg.norm(different_embedding)

    curiosity3 = rnd.compute_curiosity(different_embedding)
    print(f"Different embedding curiosity: {curiosity3:.3f}")
    print(f"Still curious about different: {curiosity3 > curiosity2}")

    # Test 4: Curiosity map
    print("\n--- Test 4: Curiosity Map ---")
    topics = ["machine learning", "cooking recipes", "quantum physics", "gardening"]
    curiosity_map = rnd.get_curiosity_map(topics)
    for topic, score in sorted(curiosity_map.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {score:.3f}")

    # Test 5: Exploration stats
    print("\n--- Test 5: Exploration Stats ---")
    # Record some curiosity measurements
    for i in range(10):
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        rnd.record_curiosity(emb, topic=f"topic_{i}")

    stats = rnd.get_exploration_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
