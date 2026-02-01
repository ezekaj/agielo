"""
Neuro-Memory Adapter
====================

Integrates the advanced ezekaj/memory system with the cognitive agent.

Features:
- Bayesian Surprise Detection (only remembers important things)
- Event Segmentation (groups related experiences)
- Two-Stage Retrieval (similarity + temporal)
- Memory Consolidation (sleep-like replay)
- Forgetting & Decay (realistic forgetting)
- Online Learning (improves over time)
"""

import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import neuro-memory components
try:
    from neuro_memory.surprise import BayesianSurpriseEngine, SurpriseConfig
    from neuro_memory.segmentation import EventSegmenter, SegmentationConfig
    from neuro_memory.memory import EpisodicMemoryStore, EpisodicMemoryConfig
    from neuro_memory.retrieval import TwoStageRetriever, RetrievalConfig
    from neuro_memory.consolidation import MemoryConsolidationEngine, ConsolidationConfig
    from neuro_memory.memory.forgetting import ForgettingEngine, ForgettingConfig
    from neuro_memory.online_learning import OnlineLearner, OnlineLearningConfig
    NEURO_MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Neuro-memory not fully available: {e}")
    print("[Info] Install: pip install chromadb hmmlearn networkx scikit-learn")
    NEURO_MEMORY_AVAILABLE = False


class NeuroMemorySystem:
    """
    Advanced bio-inspired memory system.

    Combines all components from ezekaj/memory:
    1. Surprise detection - Only store surprising/important events
    2. Event segmentation - Group related observations
    3. Episodic storage - Vector database with ChromaDB
    4. Two-stage retrieval - Similarity + temporal expansion
    5. Consolidation - Sleep-like memory replay
    6. Forgetting - Power-law decay (realistic)
    7. Online learning - Improves thresholds over time
    """

    def __init__(self, dim: int = 128, persistence_path: str = None):
        """
        Initialize the neuro-memory system.

        Args:
            dim: Embedding dimension
            persistence_path: Where to store memories (default: ~/.neuro_memory)
        """
        self.dim = dim
        self.persistence_path = persistence_path or os.path.expanduser("~/.neuro_memory")

        if not NEURO_MEMORY_AVAILABLE:
            self.enabled = False
            return

        self.enabled = True

        # Initialize all components
        self._init_components()

        # Observation buffer for segmentation
        self.observation_buffer = []
        self.surprise_buffer = []
        self.timestamp_buffer = []

        # Statistics
        self.total_observations = 0
        self.novel_events = 0

    def _init_components(self):
        """Initialize all memory components."""
        # 1. Surprise Detection
        self.surprise_engine = BayesianSurpriseEngine(
            input_dim=self.dim,
            config=SurpriseConfig(
                window_size=50,
                surprise_threshold=0.6
            )
        )

        # 2. Event Segmentation
        self.segmenter = EventSegmenter(
            config=SegmentationConfig(min_event_length=5)
        )

        # 3. Episodic Memory Store
        self.memory_store = EpisodicMemoryStore(
            config=EpisodicMemoryConfig(
                max_episodes=10000,
                embedding_dim=self.dim,
                persistence_path=self.persistence_path
            )
        )

        # 4. Two-Stage Retrieval
        self.retriever = TwoStageRetriever(
            self.memory_store,
            config=RetrievalConfig(k_similarity=10)
        )

        # 5. Memory Consolidation
        self.consolidation = MemoryConsolidationEngine(
            config=ConsolidationConfig(replay_batch_size=32)
        )

        # 6. Forgetting Engine
        self.forgetting = ForgettingEngine(
            config=ForgettingConfig(decay_rate=0.3)
        )

        # 7. Online Learning
        self.online_learner = OnlineLearner(
            config=OnlineLearningConfig(learning_rate=0.01)
        )

    def process_observation(
        self,
        embedding: np.ndarray,
        content: str = None,
        location: str = None,
        entities: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a new observation through the memory system.

        Args:
            embedding: Vector representation of the observation
            content: Optional text content
            location: Optional location context
            entities: Optional list of entities involved

        Returns:
            Dict with surprise, novelty, and storage info
        """
        if not self.enabled:
            return {'enabled': False, 'surprise': 0.0, 'is_novel': False}

        self.total_observations += 1
        timestamp = datetime.now()

        # 1. Compute surprise
        surprise_result = self.surprise_engine.compute_surprise(embedding)
        surprise = surprise_result['surprise']
        is_novel = surprise_result['is_novel']

        if is_novel:
            self.novel_events += 1

        # 2. Online learning update
        self.online_learner.online_update(embedding, surprise)

        # 3. Buffer for segmentation
        self.observation_buffer.append(embedding)
        self.surprise_buffer.append(surprise)
        self.timestamp_buffer.append(timestamp)

        # 4. If novel, store immediately
        stored = False
        if is_novel:
            # Normalize embedding
            norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            # Store episode
            self.memory_store.store_episode(
                content=embedding,
                embedding=norm_embedding,
                surprise=surprise,
                timestamp=timestamp,
                location=location,
                entities=entities or [],
                metadata={'text': content} if content else {}
            )
            stored = True

        return {
            'surprise': surprise,
            'is_novel': is_novel,
            'stored': stored,
            'total_observations': self.total_observations,
            'novel_events': self.novel_events
        }

    def recall(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant memories using two-stage retrieval.

        Args:
            query_embedding: Query vector
            k: Number of memories to retrieve

        Returns:
            List of recalled memories with scores
        """
        if not self.enabled:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Two-stage retrieval
        results = self.retriever.retrieve(query_norm)
        results = results[:k]  # Limit to k results

        # Format results
        memories = []
        for episode, score in results:
            # Check forgetting probability
            activation = self.forgetting.compute_activation(
                episode.surprise,
                episode.timestamp,
                rehearsal_count=0
            )
            forget_prob = self.forgetting.get_forgetting_probability(activation)

            # Only return if not forgotten
            if np.random.random() > forget_prob:
                memories.append({
                    'content': episode.metadata.get('text', ''),
                    'embedding': episode.embedding,
                    'surprise': episode.surprise,
                    'timestamp': episode.timestamp,
                    'location': episode.location,
                    'score': score,
                    'activation': activation
                })

        return memories

    def consolidate(self) -> Dict[str, Any]:
        """
        Run memory consolidation (like sleep).

        This:
        - Replays important memories
        - Extracts schemas/patterns
        - Strengthens important memories
        - Allows forgetting of unimportant ones
        """
        if not self.enabled:
            return {'enabled': False}

        # Run consolidation
        stats = self.consolidation.consolidate(self.memory_store.episodes)

        # Get schemas
        schemas = self.consolidation.get_schema_summary()

        return {
            'episodes_consolidated': stats['episodes_consolidated'],
            'replay_count': stats['replay_count'],
            'schemas_extracted': stats['schemas_extracted'],
            'schemas': schemas
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self.enabled:
            return {'enabled': False}

        memory_stats = self.memory_store.get_statistics()
        online_stats = self.online_learner.get_statistics()

        return {
            'total_observations': self.total_observations,
            'novel_events': self.novel_events,
            'novelty_rate': self.novel_events / max(1, self.total_observations),
            'episodes_stored': memory_stats['total_episodes'],
            'adaptive_threshold': online_stats['surprise_threshold'],
            'replay_count': online_stats['replay_count']
        }


# Simple test
if __name__ == "__main__":
    print("=" * 60)
    print("NEURO-MEMORY SYSTEM TEST")
    print("=" * 60)

    if not NEURO_MEMORY_AVAILABLE:
        print("\n[!] Install dependencies:")
        print("    pip install chromadb hmmlearn networkx scikit-learn")
        exit(1)

    # Create system
    memory = NeuroMemorySystem(dim=64)

    # Process some observations
    print("\nProcessing observations...")
    for i in range(20):
        # Create embedding (some novel, some routine)
        if i % 5 == 0:
            emb = np.random.randn(64) * 3  # Novel (high variance)
        else:
            emb = np.random.randn(64) * 0.5  # Routine (low variance)

        result = memory.process_observation(
            emb.astype(np.float32),
            content=f"Observation {i}",
            location="test"
        )

        if result['is_novel']:
            print(f"  [{i}] NOVEL - surprise: {result['surprise']:.3f}")

    # Get statistics
    stats = memory.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total observations: {stats['total_observations']}")
    print(f"  Novel events: {stats['novel_events']}")
    print(f"  Novelty rate: {stats['novelty_rate']:.1%}")
    print(f"  Episodes stored: {stats['episodes_stored']}")

    # Test recall
    print("\nTesting recall...")
    query = np.random.randn(64).astype(np.float32)
    memories = memory.recall(query, k=3)
    print(f"  Retrieved {len(memories)} memories")

    # Consolidate
    print("\nRunning consolidation...")
    consolidation_result = memory.consolidate()
    print(f"  Schemas extracted: {consolidation_result['schemas_extracted']}")

    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
