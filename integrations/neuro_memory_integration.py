"""
Neuro-Memory Integration
========================

Integrates the bio-inspired neuro_memory system into the cognitive chat.

Components integrated:
- BayesianSurpriseEngine: Detects novel/surprising information
- EpisodicMemoryStore: Human-like episodic memory with forgetting
- TwoStageRetriever: Hippocampal-style memory retrieval
- OnlineLearner: Continual learning without catastrophic forgetting

This transforms the AI from simple fact storage to human-like memory!
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
from pathlib import Path

# Add parent path for neuro_memory imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuro_memory import (
    BayesianSurpriseEngine,
    SurpriseConfig,
    EpisodicMemoryStore,
    EpisodicMemoryConfig,
    Episode,
    TwoStageRetriever,
    RetrievalConfig
)
from neuro_memory.online_learning import OnlineLearner, OnlineLearningConfig


class NeuroMemorySystem:
    """
    Bio-inspired memory system that mimics human cognition.

    Key behaviors:
    1. Only stores surprising/novel information (not everything)
    2. Memories fade over time (forgetting curve)
    3. Important memories are consolidated
    4. Retrieval mimics human recall (two-stage)
    5. Learns continuously without forgetting old knowledge
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        storage_path: str = None
    ):
        self.embedding_dim = embedding_dim
        self.storage_path = Path(storage_path or "~/.cognitive_ai_knowledge/neuro_memory").expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_surprise_engine()
        self._init_memory_store()
        self._init_retriever()
        self._init_online_learner()

        # Statistics
        self.stats = {
            'total_observations': 0,
            'surprising_events': 0,
            'memories_stored': 0,
            'memories_retrieved': 0,
            'consolidations': 0
        }

    def _init_surprise_engine(self):
        """Initialize Bayesian surprise detection."""
        config = SurpriseConfig(
            window_size=50,
            surprise_threshold=0.5,  # Lower = more selective
            use_adaptive_threshold=True,
            smoothing_alpha=0.1
        )
        self.surprise_engine = BayesianSurpriseEngine(
            input_dim=self.embedding_dim,
            config=config
        )

    def _init_memory_store(self):
        """Initialize episodic memory store."""
        config = EpisodicMemoryConfig(
            max_episodes=5000,
            embedding_dim=self.embedding_dim,
            enable_disk_offload=True,
            offload_threshold=4000,
            importance_decay=0.995,
            persistence_path=str(self.storage_path / "episodes")
        )
        self.memory_store = EpisodicMemoryStore(config)

        # Try to load existing state
        try:
            self.memory_store.load_state()
        except:
            pass

    def _init_retriever(self):
        """Initialize two-stage retriever."""
        config = RetrievalConfig(
            k_similarity=15,
            similarity_threshold=0.25,
            temporal_window=10,
            enable_temporal_expansion=True,
            similarity_weight=0.5,
            recency_weight=0.3,
            importance_weight=0.2,
            max_retrieved=5
        )
        self.retriever = TwoStageRetriever(
            memory_store=self.memory_store,
            config=config
        )

    def _init_online_learner(self):
        """Initialize continual learning system."""
        config = OnlineLearningConfig(
            learning_rate=0.01,
            ewc_lambda=0.5,
            replay_buffer_size=500,
            replay_batch_size=16,
            adaptive_threshold=True
        )
        self.online_learner = OnlineLearner(config)

    def text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.
        Uses a simple hash-based approach (replace with sentence transformers in production).
        """
        import hashlib

        # Create deterministic embedding from text
        h = hashlib.sha256(text.encode()).digest()
        # Expand to embedding_dim
        expanded = (h * ((self.embedding_dim // 32) + 1))[:self.embedding_dim]
        embedding = np.frombuffer(expanded, dtype=np.uint8).astype(np.float32)

        # Normalize to [-1, 1] range
        embedding = (embedding / 127.5) - 1.0

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def process_observation(
        self,
        content: str,
        source: str = "unknown",
        topic: str = None,
        entities: List[str] = None,
        location: str = None
    ) -> Dict:
        """
        Process a new observation through the neuro-memory pipeline.

        1. Convert to embedding
        2. Calculate surprise (is this novel?)
        3. If surprising, store in episodic memory
        4. Update online learner

        Returns:
            Dict with surprise info and whether memory was stored
        """
        self.stats['total_observations'] += 1

        # Convert to embedding
        embedding = self.text_to_embedding(content)

        # Calculate surprise
        surprise_info = self.surprise_engine.compute_surprise(embedding)

        result = {
            'surprise': surprise_info['surprise'],
            'is_novel': surprise_info['is_novel'],
            'threshold': surprise_info['threshold'],
            'stored': False,
            'episode_id': None
        }

        # Only store if surprising (human-like selective encoding)
        if surprise_info['is_novel']:
            self.stats['surprising_events'] += 1

            # Store in episodic memory
            episode = self.memory_store.store_episode(
                content=embedding,
                surprise=surprise_info['surprise'],
                timestamp=datetime.now(),
                location=location,
                entities=entities or [],
                metadata={
                    'text': content[:500],  # Store original text
                    'source': source,
                    'topic': topic or 'general'
                },
                embedding=embedding
            )

            result['stored'] = True
            result['episode_id'] = episode.episode_id
            self.stats['memories_stored'] += 1

        # Update online learner (for continual learning)
        self.online_learner.online_update(
            embedding,
            surprise_info['surprise']
        )

        return result

    def recall(
        self,
        query: str,
        k: int = 5,
        location: str = None,
        entities: List[str] = None
    ) -> List[Dict]:
        """
        Recall relevant memories using two-stage retrieval.

        Args:
            query: Text query
            k: Number of memories to retrieve
            location: Optional location filter
            entities: Optional entity filter

        Returns:
            List of recalled memories with metadata
        """
        # Convert query to embedding
        query_embedding = self.text_to_embedding(query)

        # Build filter criteria
        filter_criteria = {}
        if location:
            filter_criteria['location'] = location

        # Two-stage retrieval
        results = self.retriever.retrieve(
            query=query_embedding,
            query_time=datetime.now(),
            filter_criteria=filter_criteria if filter_criteria else None
        )

        self.stats['memories_retrieved'] += len(results)

        # Convert to readable format
        memories = []
        for episode, score in results[:k]:
            memory = {
                'text': episode.metadata.get('text', ''),
                'topic': episode.metadata.get('topic', 'unknown'),
                'source': episode.metadata.get('source', 'unknown'),
                'surprise': episode.surprise,
                'importance': episode.importance,
                'timestamp': episode.timestamp.isoformat(),
                'relevance_score': score
            }
            memories.append(memory)

        return memories

    def recall_by_context(
        self,
        location: str = None,
        entities: List[str] = None,
        time_description: str = None,
        k: int = 5
    ) -> List[Dict]:
        """
        Recall memories by contextual cues (like humans do).

        "Remember what happened at the cafe?" or "What did we discuss last week?"
        """
        memories = []

        if time_description:
            episodes = self.retriever.retrieve_by_temporal_cue(time_description, k)
        else:
            episodes = self.retriever.retrieve_by_contextual_cue(location, entities, k)

        for episode in episodes:
            memory = {
                'text': episode.metadata.get('text', ''),
                'topic': episode.metadata.get('topic', 'unknown'),
                'source': episode.metadata.get('source', 'unknown'),
                'surprise': episode.surprise,
                'importance': episode.importance,
                'timestamp': episode.timestamp.isoformat()
            }
            memories.append(memory)

        return memories

    def consolidate(self):
        """
        Trigger memory consolidation (like sleep in humans).

        - Decays unimportant memories
        - Offloads old memories to disk
        - Strengthens important connections
        """
        self.memory_store._consolidate_memory()
        self.stats['consolidations'] += 1

    def save(self):
        """Save all memory state to disk."""
        self.memory_store.save_state()

    def get_stats(self) -> Dict:
        """Get memory system statistics."""
        memory_stats = self.memory_store.get_statistics()
        learner_stats = self.online_learner.get_statistics()

        return {
            **self.stats,
            'memory': memory_stats,
            'learner': learner_stats,
            'surprise_threshold': self.surprise_engine.mean_surprise,
            'storage_path': str(self.storage_path)
        }

    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """Format recalled memories for injection into AI prompt."""
        if not memories:
            return ""

        output = "\n[Recalled from memory:]"
        for i, mem in enumerate(memories, 1):
            relevance = f"(relevance: {mem.get('relevance_score', 0):.2f})" if 'relevance_score' in mem else ""
            output += f"\n{i}. [{mem['topic']}] {mem['text'][:200]}... {relevance}"

        return output


# Integrated SelfTrainer with NeuroMemory
class NeuroSelfTrainer:
    """
    Enhanced SelfTrainer that uses bio-inspired neuro-memory.

    Replaces simple fact storage with:
    - Surprise-based selective encoding
    - Episodic memory with forgetting
    - Two-stage retrieval
    - Continual learning
    """

    def __init__(self, storage_path: str = None):
        self.neuro = NeuroMemorySystem(storage_path=storage_path)
        self.session_learning = []

    def learn(self, topic: str, content: str, source: str = "web") -> Dict:
        """
        Learn a new fact through the neuro-memory system.

        Unlike simple storage, this:
        1. Only stores if the information is surprising/novel
        2. Associates with topics, sources, timestamps
        3. Will naturally forget if unimportant
        """
        result = self.neuro.process_observation(
            content=content,
            source=source,
            topic=topic
        )

        if result['stored']:
            self.session_learning.append({
                'topic': topic,
                'content': content[:100],
                'surprise': result['surprise'],
                'time': datetime.now().isoformat()
            })

        return result

    def recall(self, query: str, k: int = 3) -> List[str]:
        """Recall relevant knowledge for a query."""
        memories = self.neuro.recall(query, k=k)
        return [m['text'] for m in memories]

    def get_knowledge_for_prompt(self, query: str) -> str:
        """Get relevant knowledge to inject into the prompt."""
        memories = self.neuro.recall(query, k=3)
        return self.neuro.format_memories_for_prompt(memories)

    def save(self):
        """Save all knowledge to disk."""
        self.neuro.save()

    def get_stats(self) -> Dict:
        """Get training statistics."""
        stats = self.neuro.get_stats()
        stats['session_learning'] = len(self.session_learning)
        return stats

    def get_session_summary(self) -> str:
        """Get summary of what was learned this session."""
        if not self.session_learning:
            return "No new surprising knowledge learned this session."

        stored = [s for s in self.session_learning if s.get('surprise', 0) > 0]

        summary = f"Learned {len(stored)} new surprising facts:\n"
        for item in stored[-5:]:
            summary += f"  • [{item['topic']}] {item['content']}... (surprise: {item.get('surprise', 0):.2f})\n"

        return summary


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("NEURO-MEMORY INTEGRATION TEST")
    print("=" * 60)

    # Test the neuro memory system
    neuro = NeuroMemorySystem()

    # Process some observations
    print("\nProcessing observations...")

    obs1 = neuro.process_observation(
        "Artificial intelligence is revolutionizing how we interact with computers",
        source="web",
        topic="AI"
    )
    print(f"Obs 1: surprise={obs1['surprise']:.3f}, stored={obs1['stored']}")

    obs2 = neuro.process_observation(
        "The hippocampus is crucial for forming new episodic memories",
        source="web",
        topic="neuroscience"
    )
    print(f"Obs 2: surprise={obs2['surprise']:.3f}, stored={obs2['stored']}")

    obs3 = neuro.process_observation(
        "Python is a programming language",  # Common knowledge, less surprising
        source="web",
        topic="programming"
    )
    print(f"Obs 3: surprise={obs3['surprise']:.3f}, stored={obs3['stored']}")

    # Test recall
    print("\nRecalling 'memory'...")
    memories = neuro.recall("memory and brain")
    for m in memories:
        print(f"  • [{m['topic']}] {m['text'][:60]}...")

    # Test NeuroSelfTrainer
    print("\n" + "=" * 60)
    print("NEURO SELF-TRAINER TEST")
    print("=" * 60)

    trainer = NeuroSelfTrainer()

    trainer.learn("quantum", "Quantum computers use qubits which can be in superposition", "article")
    trainer.learn("memory", "Working memory has limited capacity of about 7 items", "paper")
    trainer.learn("python", "Python uses indentation for code blocks", "docs")

    print(f"\nSession summary:\n{trainer.get_session_summary()}")

    print(f"\nStats: {trainer.get_stats()}")

    # Save
    trainer.save()

    print("\n" + "=" * 60)
    print("Integration complete!")
    print("=" * 60)
