"""
Episodic Memory Store
====================

Human-inspired episodic memory storage with temporal-spatial grounding.
Implements hybrid storage: vector database + disk offloading for massive contexts.

References:
- EM-LLM (ICLR 2025): Infinite context via episodic memory
- Tulving (1983): Episodic memory definition
- A-MEM (NeurIPS 2025): Agentic memory organization

Key Features:
1. Fast episodic encoding (single-shot learning)
2. Temporal-spatial indexing for contextual retrieval
3. Vector similarity search (ChromaDB/FAISS)
4. Disk offloading for contexts > 10M tokens
5. Automatic memory consolidation and pruning
"""

import gc
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
import threading
import time
from pathlib import Path
import chromadb
from chromadb.config import Settings

# Import Ebbinghaus forgetting and spaced repetition
from neuro_memory.memory.forgetting import (
    EbbinghausForgetting,
    EbbinghausConfig,
    SpacedRepetitionScheduler,
    SpacedRepetitionConfig
)

# Import numerical stability utilities for validation
from utils.numerical import validate_finite


@dataclass
class Episode:
    """
    Single episodic memory representing a specific event.
    
    Attributes:
        content: The actual observation/experience
        timestamp: When the episode occurred
        location: Spatial context (optional)
        entities: Who/what was involved
        embedding: Vector representation for similarity search
        surprise: How novel/unexpected this episode was
        importance: Consolidation priority score
        metadata: Additional contextual information
    """
    content: np.ndarray
    timestamp: datetime
    location: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    surprise: float = 0.0
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    episode_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate episode ID if not provided."""
        if self.episode_id is None:
            self.episode_id = f"ep_{self.timestamp.timestamp()}_{id(self)}"
    
    def to_dict(self) -> Dict:
        """Convert episode to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "content": self.content.tolist() if isinstance(self.content, np.ndarray) else self.content,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "entities": self.entities,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "surprise": self.surprise,
            "importance": self.importance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Episode':
        """Reconstruct episode from dictionary."""
        # Keep string content as string, only convert arrays
        content = data["content"]
        if isinstance(content, list):
            content = np.array(content)
        return cls(
            content=content,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=data.get("location"),
            entities=data.get("entities", []),
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
            surprise=data.get("surprise", 0.0),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
            episode_id=data.get("episode_id")
        )


@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory storage."""
    max_episodes: int = 10000  # Maximum episodes in hot storage
    embedding_dim: int = 512  # Dimension of episode embeddings
    enable_disk_offload: bool = True  # Offload old episodes to disk
    offload_threshold: int = 8000  # Start offloading when this many episodes
    importance_decay: float = 0.99  # Decay factor for episode importance
    consolidation_interval: int = 100  # Consolidate every N new episodes
    vector_db_backend: str = "chromadb"  # "chromadb" or "faiss"
    persistence_path: Optional[str] = "./memory_store"  # Path for persistent storage
    # Ebbinghaus forgetting settings
    enable_ebbinghaus: bool = True  # Enable Ebbinghaus forgetting model
    forgetting_background_interval: float = 3600.0  # Run forgetting task every hour (seconds)
    review_threshold: float = 0.3  # Retention threshold for review
    auto_reinforce_high_value: bool = True  # Auto-reinforce high-importance memories


class EpisodicMemoryStore:
    """
    Episodic memory storage with efficient retrieval and consolidation.
    
    Mimics hippocampal episodic memory:
    - Fast single-shot encoding
    - Temporal-spatial indexing
    - Importance-based consolidation
    - Automatic capacity management
    """
    
    def __init__(self, config: Optional[EpisodicMemoryConfig] = None):
        """
        Args:
            config: Memory configuration
        """
        self.config = config or EpisodicMemoryConfig()

        # In-memory episode storage (hot storage)
        self.episodes: List[Episode] = []

        # Thread lock for episode access - prevents race conditions
        # when background forgetting thread modifies episodes
        self._episodes_lock = threading.RLock()

        # Temporal index: timestamp -> episode_ids
        self.temporal_index: Dict[str, List[str]] = {}

        # Spatial index: location -> episode_ids
        self.spatial_index: Dict[str, List[str]] = {}

        # Entity index: entity -> episode_ids
        self.entity_index: Dict[str, List[str]] = {}

        # Statistics
        self.total_episodes_stored = 0
        self.episodes_offloaded = 0

        # Ebbinghaus forgetting tracking statistics
        self.forgotten_memories_count = 0
        self.reviewed_memories_count = 0
        self.reinforced_memories_count = 0

        # Initialize vector database
        self._initialize_vector_db()

        # Setup persistence
        if self.config.persistence_path:
            self.persistence_path = Path(self.config.persistence_path)
            self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Initialize Ebbinghaus forgetting system
        self.ebbinghaus: Optional[EbbinghausForgetting] = None
        self.spaced_repetition: Optional[SpacedRepetitionScheduler] = None
        self._forgetting_thread: Optional[threading.Thread] = None
        self._forgetting_running = False

        if self.config.enable_ebbinghaus:
            self._initialize_ebbinghaus()
        
    def _initialize_vector_db(self):
        """Initialize vector database for similarity search."""
        backend = self.config.vector_db_backend

        # Validate backend selection - FAISS not yet implemented
        if backend == "faiss":
            import warnings
            warnings.warn(
                "FAISS backend is not yet implemented. Falling back to ChromaDB. "
                "To use FAISS in the future, update the episodic_store.py implementation.",
                UserWarning
            )
            backend = "chromadb"
            self.config.vector_db_backend = "chromadb"  # Update config to reflect actual backend

        if backend == "chromadb":
            # Initialize ChromaDB with proper persistent client
            if self.config.persistence_path:
                chroma_path = Path(self.config.persistence_path) / "chroma"
                chroma_path.mkdir(parents=True, exist_ok=True)
                # Use PersistentClient for disk-backed storage
                self.chroma_client = chromadb.PersistentClient(
                    path=str(chroma_path)
                )
            else:
                # Use EphemeralClient for in-memory only
                self.chroma_client = chromadb.EphemeralClient()

            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="episodic_memories",
                metadata={"description": "Human-inspired episodic memory storage"}
            )
        else:
            # Unknown backend
            raise ValueError(
                f"Unknown vector_db_backend: '{backend}'. "
                f"Supported backends: 'chromadb'. FAISS support planned for future release."
            )

    def _initialize_ebbinghaus(self):
        """Initialize Ebbinghaus forgetting and spaced repetition systems."""
        if not self.config.persistence_path:
            # In-memory only
            self.ebbinghaus = EbbinghausForgetting(
                config=EbbinghausConfig(forget_threshold=self.config.review_threshold)
            )
            self.spaced_repetition = SpacedRepetitionScheduler(
                ebbinghaus=self.ebbinghaus
            )
        else:
            # Persistent storage
            ebbinghaus_path = self.persistence_path / "ebbinghaus_state.json"
            sr_path = self.persistence_path / "spaced_repetition_state.json"

            self.ebbinghaus = EbbinghausForgetting(
                config=EbbinghausConfig(forget_threshold=self.config.review_threshold),
                state_path=ebbinghaus_path
            )
            self.spaced_repetition = SpacedRepetitionScheduler(
                ebbinghaus=self.ebbinghaus,
                state_path=sr_path
            )

    def start_forgetting_background_task(self):
        """
        Start background task to process forgetting every hour.

        This task:
        1. Identifies memories below retention threshold
        2. Auto-reinforces high-value memories
        3. Removes truly forgotten memories from active storage
        """
        if self._forgetting_running:
            return  # Already running

        self._forgetting_running = True
        self._forgetting_thread = threading.Thread(
            target=self._forgetting_loop,
            daemon=True,
            name="EbbinghausForgettingTask"
        )
        self._forgetting_thread.start()

    def stop_forgetting_background_task(self):
        """Stop the background forgetting task."""
        self._forgetting_running = False
        if self._forgetting_thread and self._forgetting_thread.is_alive():
            self._forgetting_thread.join(timeout=5.0)

    def _forgetting_loop(self):
        """Background loop for forgetting processing."""
        while self._forgetting_running:
            try:
                self._process_forgetting()
            except Exception as e:
                print(f"[EbbinghausForgetting] Error in forgetting loop: {e}")

            # Sleep for the configured interval
            time.sleep(self.config.forgetting_background_interval)

    def _process_forgetting(self):
        """
        Process memories for forgetting:
        1. Check retention levels
        2. Auto-reinforce high-importance memories (frequently accessed, highly linked)
        3. Remove memories that should be truly forgotten

        Thread Safety:
            Uses self._episodes_lock when reading and modifying the episodes list.
            The lock is held during the entire modification operation to prevent
            race conditions with other threads accessing or modifying episodes.
        """
        if not self.ebbinghaus:
            return

        now = datetime.now().timestamp()

        # Get memories below retention threshold
        at_risk = self.ebbinghaus.get_memories_below_threshold(current_time=now)

        # Collect episodes to process - hold lock during entire operation
        # to ensure consistency between reading and modifying
        episodes_to_forget = []
        episodes_to_reinforce = []

        with self._episodes_lock:
            for memory_id, retention in at_risk:
                # Find episode in list (direct lookup, lock already held)
                episode = None
                for ep in self.episodes:
                    if ep.episode_id == memory_id:
                        episode = ep
                        break

                if episode is None:
                    continue

                # Decide: reinforce or forget based on importance
                should_reinforce = self._should_reinforce_memory(episode)

                if should_reinforce:
                    episodes_to_reinforce.append((memory_id, episode))
                else:
                    episodes_to_forget.append((memory_id, episode))

            # Remove forgotten episodes from list while still holding lock
            if episodes_to_forget:
                forget_ids = {memory_id for memory_id, _ in episodes_to_forget}
                self.episodes = [ep for ep in self.episodes if ep.episode_id not in forget_ids]

        # Process reinforcements (outside lock - no list modification)
        for memory_id, episode in episodes_to_reinforce:
            self.ebbinghaus.record_retrieval(memory_id, success=True, current_time=now)
            if self.spaced_repetition:
                self.spaced_repetition.record_review(memory_id, success=True, current_time=now)
            self.reinforced_memories_count += 1

        # Offload forgotten episodes to disk (outside lock - I/O shouldn't hold lock)
        if episodes_to_forget:
            forgotten_ids = []

            for memory_id, episode in episodes_to_forget:
                self.forgotten_memories_count += 1

                # Remove from all indices to prevent memory leaks
                self._remove_from_indices(episode)
                forgotten_ids.append(memory_id)

                if self.config.enable_disk_offload:
                    self._offload_episode(episode)

            # Remove from vector database in batch
            self._remove_from_vector_db(forgotten_ids)

            # Trigger garbage collection after bulk removal
            gc.collect()

    def _should_reinforce_memory(self, episode: Episode) -> bool:
        """
        Determine if a memory should be auto-reinforced.

        High-value memories are:
        - High importance score (from surprise)
        - Linked to many entities
        - Frequently accessed (stored in metadata)
        """
        if not self.config.auto_reinforce_high_value:
            return False

        # High importance threshold
        if episode.importance > 0.6:
            return True

        # Linked to many entities (social/contextual importance)
        if len(episode.entities) >= 3:
            return True

        # Check access count in metadata
        access_count = episode.metadata.get('access_count', 0)
        if access_count >= 5:
            return True

        return False

    def record_retrieval(self, episode_id: str, success: bool = True):
        """
        Record a memory retrieval for spaced repetition.

        Args:
            episode_id: ID of the retrieved episode
            success: Whether the retrieval was successful
        """
        now = datetime.now().timestamp()

        # Update Ebbinghaus forgetting
        if self.ebbinghaus:
            self.ebbinghaus.record_retrieval(episode_id, success=success, current_time=now)

        # Update spaced repetition schedule
        if self.spaced_repetition:
            self.spaced_repetition.record_review(episode_id, success=success, current_time=now)
            self.reviewed_memories_count += 1

        # Update episode metadata
        episode = self._get_episode_by_id(episode_id)
        if episode:
            access_count = episode.metadata.get('access_count', 0)
            episode.metadata['access_count'] = access_count + 1
            episode.metadata['last_accessed'] = datetime.now().isoformat()

    def get_memories_for_review(self, limit: int = 10) -> List[Episode]:
        """
        Get memories that are due for spaced repetition review.

        Prioritizes:
        1. Immediate reviews (failed retrievals)
        2. Most overdue memories
        3. High-importance memories at risk

        Args:
            limit: Maximum number of memories to return

        Returns:
            List of episodes needing review
        """
        if not self.spaced_repetition:
            return []

        due_memory_ids = self.spaced_repetition.get_due_for_review(limit=limit)

        # Retrieve full episodes
        episodes = []
        for memory_id in due_memory_ids:
            episode = self._get_episode_by_id(memory_id)
            if episode:
                episodes.append(episode)

        return episodes

    def get_forgetting_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the forgetting system.

        Returns:
            Dict with forgetting, retention, and review statistics
        """
        stats = {
            "forgotten_memories": self.forgotten_memories_count,
            "reviewed_memories": self.reviewed_memories_count,
            "reinforced_memories": self.reinforced_memories_count,
            "ebbinghaus": {},
            "spaced_repetition": {},
            "stability_distribution": {}
        }

        if self.ebbinghaus:
            eb_stats = self.ebbinghaus.get_statistics()
            stats["ebbinghaus"] = eb_stats
            stats["stability_distribution"] = eb_stats.get("stability_distribution", {})

        if self.spaced_repetition:
            stats["spaced_repetition"] = self.spaced_repetition.get_statistics()

        return stats

    def _validate_episode_data(
        self,
        content: np.ndarray,
        surprise: float,
        embedding: Optional[np.ndarray] = None
    ) -> None:
        """
        Validate episode data before storage.

        Checks for NaN/Inf values in numerical fields and validates ranges.

        Args:
            content: Episode content array
            surprise: Surprise score
            embedding: Optional pre-computed embedding

        Raises:
            ValueError: If any values are invalid (NaN, Inf, or out of expected range)
            TypeError: If content is not a numpy array
        """
        # Validate content type
        if not isinstance(content, np.ndarray):
            raise TypeError(
                f"content must be a numpy array, got {type(content).__name__}"
            )

        # Validate content contains finite values
        if content.size > 0 and not np.all(np.isfinite(content)):
            non_finite_count = np.sum(~np.isfinite(content))
            raise ValueError(
                f"content contains {non_finite_count} non-finite values (NaN or Inf). "
                "Episode data must contain only finite numerical values."
            )

        # Validate surprise is finite
        if not np.isfinite(surprise):
            raise ValueError(
                f"surprise must be a finite value, got {surprise}. "
                "NaN and Inf values are not allowed."
            )

        # Validate surprise is non-negative (surprise scores shouldn't be negative)
        if surprise < 0:
            raise ValueError(
                f"surprise must be non-negative, got {surprise}. "
                "Negative surprise values are not meaningful."
            )

        # Validate embedding if provided
        if embedding is not None:
            if not isinstance(embedding, np.ndarray):
                raise TypeError(
                    f"embedding must be a numpy array, got {type(embedding).__name__}"
                )
            if embedding.size > 0 and not np.all(np.isfinite(embedding)):
                non_finite_count = np.sum(~np.isfinite(embedding))
                raise ValueError(
                    f"embedding contains {non_finite_count} non-finite values (NaN or Inf). "
                    "Embedding vectors must contain only finite numerical values."
                )

    def store_episode(
        self,
        content: np.ndarray,
        surprise: float = 0.0,
        timestamp: Optional[datetime] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[np.ndarray] = None
    ) -> Episode:
        """
        Store a new episodic memory (single-shot learning).

        Args:
            content: Observation content (must be numpy array with finite values)
            surprise: Surprise/novelty score from BayesianSurpriseEngine (must be finite, non-negative)
            timestamp: When this happened (defaults to now)
            location: Where this happened
            entities: Who/what was involved
            metadata: Additional context
            embedding: Pre-computed embedding (optional, must have finite values if provided)

        Returns:
            Created Episode object

        Raises:
            ValueError: If content, surprise, or embedding contain NaN/Inf values
            TypeError: If content or embedding is not a numpy array
        """
        # Validate input data before creating episode
        self._validate_episode_data(content, surprise, embedding)

        # Create episode
        episode = Episode(
            content=content,
            timestamp=timestamp or datetime.now(),
            location=location,
            entities=entities or [],
            embedding=embedding,
            surprise=surprise,
            importance=self._compute_importance(surprise),
            metadata=metadata or {}
        )
        
        # Generate embedding if not provided
        if episode.embedding is None:
            episode.embedding = self._generate_embedding(content)

        # Add to in-memory storage (thread-safe)
        with self._episodes_lock:
            self.episodes.append(episode)

        # Update indices
        self._update_indices(episode)
        
        # Add to vector database
        self._add_to_vector_db(episode)
        
        # Update statistics
        self.total_episodes_stored += 1

        # Register with Ebbinghaus forgetting system
        if self.ebbinghaus and episode.episode_id:
            ts = episode.timestamp.timestamp()
            # Initial retention based on importance
            initial_retention = 0.5 + (0.5 * episode.importance)
            self.ebbinghaus.register_memory(
                memory_id=episode.episode_id,
                initial_retention=initial_retention,
                timestamp=ts
            )
            # Schedule for spaced repetition
            if self.spaced_repetition:
                self.spaced_repetition.schedule_memory(episode.episode_id, current_time=ts)

        # Check if consolidation/offloading needed (thread-safe read)
        with self._episodes_lock:
            episodes_count = len(self.episodes)
        if episodes_count >= self.config.offload_threshold:
            self._consolidate_memory()

        return episode
    
    def _compute_importance(self, surprise: float) -> float:
        """
        Compute importance score for episode.
        Higher surprise → higher importance → prioritized for consolidation.

        Args:
            surprise: Surprise value from Bayesian surprise engine (must be finite)

        Returns:
            Importance score [0, 1]

        Note:
            Input validation is performed in store_episode() before this method is called.
            The sigmoid computation uses clipping to ensure numerical stability.
        """
        # Clip the exponent to prevent overflow in exp()
        # For surprise=0, exponent=2.0, for surprise=100, exponent=-98 (safe)
        exponent = np.clip(-surprise + 2.0, -500, 500)
        importance = 1.0 / (1.0 + np.exp(exponent))
        return float(np.clip(importance, 0.0, 1.0))
    
    def _generate_embedding(self, content: np.ndarray) -> np.ndarray:
        """
        Generate vector embedding for content.
        In production, use a pre-trained encoder (e.g., BERT, Sentence Transformers).

        Args:
            content: Observation content (already validated in store_episode)

        Returns:
            Embedding vector (normalized, finite values guaranteed)

        Note:
            Content is validated in store_episode() before this method is called.
            This method ensures the generated embedding is also valid.
        """
        # Simple projection for now (replace with real encoder)
        if len(content) < self.config.embedding_dim:
            # Pad with zeros
            embedding = np.zeros(self.config.embedding_dim)
            embedding[:len(content)] = content
        else:
            # Use PCA-like projection (simplified)
            embedding = content[:self.config.embedding_dim]

        # Normalize (with protection against zero norm)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        # If norm is 0, embedding is all zeros which is valid

        # Final safety check - ensure embedding has no NaN/Inf
        # This shouldn't happen if content is validated, but defensive programming
        if not np.all(np.isfinite(embedding)):
            raise ValueError(
                "Generated embedding contains non-finite values. "
                "This indicates a bug in embedding generation or invalid content."
            )

        return embedding
    
    def _update_indices(self, episode: Episode):
        """Update all indices with new episode."""
        # Temporal index (by date)
        date_key = episode.timestamp.strftime("%Y-%m-%d")
        if date_key not in self.temporal_index:
            self.temporal_index[date_key] = []
        self.temporal_index[date_key].append(episode.episode_id)

        # Spatial index
        if episode.location:
            if episode.location not in self.spatial_index:
                self.spatial_index[episode.location] = []
            self.spatial_index[episode.location].append(episode.episode_id)

        # Entity index
        for entity in episode.entities:
            if entity not in self.entity_index:
                self.entity_index[entity] = []
            self.entity_index[entity].append(episode.episode_id)

    def _remove_from_indices(self, episode: Episode):
        """
        Remove episode from all indices to prevent memory leaks.

        Called when episodes are offloaded or forgotten to ensure
        index entries don't hold stale references.
        """
        episode_id = episode.episode_id

        # Remove from temporal index
        date_key = episode.timestamp.strftime("%Y-%m-%d")
        if date_key in self.temporal_index:
            try:
                self.temporal_index[date_key].remove(episode_id)
                # Clean up empty date entries
                if not self.temporal_index[date_key]:
                    del self.temporal_index[date_key]
            except ValueError:
                pass  # Episode ID not in list (already removed)

        # Remove from spatial index
        if episode.location and episode.location in self.spatial_index:
            try:
                self.spatial_index[episode.location].remove(episode_id)
                # Clean up empty location entries
                if not self.spatial_index[episode.location]:
                    del self.spatial_index[episode.location]
            except ValueError:
                pass  # Episode ID not in list

        # Remove from entity index
        for entity in episode.entities:
            if entity in self.entity_index:
                try:
                    self.entity_index[entity].remove(episode_id)
                    # Clean up empty entity entries
                    if not self.entity_index[entity]:
                        del self.entity_index[entity]
                except ValueError:
                    pass  # Episode ID not in list

    def _remove_from_vector_db(self, episode_ids: List[str]):
        """
        Remove episodes from vector database.

        Args:
            episode_ids: List of episode IDs to remove
        """
        if not episode_ids:
            return

        if self.config.vector_db_backend == "chromadb":
            try:
                self.collection.delete(ids=episode_ids)
            except Exception as e:
                # Log but don't fail - episode may already be removed
                print(f"[EpisodicMemoryStore] Warning: Failed to delete from ChromaDB: {e}")
    
    def _add_to_vector_db(self, episode: Episode):
        """Add episode to vector database for similarity search."""
        if self.config.vector_db_backend == "chromadb":
            self.collection.add(
                ids=[episode.episode_id],
                embeddings=[episode.embedding.tolist()],
                metadatas=[{
                    "timestamp": episode.timestamp.isoformat(),
                    "location": episode.location or "",
                    "entities": ",".join(episode.entities),
                    "surprise": episode.surprise,
                    "importance": episode.importance
                }]
            )
    
    def retrieve_by_similarity(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_criteria: Optional[Dict] = None
    ) -> List[Episode]:
        """
        Retrieve k most similar episodes by vector similarity.

        Args:
            query_embedding: Query vector
            k: Number of episodes to retrieve
            filter_criteria: Optional filters (e.g., {"location": "office"})

        Returns:
            List of similar episodes
        """
        # Query vector database
        where_clause = None
        if filter_criteria:
            # ChromaDB requires specific format for filters
            # Convert to $and format if multiple criteria
            if len(filter_criteria) == 1:
                # Single criterion - use directly
                key, value = list(filter_criteria.items())[0]
                if isinstance(value, list):
                    # For lists (like entities), use $contains
                    where_clause = {key: {"$in": value}}
                else:
                    where_clause = {key: value}
            elif len(filter_criteria) > 1:
                # Multiple criteria - use $and
                conditions = []
                for key, value in filter_criteria.items():
                    if isinstance(value, list):
                        conditions.append({key: {"$in": value}})
                    else:
                        conditions.append({key: value})
                where_clause = {"$and": conditions}

        # Ensure query embedding is 1D
        query_emb = np.asarray(query_embedding).flatten().tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=k,
                where=where_clause
            )
        except ValueError:
            # If filter fails, try without filter
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=k
            )

        # Retrieve full episodes
        episode_ids = results['ids'][0] if results['ids'] else []
        episodes = [self._get_episode_by_id(eid) for eid in episode_ids]

        return [ep for ep in episodes if ep is not None]
    
    def retrieve_by_temporal_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Episode]:
        """
        Retrieve episodes within a time range.

        Args:
            start_time: Start of time window
            end_time: End of time window

        Returns:
            Episodes in time range
        """
        matching_episodes = []

        # Thread-safe iteration over episodes
        with self._episodes_lock:
            for episode in self.episodes:
                if start_time <= episode.timestamp <= end_time:
                    matching_episodes.append(episode)

        return matching_episodes
    
    def retrieve_by_location(self, location: str) -> List[Episode]:
        """Retrieve all episodes at a specific location."""
        episode_ids = self.spatial_index.get(location, [])
        episodes = [self._get_episode_by_id(eid) for eid in episode_ids]
        return [ep for ep in episodes if ep is not None]
    
    def retrieve_by_entity(self, entity: str) -> List[Episode]:
        """Retrieve all episodes involving a specific entity."""
        episode_ids = self.entity_index.get(entity, [])
        episodes = [self._get_episode_by_id(eid) for eid in episode_ids]
        return [ep for ep in episodes if ep is not None]
    
    def _get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """Retrieve episode by ID (thread-safe)."""
        # Thread-safe search through episodes
        with self._episodes_lock:
            for episode in self.episodes:
                if episode.episode_id == episode_id:
                    return episode

        # Check offloaded storage (outside lock - disk I/O shouldn't hold lock)
        if self.config.enable_disk_offload:
            return self._load_offloaded_episode(episode_id)

        return None
    
    def _consolidate_memory(self):
        """
        Consolidate memory by offloading low-importance episodes to disk.
        Mimics hippocampal consolidation: important memories stay, others archived.

        Memory Cleanup:
            When episodes are offloaded, this method also:
            1. Removes episodes from temporal, spatial, and entity indices
            2. Removes episodes from the vector database
            3. Triggers garbage collection after bulk removal to free memory
        """
        if not self.config.enable_disk_offload:
            return

        episodes_to_offload = []

        # Decay importance of all episodes
        # Thread-safe consolidation
        with self._episodes_lock:
            for episode in self.episodes:
                episode.importance *= self.config.importance_decay

            # Sort by importance
            self.episodes.sort(key=lambda ep: ep.importance, reverse=True)

            # Offload bottom X% to disk
            n_to_offload = len(self.episodes) - self.config.max_episodes
            if n_to_offload > 0:
                episodes_to_offload = self.episodes[-n_to_offload:]
                self.episodes = self.episodes[:-n_to_offload]

        # Cleanup and offload outside lock to avoid holding lock during I/O
        if episodes_to_offload:
            offloaded_ids = []

            for episode in episodes_to_offload:
                # Remove from all indices to prevent memory leaks
                self._remove_from_indices(episode)
                offloaded_ids.append(episode.episode_id)

                # Offload to disk
                self._offload_episode(episode)
                self.episodes_offloaded += 1

            # Remove from vector database in batch for efficiency
            self._remove_from_vector_db(offloaded_ids)

            # Trigger garbage collection after bulk removal to free memory
            # This helps reclaim memory from the removed episodes and index entries
            gc.collect()
    
    def _offload_episode(self, episode: Episode):
        """Offload episode to disk storage."""
        if not self.persistence_path:
            return
        
        offload_dir = self.persistence_path / "offloaded"
        offload_dir.mkdir(exist_ok=True)
        
        file_path = offload_dir / f"{episode.episode_id}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(episode, f)
    
    def _load_offloaded_episode(self, episode_id: str) -> Optional[Episode]:
        """Load episode from disk storage."""
        if not self.persistence_path:
            return None
        
        file_path = self.persistence_path / "offloaded" / f"{episode_id}.pkl"
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def save_state(self):
        """Save memory state to disk (thread-safe)."""
        if not self.persistence_path:
            return

        # Thread-safe read of episodes for serialization
        with self._episodes_lock:
            episodes_data = [ep.to_dict() for ep in self.episodes]

        state = {
            "config": self.config.__dict__,
            "episodes": episodes_data,
            "temporal_index": self.temporal_index,
            "spatial_index": self.spatial_index,
            "entity_index": self.entity_index,
            "total_episodes_stored": self.total_episodes_stored,
            "episodes_offloaded": self.episodes_offloaded,
            # Ebbinghaus forgetting stats
            "forgotten_memories_count": self.forgotten_memories_count,
            "reviewed_memories_count": self.reviewed_memories_count,
            "reinforced_memories_count": self.reinforced_memories_count
        }

        state_file = self.persistence_path / "memory_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load memory state from disk."""
        if not self.persistence_path:
            return
        
        state_file = self.persistence_path / "memory_state.json"
        if not state_file.exists():
            return
        
        with open(state_file, 'r') as f:
            state = json.load(f)

        # Restore episodes (thread-safe)
        with self._episodes_lock:
            self.episodes = [Episode.from_dict(ep_dict) for ep_dict in state["episodes"]]

        # Restore indices
        self.temporal_index = state["temporal_index"]
        self.spatial_index = state["spatial_index"]
        self.entity_index = state["entity_index"]
        
        # Restore statistics
        self.total_episodes_stored = state["total_episodes_stored"]
        self.episodes_offloaded = state["episodes_offloaded"]

        # Restore Ebbinghaus forgetting stats
        self.forgotten_memories_count = state.get("forgotten_memories_count", 0)
        self.reviewed_memories_count = state.get("reviewed_memories_count", 0)
        self.reinforced_memories_count = state.get("reinforced_memories_count", 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics including forgetting system stats (thread-safe)."""
        # Thread-safe read of episodes for statistics
        with self._episodes_lock:
            episodes_count = len(self.episodes)
            mean_importance = np.mean([ep.importance for ep in self.episodes]) if self.episodes else 0.0

        stats = {
            "total_episodes": self.total_episodes_stored,
            "episodes_in_memory": episodes_count,
            "episodes_offloaded": self.episodes_offloaded,
            "unique_locations": len(self.spatial_index),
            "unique_entities": len(self.entity_index),
            "temporal_span_days": len(self.temporal_index),
            "mean_importance": mean_importance,
            # Ebbinghaus forgetting stats
            "forgotten_memories": self.forgotten_memories_count,
            "reviewed_memories": self.reviewed_memories_count,
            "reinforced_memories": self.reinforced_memories_count
        }

        # Add forgetting system stats if available
        if self.ebbinghaus:
            eb_stats = self.ebbinghaus.get_statistics()
            stats["avg_retention"] = eb_stats.get("avg_retention", 0.0)
            stats["memories_at_risk"] = eb_stats.get("memories_at_risk", 0)
            stats["stability_distribution"] = eb_stats.get("stability_distribution", {})

        if self.spaced_repetition:
            sr_stats = self.spaced_repetition.get_statistics()
            stats["due_for_review"] = sr_stats.get("due_now", 0)
            stats["upcoming_reviews_24h"] = sr_stats.get("upcoming_24h", 0)
            stats["review_success_rate"] = sr_stats.get("success_rate", 0.0)

        return stats


if __name__ == "__main__":
    print("=== Episodic Memory Store Test ===\n")
    
    # Initialize memory store
    config = EpisodicMemoryConfig(max_episodes=100, embedding_dim=128)
    memory = EpisodicMemoryStore(config)
    
    # Store some episodes
    print("Storing episodes...")
    for i in range(50):
        content = np.random.randn(10)
        surprise = np.random.rand() * 3.0  # Random surprise
        location = ["office", "home", "cafe"][i % 3]
        entities = [f"person_{i % 5}"]
        
        episode = memory.store_episode(
            content=content,
            surprise=surprise,
            location=location,
            entities=entities,
            metadata={"event": f"test_event_{i}"}
        )
    
    # Retrieve statistics
    stats = memory.get_statistics()
    print(f"\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test similarity retrieval
    query = np.random.randn(128)
    similar_episodes = memory.retrieve_by_similarity(query, k=5)
    print(f"\nRetrieved {len(similar_episodes)} similar episodes")
    
    # Test location retrieval
    office_episodes = memory.retrieve_by_location("office")
    print(f"Episodes at 'office': {len(office_episodes)}")
    
    # Test entity retrieval
    person_0_episodes = memory.retrieve_by_entity("person_0")
    print(f"Episodes with 'person_0': {len(person_0_episodes)}")
    
    print("\n✓ Episodic memory store test complete!")
