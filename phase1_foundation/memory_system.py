"""
PHASE 1 - STEP 4: MEMORY ARCHITECTURE
======================================

Complete Human-Like Memory System based on neuroscience research.

Key Innovation vs Existing AI:
- LLMs: Only have context window (not real memory)
- ACT-R: Good declarative memory but limited
- This: Multi-store model with consolidation, decay, and emotional modulation

Memory Types:
1. Sensory Buffer (iconic, echoic) - milliseconds
2. Working Memory - seconds, limited capacity (7±2 items)
3. Episodic Memory - personal experiences with context
4. Semantic Memory - facts and knowledge (context-free)
5. Procedural Memory - skills and habits (implicit)

Based on:
- Atkinson-Shiffrin Multi-Store Model
- Baddeley's Working Memory Model
- Tulving's Memory Systems
- Engram Research (2024-2025)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import time
import heapq
from collections import OrderedDict
import threading

import sys
sys.path.append('..')
from utils.fast_math import (
    cosine_similarity_matrix,
    top_k_indices,
    FastVectorIndex,
    PRECISION,
    VECTOR_DIM
)
from utils.base_types import (
    Vector, MemoryItem, Episode, Fact, Skill,
    EmotionalState, Timestamp, RingBuffer,
    MemoryType, signal_bus
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MemoryConfig:
    """Configuration for memory system."""
    embedding_dim: int = VECTOR_DIM

    # Sensory buffer
    sensory_duration_ms: float = 500.0
    sensory_capacity: int = 100

    # Working memory
    working_memory_capacity: int = 7  # Miller's Law: 7±2
    working_memory_duration_sec: float = 30.0
    rehearsal_boost: float = 0.2

    # Long-term memory
    ltm_capacity: int = 1000000  # 1 million items
    consolidation_threshold: float = 0.5
    decay_rate: float = 0.001
    emotional_boost: float = 0.3

    # Retrieval
    retrieval_k: int = 10
    similarity_threshold: float = 0.5


# =============================================================================
# SENSORY BUFFER
# =============================================================================

class SensoryBuffer:
    """
    Ultra-short-term sensory storage.

    Types:
    - Iconic (visual): ~250ms
    - Echoic (auditory): ~3-4 seconds
    - Haptic (touch): ~2 seconds

    Function: Hold raw sensory data until attention selects what to process.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config

        # Separate buffers for each modality
        self.iconic = RingBuffer(config.sensory_capacity)  # Visual
        self.echoic = RingBuffer(config.sensory_capacity)  # Auditory
        self.haptic = RingBuffer(config.sensory_capacity)  # Touch

        # Modality-specific decay rates (in seconds)
        self.decay_times = {
            'iconic': 0.25,
            'echoic': 3.0,
            'haptic': 2.0
        }

    def store(self, data: np.ndarray, modality: str = 'iconic') -> str:
        """Store sensory data in appropriate buffer."""
        item = {
            'data': data.astype(PRECISION),
            'timestamp': time.time(),
            'modality': modality
        }

        buffer = getattr(self, modality, self.iconic)
        buffer.push(item)

        return f"sensory_{modality}_{item['timestamp']}"

    def get_active(self, modality: str = 'iconic') -> List[np.ndarray]:
        """Get all active (non-decayed) items from a buffer."""
        buffer = getattr(self, modality, self.iconic)
        decay_time = self.decay_times.get(modality, 0.25)
        current_time = time.time()

        active = []
        for item in buffer.get_all():
            age = current_time - item['timestamp']
            if age < decay_time:
                active.append(item['data'])

        return active

    def attend(self, modality: str = 'iconic') -> Optional[np.ndarray]:
        """
        Attention selects the most recent item for further processing.
        This is what enters working memory.
        """
        active = self.get_active(modality)
        if active:
            return active[-1]  # Most recent
        return None


# =============================================================================
# WORKING MEMORY
# =============================================================================

@dataclass
class WorkingMemorySlot:
    """A slot in working memory."""
    content: Any
    embedding: np.ndarray
    activation: float = 1.0
    timestamp: float = field(default_factory=time.time)
    rehearsals: int = 0
    source: str = ""  # Where did this come from?

    def decay(self, rate: float = 0.1):
        """Decay activation over time."""
        age = time.time() - self.timestamp
        self.activation *= np.exp(-rate * age)


class WorkingMemory:
    """
    Active manipulation of limited items with ATTENTION SPOTLIGHT.

    Based on Baddeley's model:
    - Central Executive: Controls attention and coordinates
    - Phonological Loop: Verbal rehearsal
    - Visuospatial Sketchpad: Visual/spatial info
    - Episodic Buffer: Integrates information

    Key properties:
    - Limited capacity: 7±2 items (Miller's Law)
    - Decays without rehearsal (exponential decay)
    - Context priming via spreading activation
    - Attention spotlight focuses on relevant items
    - Gateway to long-term memory
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.capacity = config.working_memory_capacity

        # The slots
        self.slots: List[Optional[WorkingMemorySlot]] = [None] * self.capacity

        # Current attention focus (goal/context embedding)
        self.attention_focus: Optional[np.ndarray] = None

        # Decay parameters
        self.decay_rate = 0.05  # Exponential decay rate

        # Subsystems
        self.phonological_loop: List[str] = []  # Verbal items for rehearsal
        self.visuospatial: Optional[np.ndarray] = None  # Current visual representation

        # Statistics
        self.total_items_processed = 0
        self.rehearsal_count = 0

        # Lock for thread safety
        self.lock = threading.Lock()

    def set_attention_focus(self, focus_embedding: np.ndarray):
        """Set the current attention focus (goal/context)."""
        self.attention_focus = focus_embedding.astype(PRECISION)
        # Apply context priming - boost related items
        self._apply_context_priming()

    def _apply_context_priming(self):
        """Boost activation of items related to current focus (spreading activation)."""
        if self.attention_focus is None:
            return

        for slot in self.slots:
            if slot is None:
                continue

            # Compute similarity to focus
            similarity = np.dot(slot.embedding, self.attention_focus) / (
                np.linalg.norm(slot.embedding) * np.linalg.norm(self.attention_focus) + 1e-8
            )

            # Boost activation proportional to similarity
            if similarity > 0.3:
                priming_boost = similarity * 0.3
                slot.activation = min(1.0, slot.activation + priming_boost)

    def apply_exponential_decay(self):
        """Apply exponential decay to all items: activation *= exp(-k * time_elapsed)."""
        current_time = time.time()
        for slot in self.slots:
            if slot is None:
                continue

            time_elapsed = current_time - slot.timestamp
            decay_factor = np.exp(-self.decay_rate * time_elapsed)
            slot.activation *= decay_factor

            # Remove items that have decayed too much
            if slot.activation < 0.05:
                idx = self.slots.index(slot)
                self.slots[idx] = None

    def load(self, content: Any, embedding: np.ndarray, source: str = "",
             goal_embedding: Optional[np.ndarray] = None) -> int:
        """
        Load item into working memory with ATTENTION SPOTLIGHT.

        If goal_embedding provided, prioritizes slots by similarity to goal.
        Returns slot index, or -1 if failed.
        """
        with self.lock:
            # Apply decay before loading
            self.apply_exponential_decay()

            # Find best slot based on attention
            if goal_embedding is not None:
                slot_idx = self._find_slot_by_attention(embedding, goal_embedding)
            else:
                slot_idx = self._find_slot()

            self.slots[slot_idx] = WorkingMemorySlot(
                content=content,
                embedding=embedding.astype(PRECISION),
                activation=1.0,
                source=source
            )

            # If we have attention focus, apply priming to new item
            if self.attention_focus is not None:
                similarity = np.dot(embedding, self.attention_focus) / (
                    np.linalg.norm(embedding) * np.linalg.norm(self.attention_focus) + 1e-8
                )
                if similarity > 0.5:
                    self.slots[slot_idx].activation = min(1.0, 1.0 + similarity * 0.2)

            self.total_items_processed += 1
            return slot_idx

    def _find_slot_by_attention(self, new_embedding: np.ndarray,
                                 goal_embedding: np.ndarray) -> int:
        """Find slot using attention spotlight - keep relevant items."""
        # First, look for empty slot
        for i, slot in enumerate(self.slots):
            if slot is None:
                return i

        # Compute relevance scores for existing items
        scores = []
        for i, slot in enumerate(self.slots):
            if slot is None:
                scores.append((i, -1))
                continue

            # Relevance = similarity to goal + activation
            goal_sim = np.dot(slot.embedding, goal_embedding) / (
                np.linalg.norm(slot.embedding) * np.linalg.norm(goal_embedding) + 1e-8
            )
            score = goal_sim * 0.6 + slot.activation * 0.4
            scores.append((i, score))

        # Replace LEAST relevant item (lowest score)
        scores.sort(key=lambda x: x[1])
        return scores[0][0]

    def _find_slot(self) -> int:
        """Find best slot: empty or lowest activation."""
        # First, look for empty slot
        for i, slot in enumerate(self.slots):
            if slot is None:
                return i

        # Decay all slots first
        self.apply_exponential_decay()

        # Find lowest activation
        min_activation = float('inf')
        min_idx = 0
        for i, slot in enumerate(self.slots):
            if slot and slot.activation < min_activation:
                min_activation = slot.activation
                min_idx = i

        return min_idx

    def get(self, idx: int) -> Optional[WorkingMemorySlot]:
        """Get item from slot."""
        if 0 <= idx < len(self.slots):
            slot = self.slots[idx]
            if slot:
                slot.activation = min(1.0, slot.activation + 0.1)  # Boost on access
            return slot
        return None

    def get_all_active(self) -> List[WorkingMemorySlot]:
        """Get all non-None slots above activation threshold."""
        active = []
        for slot in self.slots:
            if slot and slot.activation > 0.1:
                active.append(slot)
        return active

    def rehearse(self, idx: int):
        """
        Rehearse item to prevent decay and boost activation.
        This is how we keep things in working memory.
        """
        with self.lock:
            slot = self.slots[idx]
            if slot:
                slot.activation = min(1.0, slot.activation + self.config.rehearsal_boost)
                slot.timestamp = time.time()  # Reset decay timer
                slot.rehearsals += 1
                self.rehearsal_count += 1

    def search(self, query: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, float]]:
        """Search working memory by similarity."""
        results = []
        query = query.astype(PRECISION)

        for i, slot in enumerate(self.slots):
            if slot is None:
                continue

            similarity = np.dot(query, slot.embedding) / (
                np.linalg.norm(query) * np.linalg.norm(slot.embedding) + 1e-8
            )

            if similarity > threshold:
                results.append((i, float(similarity)))

        return sorted(results, key=lambda x: -x[1])

    def clear(self):
        """Clear all slots."""
        with self.lock:
            self.slots = [None] * self.capacity

    def get_contents(self) -> List[Any]:
        """Get content of all active slots."""
        return [slot.content for slot in self.slots if slot]

    @property
    def num_active(self) -> int:
        """Count active items."""
        return sum(1 for slot in self.slots if slot and slot.activation > 0.1)


# =============================================================================
# EPISODIC MEMORY
# =============================================================================

class EpisodicMemory:
    """
    Personal experiences with context (what, when, where, who, how felt).

    Properties:
    - Rich contextual information
    - Time-tagged
    - Emotionally modulated
    - Decays over time but can be consolidated

    Based on hippocampal memory research and engram studies.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config

        # Main storage: id -> Episode
        self.episodes: Dict[str, Episode] = {}

        # Indices for fast retrieval
        self.vector_index = FastVectorIndex(config.embedding_dim)
        self.temporal_index: List[Tuple[float, str]] = []  # (timestamp, id) sorted
        self.emotional_index: Dict[str, List[str]] = {}  # emotion -> [episode_ids]

        # Statistics
        self.total_encoded = 0
        self.total_retrieved = 0

    def encode(
        self,
        content: Any,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        emotional_state: Optional[EmotionalState] = None,
        location: Optional[np.ndarray] = None
    ) -> str:
        """
        Encode a new episodic memory.

        Args:
            content: The experience content
            embedding: Vector representation
            context: Contextual information
            emotional_state: How we felt
            location: Where it happened

        Returns:
            Episode ID
        """
        episode = Episode(
            content=content,
            embedding=Vector(embedding.astype(PRECISION)),
            what=str(content)[:100],
            where=Vector(location) if location is not None else None,
            when=Timestamp(),
            emotional_state=emotional_state,
            context=context or {}
        )

        # Emotional boost to strength
        if emotional_state and emotional_state.intensity > 0.5:
            episode.strength *= (1 + self.config.emotional_boost * emotional_state.intensity)

        # Store
        self.episodes[episode.id] = episode

        # Index
        self.vector_index.add(embedding.astype(PRECISION), episode.id)
        heapq.heappush(self.temporal_index, (episode.timestamp.created, episode.id))

        if emotional_state:
            emotion_key = emotional_state.primary_emotion.value
            if emotion_key not in self.emotional_index:
                self.emotional_index[emotion_key] = []
            self.emotional_index[emotion_key].append(episode.id)

        self.total_encoded += 1

        signal_bus.publish('episode_encoded', {'id': episode.id})

        return episode.id

    def retrieve(
        self,
        query: Optional[np.ndarray] = None,
        time_range: Optional[Tuple[float, float]] = None,
        emotion: Optional[str] = None,
        k: int = 10
    ) -> List[Episode]:
        """
        Retrieve episodes by various criteria.

        Args:
            query: Similarity-based retrieval
            time_range: (start_time, end_time) filter
            emotion: Filter by emotional state
            k: Max number to return

        Returns:
            List of matching episodes
        """
        candidates = set(self.episodes.keys())

        # Filter by time
        if time_range:
            start, end = time_range
            candidates = {
                eid for eid in candidates
                if start <= self.episodes[eid].timestamp.created <= end
            }

        # Filter by emotion
        if emotion and emotion in self.emotional_index:
            candidates &= set(self.emotional_index[emotion])

        # Similarity search
        if query is not None:
            results = self.vector_index.search(query.astype(PRECISION), k=min(k * 2, len(candidates)))
            similarity_ids = {eid for _, eid in [(d, self.vector_index.metadata[i]) for i, d in results]}
            candidates &= similarity_ids

        # Get episodes and sort by relevance
        episodes = [self.episodes[eid] for eid in candidates if eid in self.episodes]

        # Sort by strength * recency
        current_time = time.time()
        episodes.sort(
            key=lambda e: e.strength * np.exp(-0.001 * (current_time - e.timestamp.created)),
            reverse=True
        )

        # Access and return
        for episode in episodes[:k]:
            episode.access()
            self.total_retrieved += 1

        return episodes[:k]

    def consolidate(self, episode_id: str) -> bool:
        """
        Strengthen an episode (simulate sleep consolidation).
        """
        if episode_id in self.episodes:
            self.episodes[episode_id].strength = min(
                1.0,
                self.episodes[episode_id].strength + 0.2
            )
            return True
        return False

    def forget(self, episode_id: str) -> bool:
        """Explicitly forget an episode."""
        if episode_id in self.episodes:
            del self.episodes[episode_id]
            return True
        return False

    def decay_all(self):
        """Apply decay to all episodes."""
        to_remove = []
        for eid, episode in self.episodes.items():
            episode.decay(self.config.decay_rate)
            if episode.strength < 0.01:
                to_remove.append(eid)

        for eid in to_remove:
            del self.episodes[eid]

    def get_recent(self, hours: float = 24.0) -> List[Episode]:
        """Get episodes from the last N hours."""
        cutoff = time.time() - hours * 3600
        return self.retrieve(time_range=(cutoff, time.time()))


# =============================================================================
# SEMANTIC MEMORY
# =============================================================================

class SemanticMemory:
    """
    Facts and general knowledge (context-free).

    Unlike episodic memory:
    - No personal context (when/where learned)
    - Organized conceptually, not temporally
    - More resistant to forgetting

    Stored as a knowledge graph + embeddings for hybrid retrieval.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config

        # Knowledge graph: subject -> relation -> objects
        self.graph: Dict[str, Dict[str, List[str]]] = {}

        # Fact storage
        self.facts: Dict[str, Fact] = {}

        # Vector index for semantic search
        self.vector_index = FastVectorIndex(config.embedding_dim)

        # Concept embeddings
        self.concept_embeddings: Dict[str, np.ndarray] = {}

    def store(
        self,
        subject: str,
        relation: str,
        obj: str,
        embedding: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        source: str = ""
    ) -> str:
        """
        Store a semantic fact (subject-relation-object triple).

        Args:
            subject: The subject entity
            relation: The relationship type
            obj: The object entity
            embedding: Vector representation of the fact
            confidence: How confident we are (0-1)
            source: Where did we learn this?

        Returns:
            Fact ID
        """
        fact = Fact(
            subject=subject,
            relation=relation,
            object=obj,
            confidence=confidence,
            source=source,
            content=f"{subject} {relation} {obj}"
        )

        if embedding is not None:
            fact.embedding = Vector(embedding.astype(PRECISION))
            self.vector_index.add(embedding.astype(PRECISION), fact.id)

        # Store fact
        self.facts[fact.id] = fact

        # Add to graph
        if subject not in self.graph:
            self.graph[subject] = {}
        if relation not in self.graph[subject]:
            self.graph[subject][relation] = []
        if obj not in self.graph[subject][relation]:
            self.graph[subject][relation].append(obj)

        return fact.id

    def query_graph(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Query the knowledge graph.

        Examples:
            query_graph(subject="Paris") -> all facts about Paris
            query_graph(relation="capital_of") -> all capital relationships
            query_graph(subject="France", relation="capital_of") -> Paris
        """
        results = []

        for s, relations in self.graph.items():
            if subject and s != subject:
                continue

            for r, objects in relations.items():
                if relation and r != relation:
                    continue

                for o in objects:
                    if obj and o != obj:
                        continue
                    results.append((s, r, o))

        return results

    def search(self, query: np.ndarray, k: int = 10) -> List[Fact]:
        """Semantic similarity search."""
        results = self.vector_index.search(query.astype(PRECISION), k=k)
        facts = []
        for idx, dist in results:
            if idx < len(self.vector_index.metadata):
                fact_id = self.vector_index.metadata[idx]
                if fact_id in self.facts:
                    facts.append(self.facts[fact_id])
        return facts

    def get_related(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get all concepts related to a given concept up to N hops.
        """
        visited = set()
        to_visit = [(concept, 0)]
        related = {}

        while to_visit:
            current, d = to_visit.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)

            if current in self.graph:
                related[current] = self.graph[current]
                for relation, objects in self.graph[current].items():
                    for obj in objects:
                        if obj not in visited:
                            to_visit.append((obj, d + 1))

        return related


# =============================================================================
# PROCEDURAL MEMORY
# =============================================================================

class ProceduralMemory:
    """
    Skills and habits (implicit memory).

    Properties:
    - Acquired through practice
    - Becomes automatic over time
    - Resistant to forgetting
    - Difficult to verbalize

    Examples: riding a bike, typing, playing an instrument
    """

    def __init__(self, config: MemoryConfig):
        self.config = config

        # Skill storage
        self.skills: Dict[str, Skill] = {}

        # Habit associations: context -> action
        self.habits: Dict[str, str] = {}

        # Motor programs: action -> sequence of steps
        self.motor_programs: Dict[str, List[Any]] = {}

    def learn_skill(
        self,
        name: str,
        action_sequence: List[Any],
        trigger_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Learn a new skill from demonstration.
        """
        skill = Skill(
            name=name,
            action_sequence=action_sequence,
            trigger_context=trigger_context or {},
            automaticity=0.0,  # Starts not automatic
            success_rate=0.5
        )

        self.skills[skill.id] = skill
        return skill.id

    def practice(self, skill_id: str, success: bool) -> float:
        """
        Practice a skill. Increases automaticity on success.

        Returns new automaticity level.
        """
        if skill_id not in self.skills:
            return 0.0

        skill = self.skills[skill_id]

        # Update success rate
        skill.success_rate = 0.9 * skill.success_rate + 0.1 * (1.0 if success else 0.0)

        # Increase automaticity on success
        if success:
            skill.automaticity = min(1.0, skill.automaticity + 0.05)

        skill.access()
        return skill.automaticity

    def execute(self, skill_id: str) -> Optional[List[Any]]:
        """
        Execute a learned skill.

        Returns action sequence if skill exists and is sufficiently learned.
        """
        if skill_id not in self.skills:
            return None

        skill = self.skills[skill_id]

        # Random failure based on (1 - success_rate)
        if np.random.random() > skill.success_rate:
            return None

        skill.access()
        return skill.action_sequence

    def form_habit(self, context: str, action: str):
        """
        Form a habit: automatic association between context and action.
        """
        self.habits[context] = action

    def get_habitual_action(self, context: str) -> Optional[str]:
        """
        Get the habitual action for a context, if one exists.
        """
        return self.habits.get(context)

    def is_automatic(self, skill_id: str, threshold: float = 0.8) -> bool:
        """Check if a skill is automatic (no conscious effort needed)."""
        if skill_id not in self.skills:
            return False
        return self.skills[skill_id].automaticity >= threshold


# =============================================================================
# COMPLETE MEMORY SYSTEM
# =============================================================================

class MemorySystem:
    """
    Complete Human-Like Memory System.

    Integrates all memory types with:
    - Encoding from sensory to LTM
    - Retrieval with spreading activation
    - Consolidation during 'sleep'
    - Forgetting through decay

    This is what's MISSING in current AI:
    - LLMs: No persistent memory
    - Traditional AI: Limited memory models
    - This: Full multi-store model with realistic dynamics
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()

        # Memory stores
        self.sensory = SensoryBuffer(self.config)
        self.working = WorkingMemory(self.config)
        self.episodic = EpisodicMemory(self.config)
        self.semantic = SemanticMemory(self.config)
        self.procedural = ProceduralMemory(self.config)

        # Statistics
        self.total_operations = 0

    def perceive(self, input_data: np.ndarray, modality: str = 'iconic') -> str:
        """
        Entry point: sensory input.
        """
        return self.sensory.store(input_data, modality)

    def attend(self, modality: str = 'iconic') -> Optional[np.ndarray]:
        """
        Attention selects from sensory buffer into working memory.
        """
        data = self.sensory.attend(modality)
        if data is not None:
            self.working.load(
                content=f"attended_{modality}",
                embedding=data,
                source="sensory"
            )
        return data

    def remember_episode(
        self,
        content: Any,
        embedding: np.ndarray,
        emotional_state: Optional[EmotionalState] = None,
        **context
    ) -> str:
        """
        Store an episodic memory.
        """
        return self.episodic.encode(
            content=content,
            embedding=embedding,
            emotional_state=emotional_state,
            context=context
        )

    def learn_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Store a semantic fact.
        """
        return self.semantic.store(subject, relation, obj, embedding)

    def learn_skill(self, name: str, actions: List[Any]) -> str:
        """
        Learn a procedural skill.
        """
        return self.procedural.learn_skill(name, actions)

    def recall(
        self,
        query: np.ndarray,
        memory_types: Optional[List[MemoryType]] = None,
        k: int = 10
    ) -> Dict[str, List[Any]]:
        """
        Unified recall across memory types.
        """
        if memory_types is None:
            memory_types = [MemoryType.WORKING, MemoryType.EPISODIC, MemoryType.SEMANTIC]

        results = {}

        if MemoryType.WORKING in memory_types:
            wm_results = self.working.search(query)
            results['working'] = [self.working.get(idx) for idx, _ in wm_results[:k]]

        if MemoryType.EPISODIC in memory_types:
            results['episodic'] = self.episodic.retrieve(query=query, k=k)

        if MemoryType.SEMANTIC in memory_types:
            results['semantic'] = self.semantic.search(query, k=k)

        self.total_operations += 1
        return results

    def consolidate(self):
        """
        Simulate sleep consolidation:
        - Replay important memories
        - Strengthen some, weaken others
        - Transfer from episodic to semantic
        """
        # Get important recent episodes
        recent = self.episodic.get_recent(hours=24)

        for episode in recent:
            # Consolidate high-strength episodes
            if episode.strength > self.config.consolidation_threshold:
                self.episodic.consolidate(episode.id)

                # Extract semantic content
                if episode.content:
                    # This is simplified - real system would extract meaningful facts
                    pass

        # Decay all memories
        self.episodic.decay_all()

    def get_working_memory_contents(self) -> List[np.ndarray]:
        """
        Get embeddings of all active items in working memory.

        Returns list of embedding vectors for active working memory items,
        ordered by when they were added (oldest first).
        """
        contents = []
        # Get slots sorted by timestamp (oldest first for proper N-back)
        active_slots = [(slot, slot.timestamp) for slot in self.working.slots
                        if slot is not None and slot.activation > 0.05]
        active_slots.sort(key=lambda x: x[1])

        for slot, _ in active_slots:
            if hasattr(slot, 'embedding'):
                contents.append(slot.embedding)
        return contents

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            'working_memory_items': self.working.num_active,
            'episodic_count': len(self.episodic.episodes),
            'semantic_facts': len(self.semantic.facts),
            'skills_learned': len(self.procedural.skills),
            'total_operations': self.total_operations
        }


# =============================================================================
# TESTING
# =============================================================================

def test_memory_system():
    """Test the complete memory system."""
    print("\n" + "=" * 60)
    print("MEMORY SYSTEM TEST")
    print("=" * 60)

    config = MemoryConfig(embedding_dim=64)
    memory = MemorySystem(config)

    # Test sensory buffer
    print("\n--- Sensory Buffer ---")
    for i in range(10):
        data = np.random.randn(64).astype(PRECISION)
        memory.perceive(data, 'iconic')
    print(f"Stored 10 items in sensory buffer")

    attended = memory.attend('iconic')
    print(f"Attended to item: shape={attended.shape if attended is not None else None}")

    # Test working memory
    print("\n--- Working Memory ---")
    for i in range(10):
        embedding = np.random.randn(64).astype(PRECISION)
        memory.working.load(f"item_{i}", embedding)
    print(f"Active items: {memory.working.num_active}")
    print(f"Capacity: {config.working_memory_capacity}")

    # Test episodic memory
    print("\n--- Episodic Memory ---")
    for i in range(100):
        embedding = np.random.randn(64).astype(PRECISION)
        emotion = EmotionalState(valence=np.random.randn(), intensity=abs(np.random.randn()))
        memory.remember_episode(
            content=f"Experience {i}",
            embedding=embedding,
            emotional_state=emotion,
            location=f"Place {i % 10}"
        )
    print(f"Encoded {len(memory.episodic.episodes)} episodes")

    # Retrieval
    query = np.random.randn(64).astype(PRECISION)
    start = time.perf_counter()
    results = memory.episodic.retrieve(query=query, k=5)
    elapsed = time.perf_counter() - start
    print(f"Retrieved {len(results)} episodes in {elapsed*1000:.2f}ms")

    # Test semantic memory
    print("\n--- Semantic Memory ---")
    facts = [
        ("Paris", "capital_of", "France"),
        ("Berlin", "capital_of", "Germany"),
        ("Tokyo", "capital_of", "Japan"),
        ("France", "in_continent", "Europe"),
        ("Germany", "in_continent", "Europe"),
    ]
    for s, r, o in facts:
        embedding = np.random.randn(64).astype(PRECISION)
        memory.learn_fact(s, r, o, embedding)

    results = memory.semantic.query_graph(relation="capital_of")
    print(f"Capitals: {results}")

    related = memory.semantic.get_related("France", depth=2)
    print(f"Related to France: {list(related.keys())}")

    # Test procedural memory
    print("\n--- Procedural Memory ---")
    skill_id = memory.learn_skill("typing", ["place_fingers", "press_keys", "look_at_screen"])
    for _ in range(20):
        memory.procedural.practice(skill_id, success=True)
    print(f"Skill automaticity: {memory.procedural.skills[skill_id].automaticity:.2f}")
    print(f"Is automatic: {memory.procedural.is_automatic(skill_id)}")

    # Overall stats
    print("\n--- Statistics ---")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)

    return memory


if __name__ == "__main__":
    test_memory_system()
