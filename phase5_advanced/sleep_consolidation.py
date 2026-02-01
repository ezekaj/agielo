"""
Sleep and Memory Consolidation System

Implements:
1. Memory Replay (hippocampal replay during sleep)
2. Synaptic Homeostasis (downscaling)
3. Memory Consolidation (episodic -> semantic)
4. Dream Generation (recombination of memories)
5. Circadian Rhythm Effects
6. Sleep Stage Simulation (NREM, REM)

Based on research:
- Tononi & Cirelli: Synaptic Homeostasis Hypothesis
- Born & Diekelmann: Active system consolidation
- Stickgold: Memory consolidation in sleep
- Walker: Sleep and emotional memory

Performance: Efficient batch replay, O(n) consolidation
Comparison vs existing:
- LLMs: No sleep or consolidation
- ACT-R: No sleep
- Neural networks: Catastrophic forgetting
- This: Sleep prevents forgetting, consolidates learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time


class SleepStage(Enum):
    """Sleep stages."""
    WAKE = auto()
    N1 = auto()      # Light sleep
    N2 = auto()      # True sleep
    N3 = auto()      # Deep sleep (slow-wave)
    REM = auto()     # Rapid eye movement


class MemoryType(Enum):
    """Types of memories for consolidation."""
    EPISODIC = auto()
    SEMANTIC = auto()
    PROCEDURAL = auto()
    EMOTIONAL = auto()


@dataclass
class MemoryTrace:
    """A memory trace for consolidation."""
    content: np.ndarray
    memory_type: MemoryType
    strength: float            # 0-1
    emotional_salience: float  # Emotional importance
    creation_time: float
    last_replay: float = 0.0
    replay_count: int = 0
    consolidated: bool = False
    source_episode: Optional[str] = None


@dataclass
class SleepSession:
    """A sleep session with stages."""
    start_time: float
    duration: float
    stages: List[Tuple[SleepStage, float]]  # (stage, duration)
    memories_replayed: int = 0
    consolidation_score: float = 0.0


@dataclass
class Dream:
    """A dream experience."""
    content: np.ndarray
    narrative: List[str]
    source_memories: List[str]
    emotional_tone: float
    bizarreness: float  # How unusual/creative
    timestamp: float = field(default_factory=time.time)


class HippocampalReplay:
    """
    Hippocampal replay of memories during sleep.

    During NREM sleep, memories are replayed in compressed time,
    strengthening connections and enabling consolidation.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Replay buffer
        self.replay_buffer: List[MemoryTrace] = []

        # Replay speed (compressed time)
        self.replay_compression = 10.0  # 10x faster than real time

        # Replay threshold
        self.min_strength_for_replay = 0.2

    def add_to_buffer(self, memory: MemoryTrace):
        """Add memory to replay buffer."""
        self.replay_buffer.append(memory)

        # Limit buffer size
        if len(self.replay_buffer) > 1000:
            # Remove weakest memories
            self.replay_buffer.sort(key=lambda m: m.strength, reverse=True)
            self.replay_buffer = self.replay_buffer[:500]

    def select_for_replay(self, num_memories: int = 50) -> List[MemoryTrace]:
        """
        Select memories for replay using PROBABILISTIC sampling.

        Probability proportional to:
        - Emotional intensity (stronger emotions = more likely to replay)
        - Strength (already strong memories get reinforced)
        - Rehearsal count (frequently rehearsed = important)
        - Recency (recent memories prioritized)

        This mimics how the brain preferentially replays emotionally
        significant and frequently accessed memories during REM sleep.
        """
        if not self.replay_buffer:
            return []

        # Calculate replay probability for each memory
        candidates = []
        probabilities = []

        for memory in self.replay_buffer:
            if memory.strength < self.min_strength_for_replay:
                continue

            # Compute composite score for probability
            # Emotional salience is key driver (REM consolidation)
            emotional_weight = memory.emotional_salience * 2.0

            # Strength indicates importance
            strength_weight = memory.strength

            # Rehearsal count shows repeated access (important)
            rehearsal_weight = np.log1p(memory.replay_count) * 0.5

            # Recency bias (recent memories prioritized)
            hours_old = (time.time() - memory.creation_time) / 3600
            recency_weight = 1.0 / (1.0 + hours_old * 0.1)

            # Combined probability score
            prob_score = (
                emotional_weight * 0.35 +
                strength_weight * 0.25 +
                rehearsal_weight * 0.20 +
                recency_weight * 0.20
            )

            candidates.append(memory)
            probabilities.append(max(0.01, prob_score))  # Minimum probability

        if not candidates:
            return []

        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        # Sample without replacement based on probabilities
        num_to_select = min(num_memories, len(candidates))

        # Use numpy for weighted sampling
        indices = np.random.choice(
            len(candidates),
            size=num_to_select,
            replace=False,
            p=probabilities
        )

        selected = [candidates[i] for i in indices]
        return selected

    def replay(self, memories: List[MemoryTrace]) -> Dict[str, Any]:
        """Execute replay of memories."""
        replayed = []

        for memory in memories:
            # Strengthen memory through replay
            memory.strength = min(1.0, memory.strength + 0.05)
            memory.replay_count += 1
            memory.last_replay = time.time()

            replayed.append({
                'type': memory.memory_type.name,
                'strength_after': memory.strength,
                'replay_count': memory.replay_count
            })

        return {
            'num_replayed': len(replayed),
            'memories': replayed
        }


class SynapticHomeostasis:
    """
    Synaptic homeostasis - downscaling during sleep.

    During waking, synapses strengthen (learning).
    During sleep, overall synaptic strength is scaled down,
    preserving relative differences but preventing saturation.
    """

    def __init__(self):
        # Homeostatic setpoint
        self.target_total_strength = 1.0

        # Scaling rate
        self.downscaling_rate = 0.1

    def compute_total_strength(self, weights: np.ndarray) -> float:
        """Compute total synaptic strength."""
        return float(np.sum(np.abs(weights)))

    def downscale(self, weights: np.ndarray, duration: float = 1.0) -> np.ndarray:
        """
        Downscale synaptic weights while preserving relative strengths.

        This is the "smart forgetting" that sleep provides:
        - Weak connections fade
        - Strong connections relatively preserved
        - Overall capacity reset for new learning
        """
        current_total = self.compute_total_strength(weights)

        if current_total <= self.target_total_strength:
            return weights  # No scaling needed

        # Compute scaling factor
        scale_factor = 1.0 - self.downscaling_rate * duration

        # Apply multiplicative scaling
        scaled = weights * scale_factor

        # Threshold very weak connections to zero
        threshold = 0.01
        scaled[np.abs(scaled) < threshold] = 0.0

        return scaled

    def selective_downscale(self,
                            weights: np.ndarray,
                            importance: np.ndarray,
                            duration: float = 1.0) -> np.ndarray:
        """
        Selective downscaling - protect important connections.

        Importance could come from:
        - Recent use
        - Emotional tagging
        - Replay reinforcement
        """
        # Scale less for high-importance connections
        protection = 1.0 - self.downscaling_rate * duration * (1.0 - importance)
        scaled = weights * protection

        return scaled


class MemoryConsolidator:
    """
    Active system consolidation with knowledge graph building.

    Transfer memories from hippocampus (episodic) to cortex (semantic).
    Builds semantic relations from repeated patterns.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Episodic store (hippocampus-like)
        self.episodic_store: List[MemoryTrace] = []

        # Semantic store (cortex-like)
        self.semantic_store: Dict[str, np.ndarray] = {}

        # Knowledge graph: subject -> relation -> objects (with strengths)
        self.knowledge_graph: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Pattern co-occurrence tracker
        self.cooccurrence: Dict[str, Dict[str, int]] = {}

        # Consolidation threshold
        self.consolidation_threshold = 3  # Minimum replays before consolidation

    def attempt_consolidation(self, memory: MemoryTrace) -> Optional[Dict[str, Any]]:
        """Attempt to consolidate episodic memory into semantic."""
        if memory.replay_count < self.consolidation_threshold:
            return None

        if memory.memory_type != MemoryType.EPISODIC:
            return None

        # Extract semantic content (abstraction)
        semantic_content = self._extract_semantic(memory)

        # Find if similar semantic exists
        best_match = None
        best_similarity = 0.0

        for key, existing in self.semantic_store.items():
            similarity = np.dot(semantic_content, existing) / (
                np.linalg.norm(semantic_content) * np.linalg.norm(existing) + 1e-8
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = key

        if best_similarity > 0.8 and best_match:
            # Update existing semantic memory
            self.semantic_store[best_match] = 0.9 * self.semantic_store[best_match] + 0.1 * semantic_content
            action = 'updated'
        else:
            # Create new semantic memory
            key = f"semantic_{len(self.semantic_store)}"
            self.semantic_store[key] = semantic_content
            action = 'created'
            best_match = key

        memory.consolidated = True

        return {
            'action': action,
            'semantic_key': best_match,
            'similarity_to_existing': best_similarity
        }

    def _extract_semantic(self, memory: MemoryTrace) -> np.ndarray:
        """Extract semantic content from episodic memory."""
        semantic = memory.content.copy()
        semantic += np.random.randn(self.dim) * 0.1
        semantic /= (np.linalg.norm(semantic) + 1e-8)
        return semantic

    def consolidate_batch(self, memories: List[MemoryTrace]) -> Dict[str, Any]:
        """Consolidate a batch of memories."""
        results = {
            'attempted': len(memories),
            'consolidated': 0,
            'relations_strengthened': 0,
            'details': []
        }

        for memory in memories:
            result = self.attempt_consolidation(memory)
            if result:
                results['consolidated'] += 1
                results['details'].append(result)

        # Build knowledge graph from co-occurring memories
        relations = self._extract_relations_from_batch(memories)
        results['relations_strengthened'] = relations

        return results

    def _extract_relations_from_batch(self, memories: List[MemoryTrace]) -> int:
        """Extract and strengthen relations from co-occurring memories."""
        strengthened = 0

        # Track co-occurrences in this batch
        mem_keys = []
        for m in memories:
            if m.source_episode:
                mem_keys.append(m.source_episode)

        # Update co-occurrence counts
        for i, key1 in enumerate(mem_keys):
            if key1 not in self.cooccurrence:
                self.cooccurrence[key1] = {}
            for key2 in mem_keys[i+1:]:
                self.cooccurrence[key1][key2] = self.cooccurrence[key1].get(key2, 0) + 1

                # If co-occurred enough times, create/strengthen relation
                if self.cooccurrence[key1][key2] >= 3:
                    self._strengthen_relation(key1, "related_to", key2)
                    strengthened += 1

        return strengthened

    def _strengthen_relation(self, subject: str, relation: str, obj: str, amount: float = 0.1):
        """Strengthen a relation in the knowledge graph."""
        if subject not in self.knowledge_graph:
            self.knowledge_graph[subject] = {}
        if relation not in self.knowledge_graph[subject]:
            self.knowledge_graph[subject][relation] = {}

        current = self.knowledge_graph[subject][relation].get(obj, 0.0)
        self.knowledge_graph[subject][relation][obj] = min(1.0, current + amount)

    def add_relation(self, subject: str, relation: str, obj: str, strength: float = 0.5):
        """Explicitly add a relation to the knowledge graph."""
        self._strengthen_relation(subject, relation, obj, strength)

    def get_relations(self, subject: str) -> Dict[str, Dict[str, float]]:
        """Get all relations for a subject."""
        return self.knowledge_graph.get(subject, {})


class DreamGenerator:
    """
    Generate dreams by recombining memory elements.

    Dreams may serve:
    - Emotional processing
    - Creative problem solving
    - Memory integration
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Dream content buffer
        self.dream_elements: List[np.ndarray] = []

        # Bizarreness control
        self.bizarreness_level = 0.3

    def add_element(self, element: np.ndarray):
        """Add element to dream buffer."""
        self.dream_elements.append(element.copy())

    def generate_dream(self,
                       seed_memories: List[MemoryTrace],
                       emotional_bias: float = 0.0,
                       length: int = 5) -> Dream:
        """
        Generate a dream sequence.

        Dreams recombine elements in novel ways.
        """
        if not seed_memories:
            # Random dream
            content = np.random.randn(self.dim)
            return Dream(
                content=content,
                narrative=["abstract_dream"],
                source_memories=[],
                emotional_tone=0.0,
                bizarreness=1.0
            )

        # Start with seed
        current = seed_memories[0].content.copy()
        narrative = []
        sources = []

        for i in range(length):
            # Mix in other memories
            if len(seed_memories) > 1:
                other = seed_memories[np.random.randint(len(seed_memories))]
                mix_ratio = 0.3 + self.bizarreness_level * np.random.random()
                current = (1 - mix_ratio) * current + mix_ratio * other.content
                sources.append(other.source_episode or f"memory_{i}")

            # Add random element (bizarreness)
            if np.random.random() < self.bizarreness_level:
                noise = np.random.randn(self.dim) * 0.3
                current = current + noise

            # Normalize
            current /= (np.linalg.norm(current) + 1e-8)

            narrative.append(f"dream_segment_{i}")

        # Compute emotional tone from sources
        emotional_tone = np.mean([m.emotional_salience for m in seed_memories])
        emotional_tone += emotional_bias

        # Compute bizarreness
        bizarreness = self._compute_bizarreness(current, seed_memories)

        return Dream(
            content=current,
            narrative=narrative,
            source_memories=sources,
            emotional_tone=float(np.clip(emotional_tone, -1, 1)),
            bizarreness=float(bizarreness)
        )

    def _compute_bizarreness(self,
                              dream_content: np.ndarray,
                              sources: List[MemoryTrace]) -> float:
        """Compute how bizarre the dream is compared to sources."""
        if not sources:
            return 1.0

        # Distance from all source memories
        distances = [
            np.linalg.norm(dream_content - m.content)
            for m in sources
        ]

        avg_distance = np.mean(distances)
        return float(np.clip(avg_distance, 0, 1))


class SleepCycle:
    """
    Simulate sleep cycles with different stages.
    """

    def __init__(self):
        # Typical 90-minute cycle
        self.cycle_duration = 90 * 60  # seconds

        # Stage proportions in a cycle
        self.stage_proportions = {
            SleepStage.N1: 0.05,
            SleepStage.N2: 0.50,
            SleepStage.N3: 0.20,
            SleepStage.REM: 0.25
        }

        # Current stage
        self.current_stage = SleepStage.WAKE
        self.stage_start_time = 0.0

        # Cycle count
        self.cycles_completed = 0

    def start_sleep(self) -> SleepStage:
        """Start sleep session."""
        self.current_stage = SleepStage.N1
        self.stage_start_time = time.time()
        self.cycles_completed = 0
        return self.current_stage

    def advance_stage(self) -> Tuple[SleepStage, float]:
        """Advance to next sleep stage."""
        stage_order = [SleepStage.N1, SleepStage.N2, SleepStage.N3, SleepStage.N2, SleepStage.REM]

        # Find current position in cycle
        try:
            current_idx = stage_order.index(self.current_stage)
            next_idx = (current_idx + 1) % len(stage_order)
        except ValueError:
            next_idx = 0

        # Move to next stage
        self.current_stage = stage_order[next_idx]

        # Compute stage duration
        proportion = self.stage_proportions.get(self.current_stage, 0.25)
        duration = self.cycle_duration * proportion

        self.stage_start_time = time.time()

        # Check for cycle completion
        if self.current_stage == SleepStage.REM:
            self.cycles_completed += 1

        return self.current_stage, duration

    def wake_up(self) -> SleepSession:
        """End sleep session."""
        session = SleepSession(
            start_time=self.stage_start_time - self.cycles_completed * self.cycle_duration,
            duration=self.cycles_completed * self.cycle_duration,
            stages=[]  # Would track actual stages
        )

        self.current_stage = SleepStage.WAKE
        return session


class SleepConsolidationSystem:
    """
    Complete sleep and consolidation system.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Components
        self.replay = HippocampalReplay(dim)
        self.homeostasis = SynapticHomeostasis()
        self.consolidator = MemoryConsolidator(dim)
        self.dream_gen = DreamGenerator(dim)
        self.sleep_cycle = SleepCycle()

        # State
        self.is_sleeping = False
        self.total_sleep_time = 0.0
        self.dreams: List[Dream] = []

        # Weights to be consolidated
        self.weights_to_process: List[np.ndarray] = []

    def add_memory_for_consolidation(self, memory: MemoryTrace):
        """Add memory for later consolidation during sleep."""
        self.replay.add_to_buffer(memory)
        self.consolidator.episodic_store.append(memory)

    def start_sleep(self) -> Dict[str, Any]:
        """Start a sleep session."""
        self.is_sleeping = True
        stage = self.sleep_cycle.start_sleep()

        return {
            'status': 'sleep_started',
            'initial_stage': stage.name
        }

    def sleep_step(self, duration: float = 1.0) -> Dict[str, Any]:
        """Execute one step of sleep processing."""
        if not self.is_sleeping:
            return {'error': 'not_sleeping'}

        results = {
            'stage': self.sleep_cycle.current_stage.name,
            'replay': None,
            'consolidation': None,
            'dream': None,
            'downscaling': None
        }

        # Stage-specific processing
        if self.sleep_cycle.current_stage in [SleepStage.N2, SleepStage.N3]:
            # NREM: Memory replay and consolidation
            selected = self.replay.select_for_replay(num_memories=int(10 * duration))
            results['replay'] = self.replay.replay(selected)

            # Attempt consolidation
            results['consolidation'] = self.consolidator.consolidate_batch(selected)

            # Synaptic downscaling
            if self.weights_to_process:
                for i, weights in enumerate(self.weights_to_process):
                    self.weights_to_process[i] = self.homeostasis.downscale(weights, duration)
                results['downscaling'] = {'processed': len(self.weights_to_process)}

        elif self.sleep_cycle.current_stage == SleepStage.REM:
            # REM: Dreams and emotional processing
            seed_memories = self.replay.select_for_replay(num_memories=5)
            if seed_memories:
                dream = self.dream_gen.generate_dream(seed_memories, length=10)
                self.dreams.append(dream)
                results['dream'] = {
                    'bizarreness': dream.bizarreness,
                    'emotional_tone': dream.emotional_tone,
                    'length': len(dream.narrative)
                }

        # Advance stage
        next_stage, stage_duration = self.sleep_cycle.advance_stage()
        results['next_stage'] = next_stage.name
        results['stage_duration'] = stage_duration

        self.total_sleep_time += duration

        return results

    def wake_up(self) -> Dict[str, Any]:
        """End sleep session."""
        if not self.is_sleeping:
            return {'error': 'not_sleeping'}

        session = self.sleep_cycle.wake_up()
        self.is_sleeping = False

        return {
            'status': 'awake',
            'session_duration': session.duration,
            'cycles_completed': self.sleep_cycle.cycles_completed,
            'dreams_count': len(self.dreams),
            'memories_consolidated': len([
                m for m in self.consolidator.episodic_store if m.consolidated
            ])
        }

    def quick_nap(self, duration: float = 20.0) -> Dict[str, Any]:
        """Quick nap for immediate memory boost."""
        self.start_sleep()

        # Mostly N2 sleep
        self.sleep_cycle.current_stage = SleepStage.N2

        results = []
        steps = int(duration / 5)
        for _ in range(steps):
            step_result = self.sleep_step(5.0)
            results.append(step_result)

        wake_result = self.wake_up()

        return {
            'nap_duration': duration,
            'replay_summary': sum(r['replay']['num_replayed'] for r in results if r['replay']),
            'wake_result': wake_result
        }

    def get_state(self) -> Dict[str, Any]:
        """Get sleep system state."""
        return {
            'is_sleeping': self.is_sleeping,
            'current_stage': self.sleep_cycle.current_stage.name,
            'total_sleep_time': self.total_sleep_time,
            'replay_buffer_size': len(self.replay.replay_buffer),
            'episodic_memories': len(self.consolidator.episodic_store),
            'semantic_memories': len(self.consolidator.semantic_store),
            'dreams_recorded': len(self.dreams),
            'cycles_completed': self.sleep_cycle.cycles_completed
        }

    def get_consolidation_queue_size(self) -> int:
        """Get number of memories waiting for consolidation."""
        return len(self.replay.replay_buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats for the sleep system."""
        consolidated = sum(1 for m in self.consolidator.episodic_store if m.consolidated)
        return {
            'total_consolidated': consolidated,
            'episodic_memories': len(self.consolidator.episodic_store),
            'semantic_memories': len(self.consolidator.semantic_store),
            'knowledge_graph_nodes': len(self.consolidator.knowledge_graph),
            'replay_buffer': len(self.replay.replay_buffer),
            'dreams_recorded': len(self.dreams),
            'total_sleep_time': self.total_sleep_time
        }

    def run_consolidation_cycle(self, duration: float = 10.0):
        """Run a complete consolidation cycle (simulated sleep)."""
        self.start_sleep()

        # Cycle through stages
        steps = max(1, int(duration / 2))
        for _ in range(steps):
            self.sleep_step(2.0)

        return self.wake_up()
