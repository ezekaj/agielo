"""
Active Learning System
======================

Implements intrinsic motivation for learning:
- Learn what you DON'T know (low confidence)
- Learn what you're CURIOUS about (high interest)
- Prioritize topics at the edge of knowledge
- Explore novel areas using RND (Random Network Distillation)

Based on:
- Intrinsic Motivation (Oudeyer & Kaplan, 2007)
- Curiosity-driven Learning (Pathak et al., 2017)
- Optimal Learning Theory (MacKay, 1992)
- Random Network Distillation (Burda et al., 2018)
"""

import numpy as np
import json
import os
import time
import atexit
import weakref
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import threading

# Import constants from config
from config.constants import (
    EXPOSURE_DECAY_RATE,
    UNCERTAINTY_PRIORITY_WEIGHT,
    CURIOSITY_PRIORITY_WEIGHT,
    NOVELTY_PRIORITY_WEIGHT,
    DEFAULT_TOPIC_CONFIDENCE,
    DEFAULT_TOPIC_CURIOSITY,
    RND_CURIOSITY_WEIGHT,
    RND_HIGH_CURIOSITY_THRESHOLD,
    LOW_CURIOSITY_THRESHOLD,
    KNOWN_TOPIC_PRIORITY_FACTOR,
    DEFAULT_CURIOSITY_BOOST,
    HIGH_CONFIDENCE_THRESHOLD,
    CURIOSITY_DECAY_WEIGHT,
    CURIOSITY_UPDATE_WEIGHT,
)

# Try to import RND curiosity module
try:
    from integrations.rnd_curiosity import RNDCuriosity, DEFAULT_EMBEDDING_DIM
    RND_AVAILABLE = True
except ImportError:
    RND_AVAILABLE = False
    DEFAULT_EMBEDDING_DIM = 384

# Try to import semantic embeddings for topic-to-embedding conversion
try:
    from integrations.semantic_embeddings import get_embedder
    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False


@dataclass
class Topic:
    """A topic the AI knows about or is curious about."""
    name: str
    confidence: float = DEFAULT_TOPIC_CONFIDENCE  # How well we know it (0=unknown, 1=expert)
    curiosity: float = DEFAULT_TOPIC_CURIOSITY    # How interested we are (0=boring, 1=fascinating)
    exposure_count: int = 0      # How many times we've seen it
    success_count: int = 0       # How many times we answered correctly
    last_seen: float = 0.0       # Timestamp
    related_topics: List[str] = field(default_factory=list)

    @property
    def learning_priority(self) -> float:
        """
        Calculate learning priority.

        High priority = low confidence + high curiosity
        This is the "zone of proximal development"
        """
        # Optimal learning happens at the edge of knowledge
        # Too easy (high confidence) = boring
        # Too hard (low confidence + low exposure) = frustrating

        # Sweet spot: moderate confidence, high curiosity
        uncertainty = 1.0 - self.confidence
        novelty_bonus = 1.0 / (1.0 + self.exposure_count * EXPOSURE_DECAY_RATE)

        priority = (uncertainty * UNCERTAINTY_PRIORITY_WEIGHT +  # Want to learn unknowns
                   self.curiosity * CURIOSITY_PRIORITY_WEIGHT +   # Want to learn interesting things
                   novelty_bonus * NOVELTY_PRIORITY_WEIGHT)       # Slight preference for new topics

        return min(1.0, priority)

    @property
    def accuracy(self) -> float:
        """Success rate for this topic."""
        if self.exposure_count == 0:
            return 0.5
        return self.success_count / self.exposure_count


@dataclass
class LearningEvent:
    """Record of a learning interaction."""
    topic: str
    timestamp: float
    was_correct: bool
    confidence_before: float
    confidence_after: float
    curiosity_delta: float = 0.0
    rnd_curiosity: float = 0.0  # RND curiosity score at time of event
    is_novel: bool = False  # Whether this was a novel topic discovery


# Track instances for atexit cleanup using weak references
_active_learner_instances: List[weakref.ref] = []


def _cleanup_all_instances() -> None:
    """Cleanup function called at program exit to save all ActiveLearner state."""
    for ref in _active_learner_instances:
        instance = ref()
        if instance is not None:
            instance.cleanup()


# Register the cleanup function with atexit
atexit.register(_cleanup_all_instances)


def _register_instance(instance: 'ActiveLearner') -> None:
    """Register an ActiveLearner instance for cleanup on exit."""
    _active_learner_instances.append(weakref.ref(instance))


class ActiveLearner:
    """
    Active learning system that prioritizes what to learn.

    Key principles:
    1. Learn at the edge of knowledge (not too easy, not too hard)
    2. Follow curiosity (intrinsic motivation)
    3. Reduce uncertainty where it matters
    4. Build on existing knowledge (scaffolding)
    5. Explore novel areas using RND curiosity
    """

    def __init__(self, storage_path: Optional[str] = None, use_rnd: bool = True) -> None:
        self.storage_path = Path(storage_path or os.path.expanduser("~/.cognitive_ai_knowledge/active_learning"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Topic knowledge base
        self.topics: Dict[str, Topic] = {}

        # Learning history
        self.history: List[LearningEvent] = []

        # Curiosity model - what drives interest
        self.curiosity_drivers = {
            'novelty': 0.3,        # New things are interesting
            'surprise': 0.3,       # Unexpected outcomes
            'complexity': 0.2,     # Moderately complex things
            'relevance': 0.2,      # Related to known topics
        }

        # RND curiosity integration
        self.use_rnd = use_rnd and RND_AVAILABLE
        self.rnd_curiosity: Optional[RNDCuriosity] = None
        self._rnd_weight = RND_CURIOSITY_WEIGHT  # Weight for RND curiosity in combined score

        # RND tracking statistics
        self._rnd_stats = {
            'curiosity_history': [],  # List of (timestamp, avg_curiosity) tuples
            'novel_discoveries': 0,    # Count of novel topic discoveries
            'total_rnd_updates': 0,    # Total predictor updates
            'curiosity_decay_rate': 0.0,  # Measured decay over time
        }

        if self.use_rnd:
            try:
                rnd_storage = self.storage_path / 'rnd_curiosity'
                self.rnd_curiosity = RNDCuriosity(
                    input_dim=DEFAULT_EMBEDDING_DIM,
                    storage_path=str(rnd_storage)
                )
                print(f"[ActiveLearner] RND curiosity enabled (dim={DEFAULT_EMBEDDING_DIM})")
            except Exception as e:
                print(f"[ActiveLearner] Failed to initialize RND: {e}")
                self.use_rnd = False
                self.rnd_curiosity = None

        # Load saved state
        self._load_state()

        # Thread lock
        self._lock = threading.Lock()

        # Register for cleanup on program exit
        _register_instance(self)

    def should_learn(self, topic: str, content: str = "") -> Tuple[bool, float, str]:
        """
        Decide if we should learn about this topic.

        Combines traditional curiosity (confidence-based) with RND curiosity
        (novelty-based) for a comprehensive learning decision.

        Returns:
            (should_learn, priority, reason)
        """
        with self._lock:
            # Get or create topic
            if topic not in self.topics:
                self.topics[topic] = Topic(name=topic)
                # New topic - check RND curiosity for extra signal
                rnd_curiosity = self._compute_rnd_curiosity(topic, content)
                if rnd_curiosity > 0.7:
                    self._record_novel_discovery()
                    return True, 0.95, "new_topic_rnd_novel"
                return True, 0.9, "new_topic"

            t = self.topics[topic]
            base_priority = t.learning_priority

            # Compute RND curiosity for this topic
            rnd_curiosity = self._compute_rnd_curiosity(topic, content)

            # Combine existing priority with RND curiosity
            # final_score = 0.5 * existing + 0.5 * rnd
            combined_priority = (1.0 - self._rnd_weight) * base_priority + self._rnd_weight * rnd_curiosity

            # Already expert? Low priority unless RND says it's novel
            if t.confidence > 0.9:
                if rnd_curiosity > 0.8:
                    # Expert but RND sees novelty - might be new aspect
                    return True, combined_priority, "expert_but_rnd_novel"
                return False, 0.1, "already_expert"

            # Low traditional curiosity? Check RND curiosity as override
            if t.curiosity < 0.3 and t.confidence > 0.5:
                if rnd_curiosity > 0.7:
                    # Low traditional curiosity but RND sees novelty
                    return True, combined_priority, "rnd_override_low_curiosity"
                return False, 0.2, "low_curiosity"

            # Perfect zone: uncertain but curious (either traditional or RND)
            if t.confidence < 0.7 and (t.curiosity > LOW_CURIOSITY_THRESHOLD or rnd_curiosity > RND_HIGH_CURIOSITY_THRESHOLD):
                reason = "curious_uncertain"
                if rnd_curiosity > RND_HIGH_CURIOSITY_THRESHOLD and t.curiosity <= LOW_CURIOSITY_THRESHOLD:
                    reason = "rnd_curious_uncertain"
                return True, combined_priority, reason

            # Recently seen? Lower priority but RND can override
            time_since = datetime.now().timestamp() - t.last_seen
            if time_since < 300:  # 5 minutes
                if rnd_curiosity > 0.8:
                    # RND still sees novelty despite recent exposure
                    return True, combined_priority * 0.7, "recently_seen_rnd_override"
                return False, combined_priority * 0.5, "recently_seen"

            # Default: learn if priority is high enough
            return combined_priority > 0.4, combined_priority, "priority_based"

    def _compute_rnd_curiosity(self, topic: str, content: str = "") -> float:
        """
        Compute RND curiosity for a topic.

        Args:
            topic: Topic name
            content: Optional content to enrich embedding

        Returns:
            RND curiosity score (0-1), or 0.5 if RND unavailable
        """
        if not self.use_rnd or self.rnd_curiosity is None:
            return 0.5  # Neutral score when RND unavailable

        try:
            # Get embedding for topic
            embedding = self._get_topic_embedding(topic, content)
            if embedding is None:
                return 0.5

            # Compute RND curiosity
            return self.rnd_curiosity.compute_curiosity(embedding)
        except Exception as e:
            # Fail silently, return neutral score
            return 0.5

    def _get_topic_embedding(self, topic: str, content: str = "") -> Optional[np.ndarray]:
        """
        Get embedding vector for a topic.

        Args:
            topic: Topic name
            content: Optional content to enrich embedding

        Returns:
            Embedding array or None if unavailable
        """
        # Combine topic and content for richer embedding
        text = topic
        if content:
            text = f"{topic}: {content[:200]}"  # Limit content length

        # Try semantic embedder first
        if EMBEDDER_AVAILABLE:
            try:
                embedder = get_embedder()
                return embedder.embed(text)
            except Exception:
                pass

        # Fallback: simple hash-based embedding
        import hashlib
        embedding = np.zeros(DEFAULT_EMBEDDING_DIM, dtype=np.float32)
        text_hash = hashlib.sha256(text.encode()).digest()
        for i, byte in enumerate(text_hash):
            pos = i % DEFAULT_EMBEDDING_DIM
            embedding[pos] = byte / 255.0 - 0.5

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def _record_novel_discovery(self) -> None:
        """Record a novel topic discovery for statistics."""
        self._rnd_stats['novel_discoveries'] += 1

    def record_exposure(self, topic: str, was_successful: bool,
                       surprise_level: float = 0.5, complexity: float = 0.5,
                       content: str = "") -> None:
        """
        Record that we encountered/learned about a topic.

        Args:
            topic: Topic name
            was_successful: Did we answer correctly / learn successfully?
            surprise_level: How surprising was the outcome (0-1)
            complexity: How complex was the content (0-1)
            content: Optional content for RND embedding
        """
        with self._lock:
            is_novel = topic not in self.topics
            if is_novel:
                self.topics[topic] = Topic(name=topic)

            t = self.topics[topic]
            old_confidence = t.confidence

            # Update exposure
            t.exposure_count += 1
            t.last_seen = datetime.now().timestamp()

            if was_successful:
                t.success_count += 1
                # Increase confidence
                t.confidence = min(1.0, t.confidence + 0.1 * (1 - t.confidence))
            else:
                # Decrease confidence but not too much
                t.confidence = max(0.1, t.confidence - 0.05)

            # Update curiosity based on surprise
            if surprise_level > 0.6:
                # Surprising outcomes increase curiosity
                t.curiosity = min(1.0, t.curiosity + 0.1)
            elif surprise_level < 0.3:
                # Predictable outcomes decrease curiosity
                t.curiosity = max(0.1, t.curiosity - 0.05)

            # Complexity affects curiosity
            # Sweet spot is moderate complexity
            complexity_interest = 1.0 - abs(complexity - LOW_CURIOSITY_THRESHOLD) * 2
            t.curiosity = CURIOSITY_DECAY_WEIGHT * t.curiosity + CURIOSITY_UPDATE_WEIGHT * complexity_interest

            # Compute RND curiosity before updating predictor
            rnd_curiosity_score = 0.0
            if self.use_rnd and self.rnd_curiosity is not None:
                rnd_curiosity_score = self._compute_rnd_curiosity(topic, content)

            # Update RND predictor after successful learning
            if was_successful and self.use_rnd and self.rnd_curiosity is not None:
                self._update_rnd_predictor_unlocked(topic, content)

            # Record event
            event = LearningEvent(
                topic=topic,
                timestamp=datetime.now().timestamp(),
                was_correct=was_successful,
                confidence_before=old_confidence,
                confidence_after=t.confidence,
                curiosity_delta=t.curiosity - 0.5,
                rnd_curiosity=rnd_curiosity_score,
                is_novel=is_novel
            )
            self.history.append(event)

            # Track novel discoveries
            if is_novel and rnd_curiosity_score > 0.7:
                self._rnd_stats['novel_discoveries'] += 1

            # Update curiosity decay tracking
            self._update_curiosity_decay_stats()

            # Keep history bounded
            if len(self.history) > 1000:
                self.history = self.history[-500:]

            # Auto-save periodically
            if len(self.history) % 50 == 0:
                self._save_state()

    def _update_rnd_predictor_unlocked(self, topic: str, content: str = "") -> None:
        """Update RND predictor after successful learning (without lock)."""
        if not self.use_rnd or self.rnd_curiosity is None:
            return

        try:
            embedding = self._get_topic_embedding(topic, content)
            if embedding is not None:
                # Update predictor to reduce curiosity for this area
                self.rnd_curiosity.update_predictor(embedding, n_steps=3)
                self._rnd_stats['total_rnd_updates'] += 1
        except Exception:
            pass

    def _update_curiosity_decay_stats(self) -> None:
        """Update curiosity decay tracking statistics."""
        # Only update periodically to avoid overhead
        if len(self.history) % 10 != 0:
            return

        # Compute average RND curiosity over recent history
        recent_events = [e for e in self.history[-50:] if e.rnd_curiosity > 0]
        if recent_events:
            avg_curiosity = np.mean([e.rnd_curiosity for e in recent_events])
            timestamp = datetime.now().timestamp()
            self._rnd_stats['curiosity_history'].append((timestamp, avg_curiosity))

            # Keep bounded
            if len(self._rnd_stats['curiosity_history']) > 100:
                self._rnd_stats['curiosity_history'] = self._rnd_stats['curiosity_history'][-50:]

            # Compute decay rate if enough data
            if len(self._rnd_stats['curiosity_history']) >= 10:
                recent = self._rnd_stats['curiosity_history'][-10:]
                if recent[-1][0] != recent[0][0]:  # Avoid division by zero
                    time_delta = recent[-1][0] - recent[0][0]
                    curiosity_delta = recent[-1][1] - recent[0][1]
                    self._rnd_stats['curiosity_decay_rate'] = curiosity_delta / time_delta

    def get_learning_recommendations(self, k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Get top-k topics we should learn about.

        Returns:
            List of (topic, priority, reason) tuples
        """
        with self._lock:
            recommendations = []

            for name, topic in self.topics.items():
                priority = topic.learning_priority

                if topic.confidence > 0.9:
                    reason = "refresh_expert"
                    priority *= KNOWN_TOPIC_PRIORITY_FACTOR
                elif topic.curiosity > 0.7:
                    reason = "high_curiosity"
                elif topic.confidence < 0.3:
                    reason = "fill_gap"
                else:
                    reason = "balanced_learning"

                recommendations.append((name, priority, reason))

            # Sort by priority
            recommendations.sort(key=lambda x: x[1], reverse=True)

            return recommendations[:k]

    def get_related_topics(self, topic: str, k: int = 3) -> List[str]:
        """Find topics related to a given topic (for scaffolding)."""
        if topic not in self.topics:
            return []

        t = self.topics[topic]

        # First, check explicit relations
        if t.related_topics:
            return t.related_topics[:k]

        # Find topics seen around the same time
        related = []
        topic_times = [(name, top.last_seen) for name, top in self.topics.items()
                      if name != topic and top.last_seen > 0]

        if not topic_times:
            return []

        target_time = t.last_seen
        topic_times.sort(key=lambda x: abs(x[1] - target_time))

        return [name for name, _ in topic_times[:k]]

    def add_topic_relation(self, topic1: str, topic2: str) -> None:
        """Record that two topics are related."""
        with self._lock:
            for topic in [topic1, topic2]:
                if topic not in self.topics:
                    self.topics[topic] = Topic(name=topic)

            if topic2 not in self.topics[topic1].related_topics:
                self.topics[topic1].related_topics.append(topic2)
            if topic1 not in self.topics[topic2].related_topics:
                self.topics[topic2].related_topics.append(topic1)

    def boost_curiosity(self, topic: str, amount: float = DEFAULT_CURIOSITY_BOOST) -> None:
        """Manually boost curiosity for a topic (e.g., user showed interest)."""
        with self._lock:
            if topic not in self.topics:
                self.topics[topic] = Topic(name=topic)
            self.topics[topic].curiosity = min(1.0, self.topics[topic].curiosity + amount)

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics including RND curiosity metrics."""
        with self._lock:
            if not self.topics:
                return {'total_topics': 0, 'avg_confidence': 0, 'avg_curiosity': 0}

            confidences = [t.confidence for t in self.topics.values()]
            curiosities = [t.curiosity for t in self.topics.values()]

            recent_events = [e for e in self.history[-100:]]
            recent_accuracy = sum(1 for e in recent_events if e.was_correct) / max(1, len(recent_events))

            stats = {
                'total_topics': len(self.topics),
                'avg_confidence': float(np.mean(confidences)),
                'avg_curiosity': float(np.mean(curiosities)),
                'recent_accuracy': float(recent_accuracy),
                'total_learning_events': len(self.history),
                'high_curiosity_topics': sum(1 for t in self.topics.values() if t.curiosity > 0.7),
                'low_confidence_topics': sum(1 for t in self.topics.values() if t.confidence < 0.3),
            }

            # Add RND stats if available
            if self.use_rnd:
                stats.update(self.get_rnd_stats_unlocked())

            return stats

    def get_rnd_stats(self) -> Dict[str, Any]:
        """Get RND curiosity-specific statistics."""
        with self._lock:
            return self.get_rnd_stats_unlocked()

    def get_rnd_stats_unlocked(self) -> Dict[str, Any]:
        """Get RND curiosity stats without lock (for internal use)."""
        if not self.use_rnd or self.rnd_curiosity is None:
            return {
                'rnd_enabled': False,
                'rnd_available': RND_AVAILABLE,
            }

        # Get recent RND curiosity values from history
        recent_rnd = [e.rnd_curiosity for e in self.history[-50:] if e.rnd_curiosity > 0]
        avg_rnd_curiosity = float(np.mean(recent_rnd)) if recent_rnd else 0.5

        # Count novel discoveries
        novel_count = sum(1 for e in self.history if e.is_novel and e.rnd_curiosity > 0.7)

        # Compute novelty discovery rate (per 100 events)
        if len(self.history) > 0:
            novel_rate = (self._rnd_stats['novel_discoveries'] / len(self.history)) * 100
        else:
            novel_rate = 0.0

        # Get exploration stats from RND module
        rnd_exploration_stats = {}
        try:
            rnd_exploration_stats = self.rnd_curiosity.get_exploration_stats()
        except Exception:
            pass

        return {
            'rnd_enabled': True,
            'rnd_weight': self._rnd_weight,
            'avg_rnd_curiosity': avg_rnd_curiosity,
            'novel_discoveries': self._rnd_stats['novel_discoveries'],
            'novelty_rate_per_100': float(novel_rate),
            'total_rnd_updates': self._rnd_stats['total_rnd_updates'],
            'curiosity_decay_rate': float(self._rnd_stats['curiosity_decay_rate']),
            'rnd_exploration': rnd_exploration_stats,
        }

    def set_rnd_weight(self, weight: float) -> None:
        """
        Set the weight for RND curiosity in combined scoring.

        Args:
            weight: Weight between 0 and 1 (default 0.5)
        """
        self._rnd_weight = max(0.0, min(1.0, weight))

    def get_low_confidence_topics(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get topics with low confidence (gaps in knowledge).

        Used by AutonomousWorker to identify what to learn.
        """
        with self._lock:
            low_conf = []
            for name, topic in self.topics.items():
                if topic.confidence < 0.5:
                    low_conf.append({
                        'name': name,
                        'confidence': topic.confidence,
                        'curiosity': topic.curiosity,
                        'priority': topic.learning_priority
                    })

            # Sort by priority (highest first)
            low_conf.sort(key=lambda x: x['priority'], reverse=True)
            return low_conf[:n]

    def get_high_curiosity_areas(self, n: int = 5) -> List[str]:
        """
        Get areas with high curiosity (novel/unexplored).

        Used by AutonomousWorker for exploration.
        """
        with self._lock:
            high_curiosity = []
            for name, topic in self.topics.items():
                if topic.curiosity > 0.6:
                    high_curiosity.append((name, topic.curiosity))

            # Sort by curiosity (highest first)
            high_curiosity.sort(key=lambda x: x[1], reverse=True)
            return [name for name, _ in high_curiosity[:n]]

    def record_learning(self, topic: str, was_correct: bool,
                        confidence_before: float, confidence_after: float) -> None:
        """
        Record a learning event.

        Used by AutonomousWorker to track learning progress.
        """
        with self._lock:
            # Create topic if doesn't exist
            if topic not in self.topics:
                self.topics[topic] = Topic(name=topic)

            t = self.topics[topic]
            t.exposure_count += 1
            if was_correct:
                t.success_count += 1
            t.confidence = confidence_after
            t.last_seen = time.time()

            # Record event
            event = LearningEvent(
                topic=topic,
                timestamp=time.time(),
                was_correct=was_correct,
                confidence_before=confidence_before,
                confidence_after=confidence_after,
                rnd_curiosity=self._compute_rnd_curiosity(topic, "")
            )
            self.history.append(event)

            # Save periodically
            if len(self.history) % 10 == 0:
                self._save_state()

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            state = {
                'topics': {
                    name: {
                        'name': t.name,
                        'confidence': t.confidence,
                        'curiosity': t.curiosity,
                        'exposure_count': t.exposure_count,
                        'success_count': t.success_count,
                        'last_seen': t.last_seen,
                        'related_topics': t.related_topics,
                    }
                    for name, t in self.topics.items()
                },
                'history': [
                    {
                        'topic': e.topic,
                        'timestamp': e.timestamp,
                        'was_correct': e.was_correct,
                        'confidence_before': e.confidence_before,
                        'confidence_after': e.confidence_after,
                        'rnd_curiosity': e.rnd_curiosity,
                        'is_novel': e.is_novel,
                    }
                    for e in self.history[-200:]  # Keep last 200
                ],
                'rnd_stats': {
                    'curiosity_history': self._rnd_stats['curiosity_history'][-50:],
                    'novel_discoveries': self._rnd_stats['novel_discoveries'],
                    'total_rnd_updates': self._rnd_stats['total_rnd_updates'],
                    'curiosity_decay_rate': self._rnd_stats['curiosity_decay_rate'],
                },
                'rnd_weight': self._rnd_weight,
            }

            with open(self.storage_path / 'state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[ActiveLearner] Save error: {e}")

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.storage_path / 'state.json'
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Load topics
            for name, data in state.get('topics', {}).items():
                self.topics[name] = Topic(
                    name=data['name'],
                    confidence=data['confidence'],
                    curiosity=data['curiosity'],
                    exposure_count=data['exposure_count'],
                    success_count=data['success_count'],
                    last_seen=data['last_seen'],
                    related_topics=data.get('related_topics', [])
                )

            # Load history
            for data in state.get('history', []):
                self.history.append(LearningEvent(
                    topic=data['topic'],
                    timestamp=data['timestamp'],
                    was_correct=data['was_correct'],
                    confidence_before=data['confidence_before'],
                    confidence_after=data['confidence_after'],
                    rnd_curiosity=data.get('rnd_curiosity', 0.0),
                    is_novel=data.get('is_novel', False),
                ))

            # Load RND stats
            rnd_stats = state.get('rnd_stats', {})
            if rnd_stats:
                self._rnd_stats['curiosity_history'] = rnd_stats.get('curiosity_history', [])
                self._rnd_stats['novel_discoveries'] = rnd_stats.get('novel_discoveries', 0)
                self._rnd_stats['total_rnd_updates'] = rnd_stats.get('total_rnd_updates', 0)
                self._rnd_stats['curiosity_decay_rate'] = rnd_stats.get('curiosity_decay_rate', 0.0)

            # Load RND weight
            self._rnd_weight = state.get('rnd_weight', 0.5)

        except Exception as e:
            print(f"[ActiveLearner] Load error: {e}")

    def cleanup(self) -> None:
        """
        Cleanup resources and save state on exit.

        This method is registered with atexit to ensure state is saved
        when the program terminates. It saves:
        - Topic knowledge base
        - Learning history
        - RND statistics
        - RND model state (if RND is enabled)
        """
        try:
            with self._lock:
                # Save active learner state
                self._save_state()

                # Save RND curiosity state if enabled
                if self.use_rnd and self.rnd_curiosity is not None:
                    try:
                        self.rnd_curiosity._save_state()
                    except Exception as e:
                        print(f"[ActiveLearner] RND cleanup error: {e}")

                print(f"[ActiveLearner] Cleanup complete: saved {len(self.topics)} topics, {len(self.history)} events")
        except Exception as e:
            print(f"[ActiveLearner] Cleanup error: {e}")


class CurriculumLearner:
    """
    Curriculum learning - structured progression from simple to complex.

    Builds on ActiveLearner by organizing topics into a learning path.
    """

    def __init__(self, active_learner: ActiveLearner) -> None:
        self.learner = active_learner

        # Topic dependencies: topic -> prerequisites
        self.prerequisites: Dict[str, List[str]] = {}

        # Difficulty levels
        self.difficulty: Dict[str, float] = {}

    def set_prerequisites(self, topic: str, prereqs: List[str]) -> None:
        """Set prerequisites for a topic."""
        self.prerequisites[topic] = prereqs

    def set_difficulty(self, topic: str, difficulty: float) -> None:
        """Set difficulty level (0-1) for a topic."""
        self.difficulty[topic] = difficulty

    def is_ready_for(self, topic: str) -> Tuple[bool, List[str]]:
        """
        Check if we're ready to learn a topic.

        Returns:
            (is_ready, missing_prereqs)
        """
        prereqs = self.prerequisites.get(topic, [])
        missing = []

        for prereq in prereqs:
            if prereq in self.learner.topics:
                if self.learner.topics[prereq].confidence < 0.5:
                    missing.append(prereq)
            else:
                missing.append(prereq)

        return len(missing) == 0, missing

    def get_next_topic(self) -> Optional[str]:
        """Get the next topic we should learn based on curriculum."""
        candidates = []

        for topic in self.learner.topics:
            ready, missing = self.is_ready_for(topic)
            if not ready:
                continue

            t = self.learner.topics[topic]
            if t.confidence >= HIGH_CONFIDENCE_THRESHOLD:  # Already know it
                continue

            # Score based on difficulty appropriateness
            difficulty = self.difficulty.get(topic, 0.5)
            current_level = t.confidence

            # Optimal difficulty is slightly above current level
            difficulty_match = 1.0 - abs(difficulty - current_level - 0.2)

            score = t.learning_priority * 0.5 + difficulty_match * 0.5
            candidates.append((topic, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


# Global instance
_active_learner: Optional[ActiveLearner] = None


def get_active_learner() -> ActiveLearner:
    """Get the global active learner instance."""
    global _active_learner
    if _active_learner is None:
        _active_learner = ActiveLearner()
    return _active_learner


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("ACTIVE LEARNING TEST")
    print("=" * 60)

    learner = ActiveLearner("/tmp/test_active_learning")

    # Simulate learning sessions
    topics = [
        ("python", True, 0.3),    # Know it well
        ("python", True, 0.2),
        ("machine learning", True, 0.7),  # Surprising success
        ("machine learning", False, 0.8),  # Surprising failure
        ("quantum computing", False, 0.9),  # Don't know, very surprising
        ("quantum computing", False, 0.8),
        ("cooking", True, 0.2),   # Know it, boring
        ("cooking", True, 0.1),
    ]

    for topic, success, surprise in topics:
        learner.record_exposure(topic, success, surprise)
        should, priority, reason = learner.should_learn(topic)
        print(f"  {topic}: should_learn={should}, priority={priority:.2f}, reason={reason}")

    print("\n--- Learning Recommendations ---")
    recs = learner.get_learning_recommendations(k=5)
    for topic, priority, reason in recs:
        t = learner.topics[topic]
        print(f"  {topic}: priority={priority:.2f}, confidence={t.confidence:.2f}, curiosity={t.curiosity:.2f}")

    print("\n--- Statistics ---")
    stats = learner.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
