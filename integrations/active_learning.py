"""
Active Learning System
======================

Implements intrinsic motivation for learning:
- Learn what you DON'T know (low confidence)
- Learn what you're CURIOUS about (high interest)
- Prioritize topics at the edge of knowledge

Based on:
- Intrinsic Motivation (Oudeyer & Kaplan, 2007)
- Curiosity-driven Learning (Pathak et al., 2017)
- Optimal Learning Theory (MacKay, 1992)
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import threading


@dataclass
class Topic:
    """A topic the AI knows about or is curious about."""
    name: str
    confidence: float = 0.5      # How well we know it (0=unknown, 1=expert)
    curiosity: float = 0.5       # How interested we are (0=boring, 1=fascinating)
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
        novelty_bonus = 1.0 / (1.0 + self.exposure_count * 0.1)

        priority = (uncertainty * 0.4 +          # Want to learn unknowns
                   self.curiosity * 0.4 +         # Want to learn interesting things
                   novelty_bonus * 0.2)           # Slight preference for new topics

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


class ActiveLearner:
    """
    Active learning system that prioritizes what to learn.

    Key principles:
    1. Learn at the edge of knowledge (not too easy, not too hard)
    2. Follow curiosity (intrinsic motivation)
    3. Reduce uncertainty where it matters
    4. Build on existing knowledge (scaffolding)
    """

    def __init__(self, storage_path: str = None):
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

        # Load saved state
        self._load_state()

        # Thread lock
        self._lock = threading.Lock()

    def should_learn(self, topic: str, content: str = "") -> Tuple[bool, float, str]:
        """
        Decide if we should learn about this topic.

        Returns:
            (should_learn, priority, reason)
        """
        with self._lock:
            # Get or create topic
            if topic not in self.topics:
                self.topics[topic] = Topic(name=topic)
                return True, 0.9, "new_topic"

            t = self.topics[topic]
            priority = t.learning_priority

            # Already expert? Low priority
            if t.confidence > 0.9:
                return False, 0.1, "already_expert"

            # Low curiosity? Skip unless very uncertain
            if t.curiosity < 0.3 and t.confidence > 0.5:
                return False, 0.2, "low_curiosity"

            # Perfect zone: uncertain but curious
            if t.confidence < 0.7 and t.curiosity > 0.5:
                return True, priority, "curious_uncertain"

            # Recently seen? Lower priority
            time_since = datetime.now().timestamp() - t.last_seen
            if time_since < 300:  # 5 minutes
                return False, priority * 0.5, "recently_seen"

            # Default: learn if priority is high enough
            return priority > 0.4, priority, "priority_based"

    def record_exposure(self, topic: str, was_successful: bool,
                       surprise_level: float = 0.5, complexity: float = 0.5):
        """
        Record that we encountered/learned about a topic.

        Args:
            topic: Topic name
            was_successful: Did we answer correctly / learn successfully?
            surprise_level: How surprising was the outcome (0-1)
            complexity: How complex was the content (0-1)
        """
        with self._lock:
            if topic not in self.topics:
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
            complexity_interest = 1.0 - abs(complexity - 0.5) * 2
            t.curiosity = 0.9 * t.curiosity + 0.1 * complexity_interest

            # Record event
            event = LearningEvent(
                topic=topic,
                timestamp=datetime.now().timestamp(),
                was_correct=was_successful,
                confidence_before=old_confidence,
                confidence_after=t.confidence,
                curiosity_delta=t.curiosity - 0.5
            )
            self.history.append(event)

            # Keep history bounded
            if len(self.history) > 1000:
                self.history = self.history[-500:]

            # Auto-save periodically
            if len(self.history) % 50 == 0:
                self._save_state()

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
                    priority *= 0.3
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

    def add_topic_relation(self, topic1: str, topic2: str):
        """Record that two topics are related."""
        with self._lock:
            for topic in [topic1, topic2]:
                if topic not in self.topics:
                    self.topics[topic] = Topic(name=topic)

            if topic2 not in self.topics[topic1].related_topics:
                self.topics[topic1].related_topics.append(topic2)
            if topic1 not in self.topics[topic2].related_topics:
                self.topics[topic2].related_topics.append(topic1)

    def boost_curiosity(self, topic: str, amount: float = 0.2):
        """Manually boost curiosity for a topic (e.g., user showed interest)."""
        with self._lock:
            if topic not in self.topics:
                self.topics[topic] = Topic(name=topic)
            self.topics[topic].curiosity = min(1.0, self.topics[topic].curiosity + amount)

    def get_stats(self) -> Dict:
        """Get learning statistics."""
        with self._lock:
            if not self.topics:
                return {'total_topics': 0, 'avg_confidence': 0, 'avg_curiosity': 0}

            confidences = [t.confidence for t in self.topics.values()]
            curiosities = [t.curiosity for t in self.topics.values()]

            recent_events = [e for e in self.history[-100:]]
            recent_accuracy = sum(1 for e in recent_events if e.was_correct) / max(1, len(recent_events))

            return {
                'total_topics': len(self.topics),
                'avg_confidence': np.mean(confidences),
                'avg_curiosity': np.mean(curiosities),
                'recent_accuracy': recent_accuracy,
                'total_learning_events': len(self.history),
                'high_curiosity_topics': sum(1 for t in self.topics.values() if t.curiosity > 0.7),
                'low_confidence_topics': sum(1 for t in self.topics.values() if t.confidence < 0.3),
            }

    def _save_state(self):
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
                    }
                    for e in self.history[-200:]  # Keep last 200
                ]
            }

            with open(self.storage_path / 'state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[ActiveLearner] Save error: {e}")

    def _load_state(self):
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
                ))

        except Exception as e:
            print(f"[ActiveLearner] Load error: {e}")


class CurriculumLearner:
    """
    Curriculum learning - structured progression from simple to complex.

    Builds on ActiveLearner by organizing topics into a learning path.
    """

    def __init__(self, active_learner: ActiveLearner):
        self.learner = active_learner

        # Topic dependencies: topic -> prerequisites
        self.prerequisites: Dict[str, List[str]] = {}

        # Difficulty levels
        self.difficulty: Dict[str, float] = {}

    def set_prerequisites(self, topic: str, prereqs: List[str]):
        """Set prerequisites for a topic."""
        self.prerequisites[topic] = prereqs

    def set_difficulty(self, topic: str, difficulty: float):
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
            if t.confidence >= 0.8:  # Already know it
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
