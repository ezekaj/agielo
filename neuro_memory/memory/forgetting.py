"""
Memory Forgetting and Decay
============================

Time-based forgetting following Ebbinghaus forgetting curve.
Implements activation-based memory decay with rehearsal.

References:
- Ebbinghaus (1885): Forgetting curve
- Anderson & Schooler (1991): Rational analysis of memory
- Wixted & Ebbesen (1991): Power law of forgetting
"""

import numpy as np
import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ForgettingConfig:
    """Configuration for memory forgetting."""
    decay_rate: float = 0.5  # How fast memories decay
    rehearsal_boost: float = 1.5  # Strength multiplier on rehearsal
    min_activation: float = 0.1  # Minimum activation threshold
    use_power_law: bool = True  # Use power law vs exponential decay


class ForgettingEngine:
    """
    Manages memory decay and forgetting.

    Activation follows power law: A(t) = A0 * (1 + t)^(-d)
    where d is decay rate, t is time since encoding.

    Rehearsal resets the clock and boosts activation.
    """

    def __init__(self, config: ForgettingConfig = None):
        self.config = config or ForgettingConfig()

    def compute_activation(
        self,
        initial_activation: float,
        timestamp: datetime,
        rehearsal_count: int = 0,
        current_time: datetime = None
    ) -> float:
        """
        Compute current memory activation.

        Args:
            initial_activation: Activation at encoding (e.g., surprise score)
            timestamp: When memory was encoded
            rehearsal_count: Number of times memory was retrieved
            current_time: Current time (defaults to now)

        Returns:
            Current activation level

        Raises:
            ValueError: If initial_activation is not finite or is negative
        """
        # Validate initial_activation: must be finite and non-negative
        if not np.isfinite(initial_activation):
            raise ValueError(
                f"initial_activation must be finite, got {initial_activation}"
            )
        if initial_activation < 0:
            raise ValueError(
                f"initial_activation must be non-negative, got {initial_activation}"
            )

        current_time = current_time or datetime.now()
        time_elapsed = (current_time - timestamp).total_seconds() / 3600  # hours

        # Apply rehearsal boost
        boosted_activation = initial_activation * (self.config.rehearsal_boost ** rehearsal_count)

        # Apply decay
        if self.config.use_power_law:
            # Power law decay: A(t) = A0 * (1 + t)^(-d)
            activation = boosted_activation * ((1 + time_elapsed) ** (-self.config.decay_rate))
        else:
            # Exponential decay: A(t) = A0 * e^(-dt)
            # Clip exponent to prevent underflow (exp(-500) -> 0, exp(500) -> overflow)
            decay_exponent = np.clip(-self.config.decay_rate * time_elapsed, -500, 0)
            activation = boosted_activation * np.exp(decay_exponent)

        return float(max(activation, self.config.min_activation))

    def should_forget(
        self,
        activation: float
    ) -> bool:
        """
        Determine if memory should be forgotten.

        Args:
            activation: Current activation level

        Returns:
            True if activation below threshold

        Raises:
            ValueError: If activation is not finite or is negative
        """
        # Validate activation: must be finite and non-negative
        if not np.isfinite(activation):
            raise ValueError(
                f"activation must be finite, got {activation}"
            )
        if activation < 0:
            raise ValueError(
                f"activation must be non-negative, got {activation}"
            )

        return activation < self.config.min_activation

    def get_forgetting_probability(
        self,
        activation: float
    ) -> float:
        """
        Probability of forgetting given activation.

        Uses sigmoid function around min_activation threshold.
        Higher activation = lower forgetting probability.

        Args:
            activation: Current activation level

        Returns:
            Probability in [0, 1]

        Raises:
            ValueError: If activation is not finite or is negative
        """
        # Validate activation: must be finite and non-negative
        if not np.isfinite(activation):
            raise ValueError(
                f"activation must be finite, got {activation}"
            )
        if activation < 0:
            raise ValueError(
                f"activation must be non-negative, got {activation}"
            )

        # Sigmoid function around min_activation threshold
        x = (activation - self.config.min_activation) / max(self.config.min_activation, 1e-10)

        # Clip to prevent numerical overflow in exp()
        # exp(500) overflows, exp(-500) underflows to 0 (safe)
        x_scaled = np.clip(x * 5, -500, 500)

        prob = 1 / (1 + np.exp(x_scaled))
        return float(prob)


@dataclass
class MemoryState:
    """
    State tracking for a single memory item with Ebbinghaus forgetting.

    Tracks the stability score which increases with each successful retrieval,
    implementing spaced repetition principles.
    """
    memory_id: str
    created_at: float  # Unix timestamp
    last_access: float  # Unix timestamp of last retrieval
    access_count: int = 0  # Number of successful retrievals
    stability_score: float = 1.0  # Stability S in R = e^(-t/S) formula
    initial_retention: float = 1.0  # Starting retention (usually 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "access_count": self.access_count,
            "stability_score": self.stability_score,
            "initial_retention": self.initial_retention
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryState":
        """Create from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            created_at=data["created_at"],
            last_access=data["last_access"],
            access_count=data.get("access_count", 0),
            stability_score=data.get("stability_score", 1.0),
            initial_retention=data.get("initial_retention", 1.0)
        )


@dataclass
class EbbinghausConfig:
    """Configuration for Ebbinghaus forgetting model."""
    base_stability: float = 1.0  # Initial stability (in hours)
    stability_multiplier: float = 1.5  # How much stability increases on successful retrieval
    forget_threshold: float = 0.3  # Retention threshold below which memory may be forgotten
    min_stability: float = 0.5  # Minimum stability score
    max_stability: float = 720.0  # Maximum stability (30 days in hours)


class EbbinghausForgetting:
    """
    Ebbinghaus-style forgetting with spaced repetition.

    Implements the forgetting formula: R = e^(-t/S)
    where:
    - R = retention (0-1)
    - t = time since last access
    - S = stability (increases with each successful retrieval)

    The stability score increases with each successful retrieval,
    implementing the core principle of spaced repetition: memories
    become more stable the more they are retrieved.
    """

    def __init__(
        self,
        config: Optional[EbbinghausConfig] = None,
        state_path: Optional[Path] = None
    ):
        """
        Initialize EbbinghausForgetting.

        Args:
            config: Configuration for forgetting parameters
            state_path: Path to save/load memory states
        """
        self.config = config or EbbinghausConfig()
        self.state_path = state_path
        self._memory_states: Dict[str, MemoryState] = {}

        if self.state_path and self.state_path.exists():
            self._load_state()

    def register_memory(
        self,
        memory_id: str,
        initial_retention: float = 1.0,
        timestamp: Optional[float] = None
    ) -> MemoryState:
        """
        Register a new memory for forgetting tracking.

        Args:
            memory_id: Unique identifier for the memory
            initial_retention: Starting retention level (default 1.0)
            timestamp: When memory was created (default: now)

        Returns:
            The created MemoryState
        """
        now = timestamp or datetime.now().timestamp()

        state = MemoryState(
            memory_id=memory_id,
            created_at=now,
            last_access=now,
            access_count=0,
            stability_score=self.config.base_stability,
            initial_retention=initial_retention
        )

        self._memory_states[memory_id] = state
        self._save_state()
        return state

    def get_memory_state(self, memory_id: str) -> Optional[MemoryState]:
        """Get the state for a memory."""
        return self._memory_states.get(memory_id)

    def compute_retention(
        self,
        memory_id: str,
        current_time: Optional[float] = None
    ) -> float:
        """
        Compute current retention for a memory.

        Formula: R = e^(-t/S)

        Args:
            memory_id: ID of the memory
            current_time: Current timestamp (default: now)

        Returns:
            Retention value in [0, 1], or 0 if memory not found
        """
        state = self._memory_states.get(memory_id)
        if state is None:
            return 0.0

        now = current_time or datetime.now().timestamp()

        # Time since last access in hours
        time_elapsed_hours = (now - state.last_access) / 3600.0

        # Ensure time is non-negative
        time_elapsed_hours = max(0, time_elapsed_hours)

        # Ebbinghaus formula: R = e^(-t/S)
        # Prevent division by zero and overflow
        stability = max(state.stability_score, 1e-10)
        exponent = np.clip(-time_elapsed_hours / stability, -500, 0)
        retention = np.exp(exponent)

        # Scale by initial retention
        retention = retention * state.initial_retention

        return float(max(0.0, min(1.0, retention)))

    def should_forget(
        self,
        memory_id: str,
        threshold: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> bool:
        """
        Determine if a memory should be forgotten.

        Args:
            memory_id: ID of the memory
            threshold: Retention threshold (default: config.forget_threshold)
            current_time: Current timestamp (default: now)

        Returns:
            True if retention is below threshold
        """
        threshold = threshold if threshold is not None else self.config.forget_threshold
        retention = self.compute_retention(memory_id, current_time)
        return retention < threshold

    def record_retrieval(
        self,
        memory_id: str,
        success: bool = True,
        current_time: Optional[float] = None
    ) -> Optional[MemoryState]:
        """
        Record a memory retrieval attempt.

        Successful retrieval increases stability (spaced repetition).
        Failed retrieval resets stability to minimum.

        Args:
            memory_id: ID of the memory
            success: Whether retrieval was successful
            current_time: Timestamp of retrieval (default: now)

        Returns:
            Updated MemoryState or None if memory not found
        """
        state = self._memory_states.get(memory_id)
        if state is None:
            return None

        now = current_time or datetime.now().timestamp()

        state.last_access = now
        state.access_count += 1

        if success:
            # Increase stability on successful retrieval
            new_stability = state.stability_score * self.config.stability_multiplier
            state.stability_score = min(new_stability, self.config.max_stability)
        else:
            # Reset stability on failed retrieval
            state.stability_score = self.config.min_stability

        self._save_state()
        return state

    def get_all_retentions(
        self,
        current_time: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Get current retention for all memories.

        Args:
            current_time: Current timestamp (default: now)

        Returns:
            Dict mapping memory_id to retention
        """
        return {
            memory_id: self.compute_retention(memory_id, current_time)
            for memory_id in self._memory_states
        }

    def get_memories_below_threshold(
        self,
        threshold: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Get memories with retention below threshold.

        Args:
            threshold: Retention threshold (default: config.forget_threshold)
            current_time: Current timestamp (default: now)

        Returns:
            List of (memory_id, retention) tuples sorted by retention ascending
        """
        threshold = threshold if threshold is not None else self.config.forget_threshold

        below_threshold = [
            (mid, retention)
            for mid, retention in self.get_all_retentions(current_time).items()
            if retention < threshold
        ]

        return sorted(below_threshold, key=lambda x: x[1])

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the forgetting system.

        Returns:
            Dict with statistics
        """
        if not self._memory_states:
            return {
                "total_memories": 0,
                "avg_stability": 0.0,
                "avg_retention": 0.0,
                "memories_at_risk": 0,
                "stability_distribution": {}
            }

        retentions = self.get_all_retentions()
        stabilities = [s.stability_score for s in self._memory_states.values()]
        access_counts = [s.access_count for s in self._memory_states.values()]

        # Stability distribution buckets (in hours)
        stability_buckets = {
            "< 1h": 0,
            "1-6h": 0,
            "6-24h": 0,
            "1-7d": 0,
            "> 7d": 0
        }

        for stability in stabilities:
            if stability < 1:
                stability_buckets["< 1h"] += 1
            elif stability < 6:
                stability_buckets["1-6h"] += 1
            elif stability < 24:
                stability_buckets["6-24h"] += 1
            elif stability < 168:
                stability_buckets["1-7d"] += 1
            else:
                stability_buckets["> 7d"] += 1

        return {
            "total_memories": len(self._memory_states),
            "avg_stability": float(np.mean(stabilities)),
            "max_stability": float(np.max(stabilities)),
            "min_stability": float(np.min(stabilities)),
            "avg_retention": float(np.mean(list(retentions.values()))),
            "avg_access_count": float(np.mean(access_counts)),
            "memories_at_risk": len(self.get_memories_below_threshold()),
            "stability_distribution": stability_buckets
        }

    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from tracking.

        Args:
            memory_id: ID of memory to remove

        Returns:
            True if memory was found and removed
        """
        if memory_id in self._memory_states:
            del self._memory_states[memory_id]
            self._save_state()
            return True
        return False

    def _save_state(self):
        """Save memory states to disk if path configured."""
        if self.state_path is None:
            return

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "memories": {
                    mid: state.to_dict()
                    for mid, state in self._memory_states.items()
                },
                "config": {
                    "base_stability": self.config.base_stability,
                    "stability_multiplier": self.config.stability_multiplier,
                    "forget_threshold": self.config.forget_threshold,
                    "min_stability": self.config.min_stability,
                    "max_stability": self.config.max_stability
                }
            }
            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save forgetting state: {e}")

    def _load_state(self):
        """Load memory states from disk."""
        if self.state_path is None or not self.state_path.exists():
            return

        try:
            with open(self.state_path, 'r') as f:
                data = json.load(f)

            self._memory_states = {
                mid: MemoryState.from_dict(state_data)
                for mid, state_data in data.get("memories", {}).items()
            }
        except Exception as e:
            print(f"Warning: Failed to load forgetting state: {e}")
            self._memory_states = {}


@dataclass
class SpacedRepetitionConfig:
    """Configuration for spaced repetition scheduling."""
    base_intervals: List[float] = field(default_factory=lambda: [
        24.0,    # 1 day (hours)
        72.0,    # 3 days
        168.0,   # 7 days
        336.0,   # 14 days
        720.0,   # 30 days
        2160.0   # 90 days
    ])
    immediate_review_delay: float = 0.25  # 15 minutes for immediate review
    stability_weight: float = 1.0  # How much stability influences review interval


@dataclass
class ReviewItem:
    """Represents a memory item due for review."""
    memory_id: str
    next_review: float  # Unix timestamp
    stability: float
    retention: float
    interval_level: int  # Which base interval we're at (0-5)
    is_immediate: bool = False  # Whether this is an immediate review after failure

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "next_review": self.next_review,
            "stability": self.stability,
            "retention": self.retention,
            "interval_level": self.interval_level,
            "is_immediate": self.is_immediate
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewItem":
        """Create from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            next_review=data["next_review"],
            stability=data["stability"],
            retention=data["retention"],
            interval_level=data.get("interval_level", 0),
            is_immediate=data.get("is_immediate", False)
        )


class SpacedRepetitionScheduler:
    """
    Spaced repetition scheduler that determines when memories should be reviewed.

    Uses a combination of:
    1. Base intervals (1, 3, 7, 14, 30, 90 days) as guideline
    2. Stability score to scale intervals (higher stability = longer intervals)
    3. Immediate review scheduling on failed retrieval

    After successful retrieval: next_review = now + stability * base_interval
    After failed retrieval: reset stability, schedule immediate review
    """

    def __init__(
        self,
        ebbinghaus: Optional[EbbinghausForgetting] = None,
        config: Optional[SpacedRepetitionConfig] = None,
        state_path: Optional[Path] = None
    ):
        """
        Initialize SpacedRepetitionScheduler.

        Args:
            ebbinghaus: EbbinghausForgetting instance for retention calculation
            config: Configuration for scheduling
            state_path: Path to save/load review schedule
        """
        self.ebbinghaus = ebbinghaus or EbbinghausForgetting()
        self.config = config or SpacedRepetitionConfig()
        self.state_path = state_path
        self._review_schedule: Dict[str, ReviewItem] = {}
        self._review_history: List[Dict[str, Any]] = []

        if self.state_path and self.state_path.exists():
            self._load_state()

    def schedule_memory(
        self,
        memory_id: str,
        current_time: Optional[float] = None
    ) -> ReviewItem:
        """
        Schedule a memory for initial review.

        Args:
            memory_id: ID of memory to schedule
            current_time: Current timestamp (default: now)

        Returns:
            ReviewItem with scheduled review time
        """
        now = current_time or datetime.now().timestamp()

        # Get or create memory state in ebbinghaus
        memory_state = self.ebbinghaus.get_memory_state(memory_id)
        if memory_state is None:
            memory_state = self.ebbinghaus.register_memory(memory_id, timestamp=now)

        # Calculate next review based on first interval and stability
        base_interval = self.config.base_intervals[0]  # Start at level 0
        scaled_interval = base_interval * memory_state.stability_score * self.config.stability_weight
        next_review = now + scaled_interval * 3600  # Convert hours to seconds

        retention = self.ebbinghaus.compute_retention(memory_id, now)

        review_item = ReviewItem(
            memory_id=memory_id,
            next_review=next_review,
            stability=memory_state.stability_score,
            retention=retention,
            interval_level=0,
            is_immediate=False
        )

        self._review_schedule[memory_id] = review_item
        self._save_state()

        return review_item

    def record_review(
        self,
        memory_id: str,
        success: bool,
        current_time: Optional[float] = None
    ) -> Optional[ReviewItem]:
        """
        Record a review attempt and schedule the next review.

        After successful retrieval: next_review = now + stability * base_interval
        Stability increases and interval level advances.

        After failed retrieval: reset stability, schedule immediate review.

        Args:
            memory_id: ID of memory reviewed
            success: Whether the review/retrieval was successful
            current_time: Timestamp of review (default: now)

        Returns:
            Updated ReviewItem or None if memory not found
        """
        now = current_time or datetime.now().timestamp()

        # Record in ebbinghaus (updates stability)
        memory_state = self.ebbinghaus.record_retrieval(memory_id, success, now)
        if memory_state is None:
            return None

        # Get current interval level
        current_item = self._review_schedule.get(memory_id)
        current_level = current_item.interval_level if current_item else 0

        if success:
            # Advance to next interval level (cap at max)
            new_level = min(current_level + 1, len(self.config.base_intervals) - 1)

            # Calculate next review: now + stability * base_interval
            base_interval = self.config.base_intervals[new_level]
            scaled_interval = base_interval * memory_state.stability_score * self.config.stability_weight
            next_review = now + scaled_interval * 3600

            review_item = ReviewItem(
                memory_id=memory_id,
                next_review=next_review,
                stability=memory_state.stability_score,
                retention=self.ebbinghaus.compute_retention(memory_id, now),
                interval_level=new_level,
                is_immediate=False
            )
        else:
            # Failed - schedule immediate review, reset level to 0
            next_review = now + self.config.immediate_review_delay * 3600  # 15 min default

            review_item = ReviewItem(
                memory_id=memory_id,
                next_review=next_review,
                stability=memory_state.stability_score,
                retention=self.ebbinghaus.compute_retention(memory_id, now),
                interval_level=0,  # Reset to beginning
                is_immediate=True
            )

        self._review_schedule[memory_id] = review_item

        # Record in history
        self._review_history.append({
            "memory_id": memory_id,
            "timestamp": now,
            "success": success,
            "stability": memory_state.stability_score,
            "interval_level": review_item.interval_level,
            "next_review": review_item.next_review
        })

        self._save_state()
        return review_item

    def get_due_for_review(
        self,
        limit: int = 10,
        current_time: Optional[float] = None
    ) -> List[str]:
        """
        Get memories that are due for review (reinforcement).

        Returns memories whose next_review time has passed,
        sorted by most overdue first.

        Args:
            limit: Maximum number of memory IDs to return
            current_time: Current timestamp (default: now)

        Returns:
            List of memory_ids needing review, sorted by urgency
        """
        now = current_time or datetime.now().timestamp()

        # Find all overdue items
        overdue = []
        for memory_id, item in self._review_schedule.items():
            if item.next_review <= now:
                # Calculate how overdue (more overdue = higher priority)
                overdue_hours = (now - item.next_review) / 3600
                overdue.append((memory_id, overdue_hours, item))

        # Sort by overdue time (most overdue first)
        overdue.sort(key=lambda x: x[1], reverse=True)

        # Return just the memory IDs, limited
        return [memory_id for memory_id, _, _ in overdue[:limit]]

    def get_review_item(self, memory_id: str) -> Optional[ReviewItem]:
        """Get the review item for a specific memory."""
        return self._review_schedule.get(memory_id)

    def get_upcoming_reviews(
        self,
        hours_ahead: float = 24.0,
        current_time: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Get memories due for review in the next N hours.

        Args:
            hours_ahead: How many hours ahead to look
            current_time: Current timestamp (default: now)

        Returns:
            List of (memory_id, hours_until_due) sorted by soonest first
        """
        now = current_time or datetime.now().timestamp()
        cutoff = now + hours_ahead * 3600

        upcoming = []
        for memory_id, item in self._review_schedule.items():
            if item.next_review > now and item.next_review <= cutoff:
                hours_until = (item.next_review - now) / 3600
                upcoming.append((memory_id, hours_until))

        return sorted(upcoming, key=lambda x: x[1])

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the review schedule.

        Returns:
            Dict with statistics about reviews and scheduling
        """
        if not self._review_schedule:
            return {
                "total_scheduled": 0,
                "due_now": 0,
                "upcoming_24h": 0,
                "avg_interval_level": 0.0,
                "immediate_reviews": 0,
                "review_history_count": 0,
                "success_rate": 0.0,
                "interval_level_distribution": {}
            }

        now = datetime.now().timestamp()

        # Count due and upcoming
        due_now = len(self.get_due_for_review(limit=1000, current_time=now))
        upcoming = len(self.get_upcoming_reviews(hours_ahead=24.0, current_time=now))

        # Interval level distribution
        level_counts = {}
        immediate_count = 0
        for item in self._review_schedule.values():
            level = item.interval_level
            level_counts[level] = level_counts.get(level, 0) + 1
            if item.is_immediate:
                immediate_count += 1

        avg_level = sum(
            item.interval_level for item in self._review_schedule.values()
        ) / len(self._review_schedule)

        # Success rate from history
        if self._review_history:
            successes = sum(1 for h in self._review_history if h["success"])
            success_rate = successes / len(self._review_history)
        else:
            success_rate = 0.0

        return {
            "total_scheduled": len(self._review_schedule),
            "due_now": due_now,
            "upcoming_24h": upcoming,
            "avg_interval_level": avg_level,
            "immediate_reviews": immediate_count,
            "review_history_count": len(self._review_history),
            "success_rate": success_rate,
            "interval_level_distribution": level_counts
        }

    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from the review schedule.

        Args:
            memory_id: ID of memory to remove

        Returns:
            True if memory was found and removed
        """
        if memory_id in self._review_schedule:
            del self._review_schedule[memory_id]
            self._save_state()
            return True
        return False

    def _save_state(self):
        """Save schedule state to disk if path configured."""
        if self.state_path is None:
            return

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "schedule": {
                    mid: item.to_dict()
                    for mid, item in self._review_schedule.items()
                },
                "history": self._review_history[-1000:],  # Keep last 1000 entries
                "config": {
                    "base_intervals": self.config.base_intervals,
                    "immediate_review_delay": self.config.immediate_review_delay,
                    "stability_weight": self.config.stability_weight
                }
            }
            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save spaced repetition state: {e}")

    def _load_state(self):
        """Load schedule state from disk."""
        if self.state_path is None or not self.state_path.exists():
            return

        try:
            with open(self.state_path, 'r') as f:
                data = json.load(f)

            self._review_schedule = {
                mid: ReviewItem.from_dict(item_data)
                for mid, item_data in data.get("schedule", {}).items()
            }
            self._review_history = data.get("history", [])
        except Exception as e:
            print(f"Warning: Failed to load spaced repetition state: {e}")
            self._review_schedule = {}
            self._review_history = []


if __name__ == "__main__":
    print("=== Memory Forgetting Test ===\n")

    engine = ForgettingEngine()

    # Test memory decay over time
    initial_activation = 2.5  # High surprise
    timestamp = datetime.now() - timedelta(hours=24)

    print(f"Initial activation: {initial_activation:.3f}")
    print(f"Encoded: {timestamp.strftime('%Y-%m-%d %H:%M')}\n")

    print("Activation decay over time:")
    for hours in [0, 1, 6, 12, 24, 48, 72, 168]:  # 0h to 1 week
        test_time = timestamp + timedelta(hours=hours)
        activation = engine.compute_activation(initial_activation, timestamp,
                                               rehearsal_count=0, current_time=test_time)
        forget_prob = engine.get_forgetting_probability(activation)

        print(f"  After {hours:3d}h: activation={activation:.3f}, forget_prob={forget_prob:.3f}")

    # Test rehearsal effect
    print("\nEffect of rehearsal (after 24 hours):")
    test_time = timestamp + timedelta(hours=24)
    for rehearsals in range(5):
        activation = engine.compute_activation(initial_activation, timestamp,
                                               rehearsal_count=rehearsals, current_time=test_time)
        print(f"  {rehearsals} rehearsals: activation={activation:.3f}")

    print("\n✓ ForgettingEngine test complete!")

    # Test Ebbinghaus forgetting
    print("\n=== Ebbinghaus Forgetting Test ===\n")

    ebbinghaus = EbbinghausForgetting()

    # Register a test memory
    memory = ebbinghaus.register_memory("test_memory_1")
    print(f"Registered memory: {memory.memory_id}")
    print(f"  Initial stability: {memory.stability_score:.2f}h")

    # Test retention decay over time
    print("\nRetention decay over time (R = e^(-t/S)):")
    base_time = datetime.now().timestamp()
    for hours in [0, 1, 2, 4, 8, 12, 24]:
        test_time = base_time + hours * 3600
        retention = ebbinghaus.compute_retention("test_memory_1", test_time)
        should_forget = ebbinghaus.should_forget("test_memory_1", current_time=test_time)
        print(f"  After {hours:2d}h: retention={retention:.3f}, should_forget={should_forget}")

    # Test spaced repetition effect
    print("\nSpaced repetition effect (stability increases with retrieval):")
    ebbinghaus2 = EbbinghausForgetting()
    ebbinghaus2.register_memory("spaced_memory")

    base_time = datetime.now().timestamp()
    for i in range(5):
        retrieval_time = base_time + i * 3600  # 1 hour intervals
        ebbinghaus2.record_retrieval("spaced_memory", success=True, current_time=retrieval_time)
        state = ebbinghaus2.get_memory_state("spaced_memory")
        retention_1h_later = ebbinghaus2.compute_retention("spaced_memory", retrieval_time + 3600)
        print(f"  After {i+1} retrievals: stability={state.stability_score:.2f}h, "
              f"retention(+1h)={retention_1h_later:.3f}")

    print("\n✓ Ebbinghaus test complete!")
