"""
Base Types and Data Structures for Human Cognition AI
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import time
import uuid
from collections import deque
import threading


# =============================================================================
# ENUMS
# =============================================================================

class ProcessingMode(Enum):
    INTUITIVE = "system1"      # Fast, automatic
    DELIBERATIVE = "system2"   # Slow, controlled
    MIXED = "mixed"            # Both active


class MemoryType(Enum):
    SENSORY = "sensory"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class EmotionType(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    FEAR = "fear"
    ANGER = "anger"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class ReasoningType(Enum):
    PERCEPTUAL = "perceptual"
    DIMENSIONAL = "dimensional"
    LOGICAL = "logical"
    INTERACTIVE = "interactive"


# =============================================================================
# BASE DATA CLASSES
# =============================================================================

@dataclass
class Timestamp:
    """Precise timestamp for events"""
    created: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        return time.time() - self.created

    def is_older_than(self, seconds: float) -> bool:
        return self.age > seconds


@dataclass
class Vector:
    """N-dimensional vector for representations"""
    data: np.ndarray

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.float32)

    @property
    def dim(self) -> int:
        return len(self.data)

    def distance(self, other: 'Vector') -> float:
        return np.linalg.norm(self.data - other.data)

    def cosine_similarity(self, other: 'Vector') -> float:
        dot = np.dot(self.data, other.data)
        norm = np.linalg.norm(self.data) * np.linalg.norm(other.data)
        return dot / (norm + 1e-8)

    def normalize(self) -> 'Vector':
        norm = np.linalg.norm(self.data)
        return Vector(self.data / (norm + 1e-8))

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.data + other.data)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.data - other.data)

    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.data * scalar)


@dataclass
class Prediction:
    """A prediction with uncertainty"""
    value: Vector
    precision: float  # Inverse variance (confidence)
    source_level: int  # Hierarchy level that generated this
    timestamp: Timestamp = field(default_factory=Timestamp)

    @property
    def uncertainty(self) -> float:
        return 1.0 / (self.precision + 1e-8)


@dataclass
class PredictionError:
    """Difference between prediction and actual"""
    error: Vector
    precision_weighted_error: Vector
    magnitude: float
    level: int
    timestamp: Timestamp = field(default_factory=Timestamp)


@dataclass
class MemoryItem:
    """Base class for all memory items"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    embedding: Optional[Vector] = None
    strength: float = 1.0
    timestamp: Timestamp = field(default_factory=Timestamp)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    emotional_valence: float = 0.0  # -1 to 1
    context: Dict[str, Any] = field(default_factory=dict)

    def access(self):
        self.access_count += 1
        self.last_accessed = time.time()
        # Strengthen on access
        self.strength = min(1.0, self.strength + 0.1)

    def decay(self, rate: float = 0.01):
        time_since_access = time.time() - self.last_accessed
        self.strength *= np.exp(-rate * time_since_access)


@dataclass
class Episode(MemoryItem):
    """Episodic memory: personal experience with context"""
    what: str = ""
    where: Optional[Vector] = None  # Spatial location
    when: Optional[Timestamp] = None
    who: List[str] = field(default_factory=list)
    emotional_state: Optional['EmotionalState'] = None
    sensory_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact(MemoryItem):
    """Semantic memory: context-free knowledge"""
    subject: str = ""
    relation: str = ""
    object: str = ""
    confidence: float = 1.0
    source: str = ""


@dataclass
class Skill(MemoryItem):
    """Procedural memory: learned skill/habit"""
    name: str = ""
    trigger_context: Dict[str, Any] = field(default_factory=dict)
    action_sequence: List[Any] = field(default_factory=list)
    automaticity: float = 0.0  # 0 = conscious, 1 = automatic
    success_rate: float = 0.5


@dataclass
class EmotionalState:
    """Current emotional state"""
    valence: float = 0.0  # -1 (negative) to 1 (positive)
    arousal: float = 0.0  # 0 (calm) to 1 (excited)
    dominance: float = 0.5  # 0 (submissive) to 1 (dominant)
    primary_emotion: EmotionType = EmotionType.NEUTRAL
    intensity: float = 0.0
    timestamp: Timestamp = field(default_factory=Timestamp)

    def blend(self, other: 'EmotionalState', weight: float = 0.5) -> 'EmotionalState':
        return EmotionalState(
            valence=self.valence * (1-weight) + other.valence * weight,
            arousal=self.arousal * (1-weight) + other.arousal * weight,
            dominance=self.dominance * (1-weight) + other.dominance * weight,
            primary_emotion=other.primary_emotion if weight > 0.5 else self.primary_emotion,
            intensity=self.intensity * (1-weight) + other.intensity * weight
        )


@dataclass
class Goal:
    """A goal with priority and status"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: float = 0.5
    progress: float = 0.0
    deadline: Optional[float] = None
    subgoals: List['Goal'] = field(default_factory=list)
    is_active: bool = True
    created: Timestamp = field(default_factory=Timestamp)

    @property
    def urgency(self) -> float:
        if self.deadline is None:
            return self.priority
        time_remaining = self.deadline - time.time()
        if time_remaining <= 0:
            return 1.0
        urgency_factor = 1.0 / (1.0 + time_remaining / 3600)  # Hours
        return min(1.0, self.priority + urgency_factor)


@dataclass
class Action:
    """An action that can be taken"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[Vector] = None
    confidence: float = 0.5
    cost: float = 0.1  # Effort required

    def __hash__(self):
        return hash(self.id)


@dataclass
class Belief:
    """A belief about the world"""
    proposition: str = ""
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    source: str = ""
    timestamp: Timestamp = field(default_factory=Timestamp)


@dataclass
class MentalState:
    """Complete mental state of an agent"""
    beliefs: List[Belief] = field(default_factory=list)
    desires: List[Goal] = field(default_factory=list)
    intentions: List[Action] = field(default_factory=list)
    emotional_state: EmotionalState = field(default_factory=EmotionalState)


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class RingBuffer:
    """Fixed-size circular buffer"""

    def __init__(self, size: int):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = threading.Lock()

    def push(self, item: Any):
        with self.lock:
            self.buffer.append(item)

    def get_all(self) -> List[Any]:
        with self.lock:
            return list(self.buffer)

    def get_recent(self, n: int) -> List[Any]:
        with self.lock:
            return list(self.buffer)[-n:]

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class ExponentialMovingAverage:
    """Exponential moving average tracker"""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def get(self) -> Optional[float]:
        return self.value


class SignalBus:
    """Pub/sub system for inter-component communication"""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Callable):
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, data: Any):
        with self.lock:
            subscribers = self.subscribers.get(event_type, [])
        for callback in subscribers:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in subscriber: {e}")

    def unsubscribe(self, event_type: str, callback: Callable):
        with self.lock:
            if event_type in self.subscribers:
                self.subscribers[event_type].remove(callback)


# Global signal bus instance
signal_bus = SignalBus()
