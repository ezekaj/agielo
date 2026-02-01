"""
Time Perception System - Temporal Cognition

Implements:
1. Interval Timing (duration estimation)
2. Temporal Order Memory (sequence of events)
3. Prospective/Retrospective Time
4. Subjective Time Dilation/Contraction
5. Temporal Context Model
6. Future Thinking (prospection)
7. Circadian Rhythm

Based on research:
- Scalar Expectancy Theory (timing)
- Temporal Context Model (memory)
- Wittmann: Subjective time
- Gilbert & Wilson: Prospection
- Eagleman: Time perception

Performance: O(1) time estimates, efficient temporal memory
Comparison vs existing:
- LLMs: No sense of time passing
- ACT-R: Simple time module
- This: Full temporal cognition model
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time


class TimeScale(Enum):
    """Different time scales."""
    MILLISECONDS = auto()
    SECONDS = auto()
    MINUTES = auto()
    HOURS = auto()
    DAYS = auto()
    WEEKS = auto()
    YEARS = auto()


@dataclass
class TemporalEvent:
    """An event with temporal information."""
    event_id: str
    content: np.ndarray
    objective_time: float       # Actual timestamp
    subjective_time: float      # Perceived timestamp
    duration: float             # How long it lasted
    emotional_intensity: float  # Affects time perception
    attention_level: float      # How much attention was paid


@dataclass
class TemporalContext:
    """Context that drifts over time."""
    context_vector: np.ndarray
    timestamp: float
    drift_rate: float = 0.01


class IntervalTimer:
    """
    Interval timing - estimating durations.

    Based on Scalar Expectancy Theory:
    - Pacemaker generates pulses
    - Accumulator counts pulses
    - Memory stores reference durations
    - Comparator judges current vs remembered
    """

    def __init__(self):
        # Pacemaker rate (affected by arousal, attention)
        self.base_pacemaker_rate = 1.0  # pulses per second
        self.current_pacemaker_rate = 1.0

        # Accumulator
        self.accumulated_pulses = 0.0

        # Reference durations (learned)
        self.reference_durations: Dict[str, float] = {
            'second': 1.0,
            'minute': 60.0,
            'hour': 3600.0
        }

        # Timer state
        self.timing_started = False
        self.start_time = 0.0

        # Variability (scalar property)
        self.coefficient_of_variation = 0.1

    def start_timing(self):
        """Start the interval timer."""
        self.timing_started = True
        self.start_time = time.time()
        self.accumulated_pulses = 0.0

    def accumulate(self, real_duration: float, arousal: float = 0.5, attention: float = 0.5):
        """Accumulate time based on internal clock."""
        # Arousal speeds up pacemaker
        arousal_factor = 0.8 + 0.4 * arousal

        # Attention affects pulse capture
        attention_factor = 0.5 + 0.5 * attention

        # Effective rate
        self.current_pacemaker_rate = self.base_pacemaker_rate * arousal_factor

        # Accumulated (with some lost to attention lapses)
        pulses = real_duration * self.current_pacemaker_rate * attention_factor
        self.accumulated_pulses += pulses

    def estimate_duration(self) -> Tuple[float, float]:
        """Estimate elapsed duration with uncertainty."""
        if not self.timing_started:
            return 0.0, 1.0

        # Base estimate from accumulated pulses
        estimated = self.accumulated_pulses / self.base_pacemaker_rate

        # Scalar variability (Weber's law for time)
        uncertainty = estimated * self.coefficient_of_variation

        return estimated, uncertainty

    def stop_timing(self) -> Dict[str, float]:
        """Stop timing and return results."""
        estimated, uncertainty = self.estimate_duration()
        actual = time.time() - self.start_time

        self.timing_started = False

        return {
            'estimated_duration': estimated,
            'actual_duration': actual,
            'error': abs(estimated - actual),
            'uncertainty': uncertainty
        }

    def calibrate(self, actual_duration: float, perceived_duration: float):
        """Calibrate timer based on feedback."""
        if actual_duration > 0:
            ratio = perceived_duration / actual_duration
            # Adjust pacemaker rate slowly
            self.base_pacemaker_rate *= (1.0 + 0.1 * (1.0 - ratio))


class TemporalOrderMemory:
    """
    Memory for temporal order of events.

    Based on Temporal Context Model (TCM):
    - Context drifts gradually over time
    - Events encoded with current context
    - Retrieval uses context similarity
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Context vector that drifts
        self.current_context = np.random.randn(dim)
        self.current_context /= np.linalg.norm(self.current_context)

        # Drift rate
        self.drift_rate = 0.05

        # Event sequence
        self.events: List[TemporalEvent] = []

        # Context history
        self.context_history: List[TemporalContext] = []

    def drift_context(self, input_vector: Optional[np.ndarray] = None):
        """Let context drift (time passing)."""
        # Random drift
        noise = np.random.randn(self.dim) * self.drift_rate
        self.current_context = self.current_context + noise

        # Input-driven drift
        if input_vector is not None:
            self.current_context = 0.9 * self.current_context + 0.1 * input_vector

        # Normalize
        self.current_context /= (np.linalg.norm(self.current_context) + 1e-8)

        # Record context
        self.context_history.append(TemporalContext(
            context_vector=self.current_context.copy(),
            timestamp=time.time(),
            drift_rate=self.drift_rate
        ))

    def encode_event(self,
                     event_id: str,
                     content: np.ndarray,
                     duration: float = 0.0,
                     emotional_intensity: float = 0.5,
                     attention_level: float = 0.5) -> TemporalEvent:
        """Encode event with temporal context."""
        # Combine content with context
        encoded_content = 0.7 * content + 0.3 * self.current_context

        event = TemporalEvent(
            event_id=event_id,
            content=encoded_content,
            objective_time=time.time(),
            subjective_time=time.time(),  # Will be adjusted
            duration=duration,
            emotional_intensity=emotional_intensity,
            attention_level=attention_level
        )

        self.events.append(event)

        # Drift context after encoding
        self.drift_context(content)

        return event

    def retrieve_by_context(self,
                            query_context: np.ndarray,
                            k: int = 5) -> List[TemporalEvent]:
        """Retrieve events similar to query context."""
        if not self.events:
            return []

        # Score by context similarity
        scored = []
        for event in self.events:
            similarity = np.dot(event.content, query_context) / (
                np.linalg.norm(event.content) * np.linalg.norm(query_context) + 1e-8
            )
            scored.append((similarity, event))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    def get_temporal_order(self, event_ids: List[str]) -> List[str]:
        """Get events in temporal order."""
        id_to_event = {e.event_id: e for e in self.events}
        relevant = [id_to_event[eid] for eid in event_ids if eid in id_to_event]
        relevant.sort(key=lambda e: e.objective_time)
        return [e.event_id for e in relevant]

    def estimate_time_between(self, event1_id: str, event2_id: str) -> float:
        """Estimate time between two events using context distance."""
        e1 = next((e for e in self.events if e.event_id == event1_id), None)
        e2 = next((e for e in self.events if e.event_id == event2_id), None)

        if not e1 or not e2:
            return 0.0

        # Context distance as proxy for time
        context_distance = np.linalg.norm(e1.content - e2.content)

        # Convert to time estimate (rough)
        estimated_time = context_distance / self.drift_rate

        return estimated_time


class SubjectiveTime:
    """
    Subjective time perception.

    Time seems to:
    - Speed up when busy/happy
    - Slow down when bored/fearful
    - Compress in retrospect for routine
    - Expand in retrospect for novel events
    """

    def __init__(self):
        # Factors affecting subjective time
        self.arousal = 0.5
        self.attention = 0.5
        self.emotional_valence = 0.0
        self.novelty = 0.5

        # Prospective vs retrospective
        self.prospective_mode = True

    def compute_subjective_rate(self) -> float:
        """
        Compute how fast time feels like it's passing.

        > 1.0 = time feels fast
        < 1.0 = time feels slow
        """
        rate = 1.0

        # High arousal -> faster
        rate *= (0.8 + 0.4 * self.arousal)

        # High attention -> faster (in the moment)
        if self.prospective_mode:
            rate *= (0.7 + 0.6 * self.attention)

        # Positive emotion -> faster
        rate *= (0.9 + 0.2 * self.emotional_valence)

        return rate

    def compute_retrospective_duration(self,
                                        actual_duration: float,
                                        events: List[TemporalEvent]) -> float:
        """
        Compute how long a period seems in retrospect.

        More events = feels longer
        More novel = feels longer
        More emotional = feels longer
        """
        if not events:
            return actual_duration

        # Number of events increases perceived duration
        event_factor = 1.0 + 0.1 * len(events)

        # Novelty increases perceived duration
        novelty = np.mean([e.attention_level for e in events])
        novelty_factor = 0.8 + 0.4 * novelty

        # Emotional intensity increases perceived duration
        emotional = np.mean([e.emotional_intensity for e in events])
        emotional_factor = 0.9 + 0.2 * emotional

        retrospective = actual_duration * event_factor * novelty_factor * emotional_factor

        return retrospective

    def time_dilation_moment(self,
                              fear_level: float = 0.0,
                              surprise_level: float = 0.0) -> float:
        """
        Compute time dilation in the moment.

        Fear and surprise can cause time to seem to slow down.
        """
        # Strong emotions slow subjective time
        dilation = 1.0 - 0.5 * max(fear_level, surprise_level)

        return max(0.2, dilation)  # At most 5x slower


class FutureThinking:
    """
    Prospection - thinking about the future.

    Uses similar mechanisms to remembering:
    - Constructive simulation
    - Temporal discounting
    - Affective forecasting
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Future scenarios
        self.future_scenarios: List[Dict[str, Any]] = []

        # Temporal discounting rate (hyperbolic)
        self.discount_rate = 0.1

    def imagine_future(self,
                       goal_state: np.ndarray,
                       time_ahead: float,
                       current_context: np.ndarray) -> Dict[str, Any]:
        """Imagine future scenario."""
        # Constructive simulation: blend goal with current context
        vividness = 1.0 / (1.0 + time_ahead / 3600)  # Decreases with time
        imagined = vividness * goal_state + (1 - vividness) * current_context

        scenario = {
            'imagined_state': imagined,
            'time_ahead': time_ahead,
            'vividness': vividness,
            'timestamp': time.time()
        }

        self.future_scenarios.append(scenario)
        return scenario

    def temporal_discount(self, value: float, delay: float) -> float:
        """
        Hyperbolic temporal discounting.

        Humans discount future rewards hyperbolically, not exponentially.
        """
        # Hyperbolic: V = A / (1 + kD)
        discounted = value / (1 + self.discount_rate * delay)
        return discounted

    def affective_forecast(self,
                           event_valence: float,
                           time_until: float) -> float:
        """
        Predict emotional impact of future event.

        People tend to overestimate future emotional impact (impact bias).
        """
        # Base prediction
        predicted_impact = event_valence

        # Impact bias: we overestimate
        impact_bias = 1.3

        # Focalism: we focus too much on the event
        focalism_factor = 1.2

        # But effects diminish with time
        time_decay = 1.0 / (1.0 + time_until / 86400)  # Day constant

        forecast = predicted_impact * impact_bias * focalism_factor * time_decay

        return np.clip(forecast, -1, 1)


class CircadianRhythm:
    """
    Circadian rhythm effects on cognition.
    """

    def __init__(self):
        # 24-hour cycle
        self.period = 24 * 3600  # seconds

        # Phase (0 = midnight)
        self.phase = 0.0

        # Individual chronotype (morning vs evening person)
        self.chronotype = 0.0  # -1 = morning, +1 = evening

    def update(self, elapsed_time: float):
        """Update circadian phase."""
        self.phase = (self.phase + elapsed_time) % self.period

    def get_alertness(self) -> float:
        """Get current alertness level based on circadian phase."""
        # Convert phase to hours
        hour = (self.phase / 3600) % 24

        # Adjust for chronotype
        adjusted_hour = hour - self.chronotype * 2

        # Alertness curve (two peaks: morning and afternoon)
        # Dip around 2-4 PM ("post-lunch dip")
        morning_peak = np.exp(-((adjusted_hour - 10) ** 2) / 20)
        afternoon_peak = np.exp(-((adjusted_hour - 16) ** 2) / 20)
        night_low = np.exp(-((adjusted_hour - 4) ** 2) / 10)

        alertness = 0.5 + 0.3 * morning_peak + 0.2 * afternoon_peak - 0.4 * night_low

        return np.clip(alertness, 0.1, 1.0)

    def get_optimal_time_for(self, task_type: str) -> float:
        """Get optimal time of day for task type."""
        optimal_hours = {
            'analytical': 10.0,     # Morning
            'creative': 14.0,       # Afternoon (slightly tired = creative)
            'memory': 10.0,         # Morning
            'physical': 17.0,       # Late afternoon
            'sleep': 23.0,          # Night
        }

        hour = optimal_hours.get(task_type, 12.0)

        # Adjust for chronotype
        hour += self.chronotype * 2

        return hour


class TimePerceptionSystem:
    """
    Complete time perception system.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Components
        self.interval_timer = IntervalTimer()
        self.temporal_memory = TemporalOrderMemory(dim)
        self.subjective = SubjectiveTime()
        self.future = FutureThinking(dim)
        self.circadian = CircadianRhythm()

        # State
        self.current_time = time.time()
        self.session_start = time.time()

    def update(self, elapsed: float, events: Optional[List[Dict]] = None):
        """Update time perception."""
        # Update circadian
        self.circadian.update(elapsed)

        # Drift temporal context
        self.temporal_memory.drift_context()

        # Update timer if running
        if self.interval_timer.timing_started:
            alertness = self.circadian.get_alertness()
            self.interval_timer.accumulate(
                elapsed,
                arousal=alertness,
                attention=self.subjective.attention
            )

        # Encode any events
        if events:
            for event in events:
                self.temporal_memory.encode_event(
                    event.get('id', f'event_{time.time()}'),
                    event.get('embedding', np.zeros(self.dim)),
                    duration=event.get('duration', 0),
                    emotional_intensity=event.get('emotion', 0.5),
                    attention_level=event.get('attention', 0.5)
                )

        self.current_time = time.time()

    def estimate_elapsed_time(self) -> Dict[str, float]:
        """Estimate how much time has elapsed."""
        actual = time.time() - self.session_start

        # Subjective rate
        rate = self.subjective.compute_subjective_rate()
        subjective = actual * rate

        # Retrospective estimate
        retrospective = self.subjective.compute_retrospective_duration(
            actual, self.temporal_memory.events
        )

        return {
            'actual_seconds': actual,
            'subjective_seconds': subjective,
            'retrospective_seconds': retrospective,
            'time_perception_rate': rate
        }

    def how_long_ago(self, event_id: str) -> Dict[str, float]:
        """Estimate how long ago an event occurred."""
        event = next(
            (e for e in self.temporal_memory.events if e.event_id == event_id),
            None
        )

        if not event:
            return {'error': 'event_not_found'}

        actual = time.time() - event.objective_time

        # Context-based estimate
        context_distance = np.linalg.norm(
            self.temporal_memory.current_context - event.content
        )
        context_estimate = context_distance / self.temporal_memory.drift_rate

        return {
            'actual_seconds': actual,
            'context_estimate': context_estimate,
            'event_emotional_intensity': event.emotional_intensity
        }

    def when_will(self, event_embedding: np.ndarray, estimated_delay: float) -> Dict[str, Any]:
        """Think about future event."""
        scenario = self.future.imagine_future(
            event_embedding,
            estimated_delay,
            self.temporal_memory.current_context
        )

        # Affective forecast
        # Assume positive event
        affective = self.future.affective_forecast(0.5, estimated_delay)

        # Temporal discount
        value = 1.0
        discounted_value = self.future.temporal_discount(value, estimated_delay)

        return {
            'scenario': scenario,
            'affective_forecast': affective,
            'discounted_value': discounted_value,
            'vividness': scenario['vividness']
        }

    def get_state(self) -> Dict[str, Any]:
        """Get time perception state."""
        return {
            'session_duration': time.time() - self.session_start,
            'events_encoded': len(self.temporal_memory.events),
            'future_scenarios': len(self.future.future_scenarios),
            'circadian_alertness': self.circadian.get_alertness(),
            'circadian_phase_hours': self.circadian.phase / 3600,
            'subjective_time_rate': self.subjective.compute_subjective_rate()
        }
