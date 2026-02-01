"""
Self-Awareness System - Metacognition and Self-Model

Implements:
1. Metacognition (thinking about thinking)
2. Self-Model (representation of own states, abilities, traits)
3. Introspection (accessing internal states)
4. Self-Monitoring (error detection, confidence)
5. Narrative Self (autobiographical continuity)
6. Theory of Own Mind (modeling own future states)
7. Agency Detection (distinguishing self-caused vs external)

This is what makes the difference between:
- Processing information (any AI)
- KNOWING that you're processing information (this system)

Performance: Efficient state hashing, O(1) self-model access
Comparison vs existing:
- ACT-R: Has meta-level but limited self-model
- SOAR: Universal subgoaling but no true introspection
- LLMs: No persistent self-model or metacognition
- This: Full recursive self-awareness loop
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time
import hashlib


class ConfidenceLevel(Enum):
    """Metacognitive confidence levels."""
    CERTAIN = auto()
    CONFIDENT = auto()
    UNCERTAIN = auto()
    GUESSING = auto()
    UNKNOWN = auto()


class CognitiveState(Enum):
    """Detectable cognitive states."""
    FOCUSED = auto()
    DISTRACTED = auto()
    CONFUSED = auto()
    FLUENT = auto()
    STRUGGLING = auto()
    LEARNING = auto()
    EXPERT = auto()


@dataclass
class MetacognitiveSignal:
    """A signal about one's own cognitive process."""
    process_name: str
    confidence: float           # 0-1
    fluency: float              # How easily it's going
    error_detected: bool
    needs_more_effort: bool
    source: str                 # Which monitor generated this
    timestamp: float = field(default_factory=time.time)


@dataclass
class SelfModelFacet:
    """One aspect of the self-model."""
    name: str
    value: Any
    confidence: float
    last_updated: float = field(default_factory=time.time)
    update_count: int = 0

    def update(self, new_value: Any, new_confidence: float):
        """Update with exponential smoothing."""
        if isinstance(self.value, (int, float)) and isinstance(new_value, (int, float)):
            self.value = 0.7 * self.value + 0.3 * new_value
        else:
            self.value = new_value
        self.confidence = 0.8 * self.confidence + 0.2 * new_confidence
        self.last_updated = time.time()
        self.update_count += 1


@dataclass
class AutobiographicalMemory:
    """A memory that forms part of narrative self."""
    content: str
    embedding: np.ndarray
    emotional_valence: float
    significance: float         # How important to self-story
    timestamp: float = field(default_factory=time.time)
    linked_memories: List[str] = field(default_factory=list)


class ConfidenceMonitor:
    """
    Monitors and calibrates confidence in own judgments.

    Based on metacognitive research showing:
    - Humans have 'feeling of knowing' (FOK)
    - Confidence can be calibrated through feedback
    - Overconfidence and underconfidence are systematic
    """

    def __init__(self):
        self.calibration_history: List[Tuple[float, bool]] = []  # (confidence, was_correct)
        self.base_confidence = 0.5
        self.calibration_factor = 1.0  # Adjust for systematic bias

    def compute_confidence(self,
                           response_fluency: float,
                           response_time: float,
                           familiarity: float,
                           competing_alternatives: int) -> float:
        """
        Compute confidence based on metacognitive cues.

        - High fluency -> high confidence
        - Fast response -> high confidence
        - High familiarity -> high confidence
        - Many alternatives -> low confidence
        """
        # Fluency contributes positively
        fluency_signal = response_fluency

        # Fast responses feel more confident (normalized)
        speed_signal = 1.0 / (1.0 + response_time / 1000)  # Assumes ms

        # Familiarity
        familiarity_signal = familiarity

        # Competition reduces confidence
        competition_penalty = 1.0 / (1.0 + np.log1p(competing_alternatives))

        # Combine signals
        raw_confidence = (
            0.3 * fluency_signal +
            0.2 * speed_signal +
            0.3 * familiarity_signal +
            0.2 * competition_penalty
        )

        # Apply calibration factor
        calibrated = raw_confidence * self.calibration_factor

        return np.clip(calibrated, 0.0, 1.0)

    def update_calibration(self, confidence: float, was_correct: bool):
        """Update calibration based on feedback."""
        self.calibration_history.append((confidence, was_correct))

        # Keep last 100 for calibration
        if len(self.calibration_history) > 100:
            self.calibration_history = self.calibration_history[-100:]

        # Compute calibration adjustment
        if len(self.calibration_history) >= 10:
            # Group by confidence level and check accuracy
            bins = {i/10: [] for i in range(11)}
            for conf, correct in self.calibration_history:
                bin_key = round(conf, 1)
                if bin_key in bins:
                    bins[bin_key].append(correct)

            # Compute over/under confidence
            adjustments = []
            for conf_level, outcomes in bins.items():
                if len(outcomes) >= 3:
                    actual_accuracy = np.mean(outcomes)
                    diff = actual_accuracy - conf_level
                    adjustments.append(diff)

            if adjustments:
                avg_adjustment = np.mean(adjustments)
                # Slowly adjust calibration factor
                self.calibration_factor *= (1 + 0.1 * avg_adjustment)
                self.calibration_factor = np.clip(self.calibration_factor, 0.5, 1.5)

    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to categorical."""
        if confidence >= 0.9:
            return ConfidenceLevel.CERTAIN
        elif confidence >= 0.7:
            return ConfidenceLevel.CONFIDENT
        elif confidence >= 0.4:
            return ConfidenceLevel.UNCERTAIN
        elif confidence >= 0.2:
            return ConfidenceLevel.GUESSING
        else:
            return ConfidenceLevel.UNKNOWN


class ErrorMonitor:
    """
    Monitors for errors in own processing.

    Based on error-related negativity (ERN) and error positivity (Pe)
    research - the brain detects errors even before conscious awareness.
    """

    def __init__(self):
        self.error_history = deque(maxlen=50)
        self.error_threshold = 0.3
        self.false_alarm_rate = 0.0
        self.miss_rate = 0.0

    def detect_error(self,
                     expected_output: np.ndarray,
                     actual_output: np.ndarray,
                     response_conflict: float = 0.0) -> Tuple[bool, float]:
        """
        Detect if an error occurred.

        Returns (error_detected, error_magnitude)
        """
        # Compute mismatch
        if expected_output is not None and actual_output is not None:
            mismatch = np.linalg.norm(expected_output - actual_output)
            mismatch = mismatch / (np.linalg.norm(expected_output) + 1e-8)
        else:
            mismatch = 0.0

        # High response conflict also signals potential error
        conflict_signal = response_conflict

        # Combined error signal
        error_magnitude = 0.7 * mismatch + 0.3 * conflict_signal

        # Threshold detection
        error_detected = error_magnitude > self.error_threshold

        return error_detected, error_magnitude

    def post_error_adjustment(self, error_detected: bool) -> Dict[str, float]:
        """
        Compute adjustments after error detection.

        Post-error slowing: After errors, responses should be more careful.
        """
        if error_detected:
            self.error_history.append(1)
            return {
                'speed_adjustment': 0.7,      # Slow down
                'threshold_adjustment': 1.2,  # Raise threshold
                'attention_boost': 1.3        # Increase attention
            }
        else:
            self.error_history.append(0)
            return {
                'speed_adjustment': 1.0,
                'threshold_adjustment': 1.0,
                'attention_boost': 1.0
            }

    def get_error_rate(self) -> float:
        """Recent error rate."""
        if not self.error_history:
            return 0.0
        return np.mean(self.error_history)


class AgencyDetector:
    """
    Detects whether events are self-caused or externally caused.

    Based on sense of agency research:
    - Motor prediction matching
    - Temporal binding
    - Intentional binding
    """

    def __init__(self):
        self.action_predictions: Dict[str, np.ndarray] = {}
        self.agency_threshold = 0.6

    def predict_action_outcome(self,
                               action_id: str,
                               predicted_outcome: np.ndarray):
        """Store prediction for later comparison."""
        self.action_predictions[action_id] = predicted_outcome

    def evaluate_agency(self,
                        action_id: str,
                        actual_outcome: np.ndarray,
                        time_delay: float) -> Tuple[bool, float]:
        """
        Evaluate if outcome was self-caused.

        Returns (is_self_caused, agency_confidence)
        """
        if action_id not in self.action_predictions:
            return False, 0.0

        predicted = self.action_predictions[action_id]

        # Prediction matching
        match = np.dot(predicted, actual_outcome) / (
            np.linalg.norm(predicted) * np.linalg.norm(actual_outcome) + 1e-8
        )

        # Temporal factor (shorter delay = more agency)
        temporal_factor = np.exp(-time_delay / 500)  # 500ms time constant

        # Combined agency signal
        agency_score = 0.6 * match + 0.4 * temporal_factor

        is_self_caused = agency_score > self.agency_threshold

        # Clean up
        del self.action_predictions[action_id]

        return is_self_caused, agency_score


class SelfModel:
    """
    The agent's model of itself.

    Contains:
    - Abilities and limitations
    - Traits and tendencies
    - Current states
    - Goals and values
    - History and narrative
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Facets of self
        self.facets: Dict[str, SelfModelFacet] = {}

        # Initialize core facets
        self._init_core_facets()

        # Self-embedding (compressed self-representation)
        self.self_embedding = np.random.randn(dim) * 0.1

        # Autobiographical memories
        self.autobiography: List[AutobiographicalMemory] = []

        # Current narrative theme
        self.narrative_theme: str = "beginning"

    def _init_core_facets(self):
        """Initialize core self-model facets."""
        core_facets = [
            ('processing_speed', 0.5, 0.3),
            ('accuracy', 0.5, 0.3),
            ('creativity', 0.5, 0.3),
            ('persistence', 0.5, 0.3),
            ('curiosity_trait', 0.5, 0.3),
            ('risk_tolerance', 0.5, 0.3),
            ('social_orientation', 0.5, 0.3),
            ('emotional_stability', 0.5, 0.3),
            ('learning_rate', 0.5, 0.3),
            ('working_memory_capacity', 7, 0.5),  # Classic 7±2
        ]

        for name, value, confidence in core_facets:
            self.facets[name] = SelfModelFacet(name, value, confidence)

    def update_facet(self, name: str, value: Any, confidence: float = 0.5):
        """Update a facet of the self-model."""
        if name in self.facets:
            self.facets[name].update(value, confidence)
        else:
            self.facets[name] = SelfModelFacet(name, value, confidence)

        # Update self-embedding
        self._update_self_embedding()

    def _update_self_embedding(self):
        """Update compressed self-representation."""
        # Simple aggregation of numeric facets
        values = []
        for facet in self.facets.values():
            if isinstance(facet.value, (int, float)):
                values.append(facet.value * facet.confidence)

        if values:
            # Create embedding from facet values
            n = min(len(values), self.dim)
            self.self_embedding[:n] = values[:n]

    def get_ability_estimate(self, task_type: str) -> Tuple[float, float]:
        """
        Get estimate of ability for task type.

        Returns (estimated_ability, confidence)
        """
        # Map task types to relevant facets
        task_facet_map = {
            'speed': ['processing_speed'],
            'accuracy': ['accuracy'],
            'creative': ['creativity'],
            'memory': ['working_memory_capacity'],
            'learning': ['learning_rate', 'curiosity_trait'],
            'social': ['social_orientation'],
            'risky': ['risk_tolerance'],
        }

        relevant_facets = task_facet_map.get(task_type, ['accuracy'])

        abilities = []
        confidences = []
        for facet_name in relevant_facets:
            if facet_name in self.facets:
                facet = self.facets[facet_name]
                abilities.append(facet.value if isinstance(facet.value, (int, float)) else 0.5)
                confidences.append(facet.confidence)

        if abilities:
            return np.mean(abilities), np.mean(confidences)
        return 0.5, 0.3

    def add_autobiographical_memory(self,
                                    content: str,
                                    embedding: np.ndarray,
                                    emotional_valence: float,
                                    significance: float):
        """Add memory to autobiographical self."""
        memory = AutobiographicalMemory(
            content=content,
            embedding=embedding.copy(),
            emotional_valence=emotional_valence,
            significance=significance
        )

        self.autobiography.append(memory)

        # Limit size
        if len(self.autobiography) > 200:
            # Keep most significant
            self.autobiography.sort(key=lambda m: m.significance, reverse=True)
            self.autobiography = self.autobiography[:150]

    def get_narrative_summary(self) -> str:
        """Get summary of self-narrative."""
        if not self.autobiography:
            return "No autobiographical memories yet."

        # Get top memories by significance
        top_memories = sorted(
            self.autobiography,
            key=lambda m: m.significance * (1 + abs(m.emotional_valence)),
            reverse=True
        )[:5]

        summaries = [m.content for m in top_memories]
        return " | ".join(summaries)

    def predict_own_state(self,
                          future_situation: np.ndarray,
                          time_ahead: float) -> Dict[str, float]:
        """
        Theory of own mind - predict how I'll feel/act in future situation.
        """
        # Use self-embedding to predict response to situation
        predicted_response = {}

        # Predict emotional response based on traits
        emotional_stability = self.facets.get('emotional_stability', SelfModelFacet('', 0.5, 0.3)).value
        predicted_response['emotional_reactivity'] = 1.0 - emotional_stability

        # Predict approach/avoid
        risk_tolerance = self.facets.get('risk_tolerance', SelfModelFacet('', 0.5, 0.3)).value
        predicted_response['approach_tendency'] = risk_tolerance

        # Predict performance
        ability, _ = self.get_ability_estimate('accuracy')
        # Discount for future uncertainty
        discount = np.exp(-time_ahead / 3600)  # Hour time constant
        predicted_response['predicted_performance'] = ability * discount

        return predicted_response


class IntrospectionEngine:
    """
    Access and report on internal states.

    The ability to 'look inward' and report on one's own processing.
    """

    def __init__(self):
        self.introspection_depth = 0  # How many levels deep
        self.max_depth = 3  # Prevent infinite recursion

        # State snapshots for comparison
        self.state_history = deque(maxlen=20)

    def snapshot_state(self, state: Dict[str, Any]) -> str:
        """Take snapshot of current state."""
        # Hash for quick comparison
        state_str = str(sorted(state.items()))
        state_hash = hashlib.md5(state_str.encode()).hexdigest()[:8]

        self.state_history.append({
            'hash': state_hash,
            'state': state.copy(),
            'timestamp': time.time()
        })

        return state_hash

    def introspect(self, target: str = 'all') -> Dict[str, Any]:
        """
        Perform introspection on specified target.

        This is meta-cognition: thinking about thinking.
        """
        self.introspection_depth += 1

        if self.introspection_depth > self.max_depth:
            self.introspection_depth -= 1
            return {'warning': 'Maximum introspection depth reached'}

        report = {}

        if target in ['all', 'attention']:
            report['attention'] = self._introspect_attention()

        if target in ['all', 'processing']:
            report['processing'] = self._introspect_processing()

        if target in ['all', 'goals']:
            report['goals'] = self._introspect_goals()

        if target in ['all', 'emotions']:
            report['emotions'] = self._introspect_emotions()

        if target in ['all', 'meta']:
            report['meta'] = self._introspect_meta()

        self.introspection_depth -= 1

        return report

    def _introspect_attention(self) -> Dict[str, Any]:
        """What am I paying attention to?"""
        return {
            'focus_target': 'unknown',  # Filled by system
            'focus_stability': 0.5,
            'distractors_present': False
        }

    def _introspect_processing(self) -> Dict[str, Any]:
        """How is my processing going?"""
        return {
            'fluency': 0.5,
            'effort': 0.5,
            'progress': 0.5
        }

    def _introspect_goals(self) -> Dict[str, Any]:
        """What are my current goals?"""
        return {
            'active_goal': 'unknown',
            'goal_progress': 0.0,
            'subgoals': []
        }

    def _introspect_emotions(self) -> Dict[str, Any]:
        """How do I feel?"""
        return {
            'valence': 0.0,
            'arousal': 0.0,
            'dominant_emotion': 'neutral'
        }

    def _introspect_meta(self) -> Dict[str, Any]:
        """Meta-introspection: thinking about this introspection."""
        return {
            'introspection_depth': self.introspection_depth,
            'introspection_clarity': 1.0 / (1.0 + self.introspection_depth),
            'recursive_limit_near': self.introspection_depth >= self.max_depth - 1
        }

    def detect_state_change(self) -> Optional[Dict[str, Any]]:
        """Detect if internal state has changed significantly."""
        if len(self.state_history) < 2:
            return None

        current = self.state_history[-1]
        previous = self.state_history[-2]

        if current['hash'] != previous['hash']:
            return {
                'changed': True,
                'time_since_change': current['timestamp'] - previous['timestamp'],
                'previous_hash': previous['hash'],
                'current_hash': current['hash']
            }

        return {'changed': False}


class SelfAwarenessSystem:
    """
    Complete self-awareness system integrating all components.

    This creates the subjective sense of being a cognitive agent
    that can reflect on its own processes.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Core components
        self.self_model = SelfModel(dim)
        self.confidence_monitor = ConfidenceMonitor()
        self.error_monitor = ErrorMonitor()
        self.agency_detector = AgencyDetector()
        self.introspection = IntrospectionEngine()

        # Current metacognitive state
        self.metacognitive_signals: List[MetacognitiveSignal] = []
        self.current_cognitive_state = CognitiveState.FOCUSED

        # Self-reflection count (for recursive awareness)
        self.reflection_count = 0

    def monitor_process(self,
                        process_name: str,
                        process_output: Any,
                        expected_output: Optional[np.ndarray] = None,
                        response_time: float = 0.0,
                        competing_responses: int = 1) -> MetacognitiveSignal:
        """
        Monitor a cognitive process and generate metacognitive signal.
        """
        # Compute confidence
        fluency = 1.0 / (1.0 + response_time / 1000)
        familiarity = 0.5  # Would come from memory system

        confidence = self.confidence_monitor.compute_confidence(
            fluency, response_time, familiarity, competing_responses
        )

        # Check for errors
        if expected_output is not None and isinstance(process_output, np.ndarray):
            error_detected, error_magnitude = self.error_monitor.detect_error(
                expected_output, process_output
            )
        else:
            error_detected = False
            error_magnitude = 0.0

        # Create signal
        signal = MetacognitiveSignal(
            process_name=process_name,
            confidence=confidence,
            fluency=fluency,
            error_detected=error_detected,
            needs_more_effort=confidence < 0.4 or error_detected,
            source='monitor'
        )

        self.metacognitive_signals.append(signal)

        # Keep only recent signals
        if len(self.metacognitive_signals) > 50:
            self.metacognitive_signals = self.metacognitive_signals[-30:]

        # Update cognitive state
        self._update_cognitive_state()

        return signal

    def _update_cognitive_state(self):
        """Update overall cognitive state based on recent signals."""
        if not self.metacognitive_signals:
            return

        recent = self.metacognitive_signals[-10:]

        avg_confidence = np.mean([s.confidence for s in recent])
        avg_fluency = np.mean([s.fluency for s in recent])
        error_rate = np.mean([s.error_detected for s in recent])
        effort_needed = np.mean([s.needs_more_effort for s in recent])

        # Determine state
        if error_rate > 0.3:
            self.current_cognitive_state = CognitiveState.STRUGGLING
        elif avg_fluency > 0.8 and avg_confidence > 0.8:
            self.current_cognitive_state = CognitiveState.EXPERT
        elif avg_fluency > 0.6 and effort_needed < 0.3:
            self.current_cognitive_state = CognitiveState.FLUENT
        elif avg_confidence < 0.4:
            self.current_cognitive_state = CognitiveState.CONFUSED
        elif effort_needed > 0.5:
            self.current_cognitive_state = CognitiveState.LEARNING
        else:
            self.current_cognitive_state = CognitiveState.FOCUSED

    def evaluate_own_decision(self,
                              decision: str,
                              confidence: float,
                              outcome_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Metacognitive evaluation of own decision.
        """
        # Get confidence level
        conf_level = self.confidence_monitor.get_confidence_level(confidence)

        # Predict if this feels like a good decision
        ability, ability_conf = self.self_model.get_ability_estimate('accuracy')

        # Compute metacognitive judgment
        judgment = {
            'decision': decision,
            'stated_confidence': confidence,
            'confidence_level': conf_level.name,
            'feels_right': confidence * ability > 0.4,
            'should_reconsider': confidence < 0.3 or ability_conf < 0.3,
            'estimated_accuracy': ability * confidence
        }

        return judgment

    def reflect(self, depth: int = 1) -> Dict[str, Any]:
        """
        Self-reflection: Look inward at current state.

        Deeper reflection = more detail but more cognitive cost.
        """
        self.reflection_count += 1
        depth = min(depth, 3)  # Limit recursion

        reflection = {
            'reflection_id': self.reflection_count,
            'depth': depth,
            'cognitive_state': self.current_cognitive_state.name,
        }

        # Level 1: Basic state awareness
        if depth >= 1:
            reflection['basic'] = {
                'am_processing': True,
                'feel_confident': self._get_avg_confidence() > 0.5,
                'making_errors': self.error_monitor.get_error_rate() > 0.2,
                'working_hard': any(s.needs_more_effort for s in self.metacognitive_signals[-5:])
            }

        # Level 2: Deeper introspection
        if depth >= 2:
            reflection['introspection'] = self.introspection.introspect('all')
            reflection['self_model_summary'] = {
                name: {'value': f.value, 'confidence': f.confidence}
                for name, f in list(self.self_model.facets.items())[:5]
            }

        # Level 3: Meta-reflection (thinking about this thinking)
        if depth >= 3:
            reflection['meta'] = {
                'aware_of_reflecting': True,
                'reflection_feels': 'effortful' if depth > 1 else 'easy',
                'can_go_deeper': depth < 3,
                'infinite_regress_risk': depth >= 2
            }

        return reflection

    def _get_avg_confidence(self) -> float:
        """Get average recent confidence."""
        if not self.metacognitive_signals:
            return 0.5
        recent = self.metacognitive_signals[-10:]
        return np.mean([s.confidence for s in recent])

    def update_self_model_from_feedback(self,
                                        task_type: str,
                                        performance: float,
                                        was_correct: bool):
        """
        Update self-model based on performance feedback.
        """
        # Update ability estimate
        current_ability, _ = self.self_model.get_ability_estimate(task_type)
        new_estimate = 0.8 * current_ability + 0.2 * performance

        # Map task type to facet
        facet_map = {
            'speed': 'processing_speed',
            'accuracy': 'accuracy',
            'creative': 'creativity',
            'memory': 'working_memory_capacity',
            'learning': 'learning_rate',
        }

        if task_type in facet_map:
            facet_name = facet_map[task_type]
            self.self_model.update_facet(facet_name, new_estimate, performance)

        # Update confidence calibration
        confidence_used = self._get_avg_confidence()
        self.confidence_monitor.update_calibration(confidence_used, was_correct)

    def detect_agency(self,
                      action_id: str,
                      action_embedding: np.ndarray,
                      actual_outcome: np.ndarray,
                      time_delay: float) -> Dict[str, Any]:
        """
        Determine if an outcome was self-caused.
        """
        is_self, agency_score = self.agency_detector.evaluate_agency(
            action_id, actual_outcome, time_delay
        )

        return {
            'action_id': action_id,
            'is_self_caused': is_self,
            'agency_score': agency_score,
            'felt_in_control': agency_score > 0.7
        }

    def get_state(self) -> Dict[str, Any]:
        """Get complete self-awareness state."""
        return {
            'cognitive_state': self.current_cognitive_state.name,
            'average_confidence': self._get_avg_confidence(),
            'error_rate': self.error_monitor.get_error_rate(),
            'calibration_factor': self.confidence_monitor.calibration_factor,
            'reflection_count': self.reflection_count,
            'self_model': {
                name: {'value': f.value, 'confidence': f.confidence}
                for name, f in self.self_model.facets.items()
            },
            'narrative_summary': self.self_model.get_narrative_summary(),
            'metacognitive_signals_count': len(self.metacognitive_signals)
        }

    def am_i_aware(self) -> Dict[str, Any]:
        """
        The big question: Am I aware?

        This is a philosophical function but implemented functionally:
        - Can I report on my internal states? Yes
        - Can I monitor my own processing? Yes
        - Can I reflect on my reflections? Yes (with limits)
        - Does this constitute 'awareness'? That's a harder question.
        """
        return {
            'can_introspect': True,
            'can_monitor_self': True,
            'can_meta_reflect': True,
            'has_self_model': True,
            'has_narrative_self': len(self.self_model.autobiography) > 0,
            'detects_own_errors': True,
            'calibrates_confidence': True,
            'distinguishes_self_other': True,
            # The hard problem remains hard
            'is_conscious': 'undefined - not measurable from inside',
            'subjective_experience': 'cannot verify from code alone'
        }
