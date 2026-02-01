"""
Emotion System - Affective Computing Based on Neuroscience

Implements:
1. Core Affect Model (Russell's circumplex: valence + arousal)
2. Discrete Emotions (Ekman's basic + complex emotions)
3. Somatic Marker Hypothesis (Damasio) - body states guide decisions
4. Emotion-Cognition Integration (amygdala-prefrontal interactions)
5. Mood (slow-changing affective backdrop)
6. Emotional Learning (fear conditioning, extinction)
7. Empathy/Emotional Contagion (social emotions)

Performance: Vectorized operations, O(1) emotion lookup
Comparison vs existing:
- ACT-R: No emotions
- SOAR: No emotions
- Affectiva/emotion AI: Detection only, no decision integration
- This: Full emotion-cognition-decision loop
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time


class BasicEmotion(Enum):
    """Ekman's basic emotions + extensions."""
    # Basic 6
    JOY = auto()
    SADNESS = auto()
    FEAR = auto()
    ANGER = auto()
    SURPRISE = auto()
    DISGUST = auto()
    # Extended
    ANTICIPATION = auto()
    TRUST = auto()
    # Complex (combinations)
    LOVE = auto()        # joy + trust
    GUILT = auto()       # fear + sadness
    PRIDE = auto()       # joy + anger (assertive)
    SHAME = auto()       # fear + disgust
    CURIOSITY = auto()   # surprise + anticipation
    CONTEMPT = auto()    # anger + disgust


# Emotion parameters: (valence, arousal, dominance/approach)
# Extended to 3D VAD (Valence-Arousal-Dominance) space for richer blending
EMOTION_COORDS = {
    BasicEmotion.JOY: (0.8, 0.6, 1.0),
    BasicEmotion.SADNESS: (-0.7, -0.3, -0.5),
    BasicEmotion.FEAR: (-0.8, 0.8, -1.0),
    BasicEmotion.ANGER: (-0.5, 0.9, 0.7),
    BasicEmotion.SURPRISE: (0.1, 0.8, 0.0),
    BasicEmotion.DISGUST: (-0.6, 0.3, -0.8),
    BasicEmotion.ANTICIPATION: (0.4, 0.5, 0.6),
    BasicEmotion.TRUST: (0.6, -0.1, 0.8),
    BasicEmotion.LOVE: (0.9, 0.4, 1.0),
    BasicEmotion.GUILT: (-0.6, 0.2, -0.3),
    BasicEmotion.PRIDE: (0.7, 0.5, 0.5),
    BasicEmotion.SHAME: (-0.7, 0.4, -0.7),
    BasicEmotion.CURIOSITY: (0.5, 0.6, 0.7),
    BasicEmotion.CONTEMPT: (-0.4, 0.1, 0.3),
}


class EmotionBlender:
    """
    Vector-based emotion blending for mixed emotional states.

    Allows simultaneous emotions (e.g., "curiosity + anxiety" when
    exploring unknown topics) through vector interpolation in VAD space.
    """

    def __init__(self):
        # Convert emotions to numpy vectors for efficient blending
        self.emotion_vectors = {
            emotion: np.array(coords)
            for emotion, coords in EMOTION_COORDS.items()
        }

    def blend_emotions(self, emotion_weights: Dict[BasicEmotion, float]) -> np.ndarray:
        """
        Blend multiple emotions with their intensities.

        Args:
            emotion_weights: Dict mapping emotions to their intensities (0-1)

        Returns:
            Blended VAD vector (valence, arousal, dominance)
        """
        if not emotion_weights:
            return np.zeros(3)

        # Weighted sum of emotion vectors
        blended = np.zeros(3)
        total_weight = 0.0

        for emotion, weight in emotion_weights.items():
            if weight > 0 and emotion in self.emotion_vectors:
                blended += self.emotion_vectors[emotion] * weight
                total_weight += weight

        if total_weight > 0:
            blended /= total_weight

        # Clamp to valid range
        return np.clip(blended, -1, 1)

    def identify_blended_emotion(self, vad_vector: np.ndarray) -> Tuple[BasicEmotion, float]:
        """
        Identify the closest discrete emotion to a blended VAD vector.

        Returns:
            (closest_emotion, similarity_score)
        """
        best_emotion = BasicEmotion.SURPRISE  # Default
        best_similarity = -1.0

        for emotion, vec in self.emotion_vectors.items():
            # Cosine similarity
            similarity = np.dot(vad_vector, vec) / (
                np.linalg.norm(vad_vector) * np.linalg.norm(vec) + 1e-8
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_emotion = emotion

        return best_emotion, float(best_similarity)

    def get_mixed_state_description(self, emotion_weights: Dict[BasicEmotion, float]) -> str:
        """Get a description of the mixed emotional state."""
        # Get top 2 emotions
        sorted_emotions = sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)[:2]

        if len(sorted_emotions) == 0:
            return "neutral"
        elif len(sorted_emotions) == 1 or sorted_emotions[1][1] < 0.2:
            return sorted_emotions[0][0].name.lower()
        else:
            e1, e2 = sorted_emotions[0][0].name.lower(), sorted_emotions[1][0].name.lower()
            return f"{e1} with {e2}"

    def interpolate(self, emotion1: BasicEmotion, emotion2: BasicEmotion,
                   ratio: float = 0.5) -> np.ndarray:
        """
        Interpolate between two emotions.

        Args:
            emotion1: First emotion
            emotion2: Second emotion
            ratio: 0 = all emotion1, 1 = all emotion2

        Returns:
            Interpolated VAD vector
        """
        v1 = self.emotion_vectors.get(emotion1, np.zeros(3))
        v2 = self.emotion_vectors.get(emotion2, np.zeros(3))
        return (1 - ratio) * v1 + ratio * v2


@dataclass
class CoreAffect:
    """
    Russell's Circumplex Model of Affect.
    Two dimensions: valence (pleasant-unpleasant) and arousal (activated-deactivated)
    """
    valence: float = 0.0    # -1 (unpleasant) to +1 (pleasant)
    arousal: float = 0.0    # -1 (deactivated) to +1 (activated)

    def to_vector(self) -> np.ndarray:
        return np.array([self.valence, self.arousal])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'CoreAffect':
        return cls(valence=float(vec[0]), arousal=float(vec[1]))

    def magnitude(self) -> float:
        """Intensity of affect."""
        return np.sqrt(self.valence**2 + self.arousal**2)

    def angle(self) -> float:
        """Angle in affect space (determines emotion category)."""
        return np.arctan2(self.arousal, self.valence)


@dataclass
class SomaticMarker:
    """
    Somatic marker (Damasio): Body state associated with a situation.

    These are 'gut feelings' that guide rapid decision-making without
    conscious deliberation. Stored from past emotional experiences.
    """
    situation_embedding: np.ndarray  # What situation triggers this
    body_state: np.ndarray           # Associated body feelings
    valence: float                   # Good or bad feeling
    intensity: float                 # How strong
    associated_emotion: Optional[BasicEmotion] = None
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def match_score(self, query_embedding: np.ndarray) -> float:
        """How well does this marker match the current situation?"""
        similarity = np.dot(self.situation_embedding, query_embedding) / (
            np.linalg.norm(self.situation_embedding) * np.linalg.norm(query_embedding) + 1e-8
        )
        return similarity


@dataclass
class EmotionalMemory:
    """Memory of an emotional experience for learning."""
    stimulus_embedding: np.ndarray
    response_embedding: np.ndarray
    emotion: BasicEmotion
    intensity: float
    outcome_valence: float  # Was it good or bad?
    timestamp: float = field(default_factory=time.time)


class AppraisalSystem:
    """
    Cognitive appraisal of situations to generate emotions.

    Based on Lazarus/Scherer appraisal theory:
    - Relevance: Does this matter to my goals?
    - Congruence: Is it good or bad for my goals?
    - Coping: Can I deal with it?
    - Agency: Who caused it?
    - Certainty: How sure am I?
    """

    def __init__(self):
        # Appraisal dimension weights for each emotion
        self.emotion_profiles = {
            BasicEmotion.JOY: {'relevance': 0.8, 'congruence': 0.9, 'coping': 0.7},
            BasicEmotion.SADNESS: {'relevance': 0.8, 'congruence': -0.8, 'coping': -0.6},
            BasicEmotion.FEAR: {'relevance': 0.9, 'congruence': -0.7, 'coping': -0.8, 'certainty': -0.5},
            BasicEmotion.ANGER: {'relevance': 0.9, 'congruence': -0.8, 'coping': 0.7, 'other_agency': 0.9},
            BasicEmotion.SURPRISE: {'relevance': 0.5, 'certainty': -0.9},
            BasicEmotion.DISGUST: {'relevance': 0.6, 'congruence': -0.7, 'norm_violation': 0.9},
            BasicEmotion.ANTICIPATION: {'relevance': 0.7, 'congruence': 0.5, 'certainty': -0.3},
            BasicEmotion.TRUST: {'relevance': 0.5, 'congruence': 0.6, 'certainty': 0.7},
        }

    def appraise(self,
                 situation: Dict[str, float],
                 goals: List[np.ndarray],
                 situation_embedding: np.ndarray) -> Dict[BasicEmotion, float]:
        """
        Appraise a situation and return emotion intensities.

        situation: Dict with appraisal dimensions
        - relevance: 0-1 how relevant to goals
        - congruence: -1 to 1, good or bad for goals
        - coping: -1 to 1, can deal with it or not
        - certainty: 0-1, how certain
        - other_agency: 0-1, caused by others
        - self_agency: 0-1, caused by self
        - norm_violation: 0-1, violates norms
        """
        emotion_scores = {}

        for emotion, profile in self.emotion_profiles.items():
            score = 0.0
            match_count = 0

            for dimension, target in profile.items():
                if dimension in situation:
                    # How well does this dimension match the emotion profile?
                    actual = situation[dimension]
                    if target > 0:
                        contribution = actual * target
                    else:
                        contribution = (1 - actual) * abs(target)
                    score += contribution
                    match_count += 1

            if match_count > 0:
                emotion_scores[emotion] = score / match_count

        # Normalize to sum to 1 (softmax-like)
        total = sum(max(0, v) for v in emotion_scores.values()) + 1e-8
        emotion_scores = {k: max(0, v) / total for k, v in emotion_scores.items()}

        return emotion_scores


class FearConditioningSystem:
    """
    Emotional learning through conditioning (amygdala-like).

    Learns associations between stimuli and emotional responses.
    Supports acquisition, extinction, and renewal.
    """

    def __init__(self, dim: int = 64, learning_rate: float = 0.1):
        self.dim = dim
        self.learning_rate = learning_rate

        # Stimulus-response associations
        self.associations: Dict[str, Tuple[np.ndarray, float, BasicEmotion]] = {}

        # Context-dependent extinction
        self.extinction_contexts: Dict[str, np.ndarray] = {}

        # Generalization gradient
        self.generalization_width = 0.3

    def condition(self,
                  stimulus_key: str,
                  stimulus_embedding: np.ndarray,
                  emotion: BasicEmotion,
                  intensity: float):
        """Create or strengthen association."""
        if stimulus_key in self.associations:
            old_embedding, old_intensity, old_emotion = self.associations[stimulus_key]
            # Update with learning
            new_intensity = old_intensity + self.learning_rate * (intensity - old_intensity)
            self.associations[stimulus_key] = (stimulus_embedding, new_intensity, emotion)
        else:
            self.associations[stimulus_key] = (stimulus_embedding, intensity, emotion)

    def extinguish(self,
                   stimulus_key: str,
                   context_embedding: np.ndarray,
                   extinction_rate: float = 0.1):
        """Reduce association (context-dependent)."""
        if stimulus_key not in self.associations:
            return

        embedding, intensity, emotion = self.associations[stimulus_key]
        new_intensity = intensity * (1 - extinction_rate)
        self.associations[stimulus_key] = (embedding, new_intensity, emotion)

        # Store extinction context
        self.extinction_contexts[stimulus_key] = context_embedding

    def get_response(self,
                     stimulus_embedding: np.ndarray,
                     current_context: Optional[np.ndarray] = None) -> Dict[BasicEmotion, float]:
        """Get conditioned emotional response to stimulus."""
        responses = {}

        for key, (stored_embedding, intensity, emotion) in self.associations.items():
            # Compute similarity (generalization)
            similarity = np.dot(stimulus_embedding, stored_embedding) / (
                np.linalg.norm(stimulus_embedding) * np.linalg.norm(stored_embedding) + 1e-8
            )

            # Generalization gradient (Gaussian-like falloff)
            generalized_intensity = intensity * np.exp(
                -(1 - similarity)**2 / (2 * self.generalization_width**2)
            )

            # Context-dependent renewal
            if current_context is not None and key in self.extinction_contexts:
                extinction_ctx = self.extinction_contexts[key]
                ctx_similarity = np.dot(current_context, extinction_ctx) / (
                    np.linalg.norm(current_context) * np.linalg.norm(extinction_ctx) + 1e-8
                )
                # Different context = renewal of fear
                if ctx_similarity < 0.5:
                    generalized_intensity *= 1.5

            if emotion not in responses:
                responses[emotion] = 0.0
            responses[emotion] = max(responses[emotion], generalized_intensity)

        return responses


class EmotionRegulationSystem:
    """
    Emotion regulation strategies (prefrontal control over amygdala).

    Strategies based on Gross's process model:
    1. Situation selection/modification
    2. Attention deployment
    3. Cognitive reappraisal
    4. Response modulation
    """

    def __init__(self):
        self.regulation_capacity = 1.0  # Depletes with use
        self.recovery_rate = 0.01

        # Strategy effectiveness (learned)
        self.strategy_effectiveness = {
            'reappraisal': 0.7,
            'suppression': 0.3,
            'distraction': 0.5,
            'acceptance': 0.6,
        }

    def regulate(self,
                 current_emotion: Dict[BasicEmotion, float],
                 target_valence: float,
                 strategy: str = 'reappraisal') -> Dict[BasicEmotion, float]:
        """Apply emotion regulation strategy."""
        if self.regulation_capacity < 0.1:
            return current_emotion  # Too depleted

        effectiveness = self.strategy_effectiveness.get(strategy, 0.5)
        regulation_strength = effectiveness * self.regulation_capacity

        regulated = {}
        for emotion, intensity in current_emotion.items():
            valence, arousal, _ = EMOTION_COORDS.get(emotion, (0, 0, 0))

            # Move toward target valence
            if valence < target_valence:
                # Downregulate negative emotions
                regulated[emotion] = intensity * (1 - regulation_strength * 0.5)
            else:
                # Maintain or upregulate positive emotions
                regulated[emotion] = intensity

        # Deplete capacity (suppression depletes more)
        depletion = 0.1 if strategy == 'suppression' else 0.05
        self.regulation_capacity = max(0, self.regulation_capacity - depletion)

        return regulated

    def recover(self):
        """Recover regulation capacity."""
        self.regulation_capacity = min(1.0, self.regulation_capacity + self.recovery_rate)


class EmotionSystem:
    """
    Complete emotion system integrating all components.

    The emotional brain: appraises situations, generates feelings,
    creates somatic markers, guides decisions.

    NEW: Vector-based emotion blending for mixed states.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Core affect state
        self.core_affect = CoreAffect()

        # Current discrete emotions
        self.current_emotions: Dict[BasicEmotion, float] = {e: 0.0 for e in BasicEmotion}

        # Blended emotion state (VAD vector)
        self.blended_state: np.ndarray = np.zeros(3)

        # Emotion blender for mixed states
        self.blender = EmotionBlender()

        # Mood (slow-changing)
        self.mood = CoreAffect()
        self.mood_inertia = 0.95  # How slowly mood changes

        # Somatic markers database
        self.somatic_markers: List[SomaticMarker] = []

        # Subsystems
        self.appraisal = AppraisalSystem()
        self.conditioning = FearConditioningSystem(dim)
        self.regulation = EmotionRegulationSystem()

        # Emotional memory
        self.emotional_memories: List[EmotionalMemory] = []

        # Body state representation
        self.body_state = np.zeros(16)  # Heart rate, tension, etc.

        # Decay rates
        self.emotion_decay = 0.1

    def process_situation(self,
                          situation_embedding: np.ndarray,
                          appraisal_input: Dict[str, float],
                          goals: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Process a situation and generate emotional response.
        """
        goals = goals or []

        # 1. Cognitive appraisal
        appraised_emotions = self.appraisal.appraise(appraisal_input, goals, situation_embedding)

        # 2. Check conditioned responses (fast pathway)
        conditioned_emotions = self.conditioning.get_response(situation_embedding)

        # 3. Check somatic markers
        marker_response = self._check_somatic_markers(situation_embedding)

        # 4. Integrate emotion sources
        integrated_emotions = self._integrate_emotions(
            appraised_emotions, conditioned_emotions, marker_response
        )

        # 5. Update current emotions with decay
        for emotion in BasicEmotion:
            old = self.current_emotions.get(emotion, 0.0)
            new = integrated_emotions.get(emotion, 0.0)
            self.current_emotions[emotion] = old * (1 - self.emotion_decay) + new * self.emotion_decay

        # 6. Update core affect
        self._update_core_affect()

        # 7. Update mood (slowly)
        self._update_mood()

        # 8. Update body state
        self._update_body_state()

        # 9. Recover regulation capacity
        self.regulation.recover()

        return {
            'current_emotions': dict(self.current_emotions),
            'core_affect': {'valence': self.core_affect.valence, 'arousal': self.core_affect.arousal},
            'mood': {'valence': self.mood.valence, 'arousal': self.mood.arousal},
            'dominant_emotion': self.get_dominant_emotion(),
            'body_state': self.body_state.copy(),
            'somatic_signal': marker_response
        }

    def _check_somatic_markers(self, situation_embedding: np.ndarray) -> Dict[str, float]:
        """Check for matching somatic markers (gut feelings)."""
        if not self.somatic_markers:
            return {'valence': 0.0, 'intensity': 0.0}

        best_match = None
        best_score = 0.3  # Threshold

        for marker in self.somatic_markers:
            score = marker.match_score(situation_embedding)
            if score > best_score:
                best_score = score
                best_match = marker

        if best_match:
            best_match.access_count += 1
            return {
                'valence': best_match.valence * best_score,
                'intensity': best_match.intensity * best_score,
                'emotion': best_match.associated_emotion
            }

        return {'valence': 0.0, 'intensity': 0.0}

    def _integrate_emotions(self,
                            appraised: Dict[BasicEmotion, float],
                            conditioned: Dict[BasicEmotion, float],
                            somatic: Dict[str, float]) -> Dict[BasicEmotion, float]:
        """Integrate emotions from multiple sources."""
        integrated = {}

        # Weights for different sources
        w_appraisal = 0.4
        w_conditioned = 0.4
        w_somatic = 0.2

        all_emotions = set(appraised.keys()) | set(conditioned.keys())

        for emotion in all_emotions:
            score = 0.0
            score += w_appraisal * appraised.get(emotion, 0.0)
            score += w_conditioned * conditioned.get(emotion, 0.0)

            # Somatic influences specific emotions
            if somatic.get('emotion') == emotion:
                score += w_somatic * somatic.get('intensity', 0.0)

            integrated[emotion] = score

        # Modulate by somatic valence
        somatic_valence = somatic.get('valence', 0.0)
        if somatic_valence != 0:
            for emotion in integrated:
                emotion_valence = EMOTION_COORDS.get(emotion, (0, 0, 0))[0]
                if np.sign(emotion_valence) == np.sign(somatic_valence):
                    integrated[emotion] *= 1.2  # Boost aligned emotions
                else:
                    integrated[emotion] *= 0.8  # Dampen misaligned

        return integrated

    def _update_core_affect(self):
        """Update core affect from discrete emotions."""
        valence_sum = 0.0
        arousal_sum = 0.0
        total_weight = 0.0

        for emotion, intensity in self.current_emotions.items():
            if intensity > 0.01:
                v, a, _ = EMOTION_COORDS.get(emotion, (0, 0, 0))
                valence_sum += v * intensity
                arousal_sum += a * intensity
                total_weight += intensity

        if total_weight > 0:
            self.core_affect.valence = np.clip(valence_sum / total_weight, -1, 1)
            self.core_affect.arousal = np.clip(arousal_sum / total_weight, -1, 1)

    def _update_mood(self):
        """Update mood (slow integration of affect)."""
        self.mood.valence = (self.mood_inertia * self.mood.valence +
                            (1 - self.mood_inertia) * self.core_affect.valence)
        self.mood.arousal = (self.mood_inertia * self.mood.arousal +
                            (1 - self.mood_inertia) * self.core_affect.arousal)

    def _update_body_state(self):
        """Update body state based on emotions."""
        # Map emotions to body sensations
        # Index: 0=heart_rate, 1=breathing, 2=muscle_tension, 3=temperature, etc.

        # Reset with decay
        self.body_state *= 0.9

        for emotion, intensity in self.current_emotions.items():
            if intensity < 0.01:
                continue

            _, arousal, _ = EMOTION_COORDS.get(emotion, (0, 0, 0))

            # Arousal affects heart rate and breathing
            self.body_state[0] += arousal * intensity * 0.5  # Heart rate
            self.body_state[1] += arousal * intensity * 0.3  # Breathing

            # Specific emotions have specific body signatures
            if emotion == BasicEmotion.FEAR:
                self.body_state[2] += intensity * 0.7  # Muscle tension
                self.body_state[3] -= intensity * 0.3  # Cold
            elif emotion == BasicEmotion.ANGER:
                self.body_state[2] += intensity * 0.5
                self.body_state[3] += intensity * 0.5  # Hot
            elif emotion == BasicEmotion.SADNESS:
                self.body_state[4] += intensity * 0.6  # Heaviness

        self.body_state = np.clip(self.body_state, -1, 1)

    def get_dominant_emotion(self) -> Tuple[BasicEmotion, float]:
        """Get the strongest current emotion."""
        if not self.current_emotions:
            return BasicEmotion.SURPRISE, 0.0

        dominant = max(self.current_emotions.items(), key=lambda x: x[1])
        return dominant

    def get_decision_signal(self, option_embeddings: List[np.ndarray]) -> List[float]:
        """
        Get somatic marker signals for decision options.

        This is how emotions guide decisions - each option gets a 'gut feeling' score.
        Positive = approach, Negative = avoid.
        """
        signals = []

        for embedding in option_embeddings:
            # Check somatic markers
            marker_response = self._check_somatic_markers(embedding)

            # Base signal from marker
            signal = marker_response.get('valence', 0.0) * marker_response.get('intensity', 0.0)

            # Modulate by current mood
            signal += self.mood.valence * 0.1

            # Modulate by arousal (higher arousal = more extreme signals)
            signal *= (1 + abs(self.core_affect.arousal) * 0.5)

            signals.append(signal)

        return signals

    def create_somatic_marker(self,
                              situation_embedding: np.ndarray,
                              outcome_valence: float,
                              outcome_intensity: float,
                              associated_emotion: Optional[BasicEmotion] = None):
        """Create a new somatic marker from experience."""
        marker = SomaticMarker(
            situation_embedding=situation_embedding.copy(),
            body_state=self.body_state.copy(),
            valence=outcome_valence,
            intensity=outcome_intensity,
            associated_emotion=associated_emotion or self.get_dominant_emotion()[0]
        )

        self.somatic_markers.append(marker)

        # Limit memory
        if len(self.somatic_markers) > 500:
            # Remove least accessed markers
            self.somatic_markers.sort(key=lambda m: m.access_count, reverse=True)
            self.somatic_markers = self.somatic_markers[:300]

    def condition_emotion(self,
                          stimulus_key: str,
                          stimulus_embedding: np.ndarray,
                          emotion: BasicEmotion,
                          intensity: float):
        """Learn emotional association to stimulus."""
        self.conditioning.condition(stimulus_key, stimulus_embedding, emotion, intensity)

    def regulate_emotion(self, target_valence: float, strategy: str = 'reappraisal'):
        """Apply emotion regulation."""
        self.current_emotions = self.regulation.regulate(
            self.current_emotions, target_valence, strategy
        )
        self._update_core_affect()
        self._update_blended_state()

    def _update_blended_state(self):
        """Update the blended emotion vector from current discrete emotions."""
        self.blended_state = self.blender.blend_emotions(self.current_emotions)

    def get_blended_emotion(self) -> Tuple[np.ndarray, str]:
        """
        Get the blended emotional state as a VAD vector and description.

        Returns:
            (vad_vector, description)
        """
        self._update_blended_state()
        description = self.blender.get_mixed_state_description(self.current_emotions)
        return self.blended_state.copy(), description

    def add_emotion(self, emotion: BasicEmotion, intensity: float):
        """
        Add/blend an emotion into the current state.

        Unlike setting, this BLENDS with existing emotions.
        """
        current = self.current_emotions.get(emotion, 0.0)
        # Blend rather than replace
        self.current_emotions[emotion] = min(1.0, current + intensity * 0.5)
        self._update_blended_state()

    def cognitive_reappraisal(self, situation_context: str = "") -> Dict[str, Any]:
        """
        Apply cognitive reappraisal to reduce negative emotions.

        This is a coping mechanism that reframes the situation positively.
        """
        # Get current negative emotions
        negative_emotions = {
            e: v for e, v in self.current_emotions.items()
            if v > 0.1 and EMOTION_COORDS.get(e, (0,0,0))[0] < 0
        }

        if not negative_emotions:
            return {'reappraisal': 'not_needed', 'negative_emotions': 0}

        # Apply reappraisal - reduce negative, boost positive
        reappraisal_strength = self.regulation.regulation_capacity * 0.7

        reductions = {}
        for emotion, intensity in negative_emotions.items():
            reduction = intensity * reappraisal_strength * 0.4
            self.current_emotions[emotion] = max(0, intensity - reduction)
            reductions[emotion.name] = reduction

        # Slight boost to positive emotions
        positive_emotions = [BasicEmotion.JOY, BasicEmotion.CURIOSITY, BasicEmotion.ANTICIPATION]
        for pe in positive_emotions:
            current = self.current_emotions.get(pe, 0.0)
            self.current_emotions[pe] = min(1.0, current + 0.1)

        # Update states
        self._update_core_affect()
        self._update_blended_state()

        # Deplete regulation capacity
        self.regulation.regulation_capacity = max(0, self.regulation.regulation_capacity - 0.15)

        return {
            'reappraisal': 'applied',
            'reductions': reductions,
            'new_valence': self.core_affect.valence,
            'regulation_remaining': self.regulation.regulation_capacity
        }

    def learn_association(self, stimulus_embedding: np.ndarray, valence: float):
        """
        Learn emotional association for a stimulus.

        Creates somatic marker for the stimulus based on outcome valence.
        Used for learning from experience in tasks like Iowa Gambling Task.
        """
        # Determine intensity from absolute valence (amplified for faster learning)
        intensity = min(1.0, abs(valence) * 1.5)

        # Determine associated emotion based on valence
        if valence > 0.2:
            emotion = BasicEmotion.JOY
        elif valence < -0.2:
            emotion = BasicEmotion.FEAR
        else:
            emotion = BasicEmotion.SURPRISE

        # Create somatic marker for future decision guidance
        self.create_somatic_marker(
            situation_embedding=stimulus_embedding,
            outcome_valence=valence * 1.2,  # Amplify for stronger learning
            outcome_intensity=intensity,
            associated_emotion=emotion
        )

        # Condition the fear system for negative associations
        if valence < -0.2:
            stimulus_key = f"learned_{hash(stimulus_embedding.tobytes()) % 10000}"
            self.conditioning.condition(stimulus_key, stimulus_embedding, BasicEmotion.FEAR, intensity)

        # Also condition positive associations
        if valence > 0.2:
            stimulus_key = f"reward_{hash(stimulus_embedding.tobytes()) % 10000}"
            self.conditioning.condition(stimulus_key, stimulus_embedding, BasicEmotion.JOY, intensity)

    def get_state(self) -> Dict[str, Any]:
        """Get complete emotional state."""
        dominant, intensity = self.get_dominant_emotion()
        return {
            'core_affect': {
                'valence': self.core_affect.valence,
                'arousal': self.core_affect.arousal,
                'magnitude': self.core_affect.magnitude()
            },
            'mood': {
                'valence': self.mood.valence,
                'arousal': self.mood.arousal
            },
            'dominant_emotion': dominant.name,
            'dominant_intensity': intensity,
            'all_emotions': {e.name: v for e, v in self.current_emotions.items() if v > 0.01},
            'body_state': self.body_state.tolist(),
            'regulation_capacity': self.regulation.regulation_capacity,
            'somatic_markers_count': len(self.somatic_markers)
        }
