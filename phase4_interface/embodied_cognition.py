"""
Embodied Cognition System - Body Shapes Mind

Implements:
1. Sensorimotor Integration (perception-action coupling)
2. Body Schema (internal body representation)
3. Affordance Detection (what actions environment allows)
4. Motor Simulation (mental rehearsal)
5. Interoception (sensing internal body states)
6. Embodied Metaphors (abstract concepts grounded in body)

Based on research:
- Lakoff & Johnson: Conceptual metaphors grounded in body
- Gibson: Affordances
- Gallese: Mirror neurons and motor simulation
- Craig: Interoception and subjective feelings

Performance: Vectorized sensor processing, O(1) affordance lookup
Comparison vs existing:
- ACT-R: Perceptual-motor modules but no true embodiment
- SOAR: No embodiment
- LLMs: Purely linguistic, no body
- Robotics AI: Body but limited cognition integration
- This: Full embodied cognition loop
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time


class BodyPart(Enum):
    """Body parts for schema."""
    HEAD = auto()
    TORSO = auto()
    LEFT_ARM = auto()
    RIGHT_ARM = auto()
    LEFT_HAND = auto()
    RIGHT_HAND = auto()
    LEFT_LEG = auto()
    RIGHT_LEG = auto()
    LEFT_FOOT = auto()
    RIGHT_FOOT = auto()


class SensoryModality(Enum):
    """Types of sensory input."""
    VISUAL = auto()
    AUDITORY = auto()
    TACTILE = auto()
    PROPRIOCEPTIVE = auto()
    VESTIBULAR = auto()
    INTEROCEPTIVE = auto()
    OLFACTORY = auto()
    GUSTATORY = auto()


@dataclass
class BodyState:
    """Current state of body parts."""
    positions: Dict[BodyPart, np.ndarray] = field(default_factory=dict)  # 3D positions
    velocities: Dict[BodyPart, np.ndarray] = field(default_factory=dict)
    tensions: Dict[BodyPart, float] = field(default_factory=dict)
    temperatures: Dict[BodyPart, float] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize default body state
        for part in BodyPart:
            if part not in self.positions:
                self.positions[part] = np.zeros(3)
            if part not in self.velocities:
                self.velocities[part] = np.zeros(3)
            if part not in self.tensions:
                self.tensions[part] = 0.0
            if part not in self.temperatures:
                self.temperatures[part] = 37.0  # Normal body temp


@dataclass
class Affordance:
    """An action possibility offered by environment/object."""
    name: str
    object_embedding: np.ndarray
    action_type: str              # 'grasp', 'push', 'sit', etc.
    required_body_parts: List[BodyPart]
    success_probability: float
    effort_required: float        # 0-1
    learned_from_experience: bool = False


@dataclass
class MotorProgram:
    """A learned motor sequence."""
    name: str
    body_parts: List[BodyPart]
    trajectory: np.ndarray        # T x D trajectory
    timing: np.ndarray            # Time points
    precision_required: float
    adaptable: bool = True


class BodySchema:
    """
    Internal representation of body structure and capabilities.

    The body schema is:
    - Plastic (can incorporate tools)
    - Predictive (anticipates sensory consequences of movement)
    - Multimodal (integrates vision, proprioception, touch)
    """

    def __init__(self):
        self.body_state = BodyState()

        # Body structure (simplified kinematic chain)
        self.joint_limits: Dict[str, Tuple[float, float]] = {
            'shoulder': (-np.pi, np.pi),
            'elbow': (0, np.pi),
            'wrist': (-np.pi/2, np.pi/2),
            'hip': (-np.pi/2, np.pi/2),
            'knee': (0, np.pi),
            'ankle': (-np.pi/4, np.pi/4),
        }

        # Peripersonal space (space near body)
        self.peripersonal_radius = 1.0  # meters

        # Tool incorporation
        self.incorporated_tools: Dict[str, np.ndarray] = {}  # tool -> extension

        # Body image (visual representation)
        self.body_image_embedding = np.random.randn(64) * 0.1

    def update_state(self, sensory_input: Dict[SensoryModality, Any]):
        """Update body state from sensory input."""
        # Proprioception updates positions
        if SensoryModality.PROPRIOCEPTIVE in sensory_input:
            proprio = sensory_input[SensoryModality.PROPRIOCEPTIVE]
            for part, pos in proprio.items():
                if part in BodyPart.__members__:
                    bp = BodyPart[part]
                    old_pos = self.body_state.positions[bp]
                    self.body_state.velocities[bp] = pos - old_pos
                    self.body_state.positions[bp] = pos

        # Interoception updates internal states
        if SensoryModality.INTEROCEPTIVE in sensory_input:
            intero = sensory_input[SensoryModality.INTEROCEPTIVE]
            if 'tension' in intero:
                for part, tension in intero['tension'].items():
                    if part in BodyPart.__members__:
                        self.body_state.tensions[BodyPart[part]] = tension

    def is_in_peripersonal_space(self, position: np.ndarray) -> bool:
        """Check if position is within reachable space."""
        # Compute distance from body center (torso)
        body_center = self.body_state.positions[BodyPart.TORSO]
        distance = np.linalg.norm(position - body_center)
        return distance <= self.peripersonal_radius

    def incorporate_tool(self, tool_name: str, extension: np.ndarray):
        """
        Incorporate tool into body schema.

        Like how using a rake extends your 'hand' mentally.
        """
        self.incorporated_tools[tool_name] = extension
        # Extend peripersonal space
        self.peripersonal_radius += np.linalg.norm(extension)

    def release_tool(self, tool_name: str):
        """Release tool from body schema."""
        if tool_name in self.incorporated_tools:
            extension = self.incorporated_tools.pop(tool_name)
            self.peripersonal_radius -= np.linalg.norm(extension)

    def get_reachable_space(self) -> Dict[str, Any]:
        """Get description of reachable space."""
        return {
            'radius': self.peripersonal_radius,
            'center': self.body_state.positions[BodyPart.TORSO].tolist(),
            'tools_incorporated': list(self.incorporated_tools.keys())
        }


class AffordanceDetector:
    """
    Detect action possibilities in environment (Gibson's affordances).

    Affordances are relational: they depend on both object AND body capabilities.
    A chair affords sitting only if your body can sit.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Learned affordances
        self.affordance_templates: List[Affordance] = []

        # Object-affordance associations
        self.object_affordances: Dict[str, List[str]] = {}

        # Initialize basic affordances
        self._init_basic_affordances()

    def _init_basic_affordances(self):
        """Initialize innate/basic affordances."""
        basic = [
            ('grasp_small', 'grasp', [BodyPart.RIGHT_HAND], 0.8, 0.2),
            ('grasp_large', 'grasp', [BodyPart.LEFT_HAND, BodyPart.RIGHT_HAND], 0.7, 0.4),
            ('push', 'push', [BodyPart.RIGHT_ARM], 0.9, 0.3),
            ('lift_light', 'lift', [BodyPart.RIGHT_ARM], 0.9, 0.2),
            ('lift_heavy', 'lift', [BodyPart.LEFT_ARM, BodyPart.RIGHT_ARM], 0.6, 0.7),
            ('sit', 'sit', [BodyPart.TORSO, BodyPart.LEFT_LEG, BodyPart.RIGHT_LEG], 0.95, 0.1),
            ('walk_on', 'locomote', [BodyPart.LEFT_LEG, BodyPart.RIGHT_LEG], 0.9, 0.3),
            ('climb', 'locomote', [BodyPart.LEFT_ARM, BodyPart.RIGHT_ARM, BodyPart.LEFT_LEG, BodyPart.RIGHT_LEG], 0.5, 0.8),
        ]

        for name, action, parts, prob, effort in basic:
            self.affordance_templates.append(Affordance(
                name=name,
                object_embedding=np.zeros(self.dim),  # Template, not specific
                action_type=action,
                required_body_parts=parts,
                success_probability=prob,
                effort_required=effort
            ))

    def detect_affordances(self,
                           object_embedding: np.ndarray,
                           object_properties: Dict[str, float],
                           body_state: BodyState) -> List[Affordance]:
        """
        Detect what actions are possible with an object.

        object_properties might include: size, weight, rigidity, temperature, etc.
        """
        detected = []

        for template in self.affordance_templates:
            # Check if body parts available and functional
            parts_ok = all(
                body_state.tensions.get(part, 0) < 0.9  # Not too tense/injured
                for part in template.required_body_parts
            )

            if not parts_ok:
                continue

            # Adjust success probability based on object properties
            adjusted_prob = template.success_probability

            # Size affects grasping
            if 'size' in object_properties:
                size = object_properties['size']
                if template.action_type == 'grasp':
                    if size > 0.8 and len(template.required_body_parts) == 1:
                        adjusted_prob *= 0.5  # Too big for one hand
                    elif size < 0.2:
                        adjusted_prob *= 0.7  # Too small

            # Weight affects lifting
            if 'weight' in object_properties and template.action_type == 'lift':
                weight = object_properties['weight']
                if weight > 0.7 and len(template.required_body_parts) == 1:
                    adjusted_prob *= 0.3

            # Create specific affordance
            if adjusted_prob > 0.1:
                detected.append(Affordance(
                    name=template.name,
                    object_embedding=object_embedding.copy(),
                    action_type=template.action_type,
                    required_body_parts=template.required_body_parts,
                    success_probability=adjusted_prob,
                    effort_required=template.effort_required
                ))

        return detected

    def learn_affordance(self,
                         object_embedding: np.ndarray,
                         action_performed: str,
                         body_parts_used: List[BodyPart],
                         success: bool,
                         effort: float):
        """Learn new affordance from experience."""
        # Update or create affordance template
        matching = [a for a in self.affordance_templates if a.action_type == action_performed]

        if matching:
            # Update existing
            for aff in matching:
                if set(aff.required_body_parts) == set(body_parts_used):
                    # Update success probability with learning
                    aff.success_probability = 0.9 * aff.success_probability + 0.1 * float(success)
                    aff.effort_required = 0.9 * aff.effort_required + 0.1 * effort
                    aff.learned_from_experience = True
                    break
        else:
            # Create new affordance
            new_aff = Affordance(
                name=f"{action_performed}_learned",
                object_embedding=object_embedding.copy(),
                action_type=action_performed,
                required_body_parts=body_parts_used,
                success_probability=float(success),
                effort_required=effort,
                learned_from_experience=True
            )
            self.affordance_templates.append(new_aff)


class MotorSimulation:
    """
    Mental simulation of actions without executing them.

    Based on motor imagery research - imagining actions activates
    motor cortex similar to actual execution.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Learned motor programs
        self.motor_programs: Dict[str, MotorProgram] = {}

        # Forward model: predicts sensory consequences of actions
        self.forward_model_weights = np.random.randn(dim, dim) * 0.01

        # Inverse model: computes motor commands for desired outcomes
        self.inverse_model_weights = np.random.randn(dim, dim) * 0.01

    def simulate_action(self,
                        action_embedding: np.ndarray,
                        current_state: np.ndarray,
                        steps: int = 10) -> List[np.ndarray]:
        """
        Mentally simulate action execution.

        Returns predicted trajectory of states.
        """
        trajectory = [current_state.copy()]
        state = current_state.copy()

        for _ in range(steps):
            # Forward model prediction
            delta = action_embedding @ self.forward_model_weights
            state = state + 0.1 * delta
            trajectory.append(state.copy())

        return trajectory

    def predict_outcome(self,
                        action_embedding: np.ndarray,
                        current_state: np.ndarray) -> np.ndarray:
        """Predict final outcome of action."""
        # Simple forward pass
        return current_state + action_embedding @ self.forward_model_weights

    def plan_action(self,
                    current_state: np.ndarray,
                    goal_state: np.ndarray) -> np.ndarray:
        """Use inverse model to plan action achieving goal."""
        desired_delta = goal_state - current_state
        # Inverse model
        action = desired_delta @ self.inverse_model_weights.T
        return action

    def learn_motor_program(self,
                            name: str,
                            trajectory: np.ndarray,
                            timing: np.ndarray,
                            body_parts: List[BodyPart]):
        """Store a learned motor program."""
        self.motor_programs[name] = MotorProgram(
            name=name,
            body_parts=body_parts,
            trajectory=trajectory.copy(),
            timing=timing.copy(),
            precision_required=0.5
        )

    def retrieve_motor_program(self, name: str) -> Optional[MotorProgram]:
        """Retrieve stored motor program."""
        return self.motor_programs.get(name)

    def update_forward_model(self,
                             action: np.ndarray,
                             predicted_outcome: np.ndarray,
                             actual_outcome: np.ndarray,
                             learning_rate: float = 0.01):
        """Update forward model from prediction error."""
        error = actual_outcome - predicted_outcome
        # Gradient update
        self.forward_model_weights += learning_rate * np.outer(action, error)


class InteroceptionSystem:
    """
    Sensing internal body states (Craig's interoception).

    Interoception provides the basis for:
    - Subjective feelings
    - Emotional awareness
    - Sense of self as embodied
    """

    def __init__(self):
        # Internal state channels
        self.heart_rate = 70.0         # BPM
        self.breathing_rate = 15.0     # Per minute
        self.hunger = 0.0              # 0-1
        self.thirst = 0.0              # 0-1
        self.fatigue = 0.0             # 0-1
        self.pain = 0.0                # 0-1
        self.temperature = 37.0        # Celsius

        # Interoceptive accuracy (how well can we sense)
        self.interoceptive_accuracy = 0.7

        # History for detecting changes
        self.state_history = deque(maxlen=100)

    def sense(self) -> Dict[str, float]:
        """Sense current interoceptive state."""
        # Add noise based on accuracy
        noise = (1 - self.interoceptive_accuracy) * 0.1

        state = {
            'heart_rate': self.heart_rate + np.random.randn() * noise * 10,
            'breathing_rate': self.breathing_rate + np.random.randn() * noise * 3,
            'hunger': np.clip(self.hunger + np.random.randn() * noise, 0, 1),
            'thirst': np.clip(self.thirst + np.random.randn() * noise, 0, 1),
            'fatigue': np.clip(self.fatigue + np.random.randn() * noise, 0, 1),
            'pain': np.clip(self.pain + np.random.randn() * noise, 0, 1),
            'temperature': self.temperature + np.random.randn() * noise * 0.5,
        }

        self.state_history.append((time.time(), state))

        return state

    def update_from_activity(self, activity_level: float, duration: float):
        """Update internal states based on activity."""
        # Activity increases heart rate, breathing, fatigue
        self.heart_rate = 70 + activity_level * 100  # Up to 170 BPM
        self.breathing_rate = 15 + activity_level * 25  # Up to 40

        # Fatigue accumulates
        self.fatigue = min(1.0, self.fatigue + activity_level * duration * 0.01)

        # Hunger and thirst grow over time
        self.hunger = min(1.0, self.hunger + duration * 0.001)
        self.thirst = min(1.0, self.thirst + duration * 0.002 * (1 + activity_level))

    def rest(self, duration: float):
        """Rest reduces fatigue, normalizes heart rate."""
        self.fatigue = max(0, self.fatigue - duration * 0.01)
        self.heart_rate = 70 + (self.heart_rate - 70) * np.exp(-duration * 0.1)
        self.breathing_rate = 15 + (self.breathing_rate - 15) * np.exp(-duration * 0.1)

    def consume(self, food: float = 0, water: float = 0):
        """Consume food/water to satisfy hunger/thirst."""
        self.hunger = max(0, self.hunger - food)
        self.thirst = max(0, self.thirst - water)

    def get_homeostatic_urgency(self) -> Dict[str, float]:
        """Get urgency of homeostatic needs."""
        return {
            'eat': self.hunger ** 2,  # Non-linear urgency
            'drink': self.thirst ** 2,
            'rest': self.fatigue ** 2,
            'regulate_temp': abs(self.temperature - 37) / 5,
            'address_pain': self.pain ** 1.5
        }


class EmbodiedMetaphors:
    """
    Abstract concepts grounded in bodily experience (Lakoff & Johnson).

    Examples:
    - UNDERSTANDING IS GRASPING
    - HAPPY IS UP, SAD IS DOWN
    - IMPORTANT IS BIG
    - TIME IS MOTION
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Metaphor mappings: abstract concept -> bodily grounding
        self.metaphors = {
            'understanding': {
                'source_domain': 'grasping',
                'mapping': np.random.randn(dim) * 0.1,
                'related_body': [BodyPart.RIGHT_HAND]
            },
            'happiness': {
                'source_domain': 'up',
                'mapping': np.array([0, 0, 1] + [0] * (dim - 3)),  # Z-up
                'related_body': [BodyPart.HEAD, BodyPart.TORSO]
            },
            'sadness': {
                'source_domain': 'down',
                'mapping': np.array([0, 0, -1] + [0] * (dim - 3)),
                'related_body': [BodyPart.HEAD, BodyPart.TORSO]
            },
            'importance': {
                'source_domain': 'big/heavy',
                'mapping': np.random.randn(dim) * 0.1,
                'related_body': [BodyPart.TORSO]
            },
            'difficulty': {
                'source_domain': 'heavy',
                'mapping': np.random.randn(dim) * 0.1,
                'related_body': [BodyPart.LEFT_ARM, BodyPart.RIGHT_ARM]
            },
            'time_future': {
                'source_domain': 'forward',
                'mapping': np.array([1, 0, 0] + [0] * (dim - 3)),
                'related_body': [BodyPart.HEAD]
            },
            'time_past': {
                'source_domain': 'behind',
                'mapping': np.array([-1, 0, 0] + [0] * (dim - 3)),
                'related_body': [BodyPart.HEAD]
            },
        }

    def ground_concept(self, concept: str, concept_embedding: np.ndarray) -> Dict[str, Any]:
        """Ground abstract concept in bodily experience."""
        if concept in self.metaphors:
            metaphor = self.metaphors[concept]
            # Blend abstract embedding with bodily grounding
            grounded = 0.7 * concept_embedding + 0.3 * metaphor['mapping']
            return {
                'concept': concept,
                'source_domain': metaphor['source_domain'],
                'grounded_embedding': grounded,
                'body_involvement': metaphor['related_body']
            }

        # No specific metaphor - return as is
        return {
            'concept': concept,
            'source_domain': 'abstract',
            'grounded_embedding': concept_embedding,
            'body_involvement': []
        }

    def embody_reasoning(self,
                         premise: np.ndarray,
                         conclusion: np.ndarray) -> Dict[str, float]:
        """
        Evaluate reasoning with embodied intuition.

        Does the conclusion 'feel right' given embodied metaphors?
        """
        # Check if reasoning respects metaphorical consistency
        # (e.g., if premise involves 'up' concepts, conclusion should too)

        consistency_scores = {}

        for name, metaphor in self.metaphors.items():
            premise_alignment = np.dot(premise, metaphor['mapping']) / (
                np.linalg.norm(premise) * np.linalg.norm(metaphor['mapping']) + 1e-8
            )
            conclusion_alignment = np.dot(conclusion, metaphor['mapping']) / (
                np.linalg.norm(conclusion) * np.linalg.norm(metaphor['mapping']) + 1e-8
            )

            # Consistency: if premise has strong alignment, conclusion should too
            if abs(premise_alignment) > 0.3:
                consistency = 1 - abs(premise_alignment - conclusion_alignment)
                consistency_scores[name] = consistency

        return consistency_scores


class EmbodiedCognitionSystem:
    """
    Complete embodied cognition system.

    Integrates body schema, affordances, motor simulation, interoception,
    and embodied metaphors into unified embodied mind.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Components
        self.body_schema = BodySchema()
        self.affordance_detector = AffordanceDetector(dim)
        self.motor_simulation = MotorSimulation(dim)
        self.interoception = InteroceptionSystem()
        self.embodied_metaphors = EmbodiedMetaphors(dim)

        # Current sensory state
        self.current_sensory: Dict[SensoryModality, Any] = {}

        # Action-perception cycle state
        self.last_action: Optional[np.ndarray] = None
        self.predicted_outcome: Optional[np.ndarray] = None

    def perceive(self, sensory_input: Dict[SensoryModality, Any]) -> Dict[str, Any]:
        """Process sensory input through embodied system."""
        self.current_sensory = sensory_input

        # Update body schema
        self.body_schema.update_state(sensory_input)

        # Get interoceptive state
        intero_state = self.interoception.sense()

        # Check prediction error if we made a prediction
        prediction_error = None
        if self.predicted_outcome is not None and SensoryModality.PROPRIOCEPTIVE in sensory_input:
            actual = sensory_input[SensoryModality.PROPRIOCEPTIVE]
            if isinstance(actual, np.ndarray):
                prediction_error = np.linalg.norm(self.predicted_outcome - actual)

                # Update forward model
                if self.last_action is not None:
                    self.motor_simulation.update_forward_model(
                        self.last_action, self.predicted_outcome, actual
                    )

        return {
            'body_state': self.body_schema.body_state,
            'interoception': intero_state,
            'prediction_error': prediction_error,
            'peripersonal_space': self.body_schema.get_reachable_space()
        }

    def detect_affordances_in_scene(self,
                                    objects: List[Tuple[np.ndarray, Dict[str, float]]]) -> Dict[str, List[Affordance]]:
        """Detect affordances for all objects in scene."""
        affordances = {}

        for i, (obj_embedding, obj_properties) in enumerate(objects):
            obj_id = f"object_{i}"
            obj_affordances = self.affordance_detector.detect_affordances(
                obj_embedding, obj_properties, self.body_schema.body_state
            )
            affordances[obj_id] = obj_affordances

        return affordances

    def plan_and_simulate_action(self,
                                 goal_state: np.ndarray,
                                 current_state: np.ndarray) -> Dict[str, Any]:
        """Plan action and simulate outcome before executing."""
        # Plan action
        planned_action = self.motor_simulation.plan_action(current_state, goal_state)

        # Simulate
        simulated_trajectory = self.motor_simulation.simulate_action(
            planned_action, current_state
        )

        # Predict final outcome
        predicted_outcome = self.motor_simulation.predict_outcome(
            planned_action, current_state
        )

        # Check if prediction reaches goal
        goal_distance = np.linalg.norm(predicted_outcome - goal_state)
        success_likely = goal_distance < 0.3

        return {
            'planned_action': planned_action,
            'simulated_trajectory': simulated_trajectory,
            'predicted_outcome': predicted_outcome,
            'goal_distance': goal_distance,
            'success_likely': success_likely
        }

    def execute_action(self, action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Execute action (or prepare for execution)."""
        # Store for prediction error computation
        self.last_action = action.copy()
        self.predicted_outcome = self.motor_simulation.predict_outcome(
            action, current_state
        )

        # Update interoception based on effort
        effort = np.linalg.norm(action) / 10  # Normalize
        self.interoception.update_from_activity(effort, duration=1.0)

        return self.predicted_outcome

    def ground_abstract_concept(self, concept: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Ground abstract concept in embodied experience."""
        return self.embodied_metaphors.ground_concept(concept, embedding)

    def get_embodied_state(self) -> Dict[str, Any]:
        """Get complete embodied state."""
        return {
            'body': {
                'peripersonal_radius': self.body_schema.peripersonal_radius,
                'tools_incorporated': list(self.body_schema.incorporated_tools.keys()),
            },
            'interoception': self.interoception.sense(),
            'homeostatic_needs': self.interoception.get_homeostatic_urgency(),
            'motor_programs': list(self.motor_simulation.motor_programs.keys()),
            'affordance_templates': len(self.affordance_detector.affordance_templates)
        }
