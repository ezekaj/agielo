"""
Motivation Engine - Drives and Goals Based on Neuroscience Research

Implements:
1. Intrinsic Motivation (curiosity, competence, autonomy - Self-Determination Theory)
2. Dopaminergic Reward System (prediction error-based learning)
3. Drive States (homeostatic and allostatic regulation)
4. Goal Management (hierarchical goals with temporal discounting)
5. Exploration-Exploitation Balance (UCB-inspired)

Performance: Vectorized NumPy, O(1) goal lookup, minimal allocations
Comparison vs existing:
- ACT-R: Has utility but no true curiosity drive
- SOAR: Impasses but not intrinsic motivation
- LLMs: No persistent goals or drives at all
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time

# Optional Numba acceleration
try:
    from numba import jit, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    float64 = np.float64


class DriveType(Enum):
    """Fundamental drives based on neuroscience."""
    CURIOSITY = auto()      # Information-seeking (dopaminergic)
    COMPETENCE = auto()     # Mastery/skill development
    AUTONOMY = auto()       # Self-determination
    AFFILIATION = auto()    # Social connection
    SAFETY = auto()         # Threat avoidance
    HOMEOSTASIS = auto()    # Internal balance


class GoalStatus(Enum):
    """Goal lifecycle states."""
    ACTIVE = auto()
    SUSPENDED = auto()
    ACHIEVED = auto()
    ABANDONED = auto()


@dataclass
class Drive:
    """A motivational drive with current and target levels."""
    drive_type: DriveType
    current_level: float = 0.5      # 0-1 satiation
    target_level: float = 0.7       # Desired level
    sensitivity: float = 1.0        # How strongly deviations are felt
    decay_rate: float = 0.01        # Natural decay per step

    def deficit(self) -> float:
        """How much is the drive unsatisfied?"""
        return max(0.0, self.target_level - self.current_level) * self.sensitivity

    def urgency(self) -> float:
        """How urgent is satisfying this drive? (non-linear)"""
        deficit = self.deficit()
        # Urgency grows faster as deficit increases (hyperbolic)
        return deficit ** 2 / (deficit + 0.1)

    def satisfy(self, amount: float):
        """Satisfy the drive by amount."""
        self.current_level = min(1.0, self.current_level + amount)

    def decay(self):
        """Natural decay of satiation."""
        self.current_level = max(0.0, self.current_level - self.decay_rate)


@dataclass
class Goal:
    """A goal with value, progress, and temporal properties."""
    name: str
    embedding: np.ndarray           # Semantic representation
    value: float = 1.0              # Intrinsic value
    progress: float = 0.0           # 0-1 completion
    deadline: Optional[float] = None  # Time pressure
    parent_goal: Optional[str] = None  # Hierarchical structure
    subgoals: List[str] = field(default_factory=list)
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    drives_satisfied: List[DriveType] = field(default_factory=list)

    def effective_value(self, current_time: float, discount_rate: float = 0.1) -> float:
        """Value adjusted for temporal discounting and progress."""
        base_value = self.value * (1.0 - self.progress)  # Remaining value

        # Temporal discounting (hyperbolic, like humans)
        if self.deadline:
            time_to_deadline = max(0.01, self.deadline - current_time)
            urgency_bonus = 1.0 / (1.0 + discount_rate * time_to_deadline)
            base_value *= (1.0 + urgency_bonus)

        return base_value


class DopamineSystem:
    """
    Dopaminergic reward prediction error system.

    Based on Schultz's work:
    - Phasic dopamine = reward prediction error
    - Tonic dopamine = average reward/motivation level
    - Anticipatory dopamine = expected reward drives behavior
    """

    def __init__(self, learning_rate: float = 0.1, decay: float = 0.95):
        self.learning_rate = learning_rate
        self.decay = decay

        # Value estimates for states/actions
        self.value_estimates: Dict[str, float] = {}

        # Tonic dopamine level (baseline motivation)
        self.tonic_level: float = 0.5

        # Recent reward history for tonic adjustment
        self.reward_history = deque(maxlen=100)

        # Phasic response (transient)
        self.phasic_response: float = 0.0

    def compute_rpe(self, state_key: str, actual_reward: float) -> float:
        """
        Compute Reward Prediction Error (RPE).

        RPE = actual_reward - expected_reward
        Positive RPE -> better than expected -> increase value
        Negative RPE -> worse than expected -> decrease value
        """
        expected = self.value_estimates.get(state_key, 0.0)
        rpe = actual_reward - expected

        # Update phasic response
        self.phasic_response = rpe

        # Update value estimate
        self.value_estimates[state_key] = expected + self.learning_rate * rpe

        # Update reward history
        self.reward_history.append(actual_reward)

        # Adjust tonic level based on recent history
        if len(self.reward_history) > 10:
            recent_avg = np.mean(list(self.reward_history)[-10:])
            self.tonic_level = 0.9 * self.tonic_level + 0.1 * recent_avg

        return rpe

    def get_anticipated_reward(self, state_key: str) -> float:
        """Get anticipated reward for a state (drives approach behavior)."""
        return self.value_estimates.get(state_key, 0.0) + self.tonic_level

    def get_motivation_signal(self) -> float:
        """Combined motivational signal."""
        # Tonic provides baseline, phasic provides moment-to-moment adjustment
        return np.clip(self.tonic_level + 0.5 * self.phasic_response, 0.0, 1.0)

    def decay_phasic(self):
        """Phasic response decays quickly."""
        self.phasic_response *= self.decay


class CuriosityModule:
    """
    Curiosity/Information-Seeking based on:
    1. Prediction Error (novelty)
    2. Learning Progress (competence)
    3. Information Gain (epistemic value)

    This drives exploration of unknown territories.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # World model for prediction (simple for speed)
        self.world_model_weights = np.random.randn(dim, dim) * 0.01

        # Learning progress tracker
        self.prediction_errors = deque(maxlen=100)
        self.learning_progress_window = 20

        # Novelty detector (visited state embeddings)
        self.visited_states: List[np.ndarray] = []
        self.novelty_threshold = 0.3

    def compute_novelty(self, state_embedding: np.ndarray) -> float:
        """How novel is this state? (distance to nearest visited)"""
        if len(self.visited_states) == 0:
            return 1.0  # Everything is novel at first

        # Vectorized distance computation
        visited_array = np.array(self.visited_states)
        distances = np.linalg.norm(visited_array - state_embedding, axis=1)
        min_distance = np.min(distances)

        # Convert to novelty score (sigmoid)
        novelty = 1.0 / (1.0 + np.exp(-10 * (min_distance - self.novelty_threshold)))

        return float(novelty)

    def compute_prediction_error(self,
                                  current_state: np.ndarray,
                                  next_state: np.ndarray) -> float:
        """How surprising was the transition?"""
        predicted_next = current_state @ self.world_model_weights
        error = np.linalg.norm(next_state - predicted_next)

        # Store for learning progress
        self.prediction_errors.append(error)

        return error

    def compute_learning_progress(self) -> float:
        """
        Are we getting better at predicting?
        Positive = learning happening = interesting
        Negative = not learning = boring
        Zero = already mastered = move on
        """
        if len(self.prediction_errors) < self.learning_progress_window * 2:
            return 0.5  # Not enough data

        errors = list(self.prediction_errors)
        old_errors = np.mean(errors[-2*self.learning_progress_window:-self.learning_progress_window])
        new_errors = np.mean(errors[-self.learning_progress_window:])

        # Learning progress = reduction in error
        progress = old_errors - new_errors

        # Normalize to 0-1
        return float(np.clip(progress * 10 + 0.5, 0.0, 1.0))

    def compute_information_gain(self,
                                  state_embedding: np.ndarray,
                                  uncertainty_before: float,
                                  uncertainty_after: float) -> float:
        """Expected information gain from visiting state."""
        return max(0.0, uncertainty_before - uncertainty_after)

    def get_curiosity_score(self,
                            state_embedding: np.ndarray,
                            prediction_error: Optional[float] = None) -> float:
        """Combined curiosity score."""
        novelty = self.compute_novelty(state_embedding)
        learning_progress = self.compute_learning_progress()

        # Weight novelty and learning progress
        # High novelty + positive learning progress = most interesting
        curiosity = 0.4 * novelty + 0.6 * learning_progress

        if prediction_error is not None:
            # Add prediction error component (bounded)
            curiosity += 0.2 * np.tanh(prediction_error)

        return float(np.clip(curiosity, 0.0, 1.0))

    def update_world_model(self,
                           current_state: np.ndarray,
                           next_state: np.ndarray,
                           learning_rate: float = 0.01):
        """Simple online learning for world model."""
        predicted = current_state @ self.world_model_weights
        error = next_state - predicted

        # Gradient descent update
        self.world_model_weights += learning_rate * np.outer(current_state, error)

    def mark_visited(self, state_embedding: np.ndarray):
        """Mark state as visited."""
        self.visited_states.append(state_embedding.copy())

        # Limit memory
        if len(self.visited_states) > 1000:
            # Keep only recent and diverse states
            self.visited_states = self.visited_states[-500:]


class ExplorationExploitationController:
    """
    Balances exploration (trying new things) vs exploitation (using known good options).

    Uses UCB-inspired approach combined with curiosity.
    """

    def __init__(self, exploration_bonus: float = 2.0):
        self.exploration_bonus = exploration_bonus
        self.action_counts: Dict[str, int] = {}
        self.action_values: Dict[str, float] = {}
        self.total_actions: int = 0

        # Adaptive exploration rate
        self.base_exploration = 0.3
        self.exploration_decay = 0.999
        self.current_exploration = self.base_exploration

    def ucb_score(self, action_key: str) -> float:
        """Upper Confidence Bound score for action selection."""
        if action_key not in self.action_counts:
            return float('inf')  # Unexplored = try it

        count = self.action_counts[action_key]
        value = self.action_values.get(action_key, 0.0)

        # UCB formula
        exploration_term = self.exploration_bonus * np.sqrt(
            np.log(self.total_actions + 1) / (count + 1)
        )

        return value + exploration_term

    def select_action(self,
                      available_actions: List[str],
                      action_values: Optional[Dict[str, float]] = None,
                      curiosity_scores: Optional[Dict[str, float]] = None) -> str:
        """Select action balancing exploration and exploitation."""
        if not available_actions:
            raise ValueError("No actions available")

        scores = {}
        for action in available_actions:
            # Base UCB score
            ucb = self.ucb_score(action)

            # Add known value if provided
            if action_values and action in action_values:
                ucb = 0.5 * ucb + 0.5 * action_values[action]

            # Add curiosity bonus if provided
            if curiosity_scores and action in curiosity_scores:
                ucb += self.current_exploration * curiosity_scores[action]

            scores[action] = ucb

        # Select best
        best_action = max(scores, key=scores.get)

        return best_action

    def update(self, action_key: str, reward: float):
        """Update after taking action."""
        self.total_actions += 1

        if action_key not in self.action_counts:
            self.action_counts[action_key] = 0
            self.action_values[action_key] = 0.0

        self.action_counts[action_key] += 1

        # Incremental mean update
        n = self.action_counts[action_key]
        old_value = self.action_values[action_key]
        self.action_values[action_key] = old_value + (reward - old_value) / n

        # Decay exploration over time
        self.current_exploration *= self.exploration_decay
        self.current_exploration = max(0.05, self.current_exploration)


class MotivationEngine:
    """
    Complete motivation system integrating all components.

    This is what gives the AI "wants" and "drives" behavior.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Core drives
        self.drives: Dict[DriveType, Drive] = {
            DriveType.CURIOSITY: Drive(DriveType.CURIOSITY, sensitivity=1.2, decay_rate=0.02),
            DriveType.COMPETENCE: Drive(DriveType.COMPETENCE, sensitivity=1.0, decay_rate=0.01),
            DriveType.AUTONOMY: Drive(DriveType.AUTONOMY, sensitivity=0.8, decay_rate=0.005),
            DriveType.AFFILIATION: Drive(DriveType.AFFILIATION, sensitivity=0.9, decay_rate=0.01),
            DriveType.SAFETY: Drive(DriveType.SAFETY, target_level=0.8, sensitivity=1.5, decay_rate=0.005),
        }

        # Goals
        self.goals: Dict[str, Goal] = {}
        self.goal_stack: List[str] = []  # Active goal pursuit

        # Subsystems
        self.dopamine = DopamineSystem()
        self.curiosity = CuriosityModule(dim)
        self.explorer = ExplorationExploitationController()

        # State
        self.current_focus: Optional[str] = None
        self.motivation_level: float = 0.5

    def add_goal(self,
                 name: str,
                 embedding: np.ndarray,
                 value: float = 1.0,
                 deadline: Optional[float] = None,
                 parent: Optional[str] = None,
                 drives: Optional[List[DriveType]] = None):
        """Add a goal to pursue."""
        goal = Goal(
            name=name,
            embedding=embedding,
            value=value,
            deadline=deadline,
            parent_goal=parent,
            drives_satisfied=drives or []
        )

        self.goals[name] = goal

        # Link to parent
        if parent and parent in self.goals:
            self.goals[parent].subgoals.append(name)

    def get_most_urgent_goal(self) -> Optional[Goal]:
        """Get the goal with highest effective value considering drives."""
        if not self.goals:
            return None

        current_time = time.time()
        best_goal = None
        best_score = -float('inf')

        for goal in self.goals.values():
            if goal.status != GoalStatus.ACTIVE:
                continue

            # Base value
            score = goal.effective_value(current_time)

            # Add drive urgency bonus
            for drive_type in goal.drives_satisfied:
                if drive_type in self.drives:
                    score += self.drives[drive_type].urgency() * 2.0

            if score > best_score:
                best_score = score
                best_goal = goal

        return best_goal

    def compute_action_value(self,
                             action_embedding: np.ndarray,
                             expected_outcomes: Dict[str, float]) -> float:
        """
        Compute value of taking an action.

        Integrates:
        - Goal progress
        - Drive satisfaction
        - Curiosity
        - Anticipated reward
        """
        value = 0.0

        # Curiosity component
        curiosity_score = self.curiosity.get_curiosity_score(action_embedding)
        value += self.drives[DriveType.CURIOSITY].sensitivity * curiosity_score * 0.3

        # Expected outcomes
        for outcome, prob in expected_outcomes.items():
            anticipated = self.dopamine.get_anticipated_reward(outcome)
            value += prob * anticipated

        # Current goal alignment
        if self.current_focus and self.current_focus in self.goals:
            goal = self.goals[self.current_focus]
            alignment = float(np.dot(action_embedding, goal.embedding) /
                            (np.linalg.norm(action_embedding) * np.linalg.norm(goal.embedding) + 1e-8))
            value += alignment * goal.value * 0.5

        return value

    def process_reward(self,
                       state_key: str,
                       reward: float,
                       drives_affected: Optional[Dict[DriveType, float]] = None):
        """Process received reward and update motivation."""
        # Dopamine RPE
        rpe = self.dopamine.compute_rpe(state_key, reward)

        # Update drives if specified
        if drives_affected:
            for drive_type, amount in drives_affected.items():
                if drive_type in self.drives:
                    self.drives[drive_type].satisfy(amount)

        # Update overall motivation
        self.motivation_level = self.dopamine.get_motivation_signal()

        return rpe

    def step(self, state_embedding: np.ndarray) -> Dict[str, Any]:
        """
        One motivation step.

        Returns current motivational state and recommended focus.
        """
        # Decay drives naturally
        for drive in self.drives.values():
            drive.decay()

        # Decay phasic dopamine
        self.dopamine.decay_phasic()

        # Update curiosity
        self.curiosity.mark_visited(state_embedding)

        # Get most urgent goal
        urgent_goal = self.get_most_urgent_goal()
        if urgent_goal:
            self.current_focus = urgent_goal.name

        # Compute drive urgencies
        urgencies = {dt: d.urgency() for dt, d in self.drives.items()}
        most_urgent_drive = max(urgencies, key=urgencies.get)

        # Overall motivation
        self.motivation_level = np.clip(
            self.dopamine.tonic_level +
            0.3 * max(urgencies.values()) +
            0.2 * self.curiosity.compute_learning_progress(),
            0.0, 1.0
        )

        return {
            'current_focus': self.current_focus,
            'most_urgent_drive': most_urgent_drive,
            'drive_urgencies': urgencies,
            'motivation_level': self.motivation_level,
            'exploration_rate': self.explorer.current_exploration,
            'curiosity_score': self.curiosity.get_curiosity_score(state_embedding),
            'dopamine_tonic': self.dopamine.tonic_level,
            'dopamine_phasic': self.dopamine.phasic_response
        }

    def select_action(self,
                      available_actions: List[Tuple[str, np.ndarray]],
                      action_outcomes: Optional[Dict[str, Dict[str, float]]] = None) -> Tuple[str, float]:
        """
        Select action based on motivation.

        Returns (action_name, expected_value)
        """
        if not available_actions:
            return None, 0.0

        action_values = {}
        curiosity_scores = {}

        for action_name, action_embedding in available_actions:
            # Curiosity for each action
            curiosity_scores[action_name] = self.curiosity.get_curiosity_score(action_embedding)

            # Expected value
            outcomes = action_outcomes.get(action_name, {}) if action_outcomes else {}
            action_values[action_name] = self.compute_action_value(action_embedding, outcomes)

        # Use exploration-exploitation controller
        action_names = [a[0] for a in available_actions]
        selected = self.explorer.select_action(action_names, action_values, curiosity_scores)

        return selected, action_values.get(selected, 0.0)

    def update_goal_progress(self, goal_name: str, progress_delta: float):
        """Update progress on a goal."""
        if goal_name not in self.goals:
            return

        goal = self.goals[goal_name]
        goal.progress = min(1.0, goal.progress + progress_delta)

        # Check completion
        if goal.progress >= 1.0:
            goal.status = GoalStatus.ACHIEVED

            # Satisfy associated drives
            for drive_type in goal.drives_satisfied:
                if drive_type in self.drives:
                    self.drives[drive_type].satisfy(0.3)

            # Remove from stack
            if goal_name in self.goal_stack:
                self.goal_stack.remove(goal_name)

            # Update focus
            if self.current_focus == goal_name:
                self.current_focus = None

    def get_state(self) -> Dict[str, Any]:
        """Get complete motivational state."""
        return {
            'drives': {dt.name: {'level': d.current_level, 'urgency': d.urgency()}
                      for dt, d in self.drives.items()},
            'goals': {name: {'progress': g.progress, 'status': g.status.name, 'value': g.value}
                     for name, g in self.goals.items()},
            'current_focus': self.current_focus,
            'motivation_level': self.motivation_level,
            'dopamine': {
                'tonic': self.dopamine.tonic_level,
                'phasic': self.dopamine.phasic_response
            }
        }
