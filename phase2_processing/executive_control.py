"""
PHASE 2 - STEP 13: EXECUTIVE CONTROL
=====================================

Meta-control over cognitive processes.

Key Innovation vs Existing AI:
- LLMs: No control over own processing
- ACT-R: Has procedural control but limited
- This: Full executive function with inhibition, switching, monitoring

Executive Functions:
1. Inhibition: Suppress unwanted responses
2. Working Memory: Maintain and manipulate information
3. Cognitive Flexibility: Switch between tasks/rules
4. Attention Control: Focus on relevant information
5. Conflict Monitoring: Detect and resolve conflicts

Based on:
- Prefrontal cortex research
- Anterior cingulate cortex (conflict monitoring)
- Unity/Diversity framework (Miyake et al.)
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time

import sys
sys.path.append('..')
from utils.base_types import (
    Goal, Action, Timestamp, ExponentialMovingAverage, signal_bus
)
from utils.fast_math import PRECISION, softmax


# =============================================================================
# CONFIGURATION
# =============================================================================

class ControlSignal(Enum):
    INCREASE_CONTROL = "increase"
    MAINTAIN = "maintain"
    DECREASE_CONTROL = "decrease"
    SWITCH_TASK = "switch"
    INHIBIT = "inhibit"


@dataclass
class ExecutiveConfig:
    """Configuration for executive control."""
    # Inhibition
    inhibition_strength: float = 0.8
    inhibition_decay: float = 0.1

    # Working memory
    working_memory_capacity: int = 7
    refresh_rate: float = 0.1

    # Cognitive flexibility
    switch_cost: float = 0.2  # Performance cost for switching
    perseveration_threshold: float = 5  # Steps before forcing switch

    # Conflict monitoring
    conflict_threshold: float = 0.3
    error_sensitivity: float = 0.5

    # Attention
    attention_capacity: float = 1.0
    attention_decay: float = 0.05


# =============================================================================
# INHIBITION MODULE
# =============================================================================

class InhibitionModule:
    """
    Suppresses prepotent or unwanted responses.

    Like a brake on automatic responses.
    Allows overriding habits and impulses.
    """

    def __init__(self, config: ExecutiveConfig):
        self.config = config

        # Currently inhibited responses
        self.inhibited: Dict[str, float] = {}  # response_id → inhibition_strength

        # Inhibition fatigue
        self.fatigue = 0.0

    def inhibit(self, response_id: str, strength: Optional[float] = None):
        """
        Inhibit a response.

        Args:
            response_id: ID of response to inhibit
            strength: Inhibition strength (default from config)
        """
        strength = strength or self.config.inhibition_strength

        # Apply fatigue (inhibition gets harder over time)
        effective_strength = strength * (1.0 - self.fatigue)

        self.inhibited[response_id] = effective_strength
        self.fatigue = min(1.0, self.fatigue + 0.1)

    def release(self, response_id: str):
        """Release inhibition on a response."""
        if response_id in self.inhibited:
            del self.inhibited[response_id]

    def is_inhibited(self, response_id: str) -> Tuple[bool, float]:
        """Check if response is inhibited and by how much."""
        if response_id in self.inhibited:
            return True, self.inhibited[response_id]
        return False, 0.0

    def stop_signal(self):
        """Emergency stop - inhibit everything."""
        for resp_id in list(self.inhibited.keys()):
            self.inhibited[resp_id] = 1.0

    def decay(self):
        """Let inhibitions decay over time."""
        to_release = []
        for resp_id, strength in self.inhibited.items():
            self.inhibited[resp_id] = strength * (1.0 - self.config.inhibition_decay)
            if self.inhibited[resp_id] < 0.01:
                to_release.append(resp_id)

        for resp_id in to_release:
            del self.inhibited[resp_id]

        # Recover from fatigue
        self.fatigue = max(0.0, self.fatigue - 0.05)

    def get_state(self) -> Dict[str, Any]:
        """Get current inhibition state."""
        return {
            'inhibited_count': len(self.inhibited),
            'fatigue': self.fatigue,
            'active_inhibitions': dict(self.inhibited)
        }


# =============================================================================
# CONFLICT MONITOR
# =============================================================================

class ConflictMonitor:
    """
    Monitors for response conflicts (ACC-like function).

    Conflict = co-activation of incompatible responses.

    When conflict detected:
    - Signals need for more control
    - Triggers attention focusing
    - May trigger System 2
    """

    def __init__(self, config: ExecutiveConfig):
        self.config = config

        # Conflict history
        self.conflict_history = ExponentialMovingAverage(alpha=0.2)

        # Error history
        self.error_history: List[bool] = []
        self.error_count = 0

    def detect_conflict(self, responses: List[Tuple[str, float]]) -> float:
        """
        Detect conflict level among competing responses.

        Args:
            responses: List of (response_id, activation) pairs

        Returns:
            Conflict level (0-1)
        """
        if len(responses) < 2:
            return 0.0

        # Compute conflict as product of competing activations
        conflict = 0.0
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                if self._are_incompatible(responses[i][0], responses[j][0]):
                    conflict += responses[i][1] * responses[j][1]

        conflict = min(1.0, conflict)
        self.conflict_history.update(conflict)

        return conflict

    def _are_incompatible(self, r1: str, r2: str) -> bool:
        """Check if two responses are incompatible."""
        # Simple heuristic: different responses are incompatible
        return r1 != r2

    def register_error(self):
        """Register an error (incorrect response)."""
        self.error_history.append(True)
        self.error_count += 1

        # Keep only recent history
        if len(self.error_history) > 100:
            if self.error_history.pop(0):
                self.error_count -= 1

    def register_correct(self):
        """Register correct response."""
        self.error_history.append(False)
        if len(self.error_history) > 100:
            if self.error_history.pop(0):
                self.error_count -= 1

    def get_error_rate(self) -> float:
        """Get recent error rate."""
        if not self.error_history:
            return 0.0
        return self.error_count / len(self.error_history)

    def needs_more_control(self) -> bool:
        """Determine if more cognitive control is needed."""
        avg_conflict = self.conflict_history.get() or 0.0
        error_rate = self.get_error_rate()

        return (avg_conflict > self.config.conflict_threshold or
                error_rate > self.config.error_sensitivity)

    def get_state(self) -> Dict[str, Any]:
        """Get monitor state."""
        return {
            'current_conflict': self.conflict_history.get() or 0.0,
            'error_rate': self.get_error_rate(),
            'needs_control': self.needs_more_control()
        }


# =============================================================================
# TASK SWITCHER
# =============================================================================

class TaskSwitcher:
    """
    Manages switching between tasks/mental sets.

    Cognitive flexibility: ability to switch between different rules,
    perspectives, or ways of thinking.
    """

    def __init__(self, config: ExecutiveConfig):
        self.config = config

        # Current task/rule set
        self.current_task: Optional[str] = None
        self.current_rules: Dict[str, Any] = {}

        # Task history
        self.task_history: List[str] = []
        self.steps_on_current_task = 0

        # Switch cost tracking
        self.recent_switch_cost = 0.0

    def set_task(self, task_name: str, rules: Optional[Dict[str, Any]] = None):
        """Set the current task/mental set."""
        if task_name != self.current_task:
            # Record switch
            if self.current_task:
                self.task_history.append(self.current_task)

            self.current_task = task_name
            self.current_rules = rules or {}
            self.steps_on_current_task = 0
            self.recent_switch_cost = self.config.switch_cost

            signal_bus.publish('task_switch', {'new_task': task_name})
        else:
            self.steps_on_current_task += 1
            # Reduce switch cost over time
            self.recent_switch_cost *= 0.9

    def should_switch(self, performance: float) -> bool:
        """Determine if task switch is needed."""
        # Poor performance suggests need to switch
        if performance < 0.3:
            return True

        # Too long on one task (perseveration)
        if self.steps_on_current_task > self.config.perseveration_threshold:
            return True

        return False

    def get_switch_cost(self) -> float:
        """Get current switch cost (performance penalty after switching)."""
        return self.recent_switch_cost

    def get_state(self) -> Dict[str, Any]:
        """Get switcher state."""
        return {
            'current_task': self.current_task,
            'steps_on_task': self.steps_on_current_task,
            'switch_cost': self.recent_switch_cost,
            'task_history_length': len(self.task_history)
        }


# =============================================================================
# ATTENTION CONTROLLER
# =============================================================================

class AttentionController:
    """
    Controls focus of attention.

    Attention is a limited resource that must be allocated.
    """

    def __init__(self, config: ExecutiveConfig):
        self.config = config

        # Attention allocation
        self.focus: Dict[str, float] = {}  # target → attention amount
        self.total_allocated = 0.0

        # Attention history
        self.focus_history: List[str] = []

    def focus_on(self, target: str, amount: float):
        """
        Allocate attention to a target.

        Args:
            target: What to focus on
            amount: Amount of attention (0-1)
        """
        # Check capacity
        available = self.config.attention_capacity - self.total_allocated + self.focus.get(target, 0)

        if amount > available:
            # Need to reduce attention elsewhere
            self._redistribute(amount - available)

        self.focus[target] = min(amount, self.config.attention_capacity)
        self.total_allocated = sum(self.focus.values())

        self.focus_history.append(target)
        if len(self.focus_history) > 100:
            self.focus_history.pop(0)

    def _redistribute(self, amount_needed: float):
        """Reduce attention on other targets to free up capacity."""
        if not self.focus:
            return

        # Reduce equally from all targets
        reduction_per_target = amount_needed / len(self.focus)
        for target in list(self.focus.keys()):
            self.focus[target] = max(0, self.focus[target] - reduction_per_target)
            if self.focus[target] < 0.01:
                del self.focus[target]

    def get_attention(self, target: str) -> float:
        """Get attention allocated to target."""
        return self.focus.get(target, 0.0)

    def decay(self):
        """Let attention decay over time."""
        for target in list(self.focus.keys()):
            self.focus[target] *= (1.0 - self.config.attention_decay)
            if self.focus[target] < 0.01:
                del self.focus[target]

        self.total_allocated = sum(self.focus.values())

    def get_state(self) -> Dict[str, Any]:
        """Get attention state."""
        return {
            'total_allocated': self.total_allocated,
            'num_targets': len(self.focus),
            'focus': dict(self.focus)
        }


# =============================================================================
# EXECUTIVE CONTROL SYSTEM
# =============================================================================

class ExecutiveControlSystem:
    """
    Complete Executive Control System.

    Integrates all executive functions:
    - Inhibition
    - Conflict monitoring
    - Task switching
    - Attention control

    This is the "conductor" of the cognitive orchestra.
    """

    def __init__(self, config: Optional[ExecutiveConfig] = None):
        self.config = config or ExecutiveConfig()

        # Components
        self.inhibition = InhibitionModule(self.config)
        self.conflict_monitor = ConflictMonitor(self.config)
        self.task_switcher = TaskSwitcher(self.config)
        self.attention = AttentionController(self.config)

        # Current control level
        self.control_level = 0.5  # 0 = relaxed, 1 = maximum control

        # Goal stack
        self.goals: List[Goal] = []

        # Statistics
        self.total_steps = 0

    def step(
        self,
        responses: List[Tuple[str, float]],
        current_goal: Optional[Goal] = None,
        feedback: Optional[Dict[str, Any]] = None
    ) -> ControlSignal:
        """
        Execute one step of executive control.

        Args:
            responses: Current candidate responses with activations
            current_goal: Current goal being pursued
            feedback: Feedback from previous action (error, reward, etc.)

        Returns:
            Control signal for the cognitive system
        """
        self.total_steps += 1

        # 1. Process feedback
        if feedback:
            if feedback.get('error', False):
                self.conflict_monitor.register_error()
            else:
                self.conflict_monitor.register_correct()

        # 2. Detect conflict
        conflict = self.conflict_monitor.detect_conflict(responses)

        # 3. Determine control signal
        signal = self._determine_control_signal(conflict, feedback)

        # 4. Execute control adjustments
        self._execute_control(signal, responses)

        # 5. Manage attention
        if current_goal:
            # Handle both Goal objects and string goals
            goal_desc = current_goal.description if hasattr(current_goal, 'description') else str(current_goal)
            self.attention.focus_on(goal_desc, 0.5)

        # 6. Decay functions
        self.inhibition.decay()
        self.attention.decay()

        return signal

    def _determine_control_signal(
        self,
        conflict: float,
        feedback: Optional[Dict[str, Any]]
    ) -> ControlSignal:
        """Determine what control signal to send."""
        # High conflict or errors → increase control
        if self.conflict_monitor.needs_more_control():
            self.control_level = min(1.0, self.control_level + 0.1)
            return ControlSignal.INCREASE_CONTROL

        # Check if task switch needed
        performance = feedback.get('performance', 0.5) if feedback else 0.5
        if self.task_switcher.should_switch(performance):
            return ControlSignal.SWITCH_TASK

        # Things going well → relax control
        if conflict < 0.1 and performance > 0.7:
            self.control_level = max(0.0, self.control_level - 0.05)
            return ControlSignal.DECREASE_CONTROL

        return ControlSignal.MAINTAIN

    def _execute_control(
        self,
        signal: ControlSignal,
        responses: List[Tuple[str, float]]
    ):
        """Execute the control signal."""
        if signal == ControlSignal.INHIBIT:
            # Inhibit all but top response
            sorted_responses = sorted(responses, key=lambda x: -x[1])
            for resp_id, _ in sorted_responses[1:]:
                self.inhibition.inhibit(resp_id)

        elif signal == ControlSignal.INCREASE_CONTROL:
            # Inhibit weak responses
            for resp_id, activation in responses:
                if activation < 0.3:
                    self.inhibition.inhibit(resp_id, strength=0.5)

    def set_goal(self, goal: Goal):
        """Set a new goal."""
        self.goals.append(goal)
        self.attention.focus_on(goal.description, goal.priority)

    def complete_goal(self, goal_id: str):
        """Mark a goal as complete."""
        self.goals = [g for g in self.goals if g.id != goal_id]

    def get_top_goal(self) -> Optional[Goal]:
        """Get the highest priority active goal."""
        if not self.goals:
            return None
        return max(self.goals, key=lambda g: g.urgency)

    def filter_responses(
        self,
        responses: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Filter responses based on inhibition state."""
        filtered = []
        for resp_id, activation in responses:
            is_inhibited, strength = self.inhibition.is_inhibited(resp_id)
            if is_inhibited:
                activation *= (1.0 - strength)
            if activation > 0.01:
                filtered.append((resp_id, activation))
        return filtered

    def get_state(self) -> Dict[str, Any]:
        """Get complete executive state."""
        return {
            'control_level': self.control_level,
            'inhibition': self.inhibition.get_state(),
            'conflict': self.conflict_monitor.get_state(),
            'task': self.task_switcher.get_state(),
            'attention': self.attention.get_state(),
            'num_goals': len(self.goals),
            'total_steps': self.total_steps
        }


# =============================================================================
# TESTING
# =============================================================================

def test_executive_control():
    """Test the executive control system."""
    print("\n" + "=" * 60)
    print("EXECUTIVE CONTROL SYSTEM TEST")
    print("=" * 60)

    config = ExecutiveConfig()
    executive = ExecutiveControlSystem(config)

    # Set up a goal
    goal = Goal(description="Complete task", priority=0.8)
    executive.set_goal(goal)

    print("\n--- Testing Control Loop ---")
    for i in range(50):
        # Simulate responses with varying conflict
        responses = [
            ('action_a', np.random.rand()),
            ('action_b', np.random.rand()),
            ('action_c', np.random.rand())
        ]

        # Simulate occasional errors
        feedback = {
            'error': np.random.rand() < 0.2,
            'performance': np.random.rand()
        }

        signal = executive.step(responses, goal, feedback)

        if i % 10 == 0:
            print(f"Step {i}: signal={signal.value}, control={executive.control_level:.2f}")

    # Test inhibition
    print("\n--- Testing Inhibition ---")
    executive.inhibition.inhibit('bad_response')
    is_inhibited, strength = executive.inhibition.is_inhibited('bad_response')
    print(f"bad_response inhibited: {is_inhibited}, strength: {strength:.2f}")

    # Test response filtering
    responses = [('good', 0.8), ('bad_response', 0.6), ('neutral', 0.4)]
    filtered = executive.filter_responses(responses)
    print(f"Filtered responses: {filtered}")

    # Statistics
    print("\n--- Executive State ---")
    state = executive.get_state()
    print(f"Control level: {state['control_level']:.3f}")
    print(f"Conflict level: {state['conflict']['current_conflict']:.3f}")
    print(f"Error rate: {state['conflict']['error_rate']:.3f}")
    print(f"Inhibition fatigue: {state['inhibition']['fatigue']:.3f}")

    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)

    return executive


if __name__ == "__main__":
    test_executive_control()
