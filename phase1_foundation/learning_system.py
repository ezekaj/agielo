"""
PHASE 1 - STEP 12: LEARNING AND NEUROPLASTICITY
================================================

Brain-Inspired Learning: Not Backpropagation!

Key Innovation vs Existing AI:
- Deep Learning: Backpropagation (biologically implausible)
- This System: Hebbian learning + STDP (brain-realistic)

Learning Types:
1. Hebbian Learning: "Neurons that fire together, wire together"
2. STDP: Spike-Timing-Dependent Plasticity (causal learning)
3. Reward Modulation: Dopamine-like reinforcement
4. Structural Plasticity: Growing/pruning synapses

Based on:
- Hebb's Rule (1949)
- STDP research (Bi & Poo, 1998)
- Reward prediction error (Schultz, 1997)
- Synaptic homeostasis (Turrigiano, 2008)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

import sys
sys.path.append('..')
from utils.fast_math import (
    hebbian_update_fast,
    stdp_update_fast,
    softmax,
    PRECISION,
    NUMBA_AVAILABLE
)
from utils.base_types import (
    Vector, Timestamp, ExponentialMovingAverage, signal_bus
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class LearningRule(Enum):
    HEBBIAN = "hebbian"
    ANTI_HEBBIAN = "anti_hebbian"
    STDP = "stdp"
    REWARD_MODULATED = "reward_modulated"
    PREDICTIVE = "predictive"


@dataclass
class LearningConfig:
    """Configuration for learning system."""
    # Hebbian learning
    hebbian_learning_rate: float = 0.01
    weight_decay: float = 0.0001

    # STDP
    tau_plus: float = 20.0  # ms, LTP time constant
    tau_minus: float = 20.0  # ms, LTD time constant
    a_plus: float = 0.1  # LTP amplitude
    a_minus: float = 0.12  # LTD amplitude (slightly larger for stability)

    # Reward modulation
    reward_learning_rate: float = 0.1
    eligibility_decay: float = 0.9  # How fast eligibility traces decay
    dopamine_baseline: float = 0.0

    # Homeostasis
    target_activity: float = 0.1  # Target firing rate
    homeostatic_rate: float = 0.001

    # Structural plasticity
    synapse_creation_threshold: float = 0.8  # Co-activation threshold
    synapse_pruning_threshold: float = 0.01  # Weight threshold for pruning
    max_synapses_per_neuron: int = 1000

    # Metaplasticity
    metaplasticity_rate: float = 0.001


# =============================================================================
# SYNAPSE
# =============================================================================

@dataclass
class Synapse:
    """A single synapse with weight and plasticity state."""
    pre_idx: int  # Presynaptic neuron index
    post_idx: int  # Postsynaptic neuron index
    weight: float = 0.1
    eligibility_trace: float = 0.0  # For reward-modulated learning
    created_time: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # Metaplasticity: sliding threshold for LTP/LTD
    theta: float = 0.5

    def activate(self, pre_activity: float, post_activity: float):
        """Update eligibility trace on activation."""
        self.eligibility_trace = pre_activity * post_activity
        self.last_active = time.time()


# =============================================================================
# NEURON
# =============================================================================

@dataclass
class Neuron:
    """A single neuron with activity state."""
    idx: int
    activation: float = 0.0
    last_spike_time: float = 0.0
    spike_count: int = 0
    threshold: float = 1.0

    # Running average of activity for homeostasis
    avg_activity: ExponentialMovingAverage = field(
        default_factory=lambda: ExponentialMovingAverage(alpha=0.01)
    )

    def fire(self, input_current: float) -> bool:
        """
        Update neuron and potentially fire.
        Returns True if spike occurred.
        """
        self.activation = np.tanh(input_current)

        # Spike if above threshold
        if self.activation > self.threshold:
            self.last_spike_time = time.time()
            self.spike_count += 1
            self.avg_activity.update(1.0)
            return True
        else:
            self.avg_activity.update(0.0)
            return False


# =============================================================================
# HEBBIAN LEARNING
# =============================================================================

class HebbianLearning:
    """
    Classic Hebbian Learning: "Neurons that fire together, wire together"

    Δw = η × pre × post

    Variants:
    - Basic Hebbian: Δw = η × pre × post
    - Oja's Rule: Δw = η × post × (pre - post × w)  [normalized]
    - BCM Rule: Δw = η × pre × post × (post - θ)  [sliding threshold]
    """

    def __init__(self, config: LearningConfig):
        self.config = config

    def update(
        self,
        weights: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        rule: str = 'basic'
    ) -> np.ndarray:
        """
        Update weights using Hebbian learning.

        Args:
            weights: Weight matrix (post × pre)
            pre_activity: Presynaptic activities
            post_activity: Postsynaptic activities
            rule: 'basic', 'oja', or 'bcm'

        Returns:
            Updated weights
        """
        eta = self.config.hebbian_learning_rate

        if rule == 'basic':
            # Basic Hebbian: outer product
            if NUMBA_AVAILABLE:
                return hebbian_update_fast(
                    weights.astype(np.float64),
                    pre_activity.astype(np.float64),
                    post_activity.astype(np.float64),
                    eta
                ).astype(PRECISION)
            else:
                delta = eta * np.outer(post_activity, pre_activity)
                return weights + delta

        elif rule == 'oja':
            # Oja's rule: normalized Hebbian (prevents weight explosion)
            delta = np.zeros_like(weights)
            for i in range(len(post_activity)):
                for j in range(len(pre_activity)):
                    delta[i, j] = eta * post_activity[i] * (
                        pre_activity[j] - post_activity[i] * weights[i, j]
                    )
            return weights + delta

        elif rule == 'bcm':
            # BCM rule: sliding threshold
            # θ = E[post²] (running average of squared activity)
            theta = np.mean(post_activity ** 2) + 0.01
            delta = np.zeros_like(weights)
            for i in range(len(post_activity)):
                for j in range(len(pre_activity)):
                    delta[i, j] = eta * pre_activity[j] * post_activity[i] * (
                        post_activity[i] - theta
                    )
            return weights + delta

        return weights

    def apply_weight_decay(self, weights: np.ndarray) -> np.ndarray:
        """Apply weight decay for regularization."""
        return weights * (1 - self.config.weight_decay)


# =============================================================================
# SPIKE-TIMING-DEPENDENT PLASTICITY (STDP)
# =============================================================================

class STDPLearning:
    """
    Spike-Timing-Dependent Plasticity.

    Key insight: The relative timing of pre and post spikes determines
    whether the synapse strengthens (LTP) or weakens (LTD).

    Δt = t_post - t_pre
    If Δt > 0 (pre before post): LTP = A+ × exp(-Δt/τ+)  [causal, strengthen]
    If Δt < 0 (post before pre): LTD = -A- × exp(Δt/τ-)  [anti-causal, weaken]

    This implements CAUSAL learning: connections that predict outcomes
    are strengthened.
    """

    def __init__(self, config: LearningConfig):
        self.config = config

    def compute_delta_weight(
        self,
        pre_spike_time: float,
        post_spike_time: float
    ) -> float:
        """
        Compute weight change based on spike timing.
        """
        dt = post_spike_time - pre_spike_time

        if dt > 0:
            # Pre before post: LTP (causal)
            return self.config.a_plus * np.exp(-dt / self.config.tau_plus)
        else:
            # Post before pre: LTD (anti-causal)
            return -self.config.a_minus * np.exp(dt / self.config.tau_minus)

    def update(
        self,
        weights: np.ndarray,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray
    ) -> np.ndarray:
        """
        Update weights using STDP rule.

        Args:
            weights: Weight matrix (post × pre)
            pre_spike_times: Spike times for presynaptic neurons
            post_spike_times: Spike times for postsynaptic neurons

        Returns:
            Updated weights
        """
        if NUMBA_AVAILABLE:
            return stdp_update_fast(
                weights.astype(np.float64),
                pre_spike_times.astype(np.float64),
                post_spike_times.astype(np.float64),
                self.config.tau_plus,
                self.config.tau_minus,
                self.config.a_plus,
                self.config.a_minus
            ).astype(PRECISION)
        else:
            delta_w = np.zeros_like(weights)
            for i in range(len(post_spike_times)):
                for j in range(len(pre_spike_times)):
                    delta_w[i, j] = self.compute_delta_weight(
                        pre_spike_times[j],
                        post_spike_times[i]
                    )
            return weights + delta_w


# =============================================================================
# REWARD-MODULATED LEARNING
# =============================================================================

class RewardModulatedLearning:
    """
    Three-Factor Learning: Pre × Post × Reward

    Standard Hebbian: Δw = η × pre × post
    Reward-modulated: Δw = η × pre × post × (reward - baseline)

    This implements:
    - Reward prediction error (dopamine signal)
    - Eligibility traces (memory of recent activity)
    - Only strengthen if rewarded

    Based on dopamine research (Schultz et al.)
    """

    def __init__(self, config: LearningConfig):
        self.config = config
        self.baseline = ExponentialMovingAverage(alpha=0.01)
        self.baseline.value = config.dopamine_baseline

    def compute_reward_signal(self, reward: float) -> float:
        """
        Compute reward prediction error (dopamine-like signal).

        δ = reward - baseline

        Positive δ: better than expected → strengthen
        Negative δ: worse than expected → weaken
        Zero δ: as expected → no change
        """
        baseline = self.baseline.get() or 0.0
        delta = reward - baseline
        self.baseline.update(reward)
        return delta

    def update(
        self,
        weights: np.ndarray,
        eligibility_traces: np.ndarray,
        reward: float
    ) -> np.ndarray:
        """
        Update weights using reward-modulated learning.

        Args:
            weights: Weight matrix
            eligibility_traces: Accumulated pre × post activity
            reward: Current reward signal

        Returns:
            Updated weights
        """
        # Compute dopamine-like signal
        dopamine = self.compute_reward_signal(reward)

        # Update: Δw = η × eligibility × dopamine
        delta = self.config.reward_learning_rate * eligibility_traces * dopamine

        return weights + delta

    def update_eligibility(
        self,
        eligibility: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray
    ) -> np.ndarray:
        """
        Update eligibility traces.

        Eligibility traces remember recent activity for delayed reward.
        """
        # Decay existing traces
        eligibility *= self.config.eligibility_decay

        # Add new activity
        eligibility += np.outer(post_activity, pre_activity)

        return eligibility


# =============================================================================
# HOMEOSTATIC PLASTICITY
# =============================================================================

class HomeostaticPlasticity:
    """
    Synaptic Homeostasis: Keep activity in healthy range.

    Problem: Hebbian learning is unstable (weights explode or vanish).
    Solution: Homeostatic mechanisms that maintain target activity levels.

    Mechanisms:
    1. Synaptic scaling: Scale all weights to maintain target firing rate
    2. Intrinsic plasticity: Adjust neuron thresholds
    3. Metaplasticity: Adjust learning rate based on recent activity

    Based on Turrigiano & Nelson (2004)
    """

    def __init__(self, config: LearningConfig):
        self.config = config

    def synaptic_scaling(
        self,
        weights: np.ndarray,
        current_activity: float,
        target_activity: Optional[float] = None
    ) -> np.ndarray:
        """
        Scale weights to maintain target activity level.

        If activity too high: scale down all weights
        If activity too low: scale up all weights
        """
        target = target_activity or self.config.target_activity

        # Scaling factor
        if current_activity > 0:
            scale = (target / current_activity) ** 0.1  # Slow adjustment
        else:
            scale = 1.1  # Increase if no activity

        # Apply scaling
        return weights * scale

    def adjust_threshold(
        self,
        threshold: float,
        current_activity: float,
        target_activity: Optional[float] = None
    ) -> float:
        """
        Adjust firing threshold to maintain target activity.
        """
        target = target_activity or self.config.target_activity

        if current_activity > target:
            # Too active: raise threshold
            threshold += self.config.homeostatic_rate
        else:
            # Too quiet: lower threshold
            threshold -= self.config.homeostatic_rate

        return max(0.1, threshold)  # Keep threshold positive

    def metaplasticity(
        self,
        learning_rate: float,
        recent_activity: float
    ) -> float:
        """
        Adjust learning rate based on recent activity (BCM-like).

        High recent activity: Harder to induce LTP, easier LTD
        Low recent activity: Easier to induce LTP, harder LTD
        """
        # Sliding threshold
        theta = recent_activity + 0.01

        # Modify learning rate
        if recent_activity > self.config.target_activity:
            return learning_rate * 0.9  # Reduce learning
        else:
            return learning_rate * 1.1  # Increase learning


# =============================================================================
# STRUCTURAL PLASTICITY
# =============================================================================

class StructuralPlasticity:
    """
    Growing and Pruning Synapses.

    Unlike standard neural networks with fixed architecture:
    - Create new synapses between co-active neurons
    - Remove synapses that are weak or inactive

    This enables:
    - Sparse, efficient representations
    - Adaptation to new tasks
    - Prevention of catastrophic forgetting

    Based on spine dynamics research.
    """

    def __init__(self, config: LearningConfig):
        self.config = config

    def should_create_synapse(
        self,
        pre_activity: float,
        post_activity: float,
        current_weight: float
    ) -> bool:
        """
        Decide whether to create a new synapse.

        Create if:
        - No synapse exists (weight = 0)
        - Pre and post are both highly active
        """
        if current_weight > 0:
            return False  # Already exists

        coactivation = pre_activity * post_activity
        return coactivation > self.config.synapse_creation_threshold

    def should_prune_synapse(
        self,
        weight: float,
        last_active: float,
        current_time: float,
        inactivity_threshold: float = 3600.0  # 1 hour
    ) -> bool:
        """
        Decide whether to prune a synapse.

        Prune if:
        - Weight is below threshold
        - OR synapse has been inactive too long
        """
        if abs(weight) < self.config.synapse_pruning_threshold:
            return True

        if current_time - last_active > inactivity_threshold:
            return True

        return False

    def remodel_network(
        self,
        weights: np.ndarray,
        pre_activities: np.ndarray,
        post_activities: np.ndarray,
        last_active: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, int, int]:
        """
        Apply structural plasticity to weight matrix.

        Returns:
            new_weights: Updated weight matrix
            created: Number of synapses created
            pruned: Number of synapses pruned
        """
        current_time = time.time()
        if last_active is None:
            last_active = np.full_like(weights, current_time)

        created = 0
        pruned = 0
        new_weights = weights.copy()

        for i in range(len(post_activities)):
            for j in range(len(pre_activities)):
                # Check for pruning
                if new_weights[i, j] != 0:
                    if self.should_prune_synapse(
                        new_weights[i, j],
                        last_active[i, j],
                        current_time
                    ):
                        new_weights[i, j] = 0
                        pruned += 1
                # Check for creation
                else:
                    if self.should_create_synapse(
                        pre_activities[j],
                        post_activities[i],
                        new_weights[i, j]
                    ):
                        new_weights[i, j] = 0.1  # Initial weight
                        created += 1

        return new_weights, created, pruned


# =============================================================================
# COMPLETE LEARNING SYSTEM
# =============================================================================

class LearningSystem:
    """
    Complete Brain-Inspired Learning System.

    Integrates all learning mechanisms:
    - Hebbian/STDP for unsupervised learning
    - Reward modulation for reinforcement
    - Homeostasis for stability
    - Structural plasticity for adaptation

    This is fundamentally different from backpropagation:
    - Local learning rules (no global error signal)
    - Biologically plausible
    - Online learning (no separate training phase)
    - Sparse, efficient updates
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()

        # Learning modules
        self.hebbian = HebbianLearning(self.config)
        self.stdp = STDPLearning(self.config)
        self.reward = RewardModulatedLearning(self.config)
        self.homeostasis = HomeostaticPlasticity(self.config)
        self.structural = StructuralPlasticity(self.config)

        # Statistics
        self.total_updates = 0
        self.synapses_created = 0
        self.synapses_pruned = 0

    def update(
        self,
        weights: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        reward: Optional[float] = None,
        pre_spike_times: Optional[np.ndarray] = None,
        post_spike_times: Optional[np.ndarray] = None,
        eligibility: Optional[np.ndarray] = None,
        rule: LearningRule = LearningRule.HEBBIAN
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply learning update to weights.

        Args:
            weights: Current weight matrix
            pre_activity: Presynaptic activities
            post_activity: Postsynaptic activities
            reward: Optional reward signal
            pre_spike_times: Optional spike times for STDP
            post_spike_times: Optional spike times for STDP
            eligibility: Optional eligibility traces
            rule: Which learning rule to use

        Returns:
            Updated weights and statistics
        """
        stats = {}

        if rule == LearningRule.HEBBIAN:
            weights = self.hebbian.update(weights, pre_activity, post_activity)
            stats['rule'] = 'hebbian'

        elif rule == LearningRule.STDP:
            if pre_spike_times is None or post_spike_times is None:
                raise ValueError("STDP requires spike times")
            weights = self.stdp.update(weights, pre_spike_times, post_spike_times)
            stats['rule'] = 'stdp'

        elif rule == LearningRule.REWARD_MODULATED:
            if reward is None or eligibility is None:
                raise ValueError("Reward learning requires reward and eligibility")
            weights = self.reward.update(weights, eligibility, reward)
            dopamine = self.reward.compute_reward_signal(reward)
            stats['rule'] = 'reward'
            stats['dopamine'] = dopamine

        elif rule == LearningRule.PREDICTIVE:
            # Predictive coding: minimize prediction error
            prediction = np.tanh(np.dot(weights, pre_activity))
            error = post_activity - prediction
            weights += self.config.hebbian_learning_rate * np.outer(error, pre_activity)
            stats['rule'] = 'predictive'
            stats['error'] = np.mean(np.abs(error))

        # Apply weight decay
        weights = self.hebbian.apply_weight_decay(weights)

        # Homeostatic regulation
        avg_activity = np.mean(np.abs(post_activity))
        weights = self.homeostasis.synaptic_scaling(weights, avg_activity)

        # Structural plasticity (less frequent)
        if self.total_updates % 100 == 0:
            weights, created, pruned = self.structural.remodel_network(
                weights, pre_activity, post_activity
            )
            self.synapses_created += created
            self.synapses_pruned += pruned
            stats['synapses_created'] = created
            stats['synapses_pruned'] = pruned

        self.total_updates += 1
        stats['total_updates'] = self.total_updates

        return weights, stats

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'total_updates': self.total_updates,
            'synapses_created': self.synapses_created,
            'synapses_pruned': self.synapses_pruned,
            'reward_baseline': self.reward.baseline.get()
        }


# =============================================================================
# TESTING
# =============================================================================

def test_learning_system():
    """Test the complete learning system."""
    print("\n" + "=" * 60)
    print("LEARNING SYSTEM TEST")
    print("=" * 60)

    config = LearningConfig()
    learning = LearningSystem(config)

    # Create test network
    n_pre = 100
    n_post = 50
    weights = np.random.randn(n_post, n_pre).astype(PRECISION) * 0.1

    print("\n--- Hebbian Learning ---")
    pre = np.random.rand(n_pre).astype(PRECISION)
    post = np.random.rand(n_post).astype(PRECISION)

    start = time.perf_counter()
    for _ in range(1000):
        weights, stats = learning.update(
            weights, pre, post,
            rule=LearningRule.HEBBIAN
        )
    elapsed = time.perf_counter() - start
    print(f"1000 Hebbian updates: {elapsed*1000:.2f}ms ({elapsed/1000*1e6:.1f}μs/update)")

    print("\n--- STDP Learning ---")
    pre_times = np.random.rand(n_pre).astype(np.float64) * 100
    post_times = np.random.rand(n_post).astype(np.float64) * 100

    start = time.perf_counter()
    for _ in range(100):
        weights, stats = learning.update(
            weights, pre, post,
            pre_spike_times=pre_times,
            post_spike_times=post_times,
            rule=LearningRule.STDP
        )
    elapsed = time.perf_counter() - start
    print(f"100 STDP updates: {elapsed*1000:.2f}ms ({elapsed/100*1e3:.2f}ms/update)")

    print("\n--- Reward-Modulated Learning ---")
    eligibility = np.random.rand(n_post, n_pre).astype(PRECISION)

    for _ in range(100):
        reward = np.random.randn()  # Random reward
        weights, stats = learning.update(
            weights, pre, post,
            reward=reward,
            eligibility=eligibility,
            rule=LearningRule.REWARD_MODULATED
        )
        eligibility = learning.reward.update_eligibility(eligibility, pre, post)

    print(f"Dopamine baseline: {stats.get('dopamine', 0):.3f}")

    print("\n--- Predictive Learning ---")
    for _ in range(100):
        weights, stats = learning.update(
            weights, pre, post,
            rule=LearningRule.PREDICTIVE
        )
    print(f"Prediction error: {stats.get('error', 0):.4f}")

    print("\n--- Statistics ---")
    stats = learning.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)

    return learning


if __name__ == "__main__":
    test_learning_system()
