"""
Cognitive Agent - Unified Human-Like AI System

This is the main integration point that combines all cognitive modules
into a unified, human-like cognitive agent.

Architecture Overview:
======================

                    ┌─────────────────────────────────────────────┐
                    │            Self-Awareness Loop              │
                    │  (Metacognition, Introspection, Self-Model) │
                    └─────────────────────┬───────────────────────┘
                                          │
    ┌─────────────────────────────────────┼─────────────────────────────────────┐
    │                              Executive Control                             │
    │                    (Inhibition, Task Switching, Attention)                │
    └─────────────────────────────────────┬─────────────────────────────────────┘
                                          │
    ┌─────────────────────────────────────┼─────────────────────────────────────┐
    │                           Dual-Process System                              │
    │                        ┌───────┴───────┐                                  │
    │                        ▼               ▼                                  │
    │               ┌─────────────┐   ┌─────────────┐                           │
    │               │  System 1   │   │  System 2   │                           │
    │               │   (Fast)    │   │   (Slow)    │                           │
    │               └─────────────┘   └─────────────┘                           │
    └─────────────────────────────────────┬─────────────────────────────────────┘
                                          │
    ┌─────────────────────────────────────┼─────────────────────────────────────┐
    │                           Processing Layer                                 │
    │   ┌──────────────┬──────────────┬──────────────┬──────────────┐           │
    │   │  Reasoning   │  Creativity  │  Time Percep │  Cog Maps    │           │
    │   └──────────────┴──────────────┴──────────────┴──────────────┘           │
    └─────────────────────────────────────┬─────────────────────────────────────┘
                                          │
    ┌─────────────────────────────────────┼─────────────────────────────────────┐
    │                           Motivation Layer                                 │
    │   ┌──────────────┬──────────────┬──────────────┐                          │
    │   │ Drives/Goals │   Emotions   │  Curiosity   │                          │
    │   └──────────────┴──────────────┴──────────────┘                          │
    └─────────────────────────────────────┬─────────────────────────────────────┘
                                          │
    ┌─────────────────────────────────────┼─────────────────────────────────────┐
    │                           Interface Layer                                  │
    │   ┌──────────────┬──────────────┬──────────────┐                          │
    │   │   Language   │   Social     │   Embodied   │                          │
    │   └──────────────┴──────────────┴──────────────┘                          │
    └─────────────────────────────────────┬─────────────────────────────────────┘
                                          │
    ┌─────────────────────────────────────┼─────────────────────────────────────┐
    │                           Foundation Layer                                 │
    │   ┌──────────────┬──────────────┬──────────────┐                          │
    │   │  Predictive  │   Memory     │   Learning   │                          │
    │   │   Coding     │   Systems    │   Systems    │                          │
    │   └──────────────┴──────────────┴──────────────┘                          │
    └───────────────────────────────────────────────────────────────────────────┘

Performance Optimizations:
- Vectorized NumPy operations throughout
- Optional Numba JIT compilation
- Lazy module initialization
- Efficient memory indexing (LSH)
- O(1) or O(log n) critical paths

Comparison vs Existing Systems:
- ACT-R: We add emotions, embodiment, creativity, social cognition
- SOAR: We add continuous learning, emotions, grounded language
- LLMs: We add explicit memory, goals, metacognition, embodiment
- See COMPARISON.md for detailed analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class CognitiveConfig:
    """Configuration for the cognitive agent."""
    dim: int = 64                    # Embedding dimension
    enable_emotions: bool = True
    enable_embodiment: bool = True
    enable_social: bool = True
    enable_creativity: bool = True
    enable_sleep: bool = True
    fast_mode: bool = False          # Skip expensive computations


class CognitiveAgent:
    """
    A unified cognitive agent with human-like cognition.

    This agent integrates all cognitive modules and provides
    a coherent interface for perception, action, and thought.
    """

    def __init__(self, config: Optional[CognitiveConfig] = None):
        self.config = config or CognitiveConfig()
        self.dim = self.config.dim

        # Lazy loading of modules (initialized on first use)
        self._modules_initialized = False
        self._modules = {}

        # Agent state
        self.current_state = np.zeros(self.dim)
        self.step_count = 0
        self.session_start = time.time()

        # Performance tracking
        self.timing_stats: Dict[str, List[float]] = {}

    def _ensure_modules(self):
        """Lazy initialization of cognitive modules."""
        if self._modules_initialized:
            return

        # Foundation Layer (Phase 1)
        from phase1_foundation.predictive_coding import (
            PredictiveCodingEngine, ActiveInference, PredictiveCodingConfig
        )
        from phase1_foundation.memory_system import MemorySystem, MemoryConfig
        from phase1_foundation.learning_system import LearningSystem

        pred_config = PredictiveCodingConfig(
            input_dim=self.dim,
            num_levels=3,
            level_dims=[self.dim, self.dim // 2, self.dim // 4]
        )
        self._modules['prediction'] = PredictiveCodingEngine(config=pred_config)
        self._modules['active_inference'] = ActiveInference(
            self._modules['prediction']
        )

        mem_config = MemoryConfig(embedding_dim=self.dim)
        self._modules['memory'] = MemorySystem(config=mem_config)

        # LearningSystem uses default LearningConfig (no embedding_dim needed)
        self._modules['learning'] = LearningSystem()

        # Processing Layer (Phase 2)
        from phase2_processing.dual_process import DualProcessController, DualProcessConfig
        from phase2_processing.executive_control import ExecutiveControlSystem, ExecutiveConfig
        from phase2_processing.reasoning_modules import ReasoningRouter, ReasoningConfig

        dual_config = DualProcessConfig(embedding_dim=self.dim)
        self._modules['dual_process'] = DualProcessController(config=dual_config)

        exec_config = ExecutiveConfig()
        self._modules['executive'] = ExecutiveControlSystem(config=exec_config)

        reason_config = ReasoningConfig(embedding_dim=self.dim)
        self._modules['reasoning'] = ReasoningRouter(config=reason_config)

        # Motivation Layer (Phase 3)
        from phase3_motivation.motivation_engine import MotivationEngine
        from phase3_motivation.emotion_system import EmotionSystem
        from phase3_motivation.self_awareness import SelfAwarenessSystem

        self._modules['motivation'] = MotivationEngine(dim=self.dim)
        if self.config.enable_emotions:
            self._modules['emotion'] = EmotionSystem(dim=self.dim)
        self._modules['self_awareness'] = SelfAwarenessSystem(dim=self.dim)

        # Interface Layer (Phase 4)
        from phase4_interface.language_system import LanguageSystem
        if self.config.enable_embodiment:
            from phase4_interface.embodied_cognition import EmbodiedCognitionSystem
            self._modules['embodied'] = EmbodiedCognitionSystem(dim=self.dim)
        if self.config.enable_social:
            from phase4_interface.social_cognition import SocialCognitionSystem
            self._modules['social'] = SocialCognitionSystem(dim=self.dim)
        self._modules['language'] = LanguageSystem(dim=self.dim)

        # Advanced Layer (Phase 5)
        if self.config.enable_creativity:
            from phase5_advanced.creativity_module import CreativityModule
            self._modules['creativity'] = CreativityModule(dim=self.dim)

        from phase5_advanced.cognitive_maps import CognitiveMapsSystem
        from phase5_advanced.time_perception import TimePerceptionSystem

        self._modules['cognitive_maps'] = CognitiveMapsSystem(dim=self.dim)
        self._modules['time'] = TimePerceptionSystem(dim=self.dim)

        if self.config.enable_sleep:
            from phase5_advanced.sleep_consolidation import SleepConsolidationSystem
            self._modules['sleep'] = SleepConsolidationSystem(dim=self.dim)

        self._modules_initialized = True

    def _time_operation(self, name: str, func, *args, **kwargs):
        """Time an operation for performance tracking."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if name not in self.timing_stats:
            self.timing_stats[name] = []
        self.timing_stats[name].append(elapsed)

        return result

    # ==================== Main Interface ====================

    def perceive(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Process a perceptual observation through the cognitive system.

        This is the main entry point for sensory input.
        """
        self._ensure_modules()

        results = {}

        # 1. Predictive Coding - compute prediction errors
        prediction_result = self._time_operation(
            'prediction',
            self._modules['prediction'].process,
            observation
        )
        results['prediction'] = prediction_result

        # 2. Store in sensory memory
        self._modules['memory'].perceive(observation, 'visual')

        # 3. Attend to input (transfer to working memory)
        attended = self._modules['memory'].attend('visual')
        results['attended'] = attended is not None

        # 4. Update time perception
        self._modules['time'].update(0.1, [{
            'id': f'percept_{self.step_count}',
            'embedding': observation,
            'attention': 0.7
        }])

        # 5. Check for emotional appraisal
        if 'emotion' in self._modules:
            appraisal = {
                'relevance': float(prediction_result.get('weighted_error', 0.5)),
                'congruence': 0.5,
                'certainty': 1.0 - float(prediction_result.get('weighted_error', 0.5))
            }
            emotion_result = self._modules['emotion'].process_situation(
                observation, appraisal
            )
            results['emotion'] = emotion_result

        # 6. Update motivation based on observation
        motivation_state = self._modules['motivation'].step(observation)
        results['motivation'] = motivation_state

        # 7. Metacognitive monitoring
        meta_signal = self._modules['self_awareness'].monitor_process(
            'perception',
            observation,
            expected_output=self._modules['prediction'].get_prediction()
        )
        results['metacognition'] = {
            'confidence': meta_signal.confidence,
            'error_detected': meta_signal.error_detected
        }

        self.current_state = observation
        self.step_count += 1

        return results

    def think(self, problem: np.ndarray, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Engage in deliberate thinking about a problem.
        """
        self._ensure_modules()
        context = context or {}

        results = {}

        # 1. Dual-process routing
        dual_response, dual_confidence, dual_meta = self._time_operation(
            'dual_process',
            self._modules['dual_process'].process,
            problem, problem, context.get('goal')
        )
        results['dual_process'] = {
            'response': str(dual_response) if dual_response is not None else '',
            'system_used': dual_meta.get('system_used') if isinstance(dual_meta, dict) else 'unknown',
            'confidence': dual_confidence
        }

        # 2. Executive control
        responses = [(str(i), dual_confidence if dual_confidence else 0.5)
                     for i in range(3)]
        exec_signal = self._modules['executive'].step(
            responses, context.get('goal'), None
        )
        # exec_signal is a ControlSignal enum
        from phase2_processing.executive_control import ControlSignal
        results['executive'] = {
            'should_inhibit': exec_signal == ControlSignal.INHIBIT,
            'should_switch': exec_signal == ControlSignal.SWITCH_TASK,
            'effort_required': exec_signal == ControlSignal.INCREASE_CONTROL
        }

        # 3. Reasoning (if needed)
        system_used = dual_meta.get('system_used') if isinstance(dual_meta, dict) else None
        if system_used == 'system2':
            reasoning_result = self._modules['reasoning'].reason(
                problem, context
            )
            results['reasoning'] = reasoning_result

        # 4. Memory retrieval
        recalled = self._modules['memory'].recall(problem, k=3)
        results['recalled_memories'] = len(recalled)

        # 5. Self-awareness check
        reflection = self._modules['self_awareness'].reflect(depth=1)
        results['reflection'] = reflection

        return results

    def decide(self,
               options: List[Tuple[str, np.ndarray]],
               goal: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Make a decision among options.
        """
        self._ensure_modules()

        results = {}

        # 1. Get motivation-based action values
        action_name, expected_value = self._modules['motivation'].select_action(options)
        results['motivation_choice'] = action_name
        results['expected_value'] = expected_value

        # 2. Get emotional signals (somatic markers)
        if 'emotion' in self._modules:
            option_embeddings = [opt[1] for opt in options]
            emotional_signals = self._modules['emotion'].get_decision_signal(option_embeddings)
            results['emotional_signals'] = emotional_signals

            # Integrate emotional signals with value - adjust choice based on gut feelings
            best_emotional_idx = None
            best_emotional_signal = -float('inf')
            for i, (name, _) in enumerate(options):
                if emotional_signals[i] < -0.3:
                    results['emotional_warning'] = f"Negative gut feeling about {name}"
                if emotional_signals[i] > best_emotional_signal:
                    best_emotional_signal = emotional_signals[i]
                    best_emotional_idx = i

            # If emotional signal is strong, let it influence decision
            if best_emotional_idx is not None and best_emotional_signal > 0.2:
                action_name = options[best_emotional_idx][0]
                results['emotion_influenced'] = True

        # 3. Active inference for action selection
        if goal is not None:
            # Resize goal to match predictive coding top-level dimension
            top_level_dim = self._modules['prediction'].levels[-1].output_dim
            if len(goal) != top_level_dim:
                # Project goal to correct dimension
                goal_resized = goal[:top_level_dim] if len(goal) > top_level_dim else np.pad(goal, (0, top_level_dim - len(goal)))
            else:
                goal_resized = goal
            selected_action = self._modules['active_inference'].select_action(goal_resized)
            results['active_inference_action'] = selected_action.tolist()

        # 4. Final decision (integrate all signals)
        results['final_decision'] = action_name

        return results

    def speak(self, meaning: Dict[str, Any]) -> str:
        """
        Generate language from meaning.
        """
        self._ensure_modules()

        utterance = self._modules['language'].generate(
            meaning.get('predicate', 'say'),
            meaning.get('arguments', {})
        )

        return utterance

    def understand(self, utterance: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Understand language input.
        """
        self._ensure_modules()

        understanding = self._modules['language'].understand(utterance, context)

        return understanding

    def remember(self, content: np.ndarray, episode_id: str, emotional_state: float = 0.0):
        """
        Store an episodic memory.
        """
        self._ensure_modules()

        from utils.base_types import EmotionalState, EmotionType
        es = EmotionalState(
            valence=emotional_state,
            arousal=abs(emotional_state),
            intensity=abs(emotional_state),
            primary_emotion=EmotionType.JOY if emotional_state > 0 else EmotionType.SADNESS
        )

        self._modules['memory'].remember_episode(
            content={'data': content.tolist()},
            embedding=content,
            emotional_state=es
        )

        # Also add to cognitive maps for mental time travel
        self._modules['cognitive_maps'].remember_experience(
            episode_id,
            content,
            {'step': self.step_count},
            emotional_state
        )

    def imagine(self, goal: np.ndarray, time_ahead: float = 3600) -> Dict[str, Any]:
        """
        Imagine a future scenario.
        """
        self._ensure_modules()

        # Use cognitive maps for mental time travel
        future_scenario = self._modules['cognitive_maps'].imagine_going_to(
            f'future_{self.step_count}', goal
        )

        # Use creativity for elaboration
        if 'creativity' in self._modules:
            creative_result = self._modules['creativity'].creative_process(
                goal, time_budget=50
            )
            return {
                'scenario': future_scenario,
                'creative_elaboration': creative_result
            }

        return {'scenario': future_scenario}

    def learn(self, experience: Dict[str, Any]):
        """
        Learn from an experience.
        """
        self._ensure_modules()

        # Hebbian learning
        if 'pre' in experience and 'post' in experience:
            weights = np.random.randn(self.dim, self.dim) * 0.1
            self._modules['learning'].update(
                weights,
                experience['pre'],
                experience['post']
            )

        # Reward learning
        if 'reward' in experience:
            self._modules['motivation'].process_reward(
                str(self.step_count),
                experience['reward']
            )

            # Update dual-process with feedback
            if 'input' in experience and 'response' in experience:
                self._modules['dual_process'].learn(
                    experience['input'],
                    experience['response'],
                    experience['reward']
                )

    def sleep(self, duration: float = 8 * 3600) -> Dict[str, Any]:
        """
        Engage in sleep for memory consolidation.
        """
        self._ensure_modules()

        if 'sleep' not in self._modules:
            return {'error': 'Sleep not enabled'}

        # Start sleep
        self._modules['sleep'].start_sleep()

        # Simulate sleep cycles
        results = []
        elapsed = 0
        step_size = 60  # 1-minute steps

        while elapsed < duration:
            step_result = self._modules['sleep'].sleep_step(step_size)
            results.append(step_result)
            elapsed += step_size

        # Wake up
        wake_result = self._modules['sleep'].wake_up()

        return {
            'duration': duration,
            'cycles': wake_result.get('cycles_completed', 0),
            'consolidated': wake_result.get('memories_consolidated', 0),
            'dreams': wake_result.get('dreams_count', 0)
        }

    # ==================== Social Interface ====================

    def meet(self, agent_id: str, agent_embedding: np.ndarray) -> Dict[str, Any]:
        """Meet a new agent."""
        self._ensure_modules()

        if 'social' not in self._modules:
            return {'error': 'Social cognition not enabled'}

        agent = self._modules['social'].meet_agent(
            agent_id, agent_id, agent_embedding
        )

        return {
            'agent_id': agent_id,
            'initial_trust': self._modules['social'].reputation.get_trust(agent_id)
        }

    def infer_mental_state(self, agent_id: str, observation: Dict) -> Dict[str, Any]:
        """Infer another agent's mental state (Theory of Mind)."""
        self._ensure_modules()

        if 'social' not in self._modules:
            return {'error': 'Social cognition not enabled'}

        result = self._modules['social'].theory_of_mind.simulate_perspective(
            agent_id, observation
        )

        return result

    # ==================== Introspection ====================

    def introspect(self) -> Dict[str, Any]:
        """Deep introspection into own cognitive state."""
        self._ensure_modules()

        return {
            'self_awareness': self._modules['self_awareness'].get_state(),
            'cognitive_state': self._modules['self_awareness'].current_cognitive_state.name,
            'am_i_aware': self._modules['self_awareness'].am_i_aware(),
            'reflection': self._modules['self_awareness'].reflect(depth=2)
        }

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete cognitive state for debugging/analysis."""
        self._ensure_modules()

        state = {
            'step_count': self.step_count,
            'session_duration': time.time() - self.session_start,
            'modules': {}
        }

        # Collect state from all modules
        for name, module in self._modules.items():
            if hasattr(module, 'get_state'):
                state['modules'][name] = module.get_state()

        # Add performance stats
        state['performance'] = {}
        for op_name, times in self.timing_stats.items():
            state['performance'][op_name] = {
                'mean_ms': np.mean(times) * 1000,
                'max_ms': np.max(times) * 1000,
                'count': len(times)
            }

        return state


# ==================== Benchmarking ====================

def benchmark_agent(num_steps: int = 100) -> Dict[str, Any]:
    """Benchmark the cognitive agent."""
    config = CognitiveConfig(dim=64, fast_mode=True)
    agent = CognitiveAgent(config)

    results = {
        'num_steps': num_steps,
        'timings': {}
    }

    # Benchmark perception
    start = time.perf_counter()
    for _ in range(num_steps):
        obs = np.random.randn(64)
        agent.perceive(obs)
    perception_time = time.perf_counter() - start
    results['timings']['perception_per_step_ms'] = (perception_time / num_steps) * 1000

    # Benchmark thinking
    start = time.perf_counter()
    for _ in range(num_steps):
        problem = np.random.randn(64)
        agent.think(problem)
    thinking_time = time.perf_counter() - start
    results['timings']['thinking_per_step_ms'] = (thinking_time / num_steps) * 1000

    # Benchmark decision
    start = time.perf_counter()
    for _ in range(num_steps):
        options = [(f'opt_{i}', np.random.randn(64)) for i in range(3)]
        agent.decide(options)
    decision_time = time.perf_counter() - start
    results['timings']['decision_per_step_ms'] = (decision_time / num_steps) * 1000

    # Total throughput
    total_time = perception_time + thinking_time + decision_time
    results['total_time_seconds'] = total_time
    results['steps_per_second'] = (num_steps * 3) / total_time

    return results


if __name__ == '__main__':
    print("Human Cognition AI - Cognitive Agent")
    print("=" * 50)

    # Quick demo
    agent = CognitiveAgent()

    # Perceive
    obs = np.random.randn(64)
    print("\n1. Perceiving observation...")
    result = agent.perceive(obs)
    print(f"   Prediction error: {result['prediction'].get('weighted_error', 0):.3f}")
    print(f"   Metacognitive confidence: {result['metacognition']['confidence']:.3f}")

    # Think
    print("\n2. Thinking about problem...")
    problem = np.random.randn(64)
    result = agent.think(problem)
    print(f"   System used: {result['dual_process']['system_used']}")
    print(f"   Confidence: {result['dual_process']['confidence']:.3f}")

    # Decide
    print("\n3. Making decision...")
    options = [
        ('explore', np.random.randn(64)),
        ('exploit', np.random.randn(64)),
        ('wait', np.random.randn(64))
    ]
    result = agent.decide(options)
    print(f"   Decision: {result['final_decision']}")

    # Introspect
    print("\n4. Introspecting...")
    intro = agent.introspect()
    print(f"   Cognitive state: {intro['cognitive_state']}")
    print(f"   Can introspect: {intro['am_i_aware']['can_introspect']}")

    # Benchmark
    print("\n5. Running benchmark...")
    bench = benchmark_agent(num_steps=50)
    print(f"   Perception: {bench['timings']['perception_per_step_ms']:.2f} ms/step")
    print(f"   Thinking: {bench['timings']['thinking_per_step_ms']:.2f} ms/step")
    print(f"   Decision: {bench['timings']['decision_per_step_ms']:.2f} ms/step")
    print(f"   Throughput: {bench['steps_per_second']:.1f} steps/sec")
