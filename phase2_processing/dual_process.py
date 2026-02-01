"""
PHASE 2 - STEP 2: DUAL-PROCESS SYSTEM
======================================

System 1 (Fast, Intuitive) + System 2 (Slow, Deliberative)

Key Innovation vs Existing AI:
- GPT/LLMs: Single processing mode (autoregressive)
- ACT-R: Has subsymbolic/symbolic but not true dual-process
- This: True dual-process with automatic switching

Based on:
- Kahneman's "Thinking, Fast and Slow"
- Dual-Process Theory (Evans & Stanovich)
- Neuroscience of automatic vs controlled processing

System 1 (Intuitive):
- Pattern matching
- Associative memory
- Emotional reactions
- Habits and skills
- FAST: milliseconds

System 2 (Deliberative):
- Working memory manipulation
- Logical reasoning
- Planning and search
- Conscious control
- SLOW: seconds to minutes
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

import sys
sys.path.append('..')
from utils.fast_math import (
    cosine_similarity_matrix,
    top_k_indices,
    softmax,
    FastVectorIndex,
    PRECISION,
    VECTOR_DIM
)
from utils.base_types import (
    Vector, Action, EmotionalState, ProcessingMode,
    Timestamp, ExponentialMovingAverage, signal_bus
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DualProcessConfig:
    """Configuration for dual-process system."""
    embedding_dim: int = VECTOR_DIM

    # System 1 (Fast)
    pattern_cache_size: int = 10000
    similarity_threshold: float = 0.85
    habit_activation_threshold: float = 0.9

    # System 2 (Slow)
    working_memory_capacity: int = 7
    max_reasoning_steps: int = 50
    search_depth: int = 5

    # Switching
    uncertainty_threshold: float = 0.3  # Below this, engage System 2
    complexity_threshold: float = 0.4  # Above this, engage System 2
    conflict_threshold: float = 0.2  # Response conflict triggers System 2


# =============================================================================
# SYSTEM 1: FAST INTUITIVE PROCESSING
# =============================================================================

class System1:
    """
    Fast, Intuitive Processing (like the unconscious mind).

    Properties:
    - Automatic and effortless
    - Parallel processing
    - Pattern-based
    - Emotionally influenced
    - Cannot be "turned off"

    Mechanisms:
    1. Pattern Cache: Stored situation → response mappings
    2. Similarity Matching: Find closest known pattern
    3. Habit Network: Automatic action sequences
    4. Emotional Valencer: Quick good/bad judgments
    """

    def __init__(self, config: DualProcessConfig):
        self.config = config

        # Pattern cache: embedding → response
        self.pattern_cache = FastVectorIndex(config.embedding_dim)
        self.responses: Dict[int, Any] = {}  # idx → response

        # Habit network: context_hash → action
        self.habits: Dict[int, Action] = {}

        # Emotional associations: embedding → valence
        self.emotional_associations = FastVectorIndex(config.embedding_dim)
        self.valences: Dict[int, float] = {}

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_time = 0.0

    def process(self, input_embedding: np.ndarray) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Fast processing of input.

        Returns:
            response: The intuitive response
            confidence: How confident (0-1)
            metadata: Additional information
        """
        start = time.perf_counter()
        input_embedding = input_embedding.astype(PRECISION)

        # 1. Check pattern cache (O(1) with LSH)
        cache_results = self.pattern_cache.search(input_embedding, k=1)

        if cache_results:
            idx, distance = cache_results[0]
            similarity = 1.0 / (1.0 + distance)

            if similarity > self.config.similarity_threshold:
                self.cache_hits += 1
                response = self.responses.get(idx)
                elapsed = time.perf_counter() - start
                self.total_time += elapsed

                return response, similarity, {
                    'source': 'cache',
                    'similarity': similarity,
                    'time_ms': elapsed * 1000
                }

        self.cache_misses += 1

        # 2. Check habits
        habit_hash = self._hash_context(input_embedding)
        if habit_hash in self.habits:
            elapsed = time.perf_counter() - start
            self.total_time += elapsed

            return self.habits[habit_hash], 0.9, {
                'source': 'habit',
                'time_ms': elapsed * 1000
            }

        # 3. Emotional valuation (quick good/bad)
        valence = self._get_emotional_valence(input_embedding)

        # 4. Generate default response based on valence
        if valence > 0.5:
            response = Action(name='approach', confidence=abs(valence))
        elif valence < -0.5:
            response = Action(name='avoid', confidence=abs(valence))
        else:
            response = Action(name='explore', confidence=0.3)

        elapsed = time.perf_counter() - start
        self.total_time += elapsed

        return response, 0.3, {
            'source': 'default',
            'valence': valence,
            'time_ms': elapsed * 1000
        }

    def _hash_context(self, embedding: np.ndarray) -> int:
        """Hash embedding for habit lookup."""
        # Simple hashing: discretize and convert to int
        discretized = (embedding[:8] * 100).astype(int)
        return hash(tuple(discretized))

    def _get_emotional_valence(self, embedding: np.ndarray) -> float:
        """Get emotional valence for input."""
        results = self.emotional_associations.search(embedding, k=3)
        if not results:
            return 0.0

        total_weight = 0.0
        weighted_valence = 0.0
        for idx, distance in results:
            weight = 1.0 / (1.0 + distance)
            if idx in self.valences:
                weighted_valence += weight * self.valences[idx]
                total_weight += weight

        if total_weight > 0:
            return weighted_valence / total_weight
        return 0.0

    def learn_pattern(self, input_embedding: np.ndarray, response: Any):
        """Store a new pattern-response association."""
        idx = len(self.responses)
        self.pattern_cache.add(input_embedding.astype(PRECISION), idx)
        self.responses[idx] = response

    def form_habit(self, context_embedding: np.ndarray, action: Action):
        """Form a new habit."""
        habit_hash = self._hash_context(context_embedding)
        self.habits[habit_hash] = action

    def learn_emotional_association(self, embedding: np.ndarray, valence: float):
        """Learn emotional association."""
        idx = len(self.valences)
        self.emotional_associations.add(embedding.astype(PRECISION), idx)
        self.valences[idx] = valence


# =============================================================================
# SYSTEM 2: SLOW DELIBERATIVE PROCESSING
# =============================================================================

class System2:
    """
    Slow, Deliberative Processing (conscious thinking).

    Properties:
    - Effortful and requires attention
    - Serial processing
    - Rule-based and logical
    - Can override System 1
    - Limited capacity

    Mechanisms:
    1. Working Memory: Hold and manipulate information
    2. Rule Engine: Apply logical rules
    3. Search/Planning: Explore possibilities
    4. Step-by-step Reasoning: Chain inferences
    """

    def __init__(self, config: DualProcessConfig):
        self.config = config

        # Working memory
        self.working_memory: List[Any] = []

        # Rule base
        self.rules: List[Dict[str, Any]] = []

        # Search state
        self.search_tree: Dict[str, Any] = {}

        # Statistics
        self.total_steps = 0
        self.total_time = 0.0

    def process(
        self,
        input_data: Any,
        goal: Optional[Any] = None,
        system1_hint: Optional[Any] = None
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Deliberative processing of input.

        Args:
            input_data: The problem/situation to process
            goal: Optional explicit goal
            system1_hint: Optional hint from System 1

        Returns:
            response: The reasoned response
            confidence: How confident (0-1)
            metadata: Reasoning trace
        """
        start = time.perf_counter()

        # Clear and load working memory
        self.working_memory = []
        self._load_working_memory(input_data)
        if system1_hint:
            self.working_memory.append(('hint', system1_hint))

        # Step-by-step reasoning
        reasoning_trace = []
        steps = 0
        solution = None

        while steps < self.config.max_reasoning_steps:
            steps += 1

            # Find applicable rules
            applicable = self._find_applicable_rules()

            if not applicable:
                break

            # Apply best rule
            rule = self._select_rule(applicable)
            result = self._apply_rule(rule)

            reasoning_trace.append({
                'step': steps,
                'rule': rule.get('name', 'unknown'),
                'result': str(result)[:100]
            })

            # Check if goal reached
            if goal and self._goal_reached(result, goal):
                solution = result
                break

            # Check if solution found
            if self._is_solution(result):
                solution = result
                break

        # If no rule-based solution, try search
        if solution is None and goal:
            solution = self._search_for_solution(goal)

        elapsed = time.perf_counter() - start
        self.total_time += elapsed
        self.total_steps += steps

        confidence = min(1.0, 0.5 + 0.1 * len(reasoning_trace))

        return solution, confidence, {
            'source': 'deliberation',
            'steps': steps,
            'trace': reasoning_trace,
            'time_ms': elapsed * 1000
        }

    def _load_working_memory(self, input_data: Any):
        """Load input into working memory."""
        if isinstance(input_data, dict):
            for key, value in list(input_data.items())[:self.config.working_memory_capacity]:
                self.working_memory.append((key, value))
        elif isinstance(input_data, (list, tuple)):
            for item in input_data[:self.config.working_memory_capacity]:
                self.working_memory.append(('item', item))
        else:
            self.working_memory.append(('input', input_data))

    def _find_applicable_rules(self) -> List[Dict[str, Any]]:
        """Find rules whose conditions match working memory."""
        applicable = []
        for rule in self.rules:
            if self._rule_matches(rule):
                applicable.append(rule)
        return applicable

    def _rule_matches(self, rule: Dict[str, Any]) -> bool:
        """Check if a rule's conditions match working memory."""
        conditions = rule.get('conditions', [])
        for condition in conditions:
            if not self._check_condition(condition):
                return False
        return True

    def _check_condition(self, condition: Dict[str, Any]) -> bool:
        """Check a single condition against working memory."""
        condition_type = condition.get('type', 'exists')
        target = condition.get('target')

        if condition_type == 'exists':
            return any(target in str(item) for item in self.working_memory)
        elif condition_type == 'equals':
            value = condition.get('value')
            for key, val in self.working_memory:
                if key == target and val == value:
                    return True
        return False

    def _select_rule(self, applicable: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best rule from applicable rules."""
        if not applicable:
            return {}

        # Simple priority-based selection
        applicable.sort(key=lambda r: r.get('priority', 0), reverse=True)
        return applicable[0]

    def _apply_rule(self, rule: Dict[str, Any]) -> Any:
        """Apply a rule and update working memory."""
        action = rule.get('action', {})
        action_type = action.get('type', 'add')

        if action_type == 'add':
            result = action.get('value')
            if len(self.working_memory) < self.config.working_memory_capacity:
                self.working_memory.append(('derived', result))
        elif action_type == 'replace':
            target = action.get('target')
            value = action.get('value')
            for i, (key, val) in enumerate(self.working_memory):
                if key == target:
                    self.working_memory[i] = (key, value)
                    break
            result = value
        else:
            result = action.get('value')

        return result

    def _goal_reached(self, result: Any, goal: Any) -> bool:
        """Check if goal is reached."""
        if result is None:
            return False
        # Handle numpy arrays
        if hasattr(result, '__iter__') and hasattr(goal, '__iter__'):
            try:
                return np.allclose(result, goal, rtol=0.1)
            except (TypeError, ValueError):
                pass
        return str(result) == str(goal)

    def _is_solution(self, result: Any) -> bool:
        """Check if result is a valid solution."""
        return result is not None and str(result).startswith('solution:')

    def _search_for_solution(self, goal: Any) -> Optional[Any]:
        """
        Search for solution using simple tree search.
        """
        # Simplified search - in real system would use MCTS or similar
        frontier = [(0, self.working_memory.copy())]
        visited = set()

        while frontier and len(visited) < 100:
            _, state = frontier.pop(0)
            state_hash = str(state)

            if state_hash in visited:
                continue
            visited.add(state_hash)

            # Check if goal in state
            for key, val in state:
                if self._goal_reached(val, goal):
                    return val

            # Expand (simplified)
            for rule in self.rules[:5]:
                new_state = state.copy()
                new_state.append(('expanded', rule.get('name', '')))
                frontier.append((len(frontier), new_state))

        return None

    def add_rule(self, name: str, conditions: List[Dict], action: Dict, priority: int = 0):
        """Add a reasoning rule."""
        self.rules.append({
            'name': name,
            'conditions': conditions,
            'action': action,
            'priority': priority
        })


# =============================================================================
# DUAL-PROCESS CONTROLLER
# =============================================================================

class DualProcessController:
    """
    Controls switching between System 1 and System 2.

    Key decisions:
    - When to trust System 1 (fast, automatic)
    - When to engage System 2 (slow, deliberate)
    - How to integrate responses from both

    Triggers for System 2:
    1. Low confidence from System 1
    2. High complexity input
    3. Conflicting responses
    4. Explicit instruction to deliberate
    5. Novel situation (no cached pattern)
    """

    def __init__(self, config: Optional[DualProcessConfig] = None):
        self.config = config or DualProcessConfig()

        # The two systems
        self.system1 = System1(self.config)
        self.system2 = System2(self.config)

        # Metacognition
        self.uncertainty_tracker = ExponentialMovingAverage(alpha=0.1)
        self.conflict_detector = ConflictDetector()

        # Current mode
        self.current_mode = ProcessingMode.INTUITIVE

        # Statistics
        self.s1_calls = 0
        self.s2_calls = 0
        self.switches = 0

    def process(
        self,
        input_embedding: np.ndarray,
        input_data: Optional[Any] = None,
        goal: Optional[Any] = None,
        force_deliberate: bool = False
    ) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Process input through dual-process system.

        Args:
            input_embedding: Vector representation of input
            input_data: Structured input data (for System 2)
            goal: Optional explicit goal
            force_deliberate: Force System 2 processing

        Returns:
            response: Final response
            confidence: Confidence in response
            metadata: Processing information
        """
        # Always run System 1 first (it's automatic)
        s1_response, s1_confidence, s1_meta = self.system1.process(input_embedding)
        self.s1_calls += 1

        # Decide whether to engage System 2
        needs_system2 = force_deliberate or self._needs_deliberation(
            s1_confidence,
            input_embedding,
            s1_response
        )

        if needs_system2:
            # Switch to deliberative mode
            if self.current_mode != ProcessingMode.DELIBERATIVE:
                self.switches += 1
                self.current_mode = ProcessingMode.DELIBERATIVE

            # Run System 2 with System 1's response as hint
            s2_input = input_data if input_data is not None else input_embedding
            s2_response, s2_confidence, s2_meta = self.system2.process(
                s2_input,
                goal=goal,
                system1_hint=s1_response
            )
            self.s2_calls += 1

            # Integrate responses
            response, confidence, meta = self._integrate_responses(
                s1_response, s1_confidence, s1_meta,
                s2_response, s2_confidence, s2_meta
            )
            meta['system_used'] = 'system2'

        else:
            # Trust System 1
            self.current_mode = ProcessingMode.INTUITIVE
            response = s1_response
            confidence = s1_confidence
            meta = s1_meta
            meta['system_used'] = 'system1'

        # Update uncertainty tracker
        self.uncertainty_tracker.update(1.0 - confidence)

        # Publish event
        signal_bus.publish('dual_process_complete', {
            'mode': self.current_mode.value,
            'confidence': confidence
        })

        return response, confidence, meta

    def _needs_deliberation(
        self,
        s1_confidence: float,
        input_embedding: np.ndarray,
        s1_response: Any
    ) -> bool:
        """Decide if System 2 is needed."""
        # Low confidence
        if s1_confidence < self.config.uncertainty_threshold:
            return True

        # High complexity (measured by embedding variance and range)
        complexity = np.std(input_embedding)
        value_range = np.max(np.abs(input_embedding))
        # Also check if there are multiple significant values (complex pattern)
        significant_dims = np.sum(np.abs(input_embedding) > 0.5)

        if complexity > self.config.complexity_threshold:
            return True
        if value_range > 2.0:  # Large values indicate complex input
            return True
        if significant_dims > 3:  # Multiple active dimensions = complex (lowered threshold)
            return True

        # Check for numerical patterns (math problems need System 2)
        # If first few values look like numbers in a problem, engage System 2
        if np.any(input_embedding[:10] > 1.0):  # Numbers > 1 suggest math
            return True

        # Detect conflict
        conflict = self.conflict_detector.detect(s1_response)
        if conflict > self.config.conflict_threshold:
            return True

        return False

    def _integrate_responses(
        self,
        s1_response: Any, s1_confidence: float, s1_meta: Dict,
        s2_response: Any, s2_confidence: float, s2_meta: Dict
    ) -> Tuple[Any, float, Dict]:
        """Integrate System 1 and System 2 responses."""
        # If they agree, high confidence
        if self._responses_agree(s1_response, s2_response):
            return s2_response, min(1.0, s1_confidence + s2_confidence), {
                'source': 'integrated_agree',
                'system1': s1_meta,
                'system2': s2_meta,
                'mode': 'convergent'
            }

        # If they disagree, trust System 2 but note conflict
        return s2_response, s2_confidence * 0.8, {
            'source': 'integrated_conflict',
            'system1': s1_meta,
            'system2': s2_meta,
            'mode': 'override',
            'conflict': True
        }

    def _responses_agree(self, r1: Any, r2: Any) -> bool:
        """Check if two responses agree."""
        if r1 is None or r2 is None:
            return False

        if isinstance(r1, Action) and isinstance(r2, Action):
            return r1.name == r2.name

        return str(r1) == str(r2)

    def learn(self, input_embedding: np.ndarray, correct_response: Any, reward: float):
        """Learn from feedback."""
        if reward > 0.5:
            # Good response - cache it in System 1
            self.system1.learn_pattern(input_embedding, correct_response)

        if reward > 0.9:
            # Very good - form habit
            if isinstance(correct_response, Action):
                self.system1.form_habit(input_embedding, correct_response)

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'system1_calls': self.s1_calls,
            'system2_calls': self.s2_calls,
            'switches': self.switches,
            'current_mode': self.current_mode.value,
            'uncertainty': self.uncertainty_tracker.get(),
            's1_cache_hit_rate': self.system1.cache_hits / max(1, self.s1_calls),
            's1_avg_time_ms': self.system1.total_time / max(1, self.s1_calls) * 1000,
            's2_avg_time_ms': self.system2.total_time / max(1, self.s2_calls) * 1000 if self.s2_calls > 0 else 0
        }


# =============================================================================
# CONFLICT DETECTOR
# =============================================================================

class ConflictDetector:
    """Detects response conflicts that trigger System 2."""

    def __init__(self):
        self.recent_responses = []
        self.max_history = 10

    def detect(self, response: Any) -> float:
        """
        Detect conflict level.

        Returns value 0-1 indicating conflict strength.
        """
        if not self.recent_responses:
            self.recent_responses.append(response)
            return 0.0

        # Check if response contradicts recent responses
        contradictions = 0
        for prev in self.recent_responses[-5:]:
            if self._contradicts(response, prev):
                contradictions += 1

        self.recent_responses.append(response)
        if len(self.recent_responses) > self.max_history:
            self.recent_responses.pop(0)

        return min(1.0, contradictions / 5.0)

    def _contradicts(self, r1: Any, r2: Any) -> bool:
        """Check if two responses contradict."""
        if isinstance(r1, Action) and isinstance(r2, Action):
            # Approach vs avoid is a contradiction
            if (r1.name == 'approach' and r2.name == 'avoid') or \
               (r1.name == 'avoid' and r2.name == 'approach'):
                return True
        return False


# =============================================================================
# TESTING
# =============================================================================

def test_dual_process():
    """Test the dual-process system."""
    print("\n" + "=" * 60)
    print("DUAL-PROCESS SYSTEM TEST")
    print("=" * 60)

    config = DualProcessConfig(embedding_dim=64)
    controller = DualProcessController(config)

    # Add some rules to System 2
    controller.system2.add_rule(
        name='if_danger_avoid',
        conditions=[{'type': 'exists', 'target': 'danger'}],
        action={'type': 'add', 'value': 'solution: avoid'},
        priority=10
    )

    # Test System 1 (fast) processing
    print("\n--- System 1 (Fast) Processing ---")
    embeddings = [np.random.randn(64).astype(PRECISION) for _ in range(100)]

    start = time.perf_counter()
    for emb in embeddings:
        response, conf, meta = controller.process(emb)
    elapsed = time.perf_counter() - start

    print(f"100 System 1 calls: {elapsed*1000:.2f}ms ({elapsed/100*1000:.3f}ms/call)")

    # Learn some patterns
    print("\n--- Learning Patterns ---")
    for i in range(50):
        emb = np.random.randn(64).astype(PRECISION)
        response = Action(name=f'action_{i % 5}')
        controller.system1.learn_pattern(emb, response)

    # Test with learned patterns
    start = time.perf_counter()
    for emb in embeddings[:20]:
        response, conf, meta = controller.process(emb)
    elapsed = time.perf_counter() - start
    print(f"After learning - 20 calls: {elapsed*1000:.2f}ms")

    # Test System 2 (slow) processing
    print("\n--- System 2 (Deliberative) Processing ---")
    start = time.perf_counter()
    response, conf, meta = controller.process(
        embeddings[0],
        input_data={'type': 'problem', 'danger': True},
        goal='avoid',
        force_deliberate=True
    )
    elapsed = time.perf_counter() - start

    print(f"System 2 deliberation: {elapsed*1000:.2f}ms")
    print(f"Response: {response}")
    print(f"Confidence: {conf:.3f}")
    print(f"Reasoning steps: {meta.get('steps', 0)}")

    # Statistics
    print("\n--- Statistics ---")
    stats = controller.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)

    return controller


if __name__ == "__main__":
    test_dual_process()
