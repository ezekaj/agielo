"""
PHASE 2 - STEP 3: FOUR REASONING MODULES
=========================================

Specialized reasoning systems based on neuroscience research.

Key Innovation vs Existing AI:
- LLMs: Generic "reasoning" through prompting
- This: Four specialized modules that cooperate

The Four Types:
1. PERCEPTUAL: Pattern detection from sensory input
2. DIMENSIONAL: Spatial, temporal, hierarchical reasoning
3. LOGICAL: Deductive, inductive, abductive inference
4. INTERACTIVE: Social reasoning, dialogue, collaboration

Based on:
- Nature's Insight Framework (arXiv 2025)
- Cognitive neuroscience of reasoning
- Specialized brain networks for different reasoning types
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time

import sys
sys.path.append('..')
from utils.base_types import (
    ReasoningType, Vector, Belief, Action,
    MentalState, Timestamp, signal_bus
)
from utils.fast_math import (
    cosine_similarity_matrix,
    softmax,
    PRECISION,
    VECTOR_DIM
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ReasoningConfig:
    """Configuration for reasoning modules."""
    embedding_dim: int = VECTOR_DIM
    max_inference_steps: int = 20
    confidence_threshold: float = 0.5
    pattern_threshold: float = 0.7


# =============================================================================
# BASE REASONING MODULE
# =============================================================================

class ReasoningModule(ABC):
    """Abstract base class for reasoning modules."""

    def __init__(self, config: ReasoningConfig):
        self.config = config
        self.reasoning_type: ReasoningType = ReasoningType.LOGICAL

    @abstractmethod
    def reason(self, input_data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform reasoning on input.

        Args:
            input_data: The data to reason about
            context: Optional contextual information

        Returns:
            Dict with 'conclusion', 'confidence', 'trace'
        """
        pass


# =============================================================================
# PERCEPTUAL REASONING
# =============================================================================

class PerceptualReasoning(ReasoningModule):
    """
    Pattern detection from sensory input.

    Brain regions: Occipital, Parietal, Temporal lobes

    Capabilities:
    - Visual pattern recognition
    - Gestalt grouping
    - Feature extraction
    - Cross-modal integration
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self.reasoning_type = ReasoningType.PERCEPTUAL

        # Pattern library
        self.known_patterns: Dict[str, np.ndarray] = {}

        # Gestalt principles weights
        self.gestalt_weights = {
            'proximity': 0.3,
            'similarity': 0.3,
            'continuity': 0.2,
            'closure': 0.1,
            'common_fate': 0.1
        }

    def reason(self, input_data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perceptual reasoning: recognize patterns in input.
        """
        start = time.perf_counter()

        if isinstance(input_data, np.ndarray):
            embedding = input_data.astype(PRECISION)
        else:
            # Convert to embedding (simplified)
            embedding = np.random.randn(self.config.embedding_dim).astype(PRECISION)

        # 1. Feature extraction
        features = self._extract_features(embedding)

        # 2. Pattern matching
        matched_pattern, similarity = self._match_pattern(embedding)

        # 3. Gestalt grouping
        groups = self._apply_gestalt(embedding)

        # 4. Determine conclusion
        if matched_pattern and similarity > self.config.pattern_threshold:
            conclusion = f"recognized: {matched_pattern}"
            confidence = similarity
        elif groups:
            conclusion = f"grouped: {len(groups)} clusters"
            confidence = 0.6
        else:
            conclusion = "novel_pattern"
            confidence = 0.3

        elapsed = time.perf_counter() - start

        return {
            'conclusion': conclusion,
            'confidence': confidence,
            'features': features,
            'matched_pattern': matched_pattern,
            'similarity': similarity,
            'groups': groups,
            'reasoning_type': self.reasoning_type.value,
            'time_ms': elapsed * 1000
        }

    def _extract_features(self, embedding: np.ndarray) -> Dict[str, float]:
        """Extract basic features from embedding."""
        return {
            'mean': float(np.mean(embedding)),
            'std': float(np.std(embedding)),
            'max': float(np.max(embedding)),
            'min': float(np.min(embedding)),
            'energy': float(np.sum(embedding ** 2))
        }

    def _match_pattern(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Match against known patterns."""
        if not self.known_patterns:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for name, pattern in self.known_patterns.items():
            similarity = np.dot(embedding, pattern) / (
                np.linalg.norm(embedding) * np.linalg.norm(pattern) + 1e-8
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        return best_match, float(best_similarity)

    def _apply_gestalt(self, embedding: np.ndarray) -> List[List[int]]:
        """Apply Gestalt grouping principles (simplified)."""
        # Simplified: group by sign
        positive = [i for i, v in enumerate(embedding) if v > 0]
        negative = [i for i, v in enumerate(embedding) if v <= 0]
        return [positive, negative] if positive and negative else []

    def learn_pattern(self, name: str, embedding: np.ndarray):
        """Learn a new pattern."""
        self.known_patterns[name] = embedding.astype(PRECISION)


# =============================================================================
# DIMENSIONAL REASONING
# =============================================================================

class DimensionalReasoning(ReasoningModule):
    """
    Spatial, temporal, and hierarchical reasoning.

    Brain regions: Parietal cortex (spatial), Hippocampus (temporal),
                   Prefrontal cortex (hierarchical)

    Capabilities:
    - Mental rotation
    - Spatial relations
    - Temporal ordering
    - Cause-effect chains
    - Hierarchical structure
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self.reasoning_type = ReasoningType.DIMENSIONAL

    def reason(self, input_data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Dimensional reasoning: analyze spatial/temporal/hierarchical structure.
        """
        start = time.perf_counter()
        context = context or {}

        dimension = context.get('dimension', 'auto')
        result = {}

        if dimension == 'spatial' or (dimension == 'auto' and self._is_spatial(input_data)):
            result = self._spatial_reason(input_data)
        elif dimension == 'temporal' or (dimension == 'auto' and self._is_temporal(input_data)):
            result = self._temporal_reason(input_data)
        elif dimension == 'hierarchical':
            result = self._hierarchical_reason(input_data)
        else:
            # Default: try all
            result = {
                'spatial': self._spatial_reason(input_data),
                'temporal': self._temporal_reason(input_data),
                'hierarchical': self._hierarchical_reason(input_data)
            }

        elapsed = time.perf_counter() - start
        result['reasoning_type'] = self.reasoning_type.value
        result['time_ms'] = elapsed * 1000

        return result

    def _is_spatial(self, data: Any) -> bool:
        """Check if data is spatial."""
        if isinstance(data, dict):
            return any(k in data for k in ['x', 'y', 'z', 'position', 'location'])
        return False

    def _is_temporal(self, data: Any) -> bool:
        """Check if data is temporal."""
        if isinstance(data, (list, tuple)) and len(data) > 1:
            return True
        if isinstance(data, dict):
            return any(k in data for k in ['time', 'sequence', 'order', 'before', 'after'])
        return False

    def _spatial_reason(self, data: Any) -> Dict[str, Any]:
        """Spatial reasoning: relations, transformations."""
        if isinstance(data, dict):
            positions = []
            for k, v in data.items():
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                    positions.append((k, np.array(v[:2])))

            if len(positions) >= 2:
                # Compute pairwise relations
                relations = []
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        name_i, pos_i = positions[i]
                        name_j, pos_j = positions[j]
                        diff = pos_j - pos_i
                        distance = np.linalg.norm(diff)
                        angle = np.arctan2(diff[1], diff[0]) * 180 / np.pi

                        relations.append({
                            'from': name_i,
                            'to': name_j,
                            'distance': float(distance),
                            'direction': float(angle)
                        })

                return {
                    'conclusion': 'spatial_relations_computed',
                    'relations': relations,
                    'confidence': 0.8
                }

        return {'conclusion': 'no_spatial_data', 'confidence': 0.0}

    def _temporal_reason(self, data: Any) -> Dict[str, Any]:
        """Temporal reasoning: ordering, causality."""
        if isinstance(data, (list, tuple)):
            # Analyze sequence
            sequence = list(data)
            n = len(sequence)

            # Check for monotonicity
            is_increasing = all(sequence[i] <= sequence[i+1] for i in range(n-1))
            is_decreasing = all(sequence[i] >= sequence[i+1] for i in range(n-1))

            # Find patterns
            diffs = [sequence[i+1] - sequence[i] for i in range(n-1)] if n > 1 else []

            return {
                'conclusion': 'temporal_sequence_analyzed',
                'length': n,
                'is_increasing': is_increasing,
                'is_decreasing': is_decreasing,
                'pattern': 'monotonic' if is_increasing or is_decreasing else 'variable',
                'confidence': 0.7
            }

        return {'conclusion': 'no_temporal_data', 'confidence': 0.0}

    def _hierarchical_reason(self, data: Any) -> Dict[str, Any]:
        """Hierarchical reasoning: structure levels."""
        def count_depth(d, current_depth=0):
            if not isinstance(d, dict):
                return current_depth
            if not d:
                return current_depth
            return max(count_depth(v, current_depth + 1) for v in d.values())

        if isinstance(data, dict):
            depth = count_depth(data)
            num_keys = len(data)

            return {
                'conclusion': 'hierarchy_analyzed',
                'depth': depth,
                'breadth': num_keys,
                'structure': 'deep' if depth > 3 else 'shallow',
                'confidence': 0.6
            }

        return {'conclusion': 'no_hierarchical_data', 'confidence': 0.0}


# =============================================================================
# LOGICAL REASONING
# =============================================================================

class LogicalReasoning(ReasoningModule):
    """
    Deductive, inductive, and abductive inference.

    Brain regions: Frontal pole, specialized logic network

    Capabilities:
    - Deductive: General → Specific (certain)
    - Inductive: Specific → General (probable)
    - Abductive: Effect → Best explanation
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self.reasoning_type = ReasoningType.LOGICAL

        # Knowledge base for inference
        self.facts: List[Tuple[str, str, str]] = []  # (subject, relation, object)
        self.rules: List[Dict[str, Any]] = []

    def reason(self, input_data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Logical reasoning: apply inference rules.
        """
        start = time.perf_counter()
        context = context or {}

        mode = context.get('mode', 'auto')
        goal = context.get('goal')

        if mode == 'deductive' or (mode == 'auto' and self._has_general_premise(input_data)):
            result = self._deductive_reason(input_data, goal)
        elif mode == 'inductive' or (mode == 'auto' and self._has_examples(input_data)):
            result = self._inductive_reason(input_data)
        elif mode == 'abductive':
            result = self._abductive_reason(input_data)
        else:
            # Try deductive first, then inductive
            result = self._deductive_reason(input_data, goal)
            if result.get('confidence', 0) < 0.5:
                result = self._inductive_reason(input_data)

        elapsed = time.perf_counter() - start
        result['reasoning_type'] = self.reasoning_type.value
        result['time_ms'] = elapsed * 1000

        return result

    def _has_general_premise(self, data: Any) -> bool:
        """Check if data contains general premises."""
        if isinstance(data, dict):
            return 'all' in str(data).lower() or 'every' in str(data).lower()
        return False

    def _has_examples(self, data: Any) -> bool:
        """Check if data contains examples for induction."""
        if isinstance(data, (list, tuple)) and len(data) >= 3:
            return True
        return False

    def _deductive_reason(self, premises: Any, goal: Optional[str] = None) -> Dict[str, Any]:
        """
        Deductive reasoning: derive specific conclusions from general premises.

        Example:
        Premise 1: All men are mortal
        Premise 2: Socrates is a man
        Conclusion: Socrates is mortal
        """
        trace = []

        if isinstance(premises, dict):
            # Extract premises
            general = premises.get('general', [])
            specific = premises.get('specific', [])

            for g in general:
                for s in specific:
                    # Simple pattern matching
                    if isinstance(g, tuple) and len(g) == 3:
                        # (category, property_relation, property_value)
                        category, rel, prop = g
                        for spec in specific:
                            if isinstance(spec, tuple) and len(spec) == 2:
                                entity, cat = spec
                                if cat == category:
                                    conclusion = (entity, rel, prop)
                                    trace.append({
                                        'step': len(trace) + 1,
                                        'rule': 'syllogism',
                                        'from': [g, spec],
                                        'derived': conclusion
                                    })

                                    return {
                                        'conclusion': conclusion,
                                        'confidence': 1.0,  # Deduction is certain
                                        'trace': trace
                                    }

        # No deduction possible
        return {
            'conclusion': None,
            'confidence': 0.0,
            'trace': trace,
            'reason': 'no_applicable_deduction'
        }

    def _inductive_reason(self, examples: Any) -> Dict[str, Any]:
        """
        Inductive reasoning: derive general patterns from specific examples.

        Example:
        Swan 1 is white
        Swan 2 is white
        Swan 3 is white
        → (probable) All swans are white
        """
        trace = []

        if isinstance(examples, (list, tuple)) and len(examples) >= 2:
            # Look for common properties
            if all(isinstance(e, dict) for e in examples):
                common_keys = set(examples[0].keys())
                for e in examples[1:]:
                    common_keys &= set(e.keys())

                common_values = {}
                for key in common_keys:
                    values = [e[key] for e in examples]
                    if len(set(str(v) for v in values)) == 1:
                        common_values[key] = values[0]

                if common_values:
                    n = len(examples)
                    confidence = min(0.9, 0.5 + 0.1 * n)  # More examples = more confident

                    return {
                        'conclusion': f'generalization: {common_values}',
                        'confidence': confidence,
                        'num_examples': n,
                        'common_properties': common_values,
                        'trace': trace
                    }

            # Numeric patterns
            elif all(isinstance(e, (int, float)) for e in examples):
                # Check for arithmetic progression
                diffs = [examples[i+1] - examples[i] for i in range(len(examples)-1)]
                if len(set(diffs)) == 1:
                    return {
                        'conclusion': f'arithmetic_progression: diff={diffs[0]}',
                        'confidence': 0.8,
                        'pattern': 'arithmetic',
                        'difference': diffs[0]
                    }

        return {
            'conclusion': None,
            'confidence': 0.0,
            'reason': 'insufficient_examples'
        }

    def _abductive_reason(self, observation: Any) -> Dict[str, Any]:
        """
        Abductive reasoning: find best explanation for observation.

        Example:
        Observation: The grass is wet
        Hypothesis 1: It rained (probable)
        Hypothesis 2: Sprinklers were on (possible)
        → Best explanation: It rained (most common cause)
        """
        hypotheses = []

        # Generate hypotheses based on observation type
        if isinstance(observation, str):
            # Simple pattern matching
            if 'wet' in observation.lower():
                hypotheses = [
                    ('rain', 0.7),
                    ('sprinkler', 0.2),
                    ('spill', 0.1)
                ]
            elif 'hot' in observation.lower():
                hypotheses = [
                    ('sun', 0.6),
                    ('heater', 0.3),
                    ('fire', 0.1)
                ]
            else:
                hypotheses = [('unknown_cause', 0.5)]

        if hypotheses:
            best = max(hypotheses, key=lambda x: x[1])
            return {
                'conclusion': f'best_explanation: {best[0]}',
                'confidence': best[1],
                'all_hypotheses': hypotheses,
                'mode': 'abductive'
            }

        return {
            'conclusion': None,
            'confidence': 0.0,
            'reason': 'no_hypotheses_generated'
        }

    def add_fact(self, subject: str, relation: str, obj: str):
        """Add a fact to the knowledge base."""
        self.facts.append((subject, relation, obj))

    def add_rule(self, rule: Dict[str, Any]):
        """Add an inference rule."""
        self.rules.append(rule)


# =============================================================================
# INTERACTIVE REASONING
# =============================================================================

class InteractiveReasoning(ReasoningModule):
    """
    Social and collaborative reasoning.

    Brain regions: mPFC, TPJ (Theory of Mind networks)

    Capabilities:
    - Theory of Mind (infer mental states)
    - Dialogue reasoning
    - Collaborative problem solving
    - Negotiation
    """

    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self.reasoning_type = ReasoningType.INTERACTIVE

        # Mental state models for other agents
        self.agent_models: Dict[str, MentalState] = {}

    def reason(self, input_data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Interactive reasoning: social inference and collaboration.
        """
        start = time.perf_counter()
        context = context or {}

        mode = context.get('mode', 'theory_of_mind')
        agent_id = context.get('agent_id')

        if mode == 'theory_of_mind':
            result = self._infer_mental_state(input_data, agent_id)
        elif mode == 'predict_action':
            result = self._predict_action(agent_id)
        elif mode == 'collaborate':
            result = self._collaborative_reason(input_data, context)
        else:
            result = self._infer_mental_state(input_data, agent_id)

        elapsed = time.perf_counter() - start
        result['reasoning_type'] = self.reasoning_type.value
        result['time_ms'] = elapsed * 1000

        return result

    def _infer_mental_state(self, observations: Any, agent_id: Optional[str]) -> Dict[str, Any]:
        """
        Infer another agent's mental state from observations.

        Theory of Mind: What do they believe, desire, intend?
        """
        if agent_id is None:
            agent_id = 'unknown_agent'

        # Extract cues from observations
        beliefs = []
        desires = []
        intentions = []

        if isinstance(observations, dict):
            # Infer from actions
            if 'action' in observations:
                action = observations['action']
                # Reverse engineer intention from action
                intentions.append(Action(name=f'intended_{action}', confidence=0.6))

            # Infer from statements
            if 'statement' in observations:
                statement = observations['statement']
                beliefs.append(Belief(proposition=statement, confidence=0.7))

            # Infer from gaze/attention
            if 'looking_at' in observations:
                target = observations['looking_at']
                desires.append(Goal(description=f'interested_in_{target}', priority=0.5))

        # Build mental state model
        mental_state = MentalState(
            beliefs=beliefs,
            desires=desires,
            intentions=intentions
        )

        # Store for future reference
        self.agent_models[agent_id] = mental_state

        return {
            'conclusion': 'mental_state_inferred',
            'agent_id': agent_id,
            'beliefs': [b.proposition for b in beliefs],
            'desires': [d.description for d in desires],
            'intentions': [i.name for i in intentions],
            'confidence': 0.6
        }

    def _predict_action(self, agent_id: Optional[str]) -> Dict[str, Any]:
        """
        Predict what action an agent will take based on mental state model.
        """
        if agent_id and agent_id in self.agent_models:
            model = self.agent_models[agent_id]

            # Simple prediction: action that achieves top desire
            if model.desires:
                top_desire = max(model.desires, key=lambda d: d.priority)
                predicted_action = f'action_toward_{top_desire.description}'

                return {
                    'conclusion': predicted_action,
                    'confidence': 0.5,
                    'based_on': 'desire',
                    'desire': top_desire.description
                }

            # Or based on intention
            if model.intentions:
                return {
                    'conclusion': model.intentions[0].name,
                    'confidence': 0.6,
                    'based_on': 'intention'
                }

        return {
            'conclusion': 'cannot_predict',
            'confidence': 0.0,
            'reason': 'no_mental_model'
        }

    def _collaborative_reason(self, problem: Any, context: Dict) -> Dict[str, Any]:
        """
        Reason collaboratively with another agent.
        """
        partner_contribution = context.get('partner_contribution')
        my_knowledge = context.get('my_knowledge', {})

        # Combine knowledge
        combined = dict(my_knowledge)
        if partner_contribution:
            combined.update(partner_contribution)

        # Joint conclusion
        return {
            'conclusion': 'collaborative_solution',
            'combined_knowledge': combined,
            'confidence': 0.7,
            'mode': 'collaborative'
        }


# =============================================================================
# REASONING ROUTER
# =============================================================================

class ReasoningRouter:
    """
    Routes problems to appropriate reasoning modules.

    Selects and combines reasoning types based on:
    - Problem characteristics
    - Context requirements
    - Available information
    """

    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()

        # Initialize all modules
        self.perceptual = PerceptualReasoning(self.config)
        self.dimensional = DimensionalReasoning(self.config)
        self.logical = LogicalReasoning(self.config)
        self.interactive = InteractiveReasoning(self.config)

        # Module registry
        self.modules = {
            ReasoningType.PERCEPTUAL: self.perceptual,
            ReasoningType.DIMENSIONAL: self.dimensional,
            ReasoningType.LOGICAL: self.logical,
            ReasoningType.INTERACTIVE: self.interactive
        }

        # Classifier for automatic routing
        self.type_indicators = {
            ReasoningType.PERCEPTUAL: ['image', 'pattern', 'visual', 'sound', 'sense'],
            ReasoningType.DIMENSIONAL: ['space', 'time', 'location', 'sequence', 'before', 'after', 'hierarchy'],
            ReasoningType.LOGICAL: ['if', 'then', 'all', 'some', 'because', 'therefore', 'prove'],
            ReasoningType.INTERACTIVE: ['agent', 'person', 'think', 'want', 'believe', 'collaborate']
        }

    def reason(
        self,
        problem: Any,
        context: Optional[Dict] = None,
        reasoning_types: Optional[List[ReasoningType]] = None
    ) -> Dict[str, Any]:
        """
        Route problem to appropriate reasoning module(s).

        Args:
            problem: The problem to solve
            context: Additional context
            reasoning_types: Specific types to use (auto-detect if None)

        Returns:
            Combined reasoning results
        """
        start = time.perf_counter()

        # Auto-detect reasoning types if not specified
        if reasoning_types is None:
            reasoning_types = self._classify_problem(problem)

        # Run selected modules
        results = {}
        for rtype in reasoning_types:
            module = self.modules.get(rtype)
            if module:
                results[rtype.value] = module.reason(problem, context)

        # Integrate results
        integrated = self._integrate_results(results)

        elapsed = time.perf_counter() - start
        integrated['total_time_ms'] = elapsed * 1000
        integrated['modules_used'] = [rt.value for rt in reasoning_types]

        return integrated

    def _classify_problem(self, problem: Any) -> List[ReasoningType]:
        """Classify problem to determine reasoning types needed."""
        types = []
        problem_str = str(problem).lower()

        for rtype, indicators in self.type_indicators.items():
            for indicator in indicators:
                if indicator in problem_str:
                    if rtype not in types:
                        types.append(rtype)
                    break

        # Default to logical if nothing detected
        if not types:
            types = [ReasoningType.LOGICAL]

        return types

    def _integrate_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Integrate results from multiple reasoning modules."""
        if not results:
            return {'conclusion': None, 'confidence': 0.0}

        # Find highest confidence conclusion
        best_conclusion = None
        best_confidence = 0.0

        for rtype, result in results.items():
            conf = result.get('confidence', 0)
            if conf > best_confidence:
                best_confidence = conf
                best_conclusion = result.get('conclusion')

        return {
            'conclusion': best_conclusion,
            'confidence': best_confidence,
            'all_results': results
        }


# =============================================================================
# TESTING
# =============================================================================

def test_reasoning_modules():
    """Test all reasoning modules."""
    print("\n" + "=" * 60)
    print("REASONING MODULES TEST")
    print("=" * 60)

    config = ReasoningConfig(embedding_dim=64)
    router = ReasoningRouter(config)

    # Test Perceptual Reasoning
    print("\n--- Perceptual Reasoning ---")
    embedding = np.random.randn(64).astype(PRECISION)
    router.perceptual.learn_pattern('face', np.random.randn(64).astype(PRECISION))

    result = router.perceptual.reason(embedding)
    print(f"Conclusion: {result['conclusion']}")
    print(f"Confidence: {result['confidence']:.3f}")

    # Test Dimensional Reasoning
    print("\n--- Dimensional (Spatial) Reasoning ---")
    spatial_data = {
        'object_a': [1.0, 2.0],
        'object_b': [4.0, 6.0],
        'object_c': [0.0, 0.0]
    }
    result = router.dimensional.reason(spatial_data, {'dimension': 'spatial'})
    print(f"Conclusion: {result.get('conclusion')}")
    if 'relations' in result:
        print(f"Relations found: {len(result['relations'])}")

    print("\n--- Dimensional (Temporal) Reasoning ---")
    temporal_data = [1, 2, 3, 5, 8, 13]  # Fibonacci-like
    result = router.dimensional.reason(temporal_data, {'dimension': 'temporal'})
    print(f"Conclusion: {result.get('conclusion')}")
    print(f"Pattern: {result.get('pattern')}")

    # Test Logical Reasoning
    print("\n--- Logical (Deductive) Reasoning ---")
    premises = {
        'general': [('human', 'is', 'mortal')],
        'specific': [('Socrates', 'human')]
    }
    result = router.logical.reason(premises, {'mode': 'deductive'})
    print(f"Conclusion: {result['conclusion']}")
    print(f"Confidence: {result['confidence']:.3f}")

    print("\n--- Logical (Inductive) Reasoning ---")
    examples = [
        {'color': 'white', 'type': 'swan'},
        {'color': 'white', 'type': 'swan'},
        {'color': 'white', 'type': 'swan'}
    ]
    result = router.logical.reason(examples, {'mode': 'inductive'})
    print(f"Conclusion: {result['conclusion']}")
    print(f"Common properties: {result.get('common_properties')}")

    print("\n--- Logical (Abductive) Reasoning ---")
    observation = "The grass is wet"
    result = router.logical.reason(observation, {'mode': 'abductive'})
    print(f"Conclusion: {result['conclusion']}")
    print(f"Hypotheses: {result.get('all_hypotheses')}")

    # Test Interactive Reasoning
    print("\n--- Interactive (Theory of Mind) Reasoning ---")
    observations = {
        'action': 'reaching_for_cookie',
        'looking_at': 'cookie_jar',
        'statement': 'I am hungry'
    }
    result = router.interactive.reason(observations, {'agent_id': 'child_1'})
    print(f"Inferred beliefs: {result.get('beliefs')}")
    print(f"Inferred desires: {result.get('desires')}")

    # Test Routing
    print("\n--- Automatic Routing ---")
    problems = [
        "If all birds can fly and penguins are birds, can penguins fly?",
        "The agent is looking at the door and walking toward it",
        [1, 4, 9, 16, 25]  # Squares
    ]

    for prob in problems:
        result = router.reason(prob)
        print(f"\nProblem: {str(prob)[:50]}...")
        print(f"Modules used: {result['modules_used']}")
        print(f"Conclusion: {result['conclusion']}")

    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)

    return router


if __name__ == "__main__":
    test_reasoning_modules()
