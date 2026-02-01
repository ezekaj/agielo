"""
Creativity Module - Generative Cognition

Implements:
1. Divergent Thinking (generating many ideas)
2. Convergent Thinking (selecting best ideas)
3. Conceptual Blending (combining concepts)
4. Analogical Reasoning (transfer across domains)
5. Incubation (unconscious processing)
6. Insight (Aha! moments)
7. Default Mode Network simulation (mind-wandering)

Based on research:
- Guilford: Divergent/Convergent thinking
- Fauconnier & Turner: Conceptual blending
- Gentner: Structure mapping in analogy
- Beeman: Right hemisphere and insight
- Raichle: Default Mode Network

Performance: Efficient concept space exploration, parallel idea generation
Comparison vs existing:
- LLMs: Generate text but no structured creativity
- ACT-R: No creativity mechanisms
- SOAR: Impasses but not true creativity
- This: Full creative cognition model
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time
import heapq


class CreativeMode(Enum):
    """Modes of creative thinking."""
    DIVERGENT = auto()     # Generate many ideas
    CONVERGENT = auto()    # Select best ideas
    EXPLORATORY = auto()   # Explore conceptual space
    TRANSFORMATIONAL = auto()  # Change the space itself
    COMBINATIONAL = auto() # Combine existing ideas


@dataclass
class Concept:
    """A concept with structure for analogical mapping."""
    name: str
    embedding: np.ndarray
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (relation, target, strength)
    domain: str = "general"


@dataclass
class Blend:
    """Result of conceptual blending."""
    input_space_1: Concept
    input_space_2: Concept
    generic_space: Dict[str, Any]
    blend_space: Concept
    emergent_properties: List[str]
    compression: float  # How much structure is compressed


@dataclass
class Idea:
    """A creative idea."""
    content: np.ndarray
    description: str
    novelty: float
    usefulness: float
    source_concepts: List[str]
    generation_mode: CreativeMode
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnalogicalMapping:
    """Mapping between source and target domains."""
    source_domain: str
    target_domain: str
    mappings: Dict[str, str]  # source_element -> target_element
    mapping_score: float
    inferences: List[str]  # New knowledge inferred


class DivergentThinking:
    """
    Generate many diverse ideas.

    Uses:
    - Random exploration
    - Semantic spreading activation
    - Constraint relaxation
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Concept space for exploration
        self.concept_embeddings: Dict[str, np.ndarray] = {}

        # Fluency parameters
        self.exploration_radius = 0.5
        self.jump_probability = 0.1

    def generate_ideas(self,
                       seed_concept: np.ndarray,
                       num_ideas: int = 10,
                       constraints: Optional[List[Callable]] = None) -> List[Idea]:
        """Generate diverse ideas from seed concept."""
        ideas = []
        constraints = constraints or []

        current = seed_concept.copy()

        for i in range(num_ideas):
            # Decide exploration strategy
            if np.random.random() < self.jump_probability:
                # Random jump (for flexibility)
                new_point = np.random.randn(self.dim)
                new_point = new_point / (np.linalg.norm(new_point) + 1e-8)
            else:
                # Local exploration (spreading activation)
                noise = np.random.randn(self.dim) * self.exploration_radius
                new_point = current + noise
                new_point = new_point / (np.linalg.norm(new_point) + 1e-8)

            # Check constraints
            valid = all(c(new_point) for c in constraints)
            if not valid:
                continue

            # Compute novelty (distance from seed)
            novelty = np.linalg.norm(new_point - seed_concept)

            # Create idea
            idea = Idea(
                content=new_point,
                description=f"idea_{i}",
                novelty=float(np.clip(novelty, 0, 1)),
                usefulness=0.5,  # To be evaluated
                source_concepts=[],
                generation_mode=CreativeMode.DIVERGENT
            )
            ideas.append(idea)

            # Update current for next iteration
            current = new_point

        return ideas

    def remote_association(self,
                           concepts: List[np.ndarray],
                           max_distance: float = 2.0) -> List[Idea]:
        """
        Find remote associations between concepts.

        Like the Remote Associates Test (RAT).
        """
        if len(concepts) < 2:
            return []

        ideas = []

        # Find points that connect all concepts
        centroid = np.mean(concepts, axis=0)

        # Generate variations around centroid
        for i in range(20):
            point = centroid + np.random.randn(self.dim) * 0.3

            # Score by how well it connects all concepts
            distances = [np.linalg.norm(point - c) for c in concepts]
            avg_distance = np.mean(distances)
            distance_std = np.std(distances)

            # Good association: connects all roughly equally
            quality = 1.0 / (1.0 + distance_std)

            if avg_distance < max_distance and quality > 0.5:
                idea = Idea(
                    content=point,
                    description=f"remote_assoc_{i}",
                    novelty=avg_distance / max_distance,
                    usefulness=quality,
                    source_concepts=[],
                    generation_mode=CreativeMode.COMBINATIONAL
                )
                ideas.append(idea)

        return ideas


class ConceptualBlending:
    """
    Blend concepts to create new meanings.

    Based on Fauconnier & Turner's theory.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def blend(self,
              concept1: Concept,
              concept2: Concept,
              blend_ratio: float = 0.5) -> Blend:
        """
        Blend two concepts.

        The blend inherits structure from both inputs but may have
        emergent properties that neither input has.
        """
        # 1. Find generic space (shared structure)
        shared_properties = {}
        for key in concept1.properties:
            if key in concept2.properties:
                shared_properties[key] = (concept1.properties[key], concept2.properties[key])

        # 2. Create blend space embedding
        blend_embedding = (blend_ratio * concept1.embedding +
                          (1 - blend_ratio) * concept2.embedding)

        # 3. Merge properties
        blend_properties = {}
        for key, value in concept1.properties.items():
            blend_properties[f"{concept1.name}_{key}"] = value
        for key, value in concept2.properties.items():
            blend_properties[f"{concept2.name}_{key}"] = value

        # 4. Merge relations
        blend_relations = concept1.relations + concept2.relations

        # 5. Detect emergent properties
        emergent = []
        # Emergence: properties that arise from combination
        if 'animate' in concept1.properties and 'mechanical' in concept2.properties:
            emergent.append('robot_like')
        if 'container' in concept1.properties and 'emotion' in concept2.properties:
            emergent.append('emotional_depth')

        # 6. Compute compression (how much structure is merged)
        total_structure = (len(concept1.properties) + len(concept2.properties) +
                          len(concept1.relations) + len(concept2.relations))
        blend_structure = len(blend_properties) + len(blend_relations)
        compression = 1.0 - (blend_structure / (total_structure + 1))

        blend_concept = Concept(
            name=f"{concept1.name}_{concept2.name}_blend",
            embedding=blend_embedding,
            properties=blend_properties,
            relations=blend_relations,
            domain="blend"
        )

        return Blend(
            input_space_1=concept1,
            input_space_2=concept2,
            generic_space=shared_properties,
            blend_space=blend_concept,
            emergent_properties=emergent,
            compression=compression
        )

    def elaborate_blend(self, blend: Blend, iterations: int = 5) -> Blend:
        """
        Elaborate blend by running mental simulation.

        "What would happen if this blend were real?"
        """
        current_properties = dict(blend.blend_space.properties)

        for _ in range(iterations):
            # Simple elaboration: infer new properties from existing
            if 'can_move' in str(current_properties) and 'has_goals' in str(current_properties):
                current_properties['can_pursue_goals'] = True

            if 'has_memory' in str(current_properties) and 'can_learn' in str(current_properties):
                current_properties['can_improve'] = True

        blend.blend_space.properties = current_properties
        return blend


class AnalogicalReasoning:
    """
    Reason by analogy - transfer knowledge between domains.

    Based on Gentner's Structure Mapping Theory.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Known domains and their concepts
        self.domains: Dict[str, List[Concept]] = {}

    def add_concept(self, concept: Concept):
        """Add concept to a domain."""
        if concept.domain not in self.domains:
            self.domains[concept.domain] = []
        self.domains[concept.domain].append(concept)

    def find_analogy(self,
                     source: Concept,
                     target_domain: str) -> Optional[AnalogicalMapping]:
        """
        Find analogical mapping from source to target domain.
        """
        if target_domain not in self.domains:
            return None

        best_mapping = None
        best_score = -float('inf')

        for target in self.domains[target_domain]:
            mapping, score = self._compute_mapping(source, target)
            if score > best_score:
                best_score = score
                best_mapping = mapping

        return best_mapping

    def _compute_mapping(self,
                         source: Concept,
                         target: Concept) -> Tuple[AnalogicalMapping, float]:
        """
        Compute structure mapping between concepts.

        Emphasizes relational similarity over surface similarity.
        """
        mappings = {}
        score = 0.0

        # Relational mapping (more important)
        for s_rel, s_target, s_strength in source.relations:
            for t_rel, t_target, t_strength in target.relations:
                if s_rel == t_rel:  # Same relation type
                    mappings[s_target] = t_target
                    score += 1.0 * float(s_strength) * float(t_strength)

        # Property mapping (less important)
        for s_prop in source.properties:
            if s_prop in target.properties:
                score += 0.3

        # Embedding similarity (surface)
        embedding_sim = np.dot(source.embedding, target.embedding) / (
            np.linalg.norm(source.embedding) * np.linalg.norm(target.embedding) + 1e-8
        )
        score += 0.2 * embedding_sim

        # Generate inferences (what can we learn?)
        inferences = []
        for s_rel, s_target, _ in source.relations:
            if s_target not in [t for _, t, _ in target.relations]:
                if s_target in mappings:
                    inferences.append(f"{mappings[s_target]} has relation {s_rel}")

        mapping = AnalogicalMapping(
            source_domain=source.domain,
            target_domain=target.domain,
            mappings=mappings,
            mapping_score=score,
            inferences=inferences
        )

        return mapping, score

    def transfer_knowledge(self,
                           mapping: AnalogicalMapping,
                           source_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from source to target via mapping."""
        transferred = {}

        for source_key, value in source_knowledge.items():
            if source_key in mapping.mappings:
                target_key = mapping.mappings[source_key]
                transferred[target_key] = value

        return transferred


class InsightEngine:
    """
    Generate insights (Aha! moments).

    Based on research showing:
    - Insights involve restructuring of problem
    - Right hemisphere involvement
    - Sudden, unexpected nature
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Problem representations
        self.current_problem: Optional[np.ndarray] = None
        self.problem_history: List[np.ndarray] = []

        # Incubation buffer
        self.incubation_buffer: List[Tuple[np.ndarray, float]] = []

        # Insight threshold
        self.insight_threshold = 0.7

    def set_problem(self, problem_embedding: np.ndarray):
        """Set current problem."""
        self.current_problem = problem_embedding.copy()
        self.problem_history.append(problem_embedding.copy())

    def incubate(self, duration: float = 1.0):
        """
        Incubation: unconscious processing while doing other things.

        During incubation:
        - Problem representation may spontaneously restructure
        - Remote associations may form
        - Fixation may be broken
        """
        if self.current_problem is None:
            return

        # Add to incubation buffer with timestamp
        self.incubation_buffer.append((self.current_problem.copy(), time.time()))

        # Simulate restructuring
        noise = np.random.randn(self.dim) * 0.1 * duration
        self.current_problem = self.current_problem + noise
        self.current_problem /= (np.linalg.norm(self.current_problem) + 1e-8)

    def check_for_insight(self,
                          solution_candidates: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Check if any solution produces an insight.

        Insight = sudden recognition that a candidate solves the problem.
        """
        if self.current_problem is None:
            return None

        for i, candidate in enumerate(solution_candidates):
            # Compute fit
            fit = np.dot(self.current_problem, candidate) / (
                np.linalg.norm(self.current_problem) * np.linalg.norm(candidate) + 1e-8
            )

            # Check against history - insight requires restructuring
            history_fits = []
            for old_problem in self.problem_history:
                old_fit = np.dot(old_problem, candidate) / (
                    np.linalg.norm(old_problem) * np.linalg.norm(candidate) + 1e-8
                )
                history_fits.append(old_fit)

            # Insight: current fit is high but historical fit was low
            if history_fits:
                avg_old_fit = np.mean(history_fits)
                insight_magnitude = fit - avg_old_fit

                if fit > self.insight_threshold and insight_magnitude > 0.3:
                    return {
                        'insight': True,
                        'solution_index': i,
                        'fit': float(fit),
                        'insight_magnitude': float(insight_magnitude),
                        'restructuring_occurred': True
                    }

        return None


class DefaultModeNetwork:
    """
    Simulated Default Mode Network (DMN).

    The DMN is active during:
    - Mind wandering
    - Self-reflection
    - Future planning
    - Creative thinking

    It allows spontaneous, undirected thought.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Memory traces that can be activated
        self.memory_traces: List[np.ndarray] = []

        # Current activation state
        self.activation_state: np.ndarray = np.zeros(dim)

        # Wandering parameters
        self.drift_rate = 0.1
        self.spontaneous_activation_prob = 0.05

    def add_memory_trace(self, trace: np.ndarray):
        """Add memory trace for potential spontaneous activation."""
        self.memory_traces.append(trace.copy())

        # Limit size
        if len(self.memory_traces) > 100:
            self.memory_traces = self.memory_traces[-50:]

    def wander(self, steps: int = 10) -> List[np.ndarray]:
        """
        Let mind wander - undirected thought.

        Returns sequence of thought states.
        """
        thoughts = []

        for _ in range(steps):
            # Drift
            drift = np.random.randn(self.dim) * self.drift_rate
            self.activation_state = self.activation_state + drift

            # Spontaneous memory activation
            if self.memory_traces and np.random.random() < self.spontaneous_activation_prob:
                idx = np.random.randint(len(self.memory_traces))
                memory = self.memory_traces[idx]
                self.activation_state = 0.7 * self.activation_state + 0.3 * memory

            # Normalize
            self.activation_state /= (np.linalg.norm(self.activation_state) + 1e-8)

            thoughts.append(self.activation_state.copy())

        return thoughts

    def get_spontaneous_associations(self,
                                      seed: np.ndarray,
                                      num_associations: int = 5) -> List[np.ndarray]:
        """Get spontaneous associations to a seed concept."""
        associations = []
        current = seed.copy()

        for _ in range(num_associations):
            # Find similar memory traces
            if self.memory_traces:
                similarities = [
                    np.dot(current, m) / (np.linalg.norm(current) * np.linalg.norm(m) + 1e-8)
                    for m in self.memory_traces
                ]

                # Weighted random selection (more similar = more likely)
                probs = np.exp(np.array(similarities) * 2)
                probs = probs / probs.sum()
                idx = np.random.choice(len(self.memory_traces), p=probs)

                associated = self.memory_traces[idx]
                associations.append(associated)

                # Update current
                current = 0.5 * current + 0.5 * associated
                current /= (np.linalg.norm(current) + 1e-8)

        return associations


class CreativityModule:
    """
    Complete creativity system.

    Integrates divergent thinking, blending, analogy, insight, and DMN.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Components
        self.divergent = DivergentThinking(dim)
        self.blending = ConceptualBlending(dim)
        self.analogy = AnalogicalReasoning(dim)
        self.insight = InsightEngine(dim)
        self.dmn = DefaultModeNetwork(dim)

        # Generated ideas
        self.idea_pool: List[Idea] = []

        # Current creative mode
        self.current_mode = CreativeMode.EXPLORATORY

    def creative_process(self,
                         problem_embedding: np.ndarray,
                         constraints: Optional[List[Callable]] = None,
                         time_budget: int = 100) -> Dict[str, Any]:
        """
        Full creative problem-solving process.

        1. Divergent phase: Generate many ideas
        2. Incubation: Let DMN process
        3. Insight check: Look for Aha! moment
        4. Convergent phase: Select best ideas
        """
        results = {
            'ideas': [],
            'insights': [],
            'best_idea': None
        }

        # Set problem for insight detection
        self.insight.set_problem(problem_embedding)

        # Phase 1: Divergent thinking
        self.current_mode = CreativeMode.DIVERGENT
        divergent_ideas = self.divergent.generate_ideas(
            problem_embedding, num_ideas=time_budget // 3, constraints=constraints
        )
        results['ideas'].extend(divergent_ideas)
        self.idea_pool.extend(divergent_ideas)

        # Phase 2: Incubation via DMN
        self.dmn.add_memory_trace(problem_embedding)
        wandering_thoughts = self.dmn.wander(steps=time_budget // 3)

        # Check for insight during wandering
        insight = self.insight.check_for_insight(wandering_thoughts)
        if insight:
            results['insights'].append(insight)

        # Phase 3: Combinational (blending random ideas)
        self.current_mode = CreativeMode.COMBINATIONAL
        if len(self.idea_pool) >= 2:
            for _ in range(min(10, time_budget // 10)):
                idx1, idx2 = np.random.choice(len(self.idea_pool), 2, replace=False)
                idea1, idea2 = self.idea_pool[idx1], self.idea_pool[idx2]

                # Create concepts from ideas
                c1 = Concept(idea1.description, idea1.content, {}, [], "generated")
                c2 = Concept(idea2.description, idea2.content, {}, [], "generated")

                blend = self.blending.blend(c1, c2)

                blend_idea = Idea(
                    content=blend.blend_space.embedding,
                    description=blend.blend_space.name,
                    novelty=blend.compression,
                    usefulness=0.5,
                    source_concepts=[idea1.description, idea2.description],
                    generation_mode=CreativeMode.COMBINATIONAL
                )
                results['ideas'].append(blend_idea)

        # Phase 4: Convergent - evaluate and select
        self.current_mode = CreativeMode.CONVERGENT
        if results['ideas']:
            # Score ideas by novelty * usefulness
            scored = [(i, i.novelty * i.usefulness) for i in results['ideas']]
            scored.sort(key=lambda x: x[1], reverse=True)
            results['best_idea'] = scored[0][0]

        return results

    def brainstorm(self,
                   seed_concepts: List[np.ndarray],
                   num_ideas: int = 20) -> List[Idea]:
        """Brainstorming session."""
        ideas = []

        # Generate from each seed
        for seed in seed_concepts:
            seed_ideas = self.divergent.generate_ideas(seed, num_ideas // len(seed_concepts))
            ideas.extend(seed_ideas)

        # Remote associations between seeds
        remote = self.divergent.remote_association(seed_concepts)
        ideas.extend(remote)

        return ideas

    def find_creative_analogy(self,
                              source_concept: Concept,
                              target_domains: List[str]) -> List[AnalogicalMapping]:
        """Find creative analogies across domains."""
        mappings = []

        for domain in target_domains:
            mapping = self.analogy.find_analogy(source_concept, domain)
            if mapping and mapping.mapping_score > 0.3:
                mappings.append(mapping)

        return mappings

    def get_state(self) -> Dict[str, Any]:
        """Get creativity module state."""
        return {
            'current_mode': self.current_mode.name,
            'idea_pool_size': len(self.idea_pool),
            'domains_known': len(self.analogy.domains),
            'memory_traces': len(self.dmn.memory_traces)
        }
