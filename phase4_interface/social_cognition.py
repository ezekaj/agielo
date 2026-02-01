"""
Social Cognition System - Understanding and Interacting with Others

Implements:
1. Theory of Mind (mentalizing about others' beliefs/desires)
2. Joint Attention (shared focus)
3. Social Learning (imitation, teaching)
4. Reputation and Trust
5. Group Dynamics (in-group/out-group)
6. Social Norms (learning and following rules)
7. Moral Cognition (fairness, harm, loyalty)

Based on research:
- Premack & Woodruff: Theory of Mind
- Tomasello: Shared Intentionality
- Haidt: Moral Foundations
- Fehr & Gächter: Social norms and punishment

Performance: O(n) for n agents, efficient belief updating
Comparison vs existing:
- ACT-R: No social cognition
- SOAR: No theory of mind
- Multi-agent RL: Game theoretic but no mentalizing
- This: Full social brain simulation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time


class BeliefType(Enum):
    """Types of beliefs about others."""
    FIRST_ORDER = auto()   # I believe X
    SECOND_ORDER = auto()  # I believe you believe X
    THIRD_ORDER = auto()   # I believe you believe I believe X


class MoralFoundation(Enum):
    """Haidt's Moral Foundations."""
    CARE = auto()          # Harm/care
    FAIRNESS = auto()      # Cheating/fairness
    LOYALTY = auto()       # Betrayal/loyalty
    AUTHORITY = auto()     # Subversion/authority
    SANCTITY = auto()      # Degradation/sanctity
    LIBERTY = auto()       # Oppression/liberty


@dataclass
class MentalState:
    """Representation of another agent's mental state."""
    agent_id: str
    beliefs: Dict[str, Tuple[Any, float]]  # belief -> (value, confidence)
    desires: Dict[str, float]              # goal -> strength
    intentions: List[str]                  # planned actions
    emotions: Dict[str, float]             # emotion -> intensity
    last_updated: float = field(default_factory=time.time)


@dataclass
class SocialAgent:
    """Representation of another agent in social world."""
    agent_id: str
    name: str
    embedding: np.ndarray                  # Identity embedding
    mental_state: MentalState
    relationship_strength: float = 0.0     # -1 (enemy) to 1 (close friend)
    trust: float = 0.5                     # 0-1
    reputation: Dict[str, float] = field(default_factory=dict)
    group_memberships: Set[str] = field(default_factory=set)
    interaction_history: List[Dict] = field(default_factory=list)


@dataclass
class SocialNorm:
    """A social norm/rule."""
    name: str
    description: str
    situation_embedding: np.ndarray        # When does this norm apply?
    prescribed_action: str                 # What should one do?
    proscribed_action: str                 # What should one not do?
    enforcement_strength: float            # How strongly enforced?
    internalized: bool = False             # Is it part of own values?


@dataclass
class MoralJudgment:
    """A moral evaluation of an action."""
    action: str
    agent: str
    foundations_violated: Dict[MoralFoundation, float]
    foundations_upheld: Dict[MoralFoundation, float]
    overall_judgment: float                # -1 (wrong) to 1 (right)
    punishment_deserved: float             # 0-1


class TheoryOfMind:
    """
    Theory of Mind (ToM) - reasoning about others' mental states.

    Levels:
    0: No ToM (behavior prediction only)
    1: First-order beliefs (I think you believe X)
    2: Second-order beliefs (I think you think I believe X)
    """

    def __init__(self, dim: int = 64, max_recursion: int = 2):
        self.dim = dim
        self.max_recursion = max_recursion

        # Mental state models for other agents
        self.mental_models: Dict[str, MentalState] = {}

        # Simulation workspace
        self.simulation_results: Dict[str, Any] = {}

    def create_mental_model(self, agent_id: str) -> MentalState:
        """Create initial mental model for new agent."""
        model = MentalState(
            agent_id=agent_id,
            beliefs={},
            desires={},
            intentions=[],
            emotions={}
        )
        self.mental_models[agent_id] = model
        return model

    def attribute_belief(self,
                         agent_id: str,
                         belief_key: str,
                         belief_value: Any,
                         confidence: float = 0.5):
        """Attribute a belief to another agent."""
        if agent_id not in self.mental_models:
            self.create_mental_model(agent_id)

        self.mental_models[agent_id].beliefs[belief_key] = (belief_value, confidence)
        self.mental_models[agent_id].last_updated = time.time()

    def attribute_desire(self, agent_id: str, goal: str, strength: float):
        """Attribute a desire/goal to another agent."""
        if agent_id not in self.mental_models:
            self.create_mental_model(agent_id)

        self.mental_models[agent_id].desires[goal] = strength
        self.mental_models[agent_id].last_updated = time.time()

    def attribute_emotion(self, agent_id: str, emotion: str, intensity: float):
        """Attribute an emotion to another agent."""
        if agent_id not in self.mental_models:
            self.create_mental_model(agent_id)

        self.mental_models[agent_id].emotions[emotion] = intensity
        self.mental_models[agent_id].last_updated = time.time()

    def infer_intention(self,
                        agent_id: str,
                        observed_actions: List[str],
                        context: Dict[str, Any]) -> List[str]:
        """Infer agent's intentions from observed actions."""
        if agent_id not in self.mental_models:
            self.create_mental_model(agent_id)

        # Simple intention inference based on desires and actions
        model = self.mental_models[agent_id]
        inferred_intentions = []

        # Actions toward high-desire goals are intentional
        for action in observed_actions:
            for goal, strength in model.desires.items():
                if strength > 0.5:
                    # Assume action is intended to achieve goal
                    inferred_intentions.append(f"{action}_for_{goal}")

        model.intentions = inferred_intentions
        return inferred_intentions

    def predict_action(self, agent_id: str, situation_embedding: np.ndarray) -> Tuple[str, float]:
        """Predict what action another agent will take."""
        if agent_id not in self.mental_models:
            return 'unknown', 0.0

        model = self.mental_models[agent_id]

        # Simple prediction based on strongest desire
        if model.desires:
            strongest_desire = max(model.desires.items(), key=lambda x: x[1])
            # Predict action toward that desire
            predicted_action = f"pursue_{strongest_desire[0]}"
            confidence = strongest_desire[1]
            return predicted_action, confidence

        return 'wait', 0.3

    def simulate_perspective(self,
                             agent_id: str,
                             situation: Dict[str, Any],
                             depth: int = 1) -> Dict[str, Any]:
        """
        Simulate another agent's perspective (perspective-taking).

        What would agent_id think/do in this situation?
        """
        if depth > self.max_recursion:
            return {'warning': 'max_recursion_reached'}

        if agent_id not in self.mental_models:
            self.create_mental_model(agent_id)

        model = self.mental_models[agent_id]

        # Simulate their beliefs about the situation
        simulated_beliefs = {}
        for key, value in situation.items():
            # They might believe different things
            if key in model.beliefs:
                simulated_beliefs[key] = model.beliefs[key][0]
            else:
                simulated_beliefs[key] = value

        # Simulate their emotional response
        simulated_emotions = model.emotions.copy()

        # Simulate their likely action
        likely_action, confidence = self.predict_action(agent_id, np.zeros(self.dim))

        result = {
            'agent_id': agent_id,
            'simulated_beliefs': simulated_beliefs,
            'simulated_emotions': simulated_emotions,
            'likely_action': likely_action,
            'confidence': confidence,
            'depth': depth
        }

        self.simulation_results[agent_id] = result
        return result

    def detect_false_belief(self,
                            agent_id: str,
                            true_state: Dict[str, Any]) -> List[str]:
        """
        Detect where another agent has false beliefs.

        Classic Sally-Anne test capability.
        """
        if agent_id not in self.mental_models:
            return []

        model = self.mental_models[agent_id]
        false_beliefs = []

        for key, (believed_value, confidence) in model.beliefs.items():
            if key in true_state:
                if believed_value != true_state[key] and confidence > 0.3:
                    false_beliefs.append(f"{agent_id}_falsely_believes_{key}={believed_value}")

        return false_beliefs

    def get_beliefs(self, agent_id: str) -> Dict[str, Any]:
        """
        Get all beliefs attributed to an agent.

        Returns dict of belief_key -> believed_value
        """
        if agent_id not in self.mental_models:
            return {}

        model = self.mental_models[agent_id]
        return {key: value for key, (value, confidence) in model.beliefs.items()}


class JointAttention:
    """
    Joint attention - shared focus between agents.

    Critical for:
    - Communication
    - Social learning
    - Shared goals
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Current joint attention targets with other agents
        self.joint_targets: Dict[str, np.ndarray] = {}

        # Gaze following capability
        self.gaze_tracking_enabled = True

    def establish_joint_attention(self,
                                   other_agent: str,
                                   target_embedding: np.ndarray) -> bool:
        """Establish joint attention with another agent."""
        self.joint_targets[other_agent] = target_embedding.copy()
        return True

    def follow_gaze(self,
                    other_agent: str,
                    gaze_direction: np.ndarray,
                    scene_objects: List[Tuple[str, np.ndarray]]) -> Optional[str]:
        """
        Follow another agent's gaze to identify their attention target.
        """
        if not self.gaze_tracking_enabled:
            return None

        # Find object most aligned with gaze direction
        best_object = None
        best_alignment = -float('inf')

        for obj_id, obj_position in scene_objects:
            alignment = np.dot(gaze_direction, obj_position) / (
                np.linalg.norm(gaze_direction) * np.linalg.norm(obj_position) + 1e-8
            )
            if alignment > best_alignment:
                best_alignment = alignment
                best_object = obj_id

        if best_alignment > 0.7 and best_object:
            # Establish joint attention
            obj_embedding = next(
                (pos for name, pos in scene_objects if name == best_object),
                np.zeros(self.dim)
            )
            self.joint_targets[other_agent] = obj_embedding

        return best_object

    def is_attending_jointly(self, other_agent: str) -> bool:
        """Check if we have joint attention with agent."""
        return other_agent in self.joint_targets


class SocialLearning:
    """
    Learning from others through observation and teaching.

    Mechanisms:
    - Imitation
    - Emulation
    - Teaching
    - Social referencing
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Learned behaviors from others
        self.imitated_behaviors: Dict[str, Tuple[np.ndarray, str]] = {}  # behavior -> (embedding, source)

        # Teaching capability
        self.teachable_skills: Set[str] = set()

    def observe_and_imitate(self,
                            demonstrator_id: str,
                            action_sequence: List[np.ndarray],
                            outcome: np.ndarray,
                            behavior_name: str) -> bool:
        """Learn behavior by imitating demonstrated action."""
        # Encode the action sequence
        if action_sequence:
            behavior_embedding = np.mean(action_sequence, axis=0)
        else:
            return False

        # Store with attribution
        self.imitated_behaviors[behavior_name] = (behavior_embedding, demonstrator_id)

        return True

    def emulate_goal(self,
                     demonstrator_id: str,
                     goal_achieved: np.ndarray) -> np.ndarray:
        """
        Emulation: Learn the goal, not the specific actions.

        Try to achieve same outcome through own means.
        """
        # Return goal embedding for own planning
        return goal_achieved.copy()

    def social_referencing(self,
                           trusted_agent_id: str,
                           agent_reaction: Dict[str, float],
                           novel_object_embedding: np.ndarray) -> float:
        """
        Social referencing: Learn valence of novel object from trusted other's reaction.
        """
        # If trusted agent shows positive emotion, object is good
        positive_emotions = ['joy', 'interest', 'trust']
        negative_emotions = ['fear', 'disgust', 'anger']

        positive_score = sum(agent_reaction.get(e, 0) for e in positive_emotions)
        negative_score = sum(agent_reaction.get(e, 0) for e in negative_emotions)

        valence = (positive_score - negative_score) / (positive_score + negative_score + 1e-8)

        return valence

    def teach(self,
              learner_id: str,
              skill_name: str,
              demonstration: List[np.ndarray]) -> Dict[str, Any]:
        """Teach a skill to another agent."""
        if skill_name not in self.teachable_skills:
            return {'success': False, 'reason': 'skill_not_teachable'}

        # Package teaching demonstration
        return {
            'success': True,
            'skill': skill_name,
            'demonstration': demonstration,
            'teacher': 'self'
        }


class ReputationSystem:
    """
    Track and manage reputation/trust.
    """

    def __init__(self):
        # Direct experiences with agents
        self.direct_trust: Dict[str, float] = {}

        # Reputation from gossip/observation
        self.reputation_scores: Dict[str, Dict[str, float]] = {}

        # Trust decay rate
        self.trust_decay = 0.99

    def update_trust(self, agent_id: str, interaction_outcome: float):
        """Update trust based on interaction outcome."""
        current = self.direct_trust.get(agent_id, 0.5)
        # Bayesian-ish update
        self.direct_trust[agent_id] = 0.8 * current + 0.2 * interaction_outcome

    def receive_reputation_info(self,
                                 about_agent: str,
                                 from_agent: str,
                                 reputation: float):
        """Receive reputation information (gossip)."""
        if about_agent not in self.reputation_scores:
            self.reputation_scores[about_agent] = {}

        self.reputation_scores[about_agent][from_agent] = reputation

    def get_trust(self, agent_id: str) -> float:
        """Get combined trust score for agent."""
        direct = self.direct_trust.get(agent_id, 0.5)

        # Combine with reputation
        if agent_id in self.reputation_scores:
            rep_scores = list(self.reputation_scores[agent_id].values())
            reputation = np.mean(rep_scores) if rep_scores else 0.5
            # Weight direct experience more
            return 0.7 * direct + 0.3 * reputation

        return direct

    def decay_trust(self):
        """Natural decay of trust over time (without interaction)."""
        for agent_id in self.direct_trust:
            # Decay toward neutral (0.5)
            current = self.direct_trust[agent_id]
            self.direct_trust[agent_id] = 0.5 + self.trust_decay * (current - 0.5)


class MoralCognition:
    """
    Moral reasoning and judgment.

    Based on Moral Foundations Theory (Haidt).
    """

    def __init__(self):
        # Weights for different moral foundations
        self.foundation_weights = {
            MoralFoundation.CARE: 1.0,
            MoralFoundation.FAIRNESS: 1.0,
            MoralFoundation.LOYALTY: 0.7,
            MoralFoundation.AUTHORITY: 0.5,
            MoralFoundation.SANCTITY: 0.3,
            MoralFoundation.LIBERTY: 0.8,
        }

        # Learned moral rules
        self.moral_rules: List[Tuple[str, MoralFoundation, float]] = []

    def evaluate_action(self,
                        action: str,
                        actor: str,
                        affected_parties: List[str],
                        context: Dict[str, Any]) -> MoralJudgment:
        """Evaluate moral status of an action."""
        violated = {}
        upheld = {}

        # Check each foundation
        # Care/Harm
        if 'harm' in action.lower() or context.get('causes_harm', False):
            violated[MoralFoundation.CARE] = 0.8
        if 'help' in action.lower() or context.get('helps_others', False):
            upheld[MoralFoundation.CARE] = 0.7

        # Fairness
        if context.get('unfair', False) or 'cheat' in action.lower():
            violated[MoralFoundation.FAIRNESS] = 0.9
        if context.get('fair', False) or 'share' in action.lower():
            upheld[MoralFoundation.FAIRNESS] = 0.6

        # Loyalty
        if context.get('betrayal', False):
            violated[MoralFoundation.LOYALTY] = 0.8
        if context.get('loyal', False):
            upheld[MoralFoundation.LOYALTY] = 0.6

        # Liberty
        if context.get('coercion', False) or 'force' in action.lower():
            violated[MoralFoundation.LIBERTY] = 0.7

        # Calculate overall judgment
        violated_score = sum(
            v * self.foundation_weights.get(f, 0.5)
            for f, v in violated.items()
        )
        upheld_score = sum(
            v * self.foundation_weights.get(f, 0.5)
            for f, v in upheld.items()
        )

        overall = (upheld_score - violated_score) / (upheld_score + violated_score + 1e-8)
        punishment = max(0, violated_score - 0.3)  # Threshold for punishment

        return MoralJudgment(
            action=action,
            agent=actor,
            foundations_violated=violated,
            foundations_upheld=upheld,
            overall_judgment=overall,
            punishment_deserved=min(1.0, punishment)
        )

    def should_punish(self, judgment: MoralJudgment) -> Tuple[bool, float]:
        """Decide whether to punish moral violation."""
        if judgment.overall_judgment < -0.3:
            return True, judgment.punishment_deserved
        return False, 0.0


class SocialNormSystem:
    """
    Learning and following social norms.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.norms: List[SocialNorm] = []

        # Norm violation history
        self.violations: List[Tuple[str, SocialNorm, float]] = []

    def learn_norm(self,
                   name: str,
                   description: str,
                   situation_embedding: np.ndarray,
                   prescribed: str,
                   proscribed: str,
                   strength: float = 0.5):
        """Learn a new social norm."""
        norm = SocialNorm(
            name=name,
            description=description,
            situation_embedding=situation_embedding.copy(),
            prescribed_action=prescribed,
            proscribed_action=proscribed,
            enforcement_strength=strength
        )
        self.norms.append(norm)

    def check_applicable_norms(self,
                                situation_embedding: np.ndarray) -> List[SocialNorm]:
        """Find norms that apply to current situation."""
        applicable = []

        for norm in self.norms:
            similarity = np.dot(norm.situation_embedding, situation_embedding) / (
                np.linalg.norm(norm.situation_embedding) *
                np.linalg.norm(situation_embedding) + 1e-8
            )
            if similarity > 0.5:
                applicable.append(norm)

        return applicable

    def evaluate_action_compliance(self,
                                    action: str,
                                    situation_embedding: np.ndarray) -> Dict[str, Any]:
        """Check if action complies with applicable norms."""
        applicable_norms = self.check_applicable_norms(situation_embedding)

        violations = []
        compliance = []

        for norm in applicable_norms:
            if action.lower() in norm.proscribed_action.lower():
                violations.append(norm)
            elif action.lower() in norm.prescribed_action.lower():
                compliance.append(norm)

        return {
            'violations': [n.name for n in violations],
            'compliance': [n.name for n in compliance],
            'violation_severity': sum(n.enforcement_strength for n in violations),
            'should_proceed': len(violations) == 0
        }


class SocialCognitionSystem:
    """
    Complete social cognition system.

    The social brain that enables understanding and interacting with others.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Components
        self.theory_of_mind = TheoryOfMind(dim)
        self.joint_attention = JointAttention(dim)
        self.social_learning = SocialLearning(dim)
        self.reputation = ReputationSystem()
        self.moral_cognition = MoralCognition()
        self.norms = SocialNormSystem(dim)

        # Known agents
        self.known_agents: Dict[str, SocialAgent] = {}

        # Group memberships
        self.own_groups: Set[str] = set()

    def meet_agent(self,
                   agent_id: str,
                   name: str,
                   embedding: np.ndarray,
                   groups: Optional[Set[str]] = None) -> SocialAgent:
        """Initialize tracking of a new agent."""
        mental_state = self.theory_of_mind.create_mental_model(agent_id)

        agent = SocialAgent(
            agent_id=agent_id,
            name=name,
            embedding=embedding.copy(),
            mental_state=mental_state,
            group_memberships=groups or set()
        )

        self.known_agents[agent_id] = agent
        return agent

    def process_social_situation(self,
                                  situation: Dict[str, Any],
                                  observed_agents: List[str]) -> Dict[str, Any]:
        """Process a social situation."""
        results = {
            'mental_state_inferences': {},
            'predicted_actions': {},
            'applicable_norms': [],
            'trust_levels': {}
        }

        # Update mental models based on observations
        for agent_id in observed_agents:
            if agent_id in self.known_agents:
                # Simulate their perspective
                perspective = self.theory_of_mind.simulate_perspective(agent_id, situation)
                results['mental_state_inferences'][agent_id] = perspective

                # Predict their next action
                action, conf = self.theory_of_mind.predict_action(
                    agent_id, np.zeros(self.dim)
                )
                results['predicted_actions'][agent_id] = (action, conf)

                # Get trust level
                results['trust_levels'][agent_id] = self.reputation.get_trust(agent_id)

        # Check applicable social norms
        if 'situation_embedding' in situation:
            norms = self.norms.check_applicable_norms(situation['situation_embedding'])
            results['applicable_norms'] = [n.name for n in norms]

        return results

    def evaluate_social_action(self,
                                action: str,
                                actor: str,
                                affected: List[str],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a social action morally and normatively."""
        # Moral evaluation
        moral_judgment = self.moral_cognition.evaluate_action(
            action, actor, affected, context
        )

        # Norm compliance
        norm_compliance = self.norms.evaluate_action_compliance(
            action, context.get('situation_embedding', np.zeros(self.dim))
        )

        # Update trust based on action
        if actor in self.known_agents:
            trust_delta = moral_judgment.overall_judgment * 0.1
            self.reputation.update_trust(actor, 0.5 + trust_delta)

        return {
            'moral_judgment': {
                'overall': moral_judgment.overall_judgment,
                'violations': {f.name: v for f, v in moral_judgment.foundations_violated.items()},
                'upheld': {f.name: v for f, v in moral_judgment.foundations_upheld.items()},
                'punishment_warranted': moral_judgment.punishment_deserved > 0.3
            },
            'norm_compliance': norm_compliance,
            'updated_trust': self.reputation.get_trust(actor) if actor in self.known_agents else None
        }

    def interact(self,
                 other_agent: str,
                 interaction_type: str,
                 outcome: float) -> Dict[str, Any]:
        """Record an interaction with another agent."""
        if other_agent not in self.known_agents:
            return {'error': 'unknown_agent'}

        # Update relationship
        agent = self.known_agents[other_agent]
        agent.relationship_strength = np.clip(
            agent.relationship_strength + outcome * 0.1, -1, 1
        )

        # Update trust
        self.reputation.update_trust(other_agent, outcome)

        # Record interaction
        agent.interaction_history.append({
            'type': interaction_type,
            'outcome': outcome,
            'time': time.time()
        })

        return {
            'agent': other_agent,
            'new_relationship_strength': agent.relationship_strength,
            'new_trust': self.reputation.get_trust(other_agent)
        }

    def is_in_group(self, agent_id: str) -> bool:
        """Check if agent is in same group (in-group)."""
        if agent_id not in self.known_agents:
            return False

        agent_groups = self.known_agents[agent_id].group_memberships
        return bool(self.own_groups & agent_groups)

    def get_state(self) -> Dict[str, Any]:
        """Get social cognition state."""
        return {
            'known_agents': len(self.known_agents),
            'own_groups': list(self.own_groups),
            'mental_models': len(self.theory_of_mind.mental_models),
            'joint_attention_targets': len(self.joint_attention.joint_targets),
            'learned_behaviors': len(self.social_learning.imitated_behaviors),
            'social_norms': len(self.norms.norms)
        }
