"""
Self-Improver: GVU Framework Implementation
============================================

Based on arXiv research:
- "Self-Improving AI Agents through Self-Play" (2512.02731)
- "RISE: Recursive IntroSpEction" (2407.18219)
- "LADDER: Self-Improving LLMs Through Recursive Problem Decomposition"

The GVU Loop:
1. GENERATE - Try a new approach/response
2. VERIFY - Test if it's better than before
3. UPDATE - Keep improvement or revert

This creates TRUE self-improvement:
- AI modifies its own behavior
- Tests the modification
- Keeps what works, discards what doesn't
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Attempt:
    """A single attempt at responding."""
    prompt: str
    response: str
    strategy: str  # Which strategy was used
    score: float   # How good was it (0-1)
    timestamp: str


@dataclass
class Strategy:
    """A response strategy the AI can use."""
    name: str
    description: str
    prompt_modifier: str  # How to modify the prompt
    success_count: int = 0
    fail_count: int = 0
    avg_score: float = 0.5

    @property
    def effectiveness(self) -> float:
        """Calculate how effective this strategy is."""
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5  # Unknown
        return self.success_count / total


class SelfImprover:
    """
    Self-improvement through Generate-Verify-Update loop.

    The AI:
    1. Tries different strategies for responding
    2. Evaluates which worked better
    3. Updates its preferences based on results
    4. Gradually improves over time
    """

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.expanduser("~/.cognitive_ai_knowledge/self_improve")
        os.makedirs(self.storage_path, exist_ok=True)

        # Load or initialize strategies
        self.strategies: Dict[str, Strategy] = {}
        self.attempts: List[Attempt] = []
        self.improvements: List[Dict] = []

        self._load_state()
        self._init_default_strategies()

    def _init_default_strategies(self):
        """Initialize default response strategies."""
        defaults = [
            Strategy(
                name="concise",
                description="Short, direct answers",
                prompt_modifier="Be concise and direct. Answer in 2-3 sentences max."
            ),
            Strategy(
                name="detailed",
                description="Thorough, comprehensive answers",
                prompt_modifier="Provide a detailed, comprehensive explanation with examples."
            ),
            Strategy(
                name="structured",
                description="Organized with headers/bullets",
                prompt_modifier="Structure your response with clear sections and bullet points."
            ),
            Strategy(
                name="socratic",
                description="Ask questions to understand better",
                prompt_modifier="Before answering, ask 1-2 clarifying questions to understand better."
            ),
            Strategy(
                name="step_by_step",
                description="Break down into steps",
                prompt_modifier="Break down your response into clear, numbered steps."
            ),
            Strategy(
                name="analogies",
                description="Use analogies to explain",
                prompt_modifier="Use simple analogies and real-world examples to explain."
            ),
        ]

        for s in defaults:
            if s.name not in self.strategies:
                self.strategies[s.name] = s

    # ==================== GENERATE ====================
    def generate_approach(self, user_input: str) -> Tuple[str, str]:
        """
        GENERATE: Choose a strategy and modify the prompt.

        Returns:
            (modified_prompt, strategy_name)
        """
        # Pick the best performing strategy (with some exploration)
        strategy = self._select_strategy()

        # Modify the prompt with this strategy
        modified_prompt = f"[Strategy: {strategy.description}]\n{strategy.prompt_modifier}\n\nUser: {user_input}"

        return modified_prompt, strategy.name

    def _select_strategy(self) -> Strategy:
        """Select a strategy using epsilon-greedy (exploit best, explore sometimes)."""
        import random

        # 20% chance to explore (try random strategy)
        if random.random() < 0.2:
            return random.choice(list(self.strategies.values()))

        # 80% chance to exploit (use best strategy)
        best = max(self.strategies.values(), key=lambda s: s.effectiveness)
        return best

    # ==================== VERIFY ====================
    def verify_response(self, prompt: str, response: str, strategy_name: str) -> float:
        """
        VERIFY: Evaluate how good the response was.

        Uses multiple signals:
        1. Response length (not too short, not too long)
        2. Relevance to prompt
        3. Structure quality
        4. User engagement (if available)

        Returns:
            Score from 0.0 to 1.0
        """
        scores = []

        # 1. Length score (sweet spot: 100-500 chars)
        length = len(response)
        if length < 50:
            scores.append(0.2)
        elif length < 100:
            scores.append(0.5)
        elif length < 500:
            scores.append(0.9)
        elif length < 1000:
            scores.append(0.7)
        else:
            scores.append(0.5)

        # 2. Relevance (keyword overlap)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
        scores.append(min(overlap * 2, 1.0))

        # 3. Structure score (has organization)
        structure_signals = ['1.', '2.', '-', '*', ':', '\n\n']
        structure_count = sum(1 for s in structure_signals if s in response)
        scores.append(min(structure_count / 3, 1.0))

        # 4. Completeness (doesn't end abruptly)
        if response.rstrip().endswith(('.', '!', '?', '```')):
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Average all scores
        final_score = sum(scores) / len(scores)

        # Record the attempt
        self.attempts.append(Attempt(
            prompt=prompt[:200],
            response=response[:500],
            strategy=strategy_name,
            score=final_score,
            timestamp=datetime.now().isoformat()
        ))

        return final_score

    def verify_with_user_feedback(self, strategy_name: str, was_helpful: bool):
        """
        VERIFY with explicit user feedback.

        Call this when user indicates satisfaction/dissatisfaction.
        """
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            if was_helpful:
                strategy.success_count += 1
            else:
                strategy.fail_count += 1

            # Update average score
            total = strategy.success_count + strategy.fail_count
            strategy.avg_score = strategy.success_count / total if total > 0 else 0.5

            self._save_state()

    # ==================== UPDATE ====================
    def update_from_result(self, strategy_name: str, score: float):
        """
        UPDATE: Adjust strategy effectiveness based on score.

        If score > 0.6: count as success
        If score < 0.4: count as failure
        """
        if strategy_name not in self.strategies:
            return

        strategy = self.strategies[strategy_name]

        if score > 0.6:
            strategy.success_count += 1
        elif score < 0.4:
            strategy.fail_count += 1

        # Recalculate average
        total = strategy.success_count + strategy.fail_count
        if total > 0:
            strategy.avg_score = strategy.success_count / total

        # Record improvement attempt
        self.improvements.append({
            'strategy': strategy_name,
            'score': score,
            'timestamp': datetime.now().isoformat(),
            'effectiveness_after': strategy.effectiveness
        })

        self._save_state()

    # ==================== REFLECT ====================
    def reflect_on_improvements(self) -> Dict:
        """
        Reflect on what's working and what's not.

        Returns analysis of improvement progress.
        """
        if not self.attempts:
            return {'status': 'no data yet'}

        # Calculate stats
        recent = self.attempts[-20:]  # Last 20 attempts
        avg_recent = sum(a.score for a in recent) / len(recent)

        older = self.attempts[:-20] if len(self.attempts) > 20 else []
        avg_older = sum(a.score for a in older) / len(older) if older else 0.5

        # Find best and worst strategies
        strategy_scores = {}
        for attempt in self.attempts:
            if attempt.strategy not in strategy_scores:
                strategy_scores[attempt.strategy] = []
            strategy_scores[attempt.strategy].append(attempt.score)

        best_strategy = max(strategy_scores.items(),
                           key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0,
                           default=(None, []))
        worst_strategy = min(strategy_scores.items(),
                            key=lambda x: sum(x[1])/len(x[1]) if x[1] else 1,
                            default=(None, []))

        improvement = avg_recent - avg_older

        return {
            'total_attempts': len(self.attempts),
            'recent_avg_score': round(avg_recent, 3),
            'older_avg_score': round(avg_older, 3),
            'improvement': round(improvement, 3),
            'is_improving': improvement > 0,
            'best_strategy': best_strategy[0],
            'worst_strategy': worst_strategy[0],
            'strategies': {
                name: {
                    'effectiveness': round(s.effectiveness, 3),
                    'attempts': s.success_count + s.fail_count
                }
                for name, s in self.strategies.items()
            }
        }

    # ==================== LEARN NEW STRATEGY ====================
    def learn_new_strategy(self, name: str, description: str, prompt_modifier: str):
        """
        Learn a completely new strategy from experience or external source.
        """
        self.strategies[name] = Strategy(
            name=name,
            description=description,
            prompt_modifier=prompt_modifier
        )
        self._save_state()

    def evolve_strategy(self, base_strategy: str, modification: str) -> str:
        """
        Create a new strategy by evolving an existing one.

        This is how the AI can invent new approaches!
        """
        if base_strategy not in self.strategies:
            return None

        base = self.strategies[base_strategy]
        new_name = f"{base_strategy}_evolved_{len(self.strategies)}"

        self.strategies[new_name] = Strategy(
            name=new_name,
            description=f"{base.description} + {modification}",
            prompt_modifier=f"{base.prompt_modifier}\nAdditionally: {modification}"
        )

        self._save_state()
        return new_name

    # ==================== PERSISTENCE ====================
    def _save_state(self):
        """Save current state to disk."""
        state = {
            'strategies': {
                name: {
                    'name': s.name,
                    'description': s.description,
                    'prompt_modifier': s.prompt_modifier,
                    'success_count': s.success_count,
                    'fail_count': s.fail_count,
                    'avg_score': s.avg_score
                }
                for name, s in self.strategies.items()
            },
            'improvements': self.improvements[-100:],  # Keep last 100
            'attempts': [
                {
                    'prompt': a.prompt,
                    'response': a.response,
                    'strategy': a.strategy,
                    'score': a.score,
                    'timestamp': a.timestamp
                }
                for a in self.attempts[-100:]  # Keep last 100
            ]
        }

        with open(os.path.join(self.storage_path, 'state.json'), 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from disk."""
        state_file = os.path.join(self.storage_path, 'state.json')
        if not os.path.exists(state_file):
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Load strategies
            for name, data in state.get('strategies', {}).items():
                self.strategies[name] = Strategy(**data)

            # Load improvements
            self.improvements = state.get('improvements', [])

            # Load attempts
            for a in state.get('attempts', []):
                self.attempts.append(Attempt(**a))

        except Exception as e:
            print(f"[SelfImprover] Error loading state: {e}")


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("SELF-IMPROVER TEST (GVU Framework)")
    print("=" * 60)

    improver = SelfImprover("/tmp/test_improver")

    # Simulate GVU loop
    test_prompts = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain transformers",
    ]

    for prompt in test_prompts:
        print(f"\n--- Prompt: {prompt[:30]}... ---")

        # GENERATE
        modified, strategy = improver.generate_approach(prompt)
        print(f"Strategy: {strategy}")

        # Simulate response
        response = f"This is a test response about {prompt}. " * 5

        # VERIFY
        score = improver.verify_response(prompt, response, strategy)
        print(f"Score: {score:.2f}")

        # UPDATE
        improver.update_from_result(strategy, score)

    # REFLECT
    print("\n" + "=" * 60)
    print("REFLECTION")
    print("=" * 60)
    reflection = improver.reflect_on_improvements()
    for k, v in reflection.items():
        print(f"  {k}: {v}")
