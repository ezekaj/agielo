"""
Self-Play Training System
=========================

AI generates questions, answers them, evaluates itself, learns from mistakes.

This implements a self-play training loop where the AI:
1. Generates questions on various topics at different difficulty levels
2. Attempts to answer the questions using current knowledge
3. Evaluates its answers against ground truth (or generated ground truth)
4. Learns from mistakes by adding corrections to training data
5. Tracks performance metrics over time

Based on research from Self-Evolving Agents Survey and COLMA systems.
"""

import os
import sys
import json
import random
import hashlib
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import EVOLUTION_DIR, KNOWLEDGE_DIR


class Difficulty(Enum):
    """Question difficulty levels."""
    EASY = "easy"           # Factual recall
    MEDIUM = "medium"       # Inference, comparison
    HARD = "hard"           # Multi-step reasoning, synthesis


@dataclass
class SelfPlayQuestion:
    """Represents a self-generated question."""
    id: str
    topic: str
    difficulty: Difficulty
    question: str
    ground_truth: Optional[str]
    generated_at: str
    metadata: Optional[Dict] = None


@dataclass
class SelfPlayAttempt:
    """Represents an attempt to answer a question."""
    question_id: str
    answer: str
    score: float  # 0-1
    evaluation: str
    is_correct: bool
    attempted_at: str


@dataclass
class SelfPlayRound:
    """Represents a complete self-play round."""
    round_id: str
    topics: List[str]
    n_questions: int
    questions_generated: int
    questions_correct: int
    correct_rate: float
    started_at: str
    completed_at: Optional[str]
    difficulty: Difficulty


class SelfPlayTrainer:
    """
    Self-play training system for autonomous improvement.

    The AI generates questions, answers them, evaluates itself,
    and learns from its mistakes to continuously improve.
    """

    # Question templates by difficulty
    EASY_TEMPLATES = [
        "What is {topic}?",
        "Define {topic} in simple terms.",
        "What is the basic concept of {topic}?",
        "Give a brief explanation of {topic}.",
        "What does {topic} refer to?",
    ]

    MEDIUM_TEMPLATES = [
        "Compare and contrast {topic} with related concepts.",
        "What are the main advantages of {topic}?",
        "How does {topic} work in practice?",
        "What are the key components of {topic}?",
        "Explain the relationship between {topic} and its applications.",
        "What are common misconceptions about {topic}?",
    ]

    HARD_TEMPLATES = [
        "Analyze the implications of {topic} for modern systems.",
        "Design a solution using {topic} for a complex problem.",
        "Critically evaluate the limitations of {topic}.",
        "Synthesize information about {topic} to propose improvements.",
        "How would you implement {topic} in a production environment?",
        "What are the long-term consequences of advances in {topic}?",
    ]

    def __init__(self, storage_dir: Path = None, llm_url: str = None):
        """
        Initialize the self-play trainer.

        Args:
            storage_dir: Directory to store self-play state
            llm_url: URL for LLM API (default: LM Studio local)
        """
        self.storage_dir = storage_dir or EVOLUTION_DIR / "self_play"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.llm_url = llm_url or "http://localhost:1234/v1/chat/completions"

        # State files
        self.state_file = self.storage_dir / "self_play_state.json"
        self.questions_file = self.storage_dir / "questions.json"
        self.attempts_file = self.storage_dir / "attempts.json"
        self.rounds_file = self.storage_dir / "rounds.json"

        # Load state
        self._load_state()

        # Metrics tracking
        self.questions_generated = self.state.get('questions_generated', 0)
        self.questions_attempted = self.state.get('questions_attempted', 0)
        self.questions_correct = self.state.get('questions_correct', 0)
        self.improvement_history: List[Dict] = self.state.get('improvement_history', [])

        print(f"[SelfPlay] Initialized with {self.questions_generated} questions generated")

    def _load_state(self):
        """Load self-play state from disk."""
        self.state = {}
        self.questions: List[SelfPlayQuestion] = []
        self.attempts: List[SelfPlayAttempt] = []
        self.rounds: List[SelfPlayRound] = []

        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}

        if self.questions_file.exists():
            try:
                with open(self.questions_file, 'r') as f:
                    data = json.load(f)
                    self.questions = [
                        SelfPlayQuestion(
                            id=q['id'],
                            topic=q['topic'],
                            difficulty=Difficulty(q['difficulty']),
                            question=q['question'],
                            ground_truth=q.get('ground_truth'),
                            generated_at=q['generated_at'],
                            metadata=q.get('metadata')
                        ) for q in data
                    ]
            except Exception:
                pass

        if self.attempts_file.exists():
            try:
                with open(self.attempts_file, 'r') as f:
                    data = json.load(f)
                    self.attempts = [
                        SelfPlayAttempt(
                            question_id=a['question_id'],
                            answer=a['answer'],
                            score=a['score'],
                            evaluation=a['evaluation'],
                            is_correct=a['is_correct'],
                            attempted_at=a['attempted_at']
                        ) for a in data
                    ]
            except Exception:
                pass

        if self.rounds_file.exists():
            try:
                with open(self.rounds_file, 'r') as f:
                    data = json.load(f)
                    self.rounds = [
                        SelfPlayRound(
                            round_id=r['round_id'],
                            topics=r['topics'],
                            n_questions=r['n_questions'],
                            questions_generated=r['questions_generated'],
                            questions_correct=r['questions_correct'],
                            correct_rate=r['correct_rate'],
                            started_at=r['started_at'],
                            completed_at=r.get('completed_at'),
                            difficulty=Difficulty(r['difficulty'])
                        ) for r in data
                    ]
            except Exception:
                pass

    def _save_state(self):
        """Save self-play state to disk."""
        self.state['questions_generated'] = self.questions_generated
        self.state['questions_attempted'] = self.questions_attempted
        self.state['questions_correct'] = self.questions_correct
        self.state['improvement_history'] = self.improvement_history
        self.state['last_updated'] = datetime.now().isoformat()

        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

        with open(self.questions_file, 'w') as f:
            json.dump([
                {
                    'id': q.id,
                    'topic': q.topic,
                    'difficulty': q.difficulty.value,
                    'question': q.question,
                    'ground_truth': q.ground_truth,
                    'generated_at': q.generated_at,
                    'metadata': q.metadata
                } for q in self.questions
            ], f, indent=2)

        with open(self.attempts_file, 'w') as f:
            json.dump([asdict(a) for a in self.attempts], f, indent=2)

        with open(self.rounds_file, 'w') as f:
            json.dump([
                {
                    'round_id': r.round_id,
                    'topics': r.topics,
                    'n_questions': r.n_questions,
                    'questions_generated': r.questions_generated,
                    'questions_correct': r.questions_correct,
                    'correct_rate': r.correct_rate,
                    'started_at': r.started_at,
                    'completed_at': r.completed_at,
                    'difficulty': r.difficulty.value
                } for r in self.rounds
            ], f, indent=2)

    def _call_llm(self, prompt: str, system_prompt: str = None, max_tokens: int = 512) -> Optional[str]:
        """
        Call the LLM API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response

        Returns:
            LLM response or None if failed
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            req = urllib.request.Request(
                self.llm_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[SelfPlay] LLM call failed: {e}")
            return None

    def generate_question(self, topic: str, difficulty: Difficulty = Difficulty.EASY) -> SelfPlayQuestion:
        """
        Generate a question for self-play training.

        Args:
            topic: Topic to generate question about
            difficulty: Question difficulty level

        Returns:
            Generated question
        """
        # Select template based on difficulty
        if difficulty == Difficulty.EASY:
            templates = self.EASY_TEMPLATES
        elif difficulty == Difficulty.MEDIUM:
            templates = self.MEDIUM_TEMPLATES
        else:
            templates = self.HARD_TEMPLATES

        template = random.choice(templates)
        question_text = template.format(topic=topic)

        # Try to generate ground truth using LLM
        ground_truth = None
        system_prompt = """You are a knowledgeable expert. Provide a clear, accurate, and concise answer.
For factual questions, be precise. For analysis questions, provide structured reasoning."""

        llm_response = self._call_llm(question_text, system_prompt, max_tokens=1024)
        if llm_response:
            ground_truth = llm_response

        # Generate question ID
        question_id = hashlib.md5(
            f"{topic}{question_text}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        question = SelfPlayQuestion(
            id=question_id,
            topic=topic,
            difficulty=difficulty,
            question=question_text,
            ground_truth=ground_truth,
            generated_at=datetime.now().isoformat(),
            metadata={'template': template}
        )

        self.questions.append(question)
        self.questions_generated += 1

        print(f"[SelfPlay] Generated question: {question_text[:50]}...")
        return question

    def attempt_answer(self, question: SelfPlayQuestion) -> str:
        """
        Attempt to answer a question using current knowledge.

        This simulates the AI trying to answer without access to
        the ground truth, using only its current knowledge.

        Args:
            question: Question to answer

        Returns:
            AI's answer
        """
        # System prompt that simulates limited knowledge
        system_prompt = """You are an AI assistant with your current knowledge.
Answer the question to the best of your ability.
If you're uncertain, acknowledge it but still provide your best answer.
Be concise but complete."""

        response = self._call_llm(question.question, system_prompt, max_tokens=1024)

        if response:
            print(f"[SelfPlay] Generated answer for: {question.question[:40]}...")
            return response
        else:
            return "Unable to generate answer due to LLM unavailability."

    def evaluate_answer(
        self,
        question: SelfPlayQuestion,
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Tuple[float, str, bool]:
        """
        Evaluate an answer against ground truth.

        Args:
            question: The question that was answered
            answer: The AI's answer
            ground_truth: Expected answer (uses question.ground_truth if not provided)

        Returns:
            (score 0-1, evaluation explanation, is_correct)
        """
        truth = ground_truth or question.ground_truth

        if not truth:
            # No ground truth available - use LLM to evaluate
            eval_prompt = f"""Evaluate this answer for accuracy and completeness.

Question: {question.question}

Answer: {answer}

Rate the answer on a scale of 0-10 and explain your evaluation.
Format: SCORE: X/10
EVALUATION: Your explanation"""

            eval_response = self._call_llm(eval_prompt, max_tokens=512)

            if eval_response:
                try:
                    # Parse score
                    if "SCORE:" in eval_response:
                        score_line = eval_response.split("SCORE:")[1].split("\n")[0]
                        score = float(score_line.split("/")[0].strip()) / 10.0
                    else:
                        score = 0.5  # Default if parsing fails

                    # Parse evaluation
                    if "EVALUATION:" in eval_response:
                        evaluation = eval_response.split("EVALUATION:")[1].strip()
                    else:
                        evaluation = eval_response

                    is_correct = score >= 0.7
                    return score, evaluation, is_correct
                except Exception:
                    return 0.5, "Evaluation parsing failed", False
            else:
                return 0.5, "Evaluation unavailable", False

        # Compare answer to ground truth using LLM
        eval_prompt = f"""Compare this answer to the ground truth and evaluate accuracy.

Question: {question.question}

Ground Truth: {truth}

Answer to Evaluate: {answer}

Rate how well the answer matches the ground truth on a scale of 0-10.
Consider:
- Factual accuracy
- Completeness
- Clarity

Format:
SCORE: X/10
EVALUATION: Your explanation"""

        eval_response = self._call_llm(eval_prompt, max_tokens=512)

        if eval_response:
            try:
                # Parse score
                if "SCORE:" in eval_response:
                    score_line = eval_response.split("SCORE:")[1].split("\n")[0]
                    score = float(score_line.split("/")[0].strip()) / 10.0
                else:
                    score = 0.5

                # Parse evaluation
                if "EVALUATION:" in eval_response:
                    evaluation = eval_response.split("EVALUATION:")[1].strip()
                else:
                    evaluation = eval_response

                is_correct = score >= 0.7
                return score, evaluation, is_correct

            except Exception:
                return 0.5, "Evaluation parsing failed", False
        else:
            # Fallback: simple text similarity
            answer_words = set(answer.lower().split())
            truth_words = set(truth.lower().split())

            if not truth_words:
                return 0.5, "No ground truth words", False

            overlap = len(answer_words & truth_words) / len(truth_words)
            is_correct = overlap >= 0.5

            return overlap, f"Word overlap: {overlap:.1%}", is_correct

    def learn_from_mistake(
        self,
        question: SelfPlayQuestion,
        wrong_answer: str,
        correct_answer: str
    ) -> bool:
        """
        Learn from a mistake by adding correction to training data.

        Args:
            question: The question that was answered wrong
            wrong_answer: The incorrect answer given
            correct_answer: The correct answer

        Returns:
            Whether learning was successful
        """
        # Add to training data for future fine-tuning
        training_file = KNOWLEDGE_DIR / "training_data.jsonl"

        training_entry = {
            'prompt': question.question,
            'completion': correct_answer,
            'topic': question.topic,
            'difficulty': question.difficulty.value,
            'source': 'self_play',
            'timestamp': datetime.now().isoformat(),
            'wrong_answer': wrong_answer[:500],  # Store for analysis
            'metadata': {
                'question_id': question.id,
                'learned_from_mistake': True
            }
        }

        try:
            with open(training_file, 'a') as f:
                f.write(json.dumps(training_entry) + '\n')

            print(f"[SelfPlay] Learned from mistake on: {question.topic}")
            return True
        except Exception as e:
            print(f"[SelfPlay] Failed to save learning: {e}")
            return False

    def run_self_play_round(
        self,
        topics: List[str],
        n_questions: int = 5,
        difficulty: Difficulty = Difficulty.EASY
    ) -> Dict[str, Any]:
        """
        Run a complete self-play training round.

        Args:
            topics: List of topics to generate questions about
            n_questions: Number of questions to generate
            difficulty: Difficulty level for questions

        Returns:
            Round results with metrics
        """
        round_id = hashlib.md5(
            f"{','.join(topics)}{n_questions}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        round_record = SelfPlayRound(
            round_id=round_id,
            topics=topics,
            n_questions=n_questions,
            questions_generated=0,
            questions_correct=0,
            correct_rate=0.0,
            started_at=datetime.now().isoformat(),
            completed_at=None,
            difficulty=difficulty
        )

        print(f"\n[SelfPlay] Starting round {round_id}")
        print(f"[SelfPlay] Topics: {topics}")
        print(f"[SelfPlay] Difficulty: {difficulty.value}")
        print(f"[SelfPlay] Questions: {n_questions}")

        results = {
            'round_id': round_id,
            'questions': [],
            'correct_count': 0,
            'total_score': 0.0,
            'mistakes_learned': 0
        }

        # Generate and attempt questions
        for i in range(n_questions):
            topic = random.choice(topics)

            # Generate question
            question = self.generate_question(topic, difficulty)
            round_record.questions_generated += 1

            # Attempt answer
            answer = self.attempt_answer(question)

            # Evaluate
            score, evaluation, is_correct = self.evaluate_answer(question, answer)

            # Record attempt
            attempt = SelfPlayAttempt(
                question_id=question.id,
                answer=answer,
                score=score,
                evaluation=evaluation,
                is_correct=is_correct,
                attempted_at=datetime.now().isoformat()
            )
            self.attempts.append(attempt)
            self.questions_attempted += 1

            results['total_score'] += score
            results['questions'].append({
                'question_id': question.id,
                'topic': topic,
                'question': question.question,
                'score': score,
                'is_correct': is_correct
            })

            if is_correct:
                results['correct_count'] += 1
                round_record.questions_correct += 1
                self.questions_correct += 1
                print(f"  [{i+1}/{n_questions}] ✓ {topic}: {score:.1%}")
            else:
                # Learn from mistake
                if question.ground_truth:
                    learned = self.learn_from_mistake(
                        question, answer, question.ground_truth
                    )
                    if learned:
                        results['mistakes_learned'] += 1
                print(f"  [{i+1}/{n_questions}] ✗ {topic}: {score:.1%}")

        # Finalize round
        round_record.correct_rate = (
            round_record.questions_correct / round_record.questions_generated
            if round_record.questions_generated > 0 else 0.0
        )
        round_record.completed_at = datetime.now().isoformat()

        self.rounds.append(round_record)

        # Update improvement history
        self.improvement_history.append({
            'timestamp': datetime.now().isoformat(),
            'round_id': round_id,
            'correct_rate': round_record.correct_rate,
            'difficulty': difficulty.value,
            'questions': n_questions
        })

        # Save state
        self._save_state()

        # Calculate results
        results['correct_rate'] = (
            results['correct_count'] / n_questions if n_questions > 0 else 0.0
        )
        results['avg_score'] = results['total_score'] / n_questions if n_questions > 0 else 0.0

        print(f"\n[SelfPlay] Round Complete")
        print(f"[SelfPlay] Correct: {results['correct_count']}/{n_questions} ({results['correct_rate']:.1%})")
        print(f"[SelfPlay] Avg Score: {results['avg_score']:.1%}")
        print(f"[SelfPlay] Mistakes Learned: {results['mistakes_learned']}")

        return results

    def get_correct_rate(self) -> float:
        """Get overall correct rate."""
        if self.questions_attempted == 0:
            return 0.0
        return self.questions_correct / self.questions_attempted

    def get_improvement_over_time(self) -> List[Dict]:
        """
        Get improvement metrics over time.

        Returns:
            List of improvement data points
        """
        return self.improvement_history

    def get_stats(self) -> Dict[str, Any]:
        """Get self-play training statistics."""
        return {
            'questions_generated': self.questions_generated,
            'questions_attempted': self.questions_attempted,
            'questions_correct': self.questions_correct,
            'correct_rate': self.get_correct_rate(),
            'total_rounds': len(self.rounds),
            'improvement_history_points': len(self.improvement_history),
            'last_round': self.rounds[-1].round_id if self.rounds else None,
            'last_round_rate': self.rounds[-1].correct_rate if self.rounds else None
        }

    def get_recent_improvement(self, n_rounds: int = 5) -> float:
        """
        Calculate improvement in recent rounds vs earlier rounds.

        Args:
            n_rounds: Number of recent rounds to consider

        Returns:
            Improvement ratio (>1 means improving, <1 means declining)
        """
        if len(self.improvement_history) < n_rounds * 2:
            return 1.0  # Not enough data

        recent = self.improvement_history[-n_rounds:]
        earlier = self.improvement_history[-(n_rounds * 2):-n_rounds]

        recent_avg = sum(r['correct_rate'] for r in recent) / len(recent)
        earlier_avg = sum(r['correct_rate'] for r in earlier) / len(earlier)

        if earlier_avg == 0:
            return 1.0

        return recent_avg / earlier_avg


# Global instance
_self_play_trainer: Optional[SelfPlayTrainer] = None


def get_self_play_trainer() -> SelfPlayTrainer:
    """Get global self-play trainer instance."""
    global _self_play_trainer
    if _self_play_trainer is None:
        _self_play_trainer = SelfPlayTrainer()
    return _self_play_trainer


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("SELF-PLAY TRAINING SYSTEM TEST")
    print("=" * 60)

    trainer = SelfPlayTrainer(Path("/tmp/test_self_play"))

    # Test 1: Generate a question
    print("\n1. Testing question generation:")
    question = trainer.generate_question("machine learning", Difficulty.EASY)
    print(f"   Question: {question.question}")
    print(f"   Ground truth: {question.ground_truth[:100] if question.ground_truth else 'None'}...")

    # Test 2: Attempt answer
    print("\n2. Testing answer attempt:")
    answer = trainer.attempt_answer(question)
    print(f"   Answer: {answer[:100]}...")

    # Test 3: Evaluate
    print("\n3. Testing evaluation:")
    score, evaluation, is_correct = trainer.evaluate_answer(question, answer)
    print(f"   Score: {score:.1%}")
    print(f"   Correct: {is_correct}")
    print(f"   Evaluation: {evaluation[:100]}...")

    # Test 4: Run a round
    print("\n4. Testing self-play round:")
    results = trainer.run_self_play_round(
        topics=["Python programming", "data structures"],
        n_questions=3,
        difficulty=Difficulty.EASY
    )
    print(f"   Results: {results['correct_count']}/{len(results['questions'])} correct")

    # Test 5: Stats
    print("\n5. Testing stats:")
    stats = trainer.get_stats()
    print(f"   Generated: {stats['questions_generated']}")
    print(f"   Correct rate: {stats['correct_rate']:.1%}")

    print("\n" + "=" * 60)
    print("Self-play tests complete!")
    print("=" * 60)
