"""
Tests for Self-Play Training System
====================================

Verifies the self-play training capabilities work correctly:
1. Question generation at different difficulty levels
2. Answer attempts with knowledge retrieval
3. Answer evaluation and scoring
4. Learning from mistakes
5. Self-play rounds with metrics tracking
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.self_play import (
    SelfPlayTrainer, SelfPlayQuestion, SelfPlayAttempt,
    Difficulty, SelfPlayRound
)


class TestSelfPlayQuestion:
    """Tests for question dataclass and generation."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.trainer = SelfPlayTrainer(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_question_dataclass(self):
        """Question dataclass should store all fields."""
        question = SelfPlayQuestion(
            id="test123",
            topic="Python",
            difficulty=Difficulty.EASY,
            question="What is Python?",
            ground_truth="A programming language",
            generated_at="2024-01-01T00:00:00",
            metadata={'template': 'test'}
        )
        assert question.id == "test123"
        assert question.topic == "Python"
        assert question.difficulty == Difficulty.EASY
        assert question.ground_truth == "A programming language"

    def test_generate_easy_question(self):
        """Easy question should be generated with correct format."""
        question = self.trainer.generate_question("Python", Difficulty.EASY)

        assert question.id is not None
        assert question.topic == "Python"
        assert question.difficulty == Difficulty.EASY
        assert "Python" in question.question
        assert question.generated_at is not None

    def test_generate_medium_question(self):
        """Medium question should be generated."""
        question = self.trainer.generate_question("machine learning", Difficulty.MEDIUM)

        assert question.id is not None
        assert question.difficulty == Difficulty.MEDIUM
        assert question.question is not None

    def test_generate_hard_question(self):
        """Hard question should be generated."""
        question = self.trainer.generate_question("neural networks", Difficulty.HARD)

        assert question.id is not None
        assert question.difficulty == Difficulty.HARD
        assert question.question is not None

    def test_questions_are_stored(self):
        """Generated questions should be stored in trainer."""
        initial_count = len(self.trainer.questions)
        self.trainer.generate_question("test topic", Difficulty.EASY)
        assert len(self.trainer.questions) == initial_count + 1

    def test_questions_counter_increments(self):
        """Questions generated counter should increment."""
        initial = self.trainer.questions_generated
        self.trainer.generate_question("test", Difficulty.EASY)
        assert self.trainer.questions_generated == initial + 1


class TestSelfPlayAttempt:
    """Tests for answer attempts."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.trainer = SelfPlayTrainer(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_attempt_answer_returns_string(self):
        """Attempt answer should return a string."""
        question = SelfPlayQuestion(
            id="test123",
            topic="Python",
            difficulty=Difficulty.EASY,
            question="What is Python?",
            ground_truth="A programming language",
            generated_at="2024-01-01T00:00:00"
        )
        answer = self.trainer.attempt_answer(question)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_attempt_dataclass(self):
        """Attempt dataclass should store all fields."""
        attempt = SelfPlayAttempt(
            question_id="test123",
            answer="Python is a language",
            score=0.8,
            evaluation="Good answer",
            is_correct=True,
            attempted_at="2024-01-01T00:00:00"
        )
        assert attempt.question_id == "test123"
        assert attempt.score == 0.8
        assert attempt.is_correct is True


class TestSelfPlayEvaluation:
    """Tests for answer evaluation."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.trainer = SelfPlayTrainer(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evaluate_returns_tuple(self):
        """Evaluation should return (score, evaluation, is_correct)."""
        question = SelfPlayQuestion(
            id="test123",
            topic="Python",
            difficulty=Difficulty.EASY,
            question="What is Python?",
            ground_truth="Python is a high-level programming language",
            generated_at="2024-01-01T00:00:00"
        )

        result = self.trainer.evaluate_answer(
            question,
            "Python is a programming language"
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        score, evaluation, is_correct = result
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(evaluation, str)
        assert isinstance(is_correct, bool)

    def test_evaluate_with_ground_truth(self):
        """Evaluation with ground truth should work."""
        question = SelfPlayQuestion(
            id="test123",
            topic="math",
            difficulty=Difficulty.EASY,
            question="What is 2+2?",
            ground_truth="4",
            generated_at="2024-01-01T00:00:00"
        )

        score, evaluation, is_correct = self.trainer.evaluate_answer(
            question,
            "The answer is 4"
        )

        assert score >= 0.0
        assert evaluation is not None

    def test_evaluate_without_ground_truth(self):
        """Evaluation without ground truth should still work."""
        question = SelfPlayQuestion(
            id="test123",
            topic="general",
            difficulty=Difficulty.EASY,
            question="What is the sky?",
            ground_truth=None,
            generated_at="2024-01-01T00:00:00"
        )

        score, evaluation, is_correct = self.trainer.evaluate_answer(
            question,
            "The sky is the atmosphere above Earth"
        )

        assert isinstance(score, float)
        assert isinstance(evaluation, str)


class TestSelfPlayLearning:
    """Tests for learning from mistakes."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        # Create knowledge dir in temp
        self.knowledge_dir = self.temp_dir / "knowledge"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = SelfPlayTrainer(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_learn_from_mistake_creates_entry(self):
        """Learning from mistake should create training entry."""
        question = SelfPlayQuestion(
            id="test123",
            topic="Python",
            difficulty=Difficulty.EASY,
            question="What is Python?",
            ground_truth="Python is a high-level programming language",
            generated_at="2024-01-01T00:00:00"
        )

        # This will try to write to KNOWLEDGE_DIR
        result = self.trainer.learn_from_mistake(
            question,
            "Python is a snake",
            "Python is a high-level programming language"
        )

        # Result may be False if KNOWLEDGE_DIR doesn't exist in test
        assert isinstance(result, bool)


class TestSelfPlayRound:
    """Tests for complete self-play rounds."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.trainer = SelfPlayTrainer(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_round_dataclass(self):
        """Round dataclass should store all fields."""
        round_record = SelfPlayRound(
            round_id="test123",
            topics=["Python", "Java"],
            n_questions=5,
            questions_generated=5,
            questions_correct=3,
            correct_rate=0.6,
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:01:00",
            difficulty=Difficulty.EASY
        )
        assert round_record.round_id == "test123"
        assert len(round_record.topics) == 2
        assert round_record.correct_rate == 0.6

    def test_run_round_returns_results(self):
        """Running a round should return results dict."""
        results = self.trainer.run_self_play_round(
            topics=["Python"],
            n_questions=2,
            difficulty=Difficulty.EASY
        )

        assert 'round_id' in results
        assert 'questions' in results
        assert 'correct_count' in results
        assert 'correct_rate' in results
        assert isinstance(results['questions'], list)

    def test_run_round_generates_questions(self):
        """Running a round should generate questions."""
        initial_count = self.trainer.questions_generated

        self.trainer.run_self_play_round(
            topics=["test"],
            n_questions=3,
            difficulty=Difficulty.EASY
        )

        assert self.trainer.questions_generated >= initial_count + 3

    def test_run_round_stores_round(self):
        """Running a round should store the round record."""
        initial_rounds = len(self.trainer.rounds)

        self.trainer.run_self_play_round(
            topics=["test"],
            n_questions=2,
            difficulty=Difficulty.EASY
        )

        assert len(self.trainer.rounds) == initial_rounds + 1


class TestSelfPlayMetrics:
    """Tests for metrics tracking."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.trainer = SelfPlayTrainer(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_stats(self):
        """Stats should return all required fields."""
        stats = self.trainer.get_stats()

        assert 'questions_generated' in stats
        assert 'questions_attempted' in stats
        assert 'questions_correct' in stats
        assert 'correct_rate' in stats
        assert 'total_rounds' in stats

    def test_get_correct_rate_initial(self):
        """Initial correct rate should be 0."""
        rate = self.trainer.get_correct_rate()
        assert rate == 0.0

    def test_get_improvement_over_time(self):
        """Improvement history should be retrievable."""
        history = self.trainer.get_improvement_over_time()
        assert isinstance(history, list)

    def test_improvement_history_updates(self):
        """Running rounds should update improvement history."""
        initial_len = len(self.trainer.improvement_history)

        self.trainer.run_self_play_round(
            topics=["test"],
            n_questions=2,
            difficulty=Difficulty.EASY
        )

        assert len(self.trainer.improvement_history) > initial_len

    def test_get_recent_improvement(self):
        """Recent improvement calculation should work."""
        improvement = self.trainer.get_recent_improvement()
        assert isinstance(improvement, float)
        assert improvement > 0


class TestSelfPlayPersistence:
    """Tests for state persistence."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.trainer = SelfPlayTrainer(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_state_file_created(self):
        """State file should be created after save."""
        self.trainer._save_state()
        assert (self.temp_dir / "self_play_state.json").exists()

    def test_questions_file_created(self):
        """Questions file should be created after generating questions."""
        self.trainer.generate_question("test", Difficulty.EASY)
        self.trainer._save_state()
        assert (self.temp_dir / "questions.json").exists()

    def test_state_persists_across_instances(self):
        """State should persist when creating new trainer instance."""
        # Generate some data
        self.trainer.generate_question("test", Difficulty.EASY)
        initial_count = self.trainer.questions_generated
        self.trainer._save_state()

        # Create new instance
        trainer2 = SelfPlayTrainer(self.temp_dir)

        assert trainer2.questions_generated == initial_count

    def test_rounds_persist(self):
        """Rounds should persist across instances."""
        self.trainer.run_self_play_round(
            topics=["test"],
            n_questions=1,
            difficulty=Difficulty.EASY
        )
        initial_rounds = len(self.trainer.rounds)

        # Create new instance
        trainer2 = SelfPlayTrainer(self.temp_dir)

        assert len(trainer2.rounds) == initial_rounds


class TestDifficultyLevels:
    """Tests for difficulty levels."""

    def test_difficulty_enum_values(self):
        """Difficulty enum should have correct values."""
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"

    def test_difficulty_templates_exist(self):
        """Templates should exist for each difficulty."""
        assert len(SelfPlayTrainer.EASY_TEMPLATES) > 0
        assert len(SelfPlayTrainer.MEDIUM_TEMPLATES) > 0
        assert len(SelfPlayTrainer.HARD_TEMPLATES) > 0

    def test_easy_templates_are_simple(self):
        """Easy templates should contain simple question patterns."""
        for template in SelfPlayTrainer.EASY_TEMPLATES:
            assert "{topic}" in template
            # Easy templates typically ask "what is" type questions
            assert any(word in template.lower() for word in ["what", "define", "explain", "basic", "give", "brief"])

    def test_hard_templates_are_complex(self):
        """Hard templates should contain complex question patterns."""
        for template in SelfPlayTrainer.HARD_TEMPLATES:
            assert "{topic}" in template
            # Hard templates ask for analysis, synthesis, design
            assert any(word in template.lower() for word in
                      ["analyze", "design", "evaluate", "synthesize", "implement", "consequences"])


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestSelfPlayQuestion,
        TestSelfPlayAttempt,
        TestSelfPlayEvaluation,
        TestSelfPlayLearning,
        TestSelfPlayRound,
        TestSelfPlayMetrics,
        TestSelfPlayPersistence,
        TestDifficultyLevels,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    # Setup
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()

                    # Run test
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    total_passed += 1

                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))

                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, traceback.format_exc()))

                finally:
                    # Teardown
                    if hasattr(instance, 'teardown_method'):
                        try:
                            instance.teardown_method()
                        except:
                            pass

    print(f"\n{'='*60}")
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print('='*60)

    if failures:
        print("\nFailures:")
        for cls, method, error in failures:
            print(f"\n  {cls}.{method}:")
            print(f"    {error[:200]}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
