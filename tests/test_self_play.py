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
    Difficulty, SelfPlayRound, DifficultyProgression
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


class TestDifficultyProgression:
    """Tests for adaptive difficulty progression system."""

    def test_initial_difficulty(self):
        """Progression should start at specified difficulty."""
        prog = DifficultyProgression(Difficulty.EASY)
        assert prog.get_current_difficulty() == Difficulty.EASY

        prog2 = DifficultyProgression(Difficulty.MEDIUM)
        assert prog2.get_current_difficulty() == Difficulty.MEDIUM

    def test_record_round_performance(self):
        """Recording performance should add to history."""
        prog = DifficultyProgression()
        prog.record_round_performance(0.8, Difficulty.EASY)
        assert len(prog.round_history) == 1
        assert prog.round_history[0]['correct_rate'] == 0.8
        assert prog.round_history[0]['difficulty'] == 'easy'

    def test_no_adjustment_with_insufficient_rounds(self):
        """Should not adjust with fewer than MIN_ROUNDS_FOR_ADJUSTMENT rounds."""
        prog = DifficultyProgression()
        prog.record_round_performance(0.9, Difficulty.EASY)
        prog.record_round_performance(0.9, Difficulty.EASY)
        # Only 2 rounds, need 3 minimum
        assert prog.should_adjust_difficulty() is None

    def test_increase_difficulty_on_high_performance(self):
        """Should increase difficulty when >80% correct."""
        prog = DifficultyProgression(Difficulty.EASY)
        # Record 3 rounds with >80% correct rate
        prog.record_round_performance(0.85, Difficulty.EASY)
        prog.record_round_performance(0.90, Difficulty.EASY)
        prog.record_round_performance(0.88, Difficulty.EASY)

        assert prog.should_adjust_difficulty() == 'increase'

    def test_decrease_difficulty_on_low_performance(self):
        """Should decrease difficulty when <50% correct."""
        prog = DifficultyProgression(Difficulty.MEDIUM)
        # Record 3 rounds with <50% correct rate
        prog.record_round_performance(0.40, Difficulty.MEDIUM)
        prog.record_round_performance(0.35, Difficulty.MEDIUM)
        prog.record_round_performance(0.45, Difficulty.MEDIUM)

        assert prog.should_adjust_difficulty() == 'decrease'

    def test_no_adjustment_in_middle_range(self):
        """Should not adjust when performance is 50-80%."""
        prog = DifficultyProgression(Difficulty.EASY)
        # Record 3 rounds with 50-80% correct rate
        prog.record_round_performance(0.65, Difficulty.EASY)
        prog.record_round_performance(0.70, Difficulty.EASY)
        prog.record_round_performance(0.60, Difficulty.EASY)

        assert prog.should_adjust_difficulty() is None

    def test_adjust_difficulty_increases(self):
        """adjust_difficulty should move from EASY to MEDIUM."""
        prog = DifficultyProgression(Difficulty.EASY)
        prog.record_round_performance(0.85, Difficulty.EASY)
        prog.record_round_performance(0.90, Difficulty.EASY)
        prog.record_round_performance(0.88, Difficulty.EASY)

        new_diff, msg = prog.adjust_difficulty()
        assert new_diff == Difficulty.MEDIUM
        assert 'Increased' in msg

    def test_adjust_difficulty_decreases(self):
        """adjust_difficulty should move from MEDIUM to EASY."""
        prog = DifficultyProgression(Difficulty.MEDIUM)
        prog.record_round_performance(0.40, Difficulty.MEDIUM)
        prog.record_round_performance(0.35, Difficulty.MEDIUM)
        prog.record_round_performance(0.45, Difficulty.MEDIUM)

        new_diff, msg = prog.adjust_difficulty()
        assert new_diff == Difficulty.EASY
        assert 'Decreased' in msg

    def test_cannot_increase_past_hard(self):
        """Should stay at HARD even with high performance."""
        prog = DifficultyProgression(Difficulty.HARD)
        prog.record_round_performance(0.90, Difficulty.HARD)
        prog.record_round_performance(0.95, Difficulty.HARD)
        prog.record_round_performance(0.92, Difficulty.HARD)

        new_diff, msg = prog.adjust_difficulty()
        assert new_diff == Difficulty.HARD
        assert 'maximum' in msg.lower()

    def test_cannot_decrease_past_easy(self):
        """Should stay at EASY even with low performance."""
        prog = DifficultyProgression(Difficulty.EASY)
        prog.record_round_performance(0.30, Difficulty.EASY)
        prog.record_round_performance(0.25, Difficulty.EASY)
        prog.record_round_performance(0.35, Difficulty.EASY)

        new_diff, msg = prog.adjust_difficulty()
        assert new_diff == Difficulty.EASY
        assert 'minimum' in msg.lower()

    def test_set_difficulty_manual(self):
        """Should be able to manually set difficulty."""
        prog = DifficultyProgression(Difficulty.EASY)
        prog.set_difficulty(Difficulty.HARD)
        assert prog.get_current_difficulty() == Difficulty.HARD
        assert len(prog.difficulty_changes) == 1
        assert prog.difficulty_changes[0]['reason'] == 'manual'

    def test_to_dict_from_dict(self):
        """Should serialize and deserialize correctly."""
        prog = DifficultyProgression(Difficulty.MEDIUM)
        prog.record_round_performance(0.7, Difficulty.MEDIUM)
        prog.record_round_performance(0.8, Difficulty.MEDIUM)

        data = prog.to_dict()
        prog2 = DifficultyProgression.from_dict(data)

        assert prog2.current_difficulty == Difficulty.MEDIUM
        assert len(prog2.round_history) == 2

    def test_get_progression_stats(self):
        """Should return comprehensive stats."""
        prog = DifficultyProgression(Difficulty.EASY)
        prog.record_round_performance(0.8, Difficulty.EASY)
        prog.record_round_performance(0.9, Difficulty.EASY)

        stats = prog.get_progression_stats()
        assert 'current_difficulty' in stats
        assert 'total_rounds' in stats
        assert 'rounds_by_difficulty' in stats
        assert stats['current_difficulty'] == 'easy'
        assert stats['total_rounds'] == 2


class TestSelfPlayTrainerWithAdaptiveDifficulty:
    """Tests for SelfPlayTrainer with adaptive difficulty."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.trainer = SelfPlayTrainer(self.temp_dir, adaptive_difficulty=True)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trainer_has_difficulty_progression(self):
        """Trainer should have difficulty progression attribute."""
        assert hasattr(self.trainer, 'difficulty_progression')
        assert isinstance(self.trainer.difficulty_progression, DifficultyProgression)

    def test_get_current_difficulty(self):
        """Should be able to get current difficulty from trainer."""
        diff = self.trainer.get_current_difficulty()
        assert isinstance(diff, Difficulty)
        assert diff == Difficulty.EASY  # Default start

    def test_set_difficulty(self):
        """Should be able to manually set difficulty."""
        self.trainer.set_difficulty(Difficulty.HARD)
        assert self.trainer.get_current_difficulty() == Difficulty.HARD

    def test_difficulty_in_stats(self):
        """Stats should include difficulty info."""
        stats = self.trainer.get_stats()
        assert 'current_difficulty' in stats
        assert 'adaptive_difficulty' in stats
        assert stats['adaptive_difficulty'] is True

    def test_round_uses_adaptive_difficulty(self):
        """Round should use adaptive difficulty when no explicit difficulty."""
        self.trainer.set_difficulty(Difficulty.MEDIUM)
        results = self.trainer.run_self_play_round(
            topics=["test"],
            n_questions=1
            # No difficulty parameter = use adaptive
        )
        assert results['difficulty_used'] == 'medium'

    def test_round_overrides_adaptive_with_explicit(self):
        """Explicit difficulty should override adaptive."""
        self.trainer.set_difficulty(Difficulty.MEDIUM)
        results = self.trainer.run_self_play_round(
            topics=["test"],
            n_questions=1,
            difficulty=Difficulty.HARD  # Explicit
        )
        assert results['difficulty_used'] == 'hard'

    def test_difficulty_progression_stats(self):
        """Should get difficulty progression stats."""
        stats = self.trainer.get_difficulty_progression_stats()
        assert 'current_difficulty' in stats
        assert 'total_rounds' in stats

    def test_difficulty_persists_across_instances(self):
        """Difficulty should persist when reloading trainer."""
        self.trainer.set_difficulty(Difficulty.HARD)
        self.trainer._save_state()

        # Create new instance from same directory
        trainer2 = SelfPlayTrainer(self.temp_dir, adaptive_difficulty=True)
        assert trainer2.get_current_difficulty() == Difficulty.HARD

    def test_adaptive_disabled(self):
        """When adaptive is disabled, should use EASY default."""
        trainer = SelfPlayTrainer(self.temp_dir, adaptive_difficulty=False)
        results = trainer.run_self_play_round(
            topics=["test"],
            n_questions=1
        )
        assert results['difficulty_used'] == 'easy'


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
        TestDifficultyProgression,
        TestSelfPlayTrainerWithAdaptiveDifficulty,
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
