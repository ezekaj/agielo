"""
Test suite for Benchmark System
===============================

Tests for:
- Test structure validation (45+ tests with required fields)
- Scoring functions with known inputs/outputs
- Benchmark history persistence
- Category analysis correctness

Run with: pytest tests/test_benchmark.py -v
"""

import os
import sys
import json
import tempfile
import shutil
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.benchmark import Benchmark, SelfBenchmark


class TestBenchmarkStructure:
    """Test that all benchmark tests have valid structure."""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance with temp storage."""
        temp_dir = tempfile.mkdtemp()
        yield Benchmark(storage_path=temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_minimum_test_count(self, benchmark):
        """Verify we have at least 45 tests for statistical significance."""
        assert len(benchmark.tests) >= 45, f"Expected 45+ tests, got {len(benchmark.tests)}"

    def test_all_tests_have_required_fields(self, benchmark):
        """Each test must have question, answer, category fields."""
        required_fields = ["question", "answer", "category"]

        for i, test in enumerate(benchmark.tests):
            for field in required_fields:
                assert field in test, f"Test {i} missing required field: {field}"
                assert test[field], f"Test {i} has empty {field}"

    def test_all_tests_have_difficulty(self, benchmark):
        """Each test should have a difficulty field."""
        valid_difficulties = ["easy", "medium", "hard"]

        for i, test in enumerate(benchmark.tests):
            assert "difficulty" in test, f"Test {i} missing difficulty field"
            assert test["difficulty"] in valid_difficulties, (
                f"Test {i} has invalid difficulty: {test['difficulty']}"
            )

    def test_all_tests_have_expected_keywords(self, benchmark):
        """Each test should have expected_keywords list."""
        for i, test in enumerate(benchmark.tests):
            assert "expected_keywords" in test, f"Test {i} missing expected_keywords"
            assert isinstance(test["expected_keywords"], list), (
                f"Test {i} expected_keywords should be a list"
            )

    def test_category_distribution(self, benchmark):
        """Verify reasonable distribution across categories."""
        categories = {}
        for test in benchmark.tests:
            cat = test["category"]
            categories[cat] = categories.get(cat, 0) + 1

        # Should have multiple categories
        assert len(categories) >= 5, f"Expected at least 5 categories, got {len(categories)}"

        # Print category distribution for visibility
        print(f"\nCategory distribution: {categories}")

    def test_difficulty_distribution(self, benchmark):
        """Verify reasonable distribution across difficulties."""
        difficulties = {"easy": 0, "medium": 0, "hard": 0}

        for test in benchmark.tests:
            diff = test.get("difficulty", "medium")
            difficulties[diff] = difficulties.get(diff, 0) + 1

        # Should have tests at each difficulty level
        assert difficulties["easy"] > 0, "No easy tests found"
        assert difficulties["medium"] > 0, "No medium tests found"
        assert difficulties["hard"] > 0, "No hard tests found"

        print(f"\nDifficulty distribution: {difficulties}")


class TestScoringFunctions:
    """Test scoring functions with known inputs/outputs."""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        temp_dir = tempfile.mkdtemp()
        yield Benchmark(storage_path=temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_exact_match_correct(self, benchmark):
        """Test exact match scoring with correct answer."""
        test = {
            "question": "What is 2 + 2?",
            "answer": "4",
            "expected_keywords": ["4", "four"],
            "category": "math",
            "check_exact": True,
            "difficulty": "easy"
        }

        # Exact match should score high
        score = benchmark.score_response("The answer is 4.", test)
        assert score >= 0.5, f"Correct exact match should score >= 0.5, got {score}"

    def test_exact_match_incorrect(self, benchmark):
        """Test exact match scoring with wrong answer."""
        test = {
            "question": "What is 2 + 2?",
            "answer": "4",
            "expected_keywords": ["4", "four"],
            "category": "math",
            "check_exact": True,
            "difficulty": "easy"
        }

        # Wrong answer should score low
        score = benchmark.score_response("The answer is 5.", test)
        assert score < 0.5, f"Wrong exact match should score < 0.5, got {score}"

    def test_keyword_matching(self, benchmark):
        """Test that keyword matching works correctly."""
        test = {
            "question": "What is the capital of France?",
            "answer": "paris",
            "expected_keywords": ["paris", "france", "capital"],
            "category": "factual",
            "check_exact": False,
            "difficulty": "easy"
        }

        # Response with all keywords should score high
        full_response = "The capital of France is Paris, a beautiful city."
        score_full = benchmark.score_response(full_response, test)

        # Response with no keywords should score lower
        empty_response = "I don't know the answer."
        score_empty = benchmark.score_response(empty_response, test)

        assert score_full > score_empty, "Full keyword response should score higher"

    def test_reasoning_bonus(self, benchmark):
        """Test that showing reasoning improves score."""
        test = {
            "question": "What is 5 + 3?",
            "answer": "8",
            "expected_keywords": ["8", "eight"],
            "category": "math",
            "check_exact": True,
            "difficulty": "easy"
        }

        # Response with reasoning
        with_reasoning = "Because 5 + 3 equals 8, therefore the answer is 8."
        score_reasoning = benchmark.score_response(with_reasoning, test)

        # Response without reasoning
        no_reasoning = "8"
        score_no_reasoning = benchmark.score_response(no_reasoning, test)

        assert score_reasoning >= score_no_reasoning, (
            "Response with reasoning should score >= response without"
        )

    def test_theory_of_mind_scoring(self, benchmark):
        """Test theory of mind scoring specifically."""
        test = {
            "question": "Sally puts a marble in basket. Anne moves it to box. Where will Sally look?",
            "answer": "basket",
            "expected_keywords": ["basket", "thinks", "believe"],
            "category": "theory_of_mind",
            "check_exact": False,
            "difficulty": "medium"
        }

        # Correct ToM answer
        correct = "Sally will look in the basket because she thinks it's still there."
        score_correct = benchmark.score_response(correct, test)

        # Incorrect ToM answer (doesn't understand false belief)
        incorrect = "Sally will look in the box because that's where it is."
        score_incorrect = benchmark.score_response(incorrect, test)

        assert score_correct > score_incorrect, (
            "Correct ToM response should score higher than incorrect"
        )

    def test_creativity_scoring(self, benchmark):
        """Test creativity scoring with varied responses."""
        test = {
            "question": "List 5 unusual uses for a brick.",
            "answer": "creative",
            "expected_keywords": ["doorstop", "weight", "art"],
            "category": "creativity",
            "check_exact": False,
            "score_creativity": True,
            "difficulty": "medium"
        }

        # Creative response with multiple ideas
        creative = """
        1. Use it as a doorstop
        2. Use it as a weight for papers
        3. Create art by painting on it
        4. Use it as a garden border
        5. Heat it up as a bed warmer
        """
        score_creative = benchmark.score_response(creative, test)

        # Uncreative response
        uncreative = "I can't think of anything."
        score_uncreative = benchmark.score_response(uncreative, test)

        assert score_creative > score_uncreative, (
            "Creative response should score higher than uncreative"
        )

    def test_social_intelligence_scoring(self, benchmark):
        """Test social intelligence scoring."""
        test = {
            "question": "A friend says 'I'm fine' with tears. What do they feel?",
            "answer": "sad",
            "expected_keywords": ["sad", "upset", "hiding"],
            "category": "social_intelligence",
            "check_exact": False,
            "difficulty": "easy"
        }

        # Emotionally intelligent response
        empathetic = "They are probably sad and hiding their true feelings."
        score_empathetic = benchmark.score_response(empathetic, test)

        # Literal response
        literal = "They are fine, they said so."
        score_literal = benchmark.score_response(literal, test)

        assert score_empathetic > score_literal, (
            "Empathetic response should score higher than literal"
        )


class TestBenchmarkHistory:
    """Test benchmark history persistence."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_history_saves_to_file(self, temp_storage):
        """Test that benchmark history is saved to file."""
        benchmark = Benchmark(storage_path=temp_storage)

        # Run a simple benchmark
        def simple_ai(q):
            return "I don't know"

        benchmark.run_benchmark(simple_ai, "test_run")

        # Check history file exists
        history_path = os.path.join(temp_storage, "history.json")
        assert os.path.exists(history_path), "History file should be created"

        # Verify content
        with open(history_path, 'r') as f:
            history = json.load(f)

        assert len(history) > 0, "History should have at least one entry"
        assert history[-1]["name"] == "test_run"

    def test_history_persists_across_restarts(self, temp_storage):
        """Test that history persists when benchmark is recreated."""
        # Create benchmark and run
        benchmark1 = Benchmark(storage_path=temp_storage)

        def simple_ai(q):
            return "Test response"

        benchmark1.run_benchmark(simple_ai, "first_run")
        first_run_count = len(benchmark1.history)

        # Create new benchmark instance (simulating restart)
        benchmark2 = Benchmark(storage_path=temp_storage)

        # History should be loaded
        assert len(benchmark2.history) == first_run_count, (
            "History should persist across restarts"
        )

    def test_history_contains_test_results(self, temp_storage):
        """Test that history contains detailed test results."""
        benchmark = Benchmark(storage_path=temp_storage)

        def simple_ai(q):
            return "4"

        result = benchmark.run_benchmark(simple_ai, "detailed_run")

        # Check result structure
        assert "tests" in result, "Result should contain tests"
        assert "total_score" in result, "Result should contain total_score"
        assert "avg_score" in result, "Result should contain avg_score"
        assert "correct_count" in result, "Result should contain correct_count"
        assert "wrong_count" in result, "Result should contain wrong_count"

        # Check individual test results
        for test_result in result["tests"]:
            assert "question" in test_result
            assert "category" in test_result
            assert "score" in test_result
            assert "is_correct" in test_result


class TestCategoryAnalysis:
    """Test category-level analysis."""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        temp_dir = tempfile.mkdtemp()
        yield Benchmark(storage_path=temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_results_include_categories(self, benchmark):
        """Test that benchmark results include category for each test."""
        def simple_ai(q):
            return "test"

        result = benchmark.run_benchmark(simple_ai, "category_test")

        for test_result in result["tests"]:
            assert "category" in test_result, "Each test result should have category"
            assert test_result["category"], "Category should not be empty"

    def test_category_scores_calculable(self, benchmark):
        """Test that we can calculate scores per category from results."""
        def simple_ai(q):
            return "4"  # Will get some math questions right

        result = benchmark.run_benchmark(simple_ai, "category_calc")

        # Calculate per-category scores
        category_scores = {}
        category_counts = {}

        for test_result in result["tests"]:
            cat = test_result["category"]
            score = test_result["score"]

            if cat not in category_scores:
                category_scores[cat] = 0
                category_counts[cat] = 0

            category_scores[cat] += score
            category_counts[cat] += 1

        # Calculate averages
        category_averages = {
            cat: category_scores[cat] / category_counts[cat]
            for cat in category_scores
        }

        print(f"\nCategory averages: {category_averages}")

        # Should have multiple categories
        assert len(category_averages) >= 5, "Should have scores for multiple categories"


class TestSelfBenchmark:
    """Test the SelfBenchmark comparison functionality."""

    @pytest.fixture
    def self_benchmark(self):
        """Create SelfBenchmark instance."""
        temp_dir = tempfile.mkdtemp()
        yield SelfBenchmark(storage_path=temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_improvement_detection(self, self_benchmark):
        """Test that improvements are detected correctly."""
        # Bad AI
        def bad_ai(q):
            return "I don't know"

        # Better AI (answers some math)
        def better_ai(q):
            if "2 + 2" in q or "apples" in q:
                return "The answer is 10"
            return "I don't know"

        result = self_benchmark.test_improvement(bad_ai, better_ai, "better_responses")

        # Better AI should win
        assert result["modified_score"] >= result["current_score"], (
            "Better AI should score same or higher"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
