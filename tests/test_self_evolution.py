#!/usr/bin/env python3
"""
Self-Evolution Error Handling Tests
====================================

Tests for robust error handling in the self-evolution module.
Covers:
- Corrupted JSONL handling in _count_training_pairs()
- Empty/whitespace content validation in mark_learned()
- reset_cycle() recovery functionality
- Exception safety in should_train()
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.self_evolution import SelfEvolution


class TestCountTrainingPairs(unittest.TestCase):
    """Tests for _count_training_pairs() error handling."""

    def setUp(self):
        """Create temporary test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.evo = SelfEvolution(storage_path=self.test_dir)
        self.training_file = os.path.expanduser("~/.cognitive_ai_knowledge/training_data_test.jsonl")
        self.evo.training_data_file = self.training_file

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        if os.path.exists(self.training_file):
            os.remove(self.training_file)

    def test_missing_file_returns_zero(self):
        """Should return 0 when training file doesn't exist."""
        self.evo.training_data_file = "/nonexistent/path/training.jsonl"
        count = self.evo._count_training_pairs()
        self.assertEqual(count, 0)

    def test_empty_file_returns_zero(self):
        """Should return 0 for empty file."""
        with open(self.training_file, 'w') as f:
            pass
        count = self.evo._count_training_pairs()
        self.assertEqual(count, 0)

    def test_valid_jsonl_counted_correctly(self):
        """Should count valid JSONL lines correctly."""
        with open(self.training_file, 'w') as f:
            f.write('{"prompt": "test1", "completion": "answer1"}\n')
            f.write('{"prompt": "test2", "completion": "answer2"}\n')
            f.write('{"prompt": "test3", "completion": "answer3"}\n')
        count = self.evo._count_training_pairs()
        self.assertEqual(count, 3)

    def test_corrupted_lines_skipped(self):
        """Should skip corrupted JSON lines and count only valid ones."""
        with open(self.training_file, 'w') as f:
            f.write('{"prompt": "valid1", "completion": "answer1"}\n')
            f.write('this is not json at all\n')
            f.write('{"broken json\n')
            f.write('{"prompt": "valid2", "completion": "answer2"}\n')
            f.write('{"missing": "fields"}\n')  # Missing prompt/completion
        count = self.evo._count_training_pairs()
        self.assertEqual(count, 2)  # Only 2 valid entries

    def test_empty_prompt_not_counted(self):
        """Should not count entries with empty prompt."""
        with open(self.training_file, 'w') as f:
            f.write('{"prompt": "", "completion": "answer1"}\n')
            f.write('{"prompt": "   ", "completion": "answer2"}\n')
            f.write('{"prompt": "valid", "completion": "answer3"}\n')
        count = self.evo._count_training_pairs()
        self.assertEqual(count, 1)

    def test_empty_completion_not_counted(self):
        """Should not count entries with empty completion."""
        with open(self.training_file, 'w') as f:
            f.write('{"prompt": "test1", "completion": ""}\n')
            f.write('{"prompt": "test2", "completion": "   "}\n')
            f.write('{"prompt": "test3", "completion": "valid"}\n')
        count = self.evo._count_training_pairs()
        self.assertEqual(count, 1)

    def test_blank_lines_ignored(self):
        """Should ignore blank lines in JSONL file."""
        with open(self.training_file, 'w') as f:
            f.write('{"prompt": "test1", "completion": "answer1"}\n')
            f.write('\n')
            f.write('   \n')
            f.write('{"prompt": "test2", "completion": "answer2"}\n')
        count = self.evo._count_training_pairs()
        self.assertEqual(count, 2)


class TestMarkLearned(unittest.TestCase):
    """Tests for mark_learned() validation."""

    def setUp(self):
        """Create temporary test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.evo = SelfEvolution(storage_path=self.test_dir)

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_valid_content_learned(self):
        """Should successfully mark valid content as learned."""
        result = self.evo.mark_learned("This is valid content")
        self.assertTrue(result)
        self.assertEqual(self.evo.state['facts_this_cycle'], 1)

    def test_empty_string_rejected(self):
        """Should reject empty string."""
        result = self.evo.mark_learned("")
        self.assertFalse(result)
        self.assertEqual(self.evo.state['facts_this_cycle'], 0)

    def test_whitespace_only_rejected(self):
        """Should reject whitespace-only content."""
        result = self.evo.mark_learned("   ")
        self.assertFalse(result)
        result = self.evo.mark_learned("\t\n")
        self.assertFalse(result)
        self.assertEqual(self.evo.state['facts_this_cycle'], 0)

    def test_none_rejected(self):
        """Should handle None gracefully."""
        result = self.evo.mark_learned(None)
        self.assertFalse(result)

    def test_duplicate_rejected(self):
        """Should reject duplicate content."""
        self.evo.mark_learned("unique content")
        result = self.evo.mark_learned("unique content")
        self.assertFalse(result)
        self.assertEqual(self.evo.state['facts_this_cycle'], 1)


class TestResetCycle(unittest.TestCase):
    """Tests for reset_cycle() recovery functionality."""

    def setUp(self):
        """Create temporary test directory with some state."""
        self.test_dir = tempfile.mkdtemp()
        self.evo = SelfEvolution(storage_path=self.test_dir)
        # Set up some state
        self.evo.mark_learned("fact 1")
        self.evo.mark_learned("fact 2")
        self.evo.state['current_cycle'] = 5
        self.evo.state['facts_this_cycle'] = 2
        self.evo.state['baseline_score'] = 0.5
        self.evo.state['current_score'] = 0.6
        self.evo.state['total_trainings'] = 2
        self.evo._save_state()

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_reset_preserves_learned_by_default(self):
        """Should preserve learned facts by default."""
        initial_facts = len(self.evo.learned_hashes)
        result = self.evo.reset_cycle()

        self.assertTrue(result['success'])
        self.assertTrue(result['preserve_learned'])
        self.assertEqual(len(self.evo.learned_hashes), initial_facts)
        self.assertEqual(self.evo.state['current_cycle'], 0)
        self.assertEqual(self.evo.state['facts_this_cycle'], 0)
        self.assertIsNone(self.evo.state['baseline_score'])
        self.assertIsNone(self.evo.state['current_score'])

    def test_reset_full_clears_everything(self):
        """Should clear all state when preserve_learned=False."""
        result = self.evo.reset_cycle(preserve_learned=False)

        self.assertTrue(result['success'])
        self.assertFalse(result['preserve_learned'])
        self.assertEqual(len(self.evo.learned_hashes), 0)
        self.assertEqual(self.evo.state['total_trainings'], 0)
        self.assertEqual(len(self.evo.state['improvements']), 0)

    def test_reset_returns_previous_state(self):
        """Should return previous state in result."""
        result = self.evo.reset_cycle()

        self.assertEqual(result['previous_state']['cycle'], 5)
        self.assertEqual(result['previous_state']['facts_this_cycle'], 2)
        self.assertEqual(result['previous_state']['total_facts'], 2)
        self.assertEqual(result['previous_state']['baseline_score'], 0.5)
        self.assertEqual(result['previous_state']['current_score'], 0.6)


class TestShouldTrain(unittest.TestCase):
    """Tests for should_train() exception safety."""

    def setUp(self):
        """Create temporary test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.evo = SelfEvolution(storage_path=self.test_dir)

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_no_training_data_returns_false(self):
        """Should return False when no training data exists."""
        should, reason = self.evo.should_train()
        self.assertFalse(should)
        # Either "Not enough training data" or "improvement" reason is valid
        self.assertTrue(
            "Not enough training data" in reason or "improvement" in reason,
            f"Unexpected reason: {reason}"
        )

    def test_corrupted_state_returns_false_safely(self):
        """Should return False without throwing on corrupted state."""
        # Corrupt the state
        self.evo.state = None
        should, reason = self.evo.should_train()
        self.assertFalse(should)
        self.assertIn("Error", reason)

    def test_missing_state_keys_handled(self):
        """Should handle missing state keys gracefully."""
        self.evo.state = {}  # Empty state
        should, reason = self.evo.should_train()
        self.assertFalse(should)
        # Should not raise KeyError


class TestPersistenceErrorHandling(unittest.TestCase):
    """Tests for file I/O error handling."""

    def setUp(self):
        """Create temporary test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.evo = SelfEvolution(storage_path=self.test_dir)

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_corrupted_hashes_file(self):
        """Should handle corrupted hashes file gracefully."""
        # Write corrupted JSON to hashes file
        with open(self.evo.learned_hashes_file, 'w') as f:
            f.write("not valid json at all")

        # Create new instance to trigger load
        evo2 = SelfEvolution(storage_path=self.test_dir)
        # Should have empty hashes, not crash
        self.assertEqual(len(evo2.learned_hashes), 0)

    def test_load_corrupted_state_file(self):
        """Should handle corrupted state file gracefully."""
        # Write corrupted JSON to state file
        with open(self.evo.state_file, 'w') as f:
            f.write("{broken json")

        # Create new instance to trigger load
        evo2 = SelfEvolution(storage_path=self.test_dir)
        # Should use defaults, not crash
        self.assertEqual(evo2.state['current_cycle'], 0)

    def test_load_corrupted_benchmark_file(self):
        """Should handle corrupted benchmark file gracefully."""
        # Write corrupted JSON to benchmark file
        with open(self.evo.benchmark_file, 'w') as f:
            f.write("invalid")

        # Create new instance to trigger load
        evo2 = SelfEvolution(storage_path=self.test_dir)
        # Should have empty history, not crash
        self.assertEqual(len(evo2.benchmark_history), 0)


if __name__ == "__main__":
    print("=" * 70)
    print("SELF-EVOLUTION ERROR HANDLING TESTS")
    print("=" * 70)

    # Run tests with verbosity
    unittest.main(verbosity=2)
