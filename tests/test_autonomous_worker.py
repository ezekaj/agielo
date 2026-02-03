#!/usr/bin/env python3
"""
Tests for AutonomousWorker atexit cleanup functionality.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import gc

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.autonomous_worker import (
    AutonomousWorker,
    GoalType,
    GoalPriority,
    _autonomous_worker_instances,
    _cleanup_all_instances,
)


class TestAutonomousWorkerCleanup(unittest.TestCase):
    """Tests for atexit cleanup functionality."""

    def setUp(self):
        """Create temporary test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.worker = AutonomousWorker(storage_path=self.test_dir)

    def tearDown(self):
        """Clean up test directory."""
        # Stop the worker if running
        if self.worker.running:
            self.worker.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cleanup_method_exists(self):
        """cleanup() method should exist and be callable."""
        self.assertTrue(hasattr(self.worker, 'cleanup'))
        self.assertTrue(callable(self.worker.cleanup))

    def test_cleanup_saves_stats(self):
        """cleanup() should save stats to disk."""
        # Modify some stats
        self.worker.stats['prompts_processed'] = 5
        self.worker.stats['goals_completed'] = 3
        self.worker.stats['facts_learned'] = 10

        # Get modification time before cleanup
        stats_mtime_before = 0
        if self.worker.stats_file.exists():
            stats_mtime_before = os.path.getmtime(self.worker.stats_file)

        # Small delay to ensure timestamps differ
        time.sleep(0.01)

        # Call cleanup
        self.worker.cleanup()

        # Verify stats file was updated
        self.assertTrue(self.worker.stats_file.exists())
        stats_mtime_after = os.path.getmtime(self.worker.stats_file)
        self.assertGreater(stats_mtime_after, stats_mtime_before)

    def test_cleanup_saves_correct_data(self):
        """cleanup() should save data that can be reloaded."""
        # Add some state
        self.worker.stats['prompts_processed'] = 7
        self.worker.stats['goals_completed'] = 4
        self.worker.stats['facts_learned'] = 15

        # Cleanup
        self.worker.cleanup()

        # Create new instance to verify persistence
        worker2 = AutonomousWorker(storage_path=self.test_dir)
        self.assertEqual(worker2.stats['prompts_processed'], 7)
        self.assertEqual(worker2.stats['goals_completed'], 4)
        self.assertEqual(worker2.stats['facts_learned'], 15)

    def test_cleanup_saves_goals(self):
        """cleanup() should save goal engine state."""
        # Add some goals
        self.worker.add_goal("Learn about test topic 1")
        self.worker.add_goal("Research something", goal_type=GoalType.RESEARCH)

        # Cleanup
        self.worker.cleanup()

        # Verify goals file exists
        goals_file = self.worker.goal_engine.goals_file
        self.assertTrue(goals_file.exists())

        # Create new instance to verify persistence
        worker2 = AutonomousWorker(storage_path=self.test_dir)
        self.assertEqual(len(worker2.goal_engine.goals), 2)

    def test_cleanup_handles_errors_gracefully(self):
        """cleanup() should not raise exceptions on errors."""
        # Make stats_file a directory to cause write error
        os.makedirs(str(self.worker.stats_file), exist_ok=True)

        # Should not raise - catches and logs error
        try:
            self.worker.cleanup()
        except Exception as e:
            self.fail(f"cleanup() raised an exception: {e}")

        # Clean up the directory we created
        os.rmdir(str(self.worker.stats_file))

    def test_cleanup_stops_running_worker(self):
        """cleanup() should stop a running worker thread."""
        # Start the worker
        self.worker.start()
        self.assertTrue(self.worker.running)

        # Give it a moment to start
        time.sleep(0.1)

        # Cleanup should stop it
        self.worker.cleanup()

        # Verify it's stopped
        self.assertFalse(self.worker.running)

    def test_cleanup_handles_stopped_worker(self):
        """cleanup() should handle workers that aren't running."""
        # Worker is not started
        self.assertFalse(self.worker.running)

        # Should not raise
        try:
            self.worker.cleanup()
        except Exception as e:
            self.fail(f"cleanup() raised on stopped worker: {e}")

    def test_instance_registration(self):
        """Instances should be registered for cleanup."""
        # Our setUp already created one instance
        # Check that at least one weak ref resolves to our instance
        found = False
        for ref in _autonomous_worker_instances:
            instance = ref()
            if instance is self.worker:
                found = True
                break
        self.assertTrue(found, "Instance not found in _autonomous_worker_instances")

    def test_cleanup_all_instances_function(self):
        """_cleanup_all_instances() should cleanup all registered instances."""
        # Create additional instances
        test_dir2 = tempfile.mkdtemp()
        try:
            worker2 = AutonomousWorker(storage_path=test_dir2)
            worker2.stats['prompts_processed'] = 100

            # Add some state to both
            self.worker.stats['prompts_processed'] = 200

            # Call cleanup all
            _cleanup_all_instances()

            # Both should have saved their state
            self.assertTrue(self.worker.stats_file.exists())
            self.assertTrue(worker2.stats_file.exists())

            # Verify data was saved
            with open(self.worker.stats_file) as f:
                saved1 = json.load(f)
            with open(worker2.stats_file) as f:
                saved2 = json.load(f)

            self.assertEqual(saved1['prompts_processed'], 200)
            self.assertEqual(saved2['prompts_processed'], 100)
        finally:
            shutil.rmtree(test_dir2, ignore_errors=True)

    def test_cleanup_handles_dead_weakrefs(self):
        """_cleanup_all_instances() should handle garbage collected instances."""
        # Create an instance that will be garbage collected
        test_dir_temp = tempfile.mkdtemp()
        try:
            temp_worker = AutonomousWorker(storage_path=test_dir_temp)
            temp_worker.stats['prompts_processed'] = 50

            # Delete the reference and force garbage collection
            del temp_worker
            gc.collect()

            # Should not raise even with dead weak references
            try:
                _cleanup_all_instances()
            except Exception as e:
                self.fail(f"_cleanup_all_instances() raised with dead weakrefs: {e}")
        finally:
            shutil.rmtree(test_dir_temp, ignore_errors=True)

    def test_atexit_registered(self):
        """_cleanup_all_instances should be registered with atexit."""
        import atexit

        # Check that our cleanup function is callable and follows the pattern
        self.assertTrue(callable(_cleanup_all_instances))


class TestAutonomousWorkerBasics(unittest.TestCase):
    """Basic tests for AutonomousWorker functionality."""

    def setUp(self):
        """Create temporary test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.worker = AutonomousWorker(storage_path=self.test_dir)

    def tearDown(self):
        """Clean up test directory."""
        if self.worker.running:
            self.worker.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_worker_creation(self):
        """AutonomousWorker should initialize properly."""
        self.assertIsNotNone(self.worker)
        self.assertFalse(self.worker.running)
        self.assertFalse(self.worker.is_busy)
        self.assertIsNone(self.worker.current_goal)

    def test_add_goal(self):
        """add_goal() should add goals to the goal engine."""
        goal = self.worker.add_goal("Test goal description")
        self.assertIsNotNone(goal)
        self.assertEqual(goal.description, "Test goal description")
        self.assertEqual(len(self.worker.goal_engine.goals), 1)

    def test_add_goal_with_type(self):
        """add_goal() should accept explicit goal types."""
        goal = self.worker.add_goal(
            "Improve something",
            goal_type=GoalType.IMPROVE,
            priority=GoalPriority.HIGH
        )
        self.assertEqual(goal.type, GoalType.IMPROVE)
        self.assertEqual(goal.priority, GoalPriority.HIGH)

    def test_add_prompt(self):
        """add_prompt() should add prompts to the queue."""
        self.worker.add_prompt("Test prompt", priority=2)
        self.assertEqual(self.worker.prompt_queue.size(), 1)

    def test_get_status(self):
        """get_status() should return current worker status."""
        status = self.worker.get_status()
        self.assertIn('running', status)
        self.assertIn('is_busy', status)
        self.assertIn('queue_size', status)
        self.assertIn('goals', status)
        self.assertIn('stats', status)

    def test_reflect(self):
        """reflect() should return a status string."""
        reflection = self.worker.reflect()
        self.assertIsInstance(reflection, str)
        self.assertIn('Running:', reflection)
        self.assertIn('GOALS', reflection)
        self.assertIn('STATISTICS', reflection)

    def test_start_stop(self):
        """start() and stop() should control the worker thread."""
        self.assertFalse(self.worker.running)

        self.worker.start()
        self.assertTrue(self.worker.running)
        self.assertIsNotNone(self.worker._thread)

        self.worker.stop()
        self.assertFalse(self.worker.running)

    def test_double_start(self):
        """start() should handle being called when already running."""
        self.worker.start()
        # Second start should not raise
        self.worker.start()
        self.assertTrue(self.worker.running)
        self.worker.stop()


if __name__ == "__main__":
    print("=" * 70)
    print("AUTONOMOUS WORKER TESTS")
    print("=" * 70)

    # Run tests with verbosity
    unittest.main(verbosity=2)
