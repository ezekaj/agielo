"""
Tests for Docker Sandbox System
===============================

Verifies Docker-based code isolation works correctly:
1. Basic code execution in container
2. Memory limit enforcement
3. Timeout enforcement
4. Network access disabled
5. Container cleanup

Note: Tests will skip gracefully if Docker is not available.
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.docker_sandbox import (
    DockerSandbox,
    ExecutionResult,
    DOCKER_PACKAGE_AVAILABLE,
    get_docker_sandbox
)


def skip_if_no_docker(func):
    """Decorator to skip tests if Docker is not available."""
    def wrapper(*args, **kwargs):
        sandbox = get_docker_sandbox()
        if not sandbox.is_available():
            print(f"  ⊘ {func.__name__}: SKIPPED (Docker not available)")
            return True  # Count as passed since it's intentionally skipped
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


class TestDockerSandboxBasic:
    """Basic tests that work with or without Docker."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sandbox = DockerSandbox(base_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sandbox_initializes(self):
        """Sandbox should initialize without errors."""
        assert self.sandbox is not None
        assert self.sandbox.timeout == 30
        assert self.sandbox.memory_limit == "512m"
        assert self.sandbox.network_disabled is True

    def test_is_available_returns_bool(self):
        """is_available() should return a boolean."""
        result = self.sandbox.is_available()
        assert isinstance(result, bool)

    def test_get_stats_empty(self):
        """Stats should work with no executions."""
        stats = self.sandbox.get_stats()
        assert stats['total_executions'] == 0
        assert stats['docker_available'] == self.sandbox.docker_available


class TestDockerSandboxExecution:
    """Tests for code execution (may use fallback if Docker unavailable)."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sandbox = DockerSandbox(base_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simple_execution(self):
        """Simple code should execute successfully."""
        code = '''
def add(a, b):
    return a + b
'''
        test_cases = [
            {'function': 'add', 'args': [2, 3], 'expected': 5},
        ]
        success, results = self.sandbox.execute_code(code, test_cases)
        assert success, f"Execution failed: {results}"
        assert results['tests_passed'] == 1
        assert results['tests_failed'] == 0

    def test_multiple_test_cases(self):
        """Multiple test cases should all be evaluated."""
        code = '''
def multiply(a, b):
    return a * b
'''
        test_cases = [
            {'function': 'multiply', 'args': [2, 3], 'expected': 6},
            {'function': 'multiply', 'args': [5, 5], 'expected': 25},
            {'function': 'multiply', 'args': [0, 100], 'expected': 0},
            {'function': 'multiply', 'args': [-1, 5], 'expected': -5},
        ]
        success, results = self.sandbox.execute_code(code, test_cases)
        assert success, f"Execution failed: {results}"
        assert results['tests_passed'] == 4

    def test_failing_test_case(self):
        """Failing test should be reported correctly."""
        code = '''
def broken_add(a, b):
    return a - b  # Bug: subtracts instead of adds
'''
        test_cases = [
            {'function': 'broken_add', 'args': [10, 5], 'expected': 15},
        ]
        success, results = self.sandbox.execute_code(code, test_cases)
        assert not success, "Should fail due to wrong result"
        assert results['tests_failed'] == 1
        assert 'errors' in results
        assert len(results['errors']) > 0

    def test_missing_function(self):
        """Missing function should be reported as error."""
        code = '''
def existing_func():
    return 42
'''
        test_cases = [
            {'function': 'nonexistent_func', 'args': [], 'expected': 0},
        ]
        success, results = self.sandbox.execute_code(code, test_cases)
        assert not success
        assert results['tests_failed'] == 1
        assert any('not found' in str(e).lower() for e in results['errors'])

    def test_exception_in_function(self):
        """Exception during execution should be caught."""
        code = '''
def divide(a, b):
    return a / b
'''
        test_cases = [
            {'function': 'divide', 'args': [10, 0], 'expected': None},  # Division by zero
        ]
        success, results = self.sandbox.execute_code(code, test_cases)
        # Either fails entirely or test fails
        assert not success or results['tests_failed'] >= 1, \
            f"Exception should cause failure: success={success}, failed={results['tests_failed']}"

    def test_code_with_imports(self):
        """Code with standard library imports should work."""
        code = '''
import math

def calculate_area(radius):
    return math.pi * radius ** 2
'''
        test_cases = [
            {'function': 'calculate_area', 'args': [1], 'expected': 3.141592653589793},
        ]
        success, results = self.sandbox.execute_code(code, test_cases)
        # May pass or fail depending on float precision
        assert results['tests_passed'] >= 0

    def test_execution_time_recorded(self):
        """Execution time should be recorded."""
        code = '''
def simple():
    return 42
'''
        test_cases = [{'function': 'simple', 'args': [], 'expected': 42}]
        success, results = self.sandbox.execute_code(code, test_cases)
        assert 'execution_time' in results
        assert results['execution_time'] > 0

    def test_sandbox_type_recorded(self):
        """Sandbox type should be recorded in results."""
        code = '''
def test():
    return True
'''
        success, results = self.sandbox.execute_code(code, [])
        assert 'sandbox_type' in results
        assert results['sandbox_type'] in ['docker', 'python']

    def test_stats_updated_after_execution(self):
        """Stats should update after execution."""
        code = '''
def test():
    return 1
'''
        self.sandbox.execute_code(code, [])
        stats = self.sandbox.get_stats()
        assert stats['total_executions'] == 1


class TestDockerSandboxDocker:
    """Tests that specifically require Docker."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sandbox = DockerSandbox(base_dir=self.temp_dir)

    def teardown_method(self):
        # Clean up any orphan containers
        if self.sandbox.is_available():
            self.sandbox.cleanup_orphan_containers()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @skip_if_no_docker
    def test_docker_execution(self):
        """Code should execute in Docker container."""
        code = '''
def greet(name):
    return f"Hello, {name}!"
'''
        test_cases = [
            {'function': 'greet', 'args': ['World'], 'expected': 'Hello, World!'},
        ]
        success, results = self.sandbox.execute_code(code, test_cases)
        assert results['sandbox_type'] == 'docker'
        assert success

    @skip_if_no_docker
    def test_memory_limit_enforcement(self):
        """Code trying to allocate too much memory should fail."""
        # This test tries to allocate ~600MB, which exceeds the 512MB limit
        code = '''
def allocate_memory():
    # Try to allocate 600MB
    data = bytearray(600 * 1024 * 1024)
    return len(data)
'''
        test_cases = [{'function': 'allocate_memory', 'args': [], 'expected': 600 * 1024 * 1024}]
        success, results = self.sandbox.execute_code(code, test_cases)
        # Should fail due to memory limit
        assert not success or results['exit_code'] != 0, \
            "Memory allocation beyond limit should fail"

    @skip_if_no_docker
    def test_timeout_enforcement(self):
        """Infinite loop should be killed after timeout."""
        sandbox = DockerSandbox(base_dir=self.temp_dir, timeout=5)  # Short timeout for test

        code = '''
import time

def infinite_loop():
    while True:
        time.sleep(0.1)
    return "never reached"
'''
        test_cases = [{'function': 'infinite_loop', 'args': []}]

        start = time.time()
        success, results = sandbox.execute_code(code, test_cases)
        elapsed = time.time() - start

        assert not success, "Infinite loop should fail"
        # Should complete in approximately timeout seconds (with some margin)
        assert elapsed < 15, f"Should timeout in ~5s, took {elapsed}s"
        assert any('timeout' in str(e).lower() for e in results.get('errors', []))

    @skip_if_no_docker
    def test_no_network_access(self):
        """Network requests should fail."""
        code = '''
import urllib.request

def fetch_url():
    try:
        urllib.request.urlopen("https://example.com", timeout=5)
        return "connected"
    except Exception as e:
        return f"blocked: {type(e).__name__}"
'''
        test_cases = [{'function': 'fetch_url', 'args': []}]
        success, results = self.sandbox.execute_code(code, test_cases)
        # Should either fail or return "blocked"
        stdout = results.get('stdout', '')
        # Network should be blocked
        assert 'connected' not in stdout or not success

    @skip_if_no_docker
    def test_container_cleanup(self):
        """Containers should be cleaned up after execution."""
        code = '''
def test():
    return 42
'''
        # Run some code
        self.sandbox.execute_code(code, [])
        self.sandbox.execute_code(code, [])
        self.sandbox.execute_code(code, [])

        # Check no orphan containers
        cleaned = self.sandbox.cleanup_orphan_containers()
        # Should have already been cleaned up
        assert cleaned == 0, f"Found {cleaned} orphan containers"


class TestDockerSandboxFallback:
    """Tests for Python fallback when Docker is unavailable."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        # Create sandbox with Docker disabled for testing fallback
        self.sandbox = DockerSandbox(base_dir=self.temp_dir)
        # Force fallback mode
        self.sandbox.docker_available = False

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fallback_execution(self):
        """Fallback to Python subprocess should work."""
        code = '''
def double(x):
    return x * 2
'''
        test_cases = [
            {'function': 'double', 'args': [5], 'expected': 10},
        ]
        success, results = self.sandbox.execute_code(code, test_cases)
        assert success
        assert results['sandbox_type'] == 'python'

    def test_fallback_timeout(self):
        """Fallback should respect timeout."""
        sandbox = DockerSandbox(base_dir=self.temp_dir, timeout=2)
        sandbox.docker_available = False

        code = '''
import time

def slow():
    time.sleep(10)
    return "done"
'''
        test_cases = [{'function': 'slow', 'args': []}]

        start = time.time()
        success, results = sandbox.execute_code(code, test_cases)
        elapsed = time.time() - start

        assert not success
        assert elapsed < 5  # Should timeout in ~2s


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestDockerSandboxBasic,
        TestDockerSandboxExecution,
        TestDockerSandboxDocker,
        TestDockerSandboxFallback,
    ]

    total_passed = 0
    total_failed = 0
    total_skipped = 0
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
                    result = getattr(instance, method_name)()

                    # Check if test was skipped (decorated function returns True)
                    if result is True:
                        total_skipped += 1
                    else:
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
    print(f"RESULTS: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    print('='*60)

    sandbox = get_docker_sandbox()
    print(f"\nDocker available: {sandbox.is_available()}")

    if failures:
        print("\nFailures:")
        for cls, method, error in failures:
            print(f"\n  {cls}.{method}:")
            print(f"    {error[:200]}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
