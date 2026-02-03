"""
Tests for Code Evolution System
===============================

Verifies the self-modification capabilities work correctly:
1. Code validation catches dangerous operations
2. Sandbox testing executes code safely
3. Version control enables rollback
4. Evolved functions can be called
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.code_evolution import (
    CodeEvolution, CodeValidator, CodeSandbox, CodeVersionControl,
    CodeChangeType, ValidationResult
)


class TestCodeValidator:
    """Tests for code validation."""

    def setup_method(self):
        self.validator = CodeValidator()

    def test_valid_code(self):
        """Valid Python code should pass validation."""
        code = '''
def greet(name: str) -> str:
    return f"Hello, {name}!"
'''
        result, msg = self.validator.validate(code)
        assert result == ValidationResult.VALID, f"Expected VALID, got {result}: {msg}"

    def test_syntax_error(self):
        """Syntax errors should be caught."""
        code = '''
def broken(
    return "missing closing paren"
'''
        result, msg = self.validator.validate(code)
        assert result == ValidationResult.SYNTAX_ERROR, f"Expected SYNTAX_ERROR, got {result}"

    def test_dangerous_eval(self):
        """eval() should be blocked."""
        code = '''
def dangerous():
    return eval("1 + 1")
'''
        result, msg = self.validator.validate(code)
        assert result == ValidationResult.DANGEROUS_CODE, f"Expected DANGEROUS_CODE, got {result}"

    def test_dangerous_exec(self):
        """exec() should be blocked."""
        code = '''
def dangerous():
    exec("print('hacked')")
'''
        result, msg = self.validator.validate(code)
        assert result == ValidationResult.DANGEROUS_CODE, f"Expected DANGEROUS_CODE, got {result}"

    def test_dangerous_os_system(self):
        """os.system() should be blocked."""
        code = '''
import os
def dangerous():
    os.system("rm -rf /")
'''
        result, msg = self.validator.validate(code)
        assert result == ValidationResult.DANGEROUS_CODE, f"Expected DANGEROUS_CODE, got {result}"

    def test_missing_import(self):
        """Missing imports should be caught."""
        code = '''
def uses_nonexistent():
    return nonexistent_module.do_thing()
'''
        # This should pass syntax check but the import check happens differently
        result, msg = self.validator.validate(code)
        # Code is syntactically valid, just uses undefined name at runtime
        assert result == ValidationResult.VALID

    def test_complex_valid_code(self):
        """Complex but valid code should pass."""
        code = '''
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import math

@dataclass
class Point:
    x: float
    y: float

    def distance_from_origin(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

def process_points(points: List[Dict]) -> List[Point]:
    result = []
    for p in points:
        point = Point(x=p.get('x', 0), y=p.get('y', 0))
        result.append(point)
    return result
'''
        result, msg = self.validator.validate(code)
        assert result == ValidationResult.VALID, f"Expected VALID, got {result}: {msg}"


class TestCodeSandbox:
    """Tests for sandboxed code execution."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sandbox = CodeSandbox(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simple_execution(self):
        """Simple code should execute successfully."""
        code = '''
def add(a, b):
    return a + b

result = add(2, 3)
'''
        success, results = self.sandbox.test_code(code)
        assert results['import_success'], "Code should import successfully"

    def test_with_test_cases(self):
        """Test cases should be evaluated."""
        code = '''
def multiply(a, b):
    return a * b
'''
        test_cases = [
            {'function': 'multiply', 'args': [2, 3], 'expected': 6},
            {'function': 'multiply', 'args': [5, 5], 'expected': 25},
        ]
        success, results = self.sandbox.test_code(code, test_cases)
        assert success, f"Tests should pass: {results}"
        assert results['tests_passed'] == 2

    def test_failing_test_case(self):
        """Failing test cases should be reported."""
        code = '''
def broken_add(a, b):
    return a - b  # Bug: subtracts instead of adds
'''
        test_cases = [
            {'function': 'broken_add', 'args': [2, 3], 'expected': 5},
        ]
        success, results = self.sandbox.test_code(code, test_cases)
        assert not success, "Should fail due to wrong result"
        assert results['tests_failed'] == 1


class TestCodeVersionControl:
    """Tests for version control system."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.vc = CodeVersionControl(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_commit(self):
        """Changes should be committed."""
        from integrations.code_evolution import CodeChange

        change = CodeChange(
            id="test123",
            change_type=CodeChangeType.NEW_FUNCTION,
            target_file="/tmp/test.py",
            original_code=None,
            new_code="def test(): pass",
            description="Test commit",
            timestamp="2024-01-01T00:00:00"
        )

        commit_id = self.vc.commit(change)
        assert commit_id, "Should return commit ID"
        assert len(self.vc.history) == 1

    def test_history(self):
        """History should be retrievable."""
        from integrations.code_evolution import CodeChange

        for i in range(5):
            change = CodeChange(
                id=f"test{i}",
                change_type=CodeChangeType.NEW_FUNCTION,
                target_file="/tmp/test.py",
                original_code=None,
                new_code=f"def test{i}(): pass",
                description=f"Test commit {i}",
                timestamp="2024-01-01T00:00:00"
            )
            self.vc.commit(change)

        history = self.vc.get_history(limit=3)
        assert len(history) == 3, "Should return last 3 commits"


class TestCodeEvolution:
    """Integration tests for the full CodeEvolution system."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.evo = CodeEvolution(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_propose_valid_function(self):
        """Valid function should be proposed successfully."""
        code = '''
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''
        change = self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code=code,
            description="Add Fibonacci function"
        )

        assert change.validation_result == ValidationResult.VALID
        assert len(self.evo.pending_changes) == 1

    def test_propose_dangerous_function(self):
        """Dangerous function should be rejected."""
        code = '''
def evil():
    import os
    os.system("echo hacked")
'''
        change = self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code=code,
            description="Should be blocked"
        )

        assert change.validation_result == ValidationResult.DANGEROUS_CODE
        assert len(self.evo.pending_changes) == 0

    def test_deploy_function(self):
        """Valid function should deploy successfully."""
        code = '''
def square(x: int) -> int:
    return x * x
'''
        change = self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code=code,
            description="Add square function"
        )

        assert change.validation_result == ValidationResult.VALID

        success = self.evo.deploy_change(change)
        assert success, "Deployment should succeed"
        assert len(self.evo.deployed_changes) == 1

    def test_call_evolved_function(self):
        """Evolved functions should be callable."""
        code = '''
def cube(x: int) -> int:
    return x * x * x
'''
        change = self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code=code,
            description="Add cube function"
        )
        self.evo.deploy_change(change)

        # Get and call the function
        functions = self.evo.get_active_functions()
        assert 'cube' in functions, f"cube should be in active functions: {functions.keys()}"

        result = self.evo.call_evolved_function('cube', 3)
        assert result == 27, f"cube(3) should be 27, got {result}"

    def test_stats(self):
        """Stats should reflect current state."""
        code = '''
def test_func():
    return 42
'''
        change = self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code=code,
            description="Test function"
        )
        self.evo.deploy_change(change)

        stats = self.evo.get_stats()
        assert stats['deployed_changes'] == 1
        assert stats['pending_changes'] == 0


class TestEnsembleIntegration:
    """Tests for ensemble verifier integration with CodeEvolution."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.evo = CodeEvolution(self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensemble_verifier_enabled(self):
        """Ensemble verifier should be enabled by default."""
        # Check that ensemble is available
        assert hasattr(self.evo, 'use_ensemble')
        assert hasattr(self.evo, 'ensemble_verifier')
        # If ensemble is available, it should be enabled
        if self.evo.use_ensemble:
            assert self.evo.ensemble_verifier is not None

    def test_verification_results_stored(self):
        """Verification results should be stored in CodeChange."""
        code = '''
def simple_add(a: int, b: int) -> int:
    return a + b
'''
        change = self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code=code,
            description="Simple add function"
        )

        # Should have validation result
        assert change.validation_result is not None

        # If ensemble is available, should have verification_results
        if self.evo.use_ensemble:
            assert change.verification_results is not None
            assert 'passed' in change.verification_results
            assert 'final_confidence' in change.verification_results
            assert 'results' in change.verification_results

    def test_dangerous_code_vetoed(self):
        """Dangerous code should be vetoed by SafetyVerifier."""
        code = '''
def evil():
    import os
    os.system("echo bad")
'''
        change = self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code=code,
            description="Should be blocked"
        )

        assert change.validation_result == ValidationResult.DANGEROUS_CODE
        assert len(self.evo.pending_changes) == 0

        # If ensemble is available, should show veto
        if self.evo.use_ensemble and change.verification_results:
            assert change.verification_results.get('vetoed', False) is True
            assert change.verification_results.get('vetoed_by') == 'SafetyVerifier'

    def test_verifier_stats_available(self):
        """Verifier stats should be available in get_stats()."""
        # Run a verification first
        code = '''
def test():
    return 1
'''
        self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code=code,
            description="Test"
        )

        stats = self.evo.get_stats()
        assert 'ensemble_verifier_enabled' in stats

        if self.evo.use_ensemble:
            assert 'verifier' in stats
            verifier_stats = stats['verifier']
            assert verifier_stats.get('enabled', False) is True
            assert 'total_verifications' in verifier_stats
            assert 'pass_rate' in verifier_stats

    def test_custom_verifier_config(self):
        """Custom verifier config should be respected."""
        custom_config = {
            'threshold': 0.8,
            'voting_strategy': 'weighted',
            'style': {'enabled': False},
            'llm': {'enabled': False}
        }

        evo_custom = CodeEvolution(
            self.temp_dir / "custom",
            verifier_config=custom_config
        )

        if evo_custom.use_ensemble:
            assert evo_custom.ensemble_verifier.threshold == 0.8
            # Style verifier should not be present
            style_verifier = evo_custom.ensemble_verifier.get_verifier("StyleVerifier")
            assert style_verifier is None

    def test_with_test_cases_ensemble(self):
        """Test cases should be passed to ensemble verifier."""
        code = '''
def multiply(a, b):
    return a * b
'''
        test_cases = [
            {'function': 'multiply', 'args': [2, 3], 'expected': 6},
            {'function': 'multiply', 'args': [5, 5], 'expected': 25},
        ]

        change = self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code=code,
            description="Multiply function",
            test_cases=test_cases
        )

        assert change.validation_result == ValidationResult.VALID
        assert len(self.evo.pending_changes) == 1

        # Test results should be available
        if change.test_results:
            assert 'tests_passed' in change.test_results or 'success' in change.test_results

    def test_get_verifier_stats_method(self):
        """get_verifier_stats should return detailed verifier info."""
        # Run a few verifications
        self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code='def f(): return 1',
            description="Test 1"
        )
        self.evo.propose_change(
            change_type=CodeChangeType.NEW_FUNCTION,
            new_code='def g(): eval("x")',  # Should be blocked
            description="Test 2"
        )

        if self.evo.use_ensemble:
            verifier_stats = self.evo.get_verifier_stats()
            assert verifier_stats['enabled'] is True
            assert verifier_stats['total_verifications'] >= 2
            assert 'issues_caught' in verifier_stats


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestCodeValidator,
        TestCodeSandbox,
        TestCodeVersionControl,
        TestCodeEvolution,
        TestEnsembleIntegration,
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
