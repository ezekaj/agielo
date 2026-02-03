"""
Tests for Ensemble Verifier System
==================================

Verifies the multi-verifier ensemble works correctly:
1. Individual verifiers function properly
2. Voting strategies combine results correctly
3. Veto power works for safety verifier
4. Weighted confidence calculations are accurate
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.ensemble_verifier import (
    EnsembleVerifier, EnsembleResult, VerificationResult, VotingStrategy,
    Verifier, SyntaxVerifier, SafetyVerifier, TypeVerifier,
    TestVerifier, LLMVerifier, StyleVerifier,
    create_ensemble_verifier
)


class TestSyntaxVerifier:
    """Tests for SyntaxVerifier."""

    def setup_method(self):
        self.verifier = SyntaxVerifier()

    def test_valid_syntax(self):
        """Valid Python code should pass."""
        code = '''
def greet(name: str) -> str:
    return f"Hello, {name}!"
'''
        result = self.verifier.verify(code)
        assert result.passed, f"Should pass: {result.message}"
        assert result.confidence >= 0.9

    def test_syntax_error(self):
        """Syntax errors should be caught."""
        code = '''
def broken(
    return "missing paren"
'''
        result = self.verifier.verify(code)
        assert not result.passed
        assert result.confidence == 0.0
        assert "Syntax error" in result.message

    def test_complex_valid_code(self):
        """Complex but valid code should pass."""
        code = '''
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

def process(items: List[Dict]) -> List[Point]:
    return [Point(x=i.get('x', 0), y=i.get('y', 0)) for i in items]
'''
        result = self.verifier.verify(code)
        assert result.passed
        assert result.confidence >= 0.9

    def test_deeply_nested_code(self):
        """Deeply nested code should get lower confidence."""
        code = '''
def deeply_nested():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            return "too deep"
'''
        result = self.verifier.verify(code)
        assert result.passed  # Still valid syntax
        assert result.confidence < 1.0  # But flagged as quality issue


class TestSafetyVerifier:
    """Tests for SafetyVerifier."""

    def setup_method(self):
        self.verifier = SafetyVerifier()

    def test_safe_code(self):
        """Safe code should pass."""
        code = '''
def safe_function(x: int) -> int:
    return x * 2
'''
        result = self.verifier.verify(code)
        assert result.passed
        assert result.confidence == 1.0
        assert result.has_veto

    def test_eval_blocked(self):
        """eval() should be blocked."""
        code = '''
def dangerous():
    return eval("1 + 1")
'''
        result = self.verifier.verify(code)
        assert not result.passed
        assert "eval" in result.message.lower() or "dangerous" in result.message.lower()

    def test_exec_blocked(self):
        """exec() should be blocked."""
        code = '''
def dangerous():
    exec("print('bad')")
'''
        result = self.verifier.verify(code)
        assert not result.passed

    def test_os_system_blocked(self):
        """os.system() should be blocked."""
        code = '''
import os
def dangerous():
    os.system("rm -rf /")
'''
        result = self.verifier.verify(code)
        assert not result.passed

    def test_subprocess_blocked(self):
        """subprocess calls should be blocked."""
        code = '''
import subprocess
def dangerous():
    subprocess.run(["echo", "bad"])
'''
        result = self.verifier.verify(code)
        assert not result.passed

    def test_suspicious_pattern_detected(self):
        """Suspicious patterns should be detected."""
        code = '''
def suspicious():
    x = __builtins__
    return x
'''
        result = self.verifier.verify(code)
        assert not result.passed
        assert 'suspicious_patterns' in result.details or 'dangerous' in result.message.lower()

    def test_docker_mode_relaxed(self):
        """Docker mode should allow more operations."""
        docker_verifier = SafetyVerifier(docker_mode=True)

        # subprocess should be allowed in Docker mode
        code = '''
import subprocess
def run_command():
    subprocess.run(["echo", "hello"])
'''
        result = docker_verifier.verify(code)
        # In docker mode, subprocess is allowed (but eval/exec still blocked)
        assert result.passed

    def test_docker_mode_still_blocks_eval(self):
        """Docker mode should still block eval/exec."""
        docker_verifier = SafetyVerifier(docker_mode=True)

        code = '''
def dangerous():
    return eval("2+2")
'''
        result = docker_verifier.verify(code)
        assert not result.passed


class TestTypeVerifier:
    """Tests for TypeVerifier."""

    def setup_method(self):
        self.verifier = TypeVerifier()

    def test_mypy_availability_check(self):
        """Should check for mypy availability."""
        # Just verify the check runs without error
        result = self.verifier.verify("x = 1")
        assert result.verifier_name == "TypeVerifier"
        # If mypy not available, should still pass with 0.5 confidence
        if not self.verifier.mypy_available:
            assert result.passed
            assert result.confidence == 0.5

    def test_valid_typed_code(self):
        """Well-typed code should pass if mypy is available."""
        code = '''
def add(a: int, b: int) -> int:
    return a + b
'''
        result = self.verifier.verify(code)
        assert result.passed


class TestTestVerifier:
    """Tests for TestVerifier."""

    def setup_method(self):
        self.verifier = TestVerifier()

    def test_no_test_cases(self):
        """No test cases should return neutral confidence."""
        code = '''
def add(a, b):
    return a + b
'''
        result = self.verifier.verify(code)
        assert result.passed
        assert result.confidence == 0.5

    def test_passing_tests(self):
        """Passing tests should return high confidence."""
        code = '''
def multiply(a, b):
    return a * b
'''
        test_cases = [
            {'function': 'multiply', 'args': [2, 3], 'expected': 6},
            {'function': 'multiply', 'args': [5, 5], 'expected': 25},
        ]
        result = self.verifier.verify(code, test_cases=test_cases)
        assert result.passed
        assert result.confidence == 1.0
        assert result.details['tests_passed'] == 2

    def test_failing_tests(self):
        """Failing tests should return low confidence."""
        code = '''
def broken(a, b):
    return a - b  # Bug: should be +
'''
        test_cases = [
            {'function': 'broken', 'args': [2, 3], 'expected': 5},
        ]
        result = self.verifier.verify(code, test_cases=test_cases)
        assert not result.passed
        assert result.details['tests_failed'] == 1

    def test_mixed_results(self):
        """Mixed results should calculate correct confidence."""
        code = '''
def add(a, b):
    if a == 0:
        return b - 1  # Bug for a=0
    return a + b
'''
        test_cases = [
            {'function': 'add', 'args': [2, 3], 'expected': 5},  # Pass
            {'function': 'add', 'args': [0, 5], 'expected': 5},  # Fail
        ]
        result = self.verifier.verify(code, test_cases=test_cases)
        assert result.confidence == 0.5  # 1 out of 2


class TestLLMVerifier:
    """Tests for LLMVerifier."""

    def setup_method(self):
        self.verifier = LLMVerifier()

    def test_no_llm_interface(self):
        """Without LLM, should return neutral result."""
        code = '''
def add(a, b):
    return a + b
'''
        result = self.verifier.verify(code)
        assert result.passed
        assert result.confidence == 0.5
        assert "not available" in result.message.lower()

    def test_with_mock_llm(self):
        """With mock LLM, should parse response."""
        def mock_llm(prompt):
            return '{"safe": true, "correct": true, "quality": true, "confidence": 0.9, "issues": [], "explanation": "Good code"}'

        verifier = LLMVerifier(llm_interface=mock_llm)
        code = '''
def add(a, b):
    return a + b
'''
        result = verifier.verify(code)
        assert result.passed
        assert result.confidence == 0.9

    def test_llm_detects_issues(self):
        """LLM should detect issues in code."""
        def mock_llm(prompt):
            return '{"safe": false, "correct": true, "quality": true, "confidence": 0.3, "issues": ["Potential security issue"], "explanation": "Unsafe"}'

        verifier = LLMVerifier(llm_interface=mock_llm)
        code = '''
def add(a, b):
    return a + b
'''
        result = verifier.verify(code)
        assert not result.passed
        assert result.confidence == 0.3


class TestStyleVerifier:
    """Tests for StyleVerifier."""

    def setup_method(self):
        self.verifier = StyleVerifier()

    def test_good_style(self):
        """Well-styled code should pass."""
        code = '''
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class MyClass:
    """A simple class."""
    pass
'''
        result = self.verifier.verify(code)
        assert result.passed
        assert result.confidence >= 0.8

    def test_long_lines(self):
        """Long lines should be flagged."""
        long_line = "x = " + "a + " * 50 + "a"
        code = f'''
def test():
    {long_line}
'''
        result = self.verifier.verify(code)
        # Should have warnings/issues about line length
        assert result.details.get('issues') or result.details.get('warnings')

    def test_naming_conventions(self):
        """Bad naming should be flagged."""
        code = '''
def BadFunction():
    pass

class lowercase_class:
    pass
'''
        result = self.verifier.verify(code)
        # Should detect naming issues
        assert result.details.get('issues')


class TestEnsembleVerifier:
    """Tests for EnsembleVerifier with multiple verifiers."""

    def setup_method(self):
        self.ensemble = EnsembleVerifier()

    def test_all_pass(self):
        """When all verifiers pass, ensemble should pass."""
        code = '''
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''
        result = self.ensemble.verify(code)
        assert result.passed
        assert result.final_confidence > 0.5

    def test_safety_veto(self):
        """SafetyVerifier should veto dangerous code."""
        code = '''
import os
def danger():
    os.system("bad command")
'''
        result = self.ensemble.verify(code)
        assert not result.passed
        assert result.vetoed
        assert result.vetoed_by == "SafetyVerifier"

    def test_syntax_failure(self):
        """Syntax errors should fail the ensemble."""
        code = '''
def broken(
    return oops
'''
        result = self.ensemble.verify(code)
        assert not result.passed

    def test_weighted_voting(self):
        """Weighted voting should calculate correctly."""
        ensemble = EnsembleVerifier(
            voting_strategy=VotingStrategy.WEIGHTED,
            threshold=0.7
        )
        code = '''
def simple():
    return 42
'''
        result = ensemble.verify(code)
        # Should use weighted average
        assert result.voting_strategy == VotingStrategy.WEIGHTED

    def test_unanimous_voting(self):
        """Unanimous voting requires all to pass."""
        ensemble = EnsembleVerifier(
            voting_strategy=VotingStrategy.UNANIMOUS
        )
        code = '''
def simple():
    return 42
'''
        result = ensemble.verify(code)
        assert result.voting_strategy == VotingStrategy.UNANIMOUS

    def test_majority_voting(self):
        """Majority voting requires >50% to pass."""
        ensemble = EnsembleVerifier(
            voting_strategy=VotingStrategy.MAJORITY
        )
        code = '''
def simple():
    return 42
'''
        result = ensemble.verify(code)
        assert result.voting_strategy == VotingStrategy.MAJORITY

    def test_with_test_cases(self):
        """Ensemble should work with test cases."""
        code = '''
def add(a, b):
    return a + b
'''
        test_cases = [
            {'function': 'add', 'args': [2, 3], 'expected': 5},
        ]
        result = self.ensemble.verify(code, test_cases=test_cases)
        assert result.passed

    def test_result_to_dict(self):
        """EnsembleResult should serialize to dict."""
        code = '''
def test():
    return 1
'''
        result = self.ensemble.verify(code)
        result_dict = result.to_dict()

        assert 'passed' in result_dict
        assert 'final_confidence' in result_dict
        assert 'voting_strategy' in result_dict
        assert 'results' in result_dict
        assert isinstance(result_dict['results'], list)

    def test_stats_collection(self):
        """Ensemble should collect stats."""
        code = '''
def test():
    return 1
'''
        # Run a few verifications
        self.ensemble.verify(code)
        self.ensemble.verify(code)

        stats = self.ensemble.get_stats()
        assert stats['total_verifications'] == 2
        assert 'pass_rate' in stats
        assert 'verifier_stats' in stats

    def test_add_remove_verifier(self):
        """Should be able to add and remove verifiers."""
        initial_count = len(self.ensemble.verifiers)

        # Add a new verifier
        new_verifier = StyleVerifier(weight=0.3)
        new_verifier.name = "CustomStyleVerifier"
        self.ensemble.add_verifier(new_verifier)
        assert len(self.ensemble.verifiers) == initial_count + 1

        # Remove it
        removed = self.ensemble.remove_verifier("CustomStyleVerifier")
        assert removed
        assert len(self.ensemble.verifiers) == initial_count

        # Try removing non-existent
        removed = self.ensemble.remove_verifier("NonExistent")
        assert not removed

    def test_get_verifier(self):
        """Should be able to get verifier by name."""
        syntax = self.ensemble.get_verifier("SyntaxVerifier")
        assert syntax is not None
        assert isinstance(syntax, SyntaxVerifier)

        none = self.ensemble.get_verifier("NonExistent")
        assert none is None


class TestEnsembleVerifierFactory:
    """Tests for create_ensemble_verifier factory function."""

    def test_default_creation(self):
        """Should create with default config."""
        ensemble = create_ensemble_verifier()
        assert len(ensemble.verifiers) >= 4  # At least syntax, safety, test, style

    def test_custom_threshold(self):
        """Should respect custom threshold."""
        config = {'threshold': 0.9}
        ensemble = create_ensemble_verifier(config=config)
        assert ensemble.threshold == 0.9

    def test_custom_voting_strategy(self):
        """Should respect custom voting strategy."""
        config = {'voting_strategy': 'unanimous'}
        ensemble = create_ensemble_verifier(config=config)
        assert ensemble.voting_strategy == VotingStrategy.UNANIMOUS

    def test_disable_verifier(self):
        """Should be able to disable verifiers."""
        config = {
            'style': {'enabled': False},
            'type': {'enabled': False},
            'llm': {'enabled': False}
        }
        ensemble = create_ensemble_verifier(config=config)

        # Should not have style, type, or llm verifiers
        names = [v.name for v in ensemble.verifiers]
        assert "StyleVerifier" not in names
        assert "TypeVerifier" not in names
        assert "LLMVerifier" not in names

    def test_custom_weights(self):
        """Should respect custom weights."""
        config = {
            'syntax': {'weight': 0.5},
            'safety': {'weight': 2.0}
        }
        ensemble = create_ensemble_verifier(config=config)

        syntax = ensemble.get_verifier("SyntaxVerifier")
        safety = ensemble.get_verifier("SafetyVerifier")

        assert syntax.weight == 0.5
        assert safety.weight == 2.0


class TestVerifierEffectiveness:
    """Tests for verifier effectiveness tracking."""

    def setup_method(self):
        self.ensemble = EnsembleVerifier()

    def test_effectiveness_calculation(self):
        """Should calculate verifier effectiveness."""
        # Run some verifications
        self.ensemble.verify('def test(): return 1')
        self.ensemble.verify('def test(): eval("x")')  # Safety will fail

        effectiveness = self.ensemble.get_verifier_effectiveness()
        assert "SafetyVerifier" in effectiveness
        assert effectiveness["SafetyVerifier"]["total_failures"] >= 1


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestSyntaxVerifier,
        TestSafetyVerifier,
        TestTypeVerifier,
        TestTestVerifier,
        TestLLMVerifier,
        TestStyleVerifier,
        TestEnsembleVerifier,
        TestEnsembleVerifierFactory,
        TestVerifierEffectiveness,
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
                    print(f"  PASS {method_name}")
                    total_passed += 1

                except AssertionError as e:
                    print(f"  FAIL {method_name}: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))

                except Exception as e:
                    print(f"  ERROR {method_name}: {type(e).__name__}: {e}")
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
            print(f"    {error[:300]}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
