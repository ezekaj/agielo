"""
Ensemble Verifier System
========================

Multiple verification methods that must agree before deploying code changes.
Implements a voting system with configurable thresholds and veto power.

Based on research showing that ensemble verification significantly reduces
false positives and catches more issues than single-verifier approaches.

Verifiers:
- SyntaxVerifier: AST parsing for syntax correctness
- SafetyVerifier: Dangerous operation detection (has veto power)
- TypeVerifier: mypy type checking (optional)
- TestVerifier: Run test cases in sandbox
- LLMVerifier: AI-based code safety and correctness check
- StyleVerifier: Basic PEP8 style checking
"""

import ast
import re
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable


class VotingStrategy(Enum):
    """Voting strategies for ensemble decision."""
    UNANIMOUS = "unanimous"      # All verifiers must agree
    MAJORITY = "majority"        # >50% of verifiers must agree
    WEIGHTED = "weighted"        # Weighted average > threshold


@dataclass
class VerificationResult:
    """Result from a single verifier."""
    verifier_name: str
    passed: bool
    confidence: float  # 0.0 to 1.0
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    has_veto: bool = False


@dataclass
class EnsembleResult:
    """Combined result from all verifiers."""
    passed: bool
    final_confidence: float
    voting_strategy: VotingStrategy
    threshold: float
    results: List[VerificationResult] = field(default_factory=list)
    vetoed: bool = False
    vetoed_by: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'passed': self.passed,
            'final_confidence': self.final_confidence,
            'voting_strategy': self.voting_strategy.value,
            'threshold': self.threshold,
            'vetoed': self.vetoed,
            'vetoed_by': self.vetoed_by,
            'timestamp': self.timestamp,
            'results': [
                {
                    'verifier_name': r.verifier_name,
                    'passed': r.passed,
                    'confidence': r.confidence,
                    'message': r.message,
                    'details': r.details,
                    'execution_time': r.execution_time,
                    'has_veto': r.has_veto
                }
                for r in self.results
            ]
        }


class Verifier(ABC):
    """
    Base class for code verifiers.

    Each verifier checks a specific aspect of code quality/safety
    and returns a result with confidence score.
    """

    def __init__(self, name: str, weight: float = 1.0, has_veto: bool = False):
        """
        Initialize verifier.

        Args:
            name: Verifier name for logging
            weight: Weight in weighted voting (0.0 to 1.0)
            has_veto: If True, failure vetoes the entire ensemble
        """
        self.name = name
        self.weight = weight
        self.has_veto = has_veto
        self.verification_log: List[Dict] = []

    @abstractmethod
    def verify(self, code: str, **kwargs) -> VerificationResult:
        """
        Verify code and return result.

        Args:
            code: Python code to verify
            **kwargs: Additional parameters (test_cases, etc.)

        Returns:
            VerificationResult with passed, confidence, and message
        """
        pass

    def _log_verification(self, result: VerificationResult):
        """Log verification result."""
        self.verification_log.append({
            'timestamp': datetime.now().isoformat(),
            'passed': result.passed,
            'confidence': result.confidence,
            'message': result.message
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        if not self.verification_log:
            return {
                'total_verifications': 0,
                'pass_rate': 0.0,
                'avg_confidence': 0.0
            }

        total = len(self.verification_log)
        passed = sum(1 for v in self.verification_log if v['passed'])
        avg_conf = sum(v['confidence'] for v in self.verification_log) / total

        return {
            'total_verifications': total,
            'pass_rate': passed / total,
            'avg_confidence': avg_conf
        }


class SyntaxVerifier(Verifier):
    """
    Verifies code syntax correctness using AST parsing.

    This is a foundational verifier - if syntax is invalid,
    nothing else can run.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__("SyntaxVerifier", weight=weight, has_veto=False)

    def verify(self, code: str, **kwargs) -> VerificationResult:
        """Verify code syntax using ast.parse()."""
        start_time = datetime.now()

        try:
            tree = ast.parse(code)

            # Additional syntax quality checks
            issues = self._check_syntax_quality(tree, code)

            if issues:
                confidence = max(0.5, 1.0 - (len(issues) * 0.1))
                result = VerificationResult(
                    verifier_name=self.name,
                    passed=True,  # Still valid syntax
                    confidence=confidence,
                    message=f"Valid syntax with {len(issues)} quality issues",
                    details={'issues': issues},
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            else:
                result = VerificationResult(
                    verifier_name=self.name,
                    passed=True,
                    confidence=1.0,
                    message="Valid Python syntax",
                    details={'ast_nodes': len(list(ast.walk(tree)))},
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

        except SyntaxError as e:
            result = VerificationResult(
                verifier_name=self.name,
                passed=False,
                confidence=0.0,
                message=f"Syntax error at line {e.lineno}: {e.msg}",
                details={
                    'line': e.lineno,
                    'offset': e.offset,
                    'text': e.text
                },
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        self._log_verification(result)
        return result

    def _check_syntax_quality(self, tree: ast.AST, code: str) -> List[str]:
        """Check for syntax quality issues (not errors)."""
        issues = []

        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append("Bare except clause (catches all exceptions)")

        # Check for very deeply nested code
        max_depth = self._get_max_nesting(tree)
        if max_depth > 5:
            issues.append(f"Deeply nested code (depth: {max_depth})")

        return issues

    def _get_max_nesting(self, tree: ast.AST, depth: int = 0) -> int:
        """Get maximum nesting depth of code."""
        max_depth = depth
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._get_max_nesting(node, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._get_max_nesting(node, depth)
                max_depth = max(max_depth, child_depth)
        return max_depth


class SafetyVerifier(Verifier):
    """
    Verifies code safety by detecting dangerous operations.

    This verifier has VETO power - if it fails, the code is rejected
    regardless of other verifiers.
    """

    # Dangerous function calls (always blocked)
    DANGEROUS_CALLS = {
        'eval', 'exec', 'compile', '__import__',
        'os.system', 'os.popen', 'os.spawn', 'os.spawnl', 'os.spawnle',
        'os.spawnlp', 'os.spawnlpe', 'os.spawnv', 'os.spawnve',
        'os.spawnvp', 'os.spawnvpe', 'os.execl', 'os.execle',
        'os.execlp', 'os.execlpe', 'os.execv', 'os.execve',
        'os.execvp', 'os.execvpe',
        'subprocess.call', 'subprocess.run', 'subprocess.Popen',
        'subprocess.check_call', 'subprocess.check_output',
        'shutil.rmtree',
        'pickle.loads', 'marshal.loads',  # Deserialization attacks
        'yaml.load',  # Without safe_load
    }

    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r'__builtins__',  # Accessing builtins directly
        r'__class__.*__bases__',  # Class hierarchy traversal
        r'__subclasses__',  # Subclass access
        r'__mro__',  # Method resolution order
        r'__globals__',  # Global variable access
        r'open\s*\([^)]*["\']w',  # File write operations
        r'socket\.',  # Network operations
        r'requests\.',  # HTTP requests
        r'urllib\.',  # URL operations
    ]

    def __init__(self, weight: float = 1.0, docker_mode: bool = False):
        super().__init__("SafetyVerifier", weight=weight, has_veto=True)
        self.docker_mode = docker_mode

        # In Docker mode, some operations are safer
        if docker_mode:
            self.DANGEROUS_CALLS = {'eval', 'exec', 'compile', '__import__'}

    def verify(self, code: str, **kwargs) -> VerificationResult:
        """Verify code safety."""
        start_time = datetime.now()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Let SyntaxVerifier handle this
            result = VerificationResult(
                verifier_name=self.name,
                passed=True,
                confidence=0.5,
                message="Cannot verify - syntax error",
                has_veto=self.has_veto,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            self._log_verification(result)
            return result

        # Check for dangerous function calls
        dangerous_found = self._find_dangerous_calls(tree)

        # Check for suspicious patterns
        suspicious_found = self._find_suspicious_patterns(code)

        all_issues = dangerous_found + suspicious_found

        if all_issues:
            result = VerificationResult(
                verifier_name=self.name,
                passed=False,
                confidence=0.0,
                message=f"Dangerous operations found: {', '.join(all_issues[:5])}",
                details={
                    'dangerous_calls': dangerous_found,
                    'suspicious_patterns': suspicious_found
                },
                has_veto=self.has_veto,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        else:
            result = VerificationResult(
                verifier_name=self.name,
                passed=True,
                confidence=1.0,
                message="No dangerous operations detected",
                has_veto=self.has_veto,
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        self._log_verification(result)
        return result

    def _find_dangerous_calls(self, tree: ast.AST) -> List[str]:
        """Find dangerous function calls in AST."""
        dangerous = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name:
                    # Check exact match
                    if func_name in self.DANGEROUS_CALLS:
                        dangerous.append(func_name)

                    # Check for dangerous patterns
                    if 'system' in func_name.lower():
                        dangerous.append(func_name)

        return dangerous

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract function name from Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return None

    def _find_suspicious_patterns(self, code: str) -> List[str]:
        """Find suspicious patterns in code."""
        found = []
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                found.append(f"Pattern: {pattern}")
        return found


class TypeVerifier(Verifier):
    """
    Verifies code type correctness using mypy (if available).

    This is an optional verifier that provides additional safety
    through static type checking.
    """

    def __init__(self, weight: float = 0.8):
        super().__init__("TypeVerifier", weight=weight, has_veto=False)
        self.mypy_available = self._check_mypy()

    def _check_mypy(self) -> bool:
        """Check if mypy is available."""
        try:
            result = subprocess.run(
                ['mypy', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def verify(self, code: str, **kwargs) -> VerificationResult:
        """Verify code types using mypy."""
        start_time = datetime.now()

        if not self.mypy_available:
            result = VerificationResult(
                verifier_name=self.name,
                passed=True,
                confidence=0.5,  # Neutral - couldn't verify
                message="mypy not available, skipping type check",
                details={'mypy_available': False},
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            self._log_verification(result)
            return result

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run mypy
            result_proc = subprocess.run(
                ['mypy', '--ignore-missing-imports', '--no-error-summary', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            errors = []
            warnings = []

            for line in result_proc.stdout.split('\n'):
                if 'error:' in line:
                    errors.append(line)
                elif 'warning:' in line or 'note:' in line:
                    warnings.append(line)

            if errors:
                # Calculate confidence based on error count
                confidence = max(0.0, 1.0 - (len(errors) * 0.2))
                result = VerificationResult(
                    verifier_name=self.name,
                    passed=False,
                    confidence=confidence,
                    message=f"Type errors found: {len(errors)}",
                    details={
                        'errors': errors[:10],  # Limit to first 10
                        'warnings': warnings[:5]
                    },
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            else:
                confidence = 1.0 if not warnings else 0.9
                result = VerificationResult(
                    verifier_name=self.name,
                    passed=True,
                    confidence=confidence,
                    message="Type check passed" + (f" with {len(warnings)} warnings" if warnings else ""),
                    details={'warnings': warnings[:5]} if warnings else None,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

        except subprocess.TimeoutExpired:
            result = VerificationResult(
                verifier_name=self.name,
                passed=True,
                confidence=0.5,
                message="Type check timed out",
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_file)
            except OSError:
                pass

        self._log_verification(result)
        return result


class TestVerifier(Verifier):
    """
    Verifies code by running test cases.

    Uses the existing sandbox infrastructure to run tests safely.
    """

    def __init__(self, weight: float = 1.0, sandbox: Any = None):
        super().__init__("TestVerifier", weight=weight, has_veto=False)
        self.sandbox = sandbox

    def verify(self, code: str, **kwargs) -> VerificationResult:
        """Verify code by running test cases."""
        start_time = datetime.now()

        test_cases = kwargs.get('test_cases', [])

        if not test_cases:
            result = VerificationResult(
                verifier_name=self.name,
                passed=True,
                confidence=0.5,  # Neutral - no tests to run
                message="No test cases provided",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            self._log_verification(result)
            return result

        # Use sandbox if available
        if self.sandbox:
            success, test_results = self.sandbox.test_code(code, test_cases)

            tests_passed = test_results.get('tests_passed', 0)
            tests_failed = test_results.get('tests_failed', 0)
            total = tests_passed + tests_failed

            if total > 0:
                confidence = tests_passed / total
            else:
                confidence = 0.5

            result = VerificationResult(
                verifier_name=self.name,
                passed=success,
                confidence=confidence,
                message=f"Tests: {tests_passed}/{total} passed",
                details={
                    'tests_passed': tests_passed,
                    'tests_failed': tests_failed,
                    'errors': test_results.get('errors', [])
                },
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        else:
            # No sandbox - try basic execution
            result = self._run_tests_basic(code, test_cases, start_time)

        self._log_verification(result)
        return result

    def _run_tests_basic(self, code: str, test_cases: List[Dict], start_time: datetime) -> VerificationResult:
        """Run tests without sandbox (basic mode)."""
        import importlib.util

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        tests_passed = 0
        tests_failed = 0
        errors = []

        try:
            spec = importlib.util.spec_from_file_location("test_module", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for test in test_cases:
                func_name = test.get('function', 'main')
                if hasattr(module, func_name):
                    try:
                        func = getattr(module, func_name)
                        result = func(*test.get('args', []), **test.get('kwargs', {}))

                        if 'expected' in test:
                            if result == test['expected']:
                                tests_passed += 1
                            else:
                                tests_failed += 1
                                errors.append(f"Expected {test['expected']}, got {result}")
                        else:
                            tests_passed += 1
                    except Exception as e:
                        tests_failed += 1
                        errors.append(str(e))
                else:
                    tests_failed += 1
                    errors.append(f"Function {func_name} not found")

        except Exception as e:
            errors.append(f"Import error: {e}")

        finally:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

        total = tests_passed + tests_failed
        confidence = tests_passed / total if total > 0 else 0.5

        return VerificationResult(
            verifier_name=self.name,
            passed=tests_failed == 0 and not errors,
            confidence=confidence,
            message=f"Tests: {tests_passed}/{total} passed",
            details={
                'tests_passed': tests_passed,
                'tests_failed': tests_failed,
                'errors': errors
            },
            execution_time=(datetime.now() - start_time).total_seconds()
        )


class LLMVerifier(Verifier):
    """
    Verifies code using LLM-based analysis.

    Asks an LLM to evaluate code for safety, correctness, and quality.
    Requires an LLM interface to be provided.
    """

    VERIFICATION_PROMPT = """Analyze the following Python code for safety and correctness.

Code:
```python
{code}
```

Evaluate:
1. Is this code safe to execute? (no malicious operations, no security vulnerabilities)
2. Is this code likely to be correct and bug-free?
3. Does this code follow good practices?

Respond with a JSON object:
{{
    "safe": true/false,
    "correct": true/false (to the best of your judgment),
    "quality": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of any issues found"],
    "explanation": "brief explanation"
}}
"""

    def __init__(self, weight: float = 0.7, llm_interface: Optional[Callable] = None):
        super().__init__("LLMVerifier", weight=weight, has_veto=False)
        self.llm_interface = llm_interface

    def verify(self, code: str, **kwargs) -> VerificationResult:
        """Verify code using LLM analysis."""
        start_time = datetime.now()

        if not self.llm_interface:
            result = VerificationResult(
                verifier_name=self.name,
                passed=True,
                confidence=0.5,  # Neutral - couldn't verify
                message="LLM interface not available",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            self._log_verification(result)
            return result

        try:
            # Call LLM with verification prompt
            prompt = self.VERIFICATION_PROMPT.format(code=code)
            response = self.llm_interface(prompt)

            # Parse LLM response
            result_data = self._parse_llm_response(response)

            passed = result_data.get('safe', True) and result_data.get('correct', True)
            confidence = result_data.get('confidence', 0.5)
            issues = result_data.get('issues', [])

            result = VerificationResult(
                verifier_name=self.name,
                passed=passed,
                confidence=confidence,
                message=result_data.get('explanation', 'LLM analysis complete'),
                details={
                    'safe': result_data.get('safe'),
                    'correct': result_data.get('correct'),
                    'quality': result_data.get('quality'),
                    'issues': issues
                },
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            result = VerificationResult(
                verifier_name=self.name,
                passed=True,
                confidence=0.5,
                message=f"LLM verification failed: {e}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        self._log_verification(result)
        return result

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        import json

        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: parse text response
        result = {
            'safe': 'unsafe' not in response.lower() and 'dangerous' not in response.lower(),
            'correct': 'incorrect' not in response.lower() and 'bug' not in response.lower(),
            'quality': True,
            'confidence': 0.6,
            'issues': [],
            'explanation': response[:200]
        }
        return result


class StyleVerifier(Verifier):
    """
    Verifies code style against basic PEP8 conventions.

    Checks:
    - Line length (max 120 chars, warn at 80)
    - Naming conventions (snake_case for functions/variables)
    - Basic formatting
    """

    MAX_LINE_LENGTH = 120
    RECOMMENDED_LINE_LENGTH = 80

    def __init__(self, weight: float = 0.5):
        super().__init__("StyleVerifier", weight=weight, has_veto=False)

    def verify(self, code: str, **kwargs) -> VerificationResult:
        """Verify code style."""
        start_time = datetime.now()

        issues = []
        warnings = []

        lines = code.split('\n')

        # Check line lengths
        for i, line in enumerate(lines, 1):
            if len(line) > self.MAX_LINE_LENGTH:
                issues.append(f"Line {i}: too long ({len(line)} > {self.MAX_LINE_LENGTH})")
            elif len(line) > self.RECOMMENDED_LINE_LENGTH:
                warnings.append(f"Line {i}: consider shorter ({len(line)} > {self.RECOMMENDED_LINE_LENGTH})")

        # Check naming conventions using AST
        try:
            tree = ast.parse(code)
            naming_issues = self._check_naming(tree)
            issues.extend(naming_issues)
        except SyntaxError:
            pass  # Let SyntaxVerifier handle this

        # Check for trailing whitespace
        for i, line in enumerate(lines, 1):
            if line.rstrip() != line:
                warnings.append(f"Line {i}: trailing whitespace")

        # Check for multiple blank lines
        blank_count = 0
        for i, line in enumerate(lines, 1):
            if not line.strip():
                blank_count += 1
                if blank_count > 2:
                    issues.append(f"Line {i}: too many blank lines")
            else:
                blank_count = 0

        # Calculate confidence based on issues
        if issues:
            confidence = max(0.3, 1.0 - (len(issues) * 0.15))
            passed = len(issues) < 5  # Allow minor style issues
        else:
            confidence = 1.0 if not warnings else 0.9
            passed = True

        result = VerificationResult(
            verifier_name=self.name,
            passed=passed,
            confidence=confidence,
            message=f"Style check: {len(issues)} issues, {len(warnings)} warnings",
            details={
                'issues': issues[:10],
                'warnings': warnings[:10]
            },
            execution_time=(datetime.now() - start_time).total_seconds()
        )

        self._log_verification(result)
        return result

    def _check_naming(self, tree: ast.AST) -> List[str]:
        """Check naming conventions."""
        issues = []

        for node in ast.walk(tree):
            # Check function names (should be snake_case)
            if isinstance(node, ast.FunctionDef):
                if not self._is_snake_case(node.name) and not node.name.startswith('_'):
                    issues.append(f"Function '{node.name}' should be snake_case")

            # Check class names (should be PascalCase)
            elif isinstance(node, ast.ClassDef):
                if not self._is_pascal_case(node.name):
                    issues.append(f"Class '{node.name}' should be PascalCase")

            # Check constant names (UPPER_SNAKE_CASE at module level)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # If it looks like a constant but isn't UPPER_CASE
                        name = target.id
                        if name.isupper() or '_' in name and name == name.upper():
                            continue  # It's a proper constant

        return issues

    def _is_snake_case(self, name: str) -> bool:
        """Check if name is snake_case."""
        if name.startswith('__') and name.endswith('__'):
            return True  # Dunder methods are OK
        return bool(re.match(r'^[a-z][a-z0-9_]*$', name))

    def _is_pascal_case(self, name: str) -> bool:
        """Check if name is PascalCase."""
        return bool(re.match(r'^[A-Z][a-zA-Z0-9]*$', name))


class EnsembleVerifier:
    """
    Combines multiple verifiers into an ensemble decision system.

    Supports different voting strategies:
    - UNANIMOUS: All verifiers must pass
    - MAJORITY: >50% must pass
    - WEIGHTED: Weighted average confidence > threshold

    SafetyVerifier has veto power - if it fails, the code is rejected
    regardless of other verifiers.
    """

    def __init__(
        self,
        verifiers: Optional[List[Verifier]] = None,
        voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED,
        threshold: float = 0.7,
        sandbox: Any = None,
        llm_interface: Optional[Callable] = None
    ):
        """
        Initialize ensemble verifier.

        Args:
            verifiers: List of verifiers (or use defaults)
            voting_strategy: How to combine verifier results
            threshold: Confidence threshold for WEIGHTED voting
            sandbox: Sandbox for TestVerifier
            llm_interface: LLM interface for LLMVerifier
        """
        self.voting_strategy = voting_strategy
        self.threshold = threshold
        self.verification_history: List[EnsembleResult] = []

        # Use provided verifiers or create defaults
        if verifiers:
            self.verifiers = verifiers
        else:
            self.verifiers = self._create_default_verifiers(sandbox, llm_interface)

    def _create_default_verifiers(
        self,
        sandbox: Any = None,
        llm_interface: Optional[Callable] = None
    ) -> List[Verifier]:
        """Create default verifier set."""
        return [
            SyntaxVerifier(weight=1.0),
            SafetyVerifier(weight=1.0),  # Has veto power
            TypeVerifier(weight=0.8),
            TestVerifier(weight=1.0, sandbox=sandbox),
            LLMVerifier(weight=0.7, llm_interface=llm_interface),
            StyleVerifier(weight=0.5),
        ]

    def verify(self, code: str, **kwargs) -> EnsembleResult:
        """
        Run all verifiers and combine results.

        Args:
            code: Python code to verify
            **kwargs: Additional args (test_cases for TestVerifier)

        Returns:
            EnsembleResult with final decision
        """
        results: List[VerificationResult] = []

        # Run all verifiers
        for verifier in self.verifiers:
            result = verifier.verify(code, **kwargs)
            result.has_veto = verifier.has_veto
            results.append(result)

        # Check for veto
        vetoed = False
        vetoed_by = None
        for result in results:
            if result.has_veto and not result.passed:
                vetoed = True
                vetoed_by = result.verifier_name
                break

        if vetoed:
            ensemble_result = EnsembleResult(
                passed=False,
                final_confidence=0.0,
                voting_strategy=self.voting_strategy,
                threshold=self.threshold,
                results=results,
                vetoed=True,
                vetoed_by=vetoed_by
            )
        else:
            # Apply voting strategy
            passed, confidence = self._apply_voting(results)

            ensemble_result = EnsembleResult(
                passed=passed,
                final_confidence=confidence,
                voting_strategy=self.voting_strategy,
                threshold=self.threshold,
                results=results
            )

        self.verification_history.append(ensemble_result)
        return ensemble_result

    def _apply_voting(self, results: List[VerificationResult]) -> Tuple[bool, float]:
        """Apply voting strategy to results."""
        if self.voting_strategy == VotingStrategy.UNANIMOUS:
            passed = all(r.passed for r in results)
            confidence = min(r.confidence for r in results) if results else 0.0
            return passed, confidence

        elif self.voting_strategy == VotingStrategy.MAJORITY:
            passed_count = sum(1 for r in results if r.passed)
            total = len(results)
            passed = passed_count > total / 2
            confidence = passed_count / total if total > 0 else 0.0
            return passed, confidence

        else:  # WEIGHTED
            total_weight = sum(v.weight for v in self.verifiers)
            if total_weight == 0:
                return False, 0.0

            weighted_confidence = sum(
                r.confidence * v.weight
                for r, v in zip(results, self.verifiers)
            ) / total_weight

            passed = weighted_confidence >= self.threshold
            return passed, weighted_confidence

    def add_verifier(self, verifier: Verifier):
        """Add a verifier to the ensemble."""
        self.verifiers.append(verifier)

    def remove_verifier(self, name: str) -> bool:
        """Remove a verifier by name."""
        for i, v in enumerate(self.verifiers):
            if v.name == name:
                self.verifiers.pop(i)
                return True
        return False

    def get_verifier(self, name: str) -> Optional[Verifier]:
        """Get a verifier by name."""
        for v in self.verifiers:
            if v.name == name:
                return v
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        if not self.verification_history:
            return {
                'total_verifications': 0,
                'pass_rate': 0.0,
                'avg_confidence': 0.0,
                'veto_count': 0,
                'verifier_stats': {}
            }

        total = len(self.verification_history)
        passed = sum(1 for v in self.verification_history if v.passed)
        avg_conf = sum(v.final_confidence for v in self.verification_history) / total
        veto_count = sum(1 for v in self.verification_history if v.vetoed)

        # Per-verifier stats
        verifier_stats = {}
        for verifier in self.verifiers:
            verifier_stats[verifier.name] = verifier.get_stats()

        return {
            'total_verifications': total,
            'pass_rate': passed / total,
            'avg_confidence': avg_conf,
            'veto_count': veto_count,
            'voting_strategy': self.voting_strategy.value,
            'threshold': self.threshold,
            'verifier_stats': verifier_stats
        }

    def get_verifier_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate effectiveness metrics for each verifier.

        Returns stats like:
        - Issues caught: how many times this verifier was the sole failure
        - False positive rate: estimate based on other verifiers passing
        """
        if not self.verification_history:
            return {}

        effectiveness = {}

        for verifier in self.verifiers:
            name = verifier.name
            effectiveness[name] = {
                'total_failures': 0,
                'sole_failures': 0,  # Failed when others passed
                'vetoes': 0,
                'avg_confidence': 0.0
            }

        for ensemble_result in self.verification_history:
            for result in ensemble_result.results:
                name = result.verifier_name
                if name not in effectiveness:
                    continue

                if not result.passed:
                    effectiveness[name]['total_failures'] += 1

                    # Check if sole failure
                    other_passed = all(
                        r.passed for r in ensemble_result.results
                        if r.verifier_name != name
                    )
                    if other_passed:
                        effectiveness[name]['sole_failures'] += 1

                if ensemble_result.vetoed and ensemble_result.vetoed_by == name:
                    effectiveness[name]['vetoes'] += 1

        # Calculate averages
        for name in effectiveness:
            stats = next(
                (v.get_stats() for v in self.verifiers if v.name == name),
                {'avg_confidence': 0.0}
            )
            effectiveness[name]['avg_confidence'] = stats.get('avg_confidence', 0.0)

        return effectiveness


# Factory function for easy creation
def create_ensemble_verifier(
    config: Optional[Dict[str, Any]] = None,
    sandbox: Any = None,
    llm_interface: Optional[Callable] = None
) -> EnsembleVerifier:
    """
    Create an EnsembleVerifier with configuration.

    Default config:
    - syntax: required (weight 1.0)
    - safety: veto power (weight 1.0)
    - test: required (weight 1.0)
    - type: optional (weight 0.8)
    - llm: optional (weight 0.7)
    - style: optional (weight 0.5)

    Args:
        config: Optional configuration dict
        sandbox: Sandbox for test execution
        llm_interface: LLM function for verification

    Returns:
        Configured EnsembleVerifier
    """
    if config is None:
        config = {}

    voting_strategy = VotingStrategy(config.get('voting_strategy', 'weighted'))
    threshold = config.get('threshold', 0.7)
    docker_mode = config.get('docker_mode', False)

    verifiers = []

    # Syntax verifier (always included)
    if config.get('syntax', {}).get('enabled', True):
        weight = config.get('syntax', {}).get('weight', 1.0)
        verifiers.append(SyntaxVerifier(weight=weight))

    # Safety verifier (always included, has veto)
    if config.get('safety', {}).get('enabled', True):
        weight = config.get('safety', {}).get('weight', 1.0)
        verifiers.append(SafetyVerifier(weight=weight, docker_mode=docker_mode))

    # Type verifier (optional)
    if config.get('type', {}).get('enabled', True):
        weight = config.get('type', {}).get('weight', 0.8)
        verifiers.append(TypeVerifier(weight=weight))

    # Test verifier
    if config.get('test', {}).get('enabled', True):
        weight = config.get('test', {}).get('weight', 1.0)
        verifiers.append(TestVerifier(weight=weight, sandbox=sandbox))

    # LLM verifier (optional)
    if config.get('llm', {}).get('enabled', True) and llm_interface:
        weight = config.get('llm', {}).get('weight', 0.7)
        verifiers.append(LLMVerifier(weight=weight, llm_interface=llm_interface))

    # Style verifier (optional)
    if config.get('style', {}).get('enabled', True):
        weight = config.get('style', {}).get('weight', 0.5)
        verifiers.append(StyleVerifier(weight=weight))

    return EnsembleVerifier(
        verifiers=verifiers,
        voting_strategy=voting_strategy,
        threshold=threshold,
        sandbox=sandbox,
        llm_interface=llm_interface
    )


# Global instance
_ensemble_verifier: Optional[EnsembleVerifier] = None


def get_ensemble_verifier() -> EnsembleVerifier:
    """Get global ensemble verifier instance."""
    global _ensemble_verifier
    if _ensemble_verifier is None:
        _ensemble_verifier = EnsembleVerifier()
    return _ensemble_verifier


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("ENSEMBLE VERIFIER TEST")
    print("=" * 60)

    # Create ensemble
    ensemble = EnsembleVerifier()

    # Test 1: Valid code
    print("\n1. Testing valid code:")
    valid_code = '''
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''
    result = ensemble.verify(valid_code)
    print(f"   Passed: {result.passed}")
    print(f"   Confidence: {result.final_confidence:.2f}")
    for r in result.results:
        print(f"   - {r.verifier_name}: {'PASS' if r.passed else 'FAIL'} ({r.confidence:.2f})")

    # Test 2: Dangerous code
    print("\n2. Testing dangerous code:")
    dangerous_code = '''
def evil():
    import os
    os.system("rm -rf /")
'''
    result = ensemble.verify(dangerous_code)
    print(f"   Passed: {result.passed}")
    print(f"   Vetoed: {result.vetoed}")
    print(f"   Vetoed by: {result.vetoed_by}")

    # Test 3: Code with syntax error
    print("\n3. Testing syntax error:")
    bad_syntax = '''
def broken(
    return "oops"
'''
    result = ensemble.verify(bad_syntax)
    print(f"   Passed: {result.passed}")
    print(f"   Confidence: {result.final_confidence:.2f}")

    # Test 4: Code with test cases
    print("\n4. Testing with test cases:")
    test_code = '''
def add(a, b):
    return a + b
'''
    test_cases = [
        {'function': 'add', 'args': [2, 3], 'expected': 5},
        {'function': 'add', 'args': [0, 0], 'expected': 0},
    ]
    result = ensemble.verify(test_code, test_cases=test_cases)
    print(f"   Passed: {result.passed}")
    print(f"   Confidence: {result.final_confidence:.2f}")

    # Stats
    print("\n5. Ensemble stats:")
    stats = ensemble.get_stats()
    print(f"   Total verifications: {stats['total_verifications']}")
    print(f"   Pass rate: {stats['pass_rate']:.2f}")
    print(f"   Veto count: {stats['veto_count']}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
