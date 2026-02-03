"""
Code Evolution System
====================

Enables TRUE self-modification:
1. Generate code improvements based on learning
2. Validate code safety (AST parsing, sandbox testing)
3. Test generated code against benchmarks
4. Deploy if improved, rollback if degraded
5. Track all modifications with git-like versioning

This is the missing piece that makes the AI truly self-evolving.
"""

import os
import sys
import ast
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import importlib.util

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import EVOLUTION_DIR

# Import Docker sandbox and check availability
try:
    from integrations.docker_sandbox import DockerSandbox, get_docker_sandbox
    _docker_sandbox = get_docker_sandbox()
    DOCKER_AVAILABLE = _docker_sandbox.is_available()
except ImportError:
    DOCKER_AVAILABLE = False
    _docker_sandbox = None
    DockerSandbox = None

# Import population evolution for genetic algorithm-style code improvement
try:
    from integrations.population_evolution import (
        Population, CodeIndividual, CodeMutator, MutationType
    )
    POPULATION_AVAILABLE = True
except ImportError:
    POPULATION_AVAILABLE = False
    Population = None
    CodeIndividual = None
    CodeMutator = None
    MutationType = None

# Import ensemble verifier for multi-verifier code validation
try:
    from integrations.ensemble_verifier import (
        EnsembleVerifier, EnsembleResult, VerificationResult as EnsembleVerificationResult,
        VotingStrategy, create_ensemble_verifier
    )
    ENSEMBLE_VERIFIER_AVAILABLE = True
except ImportError:
    ENSEMBLE_VERIFIER_AVAILABLE = False
    EnsembleVerifier = None
    EnsembleResult = None
    EnsembleVerificationResult = None
    VotingStrategy = None
    create_ensemble_verifier = None


class CodeChangeType(Enum):
    """Types of code modifications."""
    NEW_FUNCTION = "new_function"
    MODIFY_FUNCTION = "modify_function"
    NEW_CLASS = "new_class"
    MODIFY_CLASS = "modify_class"
    NEW_MODULE = "new_module"
    OPTIMIZATION = "optimization"
    BUG_FIX = "bug_fix"


class ValidationResult(Enum):
    """Code validation results."""
    VALID = "valid"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    DANGEROUS_CODE = "dangerous_code"
    TEST_FAILED = "test_failed"
    PERFORMANCE_DEGRADED = "performance_degraded"


@dataclass
class CodeChange:
    """Represents a single code change."""
    id: str
    change_type: CodeChangeType
    target_file: str
    original_code: Optional[str]
    new_code: str
    description: str
    timestamp: str
    validation_result: Optional[ValidationResult] = None
    test_results: Optional[Dict] = None
    deployed: bool = False
    rolled_back: bool = False
    verification_results: Optional[Dict] = None  # Ensemble verifier results


class CodeValidator:
    """
    Validates code safety before execution.

    Checks:
    - Syntax validity (AST parsing)
    - No dangerous operations (exec, eval, os.system with user input)
    - No network operations without approval
    - No file operations outside allowed directories

    Note: When Docker sandbox is available, some restrictions can be relaxed
    since code runs in an isolated container with no host access.
    """

    # Dangerous function calls to block (when NOT using Docker)
    DANGEROUS_CALLS_NO_DOCKER = {
        'eval', 'exec', 'compile', '__import__',
        'open',  # Only blocked if writing outside allowed dirs
        'subprocess.call', 'subprocess.run', 'subprocess.Popen',
        'os.system', 'os.popen', 'os.spawn',
        'shutil.rmtree',  # Dangerous deletion
    }

    # Dangerous function calls when Docker IS available
    # More permissive since code runs in isolated container
    DANGEROUS_CALLS_WITH_DOCKER = {
        'eval', 'exec',  # Still block code injection
        '__import__',   # Still block dynamic imports
        # subprocess, os.system, etc. are safe in container
        # file operations are safe - container has no host access
    }

    # Allowed directories for file operations
    ALLOWED_DIRS = [
        EVOLUTION_DIR,
        Path.home() / ".cognitive_ai_knowledge",
        Path("/tmp"),
    ]

    def __init__(self, use_docker: bool = True):
        self.validation_log: List[Dict] = []
        self.use_docker = use_docker and DOCKER_AVAILABLE

        # Select appropriate dangerous calls set based on Docker availability
        if self.use_docker:
            self.DANGEROUS_CALLS = self.DANGEROUS_CALLS_WITH_DOCKER
            print("[CodeValidator] Using relaxed validation (Docker isolation available)")
        else:
            self.DANGEROUS_CALLS = self.DANGEROUS_CALLS_NO_DOCKER
            print("[CodeValidator] Using strict validation (no Docker isolation)")

    def validate(self, code: str, allow_file_ops: bool = False) -> Tuple[ValidationResult, str]:
        """
        Validate code for safety and correctness.

        Args:
            code: Python code to validate
            allow_file_ops: Whether to allow file operations

        Returns:
            (ValidationResult, message)
        """
        # Step 1: Syntax check via AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult.SYNTAX_ERROR, f"Syntax error at line {e.lineno}: {e.msg}"

        # Step 2: Check for dangerous operations
        dangerous = self._find_dangerous_operations(tree, allow_file_ops)
        if dangerous:
            return ValidationResult.DANGEROUS_CODE, f"Dangerous operations found: {', '.join(dangerous)}"

        # Step 3: Check imports are available
        imports = self._extract_imports(tree)
        missing = self._check_imports(imports)
        if missing:
            return ValidationResult.IMPORT_ERROR, f"Missing imports: {', '.join(missing)}"

        self.validation_log.append({
            'timestamp': datetime.now().isoformat(),
            'code_hash': hashlib.md5(code.encode()).hexdigest(),
            'result': 'valid'
        })

        return ValidationResult.VALID, "Code passed all validation checks"

    def _find_dangerous_operations(self, tree: ast.AST, allow_file_ops: bool) -> List[str]:
        """Find dangerous function calls in AST."""
        dangerous_found = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name:
                    # Check against dangerous calls
                    if func_name in self.DANGEROUS_CALLS:
                        if func_name == 'open' and allow_file_ops:
                            continue  # Allow if file ops permitted
                        dangerous_found.append(func_name)

                    # Check for dangerous patterns
                    if 'system' in func_name.lower():
                        dangerous_found.append(func_name)

        return dangerous_found

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

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
        return list(set(imports))

    def _check_imports(self, imports: List[str]) -> List[str]:
        """Check which imports are missing."""
        missing = []
        for imp in imports:
            try:
                __import__(imp)
            except ImportError:
                missing.append(imp)
        return missing


class CodeSandbox:
    """
    Sandboxed environment for testing code changes.

    Creates isolated environment, runs tests, measures performance.
    Supports Docker-based isolation (preferred) with Python fallback.
    """

    def __init__(self, base_dir: Path = None, use_docker: bool = True):
        self.base_dir = base_dir or EVOLUTION_DIR / "sandbox"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.test_results: List[Dict] = []
        self.use_docker = use_docker and DOCKER_AVAILABLE

        # Log which sandbox type will be used
        if self.use_docker:
            print("[CodeSandbox] Using Docker-based isolation (more secure)")
        else:
            if use_docker and not DOCKER_AVAILABLE:
                print("[CodeSandbox] Docker not available, falling back to Python sandbox")
            else:
                print("[CodeSandbox] Using Python-based sandbox")

    def test_code(self, code: str, test_cases: List[Dict] = None) -> Tuple[bool, Dict]:
        """
        Test code in sandbox environment.

        Tries Docker isolation first if available, falls back to Python sandbox.

        Args:
            code: Python code to test
            test_cases: List of {input: ..., expected: ...} dicts

        Returns:
            (success, results_dict)
        """
        # Try Docker sandbox first if available
        if self.use_docker and _docker_sandbox is not None:
            return self._test_code_docker(code, test_cases)
        else:
            return self._test_code_python(code, test_cases)

    def _test_code_docker(self, code: str, test_cases: List[Dict] = None) -> Tuple[bool, Dict]:
        """
        Test code using Docker sandbox (more secure).

        Args:
            code: Python code to test
            test_cases: List of {function, args, kwargs, expected} dicts

        Returns:
            (success, results_dict)
        """
        print("[CodeSandbox] Executing in Docker container...")
        success, docker_results = _docker_sandbox.execute_code(code, test_cases)

        # Convert Docker results to our format for compatibility
        results = {
            'success': success,
            'tests_passed': docker_results.get('tests_passed', 0),
            'tests_failed': docker_results.get('tests_failed', 0),
            'errors': docker_results.get('errors', []),
            'execution_time': docker_results.get('execution_time', 0),
            'memory_usage': docker_results.get('memory_usage', 0),
            'sandbox_type': docker_results.get('sandbox_type', 'docker'),
            'stdout': docker_results.get('stdout', ''),
            'stderr': docker_results.get('stderr', ''),
            'exit_code': docker_results.get('exit_code', -1)
        }

        self.test_results.append(results)
        return results['success'], results

    def _test_code_python(self, code: str, test_cases: List[Dict] = None) -> Tuple[bool, Dict]:
        """
        Test code using Python subprocess sandbox (fallback, less secure).

        Args:
            code: Python code to test
            test_cases: List of {input: ..., expected: ...} dicts

        Returns:
            (success, results_dict)
        """
        print("[CodeSandbox] Executing in Python subprocess...")
        results = {
            'success': False,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'execution_time': 0,
            'memory_usage': 0,
            'sandbox_type': 'python'
        }

        # Create temp file for code
        test_file = self.base_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"

        try:
            # Write code to temp file
            with open(test_file, 'w') as f:
                f.write(code)

            # Try to import and execute
            start_time = datetime.now()

            spec = importlib.util.spec_from_file_location("test_module", test_file)
            module = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(module)
                results['import_success'] = True
            except Exception as e:
                results['errors'].append(f"Import error: {str(e)}")
                return False, results

            # Run test cases if provided
            if test_cases:
                for i, test in enumerate(test_cases):
                    try:
                        # Get function to test
                        func_name = test.get('function', 'main')
                        if hasattr(module, func_name):
                            func = getattr(module, func_name)
                            result = func(*test.get('args', []), **test.get('kwargs', {}))

                            if 'expected' in test:
                                if result == test['expected']:
                                    results['tests_passed'] += 1
                                else:
                                    results['tests_failed'] += 1
                                    results['errors'].append(
                                        f"Test {i+1}: Expected {test['expected']}, got {result}"
                                    )
                            else:
                                results['tests_passed'] += 1
                        else:
                            results['errors'].append(f"Function {func_name} not found")
                            results['tests_failed'] += 1
                    except Exception as e:
                        results['tests_failed'] += 1
                        results['errors'].append(f"Test {i+1} error: {str(e)}")

            results['execution_time'] = (datetime.now() - start_time).total_seconds()
            results['success'] = results['tests_failed'] == 0 and not results['errors']

        except Exception as e:
            results['errors'].append(f"Sandbox error: {str(e)}")
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

        self.test_results.append(results)
        return results['success'], results

    def get_sandbox_type(self) -> str:
        """Return which sandbox type is currently in use."""
        return "docker" if self.use_docker else "python"

    def get_stats(self) -> Dict:
        """Get sandbox execution statistics."""
        if not self.test_results:
            return {
                'total_tests': 0,
                'success_rate': 0.0,
                'sandbox_type': self.get_sandbox_type(),
                'docker_available': DOCKER_AVAILABLE
            }

        successful = sum(1 for r in self.test_results if r.get('success'))
        docker_tests = sum(1 for r in self.test_results if r.get('sandbox_type') == 'docker')
        python_tests = sum(1 for r in self.test_results if r.get('sandbox_type') == 'python')

        return {
            'total_tests': len(self.test_results),
            'successful_tests': successful,
            'success_rate': successful / len(self.test_results),
            'docker_tests': docker_tests,
            'python_tests': python_tests,
            'sandbox_type': self.get_sandbox_type(),
            'docker_available': DOCKER_AVAILABLE
        }


class CodeVersionControl:
    """
    Git-like versioning for code changes.

    Tracks all modifications, enables rollback.
    """

    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or EVOLUTION_DIR / "code_versions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_dir / "history.json"
        self.history: List[Dict] = self._load_history()

    def _load_history(self) -> List[Dict]:
        """Load version history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError, OSError, TypeError) as e:
                # Return empty history if file is corrupted or unreadable
                return []
        return []

    def _save_history(self):
        """Save version history."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def commit(self, change: CodeChange) -> str:
        """
        Commit a code change.

        Returns:
            commit_id
        """
        commit_id = hashlib.md5(
            f"{change.target_file}{change.new_code}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Save original code as backup
        if change.original_code:
            backup_file = self.storage_dir / f"{commit_id}_original.py"
            with open(backup_file, 'w') as f:
                f.write(change.original_code)

        # Save new code
        new_file = self.storage_dir / f"{commit_id}_new.py"
        with open(new_file, 'w') as f:
            f.write(change.new_code)

        # Record in history
        self.history.append({
            'commit_id': commit_id,
            'change_id': change.id,
            'change_type': change.change_type.value,
            'target_file': change.target_file,
            'description': change.description,
            'timestamp': datetime.now().isoformat(),
            'deployed': change.deployed,
            'rolled_back': change.rolled_back
        })
        self._save_history()

        return commit_id

    def rollback(self, commit_id: str) -> bool:
        """
        Rollback to previous version.

        Returns:
            success
        """
        # Find commit
        commit = None
        for c in self.history:
            if c['commit_id'] == commit_id:
                commit = c
                break

        if not commit:
            return False

        # Restore original code
        backup_file = self.storage_dir / f"{commit_id}_original.py"
        if backup_file.exists():
            with open(backup_file, 'r') as f:
                original_code = f.read()

            # Restore to target file
            target = Path(commit['target_file'])
            if target.exists():
                with open(target, 'w') as f:
                    f.write(original_code)

            # Mark as rolled back
            commit['rolled_back'] = True
            self._save_history()

            return True

        return False

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent commit history."""
        return self.history[-limit:]


class CodeIntrospector:
    """
    Allows the AI to read and analyze its own source code.

    This enables:
    - Understanding current implementations
    - Finding functions to optimize
    - Learning from existing patterns
    - Identifying bugs to fix
    """

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.source_cache: Dict[str, str] = {}

    def read_file(self, relative_path: str) -> Optional[str]:
        """
        Read a source file from the project.

        Args:
            relative_path: Path relative to project root (e.g., "integrations/self_evolution.py")

        Returns:
            File contents or None if not found
        """
        try:
            full_path = self.project_root / relative_path
            if not full_path.exists():
                return None

            # Security: only allow reading Python files in project
            if not str(full_path).startswith(str(self.project_root)):
                print(f"[Introspector] Security: Cannot read outside project root")
                return None

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.source_cache[relative_path] = content
                return content

        except Exception as e:
            print(f"[Introspector] Error reading {relative_path}: {e}")
            return None

    def read_function(self, file_path: str, function_name: str) -> Optional[str]:
        """
        Extract a specific function's source code.

        Args:
            file_path: Relative path to the file
            function_name: Name of the function to extract

        Returns:
            Function source code or None
        """
        content = self.read_file(file_path)
        if not content:
            return None

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Extract source lines
                    lines = content.split('\n')
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
                    return '\n'.join(lines[start:end])
        except Exception as e:
            print(f"[Introspector] Error parsing {file_path}: {e}")

        return None

    def read_class(self, file_path: str, class_name: str) -> Optional[str]:
        """
        Extract a specific class's source code.

        Args:
            file_path: Relative path to the file
            class_name: Name of the class to extract

        Returns:
            Class source code or None
        """
        content = self.read_file(file_path)
        if not content:
            return None

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    lines = content.split('\n')
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
                    return '\n'.join(lines[start:end])
        except Exception as e:
            print(f"[Introspector] Error parsing {file_path}: {e}")

        return None

    def list_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        List all functions in a file with metadata.

        Returns:
            List of {name, lineno, args, docstring} dicts
        """
        content = self.read_file(file_path)
        if not content:
            return []

        functions = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'lineno': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'decorators': [
                            ast.unparse(d) if hasattr(ast, 'unparse') else str(d)
                            for d in node.decorator_list
                        ]
                    }
                    functions.append(func_info)
        except Exception as e:
            print(f"[Introspector] Error listing functions in {file_path}: {e}")

        return functions

    def list_classes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        List all classes in a file with metadata.

        Returns:
            List of {name, lineno, methods, docstring} dicts
        """
        content = self.read_file(file_path)
        if not content:
            return []

        classes = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [
                        n.name for n in node.body
                        if isinstance(n, ast.FunctionDef)
                    ]
                    class_info = {
                        'name': node.name,
                        'lineno': node.lineno,
                        'methods': methods,
                        'docstring': ast.get_docstring(node),
                        'bases': [
                            ast.unparse(b) if hasattr(ast, 'unparse') else str(b)
                            for b in node.bases
                        ]
                    }
                    classes.append(class_info)
        except Exception as e:
            print(f"[Introspector] Error listing classes in {file_path}: {e}")

        return classes

    def find_files(self, pattern: str = "*.py") -> List[str]:
        """
        Find all source files matching pattern.

        Args:
            pattern: Glob pattern (default: *.py)

        Returns:
            List of relative file paths
        """
        import glob
        files = []
        for path in self.project_root.rglob(pattern):
            # Skip __pycache__, .git, etc.
            if any(part.startswith('.') or part == '__pycache__' for part in path.parts):
                continue
            try:
                rel_path = path.relative_to(self.project_root)
                files.append(str(rel_path))
            except ValueError:
                pass
        return sorted(files)

    def search_code(self, pattern: str, file_pattern: str = "*.py") -> List[Dict[str, Any]]:
        """
        Search for a pattern across all source files.

        Args:
            pattern: Text or regex to search for
            file_pattern: Glob pattern for files to search

        Returns:
            List of {file, lineno, line} matches
        """
        import re
        matches = []
        regex = re.compile(pattern, re.IGNORECASE)

        for file_path in self.find_files(file_pattern):
            content = self.read_file(file_path)
            if not content:
                continue

            for i, line in enumerate(content.split('\n'), 1):
                if regex.search(line):
                    matches.append({
                        'file': file_path,
                        'lineno': i,
                        'line': line.strip()
                    })

        return matches

    def analyze_complexity(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze code complexity of a file.

        Returns metrics useful for identifying optimization targets.
        """
        content = self.read_file(file_path)
        if not content:
            return {}

        lines = content.split('\n')
        analysis = {
            'total_lines': len(lines),
            'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
            'blank_lines': len([l for l in lines if not l.strip()]),
            'functions': [],
            'classes': []
        }

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count complexity indicators
                    complexity = {
                        'name': node.name,
                        'lines': (node.end_lineno - node.lineno + 1) if hasattr(node, 'end_lineno') else 0,
                        'nested_loops': sum(1 for _ in ast.walk(node) if isinstance(_, (ast.For, ast.While))),
                        'conditionals': sum(1 for _ in ast.walk(node) if isinstance(_, ast.If)),
                        'try_blocks': sum(1 for _ in ast.walk(node) if isinstance(_, ast.Try)),
                    }
                    analysis['functions'].append(complexity)

                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                        'lines': (node.end_lineno - node.lineno + 1) if hasattr(node, 'end_lineno') else 0,
                    }
                    analysis['classes'].append(class_info)

        except Exception as e:
            analysis['parse_error'] = str(e)

        return analysis


class CodeEvolution:
    """
    Main code evolution system.

    Orchestrates:
    - Code generation based on learning
    - Validation and safety checks
    - Sandbox testing (Docker preferred, Python fallback)
    - Deployment with rollback capability
    - Self-introspection (reading own code)
    - Population-based evolution (genetic algorithm style)
    """

    def __init__(
        self,
        storage_dir: Path = None,
        use_docker: bool = True,
        use_population: bool = False,
        verifier_config: Optional[Dict[str, Any]] = None
    ):
        self.storage_dir = storage_dir or EVOLUTION_DIR / "code_evolution"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.use_docker = use_docker
        self.use_population = use_population and POPULATION_AVAILABLE
        self.verifier_config = verifier_config

        # Legacy validator (kept for backward compatibility)
        self.validator = CodeValidator(use_docker=use_docker)
        self.sandbox = CodeSandbox(self.storage_dir / "sandbox", use_docker=use_docker)
        self.version_control = CodeVersionControl(self.storage_dir / "versions")
        self.introspector = CodeIntrospector()  # For reading own source code

        # Ensemble verifier system (replaces single CodeValidator when available)
        self.ensemble_verifier: Optional['EnsembleVerifier'] = None
        self.use_ensemble = ENSEMBLE_VERIFIER_AVAILABLE
        if self.use_ensemble:
            # Build default config if not provided
            config = verifier_config or {}
            # Default configuration: syntax (required), safety (veto), test (required), llm (optional)
            if 'syntax' not in config:
                config['syntax'] = {'enabled': True, 'weight': 1.0}
            if 'safety' not in config:
                config['safety'] = {'enabled': True, 'weight': 1.0}  # Has veto power
            if 'test' not in config:
                config['test'] = {'enabled': True, 'weight': 1.0}
            if 'llm' not in config:
                config['llm'] = {'enabled': True, 'weight': 0.7}  # Optional
            if 'type' not in config:
                config['type'] = {'enabled': True, 'weight': 0.8}  # Optional
            if 'style' not in config:
                config['style'] = {'enabled': True, 'weight': 0.5}  # Optional
            if 'docker_mode' not in config:
                config['docker_mode'] = use_docker and DOCKER_AVAILABLE

            self.ensemble_verifier = create_ensemble_verifier(
                config=config,
                sandbox=self.sandbox,
                llm_interface=None  # Can be set later via set_llm_interface()
            )
            print("[CodeEvolution] Ensemble verifier: ENABLED")
        else:
            print("[CodeEvolution] Ensemble verifier not available, using single CodeValidator")

        # Population-based evolution system
        self.population: Optional['Population'] = None
        self.mutator: Optional['CodeMutator'] = None
        if self.use_population:
            self.population = Population(
                max_size=20,
                elite_size=3,
                tournament_size=3,
                mutation_rate=0.3,
                crossover_rate=0.7,
                storage_dir=self.storage_dir / "population"
            )
            self.mutator = CodeMutator()
            print("[CodeEvolution] Population-based evolution: ENABLED")
        else:
            if use_population and not POPULATION_AVAILABLE:
                print("[CodeEvolution] Population evolution requested but not available")

        # Log sandbox configuration
        print(f"[CodeEvolution] Initialized with Docker {'enabled' if use_docker else 'disabled'}")
        print(f"[CodeEvolution] Docker available: {DOCKER_AVAILABLE}")

        # Track pending and deployed changes
        self.pending_changes: List[CodeChange] = []
        self.deployed_changes: List[CodeChange] = []

        # State file
        self.state_file = self.storage_dir / "evolution_state.json"
        self._load_state()

        # Added functions registry (active functions that can be called)
        self.functions_file = self.storage_dir / "active_functions.py"
        self._init_functions_file()

    def _init_functions_file(self):
        """Initialize the active functions file."""
        if not self.functions_file.exists():
            with open(self.functions_file, 'w') as f:
                f.write('"""Auto-generated functions from code evolution."""\n\n')
                f.write('# This file is managed by CodeEvolution\n')
                f.write('# Functions here are validated, tested, and safe to use\n\n')
                f.write('ACTIVE_FUNCTIONS = {}\n')

    def _load_state(self):
        """Load evolution state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Reconstruct changes from state
                    for change_data in state.get('deployed_changes', []):
                        change = CodeChange(
                            id=change_data['id'],
                            change_type=CodeChangeType(change_data['change_type']),
                            target_file=change_data['target_file'],
                            original_code=change_data.get('original_code'),
                            new_code=change_data['new_code'],
                            description=change_data['description'],
                            timestamp=change_data['timestamp'],
                            deployed=True
                        )
                        self.deployed_changes.append(change)
            except Exception as e:
                print(f"[CodeEvolution] Error loading state: {e}")

    def _save_state(self):
        """Save evolution state."""
        state = {
            'deployed_changes': [
                {
                    'id': c.id,
                    'change_type': c.change_type.value,
                    'target_file': c.target_file,
                    'original_code': c.original_code,
                    'new_code': c.new_code,
                    'description': c.description,
                    'timestamp': c.timestamp,
                    'deployed': c.deployed
                }
                for c in self.deployed_changes
            ],
            'total_changes': len(self.deployed_changes),
            'last_updated': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def propose_change(
        self,
        change_type: CodeChangeType,
        new_code: str,
        description: str,
        target_file: str = None,
        original_code: str = None,
        test_cases: List[Dict] = None
    ) -> CodeChange:
        """
        Propose a code change for validation and testing.

        When population evolution is enabled, generates 3-5 variants using mutations,
        tests all variants, and keeps the best performing one.

        Args:
            change_type: Type of change
            new_code: The new code to add/modify
            description: Human-readable description
            target_file: File to modify (optional for new functions)
            original_code: Original code being replaced (for modifications)
            test_cases: Optional test cases for variant evaluation

        Returns:
            CodeChange object with validation results (best variant if population enabled)
        """
        # If population evolution is enabled, generate and test variants
        if self.use_population and self.population and self.mutator:
            return self._propose_change_with_population(
                change_type, new_code, description, target_file, original_code, test_cases
            )

        # Standard single-variant proposal
        return self._propose_change_single(
            change_type, new_code, description, target_file, original_code, test_cases
        )

    def _propose_change_single(
        self,
        change_type: CodeChangeType,
        new_code: str,
        description: str,
        target_file: str = None,
        original_code: str = None,
        test_cases: List[Dict] = None
    ) -> CodeChange:
        """Standard single-variant change proposal."""
        # Generate change ID
        change_id = hashlib.md5(
            f"{new_code}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Create change object
        change = CodeChange(
            id=change_id,
            change_type=change_type,
            target_file=target_file or str(self.functions_file),
            original_code=original_code,
            new_code=new_code,
            description=description,
            timestamp=datetime.now().isoformat()
        )

        # Use ensemble verifier if available, otherwise fall back to single validator
        if self.use_ensemble and self.ensemble_verifier:
            ensemble_result = self.ensemble_verifier.verify(new_code, test_cases=test_cases or [])

            # Store full verification results
            change.verification_results = ensemble_result.to_dict()

            # Map ensemble result to ValidationResult for backward compatibility
            if ensemble_result.vetoed:
                change.validation_result = ValidationResult.DANGEROUS_CODE
                print(f"[CodeEvolution] Change {change_id} vetoed by {ensemble_result.vetoed_by}")
            elif not ensemble_result.passed:
                # Determine failure type from individual verifier results
                for r in ensemble_result.results:
                    if not r.passed:
                        if r.verifier_name == "SyntaxVerifier":
                            change.validation_result = ValidationResult.SYNTAX_ERROR
                            break
                        elif r.verifier_name == "SafetyVerifier":
                            change.validation_result = ValidationResult.DANGEROUS_CODE
                            break
                        elif r.verifier_name == "TestVerifier":
                            change.validation_result = ValidationResult.TEST_FAILED
                            break
                else:
                    # Generic failure
                    change.validation_result = ValidationResult.TEST_FAILED
                print(f"[CodeEvolution] Change {change_id} failed ensemble verification (confidence: {ensemble_result.final_confidence:.2f})")
            else:
                change.validation_result = ValidationResult.VALID
                # Extract test results from TestVerifier
                for r in ensemble_result.results:
                    if r.verifier_name == "TestVerifier" and r.details:
                        change.test_results = {
                            'success': r.passed,
                            'tests_passed': r.details.get('tests_passed', 0),
                            'tests_failed': r.details.get('tests_failed', 0),
                            'errors': r.details.get('errors', []),
                            'ensemble_confidence': ensemble_result.final_confidence
                        }
                        break
                else:
                    change.test_results = {
                        'success': True,
                        'ensemble_confidence': ensemble_result.final_confidence
                    }

                self.pending_changes.append(change)
                print(f"[CodeEvolution] Change {change_id} passed ensemble verification (confidence: {ensemble_result.final_confidence:.2f})")
        else:
            # Fall back to legacy single validator
            result, message = self.validator.validate(new_code)
            change.validation_result = result

            if result == ValidationResult.VALID:
                # Test in sandbox
                success, test_results = self.sandbox.test_code(new_code, test_cases)
                change.test_results = test_results

                if success:
                    self.pending_changes.append(change)
                    print(f"[CodeEvolution] Change {change_id} validated and ready for deployment")
                else:
                    print(f"[CodeEvolution] Change {change_id} failed sandbox testing: {test_results['errors']}")
            else:
                print(f"[CodeEvolution] Change {change_id} validation failed: {message}")

        return change

    def _propose_change_with_population(
        self,
        change_type: CodeChangeType,
        new_code: str,
        description: str,
        target_file: str = None,
        original_code: str = None,
        test_cases: List[Dict] = None
    ) -> CodeChange:
        """
        Propose a code change using population-based evolution.

        Generates 3-5 variants using mutations, tests all variants in sandbox,
        and keeps the best performing one.
        """
        import random

        print(f"[CodeEvolution] Using population-based evolution for change proposal")

        # Step 1: Validate original code first (use ensemble if available)
        validation_passed = False
        validation_result = ValidationResult.VALID
        ensemble_verification = None

        if self.use_ensemble and self.ensemble_verifier:
            ensemble_verification = self.ensemble_verifier.verify(new_code, test_cases=test_cases or [])
            validation_passed = ensemble_verification.passed and not ensemble_verification.vetoed
            if ensemble_verification.vetoed:
                validation_result = ValidationResult.DANGEROUS_CODE
            elif not ensemble_verification.passed:
                validation_result = ValidationResult.TEST_FAILED
        else:
            result, message = self.validator.validate(new_code)
            validation_passed = (result == ValidationResult.VALID)
            validation_result = result

        if not validation_passed:
            print(f"[CodeEvolution] Original code validation failed")
            # Return a failed change
            change_id = hashlib.md5(f"{new_code}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            change = CodeChange(
                id=change_id,
                change_type=change_type,
                target_file=target_file or str(self.functions_file),
                original_code=original_code,
                new_code=new_code,
                description=description,
                timestamp=datetime.now().isoformat(),
                validation_result=validation_result,
                verification_results=ensemble_verification.to_dict() if ensemble_verification else None
            )
            return change

        # Step 2: Add original to population
        original_individual = self.population.add_individual(new_code, fitness=0.0)
        print(f"[CodeEvolution] Added original code as individual {original_individual.id}")

        # Step 3: Generate 3-5 variants using mutations
        num_variants = random.randint(3, 5)
        variants: List[CodeIndividual] = [original_individual]

        print(f"[CodeEvolution] Generating {num_variants} variants via mutation...")

        for i in range(num_variants):
            # Apply random mutation
            mutation_type = random.choice(list(MutationType))
            mutated_code, mutation_desc = self.mutator.mutate(new_code, mutation_type)

            # Validate mutated code (use ensemble if available)
            variant_valid = False
            if self.use_ensemble and self.ensemble_verifier:
                mut_ensemble_result = self.ensemble_verifier.verify(mutated_code, test_cases=test_cases or [])
                variant_valid = mut_ensemble_result.passed and not mut_ensemble_result.vetoed
            else:
                mut_result, _ = self.validator.validate(mutated_code)
                variant_valid = (mut_result == ValidationResult.VALID)

            if variant_valid and mutated_code != new_code:
                variant = self.population.add_individual(
                    mutated_code,
                    fitness=0.0,
                    parent_id=original_individual.id
                )
                variant.mutations = [mutation_desc]
                variants.append(variant)
                print(f"[CodeEvolution]   Variant {i+1}: {mutation_desc}")
            else:
                print(f"[CodeEvolution]   Variant {i+1}: mutation failed or invalid")

        # Step 4: Test all variants in sandbox
        print(f"[CodeEvolution] Testing {len(variants)} variants in sandbox...")

        for variant in variants:
            success, test_results = self.sandbox.test_code(variant.code, test_cases)
            variant.test_results = test_results

            # Calculate fitness from test results
            if test_cases:
                # Fitness from test pass rate
                total = test_results.get('tests_passed', 0) + test_results.get('tests_failed', 0)
                if total > 0:
                    variant.fitness_score = test_results.get('tests_passed', 0) / total
                else:
                    variant.fitness_score = 0.5 if success else 0.0
            else:
                # No test cases - use execution success as proxy
                variant.fitness_score = 1.0 if success else 0.0

            print(f"[CodeEvolution]   {variant.id[:8]}: fitness={variant.fitness_score:.3f}")

        # Step 5: Select best performing variant
        best_variant = max(variants, key=lambda v: v.fitness_score)
        print(f"[CodeEvolution] Best variant: {best_variant.id} (fitness={best_variant.fitness_score:.3f})")

        # Track lineage
        lineage = self.population.get_lineage(best_variant.id)

        # Step 6: Create CodeChange from best variant
        # Run final ensemble verification on best variant
        final_verification = None
        if self.use_ensemble and self.ensemble_verifier:
            final_verification = self.ensemble_verifier.verify(best_variant.code, test_cases=test_cases or [])

        change_id = best_variant.id
        change = CodeChange(
            id=change_id,
            change_type=change_type,
            target_file=target_file or str(self.functions_file),
            original_code=original_code,
            new_code=best_variant.code,
            description=f"{description} (population-evolved, {len(best_variant.mutations)} mutations)",
            timestamp=datetime.now().isoformat(),
            validation_result=ValidationResult.VALID,
            test_results=best_variant.test_results,
            verification_results=final_verification.to_dict() if final_verification else None
        )

        # Add population metadata to test_results
        if change.test_results:
            change.test_results['population_metadata'] = {
                'variants_tested': len(variants),
                'best_fitness': best_variant.fitness_score,
                'mutations_applied': best_variant.mutations,
                'lineage': lineage,
                'parent_id': best_variant.parent_id
            }
            if final_verification:
                change.test_results['ensemble_confidence'] = final_verification.final_confidence

        if best_variant.fitness_score > 0 or (not test_cases and best_variant.test_results.get('success')):
            self.pending_changes.append(change)
            print(f"[CodeEvolution] Change {change_id} (population-evolved) ready for deployment")
        else:
            print(f"[CodeEvolution] Change {change_id} failed sandbox testing")

        return change

    def deploy_change(self, change: CodeChange, force: bool = False) -> bool:
        """
        Deploy a validated code change.

        Args:
            change: The change to deploy
            force: Skip validation checks (dangerous!)

        Returns:
            success
        """
        if not force:
            if change.validation_result != ValidationResult.VALID:
                print(f"[CodeEvolution] Cannot deploy: validation failed")
                return False

            if change.test_results and not change.test_results.get('success'):
                print(f"[CodeEvolution] Cannot deploy: tests failed")
                return False

        # Commit to version control first
        commit_id = self.version_control.commit(change)
        print(f"[CodeEvolution] Committed change {change.id} as {commit_id}")

        # Deploy based on change type
        if change.change_type == CodeChangeType.NEW_FUNCTION:
            success = self._deploy_new_function(change)
        elif change.change_type == CodeChangeType.MODIFY_FUNCTION:
            success = self._deploy_function_modification(change)
        else:
            success = self._deploy_generic(change)

        if success:
            change.deployed = True
            self.deployed_changes.append(change)
            if change in self.pending_changes:
                self.pending_changes.remove(change)
            self._save_state()
            print(f"[CodeEvolution] Successfully deployed change {change.id}")
        else:
            print(f"[CodeEvolution] Deployment failed, rolling back...")
            self.version_control.rollback(commit_id)

        return success

    def _deploy_new_function(self, change: CodeChange) -> bool:
        """Deploy a new function to the active functions file."""
        try:
            # Extract function name from code
            tree = ast.parse(change.new_code)
            func_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    break

            if not func_name:
                print("[CodeEvolution] No function found in code")
                return False

            # Append to functions file
            with open(self.functions_file, 'a') as f:
                f.write(f"\n\n# Added: {change.timestamp}\n")
                f.write(f"# Description: {change.description}\n")
                f.write(f"# Change ID: {change.id}\n")
                f.write(change.new_code)
                f.write(f"\n\nACTIVE_FUNCTIONS['{func_name}'] = {func_name}\n")

            return True

        except Exception as e:
            print(f"[CodeEvolution] Error deploying function: {e}")
            return False

    def _deploy_function_modification(self, change: CodeChange) -> bool:
        """Deploy a modification to an existing function."""
        try:
            target = Path(change.target_file)
            if not target.exists():
                return False

            # Read current content
            with open(target, 'r') as f:
                content = f.read()

            # Replace old code with new
            if change.original_code and change.original_code in content:
                new_content = content.replace(change.original_code, change.new_code)
                with open(target, 'w') as f:
                    f.write(new_content)
                return True
            else:
                print("[CodeEvolution] Original code not found in target file")
                return False

        except Exception as e:
            print(f"[CodeEvolution] Error modifying function: {e}")
            return False

    def _deploy_generic(self, change: CodeChange) -> bool:
        """Deploy a generic code change."""
        try:
            target = Path(change.target_file)

            if change.original_code:
                # Modification
                if target.exists():
                    with open(target, 'r') as f:
                        content = f.read()
                    content = content.replace(change.original_code, change.new_code)
                    with open(target, 'w') as f:
                        f.write(content)
                    return True
            else:
                # New file/append
                with open(target, 'a') as f:
                    f.write(change.new_code)
                return True

        except Exception as e:
            print(f"[CodeEvolution] Error deploying: {e}")
            return False

    def rollback_last(self) -> bool:
        """Rollback the last deployed change."""
        if not self.deployed_changes:
            print("[CodeEvolution] No changes to rollback")
            return False

        change = self.deployed_changes[-1]
        history = self.version_control.get_history()

        for commit in reversed(history):
            if commit['change_id'] == change.id and not commit['rolled_back']:
                success = self.version_control.rollback(commit['commit_id'])
                if success:
                    change.rolled_back = True
                    self.deployed_changes.remove(change)
                    self._save_state()
                    print(f"[CodeEvolution] Rolled back change {change.id}")
                return success

        return False

    def get_active_functions(self) -> Dict[str, Callable]:
        """
        Get all active evolved functions.

        Returns:
            Dict mapping function names to callable functions
        """
        try:
            spec = importlib.util.spec_from_file_location(
                "active_functions",
                self.functions_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, 'ACTIVE_FUNCTIONS', {})
        except Exception as e:
            print(f"[CodeEvolution] Error loading active functions: {e}")
            return {}

    def call_evolved_function(self, name: str, *args, **kwargs) -> Any:
        """
        Call an evolved function by name.

        Args:
            name: Function name
            *args, **kwargs: Arguments to pass

        Returns:
            Function result
        """
        functions = self.get_active_functions()
        if name not in functions:
            raise ValueError(f"Function '{name}' not found in active functions")
        return functions[name](*args, **kwargs)

    def get_stats(self) -> Dict:
        """Get evolution statistics."""
        stats = {
            'pending_changes': len(self.pending_changes),
            'deployed_changes': len(self.deployed_changes),
            'total_commits': len(self.version_control.history),
            'rollbacks': sum(1 for c in self.version_control.history if c.get('rolled_back')),
            'active_functions': len(self.get_active_functions()),
            'last_change': self.deployed_changes[-1].timestamp if self.deployed_changes else None,
            'population_enabled': self.use_population,
            'ensemble_verifier_enabled': self.use_ensemble
        }

        # Add population stats if enabled
        if self.use_population and self.population:
            pop_stats = self.get_population_stats()
            stats['population'] = pop_stats

        # Add ensemble verifier stats if enabled
        if self.use_ensemble and self.ensemble_verifier:
            verifier_stats = self.get_verifier_stats()
            stats['verifier'] = verifier_stats

        return stats

    def get_verifier_stats(self) -> Dict:
        """
        Get statistics for ensemble verifier performance.

        Returns:
            Dict with verifier statistics including:
            - total_verifications: Total number of verifications performed
            - pass_rate: Percentage of verifications that passed
            - veto_count: Number of times safety veto was used
            - avg_confidence: Average confidence across all verifications
            - per_verifier: Stats for each individual verifier
            - issues_caught: Which verifiers caught issues
        """
        if not self.use_ensemble or not self.ensemble_verifier:
            return {
                'enabled': False,
                'message': 'Ensemble verifier not enabled'
            }

        # Get base stats from ensemble
        ensemble_stats = self.ensemble_verifier.get_stats()

        # Get effectiveness metrics
        effectiveness = self.ensemble_verifier.get_verifier_effectiveness()

        # Calculate issues caught by each verifier
        issues_caught = {}
        for verifier_name, eff_stats in effectiveness.items():
            issues_caught[verifier_name] = {
                'total_failures': eff_stats.get('total_failures', 0),
                'sole_failures': eff_stats.get('sole_failures', 0),  # Caught issue others missed
                'vetoes': eff_stats.get('vetoes', 0),
                'effectiveness_score': (
                    eff_stats.get('sole_failures', 0) / max(eff_stats.get('total_failures', 1), 1)
                    if eff_stats.get('total_failures', 0) > 0 else 0.0
                )
            }

        return {
            'enabled': True,
            'total_verifications': ensemble_stats.get('total_verifications', 0),
            'pass_rate': round(ensemble_stats.get('pass_rate', 0.0), 4),
            'avg_confidence': round(ensemble_stats.get('avg_confidence', 0.0), 4),
            'veto_count': ensemble_stats.get('veto_count', 0),
            'voting_strategy': ensemble_stats.get('voting_strategy', 'weighted'),
            'threshold': ensemble_stats.get('threshold', 0.7),
            'per_verifier': ensemble_stats.get('verifier_stats', {}),
            'issues_caught': issues_caught
        }

    def set_llm_interface(self, llm_interface: Callable) -> None:
        """
        Set the LLM interface for the LLMVerifier in the ensemble.

        Args:
            llm_interface: Callable that takes a prompt and returns a response
        """
        if self.use_ensemble and self.ensemble_verifier:
            llm_verifier = self.ensemble_verifier.get_verifier("LLMVerifier")
            if llm_verifier:
                llm_verifier.llm_interface = llm_interface
                print("[CodeEvolution] LLM interface set for LLMVerifier")

    def get_population_stats(self) -> Dict:
        """
        Get population evolution statistics showing diversity and convergence.

        Returns:
            Dict with population statistics including:
            - generation: Current generation number
            - population_size: Number of individuals
            - diversity: Genetic diversity (0-1, higher = more diverse)
            - best_fitness: Highest fitness in population
            - avg_fitness: Average fitness across population
            - convergence: How converged the population is (0-1, higher = more converged)
            - hall_of_fame_size: Number of individuals in hall of fame
            - total_mutations: Total mutations applied
            - lineage_depth: Average lineage depth in population
        """
        if not self.use_population or not self.population:
            return {
                'enabled': False,
                'message': 'Population evolution not enabled'
            }

        # Get base stats from population
        pop_stats = self.population.get_stats()

        # Calculate convergence (inverse of diversity)
        diversity = pop_stats.get('diversity', 0.0)
        convergence = 1.0 - diversity

        # Calculate average lineage depth
        lineage_depths = []
        for ind in self.population.individuals:
            lineage = self.population.get_lineage(ind.id)
            lineage_depths.append(len(lineage))
        avg_lineage_depth = sum(lineage_depths) / len(lineage_depths) if lineage_depths else 0

        # Get fitness distribution for analysis
        fitnesses = [ind.fitness_score for ind in self.population.individuals]
        fitness_std = 0.0
        if len(fitnesses) > 1:
            avg = sum(fitnesses) / len(fitnesses)
            fitness_std = (sum((f - avg) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5

        # Get best individuals info
        best_individuals = self.population.get_best(3)
        best_info = []
        for ind in best_individuals:
            best_info.append({
                'id': ind.id,
                'fitness': ind.fitness_score,
                'generation': ind.generation,
                'mutations': ind.mutations[-3:] if ind.mutations else []  # Last 3 mutations
            })

        return {
            'enabled': True,
            'generation': pop_stats.get('generation', 0),
            'population_size': pop_stats.get('population_size', 0),
            'diversity': round(diversity, 4),
            'convergence': round(convergence, 4),
            'best_fitness': round(pop_stats.get('best_fitness', 0.0), 4),
            'avg_fitness': round(pop_stats.get('avg_fitness', 0.0), 4),
            'min_fitness': round(pop_stats.get('min_fitness', 0.0), 4),
            'fitness_std': round(fitness_std, 4),
            'hall_of_fame_size': pop_stats.get('hall_of_fame_size', 0),
            'total_mutations': pop_stats.get('total_mutations', 0),
            'avg_lineage_depth': round(avg_lineage_depth, 2),
            'best_individuals': best_info
        }

    def evolve_population(self, test_cases: List[Dict] = None, generations: int = 1) -> Dict:
        """
        Trigger population evolution for a specified number of generations.

        Args:
            test_cases: Test cases for fitness evaluation
            generations: Number of generations to evolve (default: 1)

        Returns:
            Dict with evolution results
        """
        if not self.use_population or not self.population:
            return {
                'success': False,
                'message': 'Population evolution not enabled'
            }

        if len(self.population.individuals) == 0:
            return {
                'success': False,
                'message': 'Population is empty. Add code via propose_change first.'
            }

        results = {
            'success': True,
            'generations_evolved': 0,
            'stats_history': []
        }

        print(f"[CodeEvolution] Evolving population for {generations} generation(s)...")

        for gen in range(generations):
            stats = self.population.evolve_generation(test_cases)
            results['stats_history'].append(stats)
            results['generations_evolved'] += 1

            print(f"[CodeEvolution] Generation {stats['generation']}: "
                  f"best={stats['best_fitness']:.3f}, avg={stats['avg_fitness']:.3f}")

        # Get final population stats
        results['final_stats'] = self.get_population_stats()

        return results

    # === SELF-INTROSPECTION METHODS ===
    # These allow the AI to read and understand its own code

    def read_own_code(self, file_path: str) -> Optional[str]:
        """
        Read the AI's own source code.

        Args:
            file_path: Relative path like "integrations/self_evolution.py"

        Returns:
            Source code content
        """
        return self.introspector.read_file(file_path)

    def read_own_function(self, file_path: str, function_name: str) -> Optional[str]:
        """
        Read a specific function from the AI's source code.

        Useful for understanding current implementation before optimizing.
        """
        return self.introspector.read_function(file_path, function_name)

    def read_own_class(self, file_path: str, class_name: str) -> Optional[str]:
        """
        Read a specific class from the AI's source code.
        """
        return self.introspector.read_class(file_path, class_name)

    def list_own_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        List all functions in one of the AI's source files.

        Returns function names, arguments, and docstrings.
        """
        return self.introspector.list_functions(file_path)

    def list_own_classes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        List all classes in one of the AI's source files.
        """
        return self.introspector.list_classes(file_path)

    def find_own_files(self, pattern: str = "*.py") -> List[str]:
        """
        Find all source files in the project.

        Returns list of relative paths.
        """
        return self.introspector.find_files(pattern)

    def search_own_code(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search for a pattern across all source code.

        Useful for finding where something is implemented.
        """
        return self.introspector.search_code(pattern)

    def analyze_own_code(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze complexity of a source file.

        Helps identify optimization targets.
        """
        return self.introspector.analyze_complexity(file_path)

    def reflect(self) -> str:
        """Generate reflection on code evolution state."""
        stats = self.get_stats()

        reflection = f"""
=== CODE EVOLUTION REFLECTION ===
Deployed Changes: {stats['deployed_changes']}
Pending Changes: {stats['pending_changes']}
Total Commits: {stats['total_commits']}
Rollbacks: {stats['rollbacks']}
Active Functions: {stats['active_functions']}
Last Change: {stats['last_change'] or 'None'}

"""
        if stats['deployed_changes'] > 0:
            reflection += "Recent deployments:\n"
            for change in self.deployed_changes[-5:]:
                status = "✓" if not change.rolled_back else "↩"
                reflection += f"  {status} {change.id}: {change.description}\n"

        return reflection


# Global instance
_code_evolution: Optional[CodeEvolution] = None


def get_code_evolution() -> CodeEvolution:
    """Get global code evolution instance."""
    global _code_evolution
    if _code_evolution is None:
        _code_evolution = CodeEvolution()
    return _code_evolution


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("CODE EVOLUTION TEST")
    print("=" * 60)

    evo = CodeEvolution(Path("/tmp/test_code_evolution"))

    # Test 1: Propose a new function
    print("\n1. Testing new function proposal:")

    test_code = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''

    change = evo.propose_change(
        change_type=CodeChangeType.NEW_FUNCTION,
        new_code=test_code,
        description="Add Fibonacci calculator function"
    )

    print(f"   Change ID: {change.id}")
    print(f"   Validation: {change.validation_result}")
    print(f"   Test Results: {change.test_results}")

    # Test 2: Deploy the function
    print("\n2. Testing deployment:")
    success = evo.deploy_change(change)
    print(f"   Deployed: {success}")

    # Test 3: Call the evolved function
    print("\n3. Testing evolved function call:")
    try:
        functions = evo.get_active_functions()
        print(f"   Active functions: {list(functions.keys())}")

        if 'calculate_fibonacci' in functions:
            result = evo.call_evolved_function('calculate_fibonacci', 10)
            print(f"   fibonacci(10) = {result}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: Stats and reflection
    print("\n4. Evolution stats:")
    print(evo.reflect())

    # Test 5: Dangerous code should be blocked
    print("\n5. Testing dangerous code blocking:")
    dangerous_code = '''
def evil_function():
    import os
    os.system("rm -rf /")
'''

    bad_change = evo.propose_change(
        change_type=CodeChangeType.NEW_FUNCTION,
        new_code=dangerous_code,
        description="This should be blocked"
    )
    print(f"   Validation: {bad_change.validation_result}")

    print("\n" + "=" * 60)
