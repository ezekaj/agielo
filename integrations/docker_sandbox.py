"""
Docker Sandbox System
=====================

Provides true Docker-based isolation for safe code execution.
This is a HIGH PRIORITY security improvement replacing the Python-based sandbox.

Features:
1. Full container isolation (no host access)
2. Memory limits (512MB default)
3. No network access
4. Execution timeout (30s default)
5. Auto-cleanup of containers
6. Graceful fallback when Docker is unavailable
"""

import os
import sys
import json
import time
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import EVOLUTION_DIR

# Try to import docker, set availability flag
try:
    import docker
    from docker.errors import DockerException, ContainerError, ImageNotFound, APIError
    DOCKER_PACKAGE_AVAILABLE = True
except ImportError:
    DOCKER_PACKAGE_AVAILABLE = False
    docker = None
    DockerException = Exception
    ContainerError = Exception
    ImageNotFound = Exception
    APIError = Exception


@dataclass
class ExecutionResult:
    """Result of code execution in Docker sandbox."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    memory_usage: int  # bytes
    tests_passed: int
    tests_failed: int
    errors: List[str]
    sandbox_type: str  # 'docker' or 'python'


class DockerSandbox:
    """
    Docker-based sandbox for safe code execution.

    Provides true isolation with:
    - Memory limits
    - No network access
    - Execution timeout
    - Auto-cleanup
    """

    # Default container configuration
    DEFAULT_IMAGE = "python:3.11-slim"
    SANDBOX_IMAGE = "cognitive-sandbox:latest"
    DEFAULT_MEMORY_LIMIT = "512m"
    DEFAULT_TIMEOUT = 30  # seconds

    def __init__(
        self,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
        network_disabled: bool = True,
        base_dir: Path = None
    ):
        """
        Initialize Docker sandbox.

        Args:
            memory_limit: Memory limit (e.g., "512m", "1g")
            timeout: Execution timeout in seconds
            network_disabled: Whether to disable network access
            base_dir: Base directory for sandbox files
        """
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.network_disabled = network_disabled
        self.base_dir = base_dir or EVOLUTION_DIR / "docker_sandbox"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Docker client - will be None if Docker unavailable
        self.client: Optional['docker.DockerClient'] = None
        self.docker_available = False

        # Execution history
        self.execution_history: List[ExecutionResult] = []

        # Try to initialize Docker client
        self._init_docker_client()

    def _init_docker_client(self):
        """Initialize Docker client if available."""
        if not DOCKER_PACKAGE_AVAILABLE:
            print("[DockerSandbox] docker package not installed. Install with: pip install docker")
            return

        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            self.docker_available = True
            print("[DockerSandbox] Docker connection established")

            # Ensure sandbox image exists
            self._ensure_sandbox_image()

        except DockerException as e:
            print(f"[DockerSandbox] Docker not available: {e}")
            self.docker_available = False
        except Exception as e:
            print(f"[DockerSandbox] Error initializing Docker: {e}")
            self.docker_available = False

    def _ensure_sandbox_image(self):
        """Ensure the sandbox Docker image exists, build if necessary."""
        if not self.docker_available:
            return

        try:
            self.client.images.get(self.SANDBOX_IMAGE)
            print(f"[DockerSandbox] Using existing image: {self.SANDBOX_IMAGE}")
        except ImageNotFound:
            print(f"[DockerSandbox] Building sandbox image...")
            self._build_sandbox_image()
        except Exception as e:
            print(f"[DockerSandbox] Error checking image: {e}")
            # Fall back to default Python image

    def _build_sandbox_image(self):
        """Build the sandbox Docker image from Dockerfile."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile.sandbox"

        if not dockerfile_path.exists():
            print(f"[DockerSandbox] Dockerfile.sandbox not found, using default image")
            return

        try:
            self.client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile="Dockerfile.sandbox",
                tag=self.SANDBOX_IMAGE,
                rm=True
            )
            print(f"[DockerSandbox] Built sandbox image: {self.SANDBOX_IMAGE}")
        except Exception as e:
            print(f"[DockerSandbox] Error building image: {e}")

    def is_available(self) -> bool:
        """Check if Docker sandbox is available."""
        return self.docker_available

    def execute_code(
        self,
        code: str,
        test_cases: List[Dict] = None
    ) -> Tuple[bool, Dict]:
        """
        Execute code in Docker container.

        Args:
            code: Python code to execute
            test_cases: List of {function, args, kwargs, expected} dicts

        Returns:
            (success, results_dict)
        """
        if not self.docker_available:
            return self._execute_fallback(code, test_cases)

        start_time = time.time()
        result = ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            exit_code=-1,
            execution_time=0,
            memory_usage=0,
            tests_passed=0,
            tests_failed=0,
            errors=[],
            sandbox_type="docker"
        )

        container = None
        temp_dir = None

        try:
            # Create temp directory for code
            temp_dir = tempfile.mkdtemp(dir=str(self.base_dir))
            code_file = Path(temp_dir) / "code.py"

            # Write code with test runner
            full_code = self._wrap_code_with_tests(code, test_cases)
            with open(code_file, 'w') as f:
                f.write(full_code)

            # Select image (sandbox if available, otherwise default)
            image = self.SANDBOX_IMAGE
            try:
                self.client.images.get(image)
            except ImageNotFound:
                image = self.DEFAULT_IMAGE

            # Create and run container
            container = self.client.containers.run(
                image,
                command=["python", "/sandbox/code.py"],
                volumes={temp_dir: {'bind': '/sandbox', 'mode': 'ro'}},
                mem_limit=self.memory_limit,
                network_disabled=self.network_disabled,
                detach=True,
                remove=False,  # We'll remove manually after getting logs
                stdout=True,
                stderr=True
            )

            # Wait for completion with timeout
            try:
                exit_result = container.wait(timeout=self.timeout)
                result.exit_code = exit_result.get('StatusCode', -1)
            except Exception as e:
                result.errors.append(f"Timeout after {self.timeout}s")
                result.exit_code = -1
                # Kill container on timeout
                try:
                    container.kill()
                except (DockerException, APIError) as e:
                    print(f"[DockerSandbox] Warning: Could not kill container: {e}")

            # Get logs
            result.stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
            result.stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')

            # Parse test results from stdout
            test_results = self._parse_test_results(result.stdout)
            result.tests_passed = test_results.get('passed', 0)
            result.tests_failed = test_results.get('failed', 0)
            if test_results.get('errors'):
                result.errors.extend(test_results['errors'])

            # Get container stats for memory usage
            try:
                stats = container.stats(stream=False)
                result.memory_usage = stats.get('memory_stats', {}).get('max_usage', 0)
            except (DockerException, APIError, KeyError) as e:
                print(f"[DockerSandbox] Warning: Could not retrieve container stats: {e}")

            result.success = result.exit_code == 0 and result.tests_failed == 0

        except ContainerError as e:
            result.stderr = str(e)
            result.errors.append(f"Container error: {e}")
        except APIError as e:
            result.errors.append(f"Docker API error: {e}")
        except Exception as e:
            result.errors.append(f"Execution error: {str(e)}")
            traceback.print_exc()
        finally:
            # Cleanup container
            if container:
                try:
                    container.remove(force=True)
                except (DockerException, APIError) as e:
                    print(f"[DockerSandbox] Warning: Could not remove container: {e}")

            # Cleanup temp directory
            if temp_dir:
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except (OSError, IOError) as e:
                    print(f"[DockerSandbox] Warning: Could not cleanup temp directory: {e}")

            result.execution_time = time.time() - start_time

        # Record in history
        self.execution_history.append(result)

        return result.success, self._result_to_dict(result)

    def _wrap_code_with_tests(self, code: str, test_cases: List[Dict] = None) -> str:
        """Wrap user code with test runner."""
        wrapped = f'''
import sys
import json
import traceback

# User code
{code}

# Test runner
def run_tests():
    results = {{"passed": 0, "failed": 0, "errors": []}}
    test_cases = {json.dumps(test_cases or [])}

    for i, test in enumerate(test_cases):
        try:
            func_name = test.get('function', 'main')
            args = test.get('args', [])
            kwargs = test.get('kwargs', {{}})
            expected = test.get('expected')

            # Get function
            if func_name not in globals():
                results["errors"].append(f"Test {{i+1}}: Function '{{func_name}}' not found")
                results["failed"] += 1
                continue

            func = globals()[func_name]
            result = func(*args, **kwargs)

            if expected is not None:
                if result == expected:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Test {{i+1}}: Expected {{expected}}, got {{result}}")
            else:
                results["passed"] += 1

        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Test {{i+1}}: {{str(e)}}")

    # Output results as JSON
    print("__TEST_RESULTS__")
    print(json.dumps(results))
    print("__END_RESULTS__")

    return results

if __name__ == "__main__":
    results = run_tests()
    sys.exit(0 if results["failed"] == 0 else 1)
'''
        return wrapped

    def _parse_test_results(self, stdout: str) -> Dict:
        """Parse test results from stdout."""
        try:
            # Find test results JSON
            start_marker = "__TEST_RESULTS__"
            end_marker = "__END_RESULTS__"

            if start_marker in stdout and end_marker in stdout:
                start = stdout.index(start_marker) + len(start_marker)
                end = stdout.index(end_marker)
                json_str = stdout[start:end].strip()
                return json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[DockerSandbox] Warning: Could not parse test results: {e}")

        return {"passed": 0, "failed": 0, "errors": ["Could not parse test results"]}

    def _execute_fallback(
        self,
        code: str,
        test_cases: List[Dict] = None
    ) -> Tuple[bool, Dict]:
        """
        Fallback execution using Python subprocess (less secure).
        Used when Docker is not available.
        """
        import subprocess

        start_time = time.time()
        result = ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            exit_code=-1,
            execution_time=0,
            memory_usage=0,
            tests_passed=0,
            tests_failed=0,
            errors=[],
            sandbox_type="python"
        )

        temp_dir = None

        try:
            # Create temp directory for code
            temp_dir = tempfile.mkdtemp(dir=str(self.base_dir))
            code_file = Path(temp_dir) / "code.py"

            # Write code with test runner
            full_code = self._wrap_code_with_tests(code, test_cases)
            with open(code_file, 'w') as f:
                f.write(full_code)

            # Execute with subprocess and timeout
            proc = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                timeout=self.timeout,
                cwd=temp_dir
            )

            result.exit_code = proc.returncode
            result.stdout = proc.stdout.decode('utf-8', errors='replace')
            result.stderr = proc.stderr.decode('utf-8', errors='replace')

            # Parse test results
            test_results = self._parse_test_results(result.stdout)
            result.tests_passed = test_results.get('passed', 0)
            result.tests_failed = test_results.get('failed', 0)
            if test_results.get('errors'):
                result.errors.extend(test_results['errors'])

            result.success = result.exit_code == 0 and result.tests_failed == 0

        except subprocess.TimeoutExpired:
            result.errors.append(f"Timeout after {self.timeout}s")
        except Exception as e:
            result.errors.append(f"Execution error: {str(e)}")
        finally:
            # Cleanup temp directory
            if temp_dir:
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except (OSError, IOError) as e:
                    print(f"[DockerSandbox] Warning: Could not cleanup temp directory: {e}")

            result.execution_time = time.time() - start_time

        # Record in history
        self.execution_history.append(result)

        return result.success, self._result_to_dict(result)

    def _result_to_dict(self, result: ExecutionResult) -> Dict:
        """Convert ExecutionResult to dict."""
        return {
            'success': result.success,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.exit_code,
            'execution_time': result.execution_time,
            'memory_usage': result.memory_usage,
            'tests_passed': result.tests_passed,
            'tests_failed': result.tests_failed,
            'errors': result.errors,
            'sandbox_type': result.sandbox_type
        }

    def get_stats(self) -> Dict:
        """Get sandbox execution statistics."""
        if not self.execution_history:
            return {
                'total_executions': 0,
                'docker_executions': 0,
                'python_executions': 0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'docker_available': self.docker_available
            }

        docker_execs = [r for r in self.execution_history if r.sandbox_type == 'docker']
        python_execs = [r for r in self.execution_history if r.sandbox_type == 'python']
        successful = [r for r in self.execution_history if r.success]

        return {
            'total_executions': len(self.execution_history),
            'docker_executions': len(docker_execs),
            'python_executions': len(python_execs),
            'success_rate': len(successful) / len(self.execution_history),
            'avg_execution_time': sum(r.execution_time for r in self.execution_history) / len(self.execution_history),
            'docker_available': self.docker_available
        }

    def cleanup_orphan_containers(self) -> int:
        """
        Clean up any orphaned sandbox containers.

        Returns:
            Number of containers cleaned up
        """
        if not self.docker_available:
            return 0

        cleaned = 0
        try:
            containers = self.client.containers.list(
                all=True,
                filters={'ancestor': self.SANDBOX_IMAGE}
            )

            for container in containers:
                try:
                    container.remove(force=True)
                    cleaned += 1
                except (DockerException, APIError) as e:
                    print(f"[DockerSandbox] Warning: Could not remove orphan container: {e}")

        except (DockerException, APIError) as e:
            print(f"[DockerSandbox] Error cleaning up containers: {e}")

        return cleaned


# Global instance
_docker_sandbox: Optional[DockerSandbox] = None


def get_docker_sandbox() -> DockerSandbox:
    """Get global Docker sandbox instance."""
    global _docker_sandbox
    if _docker_sandbox is None:
        _docker_sandbox = DockerSandbox()
    return _docker_sandbox


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("DOCKER SANDBOX TEST")
    print("=" * 60)

    sandbox = DockerSandbox()

    print(f"\nDocker available: {sandbox.is_available()}")
    print(f"Memory limit: {sandbox.memory_limit}")
    print(f"Timeout: {sandbox.timeout}s")
    print(f"Network disabled: {sandbox.network_disabled}")

    # Test 1: Simple code execution
    print("\n1. Testing simple code execution:")
    code = '''
def add(a, b):
    return a + b
'''
    test_cases = [
        {'function': 'add', 'args': [2, 3], 'expected': 5},
        {'function': 'add', 'args': [10, 20], 'expected': 30},
    ]

    success, results = sandbox.execute_code(code, test_cases)
    print(f"   Success: {success}")
    print(f"   Sandbox type: {results['sandbox_type']}")
    print(f"   Tests passed: {results['tests_passed']}")
    print(f"   Tests failed: {results['tests_failed']}")
    print(f"   Execution time: {results['execution_time']:.3f}s")

    # Test 2: Code with error
    print("\n2. Testing code with error:")
    bad_code = '''
def broken():
    return 1 / 0
'''
    test_cases_bad = [{'function': 'broken'}]

    success, results = sandbox.execute_code(bad_code, test_cases_bad)
    print(f"   Success: {success}")
    print(f"   Errors: {results['errors'][:2]}")

    # Test 3: Stats
    print("\n3. Sandbox stats:")
    stats = sandbox.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 60)
