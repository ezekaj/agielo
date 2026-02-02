#!/usr/bin/env python3
"""
Health check script for Human Cognition AI system.
Validates all dependencies, services, and data directories are properly configured.
"""

import sys
import os
from pathlib import Path


def check_python_imports():
    """Check if all required Python packages are importable."""
    print("Checking Python imports...")
    required_packages = [
        ("numpy", "numpy"),
        ("chromadb", "chromadb"),
        ("json", "json (stdlib)"),
        ("urllib.request", "urllib (stdlib)"),
    ]

    all_ok = True
    for module_name, display_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name} - MISSING: {e}")
            all_ok = False

    return all_ok


def check_lm_studio():
    """Check if LM Studio is running and accessible."""
    print("\nChecking LM Studio connectivity...")
    import urllib.request
    import urllib.error

    url = "http://localhost:1234/v1/models"
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print(f"  ✓ LM Studio is running at http://localhost:1234/v1")
                return True
            else:
                print(f"  ✗ LM Studio returned unexpected status: {response.status}")
                return False
    except urllib.error.URLError as e:
        print(f"  ✗ LM Studio not reachable: {e.reason}")
        print("    Hint: Start LM Studio and load a model, ensure server is running on port 1234")
        return False
    except Exception as e:
        print(f"  ✗ LM Studio check failed: {e}")
        return False


def check_chromadb():
    """Check if ChromaDB can initialize properly."""
    print("\nChecking ChromaDB initialization...")
    try:
        import chromadb
        # Test ephemeral client creation
        client = chromadb.Client()
        # Try creating a test collection
        test_collection = client.create_collection(name="health_check_test")
        client.delete_collection(name="health_check_test")
        print("  ✓ ChromaDB initializes correctly")
        return True
    except Exception as e:
        print(f"  ✗ ChromaDB initialization failed: {e}")
        return False


def check_knowledge_directory():
    """Check if the cognitive AI knowledge directory exists with correct permissions."""
    print("\nChecking knowledge directory...")
    knowledge_dir = Path.home() / ".cognitive_ai_knowledge"

    if not knowledge_dir.exists():
        print(f"  ✗ Directory does not exist: {knowledge_dir}")
        print(f"    Hint: Create it with: mkdir -p {knowledge_dir}")
        return False

    if not knowledge_dir.is_dir():
        print(f"  ✗ Path exists but is not a directory: {knowledge_dir}")
        return False

    # Check if we can read and write
    test_file = knowledge_dir / ".health_check_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print(f"  ✓ Directory exists with read/write permissions: {knowledge_dir}")
    except PermissionError:
        print(f"  ✗ Directory exists but lacks write permissions: {knowledge_dir}")
        return False
    except Exception as e:
        print(f"  ✗ Permission check failed: {e}")
        return False

    # Check for expected subdirectories/files
    expected_items = ["training_data.jsonl", "facts.json", "evolution"]
    found_items = []
    missing_items = []

    for item in expected_items:
        if (knowledge_dir / item).exists():
            found_items.append(item)
        else:
            missing_items.append(item)

    if found_items:
        print(f"  ✓ Found data files: {', '.join(found_items)}")
    if missing_items:
        print(f"  ⚠ Missing (may be created later): {', '.join(missing_items)}")

    return True


def main():
    """Run all health checks and report overall status."""
    print("=" * 60)
    print("Human Cognition AI - System Health Check")
    print("=" * 60)

    results = {
        "Python imports": check_python_imports(),
        "LM Studio": check_lm_studio(),
        "ChromaDB": check_chromadb(),
        "Knowledge directory": check_knowledge_directory(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All health checks passed! System is ready.")
        return 0
    else:
        print("Some health checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
