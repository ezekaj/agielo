# Phase 07: Code Excellence - Comprehensive Improvement Plan

## Overview
A multi-phase plan to fix all bugs, improve performance, enhance code quality, and complete missing features in the Human Cognition AI system.

**Total Issues Identified:** 20+ major issues across 40+ files
**Estimated Phases:** 5 phases
**Priority:** Fix critical bugs first, then high-priority, then quality improvements

---

## Phase 7A: Critical Bug Fixes (IMMEDIATE)

### 7A.1 - Fix Numerical Overflow in Sigmoid Functions
- [x] Fix `neuro_memory/memory/forgetting.py:109` - The `get_forgetting_probability()` method overflows when activation approaches min_activation. Replace `prob = 1 / (1 + np.exp(x * 5))` with numerically stable version using `np.clip(x * 5, -500, 500)` before exponential. Also fix line 75 (exponential decay) and line 262 (retention formula) with similar bounds checking.
  - **VERIFIED (2026-02-03)**: All three numerical overflow fixes already implemented:
    - Line 76-77: `decay_exponent = np.clip(-self.config.decay_rate * time_elapsed, -500, 0)`
    - Line 121: `x_scaled = np.clip(x * 5, -500, 500)` with input validation for non-finite values at line 113
    - Line 278: `exponent = np.clip(-time_elapsed_hours / stability, -500, 0)`
  - All edge cases tested successfully (extreme values, inf, -inf, nan)

- [x] Fix `integrations/rnd_curiosity.py:322` - Same overflow issue in curiosity calculation. Add clip bounds to prevent exp overflow. Also check lines 126-128 for proper error handling instead of bare except.
  - **VERIFIED (2026-02-03)**: All issues already fixed:
    - Line 323: `sigmoid_input = np.clip(-normalized_curiosity * self.curiosity_scale, -500, 500)` prevents exp overflow
    - Lines 126-128: No bare except exists here - this is regular reshape code in `forward_with_cache`
    - All exception handling in file uses specific types: `except ImportError:` (line 476), `except Exception as e:` (lines 635, 670)
  - Added 5 new numerical stability tests to verify fix handles extreme values (large, small, zero, mixed, high scale)
  - All 33 tests pass including edge case validation

### 7A.2 - Replace All Bare Exception Handlers
- [x] Fix `integrations/autonomous_worker.py` - Replace all bare `except:` with specific exception handling at lines 495 (URL check), and any other locations. Log errors properly instead of silently ignoring.
  - **VERIFIED (2026-02-03)**: All exception handlers already use proper exception types:
    - Line 496: `except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):` for URL checks
    - Lines 259, 270, 404, 415, 426, 440, 451, 485, 538, 883, 934: `except Exception as e:` with error logging via print()
    - Lines 951, 960: `except (AttributeError, KeyError, TypeError) as e:` for active learning stats
    - Lines 1054: `except (json.JSONDecodeError, IOError, OSError) as e:` for stats loading
    - Lines 1062: `except (IOError, OSError, TypeError) as e:` for stats saving
    - Line 1110: `except KeyboardInterrupt:` for CLI test
  - No bare `except:` clauses exist in the file - all 18 exception handlers use specific types or `Exception as e`

- [x] Fix `integrations/docker_sandbox.py` - Replace bare excepts at lines 246, 264, 282, 290, 369, 442, 513 with proper exception handling and logging for container operations.
  - **COMPLETED (2026-02-03)**: All 7 bare excepts replaced with specific exception types:
    - Line 246: `except (DockerException, APIError) as e:` for container.kill()
    - Line 264: `except (DockerException, APIError, KeyError) as e:` for container stats
    - Line 282: `except (DockerException, APIError) as e:` for container.remove()
    - Line 290: `except (OSError, IOError) as e:` for temp directory cleanup
    - Line 369: `except (ValueError, json.JSONDecodeError) as e:` for JSON parsing
    - Line 442: `except (OSError, IOError) as e:` for fallback temp cleanup
    - Line 513: `except (DockerException, APIError) as e:` for orphan container cleanup
  - All handlers now log warnings with error context instead of silently ignoring

- [x] Fix `integrations/browser_agent.py` - Find and replace all bare except clauses with specific exception types and proper error logging.
  - **COMPLETED (2026-02-03)**: Fixed 3 bare except clauses:
    - Lines 197, 202: In `click()` method, replaced bare excepts in CSS/text selector fallback logic with `except Exception as css_err:` and proper nested try-except to capture both errors
    - Line 337: In `close()` method, replaced bare except with `except (OSError, RuntimeError, AttributeError) as e:` with warning message
  - All exception handlers now use specific types or capture the exception for logging

- [x] Fix `integrations/neuro_memory_integration.py:101` - Replace bare except in memory load with specific exception and proper fallback behavior.
  - **COMPLETED (2026-02-03)**: Replaced bare `except:` with specific exception types:
    - `except (FileNotFoundError, json.JSONDecodeError, IOError, OSError, KeyError, TypeError, ValueError) as e:`
    - Added proper logging with `logging.debug()` showing exception type and message
    - Added imports for `json` and `logging` at module level
    - Fallback behavior preserved: new installations or corrupted state files start fresh

- [x] Scan remaining integration files for bare excepts: `self_evolution.py`, `self_training.py`, `super_agent.py`, `cognitive_ollama.py`, `code_evolution.py` and fix each one with specific exception handling.
  - **COMPLETED (2026-02-03)**: Scanned all 5 files and fixed 14 bare excepts total:
    - `self_evolution.py`: Fixed 5 bare excepts at lines 426, 502, 883, 895, 908
      - Lines 426, 502: JSON parsing during data prep → `(json.JSONDecodeError, KeyError, TypeError)`
      - Lines 883, 895, 908: State/history file loading → `(json.JSONDecodeError, IOError, OSError, TypeError)`
    - `self_training.py`: Fixed 3 bare excepts at lines 46, 56, 66
      - Line 46: Facts loading → `(json.JSONDecodeError, IOError, OSError, TypeError)`
      - Line 56: Embeddings loading → `(pickle.UnpicklingError, IOError, OSError, EOFError, TypeError)`
      - Line 66: Stats loading → `(json.JSONDecodeError, IOError, OSError, TypeError)`
    - `super_agent.py`: Fixed 5 bare excepts at lines 86, 223, 243, 245, 247
      - Line 86: LM Studio connection → `(urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError)`
      - Line 223: README fetch → `(urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError)`
      - Lines 243, 245: GitHub API/code fetch → `(urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError)`
      - Line 247: URL parsing → `(ValueError, IndexError, AttributeError)`
    - `code_evolution.py`: Fixed 1 bare except at line 461
      - Line 461: History loading → `(json.JSONDecodeError, IOError, OSError, TypeError)`
    - `cognitive_ollama.py`: Already clean - no bare excepts found
  - All handlers include descriptive comments explaining the expected failure scenarios
  - Tests pass: 350 passed (12 pre-existing failures in unrelated test_ebbinghaus_forgetting.py)

---

## Phase 7B: Thread Safety & High Priority Fixes

### 7B.1 - Fix Thread Safety in EpisodicMemoryStore
- [x] Add `self._episodes_lock = threading.RLock()` to `neuro_memory/memory/episodic_store.py` in `__init__`. Wrap ALL access to `self.episodes` in lock acquisition: lines 254-300 (forgetting loop), line 300 (episode removal), and all other read/write operations on the episodes list.
  - **COMPLETED (2026-02-03)**: All episode list accesses now protected with `self._episodes_lock`:
    - Line 147: Lock initialized as `threading.RLock()` in `__init__`
    - Line 458-459: `store_episode()` - episode append protected
    - Lines 485-488: `store_episode()` - offload check uses lock for count
    - Lines 649-651: `retrieve_by_temporal_range()` - iteration protected
    - Lines 668-672: `_get_episode_by_id()` - search protected
    - Lines 686-700: `_consolidate_memory()` - full modification protected
    - Lines 735-736: `save_state()` - serialization protected
    - Lines 768-769: `load_state()` - restoration protected
    - Lines 788-791: `get_statistics()` - reading protected
  - All 15 episodic memory tests pass

- [x] Fix `_forgetting_loop` method (lines 280-301) to use lock when modifying episodes list. Ensure the lock is held during the entire list modification operation to prevent race conditions.
  - **COMPLETED (2026-02-03)**: Refactored `_process_forgetting()` method to hold lock during entire episode modification:
    - Lines 294-316: Lock held during complete scan of at-risk memories
    - Lines 318-320: Episode removal happens atomically while lock is held
    - Lines 322-330: Reinforcement processing happens outside lock (no list modification)
    - Lines 332-336: Disk offloading happens outside lock (I/O shouldn't hold lock)
  - Added comprehensive docstring explaining thread safety contract

### 7B.2 - Fix Test Functions Pattern
- [x] Fix `tests/test_search.py` - Replace all `return True/False` statements with proper `assert` statements. All 4 test functions (`test_semantic_embeddings`, `test_search_ranking`, `test_memory_search`, `test_concurrent_search`) should use assertions instead of return values.
  - **COMPLETED (2026-02-03)**: Converted all 4 test functions from return-based to assert-based testing:
    - `test_semantic_embeddings()`: Added `assert all_passed` and `assert ai_related_ranked_higher` at end
    - `test_two_stage_retriever()`: Added `assert all_passed` at end
    - `test_knowledge_base_search()`: Added `assert all_passed` at end
    - `test_web_learner()`: Added `assert all_passed` at end
    - Updated `run_all_tests()` to catch `AssertionError` separately from other exceptions
    - Updated `__main__` block to use try/except for exit code
  - All 4 test suites pass successfully

- [x] Review and fix `tests/test_active_learning_rnd.py`, `tests/test_ebbinghaus_forgetting.py`, `tests/test_rnd_curiosity.py` for same pattern issues.
  - **COMPLETED (2026-02-03)**: Reviewed all three test files:
    - `test_active_learning_rnd.py`: Already correct - uses `unittest.TestCase` with proper `self.assert*` methods, no return True/False patterns. 25 tests pass.
    - `test_ebbinghaus_forgetting.py`: Fixed missing `import pytest` statement. File uses pytest-style classes (no TestCase inheritance) with `pytest.approx()` and `assert` statements correctly, but was missing the pytest import. 62 tests pass after fix.
    - `test_rnd_curiosity.py`: Already correct - uses `unittest.TestCase` with proper `self.assert*` methods and `np.testing.assert_*`, no return True/False patterns. 33 tests pass.
  - All 120 tests across the three files pass successfully

### 7B.3 - Fix FAISS Backend
- [x] In `neuro_memory/memory/episodic_store.py:202` - Either implement the FAISS backend properly OR add validation in config to prevent FAISS selection and default to chromadb with a clear error message.
  - **COMPLETED (2026-02-03)**: Implemented graceful fallback with warning instead of NotImplementedError:
    - When `vector_db_backend="faiss"` is requested, emits UserWarning explaining FAISS is not yet implemented
    - Automatically falls back to ChromaDB and updates config.vector_db_backend to reflect actual backend
    - Unknown backend values now raise ValueError with helpful message listing supported backends
    - Added 3 new tests: `test_chromadb_backend_works`, `test_faiss_backend_falls_back_to_chromadb`, `test_unknown_backend_raises_error`
  - All 18 episodic memory integration tests pass

### 7B.4 - Fix Wildcard Import
- [x] Fix `config/__init__.py:8` - Replace `from .paths import *` with explicit imports listing all symbols: `from .paths import (KNOWLEDGE_DIR, EVOLUTION_DIR, TRAINING_DATA_FILE, LEARNED_HASHES_FILE, BENCHMARK_HISTORY_FILE, EVOLUTION_STATE_FILE, ADAPTERS_DIR, LLAMA_FACTORY_OUTPUT_DIR, MLX_MODEL_PATH, HF_MODEL_PATH, LM_STUDIO_URL, DEFAULT_MODEL)`.
  - **COMPLETED (2026-02-03)**: Replaced wildcard import with explicit imports listing all 13 symbols:
    - 12 path constants: `KNOWLEDGE_DIR`, `EVOLUTION_DIR`, `TRAINING_DATA_FILE`, `LEARNED_HASHES_FILE`, `BENCHMARK_HISTORY_FILE`, `EVOLUTION_STATE_FILE`, `ADAPTERS_DIR`, `LLAMA_FACTORY_OUTPUT_DIR`, `MLX_MODEL_PATH`, `HF_MODEL_PATH`, `LM_STUDIO_URL`, `DEFAULT_MODEL`
    - 1 function: `ensure_directories` (also exported from paths.py)
  - All 16 path tests pass

---

## Phase 7C: Performance & Memory Improvements

### 7C.1 - Add Numerical Stability Throughout
- [ ] Fix `neuro_memory/retrieval/two_stage_retriever.py:412` - Add bounds checking before `np.exp()` in recency score calculation. Clip time_diff to prevent overflow: `time_diff_clipped = np.clip(time_diff, 0, 1000)`.

- [ ] Fix `neuro_memory/consolidation/memory_consolidation.py:95` - Add bounds checking to consolidation priority exponential calculations.

- [ ] Create a utility function `safe_exp(x, min_val=-500, max_val=500)` in a new `utils/numerical.py` file that clips input before calling np.exp, and use it throughout the codebase.

### 7C.2 - Fix Memory Leaks in Consolidation
- [ ] In `neuro_memory/memory/episodic_store.py` lines 678-692 - When offloading episodes during consolidation, clean up all related index entries (temporal, spatial, entity indices). Add garbage collection trigger after bulk removal.

### 7C.3 - Fix Division by Zero
- [ ] Fix `neuro_memory/online_learning.py:119` - Add check for zero sum before division: `sum_priorities = np.sum(priorities); probabilities = np.ones_like(priorities) / len(priorities) if sum_priorities == 0 else priorities / sum_priorities`.

### 7C.4 - Add Input Validation
- [ ] Add validation to `neuro_memory/memory/forgetting.py` functions - Check that activation values are finite and non-negative. Raise ValueError for invalid inputs.

- [ ] Add validation to `neuro_memory/memory/episodic_store.py` core functions - Validate episode data before storage, check for NaN/Inf in importance values.

---

## Phase 7D: Code Quality Improvements

### 7D.1 - Consolidate Duplicate Code
- [ ] In `integrations/active_learning.py` - Remove duplicate RND curiosity computation by keeping only `_compute_rnd_curiosity()` (the thread-safe version at line 223) and removing `_compute_rnd_curiosity_unlocked()` (line 375). Update all internal callers to use the locked version within appropriate lock blocks.

### 7D.2 - Add Cleanup/Atexit Handlers
- [ ] Add `atexit` cleanup to `integrations/active_learning.py` - Register handler to save state and close resources on program exit.

- [ ] Add `atexit` cleanup to `integrations/self_evolution.py` - Save evolution state on exit.

- [ ] Add `atexit` cleanup to `integrations/rnd_curiosity.py` - Save RND model state on exit.

- [ ] Add `atexit` cleanup to `integrations/autonomous_worker.py` - Stop worker thread and save stats on exit.

- [ ] Add `atexit` cleanup to `neuro_memory/memory/episodic_store.py` - Stop background forgetting thread and save state on exit.

### 7D.3 - Replace Magic Numbers with Constants
- [ ] Create `config/constants.py` with all magic numbers as named constants with documentation: `EXPOSURE_DECAY_RATE = 0.1`, `DEFAULT_DECAY_RATE = 0.5`, `IMPORTANCE_SIGMOID_SCALE = 2.0`, `BAYESIAN_SURPRISE_PERCENTILE = 75`, etc.

- [ ] Update `active_learning.py:69`, `forgetting.py:25`, `episodic_store.py:495`, `bayesian_surprise.py:307` and other files to use these constants.

### 7D.4 - Add Missing Type Hints
- [ ] Add type hints to all public functions in `integrations/active_learning.py` - Especially `set_llm_interface()` and callback functions.

- [ ] Add type hints to all public functions in `neuro_memory/memory/episodic_store.py`.

- [ ] Add type hints to all public functions in `integrations/autonomous_worker.py`.

---

## Phase 7E: Testing & Documentation

### 7E.1 - Add Integration Tests
- [ ] Create `tests/test_episodic_ebbinghaus_integration.py` - Test that EpisodicMemoryStore correctly registers episodes with Ebbinghaus system, verifies retention calculation works, and confirms spaced repetition scheduling functions correctly.

- [ ] Create `tests/test_concurrent_memory_access.py` - Test thread safety with multiple threads reading/writing episodes simultaneously. Verify no race conditions or data corruption.

- [ ] Create `tests/test_forgetting_workflow.py` - Test complete forgetting + spaced repetition workflow from episode creation through decay and review.

### 7E.2 - Add Documentation
- [ ] Add docstrings to `neuro_memory/memory/episodic_store.py._forgetting_loop()` explaining background task timing, lock usage, and thread safety contract.

- [ ] Add docstrings to `neuro_memory/memory/forgetting.py.get_forgetting_probability()` explaining why sigmoid is used and the mathematical basis.

- [ ] Document thread safety requirements in all episodic_store public methods.

### 7E.3 - Remove Unused Imports
- [ ] Scan all files with `pylint` or `flake8` and remove unused imports, particularly in `neuro_memory/surprise/bayesian_surprise.py` (torch usage) and integration modules.

---

## Summary

| Phase | Focus | Issues Fixed | Priority |
|-------|-------|--------------|----------|
| 7A | Critical Bugs | 2 overflow bugs, 20+ bare excepts | IMMEDIATE |
| 7B | Thread Safety | 4 high-priority issues | URGENT |
| 7C | Performance | 5 memory/numerical issues | HIGH |
| 7D | Code Quality | 10+ quality issues | MEDIUM |
| 7E | Testing/Docs | 6 testing/doc gaps | LOW |

**Execution Order:** 7A -> 7B -> 7C -> 7D -> 7E

Each checkbox task is self-contained and can be executed independently by an autonomous agent.
