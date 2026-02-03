# Phase 02: Benchmark & Evolution Perfection

This phase perfects the benchmark system and self-evolution loop. The benchmark should accurately measure cognitive abilities across all categories, and the evolution system should reliably track progress, prevent duplicate learning, and trigger MLX training at the right moments. By the end, running a benchmark will produce actionable insights and the system will learn from every mistake.

## Tasks

- [x] Expand the benchmark test suite for comprehensive coverage in `integrations/benchmark.py`:
  - Add 5 more math problems with varying difficulty (algebra, fractions, percentages)
  - Add 3 more theory of mind scenarios beyond Sally-Anne (smarties test, unexpected contents)
  - Add 3 causal reasoning questions ("If A causes B, and B causes C...")
  - Add 3 temporal reasoning questions ("What happened first/next/before...")
  - Add 3 spatial reasoning questions ("If I turn left, then right...")
  - Increase total tests from 28 to ~45 for better statistical significance
  - Add a `difficulty` field to each test (easy/medium/hard)

  **Completed:** Expanded from 28 to 45 tests with 15 categories. Added:
  - 5 math problems (algebra: 3x+7=22, percentages: 20% off $45, fractions: 3/4+1/2, pizza fractions, test scores)
  - 3 theory of mind tests (Smarties test, Band-Aid box unexpected contents, second-order false belief)
  - 3 causal reasoning tests (chain causation, elimination, correlation vs causation)
  - 3 temporal reasoning tests (chronological ordering, day calculation, movie end time)
  - 3 spatial reasoning tests (compass turning, cube faces, 180-degree rotation)
  - All 45 tests now have difficulty field (23 easy, 19 medium, 3 hard)
  - Created tests/test_benchmark.py with 19 pytest tests validating structure and scoring

- [ ] Improve scoring accuracy in `integrations/benchmark.py`:
  - For exact-match tests, use regex to extract just the number (e.g., "35" from "35 years old")
  - Add fuzzy matching for theory_of_mind (allow "she thinks" vs "she believes")
  - Create `_extract_numeric_answer()` helper to pull numbers from responses
  - Create `_extract_yes_no_answer()` helper for boolean questions
  - Add partial credit for showing correct reasoning even with wrong final answer
  - Test each scoring function with 5 sample responses to verify accuracy

- [ ] Add per-category benchmark tracking and analysis:
  - Modify `run_benchmark()` to return scores grouped by category
  - Create `analyze_weaknesses()` method that identifies lowest-scoring categories
  - Save category scores to benchmark history for trend analysis
  - Create `get_improvement_by_category()` to show which areas improved most
  - Print category-level summary after each benchmark run

- [ ] Perfect the duplicate detection in `integrations/self_evolution.py`:
  - The current MD5 hash can have collisions - add content length check
  - Normalize Unicode before hashing (convert to ASCII or NFC form)
  - Add `get_similar_content()` method using fuzzy matching (Levenshtein distance)
  - Create `deduplicate_existing()` to clean up any duplicates in learned_hashes
  - Add stats on how many duplicates were rejected per cycle

- [ ] Improve the training trigger logic in `integrations/self_evolution.py`:
  - Current threshold is 1% improvement OR 500+ pairs with 0 trainings
  - Add adaptive threshold: lower to 0.5% after 3 trainings, 0.25% after 5
  - Track time since last training - train if >24 hours with any improvement
  - Add `force_train()` method for manual training regardless of thresholds
  - Log detailed reasoning for why training was/wasn't triggered

- [ ] Create benchmark regression tests:
  - Create `tests/test_benchmark.py` with pytest
  - Test that all 45+ tests have valid structure (question, answer, category)
  - Test scoring functions with known inputs/outputs
  - Test that benchmark history persists correctly across restarts
  - Test category analysis produces correct results
  - Make tests runnable with `pytest tests/test_benchmark.py -v`
