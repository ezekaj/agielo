# Phase 01: System Audit & Quick Wins

This phase performs a comprehensive system audit and fixes the most impactful issues first. We'll ensure all components initialize correctly, fix obvious bugs, and validate the complete self-evolution loop runs end-to-end. By the end of this phase, running `python3 chat.py` will start without errors and complete one full learning cycle.

## Tasks

- [x] Verify all dependencies are installed and create a requirements check script:
  - Run `python3 -c "import numpy, chromadb, json, urllib.request"` to test core imports
  - Check if LM Studio is running at http://localhost:1234/v1
  - Create `scripts/check_health.py` that validates:
    - All Python imports work
    - LM Studio connectivity
    - ChromaDB can initialize
    - ~/.cognitive_ai_knowledge directory exists with correct permissions
  - Print clear error messages for any missing components
  - **Completed 2025-02-02**: Created `scripts/check_health.py` - all checks pass

- [x] Fix the benchmark scoring bug in `integrations/benchmark.py`:
  - Line 411-414 has unreachable code after return statement in `_score_social_intelligence`
  - The weighted average calculation at lines 411-414 is dead code that never executes
  - Remove the orphaned code block that starts with `# Weighted average: 50% exact...`
  - Verify all 28 benchmark tests score correctly by running the benchmark test
  - **Completed 2025-02-02**: Removed dead code (lines 411-414) from `_score_social_intelligence`. Verified all 28 tests run and score correctly.

- [ ] Add robust error handling to the self-evolution cycle in `integrations/self_evolution.py`:
  - Wrap `_count_training_pairs()` file reading in try/except to handle corrupted JSONL
  - Add validation in `mark_learned()` to reject empty or whitespace-only content
  - Add a `reset_cycle()` method to recover from stuck states
  - Ensure `should_train()` never throws even with corrupted state files

- [ ] Fix the training data path inconsistency:
  - `self_evolution.py` uses `~/.cognitive_ai_knowledge/training_data.jsonl`
  - `super_agent.py` saves to `/Users/tolga/Desktop/mcp-test/super_agent_training.jsonl` (hardcoded path at line 568)
  - Update `super_agent.py` to use the same path as self_evolution
  - Create a constants file `config/paths.py` with all shared paths

- [ ] Create the Working directory and add a system status reporter:
  - Create `Auto Run Docs/Initiation/Working/` directory
  - Create `scripts/system_status.py` that outputs:
    - Current evolution cycle number
    - Total unique facts learned
    - Number of training pairs available
    - Last benchmark score
    - Memory statistics from ChromaDB
  - Make it runnable as `python3 scripts/system_status.py`

- [ ] Run a complete end-to-end validation:
  - Start chat.py and verify initialization completes without errors
  - Verify the autonomous loop starts (check for "[Cycle X]: Starting new cycle..." message)
  - Confirm benchmark runs and scores are reported correctly
  - Verify training pairs are saved to the correct location
  - Document any remaining errors in `Auto Run Docs/Initiation/Working/audit_results.md`
