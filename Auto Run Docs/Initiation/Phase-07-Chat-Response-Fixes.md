# Phase 07: Chat Response Quality Fixes

## Problem Summary
The chat system has several issues affecting response quality:

1. **Model outputs verbose `<think>` blocks** - glm-4.7-flash produces long chain-of-thought reasoning that clutters responses
2. **"Improve yourself" triggers incorrect web search** - Meta-requests like "improve yourself" get interpreted as "AI doesn't know" and trigger random web searches
3. **No system prompt to guide behavior** - The model gets minimal guidance (just `[Cognitive State]` context)
4. **Response post-processing is missing** - No cleanup of model artifacts like `<think>` tags

## Tasks

- [x] Add a `_clean_response()` method to `integrations/cognitive_ollama.py` that:
  - Strips `<think>...</think>` blocks from responses (keep only content after `</think>`)
  - Removes other common artifacts like `[System]`, `[Internal]`, excessive newlines
  - Truncates responses over a reasonable limit (e.g., 4000 chars)
  - Call this method in `chat()` before returning the response
  - **COMPLETED**: Method exists at lines 116-141, called in `chat()` at line 222. Added 19 unit tests in `tests/test_cognitive_ollama.py` - all passing.

- [x] Add a proper system prompt in `integrations/cognitive_ollama.py`:
  - Create a `DEFAULT_SYSTEM_PROMPT` constant that instructs the model to:
    - Be concise and direct
    - Not output thinking/reasoning in `<think>` tags
    - Focus on actionable answers
  - Pass this as the default `system_prompt` if none provided
  - **COMPLETED**: `DEFAULT_SYSTEM_PROMPT` exists at lines 44-52 with all required guidance. Used in `chat()` at line 218. Tests in `tests/test_cognitive_ollama.py::TestDefaultSystemPrompt` - 4 tests passing.

- [ ] Fix the "improve yourself" false positive in `chat.py`:
  - The `meta_phrases` check at line 2103-2105 already exists but only skips knowledge injection
  - Extend this to also skip the "doesn't know" web search trigger
  - Meta-requests should get a direct response about self-improvement capabilities, not web searches

- [ ] Add response length limit to `config/constants.py`:
  - `MAX_RESPONSE_CHARS = 4000` - Maximum response length before truncation
