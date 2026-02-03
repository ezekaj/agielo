"""
Tests for cognitive_ollama.py - Chat Response Quality
======================================================

Tests the _clean_response method and DEFAULT_SYSTEM_PROMPT behavior
to ensure chat responses are clean and well-formatted.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.cognitive_ollama import CognitiveLLM, DEFAULT_SYSTEM_PROMPT
from config.constants import MAX_RESPONSE_CHARS


class TestCleanResponse:
    """Tests for the _clean_response method."""

    @pytest.fixture
    def llm(self):
        """Create a CognitiveLLM instance for testing."""
        # We don't need a real LLM connection for testing _clean_response
        return CognitiveLLM(model="test-model", backend="lmstudio")

    def test_strips_think_blocks(self, llm):
        """Test that <think>...</think> blocks are removed."""
        response = "<think>This is internal reasoning that should not be shown.</think>The actual response."
        cleaned = llm._clean_response(response)
        assert "<think>" not in cleaned
        assert "</think>" not in cleaned
        assert "internal reasoning" not in cleaned
        assert "The actual response." in cleaned

    def test_strips_multiline_think_blocks(self, llm):
        """Test that multiline <think> blocks are removed."""
        response = """<think>
Step 1: Analyze the question
Step 2: Consider options
Step 3: Formulate response
</think>
Here is my answer to your question."""
        cleaned = llm._clean_response(response)
        assert "<think>" not in cleaned
        assert "Step 1" not in cleaned
        assert "Here is my answer" in cleaned

    def test_extracts_content_after_think_block(self, llm):
        """Test extraction of content when response is mostly thinking."""
        response = "<think>Long thinking process here that takes up the whole response</think>Final answer."
        cleaned = llm._clean_response(response)
        assert "Final answer" in cleaned

    def test_removes_system_artifact(self, llm):
        """Test that [System] artifacts are removed."""
        response = "[System] Internal note\nActual user response."
        cleaned = llm._clean_response(response)
        assert "[System]" not in cleaned
        assert "Internal note" not in cleaned
        assert "Actual user response." in cleaned

    def test_removes_internal_artifact(self, llm):
        """Test that [Internal] artifacts are removed."""
        response = "[Internal] Debug info\nThe response to show."
        cleaned = llm._clean_response(response)
        assert "[Internal]" not in cleaned
        assert "Debug info" not in cleaned
        assert "The response to show." in cleaned

    def test_cleans_excessive_newlines(self, llm):
        """Test that excessive newlines are reduced."""
        response = "First paragraph.\n\n\n\n\nSecond paragraph."
        cleaned = llm._clean_response(response)
        assert "\n\n\n" not in cleaned
        assert "First paragraph." in cleaned
        assert "Second paragraph." in cleaned

    def test_truncates_long_responses(self, llm):
        """Test that responses over MAX_RESPONSE_CHARS are truncated."""
        long_response = "A" * (MAX_RESPONSE_CHARS + 500)
        cleaned = llm._clean_response(long_response)
        assert len(cleaned) <= MAX_RESPONSE_CHARS + 3  # +3 for "..."
        assert cleaned.endswith("...")

    def test_handles_empty_response(self, llm):
        """Test that empty responses are handled gracefully."""
        cleaned = llm._clean_response("")
        assert cleaned == ""

    def test_handles_none_response(self, llm):
        """Test that None responses are handled gracefully."""
        cleaned = llm._clean_response(None)
        assert cleaned is None

    def test_combined_artifacts(self, llm):
        """Test cleaning response with multiple types of artifacts."""
        response = """<think>
Let me think about this...
I should consider A, B, and C.
</think>
[System] Processing complete

Here is my actual response.


[Internal] Log entry

The final answer is 42."""
        cleaned = llm._clean_response(response)
        assert "<think>" not in cleaned
        assert "[System]" not in cleaned
        assert "[Internal]" not in cleaned
        assert "Here is my actual response." in cleaned
        assert "The final answer is 42." in cleaned

    def test_preserves_legitimate_angle_brackets(self, llm):
        """Test that legitimate use of angle brackets is preserved."""
        response = "Use the command <command> to run the script. Also try x < 5."
        cleaned = llm._clean_response(response)
        assert "<command>" in cleaned
        assert "x < 5" in cleaned

    def test_multiple_think_blocks(self, llm):
        """Test that multiple <think> blocks are all removed."""
        response = "<think>First thought</think>Response 1.<think>Second thought</think>Response 2."
        cleaned = llm._clean_response(response)
        assert "<think>" not in cleaned
        assert "</think>" not in cleaned
        assert "First thought" not in cleaned
        assert "Second thought" not in cleaned
        assert "Response 1." in cleaned
        assert "Response 2." in cleaned


class TestDefaultSystemPrompt:
    """Tests for the DEFAULT_SYSTEM_PROMPT constant."""

    def test_system_prompt_exists(self):
        """Test that DEFAULT_SYSTEM_PROMPT is defined."""
        assert DEFAULT_SYSTEM_PROMPT is not None
        assert len(DEFAULT_SYSTEM_PROMPT) > 0

    def test_system_prompt_instructs_no_think_tags(self):
        """Test that system prompt tells model not to use <think> tags."""
        assert "<think>" in DEFAULT_SYSTEM_PROMPT.lower() or "think" in DEFAULT_SYSTEM_PROMPT.lower()
        assert "not" in DEFAULT_SYSTEM_PROMPT.lower() or "do not" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_system_prompt_requests_conciseness(self):
        """Test that system prompt asks for concise responses."""
        prompt_lower = DEFAULT_SYSTEM_PROMPT.lower()
        assert "concise" in prompt_lower or "direct" in prompt_lower or "focused" in prompt_lower

    def test_system_prompt_is_reasonable_length(self):
        """Test that system prompt is not too long."""
        # Should be short enough to not waste context
        assert len(DEFAULT_SYSTEM_PROMPT) < 1000


class TestMaxResponseChars:
    """Tests for MAX_RESPONSE_CHARS constant."""

    def test_constant_exists(self):
        """Test that MAX_RESPONSE_CHARS is defined."""
        assert MAX_RESPONSE_CHARS is not None

    def test_constant_is_reasonable(self):
        """Test that MAX_RESPONSE_CHARS is a reasonable value."""
        assert MAX_RESPONSE_CHARS >= 1000  # At least 1000 chars
        assert MAX_RESPONSE_CHARS <= 10000  # Not too long

    def test_constant_is_4000(self):
        """Test that MAX_RESPONSE_CHARS is set to expected value."""
        assert MAX_RESPONSE_CHARS == 4000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
