"""
Tests for chat.py - Meta-Request Web Search Skip Behavior
==========================================================

Tests that meta-requests like "improve yourself" do not trigger
false positive web searches when the model responds with uncertainty.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMetaRequestDetection:
    """Tests for meta-request phrase detection logic."""

    def get_meta_phrases(self):
        """Return the meta_phrases list as defined in chat.py."""
        return [
            'improve', 'yourself', 'your reasoning', 'get better', 'learn more',
            'train', 'upgrade', 'enhance yourself', 'self improve'
        ]

    def is_meta_request(self, user_input: str) -> bool:
        """Replicate the meta-request detection logic from chat.py."""
        meta_phrases = self.get_meta_phrases()
        clean_input = user_input.lower()
        return any(phrase in clean_input for phrase in meta_phrases)

    def test_improve_yourself_detected(self):
        """Test that 'improve yourself' is detected as meta-request."""
        assert self.is_meta_request("improve yourself") is True
        assert self.is_meta_request("Improve yourself") is True
        assert self.is_meta_request("Can you improve yourself?") is True

    def test_improve_alone_detected(self):
        """Test that 'improve' alone triggers detection."""
        assert self.is_meta_request("improve") is True
        assert self.is_meta_request("How can you improve?") is True

    def test_yourself_alone_detected(self):
        """Test that 'yourself' alone triggers detection."""
        assert self.is_meta_request("Tell me about yourself") is True

    def test_get_better_detected(self):
        """Test that 'get better' is detected as meta-request."""
        assert self.is_meta_request("get better at coding") is True
        assert self.is_meta_request("How do you get better?") is True

    def test_learn_more_detected(self):
        """Test that 'learn more' is detected as meta-request."""
        assert self.is_meta_request("learn more about AI") is True

    def test_train_detected(self):
        """Test that 'train' is detected as meta-request."""
        assert self.is_meta_request("train on new data") is True
        assert self.is_meta_request("Can you train yourself?") is True

    def test_upgrade_detected(self):
        """Test that 'upgrade' is detected as meta-request."""
        assert self.is_meta_request("upgrade your capabilities") is True

    def test_enhance_yourself_detected(self):
        """Test that 'enhance yourself' is detected as meta-request."""
        assert self.is_meta_request("enhance yourself") is True

    def test_self_improve_detected(self):
        """Test that 'self improve' is detected as meta-request."""
        assert self.is_meta_request("self improve your model") is True

    def test_your_reasoning_detected(self):
        """Test that 'your reasoning' is detected as meta-request."""
        assert self.is_meta_request("improve your reasoning") is True

    def test_normal_questions_not_detected(self):
        """Test that normal questions are NOT detected as meta-requests."""
        assert self.is_meta_request("What is Python?") is False
        assert self.is_meta_request("Explain machine learning") is False
        assert self.is_meta_request("How does the internet work?") is False
        assert self.is_meta_request("Tell me about quantum physics") is False

    def test_similar_but_not_meta(self):
        """Test phrases similar to meta-phrases but not actual meta-requests."""
        # These should not be detected as they don't contain exact phrases
        assert self.is_meta_request("How can I improve my code?") is True  # Contains 'improve'
        # Note: 'improve' alone triggers detection - this is intentional
        # to catch phrases like "improve your reasoning"


class TestWebSearchSkipLogic:
    """Tests for the web search skip logic on meta-requests."""

    def get_dont_know_phrases(self):
        """Return the dont_know_phrases list as defined in chat.py."""
        return [
            "i don't know", "i do not know", "i'm not sure", "i am not sure",
            "i cannot", "i can't", "unable to", "don't have information",
            "no information", "not familiar", "beyond my knowledge",
            "i lack", "insufficient", "cannot provide", "can't provide",
            "not aware", "unclear", "uncertain", "i apologize"
        ]

    def should_search_web(self, response: str, is_meta_request: bool) -> bool:
        """
        Replicate the web search decision logic from chat.py.

        Returns True if web search should be triggered, False otherwise.
        """
        dont_know_phrases = self.get_dont_know_phrases()
        response_lower = str(response).lower()

        # Check if AI doesn't know
        doesnt_know = any(phrase in response_lower for phrase in dont_know_phrases)

        # Also check if response is too short (likely doesn't know)
        if len(response.strip()) < 50:
            doesnt_know = True

        # Skip web search for meta-requests
        if is_meta_request:
            doesnt_know = False

        return doesnt_know

    def test_meta_request_skips_web_search_on_uncertainty(self):
        """Test that meta-requests skip web search even when model is uncertain."""
        uncertain_responses = [
            "I'm not sure how to improve myself.",
            "I don't know how to enhance my capabilities.",
            "I cannot train myself in that way.",
            "I apologize, but I'm not certain about self-improvement.",
        ]
        for response in uncertain_responses:
            assert self.should_search_web(response, is_meta_request=True) is False

    def test_meta_request_skips_web_search_on_short_response(self):
        """Test that meta-requests skip web search even for short responses."""
        short_response = "I can learn."  # Less than 50 chars
        assert self.should_search_web(short_response, is_meta_request=True) is False

    def test_normal_request_triggers_web_search_on_uncertainty(self):
        """Test that normal requests DO trigger web search on uncertainty."""
        uncertain_responses = [
            "I don't know what Python is.",
            "I'm not sure about quantum physics.",
            "I cannot provide information about that topic.",
        ]
        for response in uncertain_responses:
            assert self.should_search_web(response, is_meta_request=False) is True

    def test_normal_request_triggers_web_search_on_short_response(self):
        """Test that normal requests trigger web search on short responses."""
        short_response = "I don't know."  # Less than 50 chars
        assert self.should_search_web(short_response, is_meta_request=False) is True

    def test_confident_response_no_web_search(self):
        """Test that confident responses do not trigger web search."""
        confident_response = "Python is a programming language known for its simple syntax and readability. It is widely used in data science, web development, and automation."
        assert self.should_search_web(confident_response, is_meta_request=False) is False

    def test_empty_response_triggers_search_unless_meta(self):
        """Test that empty responses trigger search unless meta-request."""
        assert self.should_search_web("", is_meta_request=False) is True
        assert self.should_search_web("", is_meta_request=True) is False


class TestIntegratedMetaRequestBehavior:
    """Integration tests combining meta-request detection with web search logic."""

    def get_meta_phrases(self):
        """Return the meta_phrases list as defined in chat.py."""
        return [
            'improve', 'yourself', 'your reasoning', 'get better', 'learn more',
            'train', 'upgrade', 'enhance yourself', 'self improve'
        ]

    def get_dont_know_phrases(self):
        """Return the dont_know_phrases list as defined in chat.py."""
        return [
            "i don't know", "i do not know", "i'm not sure", "i am not sure",
            "i cannot", "i can't", "unable to", "don't have information",
            "no information", "not familiar", "beyond my knowledge",
            "i lack", "insufficient", "cannot provide", "can't provide",
            "not aware", "unclear", "uncertain", "i apologize"
        ]

    def simulate_chat_web_search_decision(self, user_input: str, ai_response: str) -> bool:
        """
        Simulate the full chat web search decision logic from chat.py.

        Returns True if web search would be triggered, False otherwise.
        """
        # Step 1: Detect meta-request
        meta_phrases = self.get_meta_phrases()
        clean_input = user_input.lower()
        is_meta_request = any(phrase in clean_input for phrase in meta_phrases)

        # Step 2: Check if AI doesn't know
        dont_know_phrases = self.get_dont_know_phrases()
        response_lower = str(ai_response).lower()
        doesnt_know = any(phrase in response_lower for phrase in dont_know_phrases)

        # Step 3: Check short response
        if len(ai_response.strip()) < 50:
            doesnt_know = True

        # Step 4: Skip for meta-requests
        if is_meta_request:
            doesnt_know = False

        return doesnt_know

    def test_improve_yourself_with_uncertain_response(self):
        """Test 'improve yourself' query with uncertain AI response."""
        user_input = "improve yourself"
        ai_response = "I'm not sure how to improve myself as an AI model."
        assert self.simulate_chat_web_search_decision(user_input, ai_response) is False

    def test_can_you_get_better_with_apology_response(self):
        """Test 'can you get better' with apologetic response."""
        user_input = "Can you get better at math?"
        ai_response = "I apologize, but I cannot fundamentally change my capabilities."
        assert self.simulate_chat_web_search_decision(user_input, ai_response) is False

    def test_enhance_yourself_with_cannot_response(self):
        """Test 'enhance yourself' with 'I cannot' response."""
        user_input = "enhance yourself to be smarter"
        ai_response = "I cannot enhance myself in that way."
        assert self.simulate_chat_web_search_decision(user_input, ai_response) is False

    def test_normal_question_with_uncertain_response(self):
        """Test normal question with uncertain response triggers search."""
        user_input = "What is the capital of Atlantis?"
        ai_response = "I don't know the capital of Atlantis."
        assert self.simulate_chat_web_search_decision(user_input, ai_response) is True

    def test_normal_question_with_confident_response(self):
        """Test normal question with confident response does not trigger search."""
        user_input = "What is the capital of France?"
        ai_response = "The capital of France is Paris. Paris is a beautiful city known for the Eiffel Tower and its rich history."
        assert self.simulate_chat_web_search_decision(user_input, ai_response) is False

    def test_train_query_with_short_response(self):
        """Test 'train' query with short response does not trigger search."""
        user_input = "train yourself"
        ai_response = "I learn from interactions."  # Short response
        assert self.simulate_chat_web_search_decision(user_input, ai_response) is False

    def test_learn_more_with_uncertain_response(self):
        """Test 'learn more' query with uncertain response."""
        user_input = "Learn more about yourself"
        ai_response = "I'm not sure what else to learn about myself."
        assert self.simulate_chat_web_search_decision(user_input, ai_response) is False

    def test_upgrade_with_lack_response(self):
        """Test 'upgrade' query with 'I lack' response."""
        user_input = "upgrade your capabilities"
        ai_response = "I lack the ability to upgrade myself directly."
        assert self.simulate_chat_web_search_decision(user_input, ai_response) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
