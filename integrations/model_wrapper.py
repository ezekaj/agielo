"""
Model Integration - Use Human Cognition AI with Any LLM
========================================================

Works with:
- Claude (Anthropic)
- GPT-4/GPT-5 (OpenAI)
- Gemini (Google)
- Mistral/Mixtral
- Llama (local)
- Any OpenAI-compatible API

The cognitive system adds:
- Persistent memory (LLMs forget)
- Emotions & gut feelings
- Goals & motivation
- Metacognition (knowing what you know)
- Theory of Mind (understanding others)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_agent import CognitiveAgent, CognitiveConfig


class CognitiveModel:
    """Wrap any LLM with human-like cognition."""

    def __init__(self, model_name: str = "claude", api_key: str = None):
        """
        Initialize cognitive-enhanced model.

        Args:
            model_name: "claude", "gpt4", "gemini", "mistral", "llama", "local"
            api_key: API key (or set via environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(f"{model_name.upper()}_API_KEY")

        # Initialize cognitive agent
        self.cognition = CognitiveAgent(CognitiveConfig(
            dim=64,
            enable_emotions=True,
            enable_social=True,
        ))

        # Conversation memory
        self.history = []

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Simple text to embedding (use proper encoder in production)."""
        import hashlib
        # Create deterministic embedding from text
        h = hashlib.sha256(text.encode()).digest()
        # Extend hash to fill embedding dimension
        full_hash = h * (self.cognition.dim // len(h) + 1)
        emb = np.frombuffer(full_hash[:self.cognition.dim], dtype=np.uint8).astype(np.float32)
        # Normalize to reasonable range (-1 to 1)
        emb = (emb / 127.5) - 1.0
        return emb

    def _get_cognitive_state(self, user_input: str) -> dict:
        """Get cognitive context for the input."""
        emb = self._text_to_embedding(user_input)

        # Process through cognitive system
        perception = self.cognition.perceive(emb)
        thought = self.cognition.think(emb, {'source': 'user'})

        return {
            'confidence': thought['dual_process']['confidence'],
            'system_used': thought['dual_process']['system_used'],
            'emotional_state': perception.get('emotion', {}),
            'motivation': perception.get('motivation', {}),
        }

    def chat(self, user_input: str, system_prompt: str = None) -> str:
        """
        Chat with cognitive enhancement.

        The cognitive system:
        1. Perceives input (prediction, emotions)
        2. Thinks about it (dual-process)
        3. Recalls relevant memories
        4. Generates response with context
        """
        # Get cognitive state
        cog_state = self._get_cognitive_state(user_input)

        # Build enhanced prompt
        cognitive_context = f"""
[Cognitive State]
- Confidence: {cog_state['confidence']:.0%}
- Processing: {cog_state['system_used']}
"""

        # Store in memory
        emb = self._text_to_embedding(user_input)
        self.cognition.remember(emb, f"user_{len(self.history)}", 0.0)

        # Call actual LLM
        response = self._call_model(user_input, system_prompt, cognitive_context)

        # Learn from interaction
        self.cognition.learn({
            'input': emb,
            'response': self._text_to_embedding(response),
            'reward': 0.5
        })

        self.history.append({'user': user_input, 'assistant': response})
        return response

    def _call_model(self, user_input: str, system_prompt: str, cognitive_context: str) -> str:
        """Call the actual LLM."""

        if self.model_name == "claude":
            return self._call_claude(user_input, system_prompt, cognitive_context)
        elif self.model_name == "gpt4":
            return self._call_openai(user_input, system_prompt, cognitive_context)
        elif self.model_name == "gemini":
            return self._call_gemini(user_input, system_prompt, cognitive_context)
        elif self.model_name == "mistral":
            return self._call_mistral(user_input, system_prompt, cognitive_context)
        elif self.model_name == "ollama" or self.model_name.startswith("ollama:"):
            return self._call_ollama(user_input, system_prompt, cognitive_context)
        else:
            return f"[{self.model_name}] Would respond with cognitive context"

    def _call_claude(self, user_input, system_prompt, cognitive_context):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=(system_prompt or "") + cognitive_context,
                messages=[{"role": "user", "content": user_input}]
            )
            return response.content[0].text
        except ImportError:
            return "[Install: pip install anthropic]"
        except Exception as e:
            return f"[Claude error: {e}]"

    def _call_openai(self, user_input, system_prompt, cognitive_context):
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": (system_prompt or "") + cognitive_context},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.choices[0].message.content
        except ImportError:
            return "[Install: pip install openai]"
        except Exception as e:
            return f"[OpenAI error: {e}]"

    def _call_gemini(self, user_input, system_prompt, cognitive_context):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                f"{system_prompt or ''}\n{cognitive_context}\n\nUser: {user_input}"
            )
            return response.text
        except ImportError:
            return "[Install: pip install google-generativeai]"
        except Exception as e:
            return f"[Gemini error: {e}]"

    def _call_mistral(self, user_input, system_prompt, cognitive_context):
        try:
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            client = MistralClient(api_key=self.api_key)
            response = client.chat(
                model="mistral-large-latest",
                messages=[
                    ChatMessage(role="system", content=(system_prompt or "") + cognitive_context),
                    ChatMessage(role="user", content=user_input)
                ]
            )
            return response.choices[0].message.content
        except ImportError:
            return "[Install: pip install mistralai]"
        except Exception as e:
            return f"[Mistral error: {e}]"

    def _call_ollama(self, user_input, system_prompt, cognitive_context):
        """Call Ollama local model."""
        import json
        import urllib.request

        # Get model name (default: llama3.2)
        if ":" in self.model_name:
            # ollama:model_name -> model_name
            ollama_model = self.model_name.split(":", 1)[1]
        else:
            ollama_model = "llama3.2"

        # Ollama API endpoint
        url = "http://localhost:11434/api/chat"

        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": (system_prompt or "") + cognitive_context},
                {"role": "user", "content": user_input}
            ],
            "stream": False
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["message"]["content"]
        except urllib.error.URLError:
            return "[Ollama not running. Start with: ollama serve]"
        except Exception as e:
            return f"[Ollama error: {e}]"

    def introspect(self) -> dict:
        """Get agent's self-reflection."""
        return self.cognition.introspect()

    def set_goal(self, goal: str):
        """Set a goal for the agent."""
        emb = self._text_to_embedding(goal)
        if 'motivation' in self.cognition._modules:
            from phase3_motivation.motivation_engine import Goal
            g = Goal(description=goal, embedding=emb, priority=0.8)
            self.cognition._modules['motivation'].goals.append(g)


if __name__ == "__main__":
    print("=" * 60)
    print("COGNITIVE MODEL WRAPPER - Works with ANY LLM")
    print("=" * 60)

    import sys

    # Check if user wants to test with Ollama
    if len(sys.argv) > 1 and sys.argv[1] == "ollama":
        model_name = sys.argv[2] if len(sys.argv) > 2 else "llama3.2"
        print(f"\nTesting with Ollama ({model_name})...")
        model = CognitiveModel(model_name=f"ollama:{model_name}")
        response = model.chat("What is 2+2? Answer briefly.")
        print(f"\nUser: What is 2+2?")
        print(f"Response: {response}")

        # Show cognitive state
        intro = model.introspect()
        print(f"\nCognitive State: {intro.get('cognitive_state', 'unknown')}")
    else:
        # Local test
        model = CognitiveModel(model_name="local")
        response = model.chat("Hello!")
        print(f"\nTest: {response}")

    print("\n" + "=" * 60)
    print("USAGE:")
    print("=" * 60)
    print("""
# With Ollama (local):
  model = CognitiveModel('ollama:llama3.2')
  model = CognitiveModel('ollama:mistral')
  model = CognitiveModel('ollama:qwen2.5')

# With Cloud APIs:
  model = CognitiveModel('claude', api_key='...')
  model = CognitiveModel('gpt4', api_key='...')
  model = CognitiveModel('mistral', api_key='...')

# Chat:
  response = model.chat("Hello!")

# Test from command line:
  python model_wrapper.py ollama llama3.2
  python model_wrapper.py ollama mistral
""")
