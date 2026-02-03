"""
Cognitive LLM - LLM with Human-like Memory
==========================================

Combines:
- LM Studio (local LLM via OpenAI-compatible API)
- Human-Cognition-AI (dual-process, emotions, goals)
- Neuro-Memory (bio-inspired episodic memory)

Features:
- Only remembers surprising/important things (Bayesian surprise)
- Forgets unimportant things over time (power-law decay)
- Consolidates memories during "sleep" (schema extraction)
- Learns and improves over time (online learning)
"""

import numpy as np
import json
import urllib.request
import hashlib
import sys
import os
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_agent import CognitiveAgent, CognitiveConfig
from config.constants import (
    MAX_SYSTEM_PROMPT_CHARS,
    MAX_USER_INPUT_CHARS,
    MAX_CONTEXT_CHARS,
    MAX_RESPONSE_CHARS,
)
import re

# Try to import neuro-memory
try:
    from integrations.neuro_memory_adapter import NeuroMemorySystem, NEURO_MEMORY_AVAILABLE
except ImportError:
    NEURO_MEMORY_AVAILABLE = False
    NeuroMemorySystem = None

# Default system prompt to guide model behavior
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Be concise and direct in your responses.

IMPORTANT RULES:
- Do NOT output <think> tags or internal reasoning
- Do NOT show step-by-step thought processes
- Give direct, actionable answers
- Keep responses focused and under 500 words unless more detail is requested
"""


class CognitiveLLM:
    """
    LLM enhanced with human-like cognition and bio-inspired memory.
    Supports LM Studio (OpenAI-compatible API) and Ollama.
    """

    def __init__(
        self,
        model: str = "zai-org/glm-4.7-flash",
        api_url: str = "http://localhost:1234/v1/chat/completions",
        backend: str = "lmstudio",  # "lmstudio" or "ollama"
        memory_path: str = None
    ):
        """
        Initialize cognitive LLM.

        Args:
            model: Model name
            api_url: API URL (LM Studio or Ollama)
            backend: "lmstudio" or "ollama"
            memory_path: Where to persist memories
        """
        self.model = model
        self.api_url = api_url
        self.backend = backend
        self.dim = 128  # Embedding dimension

        # Initialize cognitive agent (emotions, dual-process, etc.)
        self.cognition = CognitiveAgent(CognitiveConfig(
            dim=self.dim,
            enable_emotions=True,
            enable_social=True,
        ))

        # Initialize neuro-memory (advanced bio-inspired memory)
        if NEURO_MEMORY_AVAILABLE:
            self.memory = NeuroMemorySystem(
                dim=self.dim,
                persistence_path=memory_path or os.path.expanduser("~/.cognitive_ollama_memory")
            )
            print(f"[Memory] Neuro-memory enabled (bio-inspired)")
        else:
            self.memory = None
            print(f"[Memory] Basic memory (install chromadb for advanced)")

        # Conversation history
        self.history = []

        # Session stats
        self.session_start = datetime.now()
        self.messages_processed = 0

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        # Simple hash-based embedding (use proper encoder in production)
        h = hashlib.sha256(text.encode()).digest()
        full_hash = h * (self.dim // len(h) + 1)
        emb = np.frombuffer(full_hash[:self.dim], dtype=np.uint8).astype(np.float32)
        emb = (emb / 127.5) - 1.0
        return emb

    def _clean_response(self, response: str) -> str:
        """Clean model response by removing <think> blocks and artifacts."""
        if not response:
            return response

        # Remove <think>...</think> blocks (keep content after)
        cleaned = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)

        # If response was mostly thinking, try to extract content after </think>
        if not cleaned.strip() and '</think>' in response:
            parts = response.split('</think>')
            if len(parts) > 1:
                cleaned = parts[-1].strip()

        # Remove other common artifacts
        cleaned = re.sub(r'\[System\].*?\n', '', cleaned)
        cleaned = re.sub(r'\[Internal\].*?\n', '', cleaned)

        # Clean up excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # Truncate if too long
        if len(cleaned) > MAX_RESPONSE_CHARS:
            cleaned = cleaned[:MAX_RESPONSE_CHARS] + "..."

        return cleaned.strip() or response.strip()

    def _get_cognitive_context(self, user_input: str) -> Dict[str, Any]:
        """Process input through cognitive system."""
        emb = self._text_to_embedding(user_input)

        # Perception + thinking
        perception = self.cognition.perceive(emb)
        thought = self.cognition.think(emb, {'source': 'user'})

        # Memory processing (with error handling for numpy issues)
        memory_info = {}
        if self.memory and self.memory.enabled:
            try:
                # Process through neuro-memory
                mem_result = self.memory.process_observation(
                    emb,
                    content=user_input,
                    location="chat"
                )
                memory_info = {
                    'surprise': float(mem_result.get('surprise', 0)),
                    'is_novel': bool(mem_result.get('is_novel', False)),
                    'stored': bool(mem_result.get('stored', False))
                }

                # Recall relevant memories
                recalled = self.memory.recall(emb, k=3)
                # Handle potential numpy array in recalled (defensive check)
                if recalled is not None and (isinstance(recalled, list) and len(recalled) > 0):
                    contents = []
                    for m in recalled:
                        c = m.get('content') if isinstance(m, dict) else None
                        if c is not None:
                            if isinstance(c, np.ndarray):
                                c = str(c) if c.size > 0 else None
                            if c:
                                contents.append(str(c))
                    memory_info['recalled'] = contents
            except Exception:
                pass  # Ignore memory errors, continue with chat

        return {
            'confidence': thought['dual_process']['confidence'],
            'system_used': thought['dual_process']['system_used'],
            'memory': memory_info
        }

    def chat(self, user_input: str, system_prompt: str = None) -> str:
        """
        Chat with cognitive enhancement.

        Args:
            user_input: User message
            system_prompt: Optional system prompt

        Returns:
            AI response
        """
        self.messages_processed += 1

        # Get cognitive context
        context = self._get_cognitive_context(user_input)

        # Build enhanced prompt
        cognitive_context = f"""
[Cognitive State]
- Confidence: {context['confidence']:.0%}
- Processing: {context['system_used']}
"""
        if context.get('memory', {}).get('is_novel'):
            cognitive_context += "- This seems like a new/important topic\n"

        if context.get('memory', {}).get('recalled'):
            cognitive_context += f"- Related memories: {len(context['memory']['recalled'])}\n"

        # Call LLM (use default system prompt if none provided)
        effective_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        response = self._call_llm(user_input, effective_system_prompt, cognitive_context)

        # Clean response (remove <think> blocks, artifacts, truncate)
        response = self._clean_response(response)

        # Store in history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'assistant': response,
            'context': context
        })

        # Learn from interaction
        response_emb = self._text_to_embedding(response)
        self.cognition.learn({
            'input': self._text_to_embedding(user_input),
            'response': response_emb,
            'reward': 0.5
        })

        return response

    def _truncate_context(self, system_prompt: str, cognitive_context: str, user_input: str) -> tuple:
        """Truncate inputs to stay within context limits (~50k tokens).

        Returns:
            tuple: (system_content, user_content) truncated to fit limits
        """
        system_content = ((system_prompt or "") + cognitive_context)[:MAX_SYSTEM_PROMPT_CHARS]
        user_content = user_input[:MAX_USER_INPUT_CHARS]

        # Final safety check on total context
        total_chars = len(system_content) + len(user_content)
        if total_chars > MAX_CONTEXT_CHARS:
            # Prioritize user input, truncate system prompt more
            excess = total_chars - MAX_CONTEXT_CHARS
            system_content = system_content[:-excess] if len(system_content) > excess else ""

        return system_content, user_content

    def _call_llm(self, user_input: str, system_prompt: str, cognitive_context: str) -> str:
        """Call LLM API (LM Studio or Ollama)."""
        if self.backend == "lmstudio":
            return self._call_lmstudio(user_input, system_prompt, cognitive_context)
        else:
            return self._call_ollama(user_input, system_prompt, cognitive_context)

    def _call_lmstudio(self, user_input: str, system_prompt: str, cognitive_context: str) -> str:
        """Call LM Studio OpenAI-compatible API."""
        system_content, user_content = self._truncate_context(system_prompt, cognitive_context, user_input)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 2048,
            "temperature": 0.7
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"]
        except urllib.error.URLError as e:
            return f"[LM Studio not running. Start LM Studio and load a model.]"
        except Exception as e:
            return f"[Error calling LM Studio: {e}]"

    def _call_ollama(self, user_input: str, system_prompt: str, cognitive_context: str) -> str:
        """Call Ollama API."""
        system_content, user_content = self._truncate_context(system_prompt, cognitive_context, user_input)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "stream": False
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["message"]["content"]
        except urllib.error.URLError as e:
            return f"[Ollama not running or model '{self.model}' not found. Start with: ollama serve]"
        except Exception as e:
            return f"[Error calling Ollama: {e}]"

    def consolidate(self) -> Dict[str, Any]:
        """
        Run memory consolidation (like sleep).

        Call this periodically to:
        - Strengthen important memories
        - Extract patterns/schemas
        - Allow forgetting of unimportant things
        """
        if self.memory and self.memory.enabled:
            return self.memory.consolidate()
        return {'enabled': False}

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            'session_start': self.session_start.isoformat(),
            'messages_processed': self.messages_processed,
            'history_length': len(self.history),
            'cognitive_state': self.cognition.introspect().get('cognitive_state', 'unknown')
        }

        if self.memory and self.memory.enabled:
            mem_stats = self.memory.get_statistics()
            stats['memory'] = mem_stats

        return stats

    def introspect(self) -> Dict[str, Any]:
        """Get detailed cognitive introspection."""
        return self.cognition.introspect()


# Backward compatibility alias
CognitiveOllama = CognitiveLLM


# Interactive chat
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("COGNITIVE LLM - LLM with Human-like Memory")
    print("=" * 60)
    print("Using: LM Studio + Qwen3 Coder 30B (~50 tok/s)")
    print("Commands: /stats /consolidate /introspect /quit")
    print("=" * 60)

    # Get model from args
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3-coder-30b"
    print(f"\nModel: {model}")

    # Initialize with LM Studio
    ai = CognitiveLLM(model=model, backend="lmstudio")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("Goodbye!")
                break

            if user_input == '/stats':
                stats = ai.get_stats()
                print(f"\n[Stats]")
                print(f"  Messages: {stats['messages_processed']}")
                print(f"  State: {stats['cognitive_state']}")
                if 'memory' in stats:
                    print(f"  Novel events: {stats['memory'].get('novel_events', 0)}")
                    print(f"  Novelty rate: {stats['memory'].get('novelty_rate', 0):.1%}")
                continue

            if user_input == '/consolidate':
                print("\n[Running memory consolidation (sleep)...]")
                result = ai.consolidate()
                if result.get('enabled', True):
                    print(f"  Schemas extracted: {result.get('schemas_extracted', 0)}")
                else:
                    print("  Memory consolidation not available")
                continue

            if user_input == '/introspect':
                intro = ai.introspect()
                print(f"\n[Introspection]")
                print(f"  State: {intro.get('cognitive_state', 'unknown')}")
                print(f"  Can introspect: {intro.get('am_i_aware', {}).get('can_introspect', False)}")
                continue

            # Chat
            response = ai.chat(user_input)
            print(f"\nAI: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[Error: {e}]")
