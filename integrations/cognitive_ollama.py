"""
Cognitive Ollama - LLM with Human-like Memory
==============================================

Combines:
- Ollama (local LLM)
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
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognitive_agent import CognitiveAgent, CognitiveConfig

# Try to import neuro-memory
try:
    from integrations.neuro_memory_adapter import NeuroMemorySystem, NEURO_MEMORY_AVAILABLE
except ImportError:
    NEURO_MEMORY_AVAILABLE = False
    NeuroMemorySystem = None


class CognitiveOllama:
    """
    Ollama LLM enhanced with human-like cognition and bio-inspired memory.
    """

    def __init__(
        self,
        model: str = "ministral-3:8b",
        ollama_url: str = "http://localhost:11434/api/chat",
        memory_path: str = None
    ):
        """
        Initialize cognitive Ollama.

        Args:
            model: Ollama model name
            ollama_url: Ollama API URL
            memory_path: Where to persist memories
        """
        self.model = model
        self.ollama_url = ollama_url
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

    def _get_cognitive_context(self, user_input: str) -> Dict[str, Any]:
        """Process input through cognitive system."""
        emb = self._text_to_embedding(user_input)

        # Perception + thinking
        perception = self.cognition.perceive(emb)
        thought = self.cognition.think(emb, {'source': 'user'})

        # Memory processing
        memory_info = {}
        if self.memory and self.memory.enabled:
            # Process through neuro-memory
            mem_result = self.memory.process_observation(
                emb,
                content=user_input,
                location="chat"
            )
            memory_info = {
                'surprise': mem_result['surprise'],
                'is_novel': mem_result['is_novel'],
                'stored': mem_result['stored']
            }

            # Recall relevant memories
            recalled = self.memory.recall(emb, k=3)
            if recalled:
                memory_info['recalled'] = [m['content'] for m in recalled if m['content']]

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

        # Call Ollama
        response = self._call_ollama(user_input, system_prompt, cognitive_context)

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

    def _call_ollama(self, user_input: str, system_prompt: str, cognitive_context: str) -> str:
        """Call Ollama API."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": (system_prompt or "") + cognitive_context},
                {"role": "user", "content": user_input}
            ],
            "stream": False
        }

        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["message"]["content"]
        except urllib.error.URLError as e:
            return f"[Ollama not running or model '{self.model}' not found. Start with: ollama serve]"
        except urllib.error.HTTPError as e:
            return f"[Ollama error: {e.code} - Model '{self.model}' may not exist. Try: ollama pull ministral-3:8b]"
        except Exception as e:
            return f"[Error calling Ollama with model '{self.model}': {e}]"

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


# Interactive chat
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("COGNITIVE OLLAMA - LLM with Human-like Memory")
    print("=" * 60)
    print("Commands: /stats /consolidate /introspect /quit")
    print("=" * 60)

    # Get model from args
    model = sys.argv[1] if len(sys.argv) > 1 else "ministral-3:8b"
    print(f"\nModel: {model}")

    # Initialize
    ai = CognitiveOllama(model=model)

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
