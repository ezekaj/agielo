"""
Self-Training Module
====================

The AI trains itself by:
1. Learning facts from the web
2. Storing them in a local knowledge base
3. Using learned knowledge in future conversations
4. Building embeddings for semantic search
5. Persisting everything to disk

This creates a growing AI that gets smarter over time!
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class KnowledgeBase:
    """Persistent knowledge base that grows over time."""

    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or os.path.expanduser("~/.cognitive_ai_knowledge"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.facts_file = self.storage_path / "facts.json"
        self.embeddings_file = self.storage_path / "embeddings.pkl"
        self.stats_file = self.storage_path / "stats.json"

        # Load existing knowledge
        self.facts = self._load_facts()
        self.embeddings = self._load_embeddings()
        self.stats = self._load_stats()

    def _load_facts(self) -> List[Dict]:
        """Load facts from disk."""
        if self.facts_file.exists():
            try:
                with open(self.facts_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load embeddings from disk."""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}

    def _load_stats(self) -> Dict:
        """Load stats from disk."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'total_facts': 0,
            'total_searches': 0,
            'total_conversations': 0,
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    def save(self):
        """Save all knowledge to disk."""
        # Save facts
        with open(self.facts_file, 'w') as f:
            json.dump(self.facts, f, indent=2)

        # Save embeddings
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

        # Update and save stats
        self.stats['total_facts'] = len(self.facts)
        self.stats['last_updated'] = datetime.now().isoformat()
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def add_fact(self, topic: str, content: str, source: str, embedding: np.ndarray = None):
        """Add a new fact to the knowledge base."""
        fact_id = f"fact_{len(self.facts)}_{datetime.now().timestamp()}"

        fact = {
            'id': fact_id,
            'topic': topic,
            'content': content,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }

        self.facts.append(fact)

        if embedding is not None:
            self.embeddings[fact_id] = embedding

        # Auto-save every 10 facts
        if len(self.facts) % 10 == 0:
            self.save()

        return fact_id

    def search_facts(self, query: str, query_embedding: np.ndarray = None, k: int = 5) -> List[Dict]:
        """Search for relevant facts - understands the whole query as one request."""
        results = []
        query_lower = query.lower()

        # Extract meaningful words from the WHOLE query (ignore small words)
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                      'until', 'while', 'about', 'against', 'what', 'which', 'who',
                      'this', 'that', 'these', 'those', 'am', 'it', 'its', 'i',
                      'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he',
                      'him', 'his', 'she', 'her', 'they', 'them', 'their',
                      'find', 'search', 'look', 'get', 'show', 'tell', 'give'}

        # Get meaningful words from query
        query_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]

        for fact in self.facts:
            score = 0
            fact_topic = fact['topic'].lower()
            fact_content = fact['content'].lower()

            # Check if WHOLE query concept matches (higher score)
            if query_lower in fact_topic or query_lower in fact_content:
                score += 5

            # Check each meaningful word against topic and content
            for word in query_words:
                if word in fact_topic:
                    score += 3  # Topic match is important
                if word in fact_content:
                    score += 1  # Content match

            # Bonus if multiple words match (understands the whole request)
            matching_words = sum(1 for w in query_words if w in fact_topic or w in fact_content)
            if matching_words > 1:
                score += matching_words * 2  # Bonus for matching multiple concepts

            if score > 0:
                results.append((fact, score))

        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, score in results[:k]]

    def get_recent_facts(self, n: int = 10) -> List[Dict]:
        """Get most recent facts."""
        return self.facts[-n:]

    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        return {
            **self.stats,
            'total_facts': len(self.facts),
            'total_embeddings': len(self.embeddings),
            'storage_path': str(self.storage_path)
        }


class SelfTrainer:
    """
    Self-training system that improves the AI over time.

    How it works:
    1. Learns facts from web searches
    2. Stores them in a persistent knowledge base
    3. Creates embeddings for semantic search
    4. Uses learned knowledge in future conversations
    5. Tracks what knowledge is useful (access patterns)
    """

    def __init__(self, storage_path: str = None):
        self.kb = KnowledgeBase(storage_path)
        self.session_learning = []  # What we learned this session

    def learn(self, topic: str, content: str, source: str = "web") -> str:
        """Learn a new fact."""
        # Simple embedding (hash-based for now)
        embedding = self._create_embedding(content)

        fact_id = self.kb.add_fact(topic, content, source, embedding)

        self.session_learning.append({
            'topic': topic,
            'content': content[:100],
            'time': datetime.now().isoformat()
        })

        return fact_id

    def _create_embedding(self, text: str) -> np.ndarray:
        """Create a simple embedding for text."""
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        full_hash = h * 4  # 128 bytes
        emb = np.frombuffer(full_hash[:128], dtype=np.uint8).astype(np.float32)
        return (emb / 127.5) - 1.0

    def recall(self, query: str, k: int = 3) -> List[str]:
        """Recall relevant knowledge for a query."""
        query_emb = self._create_embedding(query)
        facts = self.kb.search_facts(query, query_emb, k=k)

        # Update access counts
        for fact in facts:
            fact['access_count'] += 1

        return [f['content'] for f in facts]

    def get_knowledge_for_prompt(self, query: str) -> str:
        """Get relevant knowledge to inject into the prompt."""
        relevant = self.recall(query, k=3)

        if relevant:
            knowledge = "\n[Relevant knowledge I've learned:]"
            for i, fact in enumerate(relevant, 1):
                knowledge += f"\n{i}. {fact[:200]}"
            return knowledge

        return ""

    def save(self):
        """Save all knowledge to disk."""
        self.kb.save()

    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            **self.kb.get_stats(),
            'session_learning': len(self.session_learning)
        }

    def get_session_summary(self) -> str:
        """Get summary of what was learned this session."""
        if not self.session_learning:
            return "No new knowledge learned this session."

        summary = f"Learned {len(self.session_learning)} new facts:\n"
        for item in self.session_learning[-5:]:
            summary += f"  • {item['topic']}: {item['content']}...\n"

        return summary


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("SELF-TRAINING MODULE TEST")
    print("=" * 60)

    trainer = SelfTrainer()

    # Learn some facts
    trainer.learn("AI", "Artificial intelligence is the simulation of human intelligence", "test")
    trainer.learn("Memory", "Human memory uses hippocampus for episodic storage", "test")
    trainer.learn("Python", "Python is a high-level programming language", "test")

    # Recall
    print("\nRecalling 'intelligence':")
    results = trainer.recall("intelligence")
    for r in results:
        print(f"  • {r[:80]}...")

    # Save
    trainer.save()

    # Stats
    print(f"\nStats: {trainer.get_stats()}")
    print(f"\nSession summary:\n{trainer.get_session_summary()}")

    print("\n" + "=" * 60)
    print(f"Knowledge saved to: {trainer.kb.storage_path}")
    print("=" * 60)
