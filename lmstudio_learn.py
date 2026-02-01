#!/usr/bin/env python3
"""
Parallel learning using LM Studio (OpenAI-compatible API)
Faster inference on Apple Silicon with MLX backend.

Usage:
1. Start LM Studio and load a model
2. Enable local server (default: http://localhost:1234)
3. Run: python3 lmstudio_learn.py [num_workers]
"""

import sys
import os
import json
import time
import random
import urllib.request
import urllib.parse
import re
import html
import hashlib
from datetime import datetime
from multiprocessing import Process, Manager
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# LM Studio settings - can be overridden with env vars
LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "localhost")
LMSTUDIO_PORT = os.environ.get("LMSTUDIO_PORT", "1234")
LMSTUDIO_URL = f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1/chat/completions"
MODEL = "local-model"  # LM Studio uses this for the loaded model


def lmstudio_chat(prompt: str, timeout: int = 60) -> str:
    """Chat with LM Studio's OpenAI-compatible API."""
    try:
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        req = urllib.request.Request(
            LMSTUDIO_URL,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode())
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"[Error: {e}]"


class LMStudioEvolution:
    """Evolution tracking with file-based locking for parallel workers."""

    def __init__(self):
        self.base_path = os.path.expanduser("~/.cognitive_ai_knowledge/evolution")
        os.makedirs(self.base_path, exist_ok=True)
        self.state_file = os.path.join(self.base_path, "evolution_state.json")
        self.hashes_file = os.path.join(self.base_path, "learned_hashes.txt")
        self.training_file = os.path.join(self.base_path, "training_data.jsonl")
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                self.state = json.load(f)
        else:
            self.state = {
                "current_cycle": 1,
                "facts_this_cycle": 0,
                "facts_per_cycle": 100,
                "baseline_score": 0,
                "current_score": 0,
                "total_trainings": 0
            }
            self._save_state()

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_duplicate(self, content: str) -> bool:
        """Check if content was already learned."""
        h = hashlib.md5(content.encode()).hexdigest()[:16]
        if os.path.exists(self.hashes_file):
            with open(self.hashes_file) as f:
                if h in f.read():
                    return True
        return False

    def mark_learned(self, content: str) -> bool:
        """Mark content as learned. Returns True if new."""
        h = hashlib.md5(content.encode()).hexdigest()[:16]
        if self.is_duplicate(content):
            return False
        with open(self.hashes_file, 'a') as f:
            f.write(h + '\n')
        self.state['facts_this_cycle'] += 1
        self._save_state()
        return True

    def get_stats(self) -> Dict:
        self._load_state()  # Refresh from disk
        stats_file = os.path.expanduser("~/.cognitive_ai_knowledge/stats.json")
        total = 0
        if os.path.exists(stats_file):
            with open(stats_file) as f:
                total = json.load(f).get('total_facts', 0)
        return {
            'facts_this_cycle': self.state['facts_this_cycle'],
            'total_facts': total,
            'current_cycle': self.state['current_cycle']
        }

    def should_benchmark(self) -> bool:
        self._load_state()
        return self.state['facts_this_cycle'] >= self.state['facts_per_cycle']

    def save_training(self, qa_pairs: List[Dict], source: str):
        """Save Q&A pairs for training."""
        for qa in qa_pairs:
            if qa.get('q') and qa.get('a'):
                pair = {
                    "prompt": qa['q'],
                    "completion": qa['a'],
                    "source": source,
                    "timestamp": datetime.now().isoformat()
                }
                with open(self.training_file, 'a') as f:
                    f.write(json.dumps(pair) + '\n')


def fetch_arxiv(category: str = 'cs.AI') -> List[Dict]:
    """Fetch from ArXiv."""
    items = []
    try:
        url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&start={random.randint(0,100)}&max_results=5'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read().decode('utf-8')
        entries = re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL)
        for entry in entries:
            title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            if title and summary:
                items.append({
                    'title': html.unescape(title.group(1).strip()[:100]),
                    'snippet': html.unescape(summary.group(1).strip()[:800]),
                    'source': f'ArXiv-{category}'
                })
    except Exception:
        pass
    return items


def fetch_wikipedia(topic: str = None) -> List[Dict]:
    """Fetch from Wikipedia."""
    items = []
    try:
        if not topic:
            topic = random.choice([
                'Machine learning', 'Artificial intelligence', 'Neural network',
                'Deep learning', 'Natural language processing', 'Computer vision',
                'Reinforcement learning', 'Transformer (machine learning model)'
            ])
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        if data.get('extract'):
            items.append({
                'title': topic,
                'snippet': data['extract'][:800],
                'source': 'Wikipedia'
            })
    except Exception:
        pass
    return items


def fetch_duckduckgo(query: str) -> List[Dict]:
    """Fetch from DuckDuckGo instant answers."""
    items = []
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        if data.get('Abstract'):
            items.append({'title': query, 'snippet': data['Abstract'][:800], 'source': 'Web'})
        for topic in data.get('RelatedTopics', [])[:3]:
            if isinstance(topic, dict) and topic.get('Text'):
                items.append({'title': query, 'snippet': topic['Text'][:500], 'source': 'Web'})
    except Exception:
        pass
    return items


def analyze_content(content: str, title: str) -> Dict:
    """Analyze content with LM Studio to extract Q&A pairs."""
    prompt = f"""Extract knowledge from this text. Return JSON only.

TEXT: {content[:1500]}

JSON format: {{"topic":"name","summary":"one sentence","facts":["fact1","fact2"],"qa_pairs":[{{"q":"question","a":"answer"}}]}}

JSON:"""

    response = lmstudio_chat(prompt)

    try:
        json_match = re.search(r'\{[^{}]*"topic"[^{}]*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group().replace('\n', ' '))
    except Exception:
        pass

    # Fallback
    return {
        'topic': title[:50],
        'summary': content[:200],
        'facts': [content[:300]],
        'qa_pairs': [{'q': f'What is {title}?', 'a': content[:200]}]
    }


def worker(worker_id: int, sources: List[str], shared_counter):
    """Learning worker process."""
    print(f"[Worker {worker_id}] Starting with sources: {sources}")

    evolution = LMStudioEvolution()

    # Source fetch functions
    fetchers = {
        'arxiv': lambda: fetch_arxiv(random.choice(['cs.AI', 'cs.LG', 'cs.CL'])),
        'wikipedia': lambda: fetch_wikipedia(),
        'web': lambda: fetch_duckduckgo(random.choice([
            'machine learning tutorial', 'neural network basics',
            'python programming', 'algorithm design', 'data structures'
        ])),
    }

    source_idx = 0
    while True:
        try:
            # Rotate through assigned sources
            source = sources[source_idx % len(sources)]
            source_idx += 1

            # Fetch content
            fetcher = fetchers.get(source, fetchers['arxiv'])
            items = fetcher()

            for item in items[:2]:
                content = item.get('snippet', '')
                title = item.get('title', '')

                if not content or len(content) < 50:
                    continue

                if evolution.is_duplicate(content):
                    continue

                # Analyze with LM Studio
                analyzed = analyze_content(content, title)

                summary = analyzed.get('summary', content[:200])
                if evolution.is_duplicate(summary):
                    continue

                # Learn it
                if evolution.mark_learned(summary):
                    evolution.save_training(analyzed.get('qa_pairs', []), item.get('source', source))

                    stats = evolution.get_stats()
                    shared_counter.value += 1

                    print(f"[Worker {worker_id}] [{source}] {analyzed.get('topic', title)[:40]}... | "
                          f"Cycle: {stats['facts_this_cycle']}/100 | Total: {shared_counter.value}")

                    # Check for benchmark
                    if evolution.should_benchmark():
                        print(f"\n[Worker {worker_id}] === CYCLE COMPLETE - 100 facts learned! ===\n")
                        # Reset for next cycle (in production, run benchmark here)
                        evolution.state['facts_this_cycle'] = 0
                        evolution.state['current_cycle'] += 1
                        evolution._save_state()

            time.sleep(0.5)  # Small delay between fetches

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            time.sleep(2)


def check_lmstudio():
    """Check if LM Studio is running."""
    try:
        url = f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1/models"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = data.get('data', [])
            if models:
                print(f"LM Studio @ {LMSTUDIO_HOST}:{LMSTUDIO_PORT} with model: {models[0].get('id', 'unknown')}")
                return True
    except Exception:
        pass
    return False


def main():
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print("=" * 60)
    print("LM STUDIO PARALLEL LEARNER")
    print("=" * 60)

    if not check_lmstudio():
        print(f"\nERROR: LM Studio not running at {LMSTUDIO_HOST}:{LMSTUDIO_PORT}!")
        print("\nTo start LM Studio:")
        print("1. Open LM Studio app")
        print("2. Load a model (e.g., Qwen3-VL-8B)")
        print("3. Click 'Start Server' → Enable 'Serve on Local Network'")
        print("4. Run this script again")
        print("\nFor remote server, set: LMSTUDIO_HOST=192.168.x.x python3 lmstudio_learn.py")
        sys.exit(1)

    print(f"\nStarting {num_workers} parallel learning workers...")
    print("=" * 60)

    # Shared counter for total facts
    manager = Manager()
    shared_counter = manager.Value('i', 0)

    # Distribute sources among workers
    all_sources = ['arxiv', 'wikipedia', 'web']

    workers = []
    for i in range(num_workers):
        sources = [all_sources[i % len(all_sources)]]
        p = Process(target=worker, args=(i, sources, shared_counter))
        p.start()
        workers.append(p)

    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping workers...")
        for p in workers:
            p.terminate()


if __name__ == "__main__":
    main()
