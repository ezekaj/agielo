#!/usr/bin/env python3
"""
Distributed learning across multiple LM Studio servers.
Each Mac runs its own LM Studio + workers, all sharing knowledge via Git.

Usage:
  Mac 1: python3 distributed_learn.py --servers localhost:1234,192.168.1.20:1234 --workers 3
  Mac 2: python3 distributed_learn.py --servers localhost:1234,192.168.1.10:1234 --workers 3

All workers load-balance across all available servers.
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
import argparse
from datetime import datetime
from multiprocessing import Process, Manager, Lock
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ServerPool:
    """Load balancer for multiple LM Studio servers."""

    def __init__(self, servers: List[str]):
        self.servers = servers  # ["localhost:1234", "192.168.1.20:1234"]
        self.server_status = {s: True for s in servers}
        self.request_count = {s: 0 for s in servers}

    def get_server(self) -> str:
        """Get least-loaded healthy server."""
        healthy = [s for s, ok in self.server_status.items() if ok]
        if not healthy:
            # Try all servers again
            self.server_status = {s: True for s in self.servers}
            healthy = self.servers

        # Pick server with fewest requests (simple load balancing)
        return min(healthy, key=lambda s: self.request_count.get(s, 0))

    def mark_success(self, server: str):
        self.server_status[server] = True
        self.request_count[server] = self.request_count.get(server, 0) + 1

    def mark_failed(self, server: str):
        self.server_status[server] = False
        print(f"[Pool] Server {server} marked unhealthy")

    def chat(self, prompt: str, timeout: int = 90) -> str:
        """Chat with best available server."""
        for attempt in range(len(self.servers)):
            server = self.get_server()
            try:
                url = f"http://{server}/v1/chat/completions"
                data = {
                    "model": "local-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "stream": False
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    result = json.loads(response.read().decode())
                    self.mark_success(server)
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                self.mark_failed(server)
                continue

        return "[Error: All servers failed]"


class DistributedEvolution:
    """Evolution tracking with file locking for distributed workers."""

    def __init__(self):
        self.base_path = os.path.expanduser("~/.cognitive_ai_knowledge/evolution")
        os.makedirs(self.base_path, exist_ok=True)
        self.state_file = os.path.join(self.base_path, "evolution_state.json")
        self.hashes_file = os.path.join(self.base_path, "learned_hashes.txt")
        self.training_file = os.path.join(self.base_path, "training_data.jsonl")
        self.lock_file = os.path.join(self.base_path, ".lock")
        self._load_state()

    def _acquire_lock(self, timeout=5):
        """Simple file-based lock."""
        start = time.time()
        while os.path.exists(self.lock_file):
            if time.time() - start > timeout:
                # Force remove stale lock
                try:
                    os.remove(self.lock_file)
                except:
                    pass
                break
            time.sleep(0.1)
        try:
            with open(self.lock_file, 'w') as f:
                f.write(str(os.getpid()))
        except:
            pass

    def _release_lock(self):
        try:
            os.remove(self.lock_file)
        except:
            pass

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    self.state = json.load(f)
            except:
                self.state = self._default_state()
        else:
            self.state = self._default_state()
            self._save_state()

    def _default_state(self):
        return {
            "current_cycle": 1,
            "facts_this_cycle": 0,
            "facts_per_cycle": 100,
            "baseline_score": 0,
            "current_score": 0,
            "total_facts": 0,
            "total_trainings": 0
        }

    def _save_state(self):
        self._acquire_lock()
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        finally:
            self._release_lock()

    def is_duplicate(self, content: str) -> bool:
        h = hashlib.md5(content.encode()).hexdigest()[:16]
        if os.path.exists(self.hashes_file):
            try:
                with open(self.hashes_file) as f:
                    if h in f.read():
                        return True
            except:
                pass
        return False

    def mark_learned(self, content: str) -> bool:
        h = hashlib.md5(content.encode()).hexdigest()[:16]
        if self.is_duplicate(content):
            return False

        self._acquire_lock()
        try:
            with open(self.hashes_file, 'a') as f:
                f.write(h + '\n')

            # Reload state (might have changed)
            self._load_state()
            self.state['facts_this_cycle'] += 1
            self.state['total_facts'] += 1
            self._save_state()
        finally:
            self._release_lock()

        return True

    def get_stats(self) -> Dict:
        self._load_state()
        return {
            'facts_this_cycle': self.state['facts_this_cycle'],
            'total_facts': self.state['total_facts'],
            'current_cycle': self.state['current_cycle']
        }

    def should_benchmark(self) -> bool:
        self._load_state()
        return self.state['facts_this_cycle'] >= self.state['facts_per_cycle']

    def start_new_cycle(self):
        self._acquire_lock()
        try:
            self._load_state()
            self.state['facts_this_cycle'] = 0
            self.state['current_cycle'] += 1
            self._save_state()
        finally:
            self._release_lock()

    def save_training(self, qa_pairs: List[Dict], source: str):
        self._acquire_lock()
        try:
            for qa in qa_pairs:
                if qa.get('q') and qa.get('a'):
                    pair = {
                        "prompt": qa['q'],
                        "completion": qa['a'],
                        "source": source,
                        "timestamp": datetime.now().isoformat(),
                        "host": os.uname().nodename
                    }
                    with open(self.training_file, 'a') as f:
                        f.write(json.dumps(pair) + '\n')
        finally:
            self._release_lock()


def fetch_arxiv(category: str = 'cs.AI') -> List[Dict]:
    items = []
    try:
        url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&start={random.randint(0,200)}&max_results=5'
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
                    'snippet': html.unescape(summary.group(1).strip()[:1000]),
                    'source': f'ArXiv-{category}'
                })
    except Exception:
        pass
    return items


def fetch_wikipedia(topic: str = None) -> List[Dict]:
    items = []
    try:
        if not topic:
            topic = random.choice([
                'Machine learning', 'Artificial intelligence', 'Neural network',
                'Deep learning', 'Natural language processing', 'Computer vision',
                'Reinforcement learning', 'Transformer model', 'Large language model',
                'Gradient descent', 'Backpropagation', 'Convolutional neural network',
                'Recurrent neural network', 'Attention mechanism', 'BERT model',
                'GPT', 'Diffusion model', 'Generative adversarial network'
            ])
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        if data.get('extract'):
            items.append({
                'title': topic,
                'snippet': data['extract'][:1000],
                'source': 'Wikipedia'
            })
    except Exception:
        pass
    return items


def fetch_duckduckgo(query: str = None) -> List[Dict]:
    items = []
    try:
        if not query:
            query = random.choice([
                'machine learning tutorial', 'neural network basics',
                'python programming patterns', 'algorithm design',
                'data structures explained', 'deep learning concepts',
                'transformer architecture', 'attention mechanism explained'
            ])
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        if data.get('Abstract'):
            items.append({'title': query, 'snippet': data['Abstract'][:1000], 'source': 'Web'})
        for topic in data.get('RelatedTopics', [])[:3]:
            if isinstance(topic, dict) and topic.get('Text'):
                items.append({'title': query, 'snippet': topic['Text'][:500], 'source': 'Web'})
    except Exception:
        pass
    return items


def analyze_content(pool: ServerPool, content: str, title: str) -> Dict:
    """Analyze content using server pool."""
    prompt = f"""Extract knowledge from this text. Return valid JSON only.

TEXT: {content[:1500]}

Return JSON: {{"topic":"short name","summary":"one sentence summary","facts":["fact1","fact2","fact3"],"qa_pairs":[{{"q":"question?","a":"detailed answer"}}]}}

JSON:"""

    response = pool.chat(prompt)

    try:
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*"topic"[^{}]*\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group().replace('\n', ' '))
            if parsed.get('topic') and parsed.get('qa_pairs'):
                return parsed
    except:
        pass

    # Fallback
    return {
        'topic': title[:50],
        'summary': content[:200],
        'facts': [content[:300]],
        'qa_pairs': [{'q': f'What is {title}?', 'a': content[:300]}]
    }


def worker(worker_id: int, servers: List[str], sources: List[str], stats_dict):
    """Distributed learning worker."""
    pool = ServerPool(servers)
    evolution = DistributedEvolution()
    hostname = os.uname().nodename

    print(f"[{hostname}:W{worker_id}] Starting | Servers: {servers} | Sources: {sources}")

    fetchers = {
        'arxiv': lambda: fetch_arxiv(random.choice(['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE'])),
        'wikipedia': fetch_wikipedia,
        'web': fetch_duckduckgo,
    }

    source_idx = worker_id
    facts_learned = 0

    while True:
        try:
            source = sources[source_idx % len(sources)]
            source_idx += 1

            fetcher = fetchers.get(source, fetchers['arxiv'])
            items = fetcher()

            for item in items[:3]:
                content = item.get('snippet', '')
                title = item.get('title', '')

                if not content or len(content) < 50:
                    continue

                if evolution.is_duplicate(content):
                    continue

                # Analyze with load-balanced servers
                analyzed = analyze_content(pool, content, title)

                summary = analyzed.get('summary', content[:200])
                if evolution.is_duplicate(summary):
                    continue

                if evolution.mark_learned(summary):
                    evolution.save_training(analyzed.get('qa_pairs', []), item.get('source', source))
                    facts_learned += 1

                    stats = evolution.get_stats()
                    stats_dict[f'{hostname}:W{worker_id}'] = facts_learned

                    total_all = sum(stats_dict.values())
                    print(f"[{hostname}:W{worker_id}] [{source}] {analyzed.get('topic', title)[:35]}... | "
                          f"Cycle: {stats['facts_this_cycle']}/100 | Total: {stats['total_facts']} | "
                          f"Session: {total_all}")

                    if evolution.should_benchmark():
                        print(f"\n[{hostname}:W{worker_id}] === CYCLE {stats['current_cycle']} COMPLETE ===\n")
                        evolution.start_new_cycle()

            time.sleep(0.3)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{hostname}:W{worker_id}] Error: {e}")
            time.sleep(2)


def check_servers(servers: List[str]) -> List[str]:
    """Check which servers are available."""
    available = []
    for server in servers:
        try:
            url = f"http://{server}/v1/models"
            req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode())
                models = data.get('data', [])
                model_name = models[0].get('id', 'unknown') if models else 'unknown'
                print(f"[OK] {server} - {model_name}")
                available.append(server)
        except Exception as e:
            print(f"[FAIL] {server} - {e}")
    return available


def main():
    parser = argparse.ArgumentParser(description='Distributed learning across LM Studio servers')
    parser.add_argument('--servers', '-s', type=str, default='localhost:1234',
                        help='Comma-separated server list (e.g., localhost:1234,192.168.1.20:1234)')
    parser.add_argument('--workers', '-w', type=int, default=3,
                        help='Number of workers per machine')
    args = parser.parse_args()

    servers = [s.strip() for s in args.servers.split(',')]
    num_workers = args.workers
    hostname = os.uname().nodename

    print("=" * 70)
    print("DISTRIBUTED LEARNING CLUSTER")
    print("=" * 70)
    print(f"Host: {hostname}")
    print(f"Workers: {num_workers}")
    print(f"Servers: {servers}")
    print("=" * 70)

    print("\nChecking servers...")
    available = check_servers(servers)

    if not available:
        print("\nERROR: No servers available!")
        print("\nMake sure LM Studio is running with 'Serve on Local Network' enabled.")
        sys.exit(1)

    print(f"\n{len(available)} server(s) available. Starting {num_workers} workers...\n")

    # Shared stats across workers
    manager = Manager()
    stats_dict = manager.dict()

    sources = ['arxiv', 'wikipedia', 'web']

    workers = []
    for i in range(num_workers):
        worker_sources = [sources[i % len(sources)]]
        p = Process(target=worker, args=(i, available, worker_sources, stats_dict))
        p.start()
        workers.append(p)
        time.sleep(0.5)

    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping workers...")
        for p in workers:
            p.terminate()


if __name__ == "__main__":
    main()
