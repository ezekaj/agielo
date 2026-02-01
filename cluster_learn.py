#!/usr/bin/env python3
"""
Cluster learning with multiple model instances across multiple Macs.

Example setup (6 total model instances):
  Mac 1 (192.168.1.10): 3 MLX servers on ports 8080, 8081, 8082
  Mac 2 (192.168.1.20): 3 MLX servers on ports 8080, 8081, 8082

Run on Mac 1:
  python3 cluster_learn.py \
    --local-ports 8080,8081,8082 \
    --remote 192.168.1.20:8080,192.168.1.20:8081,192.168.1.20:8082 \
    --workers 4

Run on Mac 2:
  python3 cluster_learn.py \
    --local-ports 8080,8081,8082 \
    --remote 192.168.1.10:8080,192.168.1.10:8081,192.168.1.10:8082 \
    --workers 4
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
from multiprocessing import Process, Manager
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ModelCluster:
    """Manages a cluster of model instances with load balancing."""

    def __init__(self, endpoints: List[str]):
        """endpoints: list of 'host:port' strings"""
        self.endpoints = endpoints
        self.health = {e: True for e in endpoints}
        self.latency = {e: 0.0 for e in endpoints}
        self.calls = {e: 0 for e in endpoints}

    def _pick_best(self) -> str:
        """Pick healthiest, fastest endpoint."""
        healthy = [e for e, ok in self.health.items() if ok]
        if not healthy:
            self.health = {e: True for e in self.endpoints}
            healthy = self.endpoints
        # Prefer lower latency, fewer calls
        return min(healthy, key=lambda e: (self.latency.get(e, 0), self.calls.get(e, 0)))

    def chat(self, prompt: str, timeout: int = 120) -> str:
        """Send request to best available endpoint."""
        tried = set()

        for _ in range(len(self.endpoints)):
            endpoint = self._pick_best()
            if endpoint in tried:
                continue
            tried.add(endpoint)

            try:
                start = time.time()
                url = f"http://{endpoint}/v1/chat/completions"
                data = {
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    result = json.loads(response.read().decode())

                elapsed = time.time() - start
                self.latency[endpoint] = elapsed
                self.calls[endpoint] += 1
                self.health[endpoint] = True

                return result.get("choices", [{}])[0].get("message", {}).get("content", "")

            except Exception as e:
                self.health[endpoint] = False

        return "[Error: All endpoints failed]"

    def status(self) -> str:
        healthy = sum(1 for ok in self.health.values() if ok)
        return f"{healthy}/{len(self.endpoints)} endpoints healthy"


class ClusterEvolution:
    """Thread-safe evolution tracking."""

    def __init__(self):
        self.base = os.path.expanduser("~/.cognitive_ai_knowledge/evolution")
        os.makedirs(self.base, exist_ok=True)
        self.state_file = os.path.join(self.base, "evolution_state.json")
        self.hashes_file = os.path.join(self.base, "learned_hashes.txt")
        self.training_file = os.path.join(self.base, "training_data.jsonl")
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file) as f:
                    self.state = json.load(f)
            else:
                self.state = {"cycle": 1, "facts_cycle": 0, "total": 0}
        except:
            self.state = {"cycle": 1, "facts_cycle": 0, "total": 0}

    def _save(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)

    def is_dup(self, text: str) -> bool:
        h = hashlib.md5(text.encode()).hexdigest()[:16]
        try:
            if os.path.exists(self.hashes_file):
                with open(self.hashes_file) as f:
                    return h in f.read()
        except:
            pass
        return False

    def learn(self, text: str) -> bool:
        if self.is_dup(text):
            return False
        h = hashlib.md5(text.encode()).hexdigest()[:16]
        with open(self.hashes_file, 'a') as f:
            f.write(h + '\n')
        self._load()
        self.state['facts_cycle'] += 1
        self.state['total'] += 1
        self._save()
        return True

    def save_qa(self, pairs: List[Dict], source: str):
        for qa in pairs:
            if qa.get('q') and qa.get('a'):
                entry = {"prompt": qa['q'], "completion": qa['a'], "source": source,
                         "time": datetime.now().isoformat(), "host": os.uname().nodename}
                with open(self.training_file, 'a') as f:
                    f.write(json.dumps(entry) + '\n')

    def stats(self) -> Dict:
        self._load()
        return self.state

    def new_cycle(self):
        self._load()
        self.state['facts_cycle'] = 0
        self.state['cycle'] += 1
        self._save()


# ---- Content Fetchers ----

def fetch_arxiv() -> List[Dict]:
    items = []
    try:
        cat = random.choice(['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'stat.ML'])
        url = f'http://export.arxiv.org/api/query?search_query=cat:{cat}&start={random.randint(0,300)}&max_results=5'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read().decode()
        for entry in re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL):
            t = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            s = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            if t and s:
                items.append({'title': html.unescape(t.group(1).strip()[:100]),
                              'text': html.unescape(s.group(1).strip()[:1200]),
                              'src': f'ArXiv-{cat}'})
    except:
        pass
    return items


def fetch_wiki() -> List[Dict]:
    items = []
    try:
        topic = random.choice([
            'Machine learning', 'Neural network', 'Deep learning', 'Transformer model',
            'Attention mechanism', 'Backpropagation', 'Gradient descent', 'BERT',
            'GPT', 'Reinforcement learning', 'Computer vision', 'NLP',
            'Convolutional neural network', 'Recurrent neural network', 'GAN',
            'Diffusion model', 'Autoencoder', 'Word embedding', 'Fine-tuning'
        ])
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        if data.get('extract'):
            items.append({'title': topic, 'text': data['extract'][:1200], 'src': 'Wikipedia'})
    except:
        pass
    return items


def fetch_ddg() -> List[Dict]:
    items = []
    try:
        q = random.choice([
            'machine learning explained', 'how neural networks work',
            'transformer architecture tutorial', 'deep learning basics',
            'python AI programming', 'LLM training techniques'
        ])
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        if data.get('Abstract'):
            items.append({'title': q, 'text': data['Abstract'][:1000], 'src': 'Web'})
        for t in data.get('RelatedTopics', [])[:2]:
            if isinstance(t, dict) and t.get('Text'):
                items.append({'title': q, 'text': t['Text'][:500], 'src': 'Web'})
    except:
        pass
    return items


FETCHERS = [fetch_arxiv, fetch_wiki, fetch_ddg]


def analyze(cluster: ModelCluster, text: str, title: str) -> Dict:
    """Extract Q&A using cluster."""
    prompt = f"""Extract knowledge. Return JSON only.

TEXT: {text[:1500]}

JSON: {{"topic":"name","summary":"sentence","facts":["f1","f2"],"qa_pairs":[{{"q":"?","a":"answer"}}]}}

JSON:"""

    resp = cluster.chat(prompt)
    try:
        m = re.search(r'\{.*"topic".*\}', resp, re.DOTALL)
        if m:
            return json.loads(m.group().replace('\n', ' '))
    except:
        pass
    return {'topic': title[:40], 'summary': text[:150],
            'facts': [text[:200]], 'qa_pairs': [{'q': f'What is {title}?', 'a': text[:200]}]}


def worker(wid: int, endpoints: List[str], shared):
    """Learning worker."""
    cluster = ModelCluster(endpoints)
    evo = ClusterEvolution()
    host = os.uname().nodename

    print(f"[{host}:W{wid}] Started | {len(endpoints)} endpoints")

    learned = 0
    fetch_idx = wid

    while True:
        try:
            fetcher = FETCHERS[fetch_idx % len(FETCHERS)]
            fetch_idx += 1
            items = fetcher()

            for item in items[:3]:
                text = item.get('text', '')
                title = item.get('title', '')

                if len(text) < 50 or evo.is_dup(text):
                    continue

                analyzed = analyze(cluster, text, title)
                summary = analyzed.get('summary', text[:150])

                if evo.is_dup(summary):
                    continue

                if evo.learn(summary):
                    evo.save_qa(analyzed.get('qa_pairs', []), item.get('src', 'unknown'))
                    learned += 1
                    shared[f'{host}:W{wid}'] = learned

                    st = evo.stats()
                    print(f"[{host}:W{wid}] {analyzed.get('topic', title)[:30]}... | "
                          f"Cycle:{st['facts_cycle']}/100 Total:{st['total']} | {cluster.status()}")

                    if st['facts_cycle'] >= 100:
                        print(f"\n[{host}:W{wid}] === CYCLE {st['cycle']} DONE ===\n")
                        evo.new_cycle()

            time.sleep(0.2)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{host}:W{wid}] Err: {e}")
            time.sleep(1)


def check_endpoints(endpoints: List[str]) -> List[str]:
    """Check which endpoints are live."""
    live = []
    for ep in endpoints:
        try:
            req = urllib.request.Request(f"http://{ep}/v1/models",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3):
                print(f"  [OK] {ep}")
                live.append(ep)
        except Exception as e:
            print(f"  [--] {ep} ({e})")
    return live


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--local-ports', '-l', default='8080,8081,8082',
                   help='Local MLX server ports (comma-sep)')
    p.add_argument('--remote', '-r', default='',
                   help='Remote endpoints host:port (comma-sep)')
    p.add_argument('--workers', '-w', type=int, default=4)
    args = p.parse_args()

    # Build endpoint list
    local = [f"localhost:{port.strip()}" for port in args.local_ports.split(',') if port.strip()]
    remote = [r.strip() for r in args.remote.split(',') if r.strip()]
    endpoints = local + remote

    print("=" * 60)
    print("CLUSTER LEARNING")
    print("=" * 60)
    print(f"Local endpoints:  {local}")
    print(f"Remote endpoints: {remote or 'none'}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    print("\nChecking endpoints...")
    live = check_endpoints(endpoints)

    if not live:
        print("\nNo endpoints available!")
        print("\nStart MLX servers with:")
        print("  ./start_mlx_servers.sh 3")
        sys.exit(1)

    print(f"\n{len(live)} endpoint(s) ready. Starting {args.workers} workers...\n")

    mgr = Manager()
    shared = mgr.dict()

    procs = []
    for i in range(args.workers):
        proc = Process(target=worker, args=(i, live, shared))
        proc.start()
        procs.append(proc)
        time.sleep(0.3)

    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()
