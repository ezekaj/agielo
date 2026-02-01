#!/usr/bin/env python3
"""
Multi-Qwen distributed learning.
Uses all loaded Qwen models in LM Studio across multiple Macs.

Mac 1: python3 multi_qwen_learn.py --workers 6
Mac 2: python3 multi_qwen_learn.py --workers 6 --servers localhost:1234,192.168.x.10:1234
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


def get_available_models(server: str) -> List[str]:
    """Get all Qwen models from LM Studio server."""
    try:
        url = f"http://{server}/v1/models"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [m['id'] for m in data.get('data', []) if 'qwen' in m.get('id', '').lower()]
            return models
    except:
        return []


class MultiModelPool:
    """Pool of models across servers."""

    def __init__(self, servers: List[str]):
        self.endpoints = []  # (server, model) tuples

        for server in servers:
            models = get_available_models(server)
            for model in models:
                self.endpoints.append((server, model))

        self.health = {e: True for e in self.endpoints}
        self.calls = {e: 0 for e in self.endpoints}

    def _pick_best(self):
        healthy = [e for e, ok in self.health.items() if ok]
        if not healthy:
            self.health = {e: True for e in self.endpoints}
            healthy = self.endpoints
        return min(healthy, key=lambda e: self.calls.get(e, 0))

    def chat(self, prompt: str, timeout: int = 120) -> str:
        tried = set()

        for _ in range(len(self.endpoints)):
            endpoint = self._pick_best()
            if endpoint in tried:
                continue
            tried.add(endpoint)

            server, model = endpoint

            try:
                url = f"http://{server}/v1/chat/completions"
                data = {
                    "model": model,
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

                self.calls[endpoint] += 1
                self.health[endpoint] = True
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")

            except Exception as e:
                self.health[endpoint] = False

        return "[Error: All models failed]"

    def status(self) -> str:
        healthy = sum(1 for ok in self.health.values() if ok)
        return f"{healthy}/{len(self.endpoints)} models"


class Evolution:
    def __init__(self):
        self.base = os.path.expanduser("~/.cognitive_ai_knowledge/evolution")
        os.makedirs(self.base, exist_ok=True)
        self.state_file = os.path.join(self.base, "evolution_state.json")
        self.hashes_file = os.path.join(self.base, "learned_hashes.txt")
        self.training_file = os.path.join(self.base, "training_data.jsonl")
        self._load()

    def _load(self):
        try:
            with open(self.state_file) as f:
                self.state = json.load(f)
        except:
            self.state = {"cycle": 1, "facts_cycle": 0, "total": 0}

    def _save(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)

    def is_dup(self, text: str) -> bool:
        h = hashlib.md5(text.encode()).hexdigest()[:16]
        try:
            with open(self.hashes_file) as f:
                return h in f.read()
        except:
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

    def stats(self):
        self._load()
        return self.state

    def new_cycle(self):
        self._load()
        self.state['facts_cycle'] = 0
        self.state['cycle'] += 1
        self._save()


# Fetchers
def fetch_arxiv():
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


def fetch_wiki():
    items = []
    try:
        topic = random.choice([
            'Machine learning', 'Neural network', 'Deep learning', 'Transformer model',
            'Attention mechanism', 'Backpropagation', 'Gradient descent', 'BERT',
            'GPT', 'Reinforcement learning', 'Computer vision', 'Natural language processing'
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


def fetch_ddg():
    items = []
    try:
        q = random.choice(['machine learning', 'neural networks', 'deep learning', 'AI tutorial'])
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        if data.get('Abstract'):
            items.append({'title': q, 'text': data['Abstract'][:1000], 'src': 'Web'})
    except:
        pass
    return items


FETCHERS = [fetch_arxiv, fetch_wiki, fetch_ddg]


def analyze(pool: MultiModelPool, text: str, title: str) -> Dict:
    prompt = f"""Extract knowledge. Return JSON only.

TEXT: {text[:1500]}

JSON: {{"topic":"name","summary":"sentence","facts":["f1"],"qa_pairs":[{{"q":"?","a":"answer"}}]}}

JSON:"""

    resp = pool.chat(prompt)
    try:
        m = re.search(r'\{.*"topic".*\}', resp, re.DOTALL)
        if m:
            return json.loads(m.group().replace('\n', ' '))
    except:
        pass
    return {'topic': title[:40], 'summary': text[:150],
            'facts': [text[:200]], 'qa_pairs': [{'q': f'What is {title}?', 'a': text[:200]}]}


def worker(wid: int, servers: List[str], shared):
    pool = MultiModelPool(servers)
    evo = Evolution()
    host = os.uname().nodename

    print(f"[{host}:W{wid}] Started | {pool.status()}", flush=True)

    learned = 0
    fetch_idx = wid

    while True:
        try:
            fetcher = FETCHERS[fetch_idx % len(FETCHERS)]
            fetch_idx += 1
            print(f"[{host}:W{wid}] Fetching from {fetcher.__name__}...", flush=True)
            items = fetcher()
            print(f"[{host}:W{wid}] Got {len(items)} items", flush=True)

            for item in items[:3]:
                text = item.get('text', '')
                title = item.get('title', '')

                if len(text) < 50 or evo.is_dup(text):
                    continue

                analyzed = analyze(pool, text, title)
                summary = analyzed.get('summary', text[:150])

                if evo.is_dup(summary):
                    continue

                if evo.learn(summary):
                    evo.save_qa(analyzed.get('qa_pairs', []), item.get('src', ''))
                    learned += 1
                    shared[f'{host}:W{wid}'] = learned

                    st = evo.stats()
                    total_session = sum(shared.values())
                    print(f"[{host}:W{wid}] {analyzed.get('topic', '')[:30]}... | "
                          f"Cycle:{st['facts_cycle']}/100 Total:{st['total']} Session:{total_session} | {pool.status()}", flush=True)

                    if st['facts_cycle'] >= 100:
                        print(f"\n[{host}:W{wid}] === CYCLE {st['cycle']} DONE ===\n")
                        evo.new_cycle()

            time.sleep(0.2)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{host}:W{wid}] Err: {e}")
            time.sleep(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--servers', '-s', default='localhost:1234',
                   help='LM Studio servers (comma-sep)')
    p.add_argument('--workers', '-w', type=int, default=6)
    args = p.parse_args()

    servers = [s.strip() for s in args.servers.split(',')]

    print("=" * 60)
    print("MULTI-QWEN DISTRIBUTED LEARNING")
    print("=" * 60)

    # Discover all models
    all_models = []
    for server in servers:
        models = get_available_models(server)
        if models:
            print(f"[{server}] {len(models)} Qwen models: {models}")
            all_models.extend([(server, m) for m in models])
        else:
            print(f"[{server}] No models found or server offline")

    if not all_models:
        print("\nNo Qwen models found! Make sure LM Studio is running with models loaded.")
        sys.exit(1)

    print(f"\nTotal: {len(all_models)} Qwen instances across {len(servers)} server(s)")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    mgr = Manager()
    shared = mgr.dict()

    procs = []
    for i in range(args.workers):
        proc = Process(target=worker, args=(i, servers, shared))
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
