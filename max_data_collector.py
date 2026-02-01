#!/usr/bin/env python3
"""
MAXIMUM DATA COLLECTOR
======================
Collects as much training data as possible from multiple sources.
Optimized for speed - uses fast Ollama model.

Sources:
- ArXiv (AI/ML papers)
- Wikipedia (knowledge base)
- DuckDuckGo (web facts)
- HuggingFace datasets
- Common Crawl samples
- Stack Overflow

Run 24/7 until you have enough data, then train!

Usage: python3 max_data_collector.py --target 50000
"""

import os
import sys
import json
import time
import random
import hashlib
import urllib.request
import urllib.parse
import re
import html
import argparse
from datetime import datetime
from multiprocessing import Process, Manager, Value
from typing import Dict, List

# Settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "ministral-3:8b"  # Fast model for extraction
KNOWLEDGE_DIR = os.path.expanduser("~/.cognitive_ai_knowledge")
TRAINING_FILE = os.path.join(KNOWLEDGE_DIR, "training_data.jsonl")
HASHES_FILE = os.path.join(KNOWLEDGE_DIR, "learned_hashes.txt")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)


def ollama_chat(prompt: str, timeout: int = 60) -> str:
    """Fast Ollama chat."""
    try:
        data = {"model": MODEL, "prompt": prompt, "stream": False}
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode()).get("response", "")
    except:
        return ""


def is_duplicate(content: str) -> bool:
    """Check if content already learned."""
    h = hashlib.md5(content.encode()).hexdigest()[:16]
    try:
        if os.path.exists(HASHES_FILE):
            with open(HASHES_FILE) as f:
                return h in f.read()
    except:
        pass
    return False


def mark_learned(content: str):
    """Mark content as learned."""
    h = hashlib.md5(content.encode()).hexdigest()[:16]
    with open(HASHES_FILE, 'a') as f:
        f.write(h + '\n')


def save_training(pairs: List[Dict], source: str):
    """Save Q&A pairs to training file."""
    for pair in pairs:
        if pair.get('q') and pair.get('a') and len(pair['a']) > 20:
            entry = {
                "prompt": pair['q'],
                "completion": pair['a'],
                "source": source,
                "timestamp": datetime.now().isoformat()
            }
            with open(TRAINING_FILE, 'a') as f:
                f.write(json.dumps(entry) + '\n')


def extract_qa(content: str, title: str) -> List[Dict]:
    """Extract Q&A pairs from content."""
    prompt = f"""Extract 3-5 question-answer pairs from this text. Return JSON array only.

TEXT: {content[:2000]}

Return format: [{{"q":"question?","a":"detailed answer"}}]

JSON:"""

    response = ollama_chat(prompt)

    try:
        # Find JSON array in response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            pairs = json.loads(match.group())
            if isinstance(pairs, list):
                return pairs
    except:
        pass

    # Fallback: create basic Q&A
    return [{"q": f"What is {title}?", "a": content[:500]}]


# ============ DATA SOURCES ============

def fetch_arxiv(category: str = None) -> List[Dict]:
    """Fetch from ArXiv - AI/ML papers."""
    items = []
    categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'stat.ML', 'cs.RO']
    cat = category or random.choice(categories)

    try:
        url = f'http://export.arxiv.org/api/query?search_query=cat:{cat}&start={random.randint(0,500)}&max_results=10'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read().decode()

        for entry in re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL):
            title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            if title and summary:
                items.append({
                    'title': html.unescape(title.group(1).strip()[:150]),
                    'content': html.unescape(summary.group(1).strip()),
                    'source': f'ArXiv-{cat}'
                })
    except:
        pass
    return items


def fetch_wikipedia() -> List[Dict]:
    """Fetch from Wikipedia - diverse topics."""
    items = []
    topics = [
        'Machine learning', 'Artificial intelligence', 'Neural network',
        'Deep learning', 'Natural language processing', 'Computer vision',
        'Reinforcement learning', 'Transformer model', 'BERT', 'GPT',
        'Convolutional neural network', 'Recurrent neural network',
        'Gradient descent', 'Backpropagation', 'Attention mechanism',
        'Generative adversarial network', 'Autoencoder', 'Transfer learning',
        'Python programming', 'Algorithm', 'Data structure', 'Database',
        'Computer science', 'Mathematics', 'Statistics', 'Probability',
        'Linear algebra', 'Calculus', 'Physics', 'Chemistry', 'Biology',
        'History', 'Philosophy', 'Economics', 'Psychology', 'Sociology'
    ]

    try:
        topic = random.choice(topics)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())

        if data.get('extract'):
            items.append({
                'title': topic,
                'content': data['extract'],
                'source': 'Wikipedia'
            })
    except:
        pass
    return items


def fetch_wikipedia_random() -> List[Dict]:
    """Fetch random Wikipedia articles."""
    items = []
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())

        if data.get('extract') and len(data['extract']) > 100:
            items.append({
                'title': data.get('title', 'Unknown'),
                'content': data['extract'],
                'source': 'Wikipedia-Random'
            })
    except:
        pass
    return items


def fetch_duckduckgo() -> List[Dict]:
    """Fetch from DuckDuckGo instant answers."""
    items = []
    queries = [
        'machine learning tutorial', 'python programming guide',
        'neural network explained', 'deep learning basics',
        'data science fundamentals', 'algorithm design patterns',
        'software engineering best practices', 'database optimization',
        'API design principles', 'web development frameworks',
        'cloud computing concepts', 'DevOps practices',
        'cybersecurity fundamentals', 'blockchain technology',
        'quantum computing basics', 'internet of things'
    ]

    try:
        query = random.choice(queries)
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())

        if data.get('Abstract') and len(data['Abstract']) > 50:
            items.append({
                'title': query,
                'content': data['Abstract'],
                'source': 'DuckDuckGo'
            })

        for topic in data.get('RelatedTopics', [])[:5]:
            if isinstance(topic, dict) and topic.get('Text'):
                items.append({
                    'title': query,
                    'content': topic['Text'],
                    'source': 'DuckDuckGo'
                })
    except:
        pass
    return items


def fetch_quotes() -> List[Dict]:
    """Generate Q&A from programming concepts."""
    items = []
    concepts = [
        ("What is recursion in programming?", "Recursion is when a function calls itself to solve smaller instances of the same problem. It requires a base case to stop the recursion and recursive cases that break down the problem."),
        ("Explain Big O notation", "Big O notation describes the upper bound of algorithm complexity. O(1) is constant time, O(n) is linear, O(n²) is quadratic, O(log n) is logarithmic."),
        ("What is a hash table?", "A hash table is a data structure that maps keys to values using a hash function. It provides O(1) average time complexity for insertions, deletions, and lookups."),
        ("Explain the difference between stack and queue", "A stack is LIFO (Last In, First Out) - like a stack of plates. A queue is FIFO (First In, First Out) - like a line of people."),
        ("What is polymorphism in OOP?", "Polymorphism allows objects of different classes to be treated as objects of a common parent class. It enables one interface to be used for different data types."),
        ("Explain REST API principles", "REST APIs use HTTP methods (GET, POST, PUT, DELETE) for CRUD operations. They are stateless, cacheable, and use uniform interfaces with resource-based URLs."),
        ("What is a neural network?", "A neural network is a computational model inspired by biological neurons. It consists of layers of interconnected nodes that process and transform input data to produce outputs."),
        ("Explain gradient descent", "Gradient descent is an optimization algorithm that minimizes a function by iteratively moving in the direction of steepest descent, defined by the negative of the gradient."),
    ]

    q, a = random.choice(concepts)
    items.append({'title': q, 'content': a, 'source': 'Concepts'})
    return items


# All fetchers
FETCHERS = [
    fetch_arxiv,
    fetch_wikipedia,
    fetch_wikipedia_random,
    fetch_duckduckgo,
    fetch_quotes,
]


def worker(worker_id: int, counter, target: int):
    """Data collection worker."""
    hostname = os.uname().nodename
    print(f"[{hostname}:W{worker_id}] Started", flush=True)

    fetcher_idx = worker_id
    local_count = 0

    while counter.value < target:
        try:
            # Rotate through fetchers
            fetcher = FETCHERS[fetcher_idx % len(FETCHERS)]
            fetcher_idx += 1

            items = fetcher()

            for item in items:
                if counter.value >= target:
                    break

                content = item.get('content', '')
                title = item.get('title', '')
                source = item.get('source', 'Unknown')

                if len(content) < 50:
                    continue

                if is_duplicate(content):
                    continue

                # Extract Q&A pairs
                pairs = extract_qa(content, title)

                if pairs:
                    mark_learned(content)
                    save_training(pairs, source)

                    with counter.get_lock():
                        counter.value += len(pairs)

                    local_count += len(pairs)

                    print(f"[W{worker_id}] +{len(pairs)} [{source[:12]}] {title[:40]}... | Total: {counter.value}/{target}", flush=True)

            time.sleep(0.2)  # Small delay

        except KeyboardInterrupt:
            break
        except Exception as e:
            time.sleep(1)

    print(f"[W{worker_id}] Done. Collected {local_count} pairs.", flush=True)


def count_existing():
    """Count existing training pairs."""
    try:
        with open(TRAINING_FILE) as f:
            return len(f.readlines())
    except:
        return 0


def main():
    parser = argparse.ArgumentParser(description='Maximum data collector')
    parser.add_argument('--target', '-t', type=int, default=50000,
                        help='Target number of training pairs')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of parallel workers')
    args = parser.parse_args()

    existing = count_existing()
    target = args.target

    print("=" * 60)
    print("MAXIMUM DATA COLLECTOR")
    print("=" * 60)
    print(f"Current data: {existing} training pairs")
    print(f"Target: {target} training pairs")
    print(f"Need to collect: {max(0, target - existing)} more")
    print(f"Workers: {args.workers}")
    print("=" * 60)
    print()

    if existing >= target:
        print("Target already reached! Ready to train.")
        print(f"Run: python3 train_my_model.py")
        return

    # Shared counter
    manager = Manager()
    counter = Value('i', existing)

    # Start workers
    workers = []
    for i in range(args.workers):
        p = Process(target=worker, args=(i, counter, target))
        p.start()
        workers.append(p)
        time.sleep(0.3)

    try:
        # Monitor progress
        start_time = time.time()
        last_count = existing

        while counter.value < target:
            time.sleep(30)
            current = counter.value
            elapsed = time.time() - start_time
            rate = (current - existing) / (elapsed / 3600) if elapsed > 0 else 0
            remaining = (target - current) / rate if rate > 0 else 0

            print(f"\n[PROGRESS] {current}/{target} ({current/target*100:.1f}%) | "
                  f"Rate: {rate:.0f}/hour | ETA: {remaining:.1f} hours\n", flush=True)

            last_count = current

        print("\n" + "=" * 60)
        print("TARGET REACHED!")
        print("=" * 60)
        print(f"Collected {counter.value} training pairs")
        print(f"Ready to train! Run: python3 train_my_model.py")
        print("=" * 60)

    except KeyboardInterrupt:
        print(f"\nStopped. Collected {counter.value} pairs so far.")
        print("You can resume later or train with current data.")

    # Stop workers
    for p in workers:
        p.terminate()


if __name__ == "__main__":
    main()
