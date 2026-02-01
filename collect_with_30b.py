#!/usr/bin/env python3
"""
Data collector using Qwen3-VL-30B via LM Studio.
Slower but higher quality extraction.
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
from typing import Dict, List

# LM Studio settings
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-30b"

KNOWLEDGE_DIR = os.path.expanduser("~/.cognitive_ai_knowledge")
TRAINING_FILE = os.path.join(KNOWLEDGE_DIR, "training_data.jsonl")
HASHES_FILE = os.path.join(KNOWLEDGE_DIR, "learned_hashes.txt")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)


def lmstudio_chat(prompt: str, timeout: int = 300) -> str:
    """Chat with LM Studio."""
    try:
        data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024,
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
        print(f"LM Studio error: {e}")
        return ""


def is_duplicate(content: str) -> bool:
    h = hashlib.md5(content.encode()).hexdigest()[:16]
    try:
        if os.path.exists(HASHES_FILE):
            with open(HASHES_FILE) as f:
                return h in f.read()
    except:
        pass
    return False


def mark_learned(content: str):
    h = hashlib.md5(content.encode()).hexdigest()[:16]
    with open(HASHES_FILE, 'a') as f:
        f.write(h + '\n')


def save_training(pairs: List[Dict], source: str):
    count = 0
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
            count += 1
    return count


def extract_qa(content: str, title: str) -> List[Dict]:
    """Extract Q&A pairs using 30B model."""
    prompt = f"""You are a knowledge extractor. From the text below, create 5 diverse question-answer pairs that teach the key concepts.

TEXT: {content[:2500]}

Return ONLY a JSON array with this exact format:
[
  {{"q": "What is...?", "a": "Detailed answer explaining the concept..."}},
  {{"q": "How does...?", "a": "Step by step explanation..."}},
  {{"q": "Why is...?", "a": "Reasoning and explanation..."}}
]

JSON array:"""

    print("  Analyzing with 30B model...", flush=True)
    response = lmstudio_chat(prompt)

    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            pairs = json.loads(match.group())
            if isinstance(pairs, list) and len(pairs) > 0:
                return pairs
    except:
        pass

    return [{"q": f"What is {title}?", "a": content[:500]}]


# Data sources
def fetch_arxiv() -> List[Dict]:
    items = []
    categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'stat.ML']
    cat = random.choice(categories)
    try:
        url = f'http://export.arxiv.org/api/query?search_query=cat:{cat}&start={random.randint(0,300)}&max_results=5'
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
    items = []
    topics = [
        'Machine learning', 'Deep learning', 'Neural network',
        'Artificial intelligence', 'Natural language processing',
        'Computer vision', 'Reinforcement learning', 'Data science',
        'Python programming', 'Algorithm', 'Mathematics', 'Statistics'
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


def fetch_random_wiki() -> List[Dict]:
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


FETCHERS = [fetch_arxiv, fetch_wikipedia, fetch_random_wiki]


def count_existing():
    try:
        with open(TRAINING_FILE) as f:
            return len(f.readlines())
    except:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', type=int, default=50000)
    args = parser.parse_args()

    existing = count_existing()
    target = args.target
    collected = 0

    print("=" * 60)
    print("QWEN3-VL-30B DATA COLLECTOR")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Current data: {existing} pairs")
    print(f"Target: {target} pairs")
    print(f"Note: 30B is slow (~2 min/request) but high quality")
    print("=" * 60)
    print()

    fetcher_idx = 0
    start_time = time.time()

    while existing + collected < target:
        try:
            # Get content
            fetcher = FETCHERS[fetcher_idx % len(FETCHERS)]
            fetcher_idx += 1

            print(f"\n[{fetcher.__name__}] Fetching...", flush=True)
            items = fetcher()

            if not items:
                continue

            for item in items[:2]:  # Process 2 items max per fetch
                content = item.get('content', '')
                title = item.get('title', '')
                source = item.get('source', '')

                if len(content) < 50 or is_duplicate(content):
                    continue

                print(f"[{source}] {title[:50]}...", flush=True)

                # Extract Q&A with 30B
                pairs = extract_qa(content, title)
                mark_learned(content)

                # Save
                added = save_training(pairs, source)
                collected += added

                total = existing + collected
                elapsed = (time.time() - start_time) / 3600
                rate = collected / elapsed if elapsed > 0 else 0

                print(f"  +{added} pairs | Total: {total}/{target} | Rate: {rate:.0f}/hr", flush=True)

                if total >= target:
                    break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

    print("\n" + "=" * 60)
    total = existing + collected
    print(f"Collected {collected} new pairs")
    print(f"Total: {total} pairs")
    if total >= target:
        print("TARGET REACHED! Run: python3 train_my_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
