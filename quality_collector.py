#!/usr/bin/env python3
"""
QUALITY DATA COLLECTOR
======================
Sequential processing - no GPU competition.
Alternates between text (Ollama) and vision (LM Studio).

Deep Q&A extraction:
- WHAT, WHY, HOW, WHEN
- EXAMPLES, CONNECTIONS
"""

import os
import sys
import json
import time
import random
import hashlib
import urllib.request
import urllib.parse
import base64
import re
import html
from datetime import datetime
from typing import Dict, List, Optional

# Settings
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "ministral-3:8b"

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
VISION_MODEL = "qwen/qwen3-vl-8b"

KNOWLEDGE_DIR = os.path.expanduser("~/.cognitive_ai_knowledge")
TRAINING_FILE = os.path.join(KNOWLEDGE_DIR, "training_data.jsonl")
HASHES_FILE = os.path.join(KNOWLEDGE_DIR, "learned_hashes.txt")
IMAGE_CACHE = os.path.join(KNOWLEDGE_DIR, "image_cache")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE, exist_ok=True)


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def ollama_chat(prompt: str) -> str:
    """Text analysis with Ollama."""
    try:
        data = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode()).get("response", "")
    except Exception as e:
        log(f"Ollama error: {e}")
        return ""


def vision_chat(prompt: str, image_base64: str) -> str:
    """Vision analysis with Qwen-8B."""
    try:
        data = {
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        req = urllib.request.Request(
            LMSTUDIO_URL,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            result = json.loads(r.read().decode())
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        log(f"Vision error: {e}")
        return ""


def download_image(url: str) -> Optional[str]:
    try:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        cache_path = os.path.join(IMAGE_CACHE, f"{url_hash}.jpg")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            with open(cache_path, 'wb') as f:
                f.write(data)
            return base64.b64encode(data).decode()
    except:
        return None


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


def save_training(pairs: List[Dict], source: str, content_type: str) -> int:
    count = 0
    for pair in pairs:
        q = pair.get('q', '').strip()
        a = pair.get('a', '').strip()
        if q and a and len(a) >= 30:
            entry = {
                "prompt": q,
                "completion": a,
                "source": source,
                "type": content_type,
                "timestamp": datetime.now().isoformat()
            }
            with open(TRAINING_FILE, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            count += 1
    return count


def parse_qa(response: str) -> List[Dict]:
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            pairs = json.loads(match.group())
            if isinstance(pairs, list):
                return [p for p in pairs if p.get('q') and p.get('a')]
    except:
        pass
    return []


# === PROMPTS ===
TEXT_PROMPT = """Create 6 Q&A pairs about this topic with detailed answers.

Topic: {title}
Content: {content}

Cover: WHAT it is, WHY important, HOW it works, EXAMPLES, CONNECTIONS.
Each answer: 50-100 words with clear explanation.

Return JSON only: [{{"q":"?","a":"..."}}]
JSON:"""

IMAGE_PROMPT = """Analyze this image and create 6 Q&A pairs.

Cover: WHAT it shows, WHY important, HOW it works, KEY DETAILS, APPLICATIONS.
Each answer: 50-100 words.

Return JSON only: [{{"q":"?","a":"..."}}]
JSON:"""


# === FETCHERS ===
def fetch_arxiv() -> Optional[Dict]:
    try:
        cat = random.choice(['cs.AI', 'cs.LG', 'cs.CL'])
        url = f'http://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&start={random.randint(0,50)}&max_results=3'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read().decode()
        for entry in re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL):
            title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            if title and summary and len(summary.group(1)) > 150:
                return {
                    'type': 'text',
                    'title': html.unescape(title.group(1).strip()[:80]),
                    'content': html.unescape(summary.group(1).strip()[:1500]),
                    'source': f'ArXiv-{cat}'
                }
    except:
        pass
    return None


def fetch_wikipedia() -> Optional[Dict]:
    topics = ['Machine_learning', 'Deep_learning', 'Neural_network', 'Reinforcement_learning']
    try:
        topic = random.choice(topics)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        if data.get('extract') and len(data['extract']) > 100:
            return {
                'type': 'text',
                'title': data.get('title', topic),
                'content': data['extract'][:1500],
                'source': 'Wikipedia'
            }
    except:
        pass
    return None


def fetch_wiki_image() -> Optional[Dict]:
    topics = ['Neural_network', 'Machine_learning', 'Deep_learning']
    try:
        topic = random.choice(topics)
        url = f"https://en.wikipedia.org/api/rest_v1/page/media-list/{topic}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        for item in data.get('items', []):
            if item.get('type') == 'image':
                srcset = item.get('srcset', [])
                if srcset:
                    img_url = srcset[-1].get('src', '') if isinstance(srcset, list) else ''
                    if img_url and 'icon' not in img_url.lower():
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        return {
                            'type': 'image',
                            'url': img_url,
                            'context': f"From Wikipedia: {topic.replace('_', ' ')}",
                            'source': f'Wikipedia-{topic}'
                        }
    except:
        pass
    return None


def count_pairs() -> int:
    try:
        with open(TRAINING_FILE) as f:
            return len(f.readlines())
    except:
        return 0


def main():
    total = count_pairs()

    print("=" * 60)
    print("QUALITY DATA COLLECTOR")
    print("=" * 60)
    print(f"Text: Ollama ({OLLAMA_MODEL})")
    print(f"Vision: LM Studio ({VISION_MODEL})")
    print(f"Current pairs: {total}")
    print(f"Mode: Sequential (no GPU competition)")
    print("=" * 60)
    print()

    # Alternate: 3 text, 1 vision
    text_fetchers = [fetch_arxiv, fetch_wikipedia]
    cycle = 0
    session = 0

    while True:
        try:
            cycle += 1

            # Every 4th cycle, do vision
            if cycle % 4 == 0:
                log("Fetching image...")
                item = fetch_wiki_image()
                if item and not is_duplicate(item.get('url', '')):
                    log(f"[VISION] {item['source']}")
                    image_b64 = download_image(item['url'])
                    if image_b64:
                        log("Analyzing image...")
                        response = vision_chat(IMAGE_PROMPT, image_b64)
                        pairs = parse_qa(response)
                        mark_learned(item['url'])
                        added = save_training(pairs, item['source'], 'vision')
                        session += added
                        total += added
                        log(f"[VISION] +{added} pairs | Session: {session} | Total: {total}")
            else:
                # Text analysis
                fetcher = text_fetchers[cycle % len(text_fetchers)]
                item = fetcher()
                if item and not is_duplicate(item.get('content', '')):
                    log(f"[TEXT] {item['source']}: {item['title'][:50]}...")
                    prompt = TEXT_PROMPT.format(title=item['title'], content=item['content'])
                    response = ollama_chat(prompt)
                    pairs = parse_qa(response)
                    mark_learned(item['content'])
                    added = save_training(pairs, item['source'], 'text')
                    session += added
                    total += added
                    log(f"[TEXT] +{added} pairs | Session: {session} | Total: {total}")

            time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(3)

    print(f"\nSession: {session} pairs | Total: {total}")


if __name__ == "__main__":
    main()
