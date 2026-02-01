#!/usr/bin/env python3
"""
HYBRID QUALITY COLLECTOR
========================
Best of both worlds:
- Ollama (fast) for text analysis
- Qwen-8B (vision) for images

Quality prompts that extract deep understanding:
WHAT, WHY, HOW, WHEN, EXAMPLES, CONNECTIONS
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
from multiprocessing import Process, Value
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


def ollama_chat(prompt: str, timeout: int = 120) -> str:
    """Fast text analysis with Ollama."""
    try:
        data = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()).get("response", "")
    except Exception as e:
        log(f"Ollama error: {e}")
        return ""


def vision_chat(prompt: str, image_base64: str, timeout: int = 300) -> str:
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
        with urllib.request.urlopen(req, timeout=timeout) as r:
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
        if q and a and len(a) >= 30 and len(q) >= 10:
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


# Quality prompts for deep understanding
TEXT_PROMPT = """Analyze this content deeply and create 6 Q&A pairs.

CONTENT: {content}

Create questions covering:
1. WHAT - What is this about? Define key concepts.
2. WHY - Why is this important? Why does it work this way?
3. HOW - How does it work? Step by step explanation.
4. WHEN/WHERE - When is this used? What contexts?
5. EXAMPLES - Give practical examples.
6. CONNECTIONS - How does this relate to other concepts?

Each answer should be 50-150 words with clear explanations.

Return JSON array only:
[{{"q":"specific question?","a":"detailed answer with explanation..."}}]

JSON:"""

IMAGE_PROMPT = """Look at this image carefully and create 6 educational Q&A pairs.

Cover these aspects:
1. WHAT - What does this image show? Describe in detail.
2. WHY - Why is this important? Why is it shown this way?
3. HOW - How does this work? Explain the process/structure.
4. DETAILS - What specific details are visible and what do they mean?
5. APPLICATION - How is this used in real life?
6. MEANING - What concepts does this represent?

Each answer should be detailed (50-150 words).

Return JSON array only:
[{{"q":"question about image?","a":"detailed answer..."}}]

JSON:"""


def analyze_text(content: str, title: str) -> List[Dict]:
    """Deep text analysis with Ollama."""
    prompt = TEXT_PROMPT.format(content=content[:2000])
    response = ollama_chat(prompt)
    return parse_qa(response)


def analyze_image(image_b64: str, context: str) -> List[Dict]:
    """Vision analysis with Qwen-8B."""
    prompt = IMAGE_PROMPT + f"\n\nContext: {context}"
    response = vision_chat(prompt, image_b64)
    return parse_qa(response)


# === DATA SOURCES ===

def fetch_arxiv() -> Optional[Dict]:
    try:
        cat = random.choice(['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV'])
        url = f'http://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&start={random.randint(0,100)}&max_results=5'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read().decode()
        for entry in re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL):
            title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            if title and summary and len(summary.group(1)) > 200:
                return {
                    'type': 'text',
                    'title': html.unescape(title.group(1).strip()[:100]),
                    'content': html.unescape(summary.group(1).strip()),
                    'source': f'ArXiv-{cat}'
                }
    except:
        pass
    return None


def fetch_wikipedia() -> Optional[Dict]:
    topics = [
        'Machine_learning', 'Deep_learning', 'Neural_network',
        'Artificial_intelligence', 'Natural_language_processing',
        'Reinforcement_learning', 'Computer_vision', 'Transformer_model'
    ]
    try:
        topic = random.choice(topics)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        if data.get('extract') and len(data['extract']) > 150:
            return {
                'type': 'text',
                'title': data.get('title', topic),
                'content': data['extract'],
                'source': 'Wikipedia'
            }
    except:
        pass
    return None


def fetch_wiki_image() -> Optional[Dict]:
    topics = ['Neural_network', 'Machine_learning', 'Deep_learning', 'Perceptron']
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
                            'context': f"Diagram from Wikipedia article about {topic.replace('_', ' ')}",
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


def text_worker(counter):
    """Fast text collection with Ollama."""
    log("[TEXT] Worker started - using Ollama")
    fetchers = [fetch_arxiv, fetch_wikipedia]
    idx = 0
    local = 0

    while True:
        try:
            fetcher = fetchers[idx % len(fetchers)]
            idx += 1

            item = fetcher()
            if not item or is_duplicate(item.get('content', '')):
                time.sleep(1)
                continue

            log(f"[TEXT] {item['source']}: {item['title'][:50]}...")
            pairs = analyze_text(item['content'], item['title'])
            mark_learned(item['content'])

            added = save_training(pairs, item['source'], 'text')
            local += added
            with counter.get_lock():
                counter.value += added

            log(f"[TEXT] +{added} pairs | Total: {counter.value}")
            time.sleep(0.5)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"[TEXT] Error: {e}")
            time.sleep(2)


def vision_worker(counter):
    """Vision collection with Qwen-8B."""
    log("[VISION] Worker started - using Qwen-8B")
    local = 0

    while True:
        try:
            item = fetch_wiki_image()
            if not item or is_duplicate(item.get('url', '')):
                time.sleep(5)
                continue

            log(f"[VISION] {item['source']}")
            image_b64 = download_image(item['url'])
            if not image_b64:
                continue

            log("[VISION] Analyzing image...")
            pairs = analyze_image(image_b64, item['context'])
            mark_learned(item['url'])

            added = save_training(pairs, item['source'], 'vision')
            local += added
            with counter.get_lock():
                counter.value += added

            log(f"[VISION] +{added} pairs | Total: {counter.value}")
            time.sleep(2)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"[VISION] Error: {e}")
            time.sleep(5)


def main():
    existing = count_pairs()

    print("=" * 60)
    print("HYBRID QUALITY COLLECTOR")
    print("=" * 60)
    print(f"Text: Ollama ({OLLAMA_MODEL}) - FAST")
    print(f"Vision: LM Studio ({VISION_MODEL})")
    print(f"Current pairs: {existing}")
    print()
    print("Deep analysis extracting:")
    print("  WHAT, WHY, HOW, WHEN, EXAMPLES, CONNECTIONS")
    print("=" * 60)
    print()

    counter = Value('i', existing)

    # Start workers
    text_proc = Process(target=text_worker, args=(counter,))
    vision_proc = Process(target=vision_worker, args=(counter,))

    text_proc.start()
    time.sleep(2)
    vision_proc.start()

    try:
        text_proc.join()
        vision_proc.join()
    except KeyboardInterrupt:
        log("Stopping...")
        text_proc.terminate()
        vision_proc.terminate()

    print(f"\nTotal pairs: {counter.value}")


if __name__ == "__main__":
    main()
