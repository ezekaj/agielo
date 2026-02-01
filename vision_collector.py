#!/usr/bin/env python3
"""
VISION DATA COLLECTOR
=====================
Uses Qwen3-VL-30B's vision capabilities to learn from images!

Collects:
- Wikipedia diagrams and charts
- ArXiv paper figures
- Educational infographics
- Scientific visualizations

This gives you training data that text-only models can't get!
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
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# LM Studio settings
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-30b"

KNOWLEDGE_DIR = os.path.expanduser("~/.cognitive_ai_knowledge")
TRAINING_FILE = os.path.join(KNOWLEDGE_DIR, "training_data.jsonl")
HASHES_FILE = os.path.join(KNOWLEDGE_DIR, "learned_hashes.txt")
IMAGE_CACHE = os.path.join(KNOWLEDGE_DIR, "image_cache")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE, exist_ok=True)


def download_image(url: str) -> Optional[str]:
    """Download image and return base64."""
    try:
        # Create hash for caching
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        cache_path = os.path.join(IMAGE_CACHE, f"{url_hash}.jpg")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()

            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(data)

            return base64.b64encode(data).decode()
    except Exception as e:
        print(f"  Failed to download image: {e}")
        return None


def vision_chat(prompt: str, image_base64: str, timeout: int = 600) -> str:
    """Chat with vision model about an image."""
    try:
        # LM Studio vision format
        data = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
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
        print(f"  Vision API error: {e}")
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


def save_training(pairs: List[Dict], source: str) -> int:
    count = 0
    for pair in pairs:
        if pair.get('q') and pair.get('a') and len(pair['a']) > 30:
            entry = {
                "prompt": pair['q'],
                "completion": pair['a'],
                "source": source,
                "type": "vision",
                "timestamp": datetime.now().isoformat()
            }
            with open(TRAINING_FILE, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            count += 1
    return count


def analyze_image(image_base64: str, context: str = "") -> List[Dict]:
    """Analyze image and extract Q&A pairs."""

    prompt = f"""Look at this image carefully. {context}

Create 5 educational question-answer pairs about what you see.
Include questions about:
- What the image shows/depicts
- Key concepts or information visible
- How things work (if it's a diagram)
- Important details or patterns
- Practical applications or implications

Return ONLY a JSON array:
[
  {{"q": "What does this image show?", "a": "Detailed description..."}},
  {{"q": "How does X work based on this diagram?", "a": "Explanation..."}},
  ...
]

JSON array:"""

    print("  Analyzing image with vision model...", flush=True)
    response = vision_chat(prompt, image_base64)

    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            pairs = json.loads(match.group())
            if isinstance(pairs, list) and len(pairs) > 0:
                return pairs
    except:
        pass

    # Fallback: just describe the image
    desc_prompt = "Describe this image in detail. What does it show? What are the key elements?"
    description = vision_chat(desc_prompt, image_base64)

    if description:
        return [{"q": "What does this image show?", "a": description}]

    return []


# ============ IMAGE SOURCES ============

def fetch_wikipedia_images() -> List[Dict]:
    """Fetch images from Wikipedia articles."""
    items = []

    topics = [
        'Neural_network', 'Machine_learning', 'Deep_learning',
        'Artificial_intelligence', 'Neuron', 'Brain',
        'Computer_architecture', 'Algorithm', 'Graph_theory',
        'Solar_system', 'DNA', 'Cell_biology', 'Physics',
        'Mathematics', 'Chemistry', 'Engineering'
    ]

    try:
        topic = random.choice(topics)
        # Get page images
        url = f"https://en.wikipedia.org/api/rest_v1/page/media-list/{topic}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())

        for item in data.get('items', [])[:5]:
            if item.get('type') == 'image':
                src = item.get('srcset', [{}])
                if src:
                    # Get largest image
                    img_url = src[-1].get('src', '') if isinstance(src, list) else ''
                    if img_url and not any(x in img_url.lower() for x in ['icon', 'logo', 'flag']):
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        items.append({
                            'url': img_url,
                            'context': f"This is an image from the Wikipedia article about {topic.replace('_', ' ')}.",
                            'source': f'Wikipedia-{topic}'
                        })

    except Exception as e:
        print(f"  Wikipedia fetch error: {e}")

    return items


def fetch_unsplash_educational() -> List[Dict]:
    """Fetch educational images from Unsplash (free API)."""
    items = []

    queries = [
        'science laboratory', 'mathematics', 'technology',
        'computer programming', 'engineering', 'physics',
        'chemistry experiment', 'biology microscope', 'astronomy'
    ]

    try:
        query = random.choice(queries)
        # Unsplash source (no API key needed for basic usage)
        url = f"https://source.unsplash.com/800x600/?{urllib.parse.quote(query)}"

        items.append({
            'url': url,
            'context': f"This is an educational image related to {query}.",
            'source': f'Unsplash-{query}'
        })

    except Exception as e:
        print(f"  Unsplash fetch error: {e}")

    return items


def fetch_wikimedia_diagrams() -> List[Dict]:
    """Fetch diagrams from Wikimedia Commons."""
    items = []

    categories = [
        'Diagrams_of_neural_networks',
        'Machine_learning_diagrams',
        'Computer_science_diagrams',
        'Mathematics_diagrams',
        'Physics_diagrams',
        'Chemistry_diagrams',
        'Biology_diagrams',
        'Flowcharts'
    ]

    try:
        cat = random.choice(categories)
        url = f"https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:{cat}&cmtype=file&cmlimit=10&format=json"

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())

        for member in data.get('query', {}).get('categorymembers', []):
            title = member.get('title', '')
            if title.startswith('File:') and any(ext in title.lower() for ext in ['.png', '.jpg', '.svg']):
                # Get image URL
                img_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(title[5:])}"
                items.append({
                    'url': img_url,
                    'context': f"This is a diagram from Wikimedia Commons: {title[5:]}",
                    'source': f'Wikimedia-{cat}'
                })

    except Exception as e:
        print(f"  Wikimedia fetch error: {e}")

    return items[:3]


FETCHERS = [
    fetch_wikipedia_images,
    fetch_wikimedia_diagrams,
    fetch_unsplash_educational,
]


def count_existing():
    try:
        with open(TRAINING_FILE) as f:
            return len(f.readlines())
    except:
        return 0


def main():
    existing = count_existing()
    collected = 0
    images_processed = 0

    print("=" * 60)
    print("VISION DATA COLLECTOR")
    print("=" * 60)
    print(f"Model: {MODEL} (Vision)")
    print(f"Current data: {existing} pairs")
    print("Collecting from: Wikipedia, Wikimedia, Unsplash")
    print("=" * 60)
    print()
    print("Note: Vision analysis is slow (~2-5 min per image)")
    print("but gives you UNIQUE training data from visual content!")
    print()

    fetcher_idx = 0
    start_time = time.time()

    while True:
        try:
            # Get images
            fetcher = FETCHERS[fetcher_idx % len(FETCHERS)]
            fetcher_idx += 1

            print(f"\n[{fetcher.__name__}] Fetching images...", flush=True)
            items = fetcher()

            if not items:
                print("  No images found, trying next source...")
                time.sleep(2)
                continue

            for item in items[:2]:
                img_url = item.get('url', '')
                context = item.get('context', '')
                source = item.get('source', 'Unknown')

                if not img_url or is_duplicate(img_url):
                    continue

                print(f"\n[{source}]", flush=True)
                print(f"  URL: {img_url[:60]}...", flush=True)

                # Download image
                print("  Downloading...", flush=True)
                image_base64 = download_image(img_url)

                if not image_base64:
                    continue

                # Analyze with vision model
                pairs = analyze_image(image_base64, context)
                mark_learned(img_url)
                images_processed += 1

                if pairs:
                    added = save_training(pairs, source)
                    collected += added

                    total = existing + collected
                    elapsed = (time.time() - start_time) / 3600
                    rate = collected / elapsed if elapsed > 0 else 0

                    print(f"  ✓ +{added} pairs | Total: {total} | Images: {images_processed} | Rate: {rate:.0f}/hr", flush=True)
                else:
                    print("  No Q&A extracted", flush=True)

                time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

    print("\n" + "=" * 60)
    print(f"Session complete!")
    print(f"Images processed: {images_processed}")
    print(f"New pairs collected: {collected}")
    print(f"Total pairs: {existing + collected}")
    print("=" * 60)


if __name__ == "__main__":
    main()
