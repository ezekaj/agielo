#!/usr/bin/env python3
"""
PREMIUM QUALITY DATA COLLECTOR
==============================
Deep understanding with comprehensive Q&A extraction.
Uses Qwen3-VL-30B for highest quality analysis.

For each piece of content, extracts:
- WHAT: What is this? What does it show/explain?
- WHY: Why is this important? Why does it work this way?
- HOW: How does it work? How is it used?
- WHEN: When is this relevant? When was it discovered/created?
- WHO: Who uses this? Who discovered/created it?
- EXAMPLES: Real-world examples and applications
- CONNECTIONS: How does this relate to other concepts?

Quality > Speed
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

# LM Studio settings
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-30b"

KNOWLEDGE_DIR = os.path.expanduser("~/.cognitive_ai_knowledge")
TRAINING_FILE = os.path.join(KNOWLEDGE_DIR, "premium_training_data.jsonl")
HASHES_FILE = os.path.join(KNOWLEDGE_DIR, "premium_hashes.txt")
IMAGE_CACHE = os.path.join(KNOWLEDGE_DIR, "image_cache")
PROGRESS_FILE = os.path.join(KNOWLEDGE_DIR, "collection_progress.json")

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE, exist_ok=True)


def log(msg: str):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def lmstudio_chat(messages: List[Dict], timeout: int = 600) -> str:
    """Chat with LM Studio - supports both text and vision."""
    try:
        data = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,  # Comprehensive but not too long
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
        log(f"API error: {e}")
        return ""


def download_image(url: str) -> Optional[str]:
    """Download image and return base64."""
    try:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        cache_path = os.path.join(IMAGE_CACHE, f"{url_hash}.jpg")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
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

        # Quality check - skip short or low-quality answers
        if not q or not a:
            continue
        if len(a) < 50:  # Minimum answer length
            continue
        if len(q) < 10:  # Minimum question length
            continue

        entry = {
            "prompt": q,
            "completion": a,
            "source": source,
            "type": content_type,
            "quality": "premium",
            "timestamp": datetime.now().isoformat()
        }

        with open(TRAINING_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        count += 1

    return count


def deep_analyze_text(content: str, title: str, source: str) -> List[Dict]:
    """Deep analysis of text content with comprehensive Q&A."""

    prompt = f"""Create 8 Q&A pairs about this topic. Each answer should be 50-150 words with explanations.

Topic: {title}
Content: {content[:1500]}

Include:
- What is it and why is it important?
- How does it work?
- What are examples or applications?
- How does it connect to related concepts?

Return JSON array only:
[{{"q":"question?","a":"detailed answer..."}}]

JSON:"""

    log("  Performing deep text analysis...")

    messages = [{"role": "user", "content": prompt}]
    response = lmstudio_chat(messages)

    pairs = parse_qa_response(response)
    return pairs


def deep_analyze_image(image_base64: str, context: str, source: str) -> List[Dict]:
    """Deep analysis of image with comprehensive Q&A."""

    prompt = f"""Analyze this image and create 8 educational Q&A pairs.

Context: {context}

For each question, provide a detailed answer (50-150 words) covering:
- What the image shows
- Why it's important
- How it works
- Practical applications

Return JSON array only:
[{{"q":"question about the image?","a":"detailed answer..."}}]

JSON:"""

    log("  Performing deep image analysis...")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": prompt}
        ]
    }]

    response = lmstudio_chat(messages)
    return parse_qa_response(response)


def generate_follow_ups(content: str, title: str, initial_pairs: List[Dict]) -> List[Dict]:
    """Generate follow-up questions based on initial Q&A."""

    context = "\n".join([f"Q: {p['q']}\nA: {p['a'][:200]}..." for p in initial_pairs])

    prompt = f"""Based on this content and initial Q&A, generate 3 deeper follow-up questions.

Topic: {title}

Initial Q&A:
{context}

Create 3 follow-up questions that:
1. Go deeper into the concepts
2. Ask about implications or applications
3. Connect to related topics

Return JSON array:
[{{"q": "Follow-up question?", "a": "Detailed answer..."}}]

JSON:"""

    messages = [{"role": "user", "content": prompt}]
    response = lmstudio_chat(messages, timeout=300)
    return parse_qa_response(response)


def parse_qa_response(response: str) -> List[Dict]:
    """Parse Q&A JSON from response."""
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            pairs = json.loads(match.group())
            if isinstance(pairs, list):
                return [p for p in pairs if p.get('q') and p.get('a')]
    except:
        pass
    return []


# ============ CONTENT SOURCES ============

def fetch_arxiv_paper() -> Optional[Dict]:
    """Fetch a high-quality ArXiv paper."""
    categories = ['cs.AI', 'cs.LG', 'cs.CL']  # Focus on AI/ML

    try:
        cat = random.choice(categories)
        # Get recent papers (more relevant)
        url = f'http://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&start={random.randint(0,50)}&max_results=5'

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read().decode()

        entries = re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL)

        for entry in entries:
            title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)

            if title and summary:
                title_text = html.unescape(title.group(1).strip())
                summary_text = html.unescape(summary.group(1).strip())

                # Skip if too short
                if len(summary_text) < 200:
                    continue

                return {
                    'type': 'text',
                    'title': title_text,
                    'content': summary_text,
                    'source': f'ArXiv-{cat}'
                }

    except Exception as e:
        log(f"ArXiv fetch error: {e}")

    return None


def fetch_wikipedia_article() -> Optional[Dict]:
    """Fetch a high-quality Wikipedia article."""
    topics = [
        'Machine_learning', 'Deep_learning', 'Neural_network',
        'Artificial_intelligence', 'Natural_language_processing',
        'Transformer_(machine_learning_model)', 'Reinforcement_learning',
        'Convolutional_neural_network', 'Backpropagation',
        'Gradient_descent', 'Large_language_model'
    ]

    try:
        topic = random.choice(topics)

        # Get full extract, not just summary
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())

        if data.get('extract') and len(data['extract']) > 200:
            return {
                'type': 'text',
                'title': data.get('title', topic.replace('_', ' ')),
                'content': data['extract'],
                'source': 'Wikipedia'
            }

    except Exception as e:
        log(f"Wikipedia fetch error: {e}")

    return None


def fetch_wikipedia_image() -> Optional[Dict]:
    """Fetch a Wikipedia image with context."""
    topics = [
        'Neural_network', 'Machine_learning', 'Deep_learning',
        'Artificial_neural_network', 'Perceptron', 'Backpropagation'
    ]

    try:
        topic = random.choice(topics)
        url = f"https://en.wikipedia.org/api/rest_v1/page/media-list/{topic}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())

        for item in data.get('items', []):
            if item.get('type') == 'image':
                srcset = item.get('srcset', [])
                if srcset and isinstance(srcset, list):
                    img_url = srcset[-1].get('src', '')
                    if img_url and not any(x in img_url.lower() for x in ['icon', 'logo', 'flag', 'arrow']):
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url

                        caption = item.get('caption', {})
                        if isinstance(caption, dict):
                            caption = caption.get('text', '')

                        return {
                            'type': 'image',
                            'url': img_url,
                            'title': f"{topic.replace('_', ' ')} diagram",
                            'context': f"This image is from the Wikipedia article about {topic.replace('_', ' ')}. {caption}",
                            'source': f'Wikipedia-{topic}'
                        }

    except Exception as e:
        log(f"Wikipedia image fetch error: {e}")

    return None


def save_progress(stats: Dict):
    """Save collection progress."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)


def load_progress() -> Dict:
    """Load collection progress."""
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except:
        return {
            "total_pairs": 0,
            "text_items": 0,
            "image_items": 0,
            "start_time": datetime.now().isoformat()
        }


def count_training_pairs() -> int:
    """Count existing training pairs."""
    try:
        with open(TRAINING_FILE) as f:
            return len(f.readlines())
    except:
        return 0


def main():
    stats = load_progress()
    stats["total_pairs"] = count_training_pairs()

    print("=" * 70)
    print("PREMIUM QUALITY DATA COLLECTOR")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Mode: DEEP ANALYSIS (Quality > Speed)")
    print(f"Current premium pairs: {stats['total_pairs']}")
    print()
    print("Each item will be deeply analyzed for:")
    print("  - WHAT, WHY, HOW, WHEN questions")
    print("  - Examples and applications")
    print("  - Connections and implications")
    print("  - 10+ Q&A pairs per item")
    print("=" * 70)
    print()

    # Alternate between sources
    fetchers = [
        ('ArXiv Paper', fetch_arxiv_paper),
        ('Wikipedia Article', fetch_wikipedia_article),
        ('Wikipedia Image', fetch_wikipedia_image),
    ]

    fetcher_idx = 0
    session_pairs = 0

    while True:
        try:
            name, fetcher = fetchers[fetcher_idx % len(fetchers)]
            fetcher_idx += 1

            log(f"Fetching: {name}...")
            item = fetcher()

            if not item:
                log("  No item found, trying next source...")
                time.sleep(2)
                continue

            title = item.get('title', 'Unknown')
            source = item.get('source', 'Unknown')

            # Check duplicate
            content_key = item.get('content', '') or item.get('url', '')
            if is_duplicate(content_key):
                log(f"  Skipping duplicate: {title[:50]}...")
                continue

            log(f"Processing: {title[:60]}...")
            log(f"  Source: {source}")

            # Analyze based on type
            if item['type'] == 'image':
                image_b64 = download_image(item['url'])
                if not image_b64:
                    log("  Failed to download image")
                    continue

                pairs = deep_analyze_image(image_b64, item.get('context', ''), source)
                content_type = 'vision'

            else:  # text
                pairs = deep_analyze_text(item['content'], title, source)
                content_type = 'text'

            # Save
            mark_learned(content_key)
            added = save_training(pairs, source, content_type)

            stats["total_pairs"] += added
            session_pairs += added

            if content_type == 'vision':
                stats["image_items"] = stats.get("image_items", 0) + 1
            else:
                stats["text_items"] = stats.get("text_items", 0) + 1

            save_progress(stats)

            log(f"  ✓ Extracted {added} premium Q&A pairs")
            log(f"  Total: {stats['total_pairs']} pairs | Session: {session_pairs}")
            print()

            # Small delay between items
            time.sleep(2)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(5)

    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    print(f"Session pairs collected: {session_pairs}")
    print(f"Total premium pairs: {stats['total_pairs']}")
    print(f"Text items processed: {stats.get('text_items', 0)}")
    print(f"Image items processed: {stats.get('image_items', 0)}")
    print()
    print(f"Training data: {TRAINING_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
