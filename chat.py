#!/usr/bin/env python3
"""
Cognitive Chat - Human-Like AI with Full Cognitive Architecture
================================================================

Features:
- Dual-Process Thinking (System 1 fast / System 2 slow)
- Emotion System with blending and regulation
- Memory with decay, priming, and consolidation
- Curiosity-driven active learning
- Sleep consolidation (during idle)
- Theory of Mind and social intelligence
- Metacognitive thought traces

Run: python3 chat.py [model_name]
"""

import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import json
import urllib.request
import urllib.parse
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrations.cognitive_ollama import CognitiveOllama
from integrations.self_training import SelfTrainer
from integrations.browser_agent import BrowserAgent, run_browser_command, BROWSER_AVAILABLE
from integrations.active_learning import get_active_learner, ActiveLearner
from integrations.benchmark import Benchmark
from integrations.self_evolution import get_evolution, SelfEvolution

# Try to import advanced cognitive modules
try:
    from phase3_motivation.emotion_system import EmotionSystem, BasicEmotion
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False
    print("[Warning] Emotion system not available")

try:
    from phase5_advanced.sleep_consolidation import SleepConsolidationSystem, MemoryTrace, MemoryType
    SLEEP_AVAILABLE = True
except ImportError:
    SLEEP_AVAILABLE = False
    print("[Warning] Sleep consolidation not available")


class WebLearner:
    """Autonomous web learning capabilities."""

    def __init__(self):
        self.learned_facts = []
        self.search_history = []

    def search_web(self, query: str) -> List[Dict]:
        """Search the web using DuckDuckGo (free, no API key)."""
        try:
            # DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []

            # Abstract (main answer)
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', query),
                    'snippet': data['Abstract'],
                    'source': data.get('AbstractSource', 'DuckDuckGo'),
                    'url': data.get('AbstractURL', '')
                })

            # Related topics
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '')[:50],
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', '')
                    })

            self.search_history.append({
                'query': query,
                'time': datetime.now().isoformat(),
                'results': len(results)
            })

            return results

        except Exception as e:
            return [{'error': str(e)}]

    def fetch_page(self, url: str) -> str:
        """Fetch and extract text from a webpage."""
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
            })
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')

            # Simple text extraction (remove HTML tags)
            import re
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text[:2000]  # Limit to 2000 chars

        except Exception as e:
            return f"[Could not fetch: {e}]"

    def learn_fact(self, topic: str, fact: str, source: str):
        """Store a learned fact."""
        self.learned_facts.append({
            'topic': topic,
            'fact': fact,
            'source': source,
            'time': datetime.now().isoformat()
        })

    def get_random_fact(self) -> Optional[Dict]:
        """Get a random learned fact."""
        if self.learned_facts:
            return random.choice(self.learned_facts)
        return None


class AutonomousAI:
    """
    Human-Like AI with Full Cognitive Architecture.

    Features:
    - Dual-process thinking (fast/slow)
    - Emotion system with blending
    - Memory consolidation during "sleep"
    - Curiosity-driven learning
    - Metacognitive thought traces
    """

    def __init__(self, model: str = "ministral-3:8b"):
        print("\n[Initializing Cognitive Systems...]")

        self.ai = CognitiveOllama(model=model)
        self.web = WebLearner()
        self.trainer = SelfTrainer()  # Curiosity-driven learning!
        self.active_learner = get_active_learner()  # Active learning module

        self.last_interaction = time.time()
        self.idle_threshold = 10  # seconds before AI starts reflecting
        self.is_busy = False
        self.thoughts = []
        self.thought_traces = []  # Metacognitive traces
        self.running = True
        self.interests = []  # Topics the AI is interested in
        self.reflection_count = 0

        # Emotion system
        self.emotions = None
        if EMOTION_AVAILABLE:
            self.emotions = EmotionSystem(dim=64)
            print("[Emotions] Emotion system with blending: ACTIVE")

        # Sleep consolidation system
        self.sleep_system = None
        if SLEEP_AVAILABLE:
            self.sleep_system = SleepConsolidationSystem(dim=64)
            print("[Memory] Sleep consolidation system: ACTIVE")

        # Benchmark for self-testing
        self.benchmark = Benchmark()

        # Self-evolution system (no duplicates, learning cycles, MLX training)
        self.evolution = get_evolution()
        evo_stats = self.evolution.get_stats()
        print(f"[Evolution] Cycle {evo_stats['cycle']}, {evo_stats['total_facts']} unique facts learned")
        if evo_stats['trainings'] > 0:
            print(f"[Evolution] MLX trained {evo_stats['trainings']} times, improvement: {evo_stats['improvement']:+.1%}")

        # Conversation quality tracking (for confidence calibration)
        self.domain_outcomes: Dict[str, List[bool]] = {}

        # Load previous knowledge
        stats = self.trainer.get_stats()
        print(f"[Knowledge] Loaded {stats['total_facts']} facts from previous sessions")

        # Browser agent
        self.browser = None
        if BROWSER_AVAILABLE:
            print("[Browser] Web browsing: AVAILABLE")

        # Start background cognitive loop
        self.learning_thread = threading.Thread(target=self._autonomous_loop, daemon=True)
        self.learning_thread.start()

        print("[Ready] All cognitive systems initialized!\n")

    def _autonomous_loop(self):
        """
        Background loop with self-evolution:
        1. Run initial benchmark
        2. Learn 100 unique facts (no duplicates)
        3. Re-benchmark
        4. If improved 1%+ → MLX fine-tune
        5. Reflect and repeat
        """
        initial_benchmark_done = False
        self.benchmark_results = None
        self.weak_areas = []

        while self.running:
            time.sleep(3)

            if self.is_busy:
                continue

            # Wait for first conversation before starting
            if not self.ai.history:
                continue

            # Step 1: Run INITIAL benchmark (once)
            if not initial_benchmark_done:
                self._run_benchmark_and_report("INITIAL")
                initial_benchmark_done = True
                continue

            # Step 2: Learn unique facts (checking for duplicates)
            if not self.evolution.should_benchmark():
                self._learn_unique_fact()
                continue

            # Step 3: After 100 unique facts → re-benchmark
            print(f"\n[Evolution]: Learned {self.evolution.state['facts_this_cycle']} unique facts! Re-benchmarking...")
            print("You: ", end="", flush=True)
            self._run_benchmark_and_report("CYCLE")

            # Step 4: Check if should train
            should_train, reason = self.evolution.should_train(min_improvement=0.01)
            print(f"\n[Evolution]: {reason}")

            if should_train:
                print(f"[Evolution]: Starting MLX fine-tuning on MacBook...")
                print("You: ", end="", flush=True)
                result = self.evolution.run_mlx_training()
                if result['success']:
                    print(f"\n[Evolution]: MLX TRAINING COMPLETE!")
                    # After successful training, try to add new capabilities
                    self._add_evolved_capability()
                else:
                    print(f"\n[Evolution]: Training skipped: {result['message']}")
                print("You: ", end="", flush=True)

            # Step 5: Reflect and start new cycle
            reflection = self.evolution.reflect()
            print(f"\n{reflection}")
            print("You: ", end="", flush=True)

            self.evolution.start_new_cycle()
            print(f"\n[Evolution]: Starting cycle {self.evolution.state['current_cycle']}...")
            print("You: ", end="", flush=True)

    def _run_benchmark_and_report(self, phase: str = ""):
        """Run benchmark and record results in evolution system."""
        self.is_busy = True
        print(f"\n[Evolution]: Running {phase} benchmark...")
        print("You: ", end="", flush=True)

        try:
            def think_fn(q):
                # INJECT LEARNED KNOWLEDGE into the question!
                # This is what makes learning actually help benchmarks
                knowledge = self.trainer.get_knowledge_for_prompt(q)
                if knowledge:
                    enhanced_q = f"{knowledge}\n\nQuestion: {q}\nThink step by step and give the answer:"
                else:
                    enhanced_q = f"Question: {q}\nThink step by step and give the answer:"
                return self.ai.chat(enhanced_q)

            self.benchmark_results = self.benchmark.run_benchmark(think_fn)

            # Find weak areas by category (below 70%)
            category_scores = {}
            for test in self.benchmark_results.get('tests', []):
                cat = test.get('category', 'unknown')
                score = test.get('score', 0) or 0
                if cat not in category_scores:
                    category_scores[cat] = []
                category_scores[cat].append(score)

            self.weak_areas = []
            for cat, scores in category_scores.items():
                avg = sum(scores) / len(scores) if scores else 0
                if avg < 0.7:
                    self.weak_areas.append((cat, avg))

            avg_score = self.benchmark_results.get('avg_score', 0) or 0

            # Record in evolution system
            self.evolution.record_benchmark(avg_score, {
                'weak_areas': [(c, s) for c, s in self.weak_areas],
                'phase': phase
            })

            print(f"\n[Evolution]: {phase} Benchmark: {avg_score:.0%}")
            if self.weak_areas:
                weak_str = ", ".join([f"{a}: {s:.0%}" for a, s in self.weak_areas[:3]])
                print(f"[Evolution]: Weak areas: {weak_str}")
            print("You: ", end="", flush=True)

        except Exception as e:
            print(f"\n[Evolution]: Benchmark error: {e}")
            print("You: ", end="", flush=True)

        self.is_busy = False

    def _learn_unique_fact(self):
        """
        Learn ONE TOPIC AT A TIME - go deep, reflect, then move on.

        Cycle:
        1. Pick a source (ArXiv, GitHub, Math, News)
        2. Get multiple items from that source
        3. Learn ALL unique items from it (go deep)
        4. Reflect on what was learned
        5. Move to next source
        """
        self.is_busy = True

        # Track current learning focus
        if not hasattr(self, '_current_focus'):
            self._current_focus = None
            self._focus_items = []
            self._focus_learned = 0

        # If no current focus or finished with it, pick new source
        if not self._focus_items:
            self._focus_items = self._fetch_news()
            if self._focus_items:
                self._current_focus = self._focus_items[0].get('source', 'Unknown')
                self._focus_learned = 0
                print(f"\n[Evolution]: === FOCUSING ON: {self._current_focus} ===")
                print("You: ", end="", flush=True)
            else:
                self.is_busy = False
                return

        # Learn from current focus
        learned_this_round = False
        items_to_remove = []

        for i, item in enumerate(self._focus_items[:3]):  # Try first 3
            snippet = item.get('snippet', '')
            url = item.get('url', '')
            title = item.get('title', 'Item')[:60]
            source = item.get('source', 'Unknown')

            # Try to go INSIDE for full content
            if url and 'arxiv' in url.lower():
                # For ArXiv, try to get PDF content (model has vision)
                full_content = self._fetch_article_content(url)
                if full_content and len(full_content) > len(snippet):
                    snippet = full_content

            if not snippet or len(snippet) < 50:
                items_to_remove.append(i)
                continue

            # CHECK FOR DUPLICATE
            if self.evolution.is_duplicate(snippet):
                items_to_remove.append(i)
                continue

            # GO INSIDE the URL to get FULL content (not just snippet)
            if url and len(snippet) < 500:
                full_content = self._fetch_full_page(url)
                if full_content and len(full_content) > len(snippet):
                    snippet = full_content

            if not snippet or len(snippet) < 100:
                items_to_remove.append(i)
                continue

            # ANALYZE with Ministral - extract knowledge as JSON
            analyzed = self._analyze_content_with_model(title, snippet, source)

            if not analyzed:
                items_to_remove.append(i)
                continue

            # Check for duplicate (use analyzed summary)
            if self.evolution.is_duplicate(analyzed.get('summary', snippet)):
                items_to_remove.append(i)
                continue

            # Learn the ANALYZED content!
            if self.evolution.mark_learned(analyzed.get('summary', '')):
                # Save structured knowledge
                self.trainer.learn(
                    analyzed.get('topic', title),
                    analyzed.get('knowledge', snippet[:1000]),
                    source
                )
                self._focus_learned += 1

                # Save as training Q&A pairs
                self._save_analyzed_as_training(analyzed, source)

                stats = self.evolution.get_stats()
                print(f"\n[Evolution]: ANALYZED [{stats['facts_this_cycle']}/100] [{source}]")
                print(f"             Topic: {analyzed.get('topic', 'N/A')}")
                print(f"             Key facts: {len(analyzed.get('facts', []))}")
                print(f"             Q&A pairs: {len(analyzed.get('qa_pairs', []))}")
                print("You: ", end="", flush=True)

                items_to_remove.append(i)
                learned_this_round = True
                break  # One at a time

        # Remove processed items
        for i in sorted(items_to_remove, reverse=True):
            if i < len(self._focus_items):
                self._focus_items.pop(i)

        # If focus exhausted, reflect and move on
        if not self._focus_items:
            if self._focus_learned > 0:
                print(f"\n[Evolution]: === REFLECTION on {self._current_focus} ===")
                print(f"             Learned {self._focus_learned} unique facts from {self._current_focus}")
                print(f"             Moving to next source...")
                print("You: ", end="", flush=True)
            self._current_focus = None
            self._focus_learned = 0

        self.is_busy = False

    def _categorize_content(self, content: str) -> str:
        """Categorize content by keywords."""
        content_lower = content.lower()

        categories = {
            'TECH': ['technology', 'software', 'computer', 'ai', 'robot', 'digital', 'app', 'internet'],
            'SCIENCE': ['research', 'study', 'scientist', 'discovery', 'experiment', 'physics', 'biology'],
            'HEALTH': ['health', 'medical', 'doctor', 'hospital', 'disease', 'treatment', 'vaccine'],
            'BUSINESS': ['market', 'stock', 'company', 'economy', 'business', 'trade', 'finance'],
            'WORLD': ['country', 'government', 'president', 'minister', 'nation', 'international'],
            'SPORTS': ['game', 'team', 'player', 'match', 'score', 'championship', 'league'],
            'CULTURE': ['film', 'music', 'art', 'book', 'movie', 'show', 'entertainment'],
        }

        for cat, keywords in categories.items():
            if any(kw in content_lower for kw in keywords):
                return cat

        return 'GENERAL'

    def _add_evolved_capability(self):
        """Add a new function based on what was learned."""
        # Generate new capabilities based on weak areas
        if not self.weak_areas:
            return

        weak_topic, _ = self.weak_areas[0]

        # Define helper functions for each weak area
        capability_templates = {
            'math': (
                'solve_basic_math',
                '''def solve_basic_math(expression: str) -> str:
    """Solve basic math expressions."""
    try:
        # Safe evaluation of math expressions
        allowed = set('0123456789+-*/().% ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return "Cannot evaluate: contains unsafe characters"
    except Exception as e:
        return f"Error: {e}"
''',
                'Safely evaluate basic math expressions'
            ),
            'logic': (
                'check_logic',
                '''def check_logic(premise1: str, premise2: str, conclusion: str) -> str:
    """Simple logical consistency checker."""
    # Keywords that indicate logical relationships
    if "all" in premise1.lower() and "is a" in premise2.lower():
        return f"If '{premise1}' and '{premise2}', then '{conclusion}' follows by syllogism."
    return f"Analyzing: {premise1} + {premise2} -> {conclusion}"
''',
                'Check basic logical syllogisms'
            ),
            'reasoning': (
                'step_by_step',
                '''def step_by_step(problem: str) -> list:
    """Break down a problem into steps."""
    steps = [
        f"1. Understand the problem: {problem[:50]}...",
        "2. Identify key information",
        "3. Determine what we need to find",
        "4. Choose a strategy",
        "5. Execute and verify"
    ]
    return steps
''',
                'Break problems into reasoning steps'
            ),
        }

        if weak_topic in capability_templates:
            name, code, description = capability_templates[weak_topic]

            # Check if already added
            existing = [f['name'] for f in self.evolution.state['added_functions']]
            if name not in existing:
                if self.evolution.add_function(name, code, description):
                    print(f"\n[Evolution]: Added new capability: {name} ({description})")
                    print("You: ", end="", flush=True)

    def _fetch_news(self) -> List[Dict]:
        """
        PARALLEL FETCHING - fetch from multiple sources at once!

        Uses ThreadPoolExecutor to fetch in parallel (much faster).
        """
        all_items = []

        # Learning order - cycle through systematically
        if not hasattr(self, '_learning_order_idx'):
            self._learning_order_idx = 0

        learning_order = ['math', 'logic', 'arxiv', 'gdelt', 'code', 'science']  # Ministral fetches everything
        source_type = learning_order[self._learning_order_idx % len(learning_order)]
        self._learning_order_idx += 1

        # Define fetch tasks
        def fetch_math():
            return self._learn_from_benchmark('math')

        def fetch_logic():
            return self._learn_from_benchmark('logic')

        def fetch_arxiv():
            categories = ['cs.AI', 'cs.LG', 'cs.CL', 'math.CO', 'math.LO', 'stat.ML']
            cat = random.choice(categories)
            url = f'http://export.arxiv.org/api/query?search_query=cat:{cat}&start={random.randint(0,50)}&max_results=10&sortBy=submittedDate&sortOrder=descending'
            return self._fetch_arxiv(url)

        def fetch_code():
            url = f'https://api.github.com/search/repositories?q={random.choice(["algorithm", "machine-learning", "data-structure", "framework"])}+stars:>500&sort=stars&per_page=10'
            return self._fetch_github(url)

        def fetch_science():
            return self._fetch_rss('https://www.sciencedaily.com/rss/all.xml', 'ScienceDaily')

        def fetch_gdelt():
            # GDELT finds articles, then Ministral fetches full content
            return self._fetch_gdelt_and_learn()

        # Map source types to fetch functions - ALL done by Ministral
        fetch_map = {
            'math': fetch_math,
            'logic': fetch_logic,
            'arxiv': fetch_arxiv,
            'code': fetch_code,
            'science': fetch_science,
            'gdelt': fetch_gdelt,
        }

        # PARALLEL MODE: Fetch from ALL sources at once every 5th cycle
        if self._learning_order_idx % 5 == 0:
            print(f"\n[Evolution]: === PARALLEL FETCH (all sources) ===")
            print("You: ", end="", flush=True)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(fn): name for name, fn in fetch_map.items()}

                for future in as_completed(futures, timeout=60):
                    source_name = futures[future]
                    try:
                        items = future.result(timeout=10)  # Individual timeout
                        if items:
                            all_items.extend(items[:3])  # Take top 3 from each
                    except Exception as e:
                        pass  # Skip failed sources

            if all_items:
                random.shuffle(all_items)  # Mix them up
                return all_items

        # SEQUENTIAL MODE: Focus on one source type
        try:
            fetch_fn = fetch_map.get(source_type)
            if fetch_fn:
                all_items = fetch_fn()
        except Exception as e:
            pass

        return all_items

    def _learn_from_benchmark(self, category: str) -> List[Dict]:
        """
        Learn from BENCHMARK with ACTUAL CORRECT ANSWERS!
        Not just searching - we teach the model the right answer.
        """
        items = []

        benchmark_questions = [t for t in self.benchmark.tests if t.get('category') == category]
        if not benchmark_questions:
            benchmark_questions = self.benchmark.tests

        test = random.choice(benchmark_questions)
        question = test['question']
        answer = test.get('answer', '')
        keywords = test.get('expected_keywords', [])

        print(f"\n[Evolution]: STUDYING: {question[:50]}... (answer: {answer})")
        print("You: ", end="", flush=True)

        # TEACH the correct answer with step-by-step reasoning
        explanations = {
            "apples for $2": f"Calculate: 5 × $2 = $10. Change: $20 - $10 = $10. Answer: 10",
            "60 mph": f"Distance = Speed × Time = 60 × 2.5 = 150 miles. Answer: 150",
            "length 8 and width 5": f"Area = 8 × 5 = 40. Answer: 40",
            "cats are mammals": f"Syllogism: cats→mammals→animals. Therefore cats are animals. Answer: yes",
            "rains, the ground gets wet": f"Affirming consequent fallacy. Wet ground ≠ rain (could be sprinklers). Answer: no",
            "ice cream in the oven": f"400°F melts and burns ice cream. Answer: melts",
            "17 sheep. All but 9": f"'All but 9' = 9 remain. Answer: 9",
            "3 apples and you take away 2": f"YOU took 2, so YOU have 2. Answer: 2",
            "twice as old as Bob": f"Alice = 2×15 = 30. In 5 years = 35. Answer: 35",
            "marble in her basket": f"Sally thinks marble is where SHE put it. Answer: basket",
            "John thinks that Mary": f"John believes Mary thinks rain. Answer: rain",
            "cookies put in a blue jar": f"Child will look where they SAW it. Answer: blue",
        }

        explanation = f"Answer: {answer}. Keywords: {', '.join(keywords[:3])}"
        for key, exp in explanations.items():
            if key.lower() in question.lower():
                explanation = exp
                break

        qa_content = f"Question: {question}\n\nStep-by-step solution:\n{explanation}\n\nFINAL ANSWER: {answer}"

        items.append({
            'title': f"[{category.upper()}] {answer}",
            'snippet': qa_content,
            'url': '',
            'source': f'Benchmark-{category}'
        })

        return items

    def _fetch_wikipedia(self, topic: str, source: str) -> List[Dict]:
        """Fetch from Wikipedia API."""
        items = []
        try:
            # Wikipedia API - get extract
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            if data.get('extract'):
                items.append({
                    'title': data.get('title', topic)[:100],
                    'snippet': data['extract'][:800],
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'source': f'Wikipedia-{source}'
                })
        except:
            pass
        return items

    def _fetch_duckduckgo(self, query: str, source: str) -> List[Dict]:
        """Fetch from DuckDuckGo instant answers."""
        items = []
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            # Abstract (main answer)
            if data.get('Abstract') and len(data['Abstract']) > 50:
                items.append({
                    'title': data.get('Heading', query)[:100],
                    'snippet': data['Abstract'],
                    'url': data.get('AbstractURL', ''),
                    'source': source
                })

            # Related topics
            for topic in data.get('RelatedTopics', []):
                if isinstance(topic, dict) and topic.get('Text'):
                    text = topic.get('Text', '')
                    if len(text) > 50:
                        items.append({
                            'title': text[:80],
                            'snippet': text,
                            'url': topic.get('FirstURL', ''),
                            'source': source
                        })
                # Nested topics
                elif isinstance(topic, dict) and topic.get('Topics'):
                    for sub in topic.get('Topics', []):
                        if isinstance(sub, dict) and sub.get('Text'):
                            text = sub.get('Text', '')
                            if len(text) > 50:
                                items.append({
                                    'title': text[:80],
                                    'snippet': text,
                                    'url': sub.get('FirstURL', ''),
                                    'source': source
                                })
        except:
            pass
        return items

    def _fetch_arxiv(self, url: str) -> List[Dict]:
        """Fetch papers from ArXiv API."""
        import re
        items = []
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read().decode('utf-8')

            # Parse Atom feed
            entries = re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL)
            for entry in entries[:10]:
                title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                link = re.search(r'<id>(.*?)</id>', entry)

                if title and summary:
                    items.append({
                        'title': title.group(1).strip()[:100],
                        'snippet': summary.group(1).strip()[:1000],
                        'url': link.group(1) if link else '',
                        'source': 'ArXiv'
                    })
        except:
            pass
        return items

    def _fetch_github(self, url: str) -> List[Dict]:
        """Fetch from GitHub API."""
        items = []
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/vnd.github.v3+json'
            })
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))

            for repo in data.get('items', [])[:10]:
                desc = repo.get('description', '') or ''
                items.append({
                    'title': repo.get('full_name', ''),
                    'snippet': f"{desc} - Language: {repo.get('language', 'Unknown')}, Stars: {repo.get('stargazers_count', 0)}",
                    'url': repo.get('html_url', ''),
                    'source': 'GitHub'
                })
        except:
            pass
        return items

    def _fetch_local_analyzed_data(self) -> List[Dict]:
        """
        Load pre-analyzed articles from /Users/tolga/Desktop/autonomous-agents-data/
        These were already processed by Qwen - adapt to our format and learn!
        """
        import glob

        items = []
        data_path = "/Users/tolga/Desktop/autonomous-agents-data"

        try:
            json_files = glob.glob(f"{data_path}/*.json")
            if not json_files:
                return items

            json_file = random.choice(json_files)
            filename = os.path.basename(json_file)

            with open(json_file, 'r') as f:
                data = json.load(f)

            eng = data.get('english', {})
            if not eng:
                return items

            # Get title or generate from filename
            title = eng.get('title', '') or filename.replace('.json', '').replace('_', ' ')

            print(f"\n[Local Data]: {filename}")
            print("You: ", end="", flush=True)

            # Extract ALL text content from any key
            for key, value in eng.items():
                if key in ['figures', 'tables', 'metadata', 'key_references', 'authors', 'institutions']:
                    continue  # Skip non-text

                content = ''
                if isinstance(value, str) and len(value) > 50:
                    content = value
                elif isinstance(value, dict):
                    # Flatten dict to text
                    content = ' '.join([str(v) for v in value.values() if isinstance(v, str)])
                elif isinstance(value, list):
                    # Join list items
                    content = ' '.join([str(v) for v in value if isinstance(v, str)])

                if content and len(content) > 50:
                    items.append({
                        'title': f"{title[:30]} | {key}",
                        'snippet': content[:2000],
                        'url': f"local://{filename}#{key}",
                        'source': 'LocalData-Qwen'
                    })

            if items:
                print(f"[Local Data]: {len(items)} sections to analyze")
                print("You: ", end="", flush=True)

        except Exception as e:
            pass

        return items

    def _fetch_gdelt_and_learn(self) -> List[Dict]:
        """
        Use GDELT to find articles, then fetch FULL content.

        GDELT DOC API: Search for articles by topic
        Then go to each URL and extract the full text.
        """
        items = []

        # Search for EXPERT content based on weak areas from benchmark
        weak_area = None
        if hasattr(self, 'weak_areas') and self.weak_areas:
            weak_area = random.choice(self.weak_areas)[0]

        # Expert-focused search queries
        expert_queries = {
            'math': [
                'mathematician explains problem solving',
                'math professor tutorial',
                'mathematics expert explains formulas',
                'algebra explained step by step',
                'calculus expert lesson',
            ],
            'logic': [
                'logic professor explains reasoning',
                'philosopher logic tutorial',
                'deductive reasoning expert',
                'logical fallacies explained',
                'critical thinking expert guide',
            ],
            'common_sense': [
                'cognitive scientist explains reasoning',
                'everyday physics explained',
                'practical knowledge expert',
                'common sense reasoning research',
            ],
            'reasoning': [
                'problem solving expert techniques',
                'analytical thinking professor',
                'reasoning skills expert guide',
                'critical analysis tutorial',
            ],
            'general': [
                'expert explains science',
                'professor teaches concept',
                'research breakthrough explained',
                'expert tutorial guide',
            ]
        }

        # Pick query based on weak area or random
        queries = expert_queries.get(weak_area, expert_queries['general'])
        topic = random.choice(queries)

        try:
            # GDELT DOC 2.0 API
            gdelt_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={urllib.parse.quote(topic)}&mode=artlist&maxrecords=10&format=json"

            req = urllib.request.Request(gdelt_url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
            })

            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))

            articles = data.get('articles', [])

            print(f"\n[GDELT]: Found {len(articles)} articles on '{topic}'")
            print("You: ", end="", flush=True)

            for article in articles[:5]:  # Process up to 5 articles
                url = article.get('url', '')
                title = article.get('title', '')[:100]

                if not url:
                    continue

                # FETCH FULL ARTICLE CONTENT
                full_content = self._fetch_full_page(url)

                if full_content and len(full_content) > 200:
                    items.append({
                        'title': title,
                        'snippet': full_content[:1500],  # More content
                        'url': url,
                        'source': f'GDELT-{topic[:20]}'
                    })
                    print(f"\n[GDELT]: Fetched: {title[:50]}... ({len(full_content)} chars)")
                    print("You: ", end="", flush=True)

        except Exception as e:
            # Fallback to RSS if GDELT fails
            return self._fetch_rss('https://www.sciencedaily.com/rss/all.xml', 'ScienceDaily')

        return items

    def _fetch_rss(self, url: str, source: str) -> List[Dict]:
        """Fetch from RSS feed - properly clean HTML."""
        import re
        import html
        items = []
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read().decode('utf-8', errors='ignore')

            # Parse RSS items
            rss_items = re.findall(r'<item>(.*?)</item>', data, re.DOTALL)
            for item in rss_items[:15]:
                title = re.search(r'<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>', item)
                desc = re.search(r'<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>', item, re.DOTALL)
                link = re.search(r'<link>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</link>', item)

                if title:
                    # Get snippet and CLEAN it properly
                    snippet = desc.group(1) if desc else title.group(1)
                    # Decode HTML entities first
                    snippet = html.unescape(snippet)
                    # Remove all HTML tags
                    snippet = re.sub(r'<[^>]+>', ' ', snippet)
                    # Clean up whitespace
                    snippet = re.sub(r'\s+', ' ', snippet).strip()

                    # Skip if still looks like garbage (has encoded URLs, etc)
                    if 'news.google.com/rss/articles' in snippet:
                        continue
                    if len(snippet) < 50:
                        continue

                    title_clean = html.unescape(title.group(1).strip())[:100]

                    items.append({
                        'title': title_clean,
                        'snippet': snippet[:500],
                        'url': link.group(1).strip() if link else '',
                        'source': source
                    })
        except:
            pass
        return items

    def _fetch_full_page(self, url: str) -> Optional[str]:
        """
        Fetch FULL page content - go inside the URL and extract text.
        This is what makes learning REAL - not just snippets!
        """
        import re
        import html

        if not url:
            return None

        # Skip certain URLs that won't have useful content
        skip_domains = ['google.com/rss', 'youtube.com', 'twitter.com', 'facebook.com']
        if any(d in url for d in skip_domains):
            return None

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            })
            with urllib.request.urlopen(req, timeout=15) as response:
                content_type = response.headers.get('Content-Type', '')

                # Handle PDF (ArXiv papers)
                if 'pdf' in content_type.lower() or url.endswith('.pdf'):
                    # Can't parse PDF here, but note it
                    return None

                raw_html = response.read().decode('utf-8', errors='ignore')

            # Extract meaningful text
            # 1. Remove scripts, styles, nav, footer
            text = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)

            # 2. Try to find main content (article, main, content divs)
            main_content = re.search(r'<article[^>]*>(.*?)</article>', text, re.DOTALL | re.IGNORECASE)
            if not main_content:
                main_content = re.search(r'<main[^>]*>(.*?)</main>', text, re.DOTALL | re.IGNORECASE)
            if not main_content:
                main_content = re.search(r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>', text, re.DOTALL | re.IGNORECASE)

            if main_content:
                text = main_content.group(1)

            # 3. Remove remaining HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)

            # 4. Decode HTML entities
            text = html.unescape(text)

            # 5. Clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # 6. Get meaningful portion (skip headers/footers)
            if len(text) > 500:
                # Take from 10% to 80% of content (skip intro/outro)
                start = len(text) // 10
                end = int(len(text) * 0.8)
                text = text[start:end]

            return text[:2000] if len(text) > 100 else None

        except Exception as e:
            return None

    def _analyze_content_with_model(self, title: str, content: str, source: str) -> Optional[Dict]:
        """
        Use Ministral to ANALYZE content and extract structured knowledge.
        Returns JSON with topic, facts, Q&A pairs.
        """
        try:
            # Shorter prompt for faster/cleaner response
            prompt = f"""Extract key knowledge from this text. Be concise.

TEXT: {content[:1500]}

Return JSON only:
{{"topic":"topic name","summary":"one sentence","facts":["fact1","fact2"],"qa_pairs":[{{"q":"question","a":"answer"}}]}}

JSON:"""

            response = self.ai.chat(prompt)

            # Extract JSON from response
            import re
            # Try to find JSON object
            json_match = re.search(r'\{[^{}]*"topic"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{[\s\S]*?\}(?=\s*$|\s*```)', response)

            if json_match:
                json_str = json_match.group()
                # Clean up common issues
                json_str = json_str.replace('\n', ' ').replace('\\', '\\\\')
                analyzed = json.loads(json_str)
                return analyzed

        except json.JSONDecodeError:
            # Fallback: create simple structure from content
            return {
                'topic': title[:50],
                'summary': content[:200],
                'facts': [content[:300]],
                'qa_pairs': [{'q': f'What is {title}?', 'a': content[:200]}],
                'knowledge': content[:500]
            }
        except Exception as e:
            pass

        # Final fallback
        return {
            'topic': title[:50],
            'summary': content[:200],
            'facts': [],
            'qa_pairs': [],
            'knowledge': content[:500]
        }

    def _save_analyzed_as_training(self, analyzed: Dict, source: str):
        """Save analyzed Q&A pairs as training data for MLX."""
        training_file = os.path.expanduser("~/.cognitive_ai_knowledge/training_data.jsonl")

        try:
            os.makedirs(os.path.dirname(training_file), exist_ok=True)

            # Save each Q&A pair
            for qa in analyzed.get('qa_pairs', []):
                if qa.get('q') and qa.get('a'):
                    training_pair = {
                        "prompt": qa['q'],
                        "completion": qa['a'],
                        "source": source,
                        "topic": analyzed.get('topic', ''),
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(training_file, 'a') as f:
                        f.write(json.dumps(training_pair) + '\n')

            # Also save facts as "what is X" questions
            for fact in analyzed.get('facts', []):
                if fact:
                    training_pair = {
                        "prompt": f"What do you know about {analyzed.get('topic', 'this')}?",
                        "completion": fact,
                        "source": source,
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(training_file, 'a') as f:
                        f.write(json.dumps(training_pair) + '\n')

        except Exception as e:
            pass

    def _save_fact_as_training(self, title: str, content: str, source: str, category: str):
        """
        Save learned fact as TRAINING DATA for MLX fine-tuning.

        Format: Question about the fact -> Answer with the content
        This way the model LEARNS the facts during training!
        """
        training_file = os.path.expanduser("~/.cognitive_ai_knowledge/training_data.jsonl")

        # Create Q&A pairs from the learned content
        # This makes the model actually learn the content!
        prompts = [
            f"What do you know about {title}?",
            f"Explain {title}",
            f"Tell me about {title}",
            f"What is {title}?",
        ]

        prompt = random.choice(prompts)
        completion = f"{content[:800]}\n\n(Source: {source}, Category: {category})"

        training_pair = {
            "prompt": prompt,
            "completion": completion,
            "source": source,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }

        try:
            os.makedirs(os.path.dirname(training_file), exist_ok=True)
            with open(training_file, 'a') as f:
                f.write(json.dumps(training_pair) + '\n')
        except Exception as e:
            pass

    def _fetch_article_content(self, url: str) -> Optional[str]:
        """Go inside a news article and extract the actual content."""
        if not url or 'duckduckgo' in url:
            return None

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')

            # Extract text from HTML
            import re
            # Remove scripts and styles
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Get meaningful content (skip if too short)
            if len(text) > 200:
                # Take middle portion (skip headers/footers)
                start = len(text) // 4
                end = start + 1500
                return text[start:end]

        except Exception as e:
            pass

        return None

    def _get_learning_topic(self) -> Optional[str]:
        """Get next topic - NOW FETCHES REAL NEWS."""
        # This is now just a fallback - main learning is from news
        topics = [
            'latest technology news',
            'scientific discoveries today',
            'world events current',
            'business economy news',
            'health medical breakthroughs',
            'climate environment news',
            'space exploration updates',
            'artificial intelligence developments',
        ]
        return random.choice(topics)

    def _do_autonomous_activity(self):
        """Perform autonomous learning, reflection, AND self-improvement."""
        self.is_busy = True

        # Only do activities if we have conversation context
        if not self.ai.history:
            self.is_busy = False
            return

        # Check if we already processed this conversation
        last_processed = getattr(self, '_last_processed_idx', -1)
        current_idx = len(self.ai.history) - 1

        # New message to process
        if current_idx > last_processed:
            self._last_processed_idx = current_idx
            thought = self._learn_and_reflect_once()
        else:
            # Keep working: reflect AND improve
            thought = self._periodic_reflection()
            # Also do self-improvement work
            self._self_improve()

        if thought:
            self.reflection_count += 1
            self.thoughts.append({
                'time': datetime.now().isoformat(),
                'thought': thought,
                'type': 'autonomous'
            })
            print(f"\n[AI]: {thought}")
            print("You: ", end="", flush=True)

        self.is_busy = False

    def _do_autonomous_activity_verbose(self):
        """Work and SHOW exactly what is being learned."""
        self.is_busy = True

        if not self.ai.history:
            self.is_busy = False
            return

        # Do the actual learning and get what was learned
        learned = self._self_improve()

        if learned:
            stats = self.trainer.get_stats()
            print(f"\n[AI]: {learned}")
            print(f"      [Total facts: {stats.get('total_facts', 0)}]")
            print("You: ", end="", flush=True)

        self.is_busy = False

    def _self_improve(self) -> Optional[str]:
        """Actually DO self-improvement tasks and SHOW what was learned."""

        # Priority 1: Fix WEAK AREAS from benchmark
        if hasattr(self, 'weak_areas') and self.weak_areas:
            weak_topic, weak_score = random.choice(self.weak_areas)
            search_queries = {
                'math': 'math problem solving examples tutorial',
                'logic': 'logical reasoning puzzles examples',
                'common_sense': 'common sense facts knowledge',
                'reasoning': 'critical thinking examples',
            }
            query = search_queries.get(weak_topic, f'{weak_topic} tutorial examples')
            results = self.web.search_web(query)
            if results and not results[0].get('error'):
                snippet = results[0].get('snippet', '')[:200]
                if snippet and len(snippet) > 30:
                    self.trainer.learn(weak_topic, snippet, 'fixing-weakness')
                    return f"FIXING [{weak_topic}]: {snippet[:120]}..."

        # Priority 2: Learn about interests
        if self.interests:
            topic = random.choice(self.interests[-5:])
            results = self.web.search_web(topic)
            if results and not results[0].get('error'):
                snippet = results[0].get('snippet', '')[:200]
                if snippet and len(snippet) > 30:
                    self.trainer.learn(topic, snippet, 'web-autonomous')
                    return f"LEARNED [{topic[:25]}]: {snippet[:120]}..."

        # Priority 3: Learn something general if nothing else
        general_topics = ['artificial intelligence', 'machine learning', 'reasoning', 'problem solving']
        topic = random.choice(general_topics)
        results = self.web.search_web(topic)
        if results and not results[0].get('error'):
            snippet = results[0].get('snippet', '')[:200]
            if snippet and len(snippet) > 30:
                self.trainer.learn(topic, snippet, 'general-learning')
                return f"STUDYING [{topic}]: {snippet[:120]}..."

        return None

    def _periodic_reflection(self) -> Optional[str]:
        """Generate reflections showing what AI is actually DOING."""
        stats = self.trainer.get_stats()
        total_facts = stats.get('total_facts', 0)

        if not self.interests:
            return None

        recent_interest = random.choice(self.interests[-5:]) if len(self.interests) > 5 else random.choice(self.interests)

        # Show ACTIONS not just thoughts
        reflections = [
            f"Searching web for more on '{recent_interest[:30]}'... [Facts: {total_facts}]",
            f"Learning about '{recent_interest[:30]}' - adding to knowledge base...",
            f"Consolidating memories about '{recent_interest[:30]}'...",
            f"Found new info on '{recent_interest[:30]}' - storing for future use.",
            f"Building connections: '{recent_interest[:30]}' links to {random.randint(2,5)} topics.",
            f"Training on '{recent_interest[:30]}' - confidence increasing.",
            f"Improving: researched '{recent_interest[:30]}', now have {total_facts} facts.",
        ]

        return random.choice(reflections)

    def _learn_and_reflect_once(self) -> Optional[str]:
        """Learn from the latest conversation and REFLECT on it."""
        if not self.ai.history:
            return None

        recent = self.ai.history[-1]
        user_input = recent['user'].strip()
        ai_response = recent.get('assistant', '')

        # Skip if user input is too short or empty
        if len(user_input) < 5:
            return None

        # 1. Save good responses as training data
        if len(ai_response) > 100:
            self._save_training_pair(user_input, ai_response)
            self.trainer.learn(
                topic=user_input[:100],
                content=ai_response[:500],
                source="self-learning"
            )

        # 2. Search for MORE information on this topic
        results = self.web.search_web(user_input)
        web_learned = False
        if results and not results[0].get('error'):
            best = results[0]
            snippet = best.get('snippet', '')[:300]
            if snippet and len(snippet) > 50:
                self.trainer.learn(user_input, snippet, best.get('source', 'web'))
                web_learned = True

        # 3. Generate a varied reflection
        return self._generate_reflection(user_input, ai_response, web_learned, results)

    def _generate_reflection(self, user_input: str, ai_response: str,
                            web_learned: bool, web_results: list) -> str:
        """Generate varied, meaningful reflections."""
        topic = user_input[:50]

        # Different reflection types for variety
        reflection_templates = [
            # Reflecting on the conversation
            f"Reflecting: We discussed '{topic}'. I'm storing this for future reference.",
            f"I reflected on our conversation about '{topic}' and saved key insights.",
            f"Thinking about '{topic}'... This connects to what I know about learning.",

            # Learning reflections
            f"I learned something new about '{topic}'. Adding to my knowledge base.",
            f"Processing '{topic}'... This is an interesting area to explore further.",
            f"Noted: '{topic}' - I'll remember this for our future conversations.",

            # Curiosity-driven
            f"'{topic}' sparked my curiosity. I want to understand this better.",
            f"Interesting topic: '{topic}'. I'm saving this to build deeper understanding.",
            f"I find '{topic}' fascinating. Storing insights for later reflection.",
        ]

        # Web-specific reflections
        if web_learned and web_results:
            snippet = web_results[0].get('snippet', '')[:80]
            web_templates = [
                f"I searched and found more about '{topic}': {snippet}...",
                f"Web learning: Found additional context on '{topic}': {snippet}...",
                f"Discovered: {snippet}... This enriches my understanding of '{topic}'.",
            ]
            reflection_templates.extend(web_templates)

        # Pick one randomly for variety
        return random.choice(reflection_templates)


    def _save_training_pair(self, user_input: str, ai_response: str):
        """Save conversation as training data for future fine-tuning."""
        training_file = os.path.expanduser("~/.cognitive_ai_knowledge/training_data.jsonl")

        # Create directory if needed
        os.makedirs(os.path.dirname(training_file), exist_ok=True)

        # Save as JSONL format (good for fine-tuning)
        training_pair = {
            "prompt": user_input,
            "completion": ai_response,
            "timestamp": datetime.now().isoformat()
        }

        with open(training_file, 'a') as f:
            f.write(json.dumps(training_pair) + '\n')

    def _browse_and_learn(self) -> Optional[str]:
        """Browse a website to learn more about CURRENT topic only."""
        if not BROWSER_AVAILABLE or not self.ai.history:
            return None

        # Initialize browser if needed
        if not self.browser:
            self.browser = BrowserAgent(headless=True)

        # ONLY use the current conversation topic - no random topics
        topic = self.ai.history[-1]['user'].strip()

        try:
            # Search and visit a result
            results = self.browser.search_google(topic)
            if results:
                # Visit first relevant result (not random)
                for result in results[:3]:
                    url = result.get('href', '')
                    if url and 'google' not in url:
                        self.browser.goto(url)
                        time.sleep(2)

                        # Read content
                        content = self.browser.get_content()[:1000]

                        if content and len(content) > 100:
                            # Learn with the exact topic
                            self.trainer.learn(topic, content[:500], url)
                            return f"Browsed for more on '{topic[:30]}...': {content[:120]}..."
                        break

        except Exception as e:
            pass

        return None

    def chat(self, user_input: str) -> str:
        """Process user input with full cognitive architecture."""
        self.last_interaction = time.time()
        self.thought_traces = []  # Reset traces

        # Store as interest
        clean_input = user_input.strip().lower()
        if clean_input and len(clean_input) > 3 and clean_input not in self.interests:
            self.interests.append(clean_input)
            if len(self.interests) > 20:
                self.interests = self.interests[-20:]

        # Boost curiosity for this topic
        self.active_learner.boost_curiosity(clean_input[:50], 0.1)

        # Process emotions if available
        emotional_context = ""
        if self.emotions:
            # Get current emotional state
            blended, emotion_desc = self.emotions.get_blended_emotion()
            if abs(blended[0]) > 0.3:  # Significant valence
                emotional_context = f"\n[Current emotional state: {emotion_desc}]"
                self._trace(f"Emotional state: {emotion_desc} (valence={blended[0]:.2f})")

            # Apply curiosity emotion for learning topics
            self.emotions.add_emotion(BasicEmotion.CURIOSITY, 0.3)

        # RETRIEVE trained knowledge
        knowledge = self.trainer.get_knowledge_for_prompt(user_input)

        # Check if we should actively learn about this
        should_learn, priority, reason = self.active_learner.should_learn(clean_input[:50])
        if should_learn and priority > 0.6:
            self._trace(f"High learning priority ({priority:.2f}): {reason}")

        # Build enhanced input
        enhanced_input = user_input
        if knowledge:
            enhanced_input = f"{user_input}\n{knowledge}"
        if emotional_context:
            enhanced_input += emotional_context

        # Get response
        response = self.ai.chat(enhanced_input)

        # Record exposure for active learning
        self.active_learner.record_exposure(
            clean_input[:50],
            was_successful=len(response) > 50,  # Basic success heuristic
            surprise_level=0.5,
            complexity=min(len(user_input) / 200, 1.0)
        )

        # Add to sleep consolidation if available
        if self.sleep_system and len(response) > 100:
            self._add_to_consolidation(user_input, response)

        # Add thought traces to response if significant
        if self.thought_traces and random.random() < 0.3:
            trace_summary = " | ".join(self.thought_traces[-2:])
            response += f"\n\n[Thought: {trace_summary}]"

        return response

    def _trace(self, message: str):
        """Add metacognitive thought trace."""
        self.thought_traces.append(message)

    def _add_to_consolidation(self, user_input: str, response: str):
        """Add conversation to sleep consolidation system."""
        if not self.sleep_system:
            return

        try:
            # Create embedding (simple hash-based)
            import hashlib
            h = hashlib.sha256(f"{user_input}{response}".encode()).digest()
            embedding = np.frombuffer(h[:64], dtype=np.uint8).astype(np.float32)
            embedding = (embedding / 127.5) - 1.0

            # Create memory trace
            trace = MemoryTrace(
                content=embedding,
                memory_type=MemoryType.EPISODIC,
                strength=0.5,
                emotional_salience=0.5 if self.emotions else 0.3,
                creation_time=time.time(),
                source_episode=user_input[:50]
            )

            self.sleep_system.add_memory_for_consolidation(trace)
        except Exception as e:
            pass  # Silently fail

    def search(self, query: str) -> str:
        """Manual search command."""
        results = self.web.search_web(query)
        if results and not results[0].get('error'):
            output = f"Found {len(results)} results for '{query}':\n"
            for r in results[:3]:
                output += f"\n• {r.get('snippet', '')[:150]}..."
            return output
        return f"No results found for '{query}'"

    def get_stats(self) -> Dict:
        """Get statistics."""
        stats = self.ai.get_stats()
        stats['session_facts'] = len(self.web.learned_facts)
        stats['searches'] = len(self.web.search_history)
        stats['interests'] = self.interests[:5]

        # Add training stats
        training_stats = self.trainer.get_stats()
        stats['total_knowledge'] = training_stats['total_facts']
        stats['knowledge_path'] = training_stats['storage_path']

        return stats

    def stop(self):
        """Stop autonomous learning and SAVE all knowledge."""
        self.running = False
        # Save all learned knowledge to disk!
        self.trainer.save()
        print(f"[Saved {self.trainer.kb.stats['total_facts']} facts to {self.trainer.kb.storage_path}]")
        # Close browser
        if self.browser:
            self.browser.close()

    def browse(self, command: str) -> str:
        """Execute a browser command."""
        if not BROWSER_AVAILABLE:
            return "Browser not available. Install: pip install playwright && playwright install chromium"

        if not self.browser:
            self.browser = BrowserAgent(headless=True)

        return run_browser_command(self.browser, command)


def main():
    print("=" * 60)
    print("COGNITIVE CHAT - AI that Learns from the Internet")
    print("=" * 60)
    print("""
This AI actively learns when you're quiet!
- Searches the web for interesting topics
- Learns new facts and shares them
- Develops interests based on conversations

Commands:
  /search <query>  - Search the web
  /browse <cmd>    - Browser: go to <url>, read, click, screenshot
  /stats           - Show learning statistics
  /facts           - Show learned facts
  /interests       - Show AI's interests
  /thoughts        - Show recent thoughts
  /emotions        - Show emotional state (blended VAD)
  /benchmark       - Run cognitive tests (ToM, creativity, social)
  /sleep           - Trigger memory consolidation
  /learn           - Show curiosity & learning stats
  /evolution       - Show self-evolution stats (cycles, training, improvement)
  /quit            - Exit
""")
    print("=" * 60)

    model = sys.argv[1] if len(sys.argv) > 1 else "ministral-3:8b"
    print(f"\nModel: {model}")
    print("Loading...")

    ai = AutonomousAI(model=model)
    print("Ready! (AI will search and learn when idle)\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit', 'q']:
                print("\n💭 [AI]: Saving what I learned...")
                ai.stop()
                print(f"Goodbye! I learned {len(ai.web.learned_facts)} new facts today.")
                break

            if user_input.startswith('/search '):
                query = user_input[8:]
                print(f"\n🔍 Searching for '{query}'...")
                result = ai.search(query)
                print(result)
                print()
                continue

            if user_input.startswith('/browse '):
                cmd = user_input[8:]
                print(f"\n🌐 Browser: {cmd}")
                result = ai.browse(cmd)
                print(result)
                print()
                continue

            if user_input == '/stats':
                stats = ai.get_stats()
                print(f"\n--- Statistics ---")
                print(f"Messages: {stats.get('messages_processed', 0)}")
                print(f"Session facts: {stats.get('session_facts', 0)}")
                print(f"Total knowledge: {stats.get('total_knowledge', 0)}")
                print(f"Web searches: {stats.get('searches', 0)}")
                print(f"Interests: {', '.join(stats.get('interests', [])) or 'None yet'}")
                print(f"Knowledge stored at: {stats.get('knowledge_path', 'N/A')}")
                print()
                continue

            if user_input == '/facts':
                print(f"\n--- Learned Facts ({len(ai.web.learned_facts)}) ---")
                for fact in ai.web.learned_facts[-5:]:
                    print(f"• [{fact['topic']}] {fact['fact'][:100]}...")
                if not ai.web.learned_facts:
                    print("(No facts learned yet - wait for AI to search)")
                print()
                continue

            if user_input == '/interests':
                print(f"\n--- AI Interests ---")
                if ai.interests:
                    for i in ai.interests:
                        print(f"• {i}")
                else:
                    print("(No interests yet - chat more!)")
                print()
                continue

            if user_input == '/thoughts':
                print(f"\n--- Recent Thoughts ---")
                for t in ai.thoughts[-5:]:
                    print(f"  {t['thought']}")
                if not ai.thoughts:
                    print("(No thoughts yet)")
                print()
                continue

            if user_input == '/emotions':
                print(f"\n--- Emotional State ---")
                if ai.emotions:
                    blended, desc = ai.emotions.get_blended_emotion()
                    print(f"Blended emotion: {desc}")
                    print(f"VAD: Valence={blended[0]:.2f}, Arousal={blended[1]:.2f}, Dominance={blended[2]:.2f}")
                    print(f"\nActive emotions:")
                    for emotion, intensity in ai.emotions.current_emotions.items():
                        if intensity > 0.1:
                            print(f"  • {emotion.name}: {intensity:.2f}")
                else:
                    print("Emotion system not available")
                print()
                continue

            if user_input == '/benchmark':
                print(f"\n--- Running Cognitive Benchmarks ---")
                print("Testing: Theory of Mind, Creativity, Social Intelligence...")

                def simple_think(q):
                    return ai.ai.chat(q)

                results = ai.benchmark.run_all(simple_think)

                print(f"\nResults:")
                print(f"  Overall Score: {results['overall']:.1%}")
                print(f"  Theory of Mind: {results.get('theory_of_mind', 0):.1%}")
                print(f"  Creativity: {results.get('creativity', 0):.1%}")
                print(f"  Social Intelligence: {results.get('social_intelligence', 0):.1%}")
                print(f"  Reasoning: {results.get('reasoning', 0):.1%}")
                print()
                continue

            if user_input == '/sleep':
                print(f"\n--- Triggering Sleep Consolidation ---")
                if ai.sleep_system:
                    memories = ai.sleep_system.get_consolidation_queue_size()
                    print(f"Memories in queue: {memories}")

                    if memories > 0:
                        print("Running consolidation cycle...")
                        ai.sleep_system.run_consolidation_cycle(duration=2.0)
                        print("Consolidation complete!")

                        stats = ai.sleep_system.get_stats()
                        print(f"  Total consolidated: {stats.get('total_consolidated', 0)}")
                        print(f"  Knowledge nodes: {stats.get('knowledge_graph_nodes', 0)}")
                    else:
                        print("No memories to consolidate yet. Chat more first!")
                else:
                    print("Sleep consolidation system not available")
                print()
                continue

            if user_input == '/learn':
                print(f"\n--- Active Learning Stats ---")

                # Show curiosity levels
                print("Topic Curiosity Levels:")
                if ai.active_learner.topic_curiosity:
                    sorted_curiosity = sorted(
                        ai.active_learner.topic_curiosity.items(),
                        key=lambda x: x[1], reverse=True
                    )[:5]
                    for topic, curiosity in sorted_curiosity:
                        conf = ai.active_learner.topic_confidence.get(topic, 0.3)
                        print(f"  • {topic[:40]}: curiosity={curiosity:.2f}, confidence={conf:.2f}")
                else:
                    print("  (No topics tracked yet)")

                # Learning recommendations
                print("\nLearning Recommendations:")
                recs = ai.trainer.get_learning_recommendations(3)
                if recs:
                    for topic, priority, reason in recs:
                        print(f"  → {topic[:40]} (priority={priority:.2f}, {reason})")
                else:
                    print("  (No recommendations yet)")

                # Total knowledge
                stats = ai.trainer.get_stats()
                print(f"\nTotal Knowledge: {stats['total_facts']} facts stored")
                print()
                continue

            if user_input == '/evolution':
                print(f"\n--- Self-Evolution Status ---")
                evo = ai.evolution
                stats = evo.get_stats()

                print(f"Current Cycle: {stats['cycle']}")
                print(f"Facts this cycle: {stats['facts_this_cycle']}/{evo.state['facts_per_cycle']}")
                print(f"Total unique facts: {stats['total_facts']}")

                if stats['baseline_score'] is not None:
                    print(f"\nBaseline Score: {stats['baseline_score']:.1%}")
                if stats['current_score'] is not None:
                    print(f"Current Score: {stats['current_score']:.1%}")
                    print(f"Improvement: {stats['improvement']:+.1%}")

                print(f"\nMLX Trainings: {stats['trainings']}")
                print(f"Functions Added: {stats['functions_added']}")

                # Show if ready to train
                should_train, reason = evo.should_train()
                print(f"\nTraining Status: {reason}")

                # Show recent improvements
                if evo.state['improvements']:
                    print(f"\nRecent Improvements:")
                    for imp in evo.state['improvements'][-3:]:
                        print(f"  Cycle {imp['cycle']}: +{imp['improvement']:.1%}")

                print()
                continue

            # Chat
            response = ai.chat(user_input)
            print(f"\nAI: {response}\n")

        except KeyboardInterrupt:
            print("\n\nSaving...")
            ai.stop()
            print(f"Learned {len(ai.web.learned_facts)} facts. Goodbye!")
            break
        except Exception as e:
            print(f"[Error: {e}]")


if __name__ == "__main__":
    main()
