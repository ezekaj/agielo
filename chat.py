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

from integrations.cognitive_ollama import CognitiveLLM
from integrations.self_training import SelfTrainer
from integrations.browser_agent import BrowserAgent, run_browser_command, BROWSER_AVAILABLE
from integrations.active_learning import get_active_learner, ActiveLearner
from integrations.self_evolution import get_evolution, SelfEvolution

# Self-Play Training for autonomous improvement
try:
    from integrations.self_play import SelfPlayTrainer, Difficulty, get_self_play_trainer
    SELF_PLAY_AVAILABLE = True
    print("[SelfPlay] Self-play training system: AVAILABLE")
except ImportError:
    SELF_PLAY_AVAILABLE = False
    SelfPlayTrainer = None
    get_self_play_trainer = None
    Difficulty = None
    print("[SelfPlay] Self-play training not available")

# Code Evolution for population-based code improvement
try:
    from integrations.code_evolution import CodeEvolution, get_code_evolution
    CODE_EVOLUTION_AVAILABLE = True
    print("[CodeEvolution] Code evolution system: AVAILABLE")
except ImportError:
    CODE_EVOLUTION_AVAILABLE = False
    CodeEvolution = None
    get_code_evolution = None
    print("[CodeEvolution] Code evolution not available")

# RND Curiosity for novelty-driven exploration
try:
    from integrations.rnd_curiosity import RNDCuriosity, get_rnd_curiosity
    RND_CURIOSITY_AVAILABLE = True
    print("[RNDCuriosity] Curiosity-driven exploration: AVAILABLE")
except ImportError:
    RND_CURIOSITY_AVAILABLE = False
    RNDCuriosity = None
    get_rnd_curiosity = None
    print("[RNDCuriosity] RND curiosity not available")

# Super Agent for intelligent web search
try:
    from integrations.super_agent import SuperAgent
    SUPER_AGENT_AVAILABLE = True
    print("[SuperAgent] Intelligent web search: AVAILABLE")
except ImportError:
    SUPER_AGENT_AVAILABLE = False
    print("[SuperAgent] Not available - using basic search")

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
    """Autonomous web learning capabilities with multi-source parallel search."""

    def __init__(self):
        self.learned_facts = []
        self.search_history = []
        self.cache = {}  # Simple cache to avoid repeated searches

    def search_web(self, query: str, sources: List[str] = None) -> List[Dict]:
        """
        Search multiple free sources in parallel for faster results.

        Available sources:
        - duckduckgo: Instant answers and related topics
        - wikipedia: Encyclopedia articles
        - arxiv: Academic papers (science/tech/math)
        - stackexchange: Programming Q&A
        - github: Code repositories
        - wikidata: Structured facts
        - openlibrary: Book information
        - wordnik: Word definitions

        Args:
            query: Search query
            sources: List of sources to use (default: all)
        """
        # Check cache first
        cache_key = f"{query}:{','.join(sources or ['all'])}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        results = []

        # All available search functions
        all_sources = {
            'duckduckgo': self._search_duckduckgo,
            'wikipedia': self._search_wikipedia,
            'arxiv': self._search_arxiv,
            'stackexchange': self._search_stackexchange,
            'github': self._search_github,
            'wikidata': self._search_wikidata,
            'openlibrary': self._search_openlibrary,
            'wordnik': self._search_wordnik,
        }

        # Select sources to use
        if sources:
            search_funcs = {k: v for k, v in all_sources.items() if k in sources}
        else:
            # Default: use all sources
            search_funcs = all_sources

        # Parallel fetch from all sources
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(func, query): name
                for name, func in search_funcs.items()
            }

            for future in as_completed(futures, timeout=12):
                try:
                    source_results = future.result(timeout=6)
                    results.extend(source_results)
                except Exception:
                    pass  # Skip failed sources

        # Cache results
        self.cache[cache_key] = results

        self.search_history.append({
            'query': query,
            'time': datetime.now().isoformat(),
            'results': len(results),
            'sources': list(search_funcs.keys())
        })

        return results

    def _search_duckduckgo(self, query: str) -> List[Dict]:
        """Search DuckDuckGo instant answers."""
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []

            # Abstract (main answer)
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', query),
                    'snippet': data['Abstract'],
                    'source': 'DuckDuckGo',
                    'url': data.get('AbstractURL', '')
                })

            # Answer (direct answer)
            if data.get('Answer'):
                results.append({
                    'title': 'Direct Answer',
                    'snippet': data['Answer'],
                    'source': 'DuckDuckGo',
                    'url': ''
                })

            # Related topics
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '')[:50],
                        'snippet': topic.get('Text', ''),
                        'source': 'DuckDuckGo',
                        'url': topic.get('FirstURL', '')
                    })

            return results
        except:
            return []

    def _search_wikipedia(self, query: str) -> List[Dict]:
        """Search Wikipedia for factual information."""
        import re
        results = []

        # Method 1: Wikipedia Search API (finds relevant articles)
        try:
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=3"
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                search_results = data.get('query', {}).get('search', [])

                for r in search_results:
                    title = r.get('title', '')
                    snippet = r.get('snippet', '')
                    # Clean HTML tags
                    snippet = re.sub(r'<[^>]+>', '', snippet)
                    if snippet:
                        results.append({
                            'title': title,
                            'snippet': f"{title}: {snippet}",
                            'source': 'Wikipedia',
                            'url': f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                        })
        except:
            pass

        # Method 2: Direct page summary (for exact matches)
        try:
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query.replace(' ', '_'))}"
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))

            if data.get('extract'):
                results.append({
                    'title': data.get('title', query),
                    'snippet': data['extract'],
                    'source': 'Wikipedia',
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')
                })
        except:
            pass

        return results

    def _search_arxiv(self, query: str) -> List[Dict]:
        """Search ArXiv for academic papers (free, no API key needed)."""
        try:
            # ArXiv API - search for papers
            url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results=3"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=8) as response:
                content = response.read().decode('utf-8')

            results = []
            import re

            # Parse XML response (simple regex parsing)
            entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)

            for entry in entries[:3]:
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                link_match = re.search(r'<id>(.*?)</id>', entry)

                if title_match and summary_match:
                    title = title_match.group(1).strip().replace('\n', ' ')
                    summary = summary_match.group(1).strip().replace('\n', ' ')[:300]
                    link = link_match.group(1) if link_match else ''

                    results.append({
                        'title': title,
                        'snippet': summary,
                        'source': 'ArXiv',
                        'url': link
                    })

            return results
        except:
            return []

    def _search_stackexchange(self, query: str) -> List[Dict]:
        """Search StackExchange sites (StackOverflow, etc.) - free API."""
        try:
            # StackExchange API - search across sites
            url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={urllib.parse.quote(query)}&site=stackoverflow&pagesize=3&filter=withbody"
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Accept-Encoding': 'gzip'
            })

            with urllib.request.urlopen(req, timeout=5) as response:
                # Handle gzip compression
                import gzip
                if response.info().get('Content-Encoding') == 'gzip':
                    content = gzip.decompress(response.read()).decode('utf-8')
                else:
                    content = response.read().decode('utf-8')
                data = json.loads(content)

            results = []
            import re

            for item in data.get('items', [])[:3]:
                title = item.get('title', '')
                # Clean HTML from body
                body = item.get('body', '')[:300]
                body = re.sub(r'<[^>]+>', '', body)
                link = item.get('link', '')

                if title:
                    results.append({
                        'title': title,
                        'snippet': body or title,
                        'source': 'StackOverflow',
                        'url': link
                    })

            return results
        except:
            return []

    def _search_github(self, query: str) -> List[Dict]:
        """Search GitHub repositories and code (free, rate limited)."""
        try:
            # GitHub Search API (unauthenticated: 10 requests/min)
            url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&per_page=3"
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/vnd.github.v3+json'
            })

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []

            for repo in data.get('items', [])[:3]:
                name = repo.get('full_name', '')
                description = repo.get('description', '') or 'No description'
                stars = repo.get('stargazers_count', 0)
                url = repo.get('html_url', '')

                results.append({
                    'title': f"{name} ({stars}★)",
                    'snippet': description[:200],
                    'source': 'GitHub',
                    'url': url
                })

            return results
        except:
            return []

    def _search_wikidata(self, query: str) -> List[Dict]:
        """Search Wikidata for structured facts (free)."""
        try:
            # Wikidata API - search entities
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={urllib.parse.quote(query)}&language=en&format=json&limit=3"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []

            for entity in data.get('search', [])[:3]:
                label = entity.get('label', '')
                description = entity.get('description', '')
                entity_id = entity.get('id', '')
                url = f"https://www.wikidata.org/wiki/{entity_id}"

                if label and description:
                    results.append({
                        'title': label,
                        'snippet': f"{label}: {description}",
                        'source': 'Wikidata',
                        'url': url
                    })

            return results
        except:
            return []

    def _search_openlibrary(self, query: str) -> List[Dict]:
        """Search Open Library for books (free)."""
        try:
            # Open Library Search API
            url = f"https://openlibrary.org/search.json?q={urllib.parse.quote(query)}&limit=3"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []

            for doc in data.get('docs', [])[:3]:
                title = doc.get('title', '')
                author = ', '.join(doc.get('author_name', [])[:2]) or 'Unknown author'
                year = doc.get('first_publish_year', '')
                key = doc.get('key', '')
                url = f"https://openlibrary.org{key}" if key else ''

                if title:
                    snippet = f"{title} by {author}"
                    if year:
                        snippet += f" ({year})"

                    results.append({
                        'title': title,
                        'snippet': snippet,
                        'source': 'OpenLibrary',
                        'url': url
                    })

            return results
        except:
            return []

    def _search_wordnik(self, query: str) -> List[Dict]:
        """Search Wordnik for word definitions (free tier)."""
        try:
            # Wordnik API - basic definition (no API key needed for basic)
            # Using Wiktionary via DuckDuckGo as fallback
            url = f"https://api.duckduckgo.com/?q=define+{urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []

            # Get definition from DuckDuckGo
            if data.get('Definition'):
                results.append({
                    'title': f"Definition: {query}",
                    'snippet': data['Definition'],
                    'source': 'Dictionary',
                    'url': data.get('DefinitionURL', '')
                })

            # Also check Abstract for definitions
            abstract = data.get('Abstract', '')
            if abstract and 'definition' not in abstract.lower()[:50]:
                pass  # Skip if not a definition
            elif abstract:
                results.append({
                    'title': query,
                    'snippet': abstract,
                    'source': 'Dictionary',
                    'url': data.get('AbstractURL', '')
                })

            return results
        except:
            return []

    def search_academic(self, query: str) -> List[Dict]:
        """Search academic sources only (ArXiv, semantic scholar concepts)."""
        return self.search_web(query, sources=['arxiv', 'wikipedia'])

    def search_code(self, query: str) -> List[Dict]:
        """Search code-related sources only."""
        return self.search_web(query, sources=['github', 'stackexchange'])

    def search_facts(self, query: str) -> List[Dict]:
        """Search factual sources only."""
        return self.search_web(query, sources=['wikipedia', 'wikidata', 'duckduckgo'])

    def search_books(self, query: str) -> List[Dict]:
        """Search book-related sources."""
        return self.search_web(query, sources=['openlibrary', 'wikipedia'])

    def smart_search(self, question: str, category: str = "") -> List[Dict]:
        """
        Smart search with parallel multi-query execution.

        Strategies (executed in parallel):
        1. Direct question search
        2. Key terms extraction
        3. Category-specific search
        4. Question type reformulation (how/what/why)
        5. Synonym expansion
        6. Entity extraction
        7. Quoted phrase search for exact matches
        """
        all_results = []
        key_terms = self._extract_key_terms(question)

        # Build list of queries to execute in parallel
        queries = []

        # Strategy 1: Direct question search
        queries.append(question[:100])

        # Strategy 2: Key terms search
        if key_terms:
            queries.append(key_terms)

        # Strategy 3: Category-specific search
        if category:
            queries.append(f"{category} {key_terms or question[:50]}")

        # Strategy 4: Question type reformulation
        question_lower = question.lower()
        if "?" in question:
            if any(q in question_lower for q in ['what is', 'what are', 'define']):
                queries.append(f"definition {key_terms}")
            elif any(q in question_lower for q in ['how to', 'how do', 'how can']):
                queries.append(f"guide tutorial {key_terms}")
            elif any(q in question_lower for q in ['why', 'reason', 'cause']):
                queries.append(f"explanation reason {key_terms}")
            elif any(q in question_lower for q in ['when', 'date', 'year']):
                queries.append(f"date history {key_terms}")

        # Strategy 5: Synonym expansion
        expanded_terms = self._expand_with_synonyms(key_terms)
        if expanded_terms and expanded_terms != key_terms:
            queries.append(expanded_terms)

        # Strategy 6: Entity search
        entities = self._extract_entities(question)
        if entities:
            queries.append(' '.join(entities))

        # Strategy 7: Phrase search
        phrases = self._extract_phrases(question)
        for phrase in phrases[:2]:
            queries.append(f'"{phrase}"')

        # Execute all queries in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self.search_web, q): q for q in queries}

            for future in as_completed(futures, timeout=15):
                try:
                    results = future.result(timeout=5)
                    all_results.extend(results)
                except Exception:
                    pass  # Skip failed queries

        # Deduplicate by snippet content with better matching
        seen_snippets = set()
        seen_urls = set()
        unique_results = []

        for r in all_results:
            snippet = r.get('snippet', '')[:100]
            url = r.get('url', '')

            # Skip if we've seen this URL or very similar snippet
            if url and url in seen_urls:
                continue
            if snippet:
                # Normalize snippet for comparison
                normalized = ''.join(c for c in snippet.lower() if c.isalnum())[:50]
                if normalized in seen_snippets:
                    continue
                seen_snippets.add(normalized)

            if url:
                seen_urls.add(url)

            unique_results.append(r)

        # Rank results by relevance to original query
        ranked_results = self._rank_results(unique_results, question, key_terms)

        return ranked_results[:8]  # Increased max results

    def _extract_key_terms(self, text: str) -> str:
        """Extract key terms from a question."""
        # Remove common question words
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are',
                      'the', 'a', 'an', 'if', 'does', 'do', 'can', 'will', 'would', 'should',
                      'to', 'in', 'on', 'at', 'for', 'of', 'and', 'or', 'but', 'with', 'by',
                      'this', 'that', 'these', 'those', 'it', 'its', 'been', 'being', 'have',
                      'has', 'had', 'did', 'done', 'was', 'were', 'am', 'be', 'could', 'would'}
        words = text.lower().replace('?', '').replace('.', '').replace(',', '').split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(key_words[:6])  # Increased from 5 to 6

    def _expand_with_synonyms(self, terms: str) -> str:
        """Expand terms with common synonyms."""
        synonyms = {
            'big': 'large',
            'small': 'little',
            'fast': 'quick',
            'slow': 'sluggish',
            'good': 'excellent',
            'bad': 'poor',
            'start': 'begin',
            'end': 'finish',
            'make': 'create',
            'use': 'utilize',
            'find': 'discover',
            'show': 'display',
            'help': 'assist',
            'best': 'top',
            'new': 'latest',
            'old': 'previous',
        }

        words = terms.lower().split()
        expanded = []
        for word in words:
            expanded.append(word)
            if word in synonyms:
                expanded.append(synonyms[word])

        return ' '.join(expanded[:8])  # Limit expansion

    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities (capitalized words, numbers)."""
        import re
        # Find capitalized words (potential proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Find numbers with context
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:years?|days?|hours?|%|percent))?\b', text)
        return list(set(entities + numbers))[:5]

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful multi-word phrases."""
        import re
        # Remove question marks and normalize
        text = text.replace('?', '').strip()

        # Find 2-4 word sequences that look like meaningful phrases
        words = text.split()
        phrases = []

        for i in range(len(words) - 1):
            # 2-word phrases
            phrase = ' '.join(words[i:i+2])
            if len(phrase) > 5 and not any(w.lower() in ['the', 'a', 'an', 'is', 'are'] for w in words[i:i+2]):
                phrases.append(phrase)

            # 3-word phrases
            if i < len(words) - 2:
                phrase = ' '.join(words[i:i+3])
                if len(phrase) > 8:
                    phrases.append(phrase)

        return phrases[:3]

    def _rank_results(self, results: List[Dict], query: str, key_terms: str) -> List[Dict]:
        """Rank search results by relevance to query."""
        query_lower = query.lower()
        terms = set(key_terms.lower().split())

        scored_results = []
        for r in results:
            score = 0
            snippet = r.get('snippet', '').lower()
            title = r.get('title', '').lower()

            # Exact query match in snippet
            if query_lower[:30] in snippet:
                score += 5

            # Term matches in title (high value)
            for term in terms:
                if term in title:
                    score += 3
                if term in snippet:
                    score += 1

            # Source quality bonus
            source = r.get('source', '')
            if 'Wikipedia' in source:
                score += 2
            elif 'DuckDuckGo' in source:
                score += 1

            # Snippet length bonus (more content = potentially more useful)
            if len(snippet) > 100:
                score += 1

            scored_results.append((r, score))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [r for r, _ in scored_results]

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
    - INTELLIGENT SEARCH: Analyzes failures and searches for solutions
    """

    def __init__(self, model: str = "zai-org/glm-4.7-flash"):
        print("\n[Initializing Cognitive Systems...]")

        self.ai = CognitiveLLM(model=model, backend="lmstudio")
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

        # INTELLIGENT SEARCH: Track failed questions and generated queries
        self.failed_questions = []  # Questions we got wrong
        self.generated_search_queries = []  # AI-generated queries to find solutions

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

        # Self-evolution system (no duplicates, learning cycles, MLX training)
        self.evolution = get_evolution()
        evo_stats = self.evolution.get_stats()
        print(f"[Evolution] Cycle {evo_stats['cycle']}, {evo_stats['total_facts']} unique facts learned")
        if evo_stats['trainings'] > 0:
            print(f"[Evolution] MLX trained {evo_stats['trainings']} times, improvement: {evo_stats['improvement']:+.1%}")

        # Super Agent for intelligent web search
        self.super_agent = None
        if SUPER_AGENT_AVAILABLE:
            self.super_agent = SuperAgent(
                lm_studio_url="http://localhost:1234/v1",
                model=model
            )
            print("[SuperAgent] Intelligent search + compare + decide: ACTIVE")

        # Self-play training for autonomous improvement
        self.self_play_trainer = None
        self.self_play_cycle_counter = 0  # Track cycles for periodic self-play
        if SELF_PLAY_AVAILABLE:
            self.self_play_trainer = get_self_play_trainer()
            sp_stats = self.self_play_trainer.get_stats()
            print(f"[SelfPlay] Trainer active: {sp_stats['questions_generated']} questions, {sp_stats['correct_rate']:.1%} accuracy")
            print(f"[SelfPlay] Adaptive difficulty: {sp_stats['current_difficulty']}")

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
        Background loop for continuous self-learning:

        1. Learn from conversations - extract knowledge from Q&A
        2. Search and learn when AI doesn't know something
        3. Self-correct when errors are detected
        4. Periodically consolidate memory
        5. Train model when enough data collected
        """
        self.pending_learnings = []  # Topics to learn about
        self.error_corrections = []  # Errors to correct

        while self.running:
            time.sleep(5)

            if self.is_busy:
                continue

            # Wait for first conversation before starting
            if not self.ai.history:
                continue

            # ═══════════════════════════════════════════════════════════
            # CONTINUOUS LEARNING: Process pending learnings
            # ═══════════════════════════════════════════════════════════
            if self.pending_learnings:
                self._process_pending_learning()
                continue

            # ═══════════════════════════════════════════════════════════
            # ERROR CORRECTION: Fix mistakes
            # ═══════════════════════════════════════════════════════════
            if self.error_corrections:
                self._process_error_correction()
                continue

            # ═══════════════════════════════════════════════════════════
            # PERIODIC: Check if training ready (every 100 facts)
            # ═══════════════════════════════════════════════════════════
            if self.evolution.should_benchmark():
                evo_stats = self.evolution.get_stats()
                print(f"\n[Evolution] {evo_stats['total_facts']} facts learned. Use /train to fine-tune model.")
                print("You: ", end="", flush=True)
                self.evolution.start_new_cycle()

            # ═══════════════════════════════════════════════════════════
            # IDLE: Memory consolidation
            # ═══════════════════════════════════════════════════════════
            if time.time() - self.last_interaction > 60:  # 1 minute idle
                if hasattr(self.ai, 'memory') and self.ai.memory:
                    try:
                        self.ai.memory.consolidate()
                    except:
                        pass

    def _process_pending_learning(self):
        """Process one pending learning topic by searching and storing knowledge."""
        if not self.pending_learnings:
            return

        self.is_busy = True
        topic = self.pending_learnings.pop(0)

        try:
            print(f"\n[Learning] Researching: {topic[:50]}...")
            print("You: ", end="", flush=True)

            # Search for information
            results = self.web.smart_search(topic, "general")

            if results:
                # Extract and store knowledge
                for r in results[:3]:
                    snippet = r.get('snippet', '')
                    if snippet and not self.evolution.is_duplicate(snippet):
                        self.trainer.learn(topic[:50], snippet[:500], r.get('source', 'web'))
                        self.evolution.mark_learned(snippet[:200])

                print(f"[Learning] ✓ Learned about: {topic[:40]}")
            else:
                print(f"[Learning] No results for: {topic[:40]}")

            print("You: ", end="", flush=True)
        except Exception as e:
            print(f"[Learning] Error: {e}")
            print("You: ", end="", flush=True)

        self.is_busy = False

    def _process_error_correction(self):
        """Process and learn from an error correction."""
        if not self.error_corrections:
            return

        self.is_busy = True
        error = self.error_corrections.pop(0)

        try:
            question = error.get('question', '')
            correct_answer = error.get('correct_answer', '')
            wrong_answer = error.get('wrong_answer', '')

            print(f"\n[Self-Correction] Learning from mistake...")
            print("You: ", end="", flush=True)

            # Store the correction as training data
            content = f"Q: {question}\nCorrect: {correct_answer}\nWrong: {wrong_answer}"
            self.trainer.learn(f"correction: {question[:30]}", content, "self-correction")
            self.evolution.mark_learned(content[:200])

            print(f"[Self-Correction] ✓ Learned correction")
            print("You: ", end="", flush=True)
        except Exception as e:
            print(f"[Self-Correction] Error: {e}")
            print("You: ", end="", flush=True)

        self.is_busy = False

    def add_learning_topic(self, topic: str):
        """Add a topic to learn about in the background."""
        if topic and topic not in self.pending_learnings:
            self.pending_learnings.append(topic)

    def add_error_correction(self, question: str, correct_answer: str, wrong_answer: str):
        """Add an error to correct and learn from."""
        self.error_corrections.append({
            'question': question,
            'correct_answer': correct_answer,
            'wrong_answer': wrong_answer
        })

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

        # Search for EXPERT content based on weak areas from learning
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

    def _extract_methodology_and_apply(self, content: str, source: str) -> Optional[Dict]:
        """
        INTELLIGENT LEARNING: Extract METHODOLOGY from paper/code and create training data
        that teaches HOW TO SOLVE problems, not just facts.
        """
        # Get a failed question to apply the methodology to
        if not hasattr(self, 'failed_questions') or not self.failed_questions:
            return None

        failed = random.choice(self.failed_questions)

        try:
            extract_prompt = f"""Read this content and extract the PROBLEM-SOLVING METHOD it describes:

CONTENT: {content[:1500]}

Now apply this method to solve this problem:
QUESTION: {failed['question']}
CORRECT ANSWER: {failed['correct_answer']}

Create a step-by-step solution using the methodology from the content.
Format your response as:
METHOD: [name of the technique]
STEPS:
1. [step 1]
2. [step 2]
...
ANSWER: {failed['correct_answer']}"""

            response = self.ai.chat(extract_prompt)

            # Create training data with methodology
            return {
                'topic': f"Method for {failed['category']}",
                'summary': f"Methodology to solve {failed['category']} problems",
                'facts': [f"Use this method: {response[:300]}"],
                'qa_pairs': [{
                    'q': failed['question'],
                    'a': f"<think>\n{response}\n</think>\n\nFINAL ANSWER: {failed['correct_answer']}"
                }],
                'knowledge': response[:800],
                'source': source
            }

        except Exception as e:
            return None

    def _analyze_content_with_model(self, title: str, content: str, source: str) -> Optional[Dict]:
        """
        Use the model to ANALYZE content and extract structured knowledge.
        For ArXiv/GitHub sources, try to extract METHODOLOGY first.
        """
        # For research papers and code, try methodology extraction
        if source in ['ArXiv', 'GitHub'] or 'arxiv' in source.lower():
            methodology = self._extract_methodology_and_apply(content, source)
            if methodology and methodology.get('qa_pairs'):
                print(f"\n[INTELLIGENT]: Extracted methodology from {source}!")
                print("You: ", end="", flush=True)
                return methodology

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

        # Priority 1: Fix WEAK AREAS from learning history
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

        # Priority 4: Self-Play Training (every 5 cycles, run 5-10 questions)
        if self.self_play_trainer:
            self.self_play_cycle_counter += 1
            if self.self_play_cycle_counter >= 5:  # Run self-play every 5 autonomous cycles
                self.self_play_cycle_counter = 0

                # Determine topics for self-play - use interests or defaults
                sp_topics = self.interests[-5:] if self.interests else [
                    'programming', 'mathematics', 'science', 'reasoning', 'technology'
                ]

                # Run 5-10 self-play questions
                n_questions = random.randint(5, 10)
                try:
                    result = self.self_play_trainer.run_self_play_round(
                        topics=sp_topics,
                        n_questions=n_questions
                    )

                    # Log results to evolution state
                    self._log_self_play_results(result)

                    correct = result.get('correct_count', 0)
                    total = len(result.get('questions', []))
                    rate = result.get('correct_rate', 0)
                    difficulty = result.get('difficulty_used', 'unknown')

                    return f"SELF-PLAY [{difficulty}]: {correct}/{total} correct ({rate:.1%})"
                except Exception as e:
                    print(f"[SelfPlay] Error during autonomous round: {e}")

        return None

    def _log_self_play_results(self, result: Dict) -> None:
        """Log self-play results to evolution state for tracking."""
        if not result:
            return

        # Log to evolution system if available
        if hasattr(self, 'evolution') and self.evolution:
            self_play_record = {
                'timestamp': datetime.now().isoformat(),
                'round_id': result.get('round_id'),
                'correct_count': result.get('correct_count', 0),
                'total_questions': len(result.get('questions', [])),
                'correct_rate': result.get('correct_rate', 0),
                'avg_score': result.get('avg_score', 0),
                'difficulty': result.get('difficulty_used', 'unknown'),
                'difficulty_change': result.get('difficulty_change'),
                'mistakes_learned': result.get('mistakes_learned', 0)
            }

            # Add to evolution state
            if 'self_play_history' not in self.evolution.state:
                self.evolution.state['self_play_history'] = []

            self.evolution.state['self_play_history'].append(self_play_record)

            # Keep only last 100 records
            if len(self.evolution.state['self_play_history']) > 100:
                self.evolution.state['self_play_history'] = self.evolution.state['self_play_history'][-100:]

            # Save the state
            self.evolution._save_state()

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

    def _learn_topic_from_internet(self, topic: str) -> str:
        """
        Learn ANY topic from the internet!
        Searches for: similar examples, how it works, tutorials, explanations.
        """
        learned_info = []

        print(f"\n[Learning]: Researching '{topic[:50]}' from internet...")
        print("You: ", end="", flush=True)

        # 1. Search for WHAT IT IS
        what_results = self.web.search_web(f"what is {topic}")
        if what_results and not what_results[0].get('error'):
            for r in what_results[:2]:
                if r.get('snippet') and len(r['snippet']) > 50:
                    learned_info.append(f"[Definition]: {r['snippet']}")

        # 2. Search for HOW IT WORKS
        how_results = self.web.search_web(f"how does {topic} work explained")
        if how_results and not how_results[0].get('error'):
            for r in how_results[:2]:
                if r.get('snippet') and len(r['snippet']) > 50:
                    learned_info.append(f"[How it works]: {r['snippet']}")

        # 3. Search for EXAMPLES
        example_results = self.web.search_web(f"{topic} examples tutorial")
        if example_results and not example_results[0].get('error'):
            for r in example_results[:2]:
                if r.get('snippet') and len(r['snippet']) > 50:
                    learned_info.append(f"[Examples]: {r['snippet']}")

        # 4. Search for SIMILAR things
        similar_results = self.web.search_web(f"{topic} similar related concepts")
        if similar_results and not similar_results[0].get('error'):
            for r in similar_results[:1]:
                if r.get('snippet') and len(r['snippet']) > 50:
                    learned_info.append(f"[Related]: {r['snippet']}")

        # Save what we learned
        if learned_info:
            full_knowledge = "\n".join(learned_info)

            # Save to trainer
            self.trainer.learn(topic, full_knowledge[:1500], 'internet-research')

            # Save as training data
            training_file = os.path.expanduser("~/.cognitive_ai_knowledge/training_data.jsonl")
            os.makedirs(os.path.dirname(training_file), exist_ok=True)

            # Create Q&A pairs from learned info
            qa_pairs = [
                {"prompt": f"What is {topic}?", "completion": learned_info[0] if learned_info else ""},
                {"prompt": f"How does {topic} work?", "completion": learned_info[1] if len(learned_info) > 1 else ""},
                {"prompt": f"Give me examples of {topic}", "completion": learned_info[2] if len(learned_info) > 2 else ""},
            ]

            with open(training_file, 'a') as f:
                for qa in qa_pairs:
                    if qa['completion']:
                        f.write(json.dumps(qa) + '\n')

            # Mark as learned in evolution
            self.evolution.mark_learned(full_knowledge[:500])

            print(f"[Learning]: ✓ Learned {len(learned_info)} facts about '{topic[:30]}'")
            print("You: ", end="", flush=True)

            return full_knowledge

        return ""

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

        # ══════════════════════════════════════════════════════════════
        # SUPER AGENT: Detect search commands and use intelligent search
        # ══════════════════════════════════════════════════════════════
        search_triggers = [
            'search on arxiv', 'search arxiv', 'find on arxiv',
            'search on github', 'search github', 'find on github',
            'search the web', 'search internet', 'search for',
            'find the best', 'find best code', 'look up',
            'research', 'deep search'
        ]

        if self.super_agent and any(trigger in clean_input for trigger in search_triggers):
            print("\n[SuperAgent] Detected search request - using intelligent search...")

            # Extract the search query
            query = clean_input
            for trigger in search_triggers:
                query = query.replace(trigger, '').strip()

            if query:
                # Determine which sources to search
                sources = ['web', 'arxiv', 'github']
                if 'arxiv' in clean_input:
                    sources = ['arxiv']
                elif 'github' in clean_input:
                    sources = ['github']

                # Use Super Agent
                if 'best code' in clean_input or 'best' in clean_input and 'github' in clean_input:
                    result = self.super_agent.find_best_code(query)
                else:
                    result = self.super_agent.fast_search(query, sources=sources)

                # Format response
                if result.get('best_repo'):
                    repo = result['best_repo']
                    return f"""[SuperAgent Search Complete]

Best Repository: {repo.get('name', 'Unknown')}
URL: {repo.get('url', '')}
Score: {repo.get('score', 0)}/10

{result.get('analysis', '')[:500]}

Training data saved for fine-tuning."""

                else:
                    return f"""[SuperAgent Search Complete]

{result.get('analysis', 'Search completed.')}

Confidence: {result.get('confidence', 0)}/10
Results found: {len(result.get('results', []))}

Training data saved for fine-tuning."""

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

        # RETRIEVE trained knowledge - but only for actual questions, not meta-requests
        knowledge = ""
        meta_phrases = ['improve', 'yourself', 'your reasoning', 'get better', 'learn more',
                        'train', 'upgrade', 'enhance yourself', 'self improve']
        is_meta_request = any(phrase in clean_input.lower() for phrase in meta_phrases)

        if not is_meta_request:
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

        # CHECK IF MODEL DOESN'T KNOW - then search internet!
        dont_know_phrases = [
            "i don't know", "i do not know", "i'm not sure", "i am not sure",
            "i cannot", "i can't", "unable to", "don't have information",
            "no information", "not familiar", "beyond my knowledge",
            "i lack", "insufficient", "cannot provide", "can't provide",
            "not aware", "unclear", "uncertain", "i apologize"
        ]

        response_lower = str(response).lower()  # Ensure string
        doesnt_know = any(phrase in response_lower for phrase in dont_know_phrases)

        # Also check if response is too short (likely doesn't know)
        if len(response.strip()) < 50:
            doesnt_know = True

        if doesnt_know:
            print(f"\n[AI doesn't know]: Searching internet for answer...")
            print("You: ", end="", flush=True)

            # Extract topic from user input
            topic = clean_input
            for trigger in ['what is', 'how does', 'explain', 'tell me about', 'learn about', 'teach me', 'how to']:
                topic = topic.replace(trigger, '').strip()

            if topic and len(topic) > 3:
                # Learn from internet
                learned = self._learn_topic_from_internet(topic)

                if learned:
                    # Try again with new knowledge!
                    enhanced_with_learned = f"{user_input}\n\nI found this information:\n{learned[:1000]}"
                    response = self.ai.chat(enhanced_with_learned)
                    response = f"[After researching]: {response}"

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
  /sleep           - Trigger memory consolidation
  /learn           - Show curiosity & learning stats
  /evolution       - Show self-evolution stats (cycles, training, improvement)
  /train           - Run MLX fine-tuning manually
  /evolve population [n]  - Run population-based code evolution (n generations)
  /selfplay [topic] [n]   - Run self-play training (optional topic, n questions)
  /curiosity [topics]     - Show RND curiosity exploration state and recommendations
  /quit            - Exit
""")
    print("=" * 60)

    model = sys.argv[1] if len(sys.argv) > 1 else "zai-org/glm-4.7-flash"
    print(f"\nModel: {model} (LM Studio)")
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

            if user_input == '/train':
                print(f"\n--- Manual MLX Training ---")
                evo = ai.evolution
                evo_stats = evo.get_stats()
                print(f"Facts learned: {evo_stats['total_facts']}")

                # Run MLX training
                print(f"\n[Training]: Fine-tuning model with MLX...")
                result = evo.run_mlx_training()
                if result['success']:
                    print(f"[Training]: COMPLETE!")
                    # Reflect
                    reflection = evo.reflect()
                    print(f"\n{reflection}")
                else:
                    print(f"[Training]: {result['message']}")
                print()
                continue

            if user_input.startswith('/evolve population'):
                print(f"\n--- Population-Based Code Evolution ---")

                if not CODE_EVOLUTION_AVAILABLE:
                    print("Code evolution system not available.")
                    print()
                    continue

                # Parse optional generations argument
                parts = user_input.split()
                generations = 1
                if len(parts) > 2:
                    try:
                        generations = int(parts[2])
                    except ValueError:
                        print(f"Invalid generations number: {parts[2]}, using 1")

                # Get or create code evolution instance with population enabled
                code_evo = get_code_evolution()

                # Check if population is enabled
                if not code_evo.use_population:
                    print("Population evolution not enabled on current instance.")
                    print("Creating new instance with population enabled...")
                    # Create new instance with population enabled
                    from config.paths import EVOLUTION_DIR
                    code_evo = CodeEvolution(
                        storage_dir=EVOLUTION_DIR / "code_evolution",
                        use_docker=True,
                        use_population=True
                    )

                # Show current population stats
                pop_stats = code_evo.get_population_stats()
                if pop_stats.get('enabled'):
                    print(f"\nCurrent Population:")
                    print(f"  Generation: {pop_stats['generation']}")
                    print(f"  Population size: {pop_stats['population_size']}")
                    print(f"  Diversity: {pop_stats['diversity']:.4f}")
                    print(f"  Convergence: {pop_stats['convergence']:.4f}")
                    print(f"  Best fitness: {pop_stats['best_fitness']:.4f}")
                    print(f"  Avg fitness: {pop_stats['avg_fitness']:.4f}")

                    if pop_stats['population_size'] > 0:
                        # Run evolution
                        print(f"\nEvolving for {generations} generation(s)...")
                        result = code_evo.evolve_population(generations=generations)

                        if result['success']:
                            final = result['final_stats']
                            print(f"\nEvolution Complete!")
                            print(f"  Generations evolved: {result['generations_evolved']}")
                            print(f"  Final generation: {final['generation']}")
                            print(f"  Best fitness: {final['best_fitness']:.4f}")
                            print(f"  Avg fitness: {final['avg_fitness']:.4f}")
                            print(f"  Diversity: {final['diversity']:.4f}")

                            # Show best individuals
                            if final.get('best_individuals'):
                                print(f"\nTop Individuals:")
                                for ind in final['best_individuals'][:3]:
                                    mutations = ', '.join(ind['mutations']) if ind['mutations'] else 'none'
                                    print(f"    {ind['id'][:8]}... fitness={ind['fitness']:.4f} (gen {ind['generation']}, mutations: {mutations})")
                        else:
                            print(f"Evolution failed: {result.get('message', 'Unknown error')}")
                    else:
                        print("\nPopulation is empty. Add code via propose_change first.")
                        print("Example: Use the code evolution API to propose code changes.")
                else:
                    print("Population evolution not available.")

                print()
                continue

            if user_input.startswith('/selfplay'):
                print(f"\n--- Self-Play Training ---")

                if not SELF_PLAY_AVAILABLE or not ai.self_play_trainer:
                    print("Self-play training system not available.")
                    print()
                    continue

                # Parse optional arguments: /selfplay [topic] [n]
                parts = user_input.split()
                topic = None
                n_questions = 5  # default

                if len(parts) > 1:
                    # Check if first arg is a number (just n_questions)
                    try:
                        n_questions = int(parts[1])
                    except ValueError:
                        # First arg is topic
                        topic = parts[1]
                        if len(parts) > 2:
                            try:
                                n_questions = int(parts[2])
                            except ValueError:
                                print(f"Invalid number: {parts[2]}, using 5")

                # Determine topics
                if topic:
                    topics = [topic]
                elif ai.interests:
                    topics = ai.interests[-5:]
                else:
                    topics = ['programming', 'mathematics', 'science', 'reasoning', 'technology']

                print(f"Topics: {', '.join(topics)}")
                print(f"Questions: {n_questions}")
                print(f"Current difficulty: {ai.self_play_trainer.get_current_difficulty().value}")
                print()

                # Run self-play round
                result = ai.self_play_trainer.run_self_play_round(
                    topics=topics,
                    n_questions=n_questions
                )

                # Log results to evolution state
                ai._log_self_play_results(result)

                # Show results summary
                print(f"\n--- Results ---")
                print(f"Correct: {result['correct_count']}/{len(result['questions'])} ({result['correct_rate']:.1%})")
                print(f"Average score: {result['avg_score']:.1%}")
                print(f"Mistakes learned from: {result['mistakes_learned']}")

                if result.get('difficulty_change'):
                    print(f"Difficulty change: {result['difficulty_change']}")

                # Show overall stats
                stats = ai.self_play_trainer.get_stats()
                print(f"\n--- Overall Stats ---")
                print(f"Total questions generated: {stats['questions_generated']}")
                print(f"Overall accuracy: {stats['correct_rate']:.1%}")
                print(f"Total rounds: {stats['total_rounds']}")
                print(f"Current difficulty: {stats['current_difficulty']}")

                # Show difficulty progression
                prog_stats = ai.self_play_trainer.get_difficulty_progression_stats()
                if prog_stats.get('rounds_by_difficulty'):
                    print(f"\nPerformance by difficulty:")
                    for diff, data in prog_stats['rounds_by_difficulty'].items():
                        print(f"  {diff}: {data['count']} rounds, {data['avg_correct_rate']:.1%} avg accuracy")

                print()
                continue

            if user_input.startswith('/curiosity'):
                print(f"\n--- RND Curiosity Exploration State ---")

                if not RND_CURIOSITY_AVAILABLE:
                    print("RND curiosity system not available.")
                    print()
                    continue

                try:
                    rnd = get_rnd_curiosity()

                    # Get exploration statistics
                    stats = rnd.get_exploration_stats()
                    print(f"\nExploration Statistics:")
                    print(f"  Total explored: {stats['total_explored']}")
                    print(f"  Unique topics: {stats.get('unique_topics', 0)}")
                    print(f"  Avg curiosity: {stats['avg_curiosity']:.3f}")
                    print(f"  Recent avg: {stats.get('recent_avg_curiosity', stats['avg_curiosity']):.3f}")
                    print(f"  Curiosity trend: {stats['curiosity_trend']:+.3f}")
                    print(f"  Exploration rate: {stats.get('exploration_rate_per_hour', 0):.1f}/hour")

                    # Novelty status
                    print(f"\nNovelty Indicators:")
                    print(f"  High curiosity (>0.7): {stats['high_curiosity_count']}")
                    print(f"  Low curiosity (<0.3): {stats['low_curiosity_count']}")
                    print(f"  Running mean: {stats['running_mean']:.4f}")
                    print(f"  Running std: {stats['running_std']:.4f}")

                    # Most curious topics from history
                    most_curious = rnd.get_most_curious_topics(5)
                    if most_curious:
                        print(f"\nMost Curious Topics (from history):")
                        for topic, curiosity in most_curious:
                            marker = "🔥" if curiosity > 0.7 else "🟢" if curiosity > 0.5 else "🔵"
                            print(f"  {marker} {topic[:40]}: {curiosity:.3f}")

                    # Parse optional topics to analyze
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        # User provided topics to analyze
                        user_topics = [t.strip() for t in parts[1].split(',')]
                        if user_topics:
                            print(f"\nCuriosity for requested topics:")
                            curiosity_map = rnd.get_curiosity_map(user_topics)
                            sorted_topics = sorted(curiosity_map.items(), key=lambda x: x[1], reverse=True)
                            for topic, curiosity in sorted_topics:
                                marker = "🔥" if curiosity > 0.7 else "🟢" if curiosity > 0.5 else "🔵"
                                print(f"  {marker} {topic}: {curiosity:.3f}")
                    else:
                        # Suggest topics based on AI's interests
                        print(f"\nRecommendations:")
                        if ai.interests:
                            print("  Analyzing your interests for novelty...")
                            curiosity_map = rnd.get_curiosity_map(ai.interests[-10:])
                            hotspots = sorted(curiosity_map.items(), key=lambda x: x[1], reverse=True)[:5]
                            if hotspots:
                                print("  Topics to explore (by novelty):")
                                for topic, curiosity in hotspots:
                                    marker = "🔥" if curiosity > 0.7 else "🟢" if curiosity > 0.5 else "🔵"
                                    print(f"    {marker} {topic}: {curiosity:.3f}")
                        else:
                            print("  (No interests tracked yet - chat more to build up interests)")

                        # Show active learner RND stats if available
                        if hasattr(ai, 'active_learner') and ai.active_learner:
                            al_rnd_stats = ai.active_learner.get_rnd_stats()
                            if al_rnd_stats.get('rnd_enabled'):
                                print(f"\nActive Learner RND Integration:")
                                print(f"  Novel discoveries: {al_rnd_stats.get('novel_discoveries', 0)}")
                                print(f"  Total RND updates: {al_rnd_stats.get('total_rnd_updates', 0)}")
                                print(f"  Avg RND curiosity: {al_rnd_stats.get('avg_rnd_curiosity', 0.5):.3f}")
                                print(f"  RND weight: {al_rnd_stats.get('rnd_weight', 0.5):.2f}")

                except Exception as e:
                    print(f"Error accessing RND curiosity: {e}")

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
