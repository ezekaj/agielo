#!/usr/bin/env python3
"""
Super Web Agent for LM Studio
=============================
Combines:
1. Fast search (ArXiv, GitHub, DuckDuckGo) - no browser needed
2. Browser-use for full page interaction (click, fill, scroll)
3. LM Studio integration for local LLM
4. Training data generation for fine-tuning

Usage:
    agent = SuperAgent()

    # Fast search (no browser)
    result = agent.fast_search("chain of thought prompting")

    # Deep browse with browser
    result = agent.browse("https://arxiv.org", task="find papers about transformers")

    # Auto mode - decides best approach
    result = agent.auto("find the best web agent code on GitHub")
"""

import asyncio
import json
import os
import sys
import time
import re
import urllib.request
import urllib.parse
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import TRAINING_DATA_FILE, LM_STUDIO_URL, DEFAULT_MODEL

# Try to import browser-use (optional)
try:
    from browser_use import Agent, Browser, BrowserConfig
    from langchain_openai import ChatOpenAI
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    print("[Warning] browser-use not available - using fast search only")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    content: str = ""
    score: float = 0.0


class SuperAgent:
    """
    Super Web Agent combining fast search + browser control.
    Works with LM Studio for local inference.
    """

    def __init__(self,
                 lm_studio_url: str = None,
                 model: str = None,
                 headless: bool = True):

        self.lm_studio_url = lm_studio_url or LM_STUDIO_URL
        self.model = model or DEFAULT_MODEL
        self.headless = headless
        self.training_data = []

        # Check LM Studio connection
        self._check_lm_studio()

    def _check_lm_studio(self):
        """Check if LM Studio is running."""
        try:
            req = urllib.request.Request(f"{self.lm_studio_url}/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"[OK] LM Studio connected: {self.model}")
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"[Warning] LM Studio not running at {self.lm_studio_url}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    # LM STUDIO INTEGRATION
    # ═══════════════════════════════════════════════════════════════════

    def ask_model(self, prompt: str, system: str = None) -> str:
        """Ask LM Studio model a question."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }).encode('utf-8')

        req = urllib.request.Request(
            f"{self.lm_studio_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except Exception as e:
            return f"[LM Studio Error: {e}]"

    # ═══════════════════════════════════════════════════════════════════
    # FAST SEARCH (No Browser)
    # ═══════════════════════════════════════════════════════════════════

    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Fast DuckDuckGo search."""
        results = []
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8')

            snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)<', html)
            titles = re.findall(r'class="result__a"[^>]*>([^<]+)<', html)

            for title, snippet in zip(titles[:max_results], snippets[:max_results]):
                results.append(SearchResult(
                    title=title.strip(), url="", snippet=snippet.strip(), source="DuckDuckGo"
                ))
        except Exception as e:
            print(f"  DuckDuckGo error: {e}")
        return results

    def search_arxiv(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Fast ArXiv search."""
        results = []
        try:
            url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&max_results={max_results}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as response:
                xml = response.read().decode('utf-8')

            entries = re.findall(r'<entry>(.*?)</entry>', xml, re.DOTALL)
            for entry in entries[:max_results]:
                title = re.search(r'<title>([^<]+)</title>', entry)
                summary = re.search(r'<summary>([^<]+)</summary>', entry, re.DOTALL)
                link = re.search(r'<id>([^<]+)</id>', entry)

                if title and summary:
                    results.append(SearchResult(
                        title=title.group(1).strip().replace('\n', ' '),
                        url=link.group(1) if link else "",
                        snippet=summary.group(1).strip()[:300].replace('\n', ' '),
                        source="ArXiv"
                    ))
        except Exception as e:
            print(f"  ArXiv error: {e}")
        return results

    def search_github(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Fast GitHub search."""
        results = []
        try:
            url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&per_page={max_results}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            for repo in data.get('items', [])[:max_results]:
                results.append(SearchResult(
                    title=repo.get('full_name', ''),
                    url=repo.get('html_url', ''),
                    snippet=f"{repo.get('description', '')} | Stars: {repo.get('stargazers_count', 0)}",
                    source="GitHub"
                ))
        except Exception as e:
            print(f"  GitHub error: {e}")
        return results

    def fetch_page(self, url: str) -> str:
        """Fetch page content without browser."""
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')

            # Clean HTML
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:5000]
        except Exception as e:
            return f"Error: {e}"

    def fetch_github_code(self, repo_url: str) -> Dict:
        """Fetch README and code from GitHub repo."""
        result = {"readme": "", "code": "", "files": []}
        try:
            parts = repo_url.replace("https://github.com/", "").split("/")
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1].split("?")[0]

                # Fetch README
                for branch in ["main", "master"]:
                    try:
                        readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
                        req = urllib.request.Request(readme_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            result["readme"] = resp.read().decode('utf-8')[:2000]
                        break
                    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                        # Try next branch (main/master fallback)
                        continue

                # Fetch file list
                api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
                try:
                    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        files = json.loads(resp.read().decode('utf-8'))

                    py_files = [f['name'] for f in files if f.get('name', '').endswith('.py')]
                    result["files"] = py_files[:10]

                    # Fetch first Python file
                    if py_files:
                        code_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{py_files[0]}"
                        try:
                            req = urllib.request.Request(code_url, headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req, timeout=10) as resp:
                                result["code"] = resp.read().decode('utf-8')[:1500]
                        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                            # Code file not accessible - silently skip
                            pass
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as e:
                    # API request failed - silently skip
                    pass
        except (ValueError, IndexError, AttributeError) as e:
            # URL parsing failed - silently skip
            pass
        return result

    # ═══════════════════════════════════════════════════════════════════
    # FAST SEARCH (Main method)
    # ═══════════════════════════════════════════════════════════════════

    def fast_search(self, query: str, sources: List[str] = None) -> Dict:
        """
        Fast search across multiple sources (no browser needed).

        Args:
            query: Search query
            sources: ['web', 'arxiv', 'github'] - default all
        """
        if sources is None:
            sources = ['web', 'arxiv', 'github']

        print(f"\n{'='*60}")
        print(f"FAST SEARCH: {query}")
        print(f"{'='*60}")

        start_time = time.time()
        all_results = []

        # Search all sources
        if 'web' in sources:
            print(f"  Searching DuckDuckGo...", end=" ")
            results = self.search_duckduckgo(query)
            print(f"found {len(results)}")
            all_results.extend(results)

        if 'arxiv' in sources:
            print(f"  Searching ArXiv...", end=" ")
            results = self.search_arxiv(query)
            print(f"found {len(results)}")
            all_results.extend(results)

        if 'github' in sources:
            print(f"  Searching GitHub...", end=" ")
            results = self.search_github(query)
            print(f"found {len(results)}")
            all_results.extend(results)

        elapsed = time.time() - start_time
        print(f"\n  Total: {len(all_results)} results in {elapsed:.1f}s")

        # Analyze with LM Studio
        print(f"\n  Analyzing with LM Studio...")

        sources_text = "\n".join([
            f"[{r.source}] {r.title}: {r.snippet[:150]}"
            for r in all_results[:8]
        ])

        analysis_prompt = f"""Analyze these search results for: {query}

{sources_text}

1. Which result is MOST relevant and why?
2. Summarize the key information found.
3. Rate confidence (1-10).

BEST RESULT: [title]
SUMMARY: [your summary]
CONFIDENCE: [1-10]"""

        analysis = self.ask_model(analysis_prompt)

        # Extract confidence
        conf_match = re.search(r'CONFIDENCE[:\s]*(\d+)', analysis)
        confidence = int(conf_match.group(1)) if conf_match else 5

        result = {
            "query": query,
            "results": [{"title": r.title, "snippet": r.snippet, "source": r.source, "url": r.url}
                       for r in all_results],
            "analysis": analysis,
            "confidence": confidence,
            "time": elapsed,
            "method": "fast_search"
        }

        # Save for training
        self._save_training(query, result)

        print(f"\n{'='*60}")
        print(f"RESULT (confidence: {confidence}/10):")
        print(f"{'='*60}")
        print(analysis[:500])

        return result

    # ═══════════════════════════════════════════════════════════════════
    # BROWSER-USE INTEGRATION (Full browser control)
    # ═══════════════════════════════════════════════════════════════════

    async def _browser_task(self, task: str, url: str = None) -> str:
        """Run a browser task using browser-use."""
        if not BROWSER_USE_AVAILABLE:
            return "browser-use not installed"

        # Use LM Studio as the LLM backend
        llm = ChatOpenAI(
            base_url=self.lm_studio_url,
            api_key="not-needed",
            model=self.model,
            temperature=0.7
        )

        browser = Browser(config=BrowserConfig(headless=self.headless))

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser
        )

        try:
            result = await agent.run()
            return result
        finally:
            await browser.close()

    def browse(self, url: str = None, task: str = None) -> Dict:
        """
        Use browser-use for full browser interaction.

        Args:
            url: Starting URL (optional)
            task: What to do (e.g., "find papers about transformers")
        """
        if not BROWSER_USE_AVAILABLE:
            print("[Error] browser-use not available. Install with: pip install browser-use")
            return {"error": "browser-use not installed"}

        print(f"\n{'='*60}")
        print(f"BROWSER AGENT: {task}")
        print(f"{'='*60}")

        full_task = task
        if url:
            full_task = f"Go to {url} and {task}"

        start_time = time.time()

        # Run async browser task
        result = asyncio.run(self._browser_task(full_task))

        elapsed = time.time() - start_time

        return {
            "task": task,
            "url": url,
            "result": str(result),
            "time": elapsed,
            "method": "browser_use"
        }

    # ═══════════════════════════════════════════════════════════════════
    # AUTO MODE - Decides best approach
    # ═══════════════════════════════════════════════════════════════════

    def auto(self, query: str) -> Dict:
        """
        Auto mode - decides whether to use fast search or browser.

        Uses fast search for:
        - Information lookup
        - Finding papers/code
        - Comparisons

        Uses browser for:
        - Clicking/interaction needed
        - Form filling
        - Login required
        """
        print(f"\n{'='*60}")
        print(f"AUTO AGENT: {query}")
        print(f"{'='*60}")

        # Ask LM Studio to decide
        decision_prompt = f"""Task: {query}

Should I use:
A) FAST SEARCH - for finding information, papers, code (no browser needed)
B) BROWSER - for clicking, filling forms, interacting with pages

Answer A or B and explain why in one sentence.
DECISION: [A or B]
REASON: [why]"""

        decision = self.ask_model(decision_prompt)

        use_browser = "B" in decision.upper() and BROWSER_USE_AVAILABLE

        print(f"  Decision: {'BROWSER' if use_browser else 'FAST SEARCH'}")

        if use_browser:
            return self.browse(task=query)
        else:
            return self.fast_search(query)

    # ═══════════════════════════════════════════════════════════════════
    # FIND BEST CODE (Complex task)
    # ═══════════════════════════════════════════════════════════════════

    def find_best_code(self, query: str) -> Dict:
        """
        Find the best code on GitHub for a task.
        Searches, browses repos, compares, decides best.
        """
        print(f"\n{'='*60}")
        print(f"FIND BEST CODE: {query}")
        print(f"{'='*60}")

        start_time = time.time()

        # 1. Search GitHub
        print(f"\n[1/4] Searching GitHub...")
        repos = self.search_github(query, max_results=5)
        print(f"  Found {len(repos)} repos")

        if not repos:
            return {"error": "No repos found"}

        # 2. Browse each repo
        print(f"\n[2/4] Browsing repos...")
        repo_data = []
        for repo in repos:
            print(f"  {repo.title}...", end=" ")
            data = self.fetch_github_code(repo.url)
            repo_data.append({
                "name": repo.title,
                "url": repo.url,
                "description": repo.snippet,
                "readme": data.get("readme", "")[:500],
                "files": data.get("files", []),
                "code": data.get("code", "")[:500]
            })
            print(f"README: {len(data.get('readme', ''))} chars")

        # 3. Analyze with LM Studio
        print(f"\n[3/4] Analyzing repos...")

        comparison_text = ""
        for i, repo in enumerate(repo_data):
            comparison_text += f"""
REPO {i+1}: {repo['name']}
Description: {repo['description']}
Files: {', '.join(repo['files'][:5])}
README preview: {repo['readme'][:200]}
---"""

        analysis_prompt = f"""Compare these GitHub repos for: {query}

{comparison_text}

Rate each repo (1-10) and pick the BEST one.

BEST REPO: [number]
SCORE: [1-10]
REASON: [why this is best]
HOW TO USE: [quick start]"""

        analysis = self.ask_model(analysis_prompt)

        # 4. Extract best
        print(f"\n[4/4] Making decision...")

        best_match = re.search(r'BEST REPO[:\s]*(\d+)', analysis)
        best_idx = int(best_match.group(1)) - 1 if best_match else 0
        best_idx = min(max(best_idx, 0), len(repo_data) - 1)

        score_match = re.search(r'SCORE[:\s]*(\d+)', analysis)
        score = int(score_match.group(1)) if score_match else 5

        best = repo_data[best_idx]
        elapsed = time.time() - start_time

        result = {
            "query": query,
            "best_repo": {
                "name": best["name"],
                "url": best["url"],
                "description": best["description"],
                "score": score
            },
            "analysis": analysis,
            "all_repos": [{"name": r["name"], "url": r["url"]} for r in repo_data],
            "time": elapsed,
            "method": "find_best_code"
        }

        # Save for training
        self._save_training(query, result)

        print(f"\n{'='*60}")
        print(f"BEST REPO: {best['name']}")
        print(f"URL: {best['url']}")
        print(f"Score: {score}/10")
        print(f"Time: {elapsed:.1f}s")
        print(f"{'='*60}")

        return result

    # ═══════════════════════════════════════════════════════════════════
    # TRAINING DATA
    # ═══════════════════════════════════════════════════════════════════

    def _save_training(self, query: str, result: Dict):
        """Save interaction for fine-tuning."""
        entry = {
            "prompt": f"Search: {query}",
            "completion": f"""<think>
I will search for: {query}
Method: {result.get('method', 'unknown')}

{result.get('analysis', '')[:500]}
</think>

RESULT: {json.dumps(result.get('best_repo', {}), indent=2) if 'best_repo' in result else result.get('analysis', '')[:300]}""",
            "source": "SuperAgent",
            "timestamp": datetime.now().isoformat()
        }

        with open(TRAINING_DATA_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        self.training_data.append(entry)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Test the Super Agent."""
    print("="*60)
    print("SUPER AGENT TEST")
    print("="*60)

    agent = SuperAgent(
        lm_studio_url="http://localhost:1234/v1",
        model="zai-org/glm-4.7-flash"
    )

    # Test 1: Fast search
    print("\n\nTEST 1: Fast Search")
    result = agent.fast_search("chain of thought prompting LLM")

    # Test 2: Find best code
    print("\n\nTEST 2: Find Best Code")
    result = agent.find_best_code("web browsing agent python")

    # Test 3: Auto mode
    print("\n\nTEST 3: Auto Mode")
    result = agent.auto("what is the best way to fine-tune LLMs locally")

    print("\n\nAll tests completed!")
    print(f"Training data saved: {len(agent.training_data)} entries")


if __name__ == "__main__":
    main()
