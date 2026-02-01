#!/usr/bin/env python3
"""
Cognitive Chat - AI that Learns from the Internet
==================================================

The AI actively searches and learns when you're not talking!

Features:
- Searches the web for topics it's curious about
- Reads articles and learns new things
- Reflects on what it learned
- Shares discoveries with you

Run: python3 chat.py [model_name]
"""

import sys
import os
import time
import threading
import random
import json
import urllib.request
import urllib.parse
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrations.cognitive_ollama import CognitiveOllama
from integrations.self_training import SelfTrainer
from integrations.browser_agent import BrowserAgent, run_browser_command, BROWSER_AVAILABLE


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
    """AI that thinks, reflects, learns from the internet, and TRAINS ITSELF."""

    def __init__(self, model: str = "ministral-3:8b"):
        self.ai = CognitiveOllama(model=model)
        self.web = WebLearner()
        self.trainer = SelfTrainer()  # Self-training module!
        self.last_interaction = time.time()
        self.idle_threshold = 20  # seconds before AI starts learning
        self.is_busy = False
        self.thoughts = []
        self.running = True
        self.interests = []  # Topics the AI is interested in

        # Load previous knowledge count
        stats = self.trainer.get_stats()
        print(f"[Knowledge] Loaded {stats['total_facts']} facts from previous sessions")

        # Browser agent for deep web learning
        self.browser = None
        if BROWSER_AVAILABLE:
            print("[Browser] Playwright available - can browse websites!")
        else:
            print("[Browser] Install: pip install playwright && playwright install chromium")

        # Start background learning thread
        self.learning_thread = threading.Thread(target=self._autonomous_loop, daemon=True)
        self.learning_thread.start()

    def _autonomous_loop(self):
        """Background loop for autonomous learning."""
        while self.running:
            time.sleep(5)

            idle_time = time.time() - self.last_interaction

            if idle_time > self.idle_threshold and not self.is_busy:
                self._do_autonomous_activity()

    def _do_autonomous_activity(self):
        """Perform autonomous learning - ONLY when there's something NEW to learn."""
        self.is_busy = True

        # Only do activities if we have conversation context
        if not self.ai.history:
            self.is_busy = False
            return

        # Check if we already processed this conversation
        last_processed = getattr(self, '_last_processed_idx', -1)
        current_idx = len(self.ai.history) - 1

        if current_idx <= last_processed:
            # Nothing new to learn - stay quiet
            self.is_busy = False
            return

        # Mark as processed so we don't repeat
        self._last_processed_idx = current_idx

        # Do ONE meaningful action (not random repetition)
        thought = self._learn_and_reflect_once()

        if thought:
            self.thoughts.append({
                'time': datetime.now().isoformat(),
                'thought': thought,
                'type': 'autonomous'
            })
            print(f"\n[AI]: {thought}")
            print("You: ", end="", flush=True)

        self.is_busy = False

    def _learn_and_reflect_once(self) -> Optional[str]:
        """Learn from the latest conversation ONCE - no repetition."""
        if not self.ai.history:
            return None

        recent = self.ai.history[-1]
        user_input = recent['user'].strip()
        ai_response = recent.get('assistant', '')

        # Skip if user input is too short or empty
        if len(user_input) < 5:
            return None

        # 1. Save good responses as training data (silently)
        if len(ai_response) > 200:
            self._save_training_pair(user_input, ai_response)
            self.trainer.learn(
                topic=user_input[:100],
                content=ai_response[:500],
                source="self-learning"
            )

        # 2. Search for MORE information on this topic
        results = self.web.search_web(user_input)
        if results and not results[0].get('error'):
            best = results[0]
            snippet = best.get('snippet', '')[:300]
            if snippet and len(snippet) > 50:
                self.trainer.learn(user_input, snippet, best.get('source', 'web'))
                return f"Found more on '{user_input[:30]}...': {snippet[:100]}..."

        # 3. If no new info found, reflect briefly
        return f"Noted your interest in '{user_input[:40]}...' - will look for more."


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
        """Process user input, using trained knowledge."""
        self.last_interaction = time.time()

        # Store the WHOLE request as an interest (not single words)
        clean_input = user_input.strip().lower()
        if clean_input and len(clean_input) > 3 and clean_input not in self.interests:
            self.interests.append(clean_input)
            # Keep only recent interests
            if len(self.interests) > 20:
                self.interests = self.interests[-20:]

        # RETRIEVE trained knowledge relevant to the question
        knowledge = self.trainer.get_knowledge_for_prompt(user_input)

        # Add knowledge to the conversation context
        if knowledge:
            enhanced_input = f"{user_input}\n{knowledge}"
        else:
            enhanced_input = user_input

        return self.ai.chat(enhanced_input)

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
