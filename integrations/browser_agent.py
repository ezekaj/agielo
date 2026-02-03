"""
Browser Agent - AI that can browse and interact with websites
=============================================================

Uses Playwright/Selenium to:
1. Open websites
2. Read content
3. Click buttons
4. Fill forms
5. Take screenshots
6. Learn from web pages

This gives the AI eyes and hands on the web!
"""

import os
import time
import re
from datetime import datetime
from typing import Dict, List

# Try different browser automation libraries
BROWSER_AVAILABLE = False
BROWSER_TYPE = None

try:
    from playwright.sync_api import sync_playwright
    BROWSER_AVAILABLE = True
    BROWSER_TYPE = "playwright"
except ImportError:
    pass

if not BROWSER_AVAILABLE:
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        BROWSER_AVAILABLE = True
        BROWSER_TYPE = "selenium"
    except ImportError:
        pass


class BrowserAgent:
    """
    AI agent that can browse and interact with websites.

    Capabilities:
    - Navigate to URLs
    - Read page content
    - Click elements
    - Fill forms
    - Scroll pages
    - Take screenshots
    - Extract information
    """

    def __init__(self, headless: bool = True):
        """
        Initialize browser agent.

        Args:
            headless: Run browser without GUI (faster)
        """
        self.headless = headless
        self.browser = None
        self.page = None
        self.history = []
        self.screenshots_dir = os.path.expanduser("~/.cognitive_ai_screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)

        if not BROWSER_AVAILABLE:
            print("[Browser] No browser automation available")
            print("[Browser] Install: pip install playwright && playwright install")
            print("[Browser] Or: pip install selenium webdriver-manager")

    def start(self):
        """Start the browser."""
        if not BROWSER_AVAILABLE:
            return False

        if BROWSER_TYPE == "playwright":
            self._start_playwright()
        elif BROWSER_TYPE == "selenium":
            self._start_selenium()

        return self.browser is not None

    def _start_playwright(self):
        """Start Playwright browser."""
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context()
            self.page = self.context.new_page()
        except Exception as e:
            print(f"[Browser] Playwright error: {e}")

    def _start_selenium(self):
        """Start Selenium browser."""
        try:
            options = Options()
            if self.headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            service = Service(ChromeDriverManager().install())
            self.browser = webdriver.Chrome(service=service, options=options)
            self.page = self.browser
        except Exception as e:
            print(f"[Browser] Selenium error: {e}")

    def goto(self, url: str) -> Dict:
        """
        Navigate to a URL.

        Args:
            url: The URL to visit

        Returns:
            Dict with page info
        """
        if not self.page:
            if not self.start():
                return {'error': 'Browser not available'}

        try:
            if BROWSER_TYPE == "playwright":
                self.page.goto(url, timeout=30000)
                title = self.page.title()
            else:
                self.browser.get(url)
                title = self.browser.title

            self.history.append({
                'action': 'goto',
                'url': url,
                'title': title,
                'time': datetime.now().isoformat()
            })

            return {
                'success': True,
                'url': url,
                'title': title
            }

        except Exception as e:
            return {'error': str(e)}

    def get_content(self) -> str:
        """Get the text content of the current page."""
        if not self.page:
            return "[No page loaded]"

        try:
            if BROWSER_TYPE == "playwright":
                content = self.page.content()
            else:
                content = self.browser.page_source

            # Extract text from HTML
            text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            return text[:5000]  # Limit to 5000 chars

        except Exception as e:
            return f"[Error: {e}]"

    def click(self, selector: str) -> Dict:
        """
        Click an element on the page.

        Args:
            selector: CSS selector or text to click

        Returns:
            Dict with result
        """
        if not self.page:
            return {'error': 'No page loaded'}

        try:
            if BROWSER_TYPE == "playwright":
                # Try CSS selector first, then text
                try:
                    self.page.click(selector, timeout=5000)
                except Exception as css_err:
                    # CSS selector failed, try text selector as fallback
                    try:
                        self.page.click(f"text={selector}", timeout=5000)
                    except Exception as text_err:
                        raise Exception(f"CSS selector failed: {css_err}; Text selector failed: {text_err}")
            else:
                try:
                    element = self.browser.find_element(By.CSS_SELECTOR, selector)
                except Exception as css_err:
                    # CSS selector failed, try link text as fallback
                    try:
                        element = self.browser.find_element(By.LINK_TEXT, selector)
                    except Exception as link_err:
                        raise Exception(f"CSS selector failed: {css_err}; Link text failed: {link_err}")
                element.click()

            self.history.append({
                'action': 'click',
                'selector': selector,
                'time': datetime.now().isoformat()
            })

            return {'success': True, 'clicked': selector}

        except Exception as e:
            return {'error': str(e)}

    def fill(self, selector: str, text: str) -> Dict:
        """
        Fill a form field.

        Args:
            selector: CSS selector for the input
            text: Text to enter

        Returns:
            Dict with result
        """
        if not self.page:
            return {'error': 'No page loaded'}

        try:
            if BROWSER_TYPE == "playwright":
                self.page.fill(selector, text)
            else:
                element = self.browser.find_element(By.CSS_SELECTOR, selector)
                element.clear()
                element.send_keys(text)

            return {'success': True, 'filled': selector}

        except Exception as e:
            return {'error': str(e)}

    def scroll(self, direction: str = "down") -> Dict:
        """Scroll the page."""
        if not self.page:
            return {'error': 'No page loaded'}

        try:
            if BROWSER_TYPE == "playwright":
                if direction == "down":
                    self.page.evaluate("window.scrollBy(0, 500)")
                else:
                    self.page.evaluate("window.scrollBy(0, -500)")
            else:
                if direction == "down":
                    self.browser.execute_script("window.scrollBy(0, 500)")
                else:
                    self.browser.execute_script("window.scrollBy(0, -500)")

            return {'success': True, 'scrolled': direction}

        except Exception as e:
            return {'error': str(e)}

    def screenshot(self, name: str = None) -> str:
        """Take a screenshot."""
        if not self.page:
            return "[No page loaded]"

        try:
            name = name or f"screenshot_{int(time.time())}"
            path = os.path.join(self.screenshots_dir, f"{name}.png")

            if BROWSER_TYPE == "playwright":
                self.page.screenshot(path=path)
            else:
                self.browser.save_screenshot(path)

            return path

        except Exception as e:
            return f"[Error: {e}]"

    def get_links(self) -> List[Dict]:
        """Get all links on the page."""
        if not self.page:
            return []

        try:
            if BROWSER_TYPE == "playwright":
                links = self.page.eval_on_selector_all(
                    'a[href]',
                    'elements => elements.map(e => ({text: e.innerText, href: e.href}))'
                )
            else:
                elements = self.browser.find_elements(By.TAG_NAME, 'a')
                links = [{'text': e.text, 'href': e.get_attribute('href')} for e in elements]

            return [l for l in links if l['href'] and l['text']][:20]

        except Exception as e:
            return []

    def search_google(self, query: str) -> List[Dict]:
        """
        Search Google and return results.

        Args:
            query: Search query

        Returns:
            List of search results
        """
        self.goto(f"https://www.google.com/search?q={query}")
        time.sleep(2)  # Wait for results

        content = self.get_content()
        links = self.get_links()

        # Filter to actual search results
        results = [l for l in links if 'google.com' not in l.get('href', '')]

        return results[:10]

    def close(self):
        """Close the browser."""
        try:
            if BROWSER_TYPE == "playwright":
                if self.browser:
                    self.browser.close()
                if hasattr(self, 'playwright'):
                    self.playwright.stop()
            else:
                if self.browser:
                    self.browser.quit()
        except (OSError, RuntimeError, AttributeError) as e:
            # Ignore errors during browser cleanup - browser may already be closed
            # or resources may have been freed
            print(f"[Browser] Cleanup warning (safe to ignore): {e}")


# Simple command interface
def run_browser_command(agent: BrowserAgent, command: str) -> str:
    """Run a natural language browser command."""
    command = command.lower().strip()

    if command.startswith("go to ") or command.startswith("open "):
        url = command.split(" ", 2)[-1]
        if not url.startswith("http"):
            url = "https://" + url
        result = agent.goto(url)
        if result.get('success'):
            return f"Opened: {result['title']}"
        return f"Error: {result.get('error')}"

    elif command.startswith("search "):
        query = command[7:]
        results = agent.search_google(query)
        if results:
            output = f"Found {len(results)} results:\n"
            for r in results[:5]:
                output += f"• {r['text'][:50]} - {r['href'][:50]}\n"
            return output
        return "No results found"

    elif command == "read" or command == "content":
        return agent.get_content()[:1000]

    elif command.startswith("click "):
        target = command[6:]
        result = agent.click(target)
        return f"Clicked: {target}" if result.get('success') else f"Error: {result.get('error')}"

    elif command == "screenshot":
        path = agent.screenshot()
        return f"Screenshot saved: {path}"

    elif command == "links":
        links = agent.get_links()
        return "\n".join([f"• {l['text'][:30]} - {l['href'][:50]}" for l in links[:10]])

    elif command == "scroll" or command == "scroll down":
        agent.scroll("down")
        return "Scrolled down"

    elif command == "scroll up":
        agent.scroll("up")
        return "Scrolled up"

    else:
        return "Unknown command. Try: go to <url>, search <query>, read, click <element>, screenshot, links, scroll"


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("BROWSER AGENT TEST")
    print("=" * 60)

    if not BROWSER_AVAILABLE:
        print("\nBrowser automation not available!")
        print("Install one of:")
        print("  pip install playwright && playwright install chromium")
        print("  pip install selenium webdriver-manager")
        exit(1)

    print(f"\nUsing: {BROWSER_TYPE}")

    agent = BrowserAgent(headless=True)

    # Test
    print("\nSearching for 'Python programming'...")
    results = agent.search_google("Python programming")
    for r in results[:3]:
        print(f"  • {r['text'][:40]}")

    print("\nGoing to Wikipedia...")
    result = agent.goto("https://en.wikipedia.org/wiki/Artificial_intelligence")
    print(f"  Title: {result.get('title')}")

    print("\nReading content...")
    content = agent.get_content()
    print(f"  {content[:200]}...")

    agent.close()
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
