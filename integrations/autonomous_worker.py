#!/usr/bin/env python3
"""
Autonomous Prompt Worker
========================

Self-directed AI that works on prompts without user interaction.
Like Maestro but running from inside - the AI orchestrates itself.

Features:
1. GoalEngine - AI sets its own goals
2. PromptQueue - Background task processing
3. ContinuousEvolution - Never-stopping improvement
4. AutoFetch - Autonomous memory/GitHub/web fetching
5. SelfReflection - Learns from its own outputs

Usage:
    worker = AutonomousWorker()
    worker.start()  # Starts autonomous loop

    # Or add specific goals:
    worker.add_goal("improve the benchmark system")
    worker.add_goal("learn about transformer attention")
"""

import os
import sys
import json
import time
import queue
import threading
import hashlib
import random
import atexit
import weakref
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import KNOWLEDGE_DIR, LM_STUDIO_URL, DEFAULT_MODEL


# Track instances for atexit cleanup using weak references
_autonomous_worker_instances: List[weakref.ref] = []


def _cleanup_all_instances():
    """Cleanup function called at program exit to stop workers and save all state."""
    for ref in _autonomous_worker_instances:
        instance = ref()
        if instance is not None:
            instance.cleanup()


# Register the cleanup function with atexit
atexit.register(_cleanup_all_instances)


def _register_instance(instance: 'AutonomousWorker'):
    """Register an AutonomousWorker instance for cleanup on exit."""
    _autonomous_worker_instances.append(weakref.ref(instance))


class GoalType(Enum):
    """Types of autonomous goals."""
    LEARN = "learn"              # Learn about a topic
    IMPROVE = "improve"          # Improve code/system
    RESEARCH = "research"        # Research a topic deeply
    FETCH = "fetch"              # Fetch from GitHub/web
    EVOLVE = "evolve"            # Evolve own code
    BENCHMARK = "benchmark"      # Test and measure
    CONSOLIDATE = "consolidate"  # Consolidate memory
    EXPLORE = "explore"          # Explore new areas (RND-driven)


class GoalPriority(Enum):
    """Goal priority levels."""
    CRITICAL = 1     # Must do now
    HIGH = 2         # Important
    NORMAL = 3       # Standard
    LOW = 4          # Nice to have
    BACKGROUND = 5   # When idle


@dataclass
class Goal:
    """A goal the AI sets for itself."""
    id: str
    type: GoalType
    description: str
    priority: GoalPriority = GoalPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    progress: float = 0.0  # 0-1
    status: str = "pending"  # pending, in_progress, completed, failed
    metadata: Dict = field(default_factory=dict)
    result: Optional[Dict] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "deadline": self.deadline,
            "progress": self.progress,
            "status": self.status,
            "metadata": self.metadata,
            "result": self.result,
            "error": self.error,
            "attempts": self.attempts
        }

    @staticmethod
    def from_dict(data: Dict) -> 'Goal':
        return Goal(
            id=data["id"],
            type=GoalType(data["type"]),
            description=data["description"],
            priority=GoalPriority(data.get("priority", 3)),
            created_at=data.get("created_at", time.time()),
            deadline=data.get("deadline"),
            progress=data.get("progress", 0.0),
            status=data.get("status", "pending"),
            metadata=data.get("metadata", {}),
            result=data.get("result"),
            error=data.get("error"),
            attempts=data.get("attempts", 0)
        )


class GoalEngine:
    """
    The AI's goal-setting system.

    Generates goals based on:
    - Curiosity (what should I learn?)
    - Self-assessment (what am I bad at?)
    - External signals (new papers, code updates)
    - Evolution state (what needs improvement?)
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.goals_file = storage_path / "goals.json"
        self.goals: List[Goal] = []
        self._load_goals()

    def _generate_id(self) -> str:
        """Generate unique goal ID."""
        return hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:12]

    def create_goal(self,
                    goal_type: GoalType,
                    description: str,
                    priority: GoalPriority = GoalPriority.NORMAL,
                    metadata: Dict = None) -> Goal:
        """Create a new goal."""
        goal = Goal(
            id=self._generate_id(),
            type=goal_type,
            description=description,
            priority=priority,
            metadata=metadata or {}
        )
        self.goals.append(goal)
        self._save_goals()
        return goal

    def get_next_goal(self) -> Optional[Goal]:
        """Get highest priority pending goal."""
        pending = [g for g in self.goals if g.status == "pending"]
        if not pending:
            return None

        # Sort by priority (lower number = higher priority)
        pending.sort(key=lambda g: (g.priority.value, g.created_at))
        return pending[0]

    def update_goal(self, goal_id: str, **updates):
        """Update a goal's status/progress."""
        for goal in self.goals:
            if goal.id == goal_id:
                for key, value in updates.items():
                    if hasattr(goal, key):
                        setattr(goal, key, value)
                self._save_goals()
                return goal
        return None

    def complete_goal(self, goal_id: str, result: Dict = None):
        """Mark a goal as completed."""
        return self.update_goal(
            goal_id,
            status="completed",
            progress=1.0,
            result=result
        )

    def fail_goal(self, goal_id: str, error: str):
        """Mark a goal as failed."""
        goal = self.update_goal(goal_id, status="failed", error=error)
        if goal and goal.attempts < goal.max_attempts:
            # Retry by creating new goal
            goal.attempts += 1
            goal.status = "pending"
            self._save_goals()
        return goal

    def generate_goals_from_state(self,
                                   evolution_stats: Dict,
                                   active_learning_stats: Dict,
                                   rnd_curiosity_stats: Dict = None) -> List[Goal]:
        """
        Automatically generate goals based on system state.
        This is the AI setting its own goals!
        """
        new_goals = []

        # 1. Check if training needed
        facts_learned = evolution_stats.get('total_facts', 0)
        if facts_learned > 0 and facts_learned % 100 == 0:
            goal = self.create_goal(
                GoalType.BENCHMARK,
                f"Benchmark after learning {facts_learned} facts",
                GoalPriority.HIGH,
                {"facts_at_creation": facts_learned}
            )
            new_goals.append(goal)

        # 2. Check low-confidence topics (active learning)
        low_confidence_topics = active_learning_stats.get('low_confidence_topics', [])
        for topic in low_confidence_topics[:3]:  # Top 3
            goal = self.create_goal(
                GoalType.LEARN,
                f"Learn about: {topic['name']}",
                GoalPriority.NORMAL,
                {"topic": topic['name'], "current_confidence": topic.get('confidence', 0)}
            )
            new_goals.append(goal)

        # 3. Check high-curiosity topics (RND)
        if rnd_curiosity_stats:
            novel_areas = rnd_curiosity_stats.get('high_curiosity_areas', [])
            for area in novel_areas[:2]:  # Top 2
                goal = self.create_goal(
                    GoalType.EXPLORE,
                    f"Explore novel area: {area}",
                    GoalPriority.NORMAL,
                    {"area": area, "source": "rnd_curiosity"}
                )
                new_goals.append(goal)

        # 4. Periodic memory consolidation
        last_consolidation = evolution_stats.get('last_consolidation', 0)
        if time.time() - last_consolidation > 3600:  # 1 hour
            goal = self.create_goal(
                GoalType.CONSOLIDATE,
                "Consolidate memory and strengthen important connections",
                GoalPriority.LOW
            )
            new_goals.append(goal)

        return new_goals

    def _load_goals(self):
        """Load goals from disk."""
        if self.goals_file.exists():
            try:
                with open(self.goals_file, 'r') as f:
                    data = json.load(f)
                self.goals = [Goal.from_dict(g) for g in data]
            except Exception as e:
                print(f"[GoalEngine] Error loading goals: {e}")
                self.goals = []
        else:
            self.goals = []

    def _save_goals(self):
        """Save goals to disk."""
        try:
            with open(self.goals_file, 'w') as f:
                json.dump([g.to_dict() for g in self.goals], f, indent=2)
        except Exception as e:
            print(f"[GoalEngine] Error saving goals: {e}")

    def get_stats(self) -> Dict:
        """Get goal statistics."""
        by_status = {}
        by_type = {}
        for g in self.goals:
            by_status[g.status] = by_status.get(g.status, 0) + 1
            by_type[g.type.value] = by_type.get(g.type.value, 0) + 1

        return {
            "total": len(self.goals),
            "by_status": by_status,
            "by_type": by_type,
            "pending": by_status.get("pending", 0),
            "completed": by_status.get("completed", 0),
            "failed": by_status.get("failed", 0)
        }


class PromptQueue:
    """
    Queue of prompts to process autonomously.

    Prompts can come from:
    - User input (highest priority)
    - Goal system (normal priority)
    - Self-generated exploration (low priority)
    """

    def __init__(self, max_size: int = 1000):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self._processed_count = 0
        self._lock = threading.Lock()

    def add(self, prompt: str, priority: int = 3, metadata: Dict = None):
        """Add prompt to queue. Lower priority number = higher priority."""
        item = (priority, time.time(), prompt, metadata or {})
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            print("[PromptQueue] Queue full, dropping oldest low-priority item")

    def get(self, timeout: float = None) -> Optional[Tuple[str, Dict]]:
        """Get next prompt from queue."""
        try:
            priority, timestamp, prompt, metadata = self.queue.get(timeout=timeout)
            with self._lock:
                self._processed_count += 1
            return prompt, metadata
        except queue.Empty:
            return None

    def size(self) -> int:
        return self.queue.qsize()

    def get_stats(self) -> Dict:
        return {
            "queue_size": self.size(),
            "processed_count": self._processed_count
        }


class AutonomousWorker:
    """
    Main autonomous worker that runs independently.

    Like Maestro but from inside - the AI orchestrates itself:
    1. Sets its own goals (GoalEngine)
    2. Queues prompts (PromptQueue)
    3. Executes autonomously
    4. Learns and evolves
    """

    def __init__(self,
                 storage_path: str = None,
                 lm_studio_url: str = None,
                 model: str = None):

        self.storage_path = Path(storage_path or os.path.join(KNOWLEDGE_DIR, "autonomous"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.lm_studio_url = lm_studio_url or LM_STUDIO_URL
        self.model = model or DEFAULT_MODEL

        # Core components
        self.goal_engine = GoalEngine(self.storage_path / "goals")
        self.prompt_queue = PromptQueue()

        # State
        self.running = False
        self.is_busy = False
        self.current_goal: Optional[Goal] = None

        # Statistics
        self.stats = {
            "started_at": None,
            "prompts_processed": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "facts_learned": 0,
            "code_evolved": 0,
            "errors": []
        }
        self.stats_file = self.storage_path / "stats.json"
        self._load_stats()

        # Integrations (lazy loaded)
        self._evolution = None
        self._trainer = None
        self._active_learner = None
        self._super_agent = None
        self._code_evolution = None

        # Worker thread
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_prompt_complete: Optional[Callable[[str, Dict], None]] = None
        self.on_goal_complete: Optional[Callable[[Goal], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None

        # Register this instance for cleanup on program exit
        _register_instance(self)

    # ═══════════════════════════════════════════════════════════════════
    # LAZY LOADING INTEGRATIONS
    # ═══════════════════════════════════════════════════════════════════

    @property
    def evolution(self):
        """Lazy load self-evolution system."""
        if self._evolution is None:
            try:
                from integrations.self_evolution import get_evolution
                self._evolution = get_evolution()
            except Exception as e:
                print(f"[AutonomousWorker] Could not load evolution: {e}")
        return self._evolution

    @property
    def trainer(self):
        """Lazy load knowledge trainer."""
        if self._trainer is None:
            try:
                from integrations.self_training import get_knowledge_base
                self._trainer = get_knowledge_base()
            except Exception as e:
                print(f"[AutonomousWorker] Could not load trainer: {e}")
        return self._trainer

    @property
    def active_learner(self):
        """Lazy load active learning system."""
        if self._active_learner is None:
            try:
                from integrations.active_learning import ActiveLearner
                self._active_learner = ActiveLearner()
            except Exception as e:
                print(f"[AutonomousWorker] Could not load active learner: {e}")
        return self._active_learner

    @property
    def super_agent(self):
        """Lazy load super agent for web/github fetching."""
        if self._super_agent is None:
            try:
                from integrations.super_agent import SuperAgent
                self._super_agent = SuperAgent(
                    lm_studio_url=self.lm_studio_url,
                    model=self.model
                )
            except Exception as e:
                print(f"[AutonomousWorker] Could not load super agent: {e}")
        return self._super_agent

    @property
    def code_evolution(self):
        """Lazy load code evolution system."""
        if self._code_evolution is None:
            try:
                from integrations.code_evolution import get_code_evolution
                self._code_evolution = get_code_evolution()
            except Exception as e:
                print(f"[AutonomousWorker] Could not load code evolution: {e}")
        return self._code_evolution

    # ═══════════════════════════════════════════════════════════════════
    # LLM INTERACTION
    # ═══════════════════════════════════════════════════════════════════

    def ask_llm(self, prompt: str, system: str = None, max_tokens: int = 2000) -> str:
        """Ask the local LLM a question."""
        import urllib.request

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens
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
            return f"[LLM Error: {e}]"

    def check_llm_available(self) -> bool:
        """Check if LM Studio is running."""
        import urllib.request
        import urllib.error
        try:
            req = urllib.request.Request(f"{self.lm_studio_url}/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            return False

    # ═══════════════════════════════════════════════════════════════════
    # GOAL EXECUTION
    # ═══════════════════════════════════════════════════════════════════

    def execute_goal(self, goal: Goal) -> Dict:
        """Execute a single goal."""
        self.current_goal = goal
        goal.status = "in_progress"
        self.goal_engine._save_goals()

        result = {"success": False, "message": ""}

        try:
            if goal.type == GoalType.LEARN:
                result = self._execute_learn_goal(goal)
            elif goal.type == GoalType.IMPROVE:
                result = self._execute_improve_goal(goal)
            elif goal.type == GoalType.RESEARCH:
                result = self._execute_research_goal(goal)
            elif goal.type == GoalType.FETCH:
                result = self._execute_fetch_goal(goal)
            elif goal.type == GoalType.EVOLVE:
                result = self._execute_evolve_goal(goal)
            elif goal.type == GoalType.BENCHMARK:
                result = self._execute_benchmark_goal(goal)
            elif goal.type == GoalType.CONSOLIDATE:
                result = self._execute_consolidate_goal(goal)
            elif goal.type == GoalType.EXPLORE:
                result = self._execute_explore_goal(goal)
            else:
                result = {"success": False, "message": f"Unknown goal type: {goal.type}"}

            if result.get("success"):
                self.goal_engine.complete_goal(goal.id, result)
                self.stats["goals_completed"] += 1
            else:
                self.goal_engine.fail_goal(goal.id, result.get("message", "Unknown error"))
                self.stats["goals_failed"] += 1

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.goal_engine.fail_goal(goal.id, error_msg)
            self.stats["goals_failed"] += 1
            self.stats["errors"].append({
                "timestamp": time.time(),
                "goal_id": goal.id,
                "error": error_msg
            })
            result = {"success": False, "message": error_msg}

        self.current_goal = None
        self._save_stats()

        if self.on_goal_complete:
            self.on_goal_complete(goal)

        return result

    def _execute_learn_goal(self, goal: Goal) -> Dict:
        """Execute a learning goal."""
        topic = goal.metadata.get("topic") or goal.description.replace("Learn about: ", "")

        print(f"[Autonomous] Learning: {topic}")

        # Search for information
        if self.super_agent:
            search_result = self.super_agent.fast_search(topic, sources=['web', 'arxiv'])

            # Extract and store knowledge
            facts_learned = 0
            for r in search_result.get("results", [])[:5]:
                snippet = r.get('snippet', '')
                if snippet and self.trainer:
                    if not self.evolution or not self.evolution.is_duplicate(snippet):
                        self.trainer.learn(topic[:50], snippet[:500], r.get('source', 'web'))
                        if self.evolution:
                            self.evolution.mark_learned(snippet[:200])
                        facts_learned += 1

            # Update active learner
            if self.active_learner:
                self.active_learner.record_learning(topic, True, 0.5, 0.7)

            self.stats["facts_learned"] += facts_learned
            return {
                "success": True,
                "message": f"Learned {facts_learned} facts about {topic}",
                "facts_learned": facts_learned
            }

        return {"success": False, "message": "Super agent not available"}

    def _execute_improve_goal(self, goal: Goal) -> Dict:
        """Execute an improvement goal (code improvement)."""
        target = goal.metadata.get("target") or goal.description

        print(f"[Autonomous] Improving: {target}")

        if not self.code_evolution:
            return {"success": False, "message": "Code evolution not available"}

        # Read the target code
        code = self.code_evolution.read_own_code(target)
        if not code:
            return {"success": False, "message": f"Could not read {target}"}

        # Ask LLM for improvement suggestions
        prompt = f"""Analyze this code and suggest ONE specific improvement.
Return ONLY the improved code section, no explanations.

Code:
```python
{code[:2000]}
```

IMPROVED CODE:"""

        improved = self.ask_llm(prompt, system="You are a Python code optimizer. Return only code.")

        if "Error" in improved:
            return {"success": False, "message": improved}

        # Validate and apply improvement
        # (In real implementation, this would go through code_evolution validation)

        self.stats["code_evolved"] += 1
        return {
            "success": True,
            "message": f"Generated improvement for {target}",
            "improvement": improved[:500]
        }

    def _execute_research_goal(self, goal: Goal) -> Dict:
        """Execute a research goal (deep research)."""
        topic = goal.metadata.get("topic") or goal.description.replace("Research: ", "")

        print(f"[Autonomous] Researching: {topic}")

        if not self.super_agent:
            return {"success": False, "message": "Super agent not available"}

        # Deep search across all sources
        results = []

        # Search ArXiv for papers
        arxiv_results = self.super_agent.search_arxiv(topic, max_results=5)
        results.extend([{"source": "arxiv", "title": r.title, "snippet": r.snippet} for r in arxiv_results])

        # Search GitHub for implementations
        github_results = self.super_agent.search_github(topic, max_results=5)
        results.extend([{"source": "github", "title": r.title, "snippet": r.snippet, "url": r.url} for r in github_results])

        # Web search for general info
        web_results = self.super_agent.search_duckduckgo(topic, max_results=5)
        results.extend([{"source": "web", "title": r.title, "snippet": r.snippet} for r in web_results])

        # Store all findings
        for r in results:
            if self.trainer and r.get('snippet'):
                self.trainer.learn(topic[:50], r['snippet'][:500], r['source'])

        # Generate research summary
        summary_prompt = f"""Summarize these research findings about "{topic}":

{json.dumps(results[:10], indent=2)}

RESEARCH SUMMARY:"""

        summary = self.ask_llm(summary_prompt)

        return {
            "success": True,
            "message": f"Researched {topic} - found {len(results)} sources",
            "sources_found": len(results),
            "summary": summary[:1000]
        }

    def _execute_fetch_goal(self, goal: Goal) -> Dict:
        """Execute a fetch goal (GitHub/web fetching)."""
        url = goal.metadata.get("url")
        fetch_type = goal.metadata.get("fetch_type", "github")

        print(f"[Autonomous] Fetching: {url or goal.description}")

        if not self.super_agent:
            return {"success": False, "message": "Super agent not available"}

        if fetch_type == "github" and url:
            # Fetch GitHub repo
            data = self.super_agent.fetch_github_code(url)

            # Store README as knowledge
            if data.get("readme") and self.trainer:
                self.trainer.learn(f"github:{url}", data["readme"][:1000], "github")

            return {
                "success": True,
                "message": f"Fetched GitHub repo: {url}",
                "files": data.get("files", []),
                "readme_length": len(data.get("readme", ""))
            }
        else:
            # Generic web fetch
            content = self.super_agent.fetch_page(url or goal.description)

            if self.trainer and content:
                self.trainer.learn(url or goal.description[:50], content[:500], "web")

            return {
                "success": True,
                "message": f"Fetched content from {url or 'web'}",
                "content_length": len(content)
            }

    def _execute_evolve_goal(self, goal: Goal) -> Dict:
        """Execute an evolution goal (self-modification)."""
        target = goal.metadata.get("target", "self")

        print(f"[Autonomous] Evolving: {target}")

        if not self.code_evolution:
            return {"success": False, "message": "Code evolution not available"}

        # Get list of own files
        files = self.code_evolution.find_own_files()

        # Analyze for potential improvements
        improvements = []
        for file_path in files[:5]:  # Check first 5 files
            analysis = self.code_evolution.analyze_own_code(file_path)
            if analysis.get("complexity_score", 0) > 10:
                improvements.append({
                    "file": file_path,
                    "complexity": analysis.get("complexity_score"),
                    "suggestion": "High complexity - consider refactoring"
                })

        self.stats["code_evolved"] += len(improvements)

        return {
            "success": True,
            "message": f"Analyzed {len(files)} files, found {len(improvements)} improvement opportunities",
            "improvements": improvements
        }

    def _execute_benchmark_goal(self, goal: Goal) -> Dict:
        """Execute a benchmark goal."""
        print(f"[Autonomous] Benchmarking...")

        if not self.evolution:
            return {"success": False, "message": "Evolution system not available"}

        # Get current stats
        stats = self.evolution.get_stats()

        # Simple benchmark - test some questions
        test_questions = [
            "What is machine learning?",
            "Explain transformers in NLP",
            "What is gradient descent?"
        ]

        correct = 0
        for q in test_questions:
            response = self.ask_llm(q)
            # Simple check - if response is reasonable length and not error
            if len(response) > 50 and "Error" not in response:
                correct += 1

        score = correct / len(test_questions)

        # Record benchmark
        self.evolution.record_benchmark(score, {
            "test_questions": len(test_questions),
            "correct": correct
        })

        return {
            "success": True,
            "message": f"Benchmark complete: {score:.1%}",
            "score": score,
            "facts_learned": stats.get("total_facts", 0)
        }

    def _execute_consolidate_goal(self, goal: Goal) -> Dict:
        """Execute a memory consolidation goal."""
        print(f"[Autonomous] Consolidating memory...")

        # This would integrate with neuro_memory consolidation
        # For now, just trigger trainer save

        if self.trainer:
            self.trainer.save()

        if self.active_learner:
            self.active_learner._save_state()

        return {
            "success": True,
            "message": "Memory consolidated"
        }

    def _execute_explore_goal(self, goal: Goal) -> Dict:
        """Execute an exploration goal (RND-driven)."""
        area = goal.metadata.get("area") or goal.description.replace("Explore novel area: ", "")

        print(f"[Autonomous] Exploring: {area}")

        # Use active learner to find related unexplored topics
        if self.active_learner:
            related = self.active_learner.get_related_topics(area)

            # Learn about each related topic
            for topic in related[:3]:
                self.prompt_queue.add(
                    f"Learn about: {topic}",
                    priority=GoalPriority.LOW.value,
                    metadata={"source": "exploration", "parent_area": area}
                )

        return {
            "success": True,
            "message": f"Queued exploration of {area} and related topics"
        }

    # ═══════════════════════════════════════════════════════════════════
    # MAIN AUTONOMOUS LOOP
    # ═══════════════════════════════════════════════════════════════════

    def start(self):
        """Start the autonomous worker."""
        if self.running:
            print("[AutonomousWorker] Already running")
            return

        self.running = True
        self.stats["started_at"] = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[AutonomousWorker] Started autonomous loop")

    def stop(self):
        """Stop the autonomous worker."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[AutonomousWorker] Stopped")
        self._save_stats()

    def cleanup(self):
        """
        Cleanup resources and save state on exit.

        This method is registered with atexit to ensure the worker thread is stopped
        and state is saved when the program terminates. It:
        - Stops the worker thread if running
        - Saves statistics to disk
        - Saves goal engine state
        """
        try:
            # Stop the worker thread if running
            if self.running:
                self.running = False
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=5)

            # Save statistics
            self._save_stats()

            # Save goal engine state
            if self.goal_engine:
                try:
                    self.goal_engine._save_goals()
                except Exception as e:
                    print(f"[AutonomousWorker] Goal engine cleanup error: {e}")

            goals_count = len(self.goal_engine.goals) if self.goal_engine else 0
            print(f"[AutonomousWorker] Cleanup complete: saved stats and {goals_count} goals")
        except Exception as e:
            print(f"[AutonomousWorker] Cleanup error: {e}")

    def _run_loop(self):
        """Main autonomous loop."""
        while self.running:
            try:
                if self.is_busy:
                    time.sleep(1)
                    continue

                # Check LLM availability
                if not self.check_llm_available():
                    time.sleep(10)
                    continue

                # 1. Check prompt queue first (user/priority prompts)
                prompt_item = self.prompt_queue.get(timeout=0.1)
                if prompt_item:
                    prompt, metadata = prompt_item
                    self._process_prompt(prompt, metadata)
                    continue

                # 2. Get next goal from goal engine
                goal = self.goal_engine.get_next_goal()
                if goal:
                    self.is_busy = True
                    self.execute_goal(goal)
                    self.is_busy = False
                    continue

                # 3. Generate new goals if none pending
                if self.goal_engine.get_stats()["pending"] == 0:
                    self._generate_autonomous_goals()

                # 4. Idle - sleep briefly
                time.sleep(5)

            except Exception as e:
                print(f"[AutonomousWorker] Loop error: {e}")
                traceback.print_exc()
                self.stats["errors"].append({
                    "timestamp": time.time(),
                    "error": str(e)
                })
                time.sleep(5)

    def _process_prompt(self, prompt: str, metadata: Dict):
        """Process a prompt from the queue."""
        self.is_busy = True

        try:
            print(f"[Autonomous] Processing: {prompt[:50]}...")

            # Determine what kind of prompt this is
            prompt_lower = prompt.lower()

            if any(x in prompt_lower for x in ["learn", "study", "understand"]):
                topic = prompt.replace("Learn about:", "").replace("learn about", "").strip()
                goal = self.goal_engine.create_goal(GoalType.LEARN, prompt, metadata={"topic": topic})
                self.execute_goal(goal)

            elif any(x in prompt_lower for x in ["search", "find", "research"]):
                goal = self.goal_engine.create_goal(GoalType.RESEARCH, prompt)
                self.execute_goal(goal)

            elif any(x in prompt_lower for x in ["improve", "optimize", "fix"]):
                goal = self.goal_engine.create_goal(GoalType.IMPROVE, prompt)
                self.execute_goal(goal)

            elif any(x in prompt_lower for x in ["fetch", "get", "download"]):
                goal = self.goal_engine.create_goal(GoalType.FETCH, prompt)
                self.execute_goal(goal)

            elif any(x in prompt_lower for x in ["evolve", "modify", "change"]):
                goal = self.goal_engine.create_goal(GoalType.EVOLVE, prompt)
                self.execute_goal(goal)

            else:
                # Generic - ask LLM and learn from response
                response = self.ask_llm(prompt)
                if self.trainer and "Error" not in response:
                    self.trainer.learn(prompt[:50], response[:500], "self")

            self.stats["prompts_processed"] += 1

            if self.on_prompt_complete:
                self.on_prompt_complete(prompt, {"status": "completed"})

        except Exception as e:
            print(f"[AutonomousWorker] Prompt error: {e}")
            if self.on_error:
                self.on_error(e)

        self.is_busy = False

    def _generate_autonomous_goals(self):
        """Generate goals autonomously based on system state."""
        evolution_stats = self.evolution.get_stats() if self.evolution else {}

        active_learning_stats = {}
        if self.active_learner:
            try:
                active_learning_stats = {
                    "low_confidence_topics": self.active_learner.get_low_confidence_topics(n=5)
                }
            except (AttributeError, KeyError, TypeError) as e:
                print(f"[AutonomousWorker] Could not get active learning stats: {e}")

        rnd_stats = {}
        if self.active_learner and hasattr(self.active_learner, 'rnd_curiosity'):
            try:
                rnd_stats = {
                    "high_curiosity_areas": self.active_learner.get_high_curiosity_areas(n=3)
                }
            except (AttributeError, KeyError, TypeError) as e:
                print(f"[AutonomousWorker] Could not get RND stats: {e}")

        # Generate goals
        new_goals = self.goal_engine.generate_goals_from_state(
            evolution_stats,
            active_learning_stats,
            rnd_stats
        )

        if new_goals:
            print(f"[AutonomousWorker] Generated {len(new_goals)} autonomous goals")

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def add_goal(self, description: str,
                 goal_type: GoalType = None,
                 priority: GoalPriority = GoalPriority.NORMAL,
                 metadata: Dict = None) -> Goal:
        """Add a goal manually."""
        if goal_type is None:
            # Infer type from description
            desc_lower = description.lower()
            if any(x in desc_lower for x in ["learn", "study"]):
                goal_type = GoalType.LEARN
            elif any(x in desc_lower for x in ["improve", "fix", "optimize"]):
                goal_type = GoalType.IMPROVE
            elif any(x in desc_lower for x in ["research", "investigate"]):
                goal_type = GoalType.RESEARCH
            elif any(x in desc_lower for x in ["fetch", "get", "download"]):
                goal_type = GoalType.FETCH
            elif any(x in desc_lower for x in ["evolve", "modify"]):
                goal_type = GoalType.EVOLVE
            else:
                goal_type = GoalType.LEARN

        return self.goal_engine.create_goal(goal_type, description, priority, metadata)

    def add_prompt(self, prompt: str, priority: int = 3, metadata: Dict = None):
        """Add a prompt to the queue."""
        self.prompt_queue.add(prompt, priority, metadata)

    def get_status(self) -> Dict:
        """Get current worker status."""
        return {
            "running": self.running,
            "is_busy": self.is_busy,
            "current_goal": self.current_goal.to_dict() if self.current_goal else None,
            "queue_size": self.prompt_queue.size(),
            "goals": self.goal_engine.get_stats(),
            "stats": self.stats,
            "llm_available": self.check_llm_available()
        }

    def reflect(self) -> str:
        """Generate reflection on autonomous work."""
        status = self.get_status()

        reflection = f"""
=== AUTONOMOUS WORKER STATUS ===
Running: {status['running']}
Busy: {status['is_busy']}
LLM Available: {status['llm_available']}

=== GOALS ===
Pending: {status['goals']['pending']}
Completed: {status['goals']['completed']}
Failed: {status['goals']['failed']}

=== STATISTICS ===
Prompts processed: {status['stats']['prompts_processed']}
Goals completed: {status['stats']['goals_completed']}
Facts learned: {status['stats']['facts_learned']}
Code evolved: {status['stats']['code_evolved']}
Errors: {len(status['stats']['errors'])}

=== CURRENT ===
Queue size: {status['queue_size']}
Current goal: {status['current_goal']['description'] if status['current_goal'] else 'None'}
"""
        return reflection

    # ═══════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════

    def _load_stats(self):
        """Load stats from disk."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    self.stats.update(json.load(f))
            except (json.JSONDecodeError, IOError, OSError) as e:
                print(f"[AutonomousWorker] Could not load stats: {e}")

    def _save_stats(self):
        """Save stats to disk."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except (IOError, OSError, TypeError) as e:
            print(f"[AutonomousWorker] Could not save stats: {e}")


# Global instance
_worker: Optional[AutonomousWorker] = None


def get_autonomous_worker() -> AutonomousWorker:
    """Get global autonomous worker instance."""
    global _worker
    if _worker is None:
        _worker = AutonomousWorker()
    return _worker


# ═══════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AUTONOMOUS WORKER TEST")
    print("=" * 60)

    worker = AutonomousWorker(storage_path="/tmp/autonomous_test")

    # Add some goals
    worker.add_goal("Learn about transformer attention mechanisms")
    worker.add_goal("Research latest papers on chain-of-thought prompting")
    worker.add_goal("Improve the benchmark system", goal_type=GoalType.IMPROVE)

    # Add some prompts
    worker.add_prompt("What is machine learning?", priority=2)
    worker.add_prompt("Search for web browsing agents on GitHub", priority=3)

    print("\n" + worker.reflect())

    # Start worker
    print("\nStarting autonomous worker...")
    worker.start()

    # Let it run for a bit
    try:
        for i in range(30):
            time.sleep(1)
            if i % 10 == 0:
                print(f"\n[{i}s] Status: {worker.get_status()['goals']}")
    except KeyboardInterrupt:
        pass

    worker.stop()

    print("\n" + worker.reflect())
    print("\n" + "=" * 60)
