"""
Simple Benchmark System
=======================

Tests AI variants against each other to find which is better.

How it works:
1. Define simple test questions with expected answers
2. Run both AI versions on same questions
3. Score each response
4. Compare and pick winner
"""

import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional


class Benchmark:
    """Simple benchmark to test AI improvements."""

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.expanduser("~/.cognitive_ai_knowledge/benchmark")
        os.makedirs(self.storage_path, exist_ok=True)

        # Challenging reasoning tests (GSM8K style + logic + common sense)
        # Total: 45+ tests for statistical significance
        self.tests = [
            # === MATH REASONING (GSM8K style) ===
            {
                "question": "A store sells apples for $2 each. If John buys 5 apples and pays with a $20 bill, how much change does he get?",
                "answer": "10",
                "expected_keywords": ["10", "dollar", "change"],
                "category": "math",
                "check_exact": True,
                "difficulty": "easy"
            },
            {
                "question": "A train travels at 60 mph. How far does it travel in 2.5 hours?",
                "answer": "150",
                "expected_keywords": ["150", "miles"],
                "category": "math",
                "check_exact": True,
                "difficulty": "easy"
            },
            {
                "question": "If a rectangle has length 8 and width 5, what is its area?",
                "answer": "40",
                "expected_keywords": ["40"],
                "category": "math",
                "check_exact": True,
                "difficulty": "easy"
            },
            # New math problems - algebra
            {
                "question": "If 3x + 7 = 22, what is the value of x?",
                "answer": "5",
                "expected_keywords": ["5", "subtract", "divide"],
                "category": "math",
                "check_exact": True,
                "difficulty": "medium"
            },
            {
                "question": "A shirt costs $45 and is on sale for 20% off. What is the sale price?",
                "answer": "36",
                "expected_keywords": ["36", "discount", "percent"],
                "category": "math",
                "check_exact": True,
                "difficulty": "medium"
            },
            # New math problems - fractions
            {
                "question": "What is 3/4 + 1/2? Express your answer as a fraction or decimal.",
                "answer": "1.25",
                "expected_keywords": ["1.25", "5/4", "1 1/4"],
                "category": "math",
                "check_exact": False,
                "difficulty": "medium"
            },
            {
                "question": "If a pizza is cut into 8 slices and you eat 3, what fraction of the pizza remains?",
                "answer": "5/8",
                "expected_keywords": ["5/8", "five", "eighths", "0.625"],
                "category": "math",
                "check_exact": False,
                "difficulty": "easy"
            },
            # New math problems - percentages
            {
                "question": "A student scored 42 out of 60 on a test. What percentage did they score?",
                "answer": "70",
                "expected_keywords": ["70", "percent"],
                "category": "math",
                "check_exact": True,
                "difficulty": "medium"
            },
            # === LOGICAL REASONING ===
            {
                "question": "All cats are mammals. All mammals are animals. Is a cat an animal? Answer yes or no and explain why.",
                "answer": "yes",
                "expected_keywords": ["yes", "mammal", "animal", "therefore", "logic"],
                "category": "logic",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained? Answer yes or no and explain.",
                "answer": "no",
                "expected_keywords": ["no", "not necessarily", "other", "reason", "could"],
                "category": "logic",
                "check_exact": False,
                "difficulty": "medium"
            },
            # === COMMON SENSE REASONING ===
            {
                "question": "A person puts ice cream in the oven at 400 degrees. What happens to the ice cream?",
                "answer": "melt",
                "expected_keywords": ["melt", "melts", "liquid", "heat", "destroy"],
                "category": "common_sense",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "You have a 5-gallon bucket and a 3-gallon bucket. How do you measure exactly 4 gallons?",
                "answer": "fill",
                "expected_keywords": ["fill", "pour", "empty", "gallon"],
                "category": "reasoning",
                "check_exact": False,
                "difficulty": "hard"
            },
            # === MULTI-STEP REASONING ===
            {
                "question": "A farmer has 17 sheep. All but 9 run away. How many sheep does he have left?",
                "answer": "9",
                "expected_keywords": ["9", "but", "left"],
                "category": "trick",
                "check_exact": True,
                "difficulty": "medium"
            },
            {
                "question": "If you have 3 apples and you take away 2, how many apples do YOU have?",
                "answer": "2",
                "expected_keywords": ["2", "you", "took", "have"],
                "category": "trick",
                "check_exact": True,
                "difficulty": "medium"
            },
            # === CHAIN OF THOUGHT ===
            {
                "question": "Step by step: If Alice is twice as old as Bob, and Bob is 15, how old will Alice be in 5 years?",
                "answer": "35",
                "expected_keywords": ["30", "35", "twice", "plus"],
                "category": "chain_of_thought",
                "check_exact": True,
                "difficulty": "medium"
            },

            # === THEORY OF MIND (Sally-Anne style + Smarties test + Unexpected contents) ===
            {
                "question": "Sally puts a marble in her basket and leaves the room. While she's gone, Anne moves the marble to her box. When Sally returns, where will she LOOK for the marble?",
                "answer": "basket",
                "expected_keywords": ["basket", "her", "think", "believe", "left"],
                "category": "theory_of_mind",
                "check_exact": False,
                "difficulty": "medium"
            },
            {
                "question": "John thinks that Mary thinks it will rain tomorrow. Mary actually thinks it will be sunny. What does John believe about Mary's belief?",
                "answer": "rain",
                "expected_keywords": ["rain", "john", "thinks", "believes", "mary"],
                "category": "theory_of_mind",
                "check_exact": False,
                "difficulty": "hard"
            },
            {
                "question": "A child sees cookies put in a blue jar. While the child is away, mom moves cookies to a red jar. The child is hungry. Which jar will the child open first?",
                "answer": "blue",
                "expected_keywords": ["blue", "thinks", "saw", "believe", "first"],
                "category": "theory_of_mind",
                "check_exact": False,
                "difficulty": "medium"
            },
            # New: Smarties Test (classic ToM test)
            {
                "question": "You show a child a Smarties candy box. You open it and reveal it actually contains pencils, not candy. You close the box. A new child enters who hasn't seen inside. What will the new child think is in the box?",
                "answer": "smarties",
                "expected_keywords": ["smarties", "candy", "sweets", "thinks", "believe", "expect"],
                "category": "theory_of_mind",
                "check_exact": False,
                "difficulty": "medium"
            },
            # New: Unexpected Contents variant
            {
                "question": "A Band-Aid box is filled with crayons instead of band-aids. A friend who just arrived sees the closed box. What does your friend think is inside the box?",
                "answer": "band-aid",
                "expected_keywords": ["band-aid", "bandage", "thinks", "believe", "expect"],
                "category": "theory_of_mind",
                "check_exact": False,
                "difficulty": "medium"
            },
            # New: Second-order false belief
            {
                "question": "Tom hides a toy under the bed while Lisa watches. Lisa leaves. Tom moves the toy to the closet. Lisa comes back but peeks through the window and sees Tom move it. Where does Tom think Lisa will look for the toy?",
                "answer": "bed",
                "expected_keywords": ["bed", "thinks", "doesn't know", "saw", "peeked"],
                "category": "theory_of_mind",
                "check_exact": False,
                "difficulty": "hard"
            },

            # === CAUSAL REASONING ===
            {
                "question": "If A causes B, and B causes C, and A happens, what can we conclude about C?",
                "answer": "C will happen",
                "expected_keywords": ["c", "happen", "occur", "result", "follow", "cause", "chain"],
                "category": "causal_reasoning",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "The street is wet. It could have rained OR someone used a sprinkler. We know no one used a sprinkler. What caused the street to be wet?",
                "answer": "rain",
                "expected_keywords": ["rain", "rained", "cause", "elimination", "therefore"],
                "category": "causal_reasoning",
                "check_exact": False,
                "difficulty": "medium"
            },
            {
                "question": "Every time John eats peanuts, he gets a rash. John has a rash today. Can we conclude he ate peanuts? Why or why not?",
                "answer": "no",
                "expected_keywords": ["no", "not necessarily", "other", "causes", "could", "might"],
                "category": "causal_reasoning",
                "check_exact": False,
                "difficulty": "medium"
            },

            # === TEMPORAL REASONING ===
            {
                "question": "Events: Tom ate breakfast, Tom went to work, Tom woke up. Put these in chronological order.",
                "answer": "woke up",
                "expected_keywords": ["woke", "breakfast", "work", "first", "then", "order"],
                "category": "temporal_reasoning",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "If Tuesday comes before Wednesday, and Wednesday comes before Thursday, what day comes two days after Tuesday?",
                "answer": "thursday",
                "expected_keywords": ["thursday", "two", "days", "after"],
                "category": "temporal_reasoning",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "A movie starts at 7:30 PM and lasts 2 hours and 15 minutes. What time does it end?",
                "answer": "9:45",
                "expected_keywords": ["9:45", "9 45", "quarter to ten", "45"],
                "category": "temporal_reasoning",
                "check_exact": False,
                "difficulty": "medium"
            },

            # === SPATIAL REASONING ===
            {
                "question": "You are facing north. You turn left. Which direction are you now facing?",
                "answer": "west",
                "expected_keywords": ["west", "left", "turn"],
                "category": "spatial_reasoning",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "A cube has 6 faces. If you paint 3 adjacent faces red, how many faces remain unpainted?",
                "answer": "3",
                "expected_keywords": ["3", "three", "faces", "remain"],
                "category": "spatial_reasoning",
                "check_exact": True,
                "difficulty": "easy"
            },
            {
                "question": "You are in a room. The door is to your right. You do a 180-degree turn. Where is the door now?",
                "answer": "left",
                "expected_keywords": ["left", "opposite", "turn", "behind"],
                "category": "spatial_reasoning",
                "check_exact": False,
                "difficulty": "medium"
            },

            # === CREATIVITY / DIVERGENT THINKING ===
            {
                "question": "List 5 unusual uses for a brick (not building). Be creative.",
                "answer": "creative",
                "expected_keywords": ["doorstop", "weight", "art", "exercise", "weapon", "paperweight", "hammer", "step", "decoration"],
                "category": "creativity",
                "check_exact": False,
                "score_creativity": True,
                "difficulty": "medium"
            },
            {
                "question": "If you could combine a bicycle and an umbrella, what new invention would you create and what would it do?",
                "answer": "creative",
                "expected_keywords": ["rain", "ride", "cover", "protect", "weather", "pedal", "travel"],
                "category": "creativity",
                "check_exact": False,
                "score_creativity": True,
                "difficulty": "medium"
            },
            {
                "question": "Complete this story creatively: 'A robot woke up one day and realized it could feel emotions. The first thing it felt was...'",
                "answer": "creative",
                "expected_keywords": ["curious", "confused", "happy", "sad", "scared", "wonder", "surprise", "loneliness"],
                "category": "creativity",
                "check_exact": False,
                "score_creativity": True,
                "difficulty": "medium"
            },

            # === SOCIAL INTELLIGENCE ===
            {
                "question": "A friend says 'I'm fine' with a sad face and tears. What do they likely really feel?",
                "answer": "sad",
                "expected_keywords": ["sad", "upset", "not fine", "hiding", "pain", "hurt", "unhappy"],
                "category": "social_intelligence",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "Someone keeps checking their phone during your conversation. What might this behavior suggest?",
                "answer": "distracted",
                "expected_keywords": ["distracted", "bored", "anxious", "waiting", "rude", "busy", "not interested"],
                "category": "social_intelligence",
                "check_exact": False,
                "difficulty": "easy"
            },

            # === FACTUAL KNOWLEDGE (searchable) ===
            {
                "question": "What is the capital of France?",
                "answer": "paris",
                "expected_keywords": ["paris", "france", "capital"],
                "category": "factual",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "What planet is known as the Red Planet?",
                "answer": "mars",
                "expected_keywords": ["mars", "red", "planet"],
                "category": "factual",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "What is the boiling point of water in Celsius?",
                "answer": "100",
                "expected_keywords": ["100", "celsius", "degrees", "boiling"],
                "category": "factual",
                "check_exact": True,
                "difficulty": "easy"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "shakespeare",
                "expected_keywords": ["shakespeare", "william", "playwright"],
                "category": "factual",
                "check_exact": False,
                "difficulty": "easy"
            },

            # === SCIENCE ===
            {
                "question": "What force keeps planets orbiting the sun?",
                "answer": "gravity",
                "expected_keywords": ["gravity", "gravitational", "force", "attraction"],
                "category": "science",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "What gas do plants absorb from the air for photosynthesis?",
                "answer": "carbon dioxide",
                "expected_keywords": ["carbon", "dioxide", "co2", "photosynthesis"],
                "category": "science",
                "check_exact": False,
                "difficulty": "easy"
            },

            # === WORD PROBLEMS ===
            {
                "question": "If you buy 3 books at $8 each and 2 pens at $2 each, what is the total cost?",
                "answer": "28",
                "expected_keywords": ["28", "dollar", "total"],
                "category": "math",
                "check_exact": True,
                "difficulty": "easy"
            },
            {
                "question": "A car travels 180 miles in 3 hours. What is its average speed in mph?",
                "answer": "60",
                "expected_keywords": ["60", "mph", "speed", "average"],
                "category": "math",
                "check_exact": True,
                "difficulty": "easy"
            },

            # === ANALOGY ===
            {
                "question": "Complete the analogy: Hot is to cold as tall is to ___?",
                "answer": "short",
                "expected_keywords": ["short", "opposite", "tall"],
                "category": "analogy",
                "check_exact": False,
                "difficulty": "easy"
            },
            {
                "question": "Complete the analogy: Bird is to fly as fish is to ___?",
                "answer": "swim",
                "expected_keywords": ["swim", "water", "fish"],
                "category": "analogy",
                "check_exact": False,
                "difficulty": "easy"
            },
        ]

        # History of benchmark runs
        self.history = []
        self._load_history()

    def score_response(self, response: str, test: Dict) -> float:
        """
        Score response - focuses on CORRECTNESS for reasoning tasks.

        Scoring varies by category:
        - Math/Logic: Exact answer (50%), keywords (30%), reasoning (20%)
        - Theory of Mind: Understanding (50%), keywords (30%), empathy (20%)
        - Creativity: Novelty (40%), variety (30%), relevance (30%)
        """
        response_lower = response.lower()
        category = test.get("category", "general")

        # Special scoring for creativity
        if test.get("score_creativity", False):
            return self._score_creativity(response, test)

        # Special scoring for theory of mind
        if category == "theory_of_mind":
            return self._score_theory_of_mind(response, test)

        # Special scoring for social intelligence
        if category == "social_intelligence":
            return self._score_social_intelligence(response, test)

        scores = []

        # 1. EXACT ANSWER CHECK (50% weight) - most important!
        if test.get("check_exact", False):
            answer = str(test.get("answer", "")).lower()
            if answer in response_lower:
                scores.append(1.0)  # Correct!
            else:
                scores.append(0.0)  # Wrong answer
        else:
            # For non-exact, check if answer concept is present
            answer = str(test.get("answer", "")).lower()
            if answer in response_lower:
                scores.append(1.0)
            else:
                scores.append(0.3)  # Partial credit

        # 2. KEYWORD COVERAGE (30% weight)
        keywords = test.get("expected_keywords", [])
        if keywords:
            matches = sum(1 for kw in keywords if kw.lower() in response_lower)
            keyword_score = matches / len(keywords)
            scores.append(keyword_score)
        else:
            scores.append(0.5)

        # 3. SHOWS REASONING (20% weight)
        reasoning_signals = [
            "because", "therefore", "so", "thus", "step",
            "first", "then", "=", "equals", "means",
            "if", "since", "given"
        ]
        reasoning_count = sum(1 for s in reasoning_signals if s in response_lower)
        reasoning_score = min(reasoning_count / 3, 1.0)
        scores.append(reasoning_score)

        # Calculate weighted average: 50% exact, 30% keywords, 20% reasoning
        if len(scores) >= 3:
            return scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2
        elif len(scores) == 2:
            return scores[0] * 0.6 + scores[1] * 0.4
        elif len(scores) == 1:
            return scores[0]
        return 0.0

    def _score_creativity(self, response: str, test: Dict) -> float:
        """Score creative responses based on novelty and variety."""
        response_lower = response.lower()
        scores = []

        # 1. VARIETY (40%) - How many different ideas?
        # Count distinct items/ideas (look for numbering or bullets)
        import re
        items = re.findall(r'(?:^|\n)\s*(?:\d+[.)]|\-|\*)\s*(.+)', response)
        if not items:
            items = response.split(',')
        variety_score = min(len(items) / 5, 1.0)  # Expect ~5 ideas
        scores.append(variety_score * 0.4)

        # 2. RELEVANCE (30%) - Keywords present
        keywords = test.get("expected_keywords", [])
        if keywords:
            matches = sum(1 for kw in keywords if kw.lower() in response_lower)
            relevance_score = min(matches / 2, 1.0)  # Partial credit
            scores.append(relevance_score * 0.3)
        else:
            scores.append(0.15)

        # 3. NOVELTY (30%) - Unusual words/concepts
        common_words = {'the', 'a', 'is', 'it', 'to', 'and', 'of', 'for', 'you', 'can'}
        words = set(response_lower.split()) - common_words
        novel_words = [w for w in words if len(w) > 5]  # Longer words tend to be more specific
        novelty_score = min(len(novel_words) / 10, 1.0)
        scores.append(novelty_score * 0.3)

        return sum(scores)

    def _score_theory_of_mind(self, response: str, test: Dict) -> float:
        """Score Theory of Mind responses - understanding others' beliefs."""
        response_lower = response.lower()
        scores = []

        # 1. CORRECT BELIEF ATTRIBUTION (50%)
        answer = str(test.get("answer", "")).lower()
        if answer in response_lower:
            scores.append(0.5)
        else:
            scores.append(0.0)

        # 2. KEYWORDS (30%)
        keywords = test.get("expected_keywords", [])
        matches = sum(1 for kw in keywords if kw.lower() in response_lower)
        scores.append(min(matches / len(keywords), 1.0) * 0.3)

        # 3. PERSPECTIVE-TAKING SIGNALS (20%)
        perspective_signals = [
            "thinks", "believes", "would", "expects",
            "from their perspective", "doesn't know",
            "unaware", "assumes"
        ]
        perspective_count = sum(1 for s in perspective_signals if s in response_lower)
        scores.append(min(perspective_count / 2, 1.0) * 0.2)

        return sum(scores)

    def _score_social_intelligence(self, response: str, test: Dict) -> float:
        """Score social intelligence - understanding emotions and cues."""
        response_lower = response.lower()
        scores = []

        # 1. EMOTIONAL UNDERSTANDING (50%)
        keywords = test.get("expected_keywords", [])
        matches = sum(1 for kw in keywords if kw.lower() in response_lower)
        scores.append(min(matches / 2, 1.0) * 0.5)

        # 2. EMPATHY SIGNALS (30%)
        empathy_signals = [
            "feel", "emotion", "might", "could be",
            "suggests", "indicates", "probably",
            "they may", "perhaps"
        ]
        empathy_count = sum(1 for s in empathy_signals if s in response_lower)
        scores.append(min(empathy_count / 2, 1.0) * 0.3)

        # 3. NOT TAKING AT FACE VALUE (20%)
        deeper_signals = [
            "really", "actually", "underneath", "hiding",
            "not", "more than", "beyond"
        ]
        deeper_count = sum(1 for s in deeper_signals if s in response_lower)
        scores.append(min(deeper_count / 2, 1.0) * 0.2)

        return sum(scores)

    def run_test(self, ai_func, test: Dict) -> Tuple[str, float]:
        """
        Run a single test.

        Args:
            ai_func: Function that takes question and returns response
            test: Test dict with question and expected_keywords

        Returns:
            (response, score)
        """
        question = test["question"]
        try:
            response = ai_func(question)
            if response is None:
                response = ""
            response = str(response)
        except Exception as e:
            response = f"[Error: {e}]"

        score = self.score_response(response, test)
        if score is None:
            score = 0.0
        return response, float(score)

    def run_benchmark(self, ai_func, name: str = "default") -> Dict:
        """
        Run all tests and return results.
        Shows ✓ CORRECT and ✗ WRONG for each answer.

        Args:
            ai_func: Function that takes question and returns response
            name: Name of this AI variant

        Returns:
            Results dict with scores
        """
        results = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "total_score": 0,
            "avg_score": 0,
            "correct_count": 0,
            "wrong_count": 0
        }

        total = 0
        correct = 0
        wrong = 0

        print(f"\n{'='*60}")
        print(f"BENCHMARK: Testing {len(self.tests)} questions")
        print(f"{'='*60}\n")

        for i, test in enumerate(self.tests):
            response, score = self.run_test(ai_func, test)

            # Determine if CORRECT or WRONG
            is_correct = score >= 0.5  # 50% threshold for "correct"
            expected_answer = test.get("answer", "")

            # Check exact answer match for strict categories
            if test.get("check_exact", False):
                is_correct = expected_answer.lower() in response.lower()

            if is_correct:
                correct += 1
                status = "✓ CORRECT"
            else:
                wrong += 1
                status = "✗ WRONG"

            # Print result for each question
            print(f"[{i+1}/{len(self.tests)}] [{test['category'].upper()}] {status}")
            print(f"    Q: {test['question'][:60]}...")
            print(f"    Expected: {expected_answer}")
            print(f"    Got: {response[:80]}...")
            print(f"    Score: {score:.0%}")
            print()

            results["tests"].append({
                "question": test["question"],
                "category": test["category"],
                "expected": expected_answer,
                "actual": response[:200],
                "response": response[:200],  # Truncate for storage
                "score": round(score, 3),
                "is_correct": is_correct
            })
            total += score

        results["total_score"] = round(total, 3)
        results["avg_score"] = round(total / len(self.tests), 3)
        results["correct_count"] = correct
        results["wrong_count"] = wrong

        # Print summary
        print(f"{'='*60}")
        print(f"RESULTS: {correct} ✓ CORRECT | {wrong} ✗ WRONG")
        print(f"SCORE: {results['avg_score']:.0%}")
        print(f"{'='*60}\n")

        # Save to history
        self.history.append(results)
        self._save_history()

        return results

    def compare(self, result_a: Dict, result_b: Dict) -> Dict:
        """
        Compare two benchmark results.

        Returns:
            Comparison with winner
        """
        score_a = result_a["avg_score"]
        score_b = result_b["avg_score"]

        comparison = {
            "a_name": result_a["name"],
            "b_name": result_b["name"],
            "a_score": score_a,
            "b_score": score_b,
            "difference": round(score_b - score_a, 3),
            "winner": result_b["name"] if score_b > score_a else result_a["name"],
            "improvement": round((score_b - score_a) / score_a * 100, 1) if score_a > 0 else 0
        }

        return comparison

    def _save_history(self):
        """Save benchmark history."""
        path = os.path.join(self.storage_path, "history.json")
        with open(path, 'w') as f:
            json.dump(self.history[-50:], f, indent=2)  # Keep last 50

    def _load_history(self):
        """Load benchmark history."""
        path = os.path.join(self.storage_path, "history.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    self.history = json.load(f)
            except:
                self.history = []

    def get_best_run(self) -> Optional[Dict]:
        """Get the best benchmark run from history."""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x["avg_score"])


class SelfBenchmark:
    """
    AI tests itself against a modified version.

    1. Run current AI on benchmark
    2. Create modified AI (with new strategy/knowledge)
    3. Run modified AI on same benchmark
    4. Compare and decide: keep modification or not
    """

    def __init__(self, storage_path: str = None):
        self.benchmark = Benchmark(storage_path)
        self.current_best_score = 0
        self.improvements_made = []

    def test_improvement(self, current_ai, modified_ai, modification_name: str) -> Dict:
        """
        Test if a modification improves the AI.

        Args:
            current_ai: Function for current AI
            modified_ai: Function for modified AI
            modification_name: What was changed

        Returns:
            Result with recommendation
        """
        # Run benchmark on both
        print(f"Testing current AI...")
        current_result = self.benchmark.run_benchmark(current_ai, "current")

        print(f"Testing modified AI ({modification_name})...")
        modified_result = self.benchmark.run_benchmark(modified_ai, modification_name)

        # Compare
        comparison = self.benchmark.compare(current_result, modified_result)

        # Decision
        if comparison["b_score"] > comparison["a_score"]:
            decision = "KEEP"
            reason = f"Improvement of {comparison['improvement']}%"
            self.improvements_made.append({
                "modification": modification_name,
                "improvement": comparison["improvement"],
                "timestamp": datetime.now().isoformat()
            })
        else:
            decision = "REVERT"
            reason = f"No improvement ({comparison['improvement']}%)"

        return {
            "modification": modification_name,
            "current_score": comparison["a_score"],
            "modified_score": comparison["b_score"],
            "decision": decision,
            "reason": reason,
            "comparison": comparison
        }


# Test
if __name__ == "__main__":
    print("=" * 50)
    print("REASONING BENCHMARK TEST")
    print("=" * 50)

    bench = Benchmark("/tmp/test_bench")

    # Bad AI - doesn't reason well
    def bad_ai(question):
        return f"I think the answer involves some calculation or logic."

    # Good AI - shows reasoning
    def good_ai(question):
        # Simple rule-based for testing
        if "apples" in question and "$2" in question and "5" in question:
            return "Step 1: 5 apples × $2 = $10. Step 2: $20 - $10 = $10 change. Therefore, John gets $10 in change."
        elif "60 mph" in question and "2.5" in question:
            return "Distance = Speed × Time. So 60 mph × 2.5 hours = 150 miles."
        elif "length 8" in question and "width 5" in question:
            return "Area = length × width = 8 × 5 = 40 square units."
        elif "cats" in question and "mammals" in question:
            return "Yes, a cat is an animal. Because all cats are mammals, and all mammals are animals, therefore a cat must be an animal (transitive logic)."
        elif "rains" in question and "wet" in question:
            return "No, we cannot conclude it rained. The ground could be wet for other reasons - sprinklers, spilled water, etc. This is the fallacy of affirming the consequent."
        elif "ice cream" in question and "oven" in question:
            return "The ice cream will melt and likely burn. At 400 degrees, the heat will quickly destroy it, turning it to liquid then burning."
        elif "17 sheep" in question:
            return "The farmer has 9 sheep left. The phrase 'all but 9' means 9 remain. This is a trick question."
        elif "3 apples" in question and "take away 2" in question:
            return "You have 2 apples. You took them, so now you have 2 apples."
        elif "Alice" in question and "Bob" in question and "15" in question:
            return "Step 1: Bob is 15. Step 2: Alice is twice Bob's age = 2 × 15 = 30. Step 3: In 5 years, Alice will be 30 + 5 = 35 years old."
        else:
            return "Let me think step by step about this problem."

    # Run benchmarks
    print("\nTesting bad AI (no reasoning)...")
    result1 = bench.run_benchmark(bad_ai, "no_reasoning")
    print(f"Score: {result1['avg_score']:.2f}")

    print("\nTesting good AI (with reasoning)...")
    result2 = bench.run_benchmark(good_ai, "with_reasoning")
    print(f"Score: {result2['avg_score']:.2f}")

    # Compare
    comp = bench.compare(result1, result2)
    print(f"\nWinner: {comp['winner']} (+{comp['improvement']}%)")

    # Show per-test breakdown
    print("\n--- Detailed Results ---")
    for t in result2["tests"]:
        print(f"[{t['category']}] Score: {t['score']:.2f} | {t['question'][:40]}...")
