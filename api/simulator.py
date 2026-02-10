"""
Cognitive Simulation Engine

Wraps the core CognitiveAgent to run scenarios through the 5-phase
architecture, tracking module activations and returning structured results.
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("simulator")

MEMORY_CAPACITY_LIMIT = 100

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.cognitive_agent import CognitiveAgent, CognitiveConfig
    AGENT_AVAILABLE = True
except Exception:
    AGENT_AVAILABLE = False


PHASES = [
    {
        "id": 1,
        "name": "Foundation",
        "description": "Predictive coding, memory systems, and perception. Implements hierarchical prediction error minimization following the free-energy principle, multi-store memory (episodic, semantic, procedural, working), and multi-modal sensory processing.",
        "modules": ["prediction", "memory", "learning"],
        "theories": ["Free Energy Principle (Friston 2010)", "Predictive Coding (Rao & Ballard 1999)", "Memory Consolidation (Tulving 1985)"],
    },
    {
        "id": 2,
        "name": "Processing",
        "description": "Dual-process thinking (System 1/2), executive control, and reasoning. Dynamic switching between fast intuitive and slow analytical processing based on task demands.",
        "modules": ["dual_process", "executive", "reasoning"],
        "theories": ["Dual-Process Theory (Kahneman 2011)", "Executive Function (Miyake 2000)", "Mental Models (Johnson-Laird 1983)"],
    },
    {
        "id": 3,
        "name": "Motivation",
        "description": "Drive systems, emotion processing with somatic markers, and recursive self-awareness. Curiosity-driven learning and metacognitive monitoring.",
        "modules": ["motivation", "emotion", "self_awareness"],
        "theories": ["Somatic Marker Hypothesis (Damasio 1994)", "Intrinsic Motivation (Berlyne 1960)", "Metacognition (Flavell 1979)"],
    },
    {
        "id": 4,
        "name": "Interface",
        "description": "Language processing, social cognition with theory of mind, and embodied cognition. Sensorimotor grounding and natural language generation.",
        "modules": ["language", "social", "embodied"],
        "theories": ["Embodied Cognition (Barsalou 1999)", "Theory of Mind (Premack 1978)", "Usage-Based Language (Tomasello 2003)"],
    },
    {
        "id": 5,
        "name": "Advanced",
        "description": "Creativity through conceptual blending, cognitive maps for spatial reasoning, time perception, and sleep-based memory consolidation.",
        "modules": ["creativity", "cognitive_maps", "time", "sleep"],
        "theories": ["Conceptual Blending (Fauconnier 2002)", "Cognitive Maps (Tolman 1948)", "Sleep Consolidation (Tononi 2006)"],
    },
]

MODULE_DEFINITIONS = {
    "prediction": {
        "name": "Predictive Coding",
        "phase": 1,
        "description": "Hierarchical prediction error minimization following Friston's free-energy principle.",
        "parameters": {"input_dim": 64, "num_levels": 3, "learning_rate": 0.01},
    },
    "memory": {
        "name": "Memory Systems",
        "phase": 1,
        "description": "Four-store memory: episodic, semantic, procedural, and working memory.",
        "parameters": {"embedding_dim": 64, "capacity": 1000, "decay_rate": 0.5},
    },
    "learning": {
        "name": "Learning System",
        "phase": 1,
        "description": "Hebbian learning, reinforcement learning, and synaptic plasticity.",
        "parameters": {"learning_rate": 0.01, "momentum": 0.9},
    },
    "dual_process": {
        "name": "Dual-Process Thinking",
        "phase": 2,
        "description": "System 1 (fast/intuitive) and System 2 (slow/deliberate) with dynamic switching.",
        "parameters": {"embedding_dim": 64, "switch_threshold": 0.5},
    },
    "executive": {
        "name": "Executive Control",
        "phase": 2,
        "description": "Task switching, inhibitory control, and working memory management.",
        "parameters": {"inhibition_threshold": 0.7, "switching_cost": 0.15},
    },
    "reasoning": {
        "name": "Reasoning Router",
        "phase": 2,
        "description": "Deductive, inductive, and abductive reasoning engines.",
        "parameters": {"embedding_dim": 64, "confidence_threshold": 0.6},
    },
    "motivation": {
        "name": "Motivation Engine",
        "phase": 3,
        "description": "Drive system integrating homeostatic needs, incentive salience, and intrinsic motivation.",
        "parameters": {"dim": 64, "curiosity_weight": 0.3, "reward_discount": 0.95},
    },
    "emotion": {
        "name": "Emotion System",
        "phase": 3,
        "description": "Dimensional emotion modeling (valence/arousal/dominance) with somatic markers.",
        "parameters": {"dim": 64, "reactivity": 0.6, "regulation": 0.5},
    },
    "self_awareness": {
        "name": "Self-Awareness",
        "phase": 3,
        "description": "Recursive self-modeling, metacognitive monitoring, confidence calibration.",
        "parameters": {"dim": 64, "reflection_depth": 3},
    },
    "language": {
        "name": "Language Processing",
        "phase": 4,
        "description": "Semantic parsing, pragmatic reasoning, and natural language generation.",
        "parameters": {"dim": 64, "vocab_size": 50000},
    },
    "social": {
        "name": "Social Cognition",
        "phase": 4,
        "description": "Theory of mind, empathy modeling, reputation tracking.",
        "parameters": {"dim": 64, "trust_decay": 0.01},
    },
    "embodied": {
        "name": "Embodied Cognition",
        "phase": 4,
        "description": "Sensorimotor grounding, body schema, environmental coupling.",
        "parameters": {"dim": 64, "motor_noise": 0.05},
    },
    "creativity": {
        "name": "Creativity Module",
        "phase": 5,
        "description": "Conceptual blending, analogical reasoning, divergent thought generation.",
        "parameters": {"dim": 64, "novelty_threshold": 0.4},
    },
    "cognitive_maps": {
        "name": "Cognitive Maps",
        "phase": 5,
        "description": "Spatial reasoning, mental navigation, and experience replay.",
        "parameters": {"dim": 64, "map_size": 100},
    },
    "time": {
        "name": "Time Perception",
        "phase": 5,
        "description": "Internal clock, duration estimation, temporal ordering.",
        "parameters": {"dim": 64, "clock_speed": 1.0},
    },
    "sleep": {
        "name": "Sleep Consolidation",
        "phase": 5,
        "description": "Offline memory replay, synaptic homeostasis, dream generation.",
        "parameters": {"dim": 64, "cycle_duration": 90},
    },
}

SCENARIOS = {
    "sentence_processing": {
        "name": "Process a Sentence",
        "description": "Process 'The cat sat on the mat' through sensory input to full comprehension.",
        "input_text": "The cat sat on the mat",
    },
    "memory_recall": {
        "name": "Recall a Childhood Memory",
        "description": "Episodic memory retrieval with emotional coloring and verbal report.",
        "input_text": "Remember a childhood birthday party",
    },
    "financial_decision": {
        "name": "Financial Decision",
        "description": "System 1 vs System 2 conflict in evaluating an investment opportunity.",
        "input_text": "18% return, high volatility, 3-year lock-in investment",
    },
    "word_learning": {
        "name": "Learn a New Word",
        "description": "Acquire the word 'petrichor' through language parsing and memory encoding.",
        "input_text": "Petrichor means the smell of rain on dry earth",
    },
    "danger_response": {
        "name": "Respond to Danger",
        "description": "Amygdala fast-path threat response bypassing cortical processing.",
        "input_text": "Large dark shape approaching rapidly",
    },
}


def _serialize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    try:
        return str(obj)
    except Exception:
        return None


class CognitiveSimulator:
    """Runs cognitive scenarios through the real agent framework."""

    def __init__(self):
        self._agent = None
        self._initialized = False

    def _ensure_agent(self):
        if self._initialized:
            self._enforce_memory_limit()
            return
        if AGENT_AVAILABLE:
            try:
                config = CognitiveConfig(dim=64, fast_mode=True)
                self._agent = CognitiveAgent(config)
                self._agent._ensure_modules()
                self._initialized = True
            except Exception as e:
                logger.error("Agent init failed: %s", e)
                self._agent = None
                self._initialized = True
        else:
            self._initialized = True

    def _enforce_memory_limit(self):
        if self._agent is None:
            return
        if not hasattr(self._agent, '_modules') or 'memory' not in self._agent._modules:
            return
        memory = self._agent._modules['memory']
        for store_name in ('episodic', 'semantic', 'working'):
            store = getattr(memory, store_name, None)
            if store is None:
                continue
            entries = getattr(store, 'memories', None) or getattr(store, 'items', None) or getattr(store, 'buffer', None)
            if entries is not None and hasattr(entries, '__len__') and len(entries) > MEMORY_CAPACITY_LIMIT:
                logger.warning(
                    "Memory store '%s' exceeded limit (%d > %d), clearing oldest entries",
                    store_name, len(entries), MEMORY_CAPACITY_LIMIT
                )
                if isinstance(entries, list):
                    del entries[:len(entries) - MEMORY_CAPACITY_LIMIT]
                elif isinstance(entries, dict):
                    keys_to_remove = list(entries.keys())[:len(entries) - MEMORY_CAPACITY_LIMIT]
                    for k in keys_to_remove:
                        del entries[k]

    def get_modules(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": mod_id,
                "name": info["name"],
                "phase": info["phase"],
                "description": info["description"],
                "parameters": info["parameters"],
                "active": self._agent is not None and mod_id in (self._agent._modules if self._agent else {}),
            }
            for mod_id, info in MODULE_DEFINITIONS.items()
        ]

    def get_phases(self) -> List[Dict[str, Any]]:
        self._ensure_agent()
        result = []
        for phase in PHASES:
            active_modules = []
            if self._agent:
                active_modules = [m for m in phase["modules"] if m in self._agent._modules]
            result.append({
                "id": phase["id"],
                "name": phase["name"],
                "description": phase["description"],
                "modules": phase["modules"],
                "theories": phase["theories"],
                "active_module_count": len(active_modules),
                "total_module_count": len(phase["modules"]),
                "status": "active" if active_modules else "available",
            })
        return result

    def simulate(self, scenario_key: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        self._ensure_agent()
        parameters = parameters or {}

        if scenario_key not in SCENARIOS:
            return {"error": f"Unknown scenario: {scenario_key}", "available": list(SCENARIOS.keys())}

        scenario = SCENARIOS[scenario_key]
        dim = parameters.get("dim", 64)
        if not isinstance(dim, int) or dim < 1 or dim > 512:
            return {"error": "dim must be an integer between 1 and 512"}
        steps = []
        module_activations = {}
        start_time = time.time()

        if self._agent:
            try:
                return self._run_real_simulation(scenario_key, scenario, dim, parameters)
            except Exception as e:
                logger.warning("Real simulation failed, using fallback: %s", e)

        return self._run_fallback_simulation(scenario_key, scenario, dim, parameters)

    def _run_real_simulation(self, scenario_key: str, scenario: Dict, dim: int, parameters: Dict) -> Dict[str, Any]:
        steps = []
        module_activations = {}
        start_time = time.time()

        observation = np.random.randn(dim)
        np.random.seed(hash(scenario["input_text"]) % (2**32))
        observation = np.random.randn(dim)

        # Phase 1: Perception
        t0 = time.perf_counter()
        perceive_result = self._agent.perceive(observation)
        perceive_time = (time.perf_counter() - t0) * 1000

        steps.append({
            "phase": 1,
            "phase_name": "Foundation",
            "module": "prediction",
            "module_name": "Predictive Coding",
            "action": "Process sensory input",
            "input": scenario["input_text"],
            "output": f"Prediction error: {perceive_result.get('prediction', {}).get('weighted_error', 0.0):.3f}",
            "duration_ms": round(perceive_time, 2),
            "details": _serialize(perceive_result.get("prediction", {})),
        })
        module_activations["prediction"] = {
            "activation": float(perceive_result.get("prediction", {}).get("weighted_error", 0.5)),
            "phase": 1,
        }

        if perceive_result.get("emotion"):
            module_activations["emotion"] = {
                "activation": float(perceive_result["emotion"].get("valence", 0.5)),
                "phase": 3,
            }

        module_activations["motivation"] = {
            "activation": 0.6,
            "phase": 3,
        }

        # Phase 2: Thinking
        t0 = time.perf_counter()
        think_result = self._agent.think(observation)
        think_time = (time.perf_counter() - t0) * 1000

        system_used = think_result.get("dual_process", {}).get("system_used", "unknown")
        confidence = think_result.get("dual_process", {}).get("confidence", 0.0)

        steps.append({
            "phase": 2,
            "phase_name": "Processing",
            "module": "dual_process",
            "module_name": "Dual-Process Thinking",
            "action": f"Route to {system_used}",
            "input": f"Problem analysis for: {scenario['input_text'][:50]}",
            "output": f"System used: {system_used}, confidence: {confidence:.3f}",
            "duration_ms": round(think_time, 2),
            "details": _serialize(think_result.get("dual_process", {})),
        })
        module_activations["dual_process"] = {"activation": confidence, "phase": 2}

        if think_result.get("executive"):
            module_activations["executive"] = {
                "activation": 0.7 if think_result["executive"].get("effort_required") else 0.4,
                "phase": 2,
            }

        steps.append({
            "phase": 2,
            "phase_name": "Processing",
            "module": "executive",
            "module_name": "Executive Control",
            "action": "Coordinate processing",
            "input": f"System: {system_used}, confidence: {confidence:.3f}",
            "output": f"Inhibit: {think_result.get('executive', {}).get('should_inhibit', False)}, Switch: {think_result.get('executive', {}).get('should_switch', False)}",
            "duration_ms": round(think_time * 0.3, 2),
            "details": _serialize(think_result.get("executive", {})),
        })

        if think_result.get("reflection"):
            steps.append({
                "phase": 3,
                "phase_name": "Motivation",
                "module": "self_awareness",
                "module_name": "Self-Awareness",
                "action": "Metacognitive reflection",
                "input": "Monitor cognitive processing",
                "output": f"Recalled {think_result.get('recalled_memories', 0)} memories",
                "duration_ms": round(think_time * 0.2, 2),
                "details": _serialize(think_result.get("reflection", {})),
            })
            module_activations["self_awareness"] = {"activation": 0.6, "phase": 3}

        # Phase 3-4: Decision
        options = [
            ("respond", np.random.randn(dim)),
            ("elaborate", np.random.randn(dim)),
            ("wait", np.random.randn(dim)),
        ]
        t0 = time.perf_counter()
        decide_result = self._agent.decide(options, goal=observation)
        decide_time = (time.perf_counter() - t0) * 1000

        steps.append({
            "phase": 3,
            "phase_name": "Motivation",
            "module": "motivation",
            "module_name": "Motivation Engine",
            "action": "Evaluate action values",
            "input": "Options: respond, elaborate, wait",
            "output": f"Selected: {decide_result.get('final_decision', 'unknown')}",
            "duration_ms": round(decide_time, 2),
            "details": {
                "decision": decide_result.get("final_decision"),
                "expected_value": _serialize(decide_result.get("expected_value")),
            },
        })

        # Phase 4: Language output
        t0 = time.perf_counter()
        utterance = self._agent.speak({"predicate": "describe", "arguments": {"topic": scenario["input_text"][:30]}})
        speak_time = (time.perf_counter() - t0) * 1000

        steps.append({
            "phase": 4,
            "phase_name": "Interface",
            "module": "language",
            "module_name": "Language Processing",
            "action": "Generate response",
            "input": f"Meaning: describe {scenario['input_text'][:30]}",
            "output": utterance or "Generated linguistic output",
            "duration_ms": round(speak_time, 2),
        })
        module_activations["language"] = {"activation": 0.8, "phase": 4}

        total_time = (time.time() - start_time) * 1000

        return {
            "scenario": scenario_key,
            "scenario_name": scenario["name"],
            "input": scenario["input_text"],
            "steps": steps,
            "module_activations": module_activations,
            "total_duration_ms": round(total_time, 2),
            "phases_traversed": sorted(set(s["phase"] for s in steps)),
            "engine": "real",
            "agent_step_count": self._agent.step_count,
        }

    def _run_fallback_simulation(self, scenario_key: str, scenario: Dict, dim: int, parameters: Dict) -> Dict[str, Any]:
        """Deterministic fallback when the real agent is not available."""
        start_time = time.time()
        steps = []
        module_activations = {}

        scenario_flows = {
            "sentence_processing": [
                (1, "prediction", "Predictive Coding", "Process auditory input", "Phoneme sequence extracted", 1.2),
                (4, "language", "Language Processing", "Parse syntax", "Parse tree: [S [NP the cat] [VP sat [PP on [NP the mat]]]]", 2.1),
                (1, "memory", "Memory Systems", "Semantic lookup", "cat=feline, mat=surface, sat=past(sit)", 1.5),
                (2, "executive", "Executive Control", "Integrate meaning", "Agent(cat) Action(sit) Location(mat)", 1.8),
            ],
            "memory_recall": [
                (2, "executive", "Executive Control", "Generate retrieval cue", "Cue: temporal=childhood, valence=positive", 1.5),
                (1, "memory", "Memory Systems", "Episodic search", "Match: birthday party, age 7", 2.3),
                (3, "emotion", "Emotion System", "Emotional coloring", "Nostalgia (valence=0.8, arousal=0.4)", 1.8),
                (3, "self_awareness", "Self-Awareness", "Metacognitive reflection", "Vivid episodic memory confirmed", 1.2),
                (4, "language", "Language Processing", "Verbalize memory", "I remember my 7th birthday...", 1.9),
            ],
            "financial_decision": [
                (1, "prediction", "Predictive Coding", "Process investment data", "18% return, high volatility, 3yr lock", 1.1),
                (2, "dual_process", "Dual-Process Thinking", "System 1 impulse", "Quick: invest now! (excitement bias)", 0.8),
                (2, "dual_process", "Dual-Process Thinking", "System 2 override", "Analyzing risk/reward ratio...", 3.2),
                (3, "emotion", "Emotion System", "Somatic markers", "Anxiety (loss aversion) + excitement (gain prospect)", 1.5),
                (2, "executive", "Executive Control", "Resolve conflict", "Compromise: invest 30% of capital", 1.8),
            ],
            "word_learning": [
                (1, "prediction", "Predictive Coding", "Detect novel phoneme", "Unknown word: petrichor", 1.0),
                (4, "language", "Language Processing", "Parse definition", "petrichor = smell of rain on dry earth", 1.8),
                (1, "memory", "Memory Systems", "Encode to long-term", "Linked to: rain, earth, smell, nature", 2.5),
                (2, "executive", "Executive Control", "Confirm acquisition", "Word acquired, needs consolidation", 1.2),
            ],
            "danger_response": [
                (1, "prediction", "Predictive Coding", "Visual threat detection", "Large shape, high velocity, approaching", 0.5),
                (3, "emotion", "Emotion System", "Amygdala fast-path", "ALARM: fight-or-flight, cortisol surge", 0.3),
                (2, "executive", "Executive Control", "Cortical analysis (slow)", "Object identified: ball, not threat", 2.0),
                (3, "emotion", "Emotion System", "Resolution", "Relief (valence=0.6), arousal declining", 1.0),
            ],
        }

        flow = scenario_flows.get(scenario_key, scenario_flows["sentence_processing"])
        cumulative_ms = 0
        for phase, mod_id, mod_name, action, output, duration in flow:
            cumulative_ms += duration
            steps.append({
                "phase": phase,
                "phase_name": PHASES[phase - 1]["name"],
                "module": mod_id,
                "module_name": mod_name,
                "action": action,
                "input": scenario["input_text"] if len(steps) == 0 else steps[-1]["output"],
                "output": output,
                "duration_ms": round(duration, 2),
            })
            module_activations[mod_id] = {
                "activation": round(0.5 + np.random.random() * 0.4, 3),
                "phase": phase,
            }

        total_time = (time.time() - start_time) * 1000

        return {
            "scenario": scenario_key,
            "scenario_name": scenario["name"],
            "input": scenario["input_text"],
            "steps": steps,
            "module_activations": module_activations,
            "total_duration_ms": round(total_time, 2),
            "phases_traversed": sorted(set(s["phase"] for s in steps)),
            "engine": "fallback",
        }

    def analyze_text(self, text: str, lenses: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze text through cognitive lenses."""
        self._ensure_agent()
        lenses = lenses or ["memory", "emotion", "reasoning", "creativity"]
        results = {}
        dim = 64

        text_hash = hash(text) % (2**32)
        np.random.seed(text_hash)
        text_embedding = np.random.randn(dim)

        if self._agent:
            try:
                if "memory" in lenses:
                    understand = self._agent.understand(text)
                    results["memory"] = {
                        "understanding": _serialize(understand),
                        "associations": "Retrieved from semantic memory",
                    }

                if "emotion" in lenses:
                    perceive = self._agent.perceive(text_embedding)
                    emotion_data = perceive.get("emotion", {})
                    results["emotion"] = {
                        "valence": _serialize(emotion_data.get("valence", 0.0)),
                        "arousal": _serialize(emotion_data.get("arousal", 0.5)),
                        "primary_emotion": _serialize(emotion_data.get("primary_emotion", "neutral")),
                    }

                if "reasoning" in lenses:
                    think = self._agent.think(text_embedding)
                    results["reasoning"] = {
                        "system_used": think.get("dual_process", {}).get("system_used", "unknown"),
                        "confidence": _serialize(think.get("dual_process", {}).get("confidence", 0.0)),
                        "recalled_memories": think.get("recalled_memories", 0),
                    }

                if "creativity" in lenses:
                    imagine = self._agent.imagine(text_embedding)
                    results["creativity"] = _serialize(imagine)

                return {"text": text, "lenses": lenses, "analysis": results, "engine": "real"}
            except Exception as e:
                logger.warning("Analysis failed, using fallback: %s", e)

        word_count = len(text.split())
        results = {}
        if "memory" in lenses:
            results["memory"] = {
                "word_count": word_count,
                "associations": ["language", "meaning", "context"],
                "familiarity": min(1.0, word_count / 20),
            }
        if "emotion" in lenses:
            results["emotion"] = {
                "valence": round(np.random.uniform(-0.3, 0.8), 3),
                "arousal": round(np.random.uniform(0.2, 0.7), 3),
                "primary_emotion": "neutral",
            }
        if "reasoning" in lenses:
            results["reasoning"] = {
                "system_used": "system2" if word_count > 10 else "system1",
                "confidence": round(0.5 + np.random.random() * 0.4, 3),
                "complexity": "high" if word_count > 15 else "moderate" if word_count > 5 else "low",
            }
        if "creativity" in lenses:
            results["creativity"] = {
                "novelty_score": round(np.random.uniform(0.3, 0.9), 3),
                "associations": ["metaphor", "analogy", "abstraction"],
            }

        return {"text": text, "lenses": lenses, "analysis": results, "engine": "fallback"}

    def chat(self, message: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Process a chat message through the cognitive framework."""
        self._ensure_agent()
        history = history or []
        dim = 64

        np.random.seed(hash(message) % (2**32))
        msg_embedding = np.random.randn(dim)

        cognitive_trace = {}

        if self._agent:
            try:
                perceive_result = self._agent.perceive(msg_embedding)
                cognitive_trace["perception"] = {
                    "prediction_error": _serialize(perceive_result.get("prediction", {}).get("weighted_error", 0.0)),
                    "metacognition": _serialize(perceive_result.get("metacognition", {})),
                }

                think_result = self._agent.think(msg_embedding)
                cognitive_trace["thinking"] = {
                    "system_used": think_result.get("dual_process", {}).get("system_used", "unknown"),
                    "confidence": _serialize(think_result.get("dual_process", {}).get("confidence", 0.0)),
                }

                understand_result = self._agent.understand(message)
                cognitive_trace["understanding"] = _serialize(understand_result)

                utterance = self._agent.speak({
                    "predicate": "respond",
                    "arguments": {"query": message[:50], "context": "conversation"},
                })

                emotion_data = perceive_result.get("emotion", {})
                cognitive_trace["emotion"] = {
                    "valence": _serialize(emotion_data.get("valence", 0.0)),
                    "arousal": _serialize(emotion_data.get("arousal", 0.5)),
                }

                response = (
                    f"[Cognitive Processing Complete]\n"
                    f"System: {think_result.get('dual_process', {}).get('system_used', 'unknown')}\n"
                    f"Confidence: {think_result.get('dual_process', {}).get('confidence', 0.0):.2f}\n"
                    f"Linguistic output: {utterance}\n"
                    f"Memory associations: {think_result.get('recalled_memories', 0)} memories recalled"
                )

                return {
                    "message": message,
                    "response": response,
                    "cognitive_trace": cognitive_trace,
                    "engine": "real",
                }
            except Exception as e:
                logger.warning("Chat failed, using fallback: %s", e)

        word_count = len(message.split())
        system = "system2" if word_count > 8 or "?" in message else "system1"
        confidence = round(0.5 + np.random.random() * 0.45, 3)

        cognitive_trace = {
            "perception": {"prediction_error": round(np.random.random() * 0.5, 3)},
            "thinking": {"system_used": system, "confidence": confidence},
            "emotion": {"valence": round(np.random.uniform(-0.2, 0.8), 3), "arousal": round(np.random.uniform(0.2, 0.6), 3)},
        }

        response = (
            f"[Cognitive Processing Complete]\n"
            f"System: {system} | Confidence: {confidence:.2f}\n"
            f"Input analyzed: {word_count} tokens processed through cognitive pipeline.\n"
            f"The cognitive architecture processed your input through predictive coding, "
            f"dual-process evaluation ({system}), and emotional appraisal.\n"
            f"Note: Connect a language model (Ollama/LM Studio) for natural language responses."
        )

        return {
            "message": message,
            "response": response,
            "cognitive_trace": cognitive_trace,
            "engine": "fallback",
        }


simulator = CognitiveSimulator()
