# Human Cognition AI

A complete implementation of human-like cognition based on neuroscience research.

## Quick Start

```bash
cd ~/Desktop/human-cognition-ai

# Run all examples
python3 run_examples.py --all

# Run specific example
python3 run_examples.py -e 1

# Run benchmark
python3 run_examples.py --benchmark

# Run the main demo
python3 core/cognitive_agent.py
```

## Requirements

- Python 3.8+
- NumPy

```bash
pip3 install numpy
```

Optional (for better performance):
```bash
pip3 install numba  # JIT compilation for 10-100x speedup
```

## Usage

### Basic Usage

```python
from core.cognitive_agent import CognitiveAgent
import numpy as np

# Create agent
agent = CognitiveAgent()

# Perceive (process sensory input)
observation = np.random.randn(64)
result = agent.perceive(observation)
print(f"Prediction error: {result['prediction']['weighted_error']}")

# Think (deliberate reasoning)
problem = np.random.randn(64)
result = agent.think(problem)
print(f"Confidence: {result['dual_process']['confidence']}")

# Decide (choose between options)
options = [
    ('option_a', np.random.randn(64)),
    ('option_b', np.random.randn(64)),
]
result = agent.decide(options)
print(f"Decision: {result['final_decision']}")

# Introspect (self-reflection)
intro = agent.introspect()
print(f"Cognitive state: {intro['cognitive_state']}")
```

### Language Processing

```python
# Understand language
result = agent.understand("the cat chases the mouse")
print(f"Speech act: {result['speech_act']}")
print(f"Predicate: {result['proposition']['predicate']}")

# Generate language
utterance = agent.speak({
    'predicate': 'give',
    'arguments': {'agent': 'I', 'theme': 'book', 'goal': 'you'}
})
print(f"Generated: {utterance}")
```

### Memory

```python
# Store episodic memory
agent.remember(
    content=np.random.randn(64),
    episode_id='my_experience',
    emotional_state=0.8  # Positive emotion
)

# Recall memories
recalled = agent._modules['memory'].recall(query_embedding, k=5)
```

### Social Cognition (Theory of Mind)

```python
# Meet another agent
agent.meet('alice', np.random.randn(64))

# Infer their mental state
perspective = agent.infer_mental_state('alice', {'situation': 'data'})
```

### Configuration

```python
from core.cognitive_agent import CognitiveAgent, CognitiveConfig

# Minimal (fast)
config = CognitiveConfig(
    dim=32,
    enable_emotions=False,
    enable_embodiment=False,
    enable_social=False,
    enable_creativity=False,
    enable_sleep=False
)
agent = CognitiveAgent(config)

# Full (all features)
config = CognitiveConfig(
    dim=128,
    enable_emotions=True,
    enable_embodiment=True,
    enable_social=True,
    enable_creativity=True,
    enable_sleep=True
)
agent = CognitiveAgent(config)
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Self-Awareness Loop                │
│    (Metacognition, Introspection, Self-Model)   │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│              Executive Control                   │
│      (Inhibition, Task Switching, Attention)    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│              Dual-Process System                 │
│         System 1 (Fast) + System 2 (Slow)       │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│              Processing Layer                    │
│   Reasoning | Creativity | Time | Cog Maps      │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│              Motivation Layer                    │
│       Drives/Goals | Emotions | Curiosity       │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│              Interface Layer                     │
│        Language | Social | Embodied             │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│              Foundation Layer                    │
│   Predictive Coding | Memory | Learning         │
└─────────────────────────────────────────────────┘
```

## Modules

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `predictive_coding.py` | Free Energy Principle, prediction errors |
| 1 | `memory_system.py` | Working, episodic, semantic, procedural memory |
| 1 | `learning_system.py` | Hebbian, STDP, reward-modulated learning |
| 2 | `dual_process.py` | System 1 (fast) + System 2 (slow) |
| 2 | `executive_control.py` | Inhibition, task switching, attention |
| 2 | `reasoning_modules.py` | Perceptual, dimensional, logical, interactive |
| 3 | `motivation_engine.py` | Curiosity, drives, goals, dopamine |
| 3 | `emotion_system.py` | Appraisal, somatic markers, mood |
| 3 | `self_awareness.py` | Metacognition, self-model, introspection |
| 4 | `embodied_cognition.py` | Body schema, affordances, motor simulation |
| 4 | `language_system.py` | Grounded language, constructions, pragmatics |
| 4 | `social_cognition.py` | Theory of Mind, morality, norms |
| 5 | `creativity_module.py` | Divergent thinking, blending, insight |
| 5 | `cognitive_maps.py` | Place/grid cells, navigation, mental travel |
| 5 | `sleep_consolidation.py` | Memory replay, dreams, synaptic homeostasis |
| 5 | `time_perception.py` | Interval timing, subjective time, prospection |

## Performance

| Operation | Time | Throughput |
|-----------|------|------------|
| Perception | ~0.8 ms | |
| Thinking | ~0.3 ms | |
| Decision | ~0.2 ms | |
| **Total** | | **~2000+ ops/sec** |

## Comparison with Existing Systems

See [COMPARISON.md](COMPARISON.md) for detailed analysis vs:
- ACT-R
- SOAR
- LLMs (GPT, Claude)
- Reinforcement Learning

## Based On

- Friston: Free Energy Principle
- Kahneman: Dual-Process Theory
- Damasio: Somatic Marker Hypothesis
- Baddeley: Working Memory Model
- Tulving: Episodic Memory
- Premack & Woodruff: Theory of Mind
- Tononi & Cirelli: Sleep and Consciousness
