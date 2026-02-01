#!/usr/bin/env python3
"""
Human Cognition AI - Usage Examples
===================================

This file demonstrates how to run and use the cognitive agent system.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cognitive_agent import CognitiveAgent, CognitiveConfig


def example_1_basic_usage():
    """Basic usage - create agent and process observations."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)

    # Create a cognitive agent with default settings
    agent = CognitiveAgent()

    # Perceive an observation (e.g., sensory input)
    observation = np.random.randn(64)  # 64-dimensional vector
    result = agent.perceive(observation)

    print(f"Perceived observation, prediction error: {result['prediction'].get('weighted_error', 0):.4f}")
    print(f"Metacognitive confidence: {result['metacognition']['confidence']:.3f}")
    print(f"Current motivation level: {result['motivation']['motivation_level']:.3f}")

    return agent


def example_2_thinking_and_reasoning():
    """Use the agent to think about problems."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Thinking and Reasoning")
    print("="*60)

    agent = CognitiveAgent()

    # Give the agent a problem to think about
    problem = np.random.randn(64)
    context = {'goal': 'solve_problem'}

    result = agent.think(problem, context)

    print(f"System used: {result['dual_process']['system_used']}")
    print(f"Confidence: {result['dual_process']['confidence']:.3f}")
    print(f"Memories recalled: {result['recalled_memories']}")
    print(f"Executive control - should inhibit: {result['executive']['should_inhibit']}")

    return agent


def example_3_decision_making():
    """Make decisions between options."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Decision Making")
    print("="*60)

    agent = CognitiveAgent()

    # Define options with embeddings
    options = [
        ('explore_unknown', np.random.randn(64)),
        ('exploit_known', np.random.randn(64)),
        ('wait_and_observe', np.random.randn(64)),
    ]

    # Optional: set a goal
    goal = np.random.randn(64)

    result = agent.decide(options, goal)

    print(f"Decision: {result['final_decision']}")
    print(f"Expected value: {result['expected_value']:.3f}")
    if 'emotional_signals' in result:
        print(f"Emotional signals (gut feelings): {[f'{s:.2f}' for s in result['emotional_signals']]}")

    return agent


def example_4_language():
    """Process and generate language."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Language Processing")
    print("="*60)

    agent = CognitiveAgent()

    # Understand language
    sentences = [
        "the cat chases the mouse",
        "give me the book",
        "where is the car?",
    ]

    for sentence in sentences:
        result = agent.understand(sentence)
        print(f"\nInput: '{sentence}'")
        print(f"  Speech act: {result['speech_act']}")
        print(f"  Predicate: {result['proposition']['predicate']}")
        print(f"  Arguments: {result['proposition']['arguments']}")

    # Generate language
    print("\n--- Generating language ---")
    meanings = [
        {'predicate': 'chase', 'arguments': {'agent': 'dog', 'patient': 'cat'}},
        {'predicate': 'give', 'arguments': {'agent': 'I', 'theme': 'book', 'goal': 'you'}},
    ]

    for meaning in meanings:
        utterance = agent.speak(meaning)
        print(f"Meaning: {meaning}")
        print(f"Generated: '{utterance}'")

    return agent


def example_5_memory():
    """Store and recall memories."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Memory Operations")
    print("="*60)

    agent = CognitiveAgent()

    # Store episodic memories with different emotional valences
    memories = [
        ('happy_event', np.random.randn(64), 0.8),   # Positive
        ('neutral_event', np.random.randn(64), 0.0), # Neutral
        ('sad_event', np.random.randn(64), -0.6),    # Negative
    ]

    for name, content, emotion in memories:
        agent.remember(content, name, emotion)
        print(f"Stored memory: {name} (emotional valence: {emotion})")

    # Recall memories using a query
    query = memories[0][1] + np.random.randn(64) * 0.1  # Similar to first memory
    recalled = agent._modules['memory'].recall(query, k=2)
    print(f"\nRecalled {len(recalled)} memories based on query")

    return agent


def example_6_introspection():
    """Self-reflection and metacognition."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Introspection & Self-Awareness")
    print("="*60)

    agent = CognitiveAgent()

    # Do some processing first
    for _ in range(5):
        agent.perceive(np.random.randn(64))
        agent.think(np.random.randn(64))

    # Introspect
    intro = agent.introspect()

    print(f"Cognitive state: {intro['cognitive_state']}")
    print(f"\nSelf-awareness check:")
    for key, value in intro['am_i_aware'].items():
        print(f"  {key}: {value}")

    print(f"\nReflection (depth 2):")
    reflection = intro['reflection']
    if 'basic' in reflection:
        for key, value in reflection['basic'].items():
            print(f"  {key}: {value}")

    return agent


def example_7_creativity():
    """Creative problem solving."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Creativity")
    print("="*60)

    config = CognitiveConfig(enable_creativity=True)
    agent = CognitiveAgent(config)
    agent._ensure_modules()

    if 'creativity' not in agent._modules:
        print("Creativity module not available")
        return agent

    creativity = agent._modules['creativity']

    # Creative problem solving
    problem = np.random.randn(64)
    result = creativity.creative_process(problem, time_budget=50)

    print(f"Generated {len(result['ideas'])} ideas")
    print(f"Insights found: {len(result['insights'])}")
    if result['best_idea']:
        print(f"Best idea novelty: {result['best_idea'].novelty:.3f}")
        print(f"Best idea usefulness: {result['best_idea'].usefulness:.3f}")

    return agent


def example_8_social_cognition():
    """Theory of Mind and social reasoning."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Social Cognition (Theory of Mind)")
    print("="*60)

    config = CognitiveConfig(enable_social=True)
    agent = CognitiveAgent(config)
    agent._ensure_modules()

    if 'social' not in agent._modules:
        print("Social cognition module not available")
        return agent

    social = agent._modules['social']

    # Meet another agent
    other_embedding = np.random.randn(64)
    agent.meet('alice', other_embedding)
    print("Met agent 'alice'")

    # Attribute beliefs and desires to them
    social.theory_of_mind.attribute_belief('alice', 'location_of_ball', 'under_cup', 0.9)
    social.theory_of_mind.attribute_desire('alice', 'find_ball', 0.8)

    # Predict their action
    predicted_action, confidence = social.theory_of_mind.predict_action('alice', np.zeros(64))
    print(f"Predicted action for alice: {predicted_action} (confidence: {confidence:.2f})")

    # Simulate their perspective
    situation = {'ball_visible': False, 'cup_present': True}
    perspective = agent.infer_mental_state('alice', situation)
    print(f"Alice's simulated beliefs: {perspective.get('simulated_beliefs', {})}")

    return agent


def example_9_full_cognitive_loop():
    """Complete cognitive cycle: perceive -> think -> decide -> act -> learn."""
    print("\n" + "="*60)
    print("EXAMPLE 9: Full Cognitive Loop")
    print("="*60)

    agent = CognitiveAgent()

    print("Running 10 cognitive cycles...\n")

    for cycle in range(10):
        # 1. Perceive
        observation = np.random.randn(64)
        percept = agent.perceive(observation)

        # 2. Think
        thought = agent.think(observation, {'goal': 'optimize'})

        # 3. Decide
        options = [
            ('action_A', np.random.randn(64)),
            ('action_B', np.random.randn(64)),
        ]
        decision = agent.decide(options)

        # 4. Learn from outcome (simulated reward)
        reward = np.random.random()  # Random reward for demo
        agent.learn({
            'input': observation,
            'response': decision['final_decision'],
            'reward': reward
        })

        print(f"Cycle {cycle+1}: decided '{decision['final_decision']}', reward: {reward:.2f}")

    # Check final state
    state = agent.get_full_state()
    print(f"\nFinal state after 10 cycles:")
    print(f"  Total steps: {state['step_count']}")
    print(f"  Modules active: {len(state['modules'])}")

    return agent


def example_10_configuration():
    """Different configuration options."""
    print("\n" + "="*60)
    print("EXAMPLE 10: Configuration Options")
    print("="*60)

    # Minimal agent (faster, less features)
    minimal_config = CognitiveConfig(
        dim=32,  # Smaller embedding dimension
        enable_emotions=False,
        enable_embodiment=False,
        enable_social=False,
        enable_creativity=False,
        enable_sleep=False,
        fast_mode=True
    )
    minimal_agent = CognitiveAgent(minimal_config)
    print("Created minimal agent (dim=32, core modules only)")

    # Full agent (all features)
    full_config = CognitiveConfig(
        dim=128,  # Larger embedding dimension
        enable_emotions=True,
        enable_embodiment=True,
        enable_social=True,
        enable_creativity=True,
        enable_sleep=True,
        fast_mode=False
    )
    full_agent = CognitiveAgent(full_config)
    print("Created full agent (dim=128, all modules)")

    # Compare performance
    from core.cognitive_agent import benchmark_agent

    print("\nBenchmarking minimal agent...")
    minimal_bench = benchmark_agent(50)
    print(f"  Throughput: {minimal_bench['steps_per_second']:.0f} ops/sec")

    return minimal_agent, full_agent


def run_all_examples():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# HUMAN COGNITION AI - COMPLETE USAGE GUIDE")
    print("#"*60)

    example_1_basic_usage()
    example_2_thinking_and_reasoning()
    example_3_decision_making()
    example_4_language()
    example_5_memory()
    example_6_introspection()
    example_7_creativity()
    example_8_social_cognition()
    example_9_full_cognitive_loop()
    example_10_configuration()

    print("\n" + "#"*60)
    print("# ALL EXAMPLES COMPLETED")
    print("#"*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Human Cognition AI Examples')
    parser.add_argument('--example', '-e', type=int, choices=range(1, 11),
                        help='Run specific example (1-10)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Run all examples')
    parser.add_argument('--benchmark', '-b', action='store_true',
                        help='Run performance benchmark')

    args = parser.parse_args()

    if args.benchmark:
        from core.cognitive_agent import benchmark_agent
        print("Running benchmark (100 steps)...")
        result = benchmark_agent(100)
        print(f"Throughput: {result['steps_per_second']:.0f} ops/sec")
        print(f"Perception: {result['timings']['perception_per_step_ms']:.2f} ms")
        print(f"Thinking: {result['timings']['thinking_per_step_ms']:.2f} ms")
        print(f"Decision: {result['timings']['decision_per_step_ms']:.2f} ms")
    elif args.example:
        examples = {
            1: example_1_basic_usage,
            2: example_2_thinking_and_reasoning,
            3: example_3_decision_making,
            4: example_4_language,
            5: example_5_memory,
            6: example_6_introspection,
            7: example_7_creativity,
            8: example_8_social_cognition,
            9: example_9_full_cognitive_loop,
            10: example_10_configuration,
        }
        examples[args.example]()
    elif args.all:
        run_all_examples()
    else:
        # Default: show help and run basic example
        print("Human Cognition AI System")
        print("="*40)
        print("\nUsage:")
        print("  python run_examples.py --all        # Run all examples")
        print("  python run_examples.py -e 1         # Run example 1")
        print("  python run_examples.py --benchmark  # Run benchmark")
        print("\nRunning basic example...\n")
        example_1_basic_usage()
