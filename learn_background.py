#!/usr/bin/env python3
"""Background learning - no user input needed."""
import sys
import time
sys.path.insert(0, '.')

from integrations.self_evolution import get_evolution
from integrations.self_training import SelfTrainer
from integrations.benchmark import Benchmark

class BackgroundLearner:
    def __init__(self):
        self.evolution = get_evolution()
        self.trainer = SelfTrainer()
        self.benchmark = Benchmark()
        
        # Import chat functions
        from chat import AutonomousAI
        self.chat = AutonomousAI()
        self.chat.running = True
        
    def run(self):
        print("=" * 60)
        print("AGIELO BACKGROUND LEARNING")
        print("=" * 60)
        print(f"Facts: {self.evolution.get_stats()['total_facts']}")
        print("Learning autonomously...")
        print("=" * 60)
        
        # Fake a conversation to trigger learning
        self.chat.ai.history = [{"role": "user", "content": "start learning"}]
        
        while True:
            try:
                # Use the chat's learning methods
                if not self.evolution.should_benchmark():
                    self.chat._learn_unique_fact()
                else:
                    print(f"\n[BENCHMARK] Running after {self.evolution.state['facts_this_cycle']} facts...")
                    self.chat._run_benchmark_and_report("CYCLE")
                    
                    should_train, reason = self.evolution.should_train()
                    print(f"[TRAIN] {reason}")
                    
                    if should_train:
                        self.evolution.run_mlx_training()
                    
                    self.evolution.start_new_cycle()
                    
                time.sleep(3)
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    learner = BackgroundLearner()
    learner.run()
