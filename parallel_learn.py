#!/usr/bin/env python3
"""Parallel learning workers - run multiple instances."""
import sys
import os
import time
import multiprocessing as mp
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def worker(worker_id, sources):
    """Single learning worker."""
    from integrations.self_evolution import get_evolution
    from integrations.self_training import SelfTrainer
    from integrations.cognitive_ollama import CognitiveOllama
    
    print(f"[Worker {worker_id}] Starting with sources: {sources}")
    
    evolution = get_evolution()
    trainer = SelfTrainer()
    ai = CognitiveOllama(model="ministral-3:8b")
    
    # Import fetch functions from chat
    from chat import AutonomousAI
    chat = AutonomousAI.__new__(AutonomousAI)
    chat.ai = ai
    chat.evolution = evolution
    chat.trainer = trainer
    chat.running = True
    chat._learning_order_idx = worker_id  # Start at different sources
    
    while True:
        try:
            # Learn one fact
            chat._learn_unique_fact()
            stats = evolution.get_stats()
            print(f"[Worker {worker_id}] Facts: {stats['facts_this_cycle']}/100 | Total: {stats['total_facts']}")
            time.sleep(0.3)  # Faster learning
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            time.sleep(5)

def main():
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    
    print(f"Starting {num_workers} parallel learning workers...")
    print("=" * 50)
    
    # Define source groups for each worker
    source_groups = [
        ['arxiv', 'math'],
        ['gdelt', 'logic'],
        ['github', 'sciencedaily'],
        ['wikipedia', 'benchmark'],
    ]
    
    processes = []
    for i in range(num_workers):
        sources = source_groups[i % len(source_groups)]
        p = mp.Process(target=worker, args=(i, sources))
        p.start()
        processes.append(p)
        time.sleep(2)  # Stagger starts
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping workers...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
