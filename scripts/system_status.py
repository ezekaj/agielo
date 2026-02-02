#!/usr/bin/env python3
"""
System Status Reporter
======================

Outputs comprehensive status information about the Human Cognition AI system:
- Current evolution cycle number
- Total unique facts learned
- Number of training pairs available
- Last benchmark score
- Memory statistics from ChromaDB

Usage:
    python3 scripts/system_status.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.paths import (
    KNOWLEDGE_DIR, EVOLUTION_DIR, TRAINING_DATA_FILE,
    LEARNED_HASHES_FILE, BENCHMARK_HISTORY_FILE, EVOLUTION_STATE_FILE
)


def get_evolution_cycle() -> int:
    """Get the current evolution cycle number."""
    if not EVOLUTION_STATE_FILE.exists():
        return 0
    try:
        with open(EVOLUTION_STATE_FILE, 'r') as f:
            state = json.load(f)
            return state.get('current_cycle', 0)
    except (json.JSONDecodeError, IOError):
        return 0


def get_total_facts_learned() -> int:
    """Get total unique facts learned (hash count)."""
    if not LEARNED_HASHES_FILE.exists():
        return 0
    try:
        with open(LEARNED_HASHES_FILE, 'r') as f:
            hashes = json.load(f)
            return len(hashes) if isinstance(hashes, list) else 0
    except (json.JSONDecodeError, IOError):
        return 0


def get_training_pairs_count() -> int:
    """Count valid training pairs in training_data.jsonl."""
    if not TRAINING_DATA_FILE.exists():
        return 0
    try:
        valid_count = 0
        with open(TRAINING_DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if (isinstance(data, dict) and
                        data.get('prompt', '').strip() and
                        data.get('completion', '').strip()):
                        valid_count += 1
                except json.JSONDecodeError:
                    continue
        return valid_count
    except (IOError, OSError, UnicodeDecodeError):
        return 0


def get_last_benchmark_score() -> dict:
    """Get the last benchmark score and details."""
    if not BENCHMARK_HISTORY_FILE.exists():
        return {'score': None, 'timestamp': None, 'cycle': None}
    try:
        with open(BENCHMARK_HISTORY_FILE, 'r') as f:
            history = json.load(f)
            if history and isinstance(history, list) and len(history) > 0:
                last = history[-1]
                return {
                    'score': last.get('score'),
                    'timestamp': last.get('timestamp'),
                    'cycle': last.get('cycle'),
                    'details': last.get('details', {})
                }
        return {'score': None, 'timestamp': None, 'cycle': None}
    except (json.JSONDecodeError, IOError):
        return {'score': None, 'timestamp': None, 'cycle': None}


def get_memory_statistics() -> dict:
    """Get ChromaDB memory statistics."""
    stats = {
        'total_episodes': 0,
        'episodes_in_memory': 0,
        'episodes_offloaded': 0,
        'unique_locations': 0,
        'unique_entities': 0,
        'available': False
    }

    # Check if neuro_memory path exists
    neuro_memory_path = Path.home() / ".neuro_memory"
    memory_state_file = neuro_memory_path / "memory_state.json"

    if memory_state_file.exists():
        try:
            with open(memory_state_file, 'r') as f:
                state = json.load(f)
                stats['total_episodes'] = state.get('total_episodes_stored', 0)
                stats['episodes_in_memory'] = len(state.get('episodes', []))
                stats['episodes_offloaded'] = state.get('episodes_offloaded', 0)
                stats['unique_locations'] = len(state.get('spatial_index', {}))
                stats['unique_entities'] = len(state.get('entity_index', {}))
                stats['available'] = True
        except (json.JSONDecodeError, IOError):
            pass

    # Also try to get ChromaDB collection stats directly
    chroma_path = neuro_memory_path / "chroma"
    if chroma_path.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_path))
            try:
                collection = client.get_collection("episodic_memories")
                stats['chromadb_count'] = collection.count()
                stats['available'] = True
            except Exception:
                pass
        except ImportError:
            stats['chromadb_note'] = "chromadb not installed"
        except Exception:
            pass

    return stats


def format_score(score) -> str:
    """Format a score for display."""
    if score is None:
        return "N/A"
    try:
        return f"{float(score):.1%}"
    except (ValueError, TypeError):
        return str(score)


def format_timestamp(ts_str) -> str:
    """Format an ISO timestamp for display."""
    if ts_str is None:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(ts_str)


def main():
    """Output system status report."""
    print("=" * 60)
    print("Human Cognition AI - System Status Report")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Evolution Status
    print("EVOLUTION STATUS")
    print("-" * 40)
    cycle = get_evolution_cycle()
    facts = get_total_facts_learned()
    training_pairs = get_training_pairs_count()

    print(f"  Current cycle:       {cycle}")
    print(f"  Unique facts learned: {facts}")
    print(f"  Training pairs:       {training_pairs}")

    # Evolution state details
    if EVOLUTION_STATE_FILE.exists():
        try:
            with open(EVOLUTION_STATE_FILE, 'r') as f:
                state = json.load(f)
                print(f"  Facts this cycle:     {state.get('facts_this_cycle', 0)}")
                print(f"  Total trainings:      {state.get('total_trainings', 0)}")
                baseline = state.get('baseline_score')
                current = state.get('current_score')
                print(f"  Baseline score:       {format_score(baseline)}")
                print(f"  Current score:        {format_score(current)}")
                if baseline is not None and current is not None:
                    improvement = current - baseline
                    print(f"  Improvement:          {improvement:+.1%}")
        except (json.JSONDecodeError, IOError):
            pass
    print()

    # Benchmark Status
    print("BENCHMARK STATUS")
    print("-" * 40)
    benchmark = get_last_benchmark_score()
    print(f"  Last score:          {format_score(benchmark['score'])}")
    print(f"  Benchmark cycle:     {benchmark['cycle'] if benchmark['cycle'] is not None else 'N/A'}")
    print(f"  Last run:            {format_timestamp(benchmark['timestamp'])}")

    if benchmark.get('details'):
        print("  Category scores:")
        for category, score in benchmark['details'].items():
            if isinstance(score, (int, float)):
                print(f"    - {category}: {format_score(score)}")
    print()

    # Memory Statistics
    print("MEMORY STATISTICS (ChromaDB)")
    print("-" * 40)
    memory_stats = get_memory_statistics()

    if memory_stats['available']:
        print(f"  Total episodes stored:  {memory_stats['total_episodes']}")
        print(f"  Episodes in memory:     {memory_stats['episodes_in_memory']}")
        print(f"  Episodes offloaded:     {memory_stats['episodes_offloaded']}")
        print(f"  Unique locations:       {memory_stats['unique_locations']}")
        print(f"  Unique entities:        {memory_stats['unique_entities']}")
        if 'chromadb_count' in memory_stats:
            print(f"  ChromaDB collection:    {memory_stats['chromadb_count']} items")
    else:
        print("  No memory data available yet.")
        print("  (Memory statistics will appear after first chat session)")
        if 'chromadb_note' in memory_stats:
            print(f"  Note: {memory_stats['chromadb_note']}")
    print()

    # Data Paths
    print("DATA PATHS")
    print("-" * 40)
    print(f"  Knowledge dir:        {KNOWLEDGE_DIR}")
    print(f"  Evolution dir:        {EVOLUTION_DIR}")
    print(f"  Training data:        {TRAINING_DATA_FILE}")

    # Check which paths exist
    paths_status = {
        'Knowledge dir': KNOWLEDGE_DIR.exists(),
        'Evolution dir': EVOLUTION_DIR.exists(),
        'Training data': TRAINING_DATA_FILE.exists(),
        'Learned hashes': LEARNED_HASHES_FILE.exists(),
        'Benchmark history': BENCHMARK_HISTORY_FILE.exists(),
        'Evolution state': EVOLUTION_STATE_FILE.exists()
    }

    print("\n  Path status:")
    for name, exists in paths_status.items():
        status = "exists" if exists else "not found"
        print(f"    - {name}: {status}")

    print()
    print("=" * 60)
    print("Status report complete.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
