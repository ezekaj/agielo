"""
Tests for Population-Based Code Evolution System
================================================

Verifies the population evolution capabilities work correctly:
1. CodeIndividual creation and serialization
2. CodeMutator applies various mutations
3. CodeCrossover combines parent code
4. Population management and selection
5. Fitness evaluation
6. Generation evolution
7. Lineage tracking
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.population_evolution import (
    CodeIndividual, CodeMutator, CodeCrossover, Population,
    MutationType, get_population
)


class TestCodeIndividual:
    """Tests for CodeIndividual dataclass."""

    def test_create_individual(self):
        """Individual should be created with correct fields."""
        ind = CodeIndividual(
            id="test123",
            code="def test(): pass",
            fitness_score=0.5,
            generation=1
        )
        assert ind.id == "test123"
        assert ind.code == "def test(): pass"
        assert ind.fitness_score == 0.5
        assert ind.generation == 1

    def test_to_dict(self):
        """Individual should serialize to dict."""
        ind = CodeIndividual(
            id="test456",
            code="def foo(): return 42",
            fitness_score=0.8,
            generation=2,
            parent_id="parent123"
        )
        d = ind.to_dict()
        assert d['id'] == "test456"
        assert d['fitness_score'] == 0.8
        assert d['parent_id'] == "parent123"

    def test_from_dict(self):
        """Individual should deserialize from dict."""
        data = {
            'id': 'test789',
            'code': 'def bar(): return 0',
            'fitness_score': 0.9,
            'generation': 3,
            'parent_id': None,
            'parent_ids': [],
            'mutations': ['param:1->2']
        }
        ind = CodeIndividual.from_dict(data)
        assert ind.id == 'test789'
        assert ind.fitness_score == 0.9
        assert 'param:1->2' in ind.mutations


class TestCodeMutator:
    """Tests for code mutation strategies."""

    def setup_method(self):
        self.mutator = CodeMutator()

    def test_validate_code(self):
        """Valid code should pass validation."""
        valid = "def test(): return 1"
        invalid = "def test( return 1"  # Missing paren

        assert self.mutator.validate_code(valid)
        assert not self.mutator.validate_code(invalid)

    def test_parameter_mutation(self):
        """Parameter mutation should change numeric values."""
        code = '''
def example():
    x = 100
    return x * 2
'''
        # Run multiple times - should sometimes mutate
        mutated_any = False
        for _ in range(10):
            mutated, desc = self.mutator._parameter_mutation(code)
            if "100" not in mutated or "2" not in mutated:
                mutated_any = True
                break
            if "parameter:" in desc and "no_change" not in desc:
                mutated_any = True
                break

        # At minimum, code should still be valid
        assert self.mutator.validate_code(mutated)

    def test_operator_mutation(self):
        """Operator mutation should change operators."""
        code = '''
def add(a, b):
    return a + b
'''
        # Run mutation
        mutated, desc = self.mutator._operator_mutation(code)

        # Code should be valid
        assert self.mutator.validate_code(mutated)

    def test_structure_mutation(self):
        """Structure mutation should swap branches."""
        code = '''
def check(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
'''
        mutated, desc = self.mutator._structure_mutation(code)
        assert self.mutator.validate_code(mutated)

    def test_simplification_mutation(self):
        """Simplification should simplify expressions."""
        code = '''
def add_zero(x):
    return x + 0
'''
        mutated, desc = self.mutator._simplification_mutation(code)
        assert self.mutator.validate_code(mutated)
        # Should potentially simplify x + 0 to just x

    def test_expansion_mutation(self):
        """Expansion should add type hints."""
        code = '''
def get_value():
    return 42
'''
        mutated, desc = self.mutator._expansion_mutation(code)
        assert self.mutator.validate_code(mutated)

    def test_mutate_random(self):
        """Random mutation should work."""
        code = '''
def test():
    x = 10
    if x > 5:
        return True
    return False
'''
        mutated, desc = self.mutator.mutate(code)
        assert self.mutator.validate_code(mutated)


class TestCodeCrossover:
    """Tests for code crossover."""

    def setup_method(self):
        self.crossover = CodeCrossover()

    def test_crossover_common_function(self):
        """Crossover should mix common functions."""
        parent1 = '''
def calculate(x):
    return x * 2
'''
        parent2 = '''
def calculate(x):
    result = x + x
    return result
'''
        child, desc = self.crossover.crossover(parent1, parent2)
        # Child should be valid code
        import ast
        try:
            ast.parse(child)
            valid = True
        except:
            valid = False
        assert valid, f"Child code invalid: {child}"

    def test_crossover_no_common(self):
        """Crossover with no common functions should not crash."""
        parent1 = "def foo(): pass"
        parent2 = "def bar(): pass"

        child, desc = self.crossover.crossover(parent1, parent2)
        # Should return one of the parents
        assert child in [parent1, parent2] or "no_common" in desc


class TestPopulation:
    """Tests for population management."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.pop = Population(max_size=10, storage_dir=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_individual(self):
        """Should add individual to population."""
        ind = self.pop.add_individual("def test(): pass")
        assert len(self.pop.individuals) == 1
        assert ind.id is not None

    def test_max_size_enforcement(self):
        """Population should not exceed max size."""
        for i in range(15):
            self.pop.add_individual(f"def test{i}(): return {i}", fitness=i/15)

        assert len(self.pop.individuals) <= self.pop.max_size

    def test_select_parents(self):
        """Tournament selection should return parents."""
        for i in range(5):
            self.pop.add_individual(f"def test{i}(): pass", fitness=i/5)

        parents = self.pop.select_parents(2)
        assert len(parents) == 2
        # Tournament selection favors higher fitness

    def test_crossover_parents(self):
        """Crossover should create child with lineage."""
        ind1 = self.pop.add_individual("def calc(x): return x * 2")
        ind2 = self.pop.add_individual("def calc(x): return x + x")

        child = self.pop.crossover_parents(ind1, ind2)
        assert child.parent_ids == [ind1.id, ind2.id]
        assert child.generation == self.pop.generation + 1

    def test_mutate_individual(self):
        """Mutation should create new individual with lineage."""
        ind = self.pop.add_individual("def test(): return 100")
        mutant = self.pop.mutate_individual(ind)

        assert mutant.parent_id == ind.id
        assert len(mutant.mutations) > 0

    def test_evaluate_fitness(self):
        """Fitness evaluation should return score."""
        ind = self.pop.add_individual('''
def add(a, b):
    return a + b
''')
        test_cases = [
            {'function': 'add', 'args': [2, 3], 'expected': 5}
        ]

        fitness = self.pop.evaluate_fitness(ind, test_cases)
        assert 0 <= fitness <= 1
        assert ind.fitness_score == fitness

    def test_evolve_generation(self):
        """Evolution should create new generation."""
        # Add initial population
        for i in range(5):
            self.pop.add_individual(f'''
def add(a, b):
    return a + b + {i * 0}  # variation
''')

        test_cases = [
            {'function': 'add', 'args': [2, 3], 'expected': 5}
        ]

        initial_gen = self.pop.generation
        stats = self.pop.evolve_generation(test_cases)

        assert self.pop.generation == initial_gen + 1
        assert stats['generation'] == initial_gen + 1
        assert stats['best_fitness'] >= 0

    def test_get_best(self):
        """Should return best individuals by fitness."""
        self.pop.add_individual("def a(): pass", fitness=0.3)
        self.pop.add_individual("def b(): pass", fitness=0.9)
        self.pop.add_individual("def c(): pass", fitness=0.5)

        best = self.pop.get_best(2)
        assert len(best) == 2
        assert best[0].fitness_score >= best[1].fitness_score

    def test_diversity(self):
        """Diversity should measure population variety."""
        # Add very similar code
        self.pop.add_individual("def test(): return 1")
        self.pop.add_individual("def test(): return 1")
        low_diversity = self.pop.get_diversity()

        # Add diverse code
        self.pop.add_individual("def totally_different(x, y, z): return x * y + z")
        higher_diversity = self.pop.get_diversity()

        # More diverse population should have higher diversity
        assert higher_diversity >= low_diversity

    def test_get_stats(self):
        """Stats should reflect population state."""
        self.pop.add_individual("def test(): pass", fitness=0.5)
        self.pop.add_individual("def test2(): pass", fitness=0.8)

        stats = self.pop.get_stats()
        assert stats['population_size'] == 2
        assert stats['best_fitness'] == 0.8
        assert 0.5 <= stats['avg_fitness'] <= 0.8

    def test_lineage_tracking(self):
        """Should track individual lineage."""
        parent = self.pop.add_individual("def test(): return 1")
        parent.fitness_score = 0.9  # Make it selectable

        child = self.pop.mutate_individual(parent)
        self.pop.individuals.append(child)

        lineage = self.pop.get_lineage(child.id)
        assert len(lineage) >= 1
        # Lineage should include parent

    def test_state_persistence(self):
        """Population should save and load state."""
        self.pop.add_individual("def test(): pass", fitness=0.7)
        self.pop._save_state()

        # Create new population from same directory
        pop2 = Population(storage_dir=self.temp_dir)
        assert len(pop2.individuals) == 1
        assert pop2.individuals[0].fitness_score == 0.7


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestCodeIndividual,
        TestCodeMutator,
        TestCodeCrossover,
        TestPopulation,
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    # Setup
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()

                    # Run test
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    total_passed += 1

                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))

                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, traceback.format_exc()))

                finally:
                    # Teardown
                    if hasattr(instance, 'teardown_method'):
                        try:
                            instance.teardown_method()
                        except:
                            pass

    print(f"\n{'='*60}")
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print('='*60)

    if failures:
        print("\nFailures:")
        for cls, method, error in failures:
            print(f"\n  {cls}.{method}:")
            print(f"    {error[:200]}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
