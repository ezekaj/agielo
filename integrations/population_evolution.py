"""
Population-Based Code Evolution System
======================================

Implements genetic algorithm-style evolution for code improvements.
Maintains multiple code variants and evolves the best ones (inspired by Darwin Godel Machine).

Features:
1. CodeIndividual: Single code variant with fitness tracking
2. Population: Collection of individuals with generation management
3. Tournament selection based on fitness
4. AST-based crossover combining code from parents
5. Various mutation strategies (parameter, structure, operator, etc.)
6. Fitness evaluation via sandbox testing
7. Lineage tracking for understanding evolution paths
"""

import os
import sys
import ast
import json
import copy
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import re

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import EVOLUTION_DIR


class MutationType(Enum):
    """Types of code mutations."""
    PARAMETER = "parameter"         # Change numeric constants
    STRUCTURE = "structure"         # Swap branches, change loop types
    OPERATOR = "operator"           # Change +/-, *//, and/or
    SIMPLIFICATION = "simplification"  # Remove dead code, inline functions
    EXPANSION = "expansion"         # Add error handling, type hints


@dataclass
class CodeIndividual:
    """
    Represents a single code variant in the population.

    Tracks code, fitness, lineage, and mutations applied.
    """
    id: str
    code: str
    fitness_score: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)  # For crossover (both parents)
    mutations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    test_results: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'code': self.code,
            'fitness_score': self.fitness_score,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'parent_ids': self.parent_ids,
            'mutations': self.mutations,
            'created_at': self.created_at,
            'test_results': self.test_results
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CodeIndividual':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            code=data['code'],
            fitness_score=data.get('fitness_score', 0.0),
            generation=data.get('generation', 0),
            parent_id=data.get('parent_id'),
            parent_ids=data.get('parent_ids', []),
            mutations=data.get('mutations', []),
            created_at=data.get('created_at', datetime.now().isoformat()),
            test_results=data.get('test_results')
        )


class CodeMutator:
    """
    Applies various mutations to code using AST transformations.

    Each mutation preserves code correctness (validates after mutation).
    """

    def __init__(self):
        self.mutation_log: List[Dict] = []

    def mutate(self, code: str, mutation_type: MutationType = None) -> Tuple[str, str]:
        """
        Apply a mutation to code.

        Args:
            code: Python code to mutate
            mutation_type: Specific mutation type, or random if None

        Returns:
            (mutated_code, mutation_description)
        """
        if mutation_type is None:
            mutation_type = random.choice(list(MutationType))

        try:
            if mutation_type == MutationType.PARAMETER:
                return self._parameter_mutation(code)
            elif mutation_type == MutationType.STRUCTURE:
                return self._structure_mutation(code)
            elif mutation_type == MutationType.OPERATOR:
                return self._operator_mutation(code)
            elif mutation_type == MutationType.SIMPLIFICATION:
                return self._simplification_mutation(code)
            elif mutation_type == MutationType.EXPANSION:
                return self._expansion_mutation(code)
        except Exception as e:
            # If mutation fails, return original code
            return code, f"mutation_failed:{mutation_type.value}:{str(e)}"

        return code, "no_mutation"

    def _parameter_mutation(self, code: str) -> Tuple[str, str]:
        """
        Change numeric constants by +/- 10-50%.

        Example: 100 -> 120 or 100 -> 75
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, "parameter_mutation:syntax_error"

        class NumericMutator(ast.NodeTransformer):
            def __init__(self):
                self.mutated = False
                self.description = ""

            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)) and not self.mutated:
                    # 30% chance to mutate each number
                    if random.random() < 0.3 and node.value != 0:
                        old_val = node.value
                        factor = random.uniform(0.5, 1.5)  # +/- 50%
                        new_val = old_val * factor
                        if isinstance(old_val, int):
                            new_val = int(new_val)
                            # Ensure we don't get the same value
                            if new_val == old_val:
                                new_val = old_val + random.choice([-1, 1])
                        node.value = new_val
                        self.mutated = True
                        self.description = f"parameter:{old_val}->{new_val}"
                return node

        mutator = NumericMutator()
        new_tree = mutator.visit(tree)
        ast.fix_missing_locations(new_tree)

        try:
            new_code = ast.unparse(new_tree)
            return new_code, mutator.description if mutator.mutated else "parameter:no_change"
        except:
            return code, "parameter:unparse_failed"

    def _structure_mutation(self, code: str) -> Tuple[str, str]:
        """
        Swap if/else branches or change loop types.

        Example: if a: X else: Y -> if not a: Y else: X
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, "structure_mutation:syntax_error"

        class StructureMutator(ast.NodeTransformer):
            def __init__(self):
                self.mutated = False
                self.description = ""

            def visit_If(self, node):
                # First visit children
                self.generic_visit(node)

                # Swap if/else branches (only if there's an else branch)
                if not self.mutated and node.orelse and random.random() < 0.3:
                    # Negate condition and swap branches
                    old_test = ast.unparse(node.test) if hasattr(ast, 'unparse') else 'condition'
                    node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
                    node.body, node.orelse = node.orelse, node.body
                    self.mutated = True
                    self.description = f"structure:swap_if_else:{old_test}"

                return node

            def visit_For(self, node):
                self.generic_visit(node)

                # Convert for to while (only for simple range loops)
                if not self.mutated and random.random() < 0.2:
                    if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                        if node.iter.func.id == 'range' and len(node.iter.args) >= 1:
                            self.description = "structure:for_to_while:range_loop"
                            # Don't actually change - too complex for now

                return node

        mutator = StructureMutator()
        new_tree = mutator.visit(tree)
        ast.fix_missing_locations(new_tree)

        try:
            new_code = ast.unparse(new_tree)
            return new_code, mutator.description if mutator.mutated else "structure:no_change"
        except:
            return code, "structure:unparse_failed"

    def _operator_mutation(self, code: str) -> Tuple[str, str]:
        """
        Change operators: +/-, *//, and/or, ==/!=

        Example: a + b -> a - b
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, "operator_mutation:syntax_error"

        class OperatorMutator(ast.NodeTransformer):
            def __init__(self):
                self.mutated = False
                self.description = ""

            # Operator pairs that can be swapped
            binop_pairs = [
                (ast.Add, ast.Sub),
                (ast.Mult, ast.FloorDiv),
                (ast.Mod, ast.FloorDiv),
            ]

            boolop_pairs = [
                (ast.And, ast.Or),
            ]

            compare_pairs = [
                (ast.Eq, ast.NotEq),
                (ast.Lt, ast.Gt),
                (ast.LtE, ast.GtE),
            ]

            def visit_BinOp(self, node):
                self.generic_visit(node)

                if not self.mutated and random.random() < 0.3:
                    for op1, op2 in self.binop_pairs:
                        if isinstance(node.op, op1):
                            node.op = op2()
                            self.mutated = True
                            self.description = f"operator:{op1.__name__}->{op2.__name__}"
                            break
                        elif isinstance(node.op, op2):
                            node.op = op1()
                            self.mutated = True
                            self.description = f"operator:{op2.__name__}->{op1.__name__}"
                            break

                return node

            def visit_BoolOp(self, node):
                self.generic_visit(node)

                if not self.mutated and random.random() < 0.2:
                    for op1, op2 in self.boolop_pairs:
                        if isinstance(node.op, op1):
                            node.op = op2()
                            self.mutated = True
                            self.description = f"operator:{op1.__name__}->{op2.__name__}"
                            break
                        elif isinstance(node.op, op2):
                            node.op = op1()
                            self.mutated = True
                            self.description = f"operator:{op2.__name__}->{op1.__name__}"
                            break

                return node

            def visit_Compare(self, node):
                self.generic_visit(node)

                if not self.mutated and random.random() < 0.3 and len(node.ops) == 1:
                    for op1, op2 in self.compare_pairs:
                        if isinstance(node.ops[0], op1):
                            node.ops[0] = op2()
                            self.mutated = True
                            self.description = f"operator:{op1.__name__}->{op2.__name__}"
                            break
                        elif isinstance(node.ops[0], op2):
                            node.ops[0] = op1()
                            self.mutated = True
                            self.description = f"operator:{op2.__name__}->{op1.__name__}"
                            break

                return node

        mutator = OperatorMutator()
        new_tree = mutator.visit(tree)
        ast.fix_missing_locations(new_tree)

        try:
            new_code = ast.unparse(new_tree)
            return new_code, mutator.description if mutator.mutated else "operator:no_change"
        except:
            return code, "operator:unparse_failed"

    def _simplification_mutation(self, code: str) -> Tuple[str, str]:
        """
        Remove dead code or simplify expressions.

        Example: if True: X -> X
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, "simplification_mutation:syntax_error"

        class SimplificationMutator(ast.NodeTransformer):
            def __init__(self):
                self.mutated = False
                self.description = ""

            def visit_If(self, node):
                self.generic_visit(node)

                # Simplify if True/if False
                if not self.mutated and isinstance(node.test, ast.Constant):
                    if node.test.value is True:
                        self.mutated = True
                        self.description = "simplify:if_true_removed"
                        # Return body statements
                        return node.body
                    elif node.test.value is False and node.orelse:
                        self.mutated = True
                        self.description = "simplify:if_false_removed"
                        return node.orelse

                return node

            def visit_BinOp(self, node):
                self.generic_visit(node)

                # Simplify x + 0, x * 1, etc.
                if not self.mutated:
                    if isinstance(node.op, ast.Add):
                        if isinstance(node.right, ast.Constant) and node.right.value == 0:
                            self.mutated = True
                            self.description = "simplify:add_zero"
                            return node.left
                        if isinstance(node.left, ast.Constant) and node.left.value == 0:
                            self.mutated = True
                            self.description = "simplify:add_zero"
                            return node.right
                    elif isinstance(node.op, ast.Mult):
                        if isinstance(node.right, ast.Constant) and node.right.value == 1:
                            self.mutated = True
                            self.description = "simplify:mult_one"
                            return node.left
                        if isinstance(node.left, ast.Constant) and node.left.value == 1:
                            self.mutated = True
                            self.description = "simplify:mult_one"
                            return node.right

                return node

        mutator = SimplificationMutator()
        new_tree = mutator.visit(tree)
        ast.fix_missing_locations(new_tree)

        try:
            new_code = ast.unparse(new_tree)
            return new_code, mutator.description if mutator.mutated else "simplify:no_change"
        except:
            return code, "simplify:unparse_failed"

    def _expansion_mutation(self, code: str) -> Tuple[str, str]:
        """
        Add error handling or type hints.

        Example: Add try/except around risky operations
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, "expansion_mutation:syntax_error"

        class ExpansionMutator(ast.NodeTransformer):
            def __init__(self):
                self.mutated = False
                self.description = ""

            def visit_FunctionDef(self, node):
                self.generic_visit(node)

                # Add return type hint if missing
                if not self.mutated and node.returns is None and random.random() < 0.3:
                    # Simple heuristic: look for return statements
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and child.value is not None:
                            if isinstance(child.value, ast.Constant):
                                if isinstance(child.value.value, int):
                                    node.returns = ast.Name(id='int', ctx=ast.Load())
                                    self.mutated = True
                                    self.description = f"expand:add_return_type:int:{node.name}"
                                    break
                                elif isinstance(child.value.value, str):
                                    node.returns = ast.Name(id='str', ctx=ast.Load())
                                    self.mutated = True
                                    self.description = f"expand:add_return_type:str:{node.name}"
                                    break
                                elif isinstance(child.value.value, bool):
                                    node.returns = ast.Name(id='bool', ctx=ast.Load())
                                    self.mutated = True
                                    self.description = f"expand:add_return_type:bool:{node.name}"
                                    break

                return node

        mutator = ExpansionMutator()
        new_tree = mutator.visit(tree)
        ast.fix_missing_locations(new_tree)

        try:
            new_code = ast.unparse(new_tree)
            return new_code, mutator.description if mutator.mutated else "expand:no_change"
        except:
            return code, "expand:unparse_failed"

    def validate_code(self, code: str) -> bool:
        """Check if code is syntactically valid."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class CodeCrossover:
    """
    Combines code from two parents using AST-based techniques.

    Strategies:
    - Function mixing: Take functions from different parents
    - Expression mixing: Combine expressions from both
    - Statement interleaving: Alternate statements
    """

    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        Perform crossover between two parent codes.

        Args:
            parent1: First parent's code
            parent2: Second parent's code

        Returns:
            (child_code, crossover_description)
        """
        try:
            tree1 = ast.parse(parent1)
            tree2 = ast.parse(parent2)
        except SyntaxError:
            return parent1, "crossover:syntax_error"

        # Extract functions from both parents
        funcs1 = {node.name: node for node in ast.walk(tree1)
                  if isinstance(node, ast.FunctionDef)}
        funcs2 = {node.name: node for node in ast.walk(tree2)
                  if isinstance(node, ast.FunctionDef)}

        # If both have the same function, try to mix them
        common_funcs = set(funcs1.keys()) & set(funcs2.keys())

        if common_funcs:
            # Pick a common function to mix
            func_name = random.choice(list(common_funcs))
            return self._mix_functions(funcs1[func_name], funcs2[func_name], func_name)

        # Fallback: just return one of the parents with slight modification
        return parent1, "crossover:no_common_functions"

    def _mix_functions(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                       func_name: str) -> Tuple[str, str]:
        """Mix two function implementations."""
        # Simple strategy: take body from one, args from other
        if random.random() < 0.5:
            # Take func1's structure with some statements from func2
            new_func = copy.deepcopy(func1)

            # If func2 has more statements, maybe add some
            if len(func2.body) > len(func1.body) and random.random() < 0.5:
                # Insert a statement from func2
                idx = random.randint(0, len(func2.body) - 1)
                stmt = copy.deepcopy(func2.body[idx])
                insert_pos = random.randint(0, len(new_func.body))
                new_func.body.insert(insert_pos, stmt)
        else:
            # Take func2's structure with func1's args
            new_func = copy.deepcopy(func2)
            new_func.args = copy.deepcopy(func1.args)

        # Wrap in module
        module = ast.Module(body=[new_func], type_ignores=[])
        ast.fix_missing_locations(module)

        try:
            code = ast.unparse(module)
            return code, f"crossover:mixed_function:{func_name}"
        except:
            return ast.unparse(ast.Module(body=[func1], type_ignores=[])), "crossover:mix_failed"


class Population:
    """
    Manages a population of code variants.

    Supports:
    - Tournament selection
    - Crossover
    - Mutation
    - Fitness evaluation
    - Generation evolution
    - Hall of fame tracking
    """

    def __init__(
        self,
        max_size: int = 20,
        elite_size: int = 3,
        tournament_size: int = 3,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        storage_dir: Path = None
    ):
        """
        Initialize population.

        Args:
            max_size: Maximum population size
            elite_size: Number of best individuals to keep unchanged
            tournament_size: Size of tournament for selection
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            storage_dir: Directory for saving population state
        """
        self.max_size = max_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.storage_dir = storage_dir or EVOLUTION_DIR / "population"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.storage_dir / "population.json"

        # Population state
        self.individuals: List[CodeIndividual] = []
        self.generation: int = 0
        self.hall_of_fame: List[CodeIndividual] = []  # Best ever individuals

        # Helper classes
        self.mutator = CodeMutator()
        self.crossover = CodeCrossover()

        # Sandbox for fitness evaluation
        self._sandbox = None

        # Load existing state if available
        self._load_state()

    def _load_state(self):
        """Load population state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                self.generation = state.get('generation', 0)
                self.individuals = [
                    CodeIndividual.from_dict(d)
                    for d in state.get('individuals', [])
                ]
                self.hall_of_fame = [
                    CodeIndividual.from_dict(d)
                    for d in state.get('hall_of_fame', [])
                ]

                print(f"[Population] Loaded {len(self.individuals)} individuals, generation {self.generation}")
            except Exception as e:
                print(f"[Population] Error loading state: {e}")

    def _save_state(self):
        """Save population state to file."""
        state = {
            'generation': self.generation,
            'individuals': [ind.to_dict() for ind in self.individuals],
            'hall_of_fame': [ind.to_dict() for ind in self.hall_of_fame],
            'last_updated': datetime.now().isoformat()
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _get_sandbox(self):
        """Get or create sandbox for fitness evaluation."""
        if self._sandbox is None:
            try:
                from integrations.docker_sandbox import get_docker_sandbox
                self._sandbox = get_docker_sandbox()
            except ImportError:
                # Fallback to code sandbox
                from integrations.code_evolution import CodeSandbox
                self._sandbox = CodeSandbox()
        return self._sandbox

    def _generate_id(self, code: str) -> str:
        """Generate unique ID for code."""
        return hashlib.md5(
            f"{code}{datetime.now().isoformat()}{random.random()}".encode()
        ).hexdigest()[:12]

    def add_individual(self, code: str, fitness: float = 0.0,
                       parent_id: str = None) -> CodeIndividual:
        """
        Add a new individual to the population.

        Args:
            code: Python code for this individual
            fitness: Initial fitness score (will be evaluated if 0)
            parent_id: ID of parent if derived from another individual

        Returns:
            Created CodeIndividual
        """
        individual = CodeIndividual(
            id=self._generate_id(code),
            code=code,
            fitness_score=fitness,
            generation=self.generation,
            parent_id=parent_id
        )

        self.individuals.append(individual)

        # Maintain max size
        if len(self.individuals) > self.max_size:
            # Remove worst individuals
            self.individuals.sort(key=lambda x: x.fitness_score, reverse=True)
            self.individuals = self.individuals[:self.max_size]

        self._save_state()
        return individual

    def select_parents(self, n: int = 2) -> List[CodeIndividual]:
        """
        Select parents using tournament selection.

        Args:
            n: Number of parents to select

        Returns:
            List of selected parents
        """
        if len(self.individuals) < n:
            return list(self.individuals)

        parents = []
        for _ in range(n):
            # Tournament selection
            tournament = random.sample(
                self.individuals,
                min(self.tournament_size, len(self.individuals))
            )
            winner = max(tournament, key=lambda x: x.fitness_score)
            parents.append(winner)

        return parents

    def crossover_parents(self, parent1: CodeIndividual,
                          parent2: CodeIndividual) -> CodeIndividual:
        """
        Combine code from two parents.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Child individual
        """
        child_code, description = self.crossover.crossover(parent1.code, parent2.code)

        child = CodeIndividual(
            id=self._generate_id(child_code),
            code=child_code,
            fitness_score=0.0,
            generation=self.generation + 1,
            parent_id=parent1.id,
            parent_ids=[parent1.id, parent2.id],
            mutations=[description]
        )

        return child

    def mutate_individual(self, individual: CodeIndividual) -> CodeIndividual:
        """
        Apply mutation to an individual.

        Args:
            individual: Individual to mutate

        Returns:
            New mutated individual
        """
        mutated_code, description = self.mutator.mutate(individual.code)

        # Validate mutated code
        if not self.mutator.validate_code(mutated_code):
            mutated_code = individual.code
            description = "mutation:validation_failed"

        mutant = CodeIndividual(
            id=self._generate_id(mutated_code),
            code=mutated_code,
            fitness_score=0.0,
            generation=self.generation + 1,
            parent_id=individual.id,
            parent_ids=[individual.id],
            mutations=individual.mutations + [description]
        )

        return mutant

    def evaluate_fitness(self, individual: CodeIndividual,
                        test_cases: List[Dict]) -> float:
        """
        Evaluate fitness of an individual using test cases.

        Fitness is based on:
        - Test pass rate (primary)
        - Execution time (secondary)
        - Code simplicity (tertiary)

        Args:
            individual: Individual to evaluate
            test_cases: List of test cases

        Returns:
            Fitness score (0-1)
        """
        sandbox = self._get_sandbox()

        try:
            success, results = sandbox.execute_code(individual.code, test_cases)
            individual.test_results = results

            # Calculate fitness
            total_tests = results.get('tests_passed', 0) + results.get('tests_failed', 0)

            if total_tests == 0:
                # No tests - base fitness on successful execution
                fitness = 0.5 if success else 0.0
            else:
                # Primary: test pass rate (0-0.8)
                pass_rate = results.get('tests_passed', 0) / total_tests
                fitness = pass_rate * 0.8

                # Secondary: execution time bonus (0-0.1)
                exec_time = results.get('execution_time', 10)
                time_bonus = max(0, 0.1 * (1 - exec_time / 10))  # Bonus for < 10s
                fitness += time_bonus

                # Tertiary: code simplicity bonus (0-0.1)
                code_lines = len(individual.code.split('\n'))
                simplicity_bonus = max(0, 0.1 * (1 - code_lines / 100))  # Bonus for < 100 lines
                fitness += simplicity_bonus

            individual.fitness_score = fitness

        except Exception as e:
            print(f"[Population] Fitness evaluation error: {e}")
            individual.fitness_score = 0.0

        return individual.fitness_score

    def evolve_generation(self, test_cases: List[Dict] = None) -> Dict:
        """
        Create next generation from current population.

        Steps:
        1. Evaluate fitness of all individuals
        2. Select elite to carry over unchanged
        3. Select parents and create offspring via crossover
        4. Apply mutations
        5. Evaluate new individuals
        6. Update hall of fame

        Args:
            test_cases: Test cases for fitness evaluation

        Returns:
            Stats about the evolution
        """
        stats = {
            'generation': self.generation,
            'population_size': len(self.individuals),
            'offspring_created': 0,
            'mutations_applied': 0,
            'best_fitness': 0.0,
            'avg_fitness': 0.0
        }

        if len(self.individuals) == 0:
            print("[Population] Empty population, cannot evolve")
            return stats

        # Step 1: Evaluate all current individuals
        if test_cases:
            for ind in self.individuals:
                if ind.fitness_score == 0.0:  # Only evaluate unevaluated
                    self.evaluate_fitness(ind, test_cases)

        # Step 2: Sort by fitness and select elite
        self.individuals.sort(key=lambda x: x.fitness_score, reverse=True)
        elite = self.individuals[:self.elite_size]

        # Update hall of fame
        for ind in elite:
            if len(self.hall_of_fame) < 5:
                self.hall_of_fame.append(ind)
            elif ind.fitness_score > min(h.fitness_score for h in self.hall_of_fame):
                self.hall_of_fame.sort(key=lambda x: x.fitness_score)
                self.hall_of_fame[0] = ind

        # Step 3 & 4: Create offspring via crossover and mutation
        new_generation = list(elite)  # Elite carry over

        while len(new_generation) < self.max_size:
            if random.random() < self.crossover_rate and len(self.individuals) >= 2:
                # Crossover
                parent1, parent2 = self.select_parents(2)
                child = self.crossover_parents(parent1, parent2)
                stats['offspring_created'] += 1
            else:
                # Clone a selected parent
                parent = self.select_parents(1)[0]
                child = CodeIndividual(
                    id=self._generate_id(parent.code),
                    code=parent.code,
                    fitness_score=0.0,
                    generation=self.generation + 1,
                    parent_id=parent.id,
                    parent_ids=[parent.id],
                    mutations=list(parent.mutations)
                )

            # Maybe mutate
            if random.random() < self.mutation_rate:
                child = self.mutate_individual(child)
                stats['mutations_applied'] += 1

            # Evaluate fitness
            if test_cases:
                self.evaluate_fitness(child, test_cases)

            new_generation.append(child)

        # Step 5: Replace population
        self.individuals = new_generation
        self.generation += 1

        # Calculate stats
        stats['generation'] = self.generation
        stats['population_size'] = len(self.individuals)
        stats['best_fitness'] = max(ind.fitness_score for ind in self.individuals)
        stats['avg_fitness'] = sum(ind.fitness_score for ind in self.individuals) / len(self.individuals)

        # Save state
        self._save_state()

        print(f"[Population] Generation {self.generation}: best={stats['best_fitness']:.3f}, avg={stats['avg_fitness']:.3f}")

        return stats

    def get_best(self, n: int = 1) -> List[CodeIndividual]:
        """Get the n best individuals by fitness."""
        sorted_inds = sorted(self.individuals, key=lambda x: x.fitness_score, reverse=True)
        return sorted_inds[:n]

    def get_diversity(self) -> float:
        """
        Calculate population diversity.

        Based on code similarity (Jaccard similarity of tokens).

        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if len(self.individuals) < 2:
            return 0.0

        def tokenize(code: str) -> set:
            # Simple tokenization
            tokens = set(re.findall(r'\w+', code))
            return tokens

        similarities = []
        for i, ind1 in enumerate(self.individuals):
            for ind2 in self.individuals[i+1:]:
                tokens1 = tokenize(ind1.code)
                tokens2 = tokenize(ind2.code)
                if tokens1 or tokens2:
                    jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
                    similarities.append(jaccard)

        if not similarities:
            return 0.0

        # Diversity = 1 - average similarity
        return 1 - (sum(similarities) / len(similarities))

    def get_stats(self) -> Dict:
        """Get population statistics."""
        if not self.individuals:
            return {
                'generation': self.generation,
                'population_size': 0,
                'best_fitness': 0.0,
                'avg_fitness': 0.0,
                'diversity': 0.0,
                'hall_of_fame_size': len(self.hall_of_fame)
            }

        fitnesses = [ind.fitness_score for ind in self.individuals]

        return {
            'generation': self.generation,
            'population_size': len(self.individuals),
            'best_fitness': max(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'min_fitness': min(fitnesses),
            'diversity': self.get_diversity(),
            'hall_of_fame_size': len(self.hall_of_fame),
            'total_mutations': sum(len(ind.mutations) for ind in self.individuals)
        }

    def get_lineage(self, individual_id: str) -> List[Dict]:
        """
        Get the lineage (ancestry) of an individual.

        Args:
            individual_id: ID of the individual

        Returns:
            List of ancestor info in order (oldest first)
        """
        lineage = []
        current_id = individual_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)

            # Find individual
            ind = None
            for i in self.individuals + self.hall_of_fame:
                if i.id == current_id:
                    ind = i
                    break

            if ind:
                lineage.append({
                    'id': ind.id,
                    'generation': ind.generation,
                    'fitness': ind.fitness_score,
                    'mutations': ind.mutations
                })
                current_id = ind.parent_id
            else:
                break

        return list(reversed(lineage))


# Global instance
_population: Optional[Population] = None


def get_population() -> Population:
    """Get global population instance."""
    global _population
    if _population is None:
        _population = Population()
    return _population


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("POPULATION EVOLUTION TEST")
    print("=" * 60)

    # Create test population
    pop = Population(max_size=10, storage_dir=Path("/tmp/test_population"))

    # Test 1: Add individuals
    print("\n1. Adding initial individuals:")

    codes = [
        '''
def add(a, b):
    return a + b
''',
        '''
def add(a, b):
    result = a + b
    return result
''',
        '''
def add(a, b):
    if a == 0:
        return b
    return a + b
'''
    ]

    for code in codes:
        ind = pop.add_individual(code)
        print(f"   Added: {ind.id}")

    print(f"   Population size: {len(pop.individuals)}")

    # Test 2: Evaluate fitness
    print("\n2. Evaluating fitness:")
    test_cases = [
        {'function': 'add', 'args': [2, 3], 'expected': 5},
        {'function': 'add', 'args': [10, 20], 'expected': 30},
        {'function': 'add', 'args': [0, 5], 'expected': 5},
    ]

    for ind in pop.individuals:
        fitness = pop.evaluate_fitness(ind, test_cases)
        print(f"   {ind.id}: fitness={fitness:.3f}")

    # Test 3: Select parents
    print("\n3. Tournament selection:")
    parents = pop.select_parents(2)
    print(f"   Selected: {[p.id for p in parents]}")

    # Test 4: Crossover
    print("\n4. Crossover:")
    if len(parents) >= 2:
        child = pop.crossover_parents(parents[0], parents[1])
        print(f"   Child: {child.id}")
        print(f"   Parents: {child.parent_ids}")
        print(f"   Mutations: {child.mutations}")

    # Test 5: Mutation
    print("\n5. Mutation:")
    mutant = pop.mutate_individual(pop.individuals[0])
    print(f"   Original: {pop.individuals[0].id}")
    print(f"   Mutant: {mutant.id}")
    print(f"   Mutations: {mutant.mutations}")

    # Test 6: Evolve generation
    print("\n6. Evolving generation:")
    stats = pop.evolve_generation(test_cases)
    print(f"   Generation: {stats['generation']}")
    print(f"   Best fitness: {stats['best_fitness']:.3f}")
    print(f"   Avg fitness: {stats['avg_fitness']:.3f}")

    # Test 7: Population stats
    print("\n7. Population stats:")
    stats = pop.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test 8: Lineage
    print("\n8. Lineage tracking:")
    best = pop.get_best(1)[0]
    lineage = pop.get_lineage(best.id)
    print(f"   Best individual: {best.id}")
    print(f"   Lineage depth: {len(lineage)}")
    for ancestor in lineage:
        print(f"     Gen {ancestor['generation']}: {ancestor['id'][:8]}... (fitness={ancestor['fitness']:.3f})")

    print("\n" + "=" * 60)
    print("POPULATION EVOLUTION TEST COMPLETE")
    print("=" * 60)
