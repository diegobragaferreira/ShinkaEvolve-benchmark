# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import random
from typing import List, Tuple
import time
import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.optimize import minimize
import math

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

# Compile the convolution function for performance
@jit
def fast_convolve_jit_jax(a, b):
    """Fast convolution using JAX JIT compilation."""
    n = len(a)
    m = len(b)
    result = jnp.zeros(n + m - 1)
    for i in range(n):
        for j in range(m):
            result = result.at[i + j].add(a[i] * b[j])
    return result

class GradientGuidedEvaluator:
    """Evaluator with gradient information and caching."""

    def __init__(self):
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(self, sequence: List[float]) -> Tuple[float, float]:
        """
        Computes the autocorrelation constant C1 and its reciprocal 1/C1.
        Uses caching and FFT convolution for efficiency.
        """
        # Create a hashable representation for caching
        seq_tuple = tuple(sequence)
        if seq_tuple in self._cache:
            self._cache_hits += 1
            return self._cache[seq_tuple]

        self._cache_misses += 1

        if not sequence or sum(sequence) < 0.01:
            result = (float('inf'), 0.0)
            self._cache[seq_tuple] = result
            return result

        n = len(sequence)
        # Use FFT-based convolution for efficiency O(n log n)
        conv = fftconvolve(sequence, sequence, mode='full')
        max_conv = np.max(conv)
        sum_seq = sum(sequence)

        if sum_seq == 0:
            result = (float('inf'), 0.0)
            self._cache[seq_tuple] = result
            return result

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        result = (c1, inv_c1)
        self._cache[seq_tuple] = result
        return result

# Global evaluator instance
_evaluator = GradientGuidedEvaluator()

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 and 1/C1 using global cached evaluator."""
    return _evaluator.evaluate(sequence)

def generate_random_valid_sequence(length_range=(50, 500), method='mixed') -> List[float]:
    """Generate a random valid sequence within specified length range."""
    n = random.randint(*length_range)

    if method == 'step':
        # Generate step function with varied heights
        num_steps = max(2, min(20, n // 10))
        step_positions = sorted(random.sample(range(n), num_steps))
        step_heights = [random.uniform(0.1, 100.0) for _ in range(num_steps)]

        sequence = [0.0] * n
        for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
            if i < len(step_positions) - 1:
                end_pos = step_positions[i+1]
            else:
                end_pos = n
            sequence[pos:end_pos] = [height] * (end_pos - pos)

    elif method == 'gaussian':
        # Generate Gaussian-like distribution
        sequence = [random.gauss(50.0, 20.0) for _ in range(n)]
        sequence = [max(0.01, x) for x in sequence]

    else:  # default 'uniform'
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]

    return sequence

def generate_diverse_population(population_size: int, length_range=(50, 500)) -> List[List[float]]:
    """Generate a diverse initial population."""
    population = []

    # Add some step-function examples to encourage structure finding
    for _ in range(population_size // 4):
        population.append(generate_random_valid_sequence(length_range, 'step'))

    # Add some Gaussian examples
    for _ in range(population_size // 4):
        population.append(generate_random_valid_sequence(length_range, 'gaussian'))

    # Fill remaining with standard random
    while len(population) < population_size:
        population.append(generate_random_valid_sequence(length_range, 'uniform'))

    return population

def evaluate_fitness_batch(sequences: List[List[float]]) -> List[float]:
    """Evaluate multiple sequences at once for better performance."""
    fitness_scores = []
    for seq in sequences:
        _, inv_c1 = compute_autocorrelation_constant(seq)
        fitness_scores.append(inv_c1)
    return fitness_scores

def adaptive_mutation(sequence: List[float], generation: int, max_generations: int) -> List[float]:
    """Adaptive mutation with dynamic step size."""
    mutated = sequence.copy()
    # Dynamic mutation rate based on generation
    mutation_rate = max(0.01, 0.3 * (1 - generation / max_generations))
    # Dynamic step size
    step_size = 0.1 * (1 - generation / max_generations) + 0.01
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply Gaussian noise scaled by step size
            noise = random.gauss(0, step_size * mutated[i])
            mutated[i] = max(0.01, mutated[i] + noise)
    return mutated

@jit
def _compute_gradient(sequence_array, target):
    """JAX-based gradient computation."""
    @jit
    def loss_fn(seq):
        conv = fast_convolve_jit_jax(seq, seq)
        max_conv = jnp.max(conv)
        sum_seq = jnp.sum(seq)
        c1 = 2 * len(seq) * max_conv / (sum_seq ** 2)
        return c1
    
    gradient = grad(loss_fn)(sequence_array)
    return gradient

def gradient_guided_mutation(sequence: List[float], target_sequence: List[float], 
                           generation: int, max_generations: int) -> List[float]:
    """Use gradient to guide mutation."""
    mutated = sequence.copy()
    # Compute gradient
    try:
        seq_array = jnp.array(mutated)
        grad_array = _compute_gradient(seq_array, target_sequence)
        # Apply gradient direction with adaptive step size
        step_size = 0.01 * (1 - generation / max_generations) + 0.001
        mutated = [max(0.01, x - step_size * grad_array[i]) for i, x in enumerate(mutated)]
    except:
        # Fallback to regular mutation if gradient computation fails
        mutated = adaptive_mutation(mutated, generation, max_generations)
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if seq1 else seq2

    # Single-point crossover with bias towards preserving better parts
    crossover_point = random.randint(1, min_len - 1)
    child = seq1[:crossover_point] + seq2[crossover_point:]

    # Ensure minimum positive value for all elements
    child = [max(0.01, x) for x in child]
    return child

def diversity_preservation(population: List[List[float]], fitness_scores: List[float], 
                          num_keep: int = 5) -> List[List[float]]:
    """Keep diverse individuals in the population."""
    # Sort by fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]
    # Keep top performers
    preserved = [population[i] for i in sorted_indices[:num_keep]]
    # Add diverse individuals based on pairwise distance
    for i in range(len(population)):
        if len(preserved) >= len(population):
            break
        # Simple diversity metric: sum of squared differences from top individuals
        distances = []
        for p in preserved:
            dist = sum((a - b)**2 for a, b in zip(population[i], p))
            distances.append(dist)
        if min(distances) > 100:  # Threshold for diversity
            preserved.append(population[i])
    return preserved

def optimize_step_function_gradient_guided(max_time_seconds=170) -> List[float]:
    """
    Gradient-guided evolutionary optimization to find optimal sequence that maximizes 1/C1.
    """
    start_time = time.time()
    _evaluator.clear_cache()

    # Initialize population with more diverse strategies
    population_size = 50
    population = generate_diverse_population(population_size, (100, 1000))

    best_sequence = None
    best_inv_c1 = 0.0

    generation = 0
    stagnation_count = 0
    max_stagnation = 40
    max_generations = 100

    while time.time() - start_time < max_time_seconds and stagnation_count < max_stagnation:
        generation += 1

        # Evaluate fitness (1/C1) of each individual
        fitness_scores = evaluate_fitness_batch(population)

        # Track best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_inv_c1 = fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Selection with tournament selection and elitism
        selected_parents = []
        tournament_size = 5  # Larger tournament for more selection pressure

        # Elitism: keep the top performer
        elite_idx = current_best_idx
        selected_parents.append(population[elite_idx].copy())

        # Tournament selection for rest
        for _ in range(population_size - 1):  # -1 because we already added elite
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner_idx].copy())

        # Create new population through crossover, mutation and gradient refinement
        new_population = [best_sequence.copy()]  # Elitism: keep best individual

        while len(new_population) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Apply gradient-guided mutation
            child = gradient_guided_mutation(child, best_sequence, generation, max_generations)

            # Ensure validity
            child = [max(0.01, x) for x in child]
            new_population.append(child)

        # Preserve diversity
        new_population = diversity_preservation(new_population, fitness_scores)
        # Fill up to full population size
        while len(new_population) < population_size:
            new_population.append(generate_random_valid_sequence())

        population = new_population[:population_size]

    # Final cleanup and validation
    if best_sequence is not None:
        # Normalize sequence to make sure it's valid
        sum_seq = sum(best_sequence)
        if sum_seq > 0.01:
            best_sequence = [x / sum_seq * 100 for x in best_sequence]

    return best_sequence if best_sequence else generate_random_valid_sequence()

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use gradient-guided evolutionary optimization approach
        best_sequence = optimize_step_function_gradient_guided()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        # Fallback to simple random approach
        return generate_random_valid_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")