# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import random
from typing import List, Tuple
import time
import numba
from numba import jit
import math
from joblib import Parallel, delayed
from scipy import optimize

@jit(nopython=True)
def fast_convolve_jit(a, b):
    """Fast convolution using Numba JIT compilation."""
    n = len(a)
    m = len(b)
    result = np.zeros(n + m - 1)
    for i in range(n):
        for j in range(m):
            result[i + j] += a[i] * b[j]
    return result

class FastAutocorrelationEvaluator:
    """Efficient evaluator for autocorrelation constants using FFT with caching."""

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
        if n > 500:
            try:
                conv = fftconvolve(sequence, sequence, mode='full')
            except Exception:
                # Fallback to JIT for large sequences if FFT fails
                conv = fast_convolve_jit(sequence, sequence)
        else:
            conv = fast_convolve_jit(sequence, sequence)
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
_evaluator = FastAutocorrelationEvaluator()

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

def mutate_sequence(sequence: List[float], mutation_rate=0.1, mutation_strength=0.3) -> List[float]:
    """Apply mutation to a sequence with specified rate and strength."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply Gaussian noise scaled by mutation strength
            noise = random.gauss(0, mutation_strength * mutated[i])
            mutated[i] = max(0.01, mutated[i] + noise)
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

def evaluate_fitness_single(sequence: List[float]) -> float:
    """Evaluate fitness for a single sequence."""
    _, inv_c1 = compute_autocorrelation_constant(sequence)
    return inv_c1

def evaluate_fitness_batch_parallel(sequences: List[List[float]]) -> List[float]:
    """Evaluate multiple sequences in parallel for better performance."""
    fitness_scores = Parallel(n_jobs=-1)(delayed(evaluate_fitness_single)(seq) for seq in sequences)
    return fitness_scores

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    try:
        n = len(f_sequence)
        c = -np.ones(n)
        a_ub = []
        b_ub = []
        for k in range(2 * n - 1):
            row = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_sequence[i]
            a_ub.append(row)
            b_ub.append(rhs)

        # Non-negativity constraints: b_i >= 0
        a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
        b_ub_nonneg = np.zeros(n)  # Zero vector

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except Exception:
        return None

def get_good_direction_to_move_into(sequence: List[float]) -> List[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    if sum_sequence < 1e-10:
        return None
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    try:
        conv_result = np.convolve(normalized_sequence, normalized_sequence)
        rhs = np.max(conv_result)
    except Exception:
        return None
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def optimize_step_function_evolutionary(max_time_seconds=170) -> List[float]:
    """
    Evolutionary optimization to find optimal step function that maximizes 1/C1.
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

    while time.time() - start_time < max_time_seconds and stagnation_count < max_stagnation:
        generation += 1

        # Evaluate fitness (1/C1) of each individual in batches
        fitness_scores = evaluate_fitness_batch_parallel(population)

        # Track best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_inv_c1 = fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Try to improve best sequence using LP-based direction
        if best_sequence is not None:
            improved = get_good_direction_to_move_into(best_sequence)
            if improved is not None:
                _, improved_inv_c1 = compute_autocorrelation_constant(improved)
                if improved_inv_c1 > best_inv_c1:
                    best_sequence = improved
                    best_inv_c1 = improved_inv_c1

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

        # Create new population through crossover and mutation
        new_population = [best_sequence.copy()]  # Elitism: keep best individual

        while len(new_population) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation with adaptive rate based on generation
            adaptive_mutation_rate = 0.15 * (1 - generation / 100)  # Decreases over time
            child = mutate_sequence(child, mutation_rate=adaptive_mutation_rate, mutation_strength=0.2)

            new_population.append(child)

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
        # Use evolutionary optimization approach
        best_sequence = optimize_step_function_evolutionary()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        # Fallback to simple random approach
        return generate_random_valid_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")