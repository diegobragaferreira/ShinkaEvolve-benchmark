# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
from typing import List, Tuple
import time
import numba
from numba import jit
from joblib import Parallel, delayed
from collections import OrderedDict
import copy

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class LruCache:
    def __init__(self, maxsize=256):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value

class FastAutocorrelationEvaluator:
    """Efficient evaluator for autocorrelation constants using FFT with caching."""

    def __init__(self):
        self._cache = LruCache(maxsize=512)
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache = LruCache(maxsize=512)
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(self, sequence: List[float]) -> Tuple[float, float]:
        """
        Computes the autocorrelation constant C1 and its reciprocal 1/C1.
        Uses caching and FFT convolution for efficiency.
        """
        # Create a hashable representation for caching
        seq_tuple = tuple(sequence)
        cached_result = self._cache.get(seq_tuple)
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result

        self._cache_misses += 1

        if not sequence or sum(sequence) < 0.01:
            result = (float('inf'), 0.0)
            self._cache.put(seq_tuple, result)
            return result

        n = len(sequence)
        # Use FFT-based convolution for efficiency O(n log n)
        conv = fftconvolve(sequence, sequence, mode='full')
        max_conv = np.max(conv)
        sum_seq = sum(sequence)

        if sum_seq == 0:
            result = (float('inf'), 0.0)
            self._cache.put(seq_tuple, result)
            return result

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        result = (c1, inv_c1)
        self._cache.put(seq_tuple, result)
        return result

# Global evaluator instance
_evaluator = FastAutocorrelationEvaluator()

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 and 1/C1 using global cached evaluator."""
    return _evaluator.evaluate(sequence)

def compute_fitness_parallel(seqs: List[List[float]]) -> List[float]:
    """Compute fitness scores for a batch of sequences in parallel."""
    return Parallel(n_jobs=-1)(delayed(lambda s: compute_autocorrelation_constant(s)[1])(seq) for seq in seqs)

def generate_step_sequence(length: int, num_steps: int) -> List[float]:
    """Generate a step function sequence."""
    # Create a more structured step sequence
    step_positions = sorted(random.sample(range(length), num_steps)) + [length]
    step_heights = [random.uniform(10.0, 100.0) for _ in range(num_steps)]

    sequence = [0.0] * length
    for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = length
        sequence[pos:end_pos] = [height] * (end_pos - pos)
    return sequence

def generate_diverse_population(population_size: int, length_range=(100, 1000)) -> List[List[float]]:
    """Generate a diverse initial population with various patterns."""
    population = []

    # Generate step functions with varying step sizes and heights
    for _ in range(population_size // 3):
        n = random.randint(*length_range)
        num_steps = max(2, min(30, n // 10))
        population.append(generate_step_sequence(n, num_steps))

    # Add some sequences with exponential decay
    for _ in range(population_size // 3):
        n = random.randint(*length_range)
        sequence = []
        for i in range(n):
            val = 100 * np.exp(-i * 0.02)
            sequence.append(max(0.01, val))
        population.append(sequence)

    # Fill remaining with random sequences
    while len(population) < population_size:
        n = random.randint(*length_range)
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
        population.append(sequence)

    return population

def mutate_sequence(sequence: List[float], generation: int, population_size: int, mutation_strength=0.3) -> List[float]:
    """Apply adaptive mutation to a sequence with decreasing rate over generations."""
    mutated = sequence.copy()
    # Adaptive mutation rate that decreases with generation
    mutation_rate = max(0.05, 0.3 * (1 - generation / (population_size * 2)))
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

    # Single-point crossover
    crossover_point = random.randint(1, min_len - 1)
    child = seq1[:crossover_point] + seq2[crossover_point:]

    # Ensure minimum positive value for all elements
    child = [max(0.01, x) for x in child]
    return child

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence using L-BFGS optimization."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)

    # Avoid division by zero or negligible sums
    if sum_sequence < 1e-10:
        return None

    # Normalize the sequence to avoid numerical issues
    normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence

    # Define objective function to minimize (negative of 1/C1)
    def objective(x):
        # Ensure non-negativity
        x = np.maximum(x, 1e-10)
        x = x / np.sum(x) * sum_sequence  # Renormalize to original sum
        c1 = compute_c1(x)
        if c1 == 0 or np.isnan(c1):
            return float('inf')
        return -1.0 / c1  # Minimize negative of 1/C1 = maximize 1/C1

    # Define gradient function (approximated for now)
    def gradient(x):
        # This is a simplified gradient estimator
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad

    # Use L-BFGS for local optimization
    try:
        result = optimize.minimize(objective, normalized_sequence, method='L-BFGS-B',
                                  bounds=[(1e-10, 1000.0)] * len(normalized_sequence),
                                  jac=gradient, options={'maxiter': 20})
        if result.success:
            optimized_sequence = result.x
            # Renormalize to original sum
            optimized_sequence = optimized_sequence / np.sum(optimized_sequence) * sum_sequence
            return optimized_sequence.tolist()
    except Exception as e:
        # Fall back to previous method if optimization fails
        pass

    # If optimization fails, use the previous approach as fallback
    conv_result = fftconvolve(normalized_sequence, normalized_sequence, mode='full')
    conv_result = conv_result[n-1:2*n-1]
    rhs = np.max(conv_result)

    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None

    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

    t = 0.02
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]

    new_sequence = [max(0, min(x, 1000)) for x in new_sequence]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
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

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except Exception as e:
        return None

def optimize_step_function_evolutionary(max_time_seconds=170) -> List[float]:
    """
    Evolutionary optimization to find optimal step function that maximizes 1/C1.
    """
    start_time = time.time()
    _evaluator.clear_cache()

    # Initialize population with diverse strategies
    population_size = 100  # Increased size for better exploration
    population = generate_diverse_population(population_size, (100, 1000))

    best_sequence = None
    best_inv_c1 = 0.0

    generation = 0
    stagnation_count = 0
    max_stagnation = 30

    while time.time() - start_time < max_time_seconds and stagnation_count < max_stagnation:
        generation += 1

        # Evaluate fitness (1/C1) of each individual in parallel
        fitness_scores = compute_fitness_parallel(population)

        # Track best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_inv_c1 = fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Apply local search to the best sequence
        if best_sequence is not None:
            local_search_result = get_good_direction_to_move_into(best_sequence)
            if local_search_result is not None:
                # Evaluate the local search result
                _, local_inv_c1 = compute_autocorrelation_constant(local_search_result)
                if local_inv_c1 > best_inv_c1:
                    best_inv_c1 = local_inv_c1
                    best_sequence = local_search_result
                    stagnation_count = 0

        # Selection with tournament selection and elitism
        selected_parents = []
        tournament_size = 7  # Larger tournament for more selection pressure

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

            # Mutation with adaptive rate
            child = mutate_sequence(child, generation, population_size, mutation_strength=0.3)

            new_population.append(child)

        population = new_population[:population_size]

    # Final cleanup and validation
    if best_sequence is not None:
        # Normalize sequence to make sure it's valid
        sum_seq = sum(best_sequence)
        if sum_seq > 0.01:
            best_sequence = [x / sum_seq * 100 for x in best_sequence]

    return best_sequence if best_sequence else generate_step_sequence(100, 5)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use evolutionary optimization approach
        best_sequence = optimize_step_function_evolutionary()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        # Fallback to simple step sequence
        return generate_step_sequence(100, 5)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")