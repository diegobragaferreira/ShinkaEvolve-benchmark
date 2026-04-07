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
import collections

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CacheManager:
    """Manages caching for autocorrelation evaluations."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {'hits': self.hits, 'misses': self.misses, 'hit_rate': hit_rate}

# Global cache manager
_cache_manager = CacheManager()

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

class AutocorrelationEvaluator:
    """Efficient evaluator for autocorrelation constants using FFT with caching."""

    def __init__(self):
        self.cache = _cache_manager

    def evaluate(self, sequence: List[float]) -> Tuple[float, float]:
        """
        Computes the autocorrelation constant C1 and its reciprocal 1/C1.
        Uses caching and FFT convolution for efficiency.
        """
        # Create a hashable representation for caching
        seq_tuple = tuple(sequence)
        cached = self.cache.get(seq_tuple)
        if cached is not None:
            return cached

        if not sequence or sum(sequence) < 0.01:
            result = (float('inf'), 0.0)
            self.cache.put(seq_tuple, result)
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
            self.cache.put(seq_tuple, result)
            return result

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        result = (c1, inv_c1)
        self.cache.put(seq_tuple, result)
        return result

# Global evaluator instance
_evaluator = AutocorrelationEvaluator()

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 and 1/C1 using global cached evaluator."""
    return _evaluator.evaluate(sequence)

def generate_step_function(n: int, num_steps: int = None) -> List[float]:
    """Generate a step function with randomly placed steps."""
    if num_steps is None:
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
    return sequence

def generate_gaussian_distribution(n: int) -> List[float]:
    """Generate a Gaussian-like distribution."""
    sequence = [random.gauss(50.0, 20.0) for _ in range(n)]
    return [max(0.01, x) for x in sequence]

def generate_uniform_distribution(n: int) -> List[float]:
    """Generate a uniform distribution."""
    return [random.uniform(0.1, 100.0) for _ in range(n)]

def generate_pattern_based_sequence(n: int) -> List[float]:
    """Generate a sequence with a custom pattern to encourage structure."""
    # Create a combination of peaks and valleys
    sequence = [0.0] * n
    num_peaks = max(2, min(10, n // 50))
    peak_positions = sorted(random.sample(range(n), num_peaks))

    # Assign increasing heights to peaks
    peak_heights = [random.uniform(10, 100) for _ in range(num_peaks)]
    for i, (pos, height) in enumerate(zip(peak_positions, peak_heights)):
        # Set the area around the peak to the height
        radius = max(1, n // 20)
        start = max(0, pos - radius)
        end = min(n, pos + radius)
        for j in range(start, end):
            sequence[j] = height
    return sequence

def generate_diverse_population(population_size: int, length_range=(50, 500)) -> List[List[float]]:
    """Generate a diverse initial population."""
    population = []

    for _ in range(population_size):
        n = random.randint(*length_range)
        method = random.choice(['step', 'gaussian', 'uniform', 'pattern'])
        if method == 'step':
            sequence = generate_step_function(n)
        elif method == 'gaussian':
            sequence = generate_gaussian_distribution(n)
        elif method == 'uniform':
            sequence = generate_uniform_distribution(n)
        else:
            sequence = generate_pattern_based_sequence(n)
        population.append(sequence)
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

def adaptive_mutate_sequence(sequence: List[float], generation: int, max_generations: int,
                           base_mutation_rate: float = 0.15, mutation_strength: float = 0.2) -> List[float]:
    """Apply adaptive mutation with decreasing rate over generations."""
    # Decrease mutation rate as generations progress
    adaptive_rate = base_mutation_rate * (1 - generation / max_generations)
    return mutate_sequence(sequence, adaptive_rate, mutation_strength)

def crossover_sequences(seq1: List[float], seq2: List[float], crossover_rate: float = 0.8) -> List[float]:
    """Perform crossover between two sequences with probability."""
    if random.random() > crossover_rate or len(seq1) == 0 or len(seq2) == 0:
        return seq1 if random.random() < 0.5 else seq2

    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if seq1 else seq2

    # Single-point crossover with a probabilistic bias towards better parts
    crossover_point = random.randint(1, min_len - 1)
    # Bias towards choosing the better parent's segments
    if random.random() < 0.7:
        # 70% chance to preserve better parts
        if sum(seq1) > sum(seq2):
            child = seq1[:crossover_point] + seq2[crossover_point:]
        else:
            child = seq2[:crossover_point] + seq1[crossover_point:]
    else:
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

def check_diversity(population: List[List[float]], threshold: float = 0.1) -> bool:
    """Check if the population has sufficient diversity."""
    if len(population) < 2:
        return True
    # Compute average pairwise distance between sequences
    distances = []
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            dist = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
            distances.append(dist)
    avg_dist = np.mean(distances) if distances else 0
    return avg_dist > threshold

def optimize_step_function_evolutionary(max_time_seconds=170) -> List[float]:
    """
    Evolutionary optimization to find optimal step function that maximizes 1/C1.
    """
    start_time = time.time()
    _cache_manager.clear()

    # Initialize population with diverse strategies
    population_size = 50
    population = generate_diverse_population(population_size, (100, 1000))

    best_sequence = None
    best_inv_c1 = 0.0

    generation = 0
    stagnation_count = 0
    max_stagnation = 40

    # History for diversity tracking
    recent_improvements = collections.deque(maxlen=20)
    fitness_history = []

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
            recent_improvements.append(current_best_inv_c1)
        else:
            stagnation_count += 1
            recent_improvements.append(best_inv_c1)

        # Try to improve best sequence using LP-based direction
        if best_sequence is not None:
            improved = get_good_direction_to_move_into(best_sequence)
            if improved is not None:
                _, improved_inv_c1 = compute_autocorrelation_constant(improved)
                if improved_inv_c1 > best_inv_c1:
                    best_sequence = improved
                    best_inv_c1 = improved_inv_c1

        # Reintroduce diversity if stagnating
        if stagnation_count > 20 and check_diversity(population):
            # Reset part of the population with fresh sequences
            reset_count = population_size // 4
            for i in range(reset_count):
                n = random.randint(100, 1000)
                if random.random() < 0.5:
                    population[i] = generate_step_function(n)
                else:
                    population[i] = generate_pattern_based_sequence(n)
            stagnation_count = 0  # Reset stagnation count

        # Compute population diversity statistics
        diversity = np.std(fitness_scores) if len(fitness_scores) > 1 else 0.0
        fitness_history.append(current_best_inv_c1)
        if len(fitness_history) > 10:
            fitness_history.pop(0)

        # Adjust tournament size based on diversity and generation
        tournament_size = max(3, min(7, int(5 + diversity * 2)))

        # Adjust mutation rate based on fitness variance and generation
        if len(fitness_history) >= 2:
            fitness_variance = np.var(fitness_history[-5:]) if len(fitness_history) >= 5 else 0.0
            adaptive_mutation_rate = 0.15 * (1 - generation / 100) * (1 + fitness_variance)
        else:
            adaptive_mutation_rate = 0.15 * (1 - generation / 100)
        adaptive_mutation_rate = max(0.05, min(0.3, adaptive_mutation_rate))

        # Selection with dynamic tournament size and elitism
        selected_parents = []

        # Elitism: keep the top performer
        elite_idx = current_best_idx
        selected_parents.append(population[elite_idx].copy())

        # Tournament selection for rest with dynamic tournament size
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

            # Crossover with probability
            child = crossover_sequences(parent1, parent2, crossover_rate=0.8)

            # Mutation with adaptive rate
            child = adaptive_mutate_sequence(child, generation, 100, adaptive_mutation_rate, 0.2)

            new_population.append(child)

        population = new_population[:population_size]

    # Final cleanup and validation
    if best_sequence is not None:
        # Normalize sequence to make sure it's valid
        sum_seq = sum(best_sequence)
        if sum_seq > 0.01:
            best_sequence = [x / sum_seq * 100 for x in best_sequence]

    return best_sequence if best_sequence else generate_uniform_distribution(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use evolutionary optimization approach with diversity management
        best_sequence = optimize_step_function_evolutionary()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        # Fallback to simple random approach
        return generate_uniform_distribution(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")