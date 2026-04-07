# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from scipy.signal import fftconvolve
from typing import List, Optional, Tuple
import random
import time
from collections import OrderedDict
import joblib
from joblib import Parallel, delayed

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class LruCache:
    def __init__(self, maxsize=128):
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
        self._cache = LruCache(maxsize=256)
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache = LruCache(maxsize=256)
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

def compute_autocorrelation_constant_parallel(sequences: List[List[float]]) -> List[Tuple[float, float]]:
    """Parallel evaluation of autocorrelation constants for multiple sequences."""
    return Parallel(n_jobs=-1)(delayed(compute_autocorrelation_constant)(seq) for seq in sequences)

def generate_harmonic_sequence(length: int) -> List[float]:
    """Generate a harmonic-like sequence."""
    sequence = []
    for i in range(length):
        # Create harmonics to encourage structure
        harmonic = 1.0 / (1 + i * 0.1) * np.sin(i * 0.5)
        sequence.append(max(0.01, abs(harmonic) * 100))
    return sequence

def generate_exponential_sequence(length: int) -> List[float]:
    """Generate an exponentially decaying sequence."""
    sequence = []
    for i in range(length):
        # Exponential decay
        val = 100 * np.exp(-i * 0.02)
        sequence.append(max(0.01, val))
    return sequence

def generate_step_sequence(length: int, num_steps: int) -> List[float]:
    """Generate a step function sequence."""
    step_positions = sorted(random.sample(range(length), num_steps))
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

    # Generate sequences using different methods
    for _ in range(population_size // 4):
        n = random.randint(*length_range)
        population.append(generate_harmonic_sequence(n))

    for _ in range(population_size // 4):
        n = random.randint(*length_range)
        population.append(generate_exponential_sequence(n))

    for _ in range(population_size // 4):
        n = random.randint(*length_range)
        num_steps = max(2, min(20, n // 10))
        population.append(generate_step_sequence(n, num_steps))

    # Fill remaining with standard random (but use a better distribution)
    while len(population) < population_size:
        n = random.randint(*length_range)
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
        population.append(sequence)

    return population

def mutate_sequence(sequence: List[float], generation: int, population_size: int) -> List[float]:
    """Apply adaptive mutation to a sequence."""
    mutated = sequence.copy()
    # Dynamic mutation rate
    mutation_rate = max(0.05, 0.3 * (1 - generation / (population_size * 2)))
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Global perturbation - larger changes early, smaller later
            if generation < population_size // 4:
                noise = random.gauss(0, 5.0)  # Large change
            else:
                noise = random.gauss(0, 1.0)  # Small change
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

def fitness_sharing(fitness_scores: List[float], population: List[List[float]], sigma: float = 0.5) -> List[float]:
    """Apply fitness sharing to maintain diversity."""
    shared_fitness = []
    for i in range(len(population)):
        sharing_value = 0
        for j in range(len(population)):
            if i != j:
                # Euclidean distance between sequences (normalized)
                dist = np.linalg.norm(np.array(population[i]) - np.array(population[j])) / len(population[i])
                if dist < sigma:
                    sharing_value += 1 - dist / sigma
        shared_fitness.append(fitness_scores[i] / (1 + sharing_value))
    return shared_fitness

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    
    # Avoid division by zero
    if sum_sequence < 1e-10:
        return None
        
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    
    # Use FFT for faster convolution
    try:
        conv_result = np.real(ifft(fft(normalized_sequence, 2*n-1) *
                                   np.conj(fft(normalized_sequence, 2*n-1))))
        rhs = np.max(conv_result[:2*n-1])  # Only consider the actual convolution results
    except Exception as e:
        print(f"Error during FFT convolution: {e}")
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

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Precompute the convolution constraint matrix using explicit loop
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
            print('LP optimization failed:', result.message)
            return None
    except Exception as e:
        print(f'LP optimization error: {e}')
        return None

def optimize_step_function_evolutionary(max_time_seconds=170) -> List[float]:
    """
    Evolutionary optimization to find optimal step function that maximizes 1/C1.
    """
    start_time = time.time()
    _evaluator.clear_cache()

    # Initialize population with diverse strategies
    population_size = 30
    population = generate_diverse_population(population_size, (100, 1000))

    best_sequence = None
    best_inv_c1 = 0.0

    generation = 0
    stagnation_count = 0
    max_stagnation = 30

    while time.time() - start_time < max_time_seconds and stagnation_count < max_stagnation:
        generation += 1

        # Evaluate fitness (1/C1) of each individual in parallel
        fitness_scores = []
        results = compute_autocorrelation_constant_parallel(population)
        for _, inv_c1 in results:
            fitness_scores.append(inv_c1)

        # Apply fitness sharing for diversity
        shared_fitness_scores = fitness_sharing(fitness_scores, population)

        # Track best solution
        current_best_idx = np.argmax(shared_fitness_scores)
        current_best_inv_c1 = shared_fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Selection with fitness proportionate selection and tournament backup
        selected_parents = []
        
        # Fitness proportionate selection
        total_fitness = sum(shared_fitness_scores)
        if total_fitness > 0:
            probabilities = [f / total_fitness for f in shared_fitness_scores]
            selected_parents.extend(random.choices(population, probabilities, k=population_size - 1))
        else:
            # Fallback to tournament selection
            tournament_size = 5
            for _ in range(population_size - 1):
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitness = [shared_fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_parents.append(population[winner_idx].copy())

        # Elitism: keep the best individual
        selected_parents.insert(0, best_sequence.copy())

        # Create new population through crossover and mutation
        new_population = [best_sequence.copy()]  # Elitism

        while len(new_population) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation with adaptive rate
            child = mutate_sequence(child, generation, population_size)

            new_population.append(child)

        population = new_population[:population_size]

        # Early termination if no significant improvement
        if stagnation_count > 10 and best_inv_c1 > 0.6:
            break

    # Final cleanup and validation
    if best_sequence is not None:
        # Normalize sequence to make sure it's valid
        sum_seq = sum(best_sequence)
        if sum_seq > 0.01:
            best_sequence = [x / sum_seq * 100 for x in best_sequence]

    return best_sequence if best_sequence else generate_harmonic_sequence(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use evolutionary optimization approach
        best_sequence = optimize_step_function_evolutionary()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        # Fallback to harmonic sequence
        return generate_harmonic_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")