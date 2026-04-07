# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import random
from typing import List, Tuple
import time
import numba
from numba import jit
from joblib import Parallel, delayed
from scipy.optimize import minimize
import math

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

def generate_mathematical_step_function(n: int) -> List[float]:
    """
    Generate a step function with mathematical properties designed to minimize
    autocorrelation peaks. Uses a geometric decay pattern that's known to
    reduce convolution maxima.
    """
    # Use a geometric progression for step heights to reduce autocorrelation
    num_steps = max(3, min(25, n // 15))  # Adjust number of steps based on sequence length

    # Create step positions that are more evenly distributed but with clustering
    step_positions = []
    step_width = n / num_steps
    for i in range(num_steps):
        start_pos = int(i * step_width)
        # Add some clustering effect to positions
        cluster_offset = int(np.random.exponential(scale=10) * (i % 3 - 1))
        actual_start = max(0, min(n-1, start_pos + cluster_offset))
        step_positions.append(actual_start)

    # Sort positions and ensure no overlaps
    step_positions = sorted(set(step_positions))
    if len(step_positions) < num_steps:
        # Fill missing positions
        while len(step_positions) < num_steps:
            new_pos = random.randint(0, n-1)
            if new_pos not in step_positions:
                step_positions.append(new_pos)
        step_positions.sort()

    # Calculate step heights with geometric decay and some randomness
    step_heights = []
    # Base geometric decay with some randomness
    base_decay = 0.8
    for i in range(len(step_positions)):
        # Geometric decay combined with some randomness (0.8 to 1.2 multiplier)
        height_base = 100 * (base_decay ** i)
        noise = random.uniform(0.8, 1.2)
        height = max(0.01, height_base * noise)
        step_heights.append(height)

    # Ensure we have exactly num_steps
    if len(step_heights) > num_steps:
        step_heights = step_heights[:num_steps]
    elif len(step_heights) < num_steps:
        # Fill remaining with last height
        while len(step_heights) < num_steps:
            step_heights.append(step_heights[-1])

    # Create final sequence
    sequence = [0.0] * n
    for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
        # Determine end position
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = n

        # Ensure proper bounds
        pos = max(0, min(n-1, pos))
        end_pos = max(pos+1, min(n, end_pos))

        # Set step values
        if end_pos > pos:
            sequence[pos:end_pos] = [height] * (end_pos - pos)

    return sequence

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

    elif method == 'mathematical_step':
        # Use our enhanced mathematical step function generator
        return generate_mathematical_step_function(n)

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

    # Add mathematical step-function examples (increased from 1/4 to 1/3)
    for _ in range(population_size // 3):
        population.append(generate_random_valid_sequence(length_range, 'mathematical_step'))

    # Add regular step-function examples
    for _ in range(population_size // 6):
        population.append(generate_random_valid_sequence(length_range, 'step'))

    # Add some Gaussian examples
    for _ in range(population_size // 6):
        population.append(generate_random_valid_sequence(length_range, 'gaussian'))

    # Add hybrid sequences - combinations of different approaches
    for _ in range(population_size // 6):
        # Create a hybrid: start with mathematical step, then add some Gaussian noise
        n = random.randint(*length_range)
        base_seq = generate_mathematical_step_function(n)
        # Add some noise while keeping structure
        noise_level = 0.1
        hybrid_seq = [max(0.01, base_seq[i] * (1 + random.gauss(0, noise_level))) for i in range(n)]
        population.append(hybrid_seq)

    # Fill remaining with standard random
    while len(population) < population_size:
        population.append(generate_random_valid_sequence(length_range, 'uniform'))

    return population

def identify_step_boundaries(sequence: List[float], threshold_factor=0.1) -> List[int]:
    """
    Identify potential step boundaries in a sequence by looking for significant
    changes in adjacent values.
    """
    if len(sequence) < 2:
        return []

    boundaries = []
    avg_val = np.mean(sequence)
    threshold = avg_val * threshold_factor

    # Look for significant differences between consecutive elements
    for i in range(len(sequence) - 1):
        if abs(sequence[i+1] - sequence[i]) > threshold:
            boundaries.append(i)

    return sorted(set(boundaries))

def mutate_sequence(sequence: List[float], mutation_rate=0.1, mutation_strength=0.3) -> List[float]:
    """Apply mutation to a sequence with specified rate and strength."""
    mutated = sequence.copy()

    # First, identify step boundaries if this looks like a step function
    boundaries = identify_step_boundaries(sequence, threshold_factor=0.05)

    # For very short sequences or if boundaries aren't clear, apply normal mutation
    if len(boundaries) < 2 or len(sequence) < 20:
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Apply Gaussian noise scaled by mutation strength
                noise = random.gauss(0, mutation_strength * mutated[i])
                mutated[i] = max(0.01, mutated[i] + noise)
    else:
        # Apply step-aware mutation
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # If we're near a boundary, modify the whole segment
                # Otherwise, do standard mutation
                is_boundary = any(abs(i - boundary) < 3 for boundary in boundaries)

                if is_boundary and random.random() < 0.5:
                    # Change the height of the step (more aggressive change)
                    mutated[i] = max(0.01, mutated[i] * random.uniform(0.7, 1.3))
                else:
                    # Apply Gaussian noise scaled by mutation strength
                    noise = random.gauss(0, mutation_strength * mutated[i])
                    mutated[i] = max(0.01, mutated[i] + noise)

    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences with autocorrelation awareness."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if seq1 else seq2

    # Try to find a crossover point that minimizes disruption to step structure
    # First, let's check if both sequences have similar structures

    # Get some statistics about both sequences
    avg1 = np.mean(seq1) if len(seq1) > 0 else 0
    avg2 = np.mean(seq2) if len(seq2) > 0 else 0

    # Prefer crossover points that don't split potential step regions
    # Find potential step boundaries in both sequences
    boundaries1 = identify_step_boundaries(seq1, threshold_factor=0.05)
    boundaries2 = identify_step_boundaries(seq2, threshold_factor=0.05)

    # Choose crossover point that avoids major structural changes
    if len(boundaries1) > 0 and len(boundaries2) > 0:
        # Try to pick a crossover point near a boundary
        possible_points = []
        for b in boundaries1:
            if 1 <= b < min_len - 1:
                possible_points.append(b)
        for b in boundaries2:
            if 1 <= b < min_len - 1:
                possible_points.append(b)

        # Avoid crossing boundaries if possible
        if possible_points:
            # Select a point closer to a boundary, but biased toward the middle to maintain diversity
            crossover_point = random.choice(possible_points)
        else:
            crossover_point = random.randint(1, min_len - 1)
    else:
        # Fallback to simple random crossover point
        crossover_point = random.randint(1, min_len - 1)

    child = seq1[:crossover_point] + seq2[crossover_point:]

    # Ensure minimum positive value for all elements
    child = [max(0.01, x) for x in child]

    # If the result is very different from either parent, try to preserve some structure
    # by recomputing with a more careful approach if needed
    return child

def evaluate_fitness_single(sequence: List[float]) -> float:
    """Evaluate fitness for a single sequence."""
    _, inv_c1 = compute_autocorrelation_constant(sequence)
    return inv_c1

def evaluate_fitness_batch_parallel(sequences: List[List[float]]) -> List[float]:
    """Evaluate multiple sequences in parallel for better performance."""
    fitness_scores = Parallel(n_jobs=-1)(delayed(evaluate_fitness_single)(seq) for seq in sequences)
    return fitness_scores

def local_search_refinement(sequence: List[float], max_iterations: int = 10) -> List[float]:
    """Apply local search refinement to improve sequence."""
    best_seq = sequence.copy()
    best_fitness = evaluate_fitness_single(best_seq)

    for _ in range(max_iterations):
        # Try small perturbations
        mutated = best_seq.copy()
        for i in range(len(mutated)):
            if random.random() < 0.1:  # 10% chance to mutate each element
                mutated[i] *= random.uniform(0.9, 1.1)  # Small multiplicative change
                mutated[i] = max(0.01, mutated[i])  # Ensure non-negative

        mutated_fitness = evaluate_fitness_single(mutated)
        if mutated_fitness > best_fitness:
            best_seq = mutated
            best_fitness = mutated_fitness

    return best_seq

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

        # Apply local refinement to the best solution found so far
        if best_sequence is not None:
            refined = local_search_refinement(best_sequence, 5)
            _, refined_inv_c1 = compute_autocorrelation_constant(refined)
            if refined_inv_c1 > best_inv_c1:
                best_sequence = refined
                best_inv_c1 = refined_inv_c1

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
