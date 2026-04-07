# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
from typing import List, Tuple
import time
import numba
from numba import jit

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

def mutate_sequence(sequence: List[float], generation: int, population_size: int,
                   mutation_strength=0.3) -> List[float]:
    """Apply adaptive mutation to a sequence with rate based on generation and diversity."""
    mutated = sequence.copy()

    # Adaptive mutation rate that decreases with generation and increases with diversity
    # High diversity early on -> higher mutation rate
    # Low diversity later -> lower mutation rate
    diversity_factor = 1.0  # Will be calculated based on population spread

    # Calculate population diversity (standard deviation of sequence sums)
    if len(sequence) > 1:
        # Simple measure of diversity: standard deviation of sequence values
        std_dev = np.std(sequence)
        diversity_factor = 1.0 + 0.5 * np.tanh(std_dev / np.mean(sequence) if np.mean(sequence) > 0 else 1)

    # Decreasing mutation rate over generations
    base_mutation_rate = 0.3 * (1 - generation / (population_size * 2))
    mutation_rate = max(0.05, base_mutation_rate * diversity_factor)

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

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence with enhanced local search."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)

    if sum_sequence < 0.01:
        return None

    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)

    if g_fun is None:
        # Fallback: try gradient-based approach
        return gradient_improve_sequence(sequence)

    sum_sequence = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_sequence for x in g_fun]
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def gradient_improve_sequence(sequence: list[float], step_size: float = 0.01) -> list[float]:
    """Apply simple gradient-based improvement to sequence."""
    # This is a simplified gradient approximation
    # In practice, we could compute actual gradients or use finite differences

    # For now, just do a small perturbation based on sequence characteristics
    improved = sequence.copy()

    # Try small adjustments to improve the sequence
    for i in range(len(improved)):
        # Perturb slightly based on neighboring values
        neighbors = []
        if i > 0:
            neighbors.append(improved[i-1])
        if i < len(improved) - 1:
            neighbors.append(improved[i+1])

        if neighbors:
            avg_neighbor = np.mean(neighbors)
            # Adjust toward average neighbor to smooth out extremes
            if abs(avg_neighbor - improved[i]) > 0.1:
                improved[i] = improved[i] * (1 - step_size) + avg_neighbor * step_size

    return improved

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

    result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub)

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        print('LP optimization failed.')
        return None

def adaptive_tournament_selection(population: List[List[float]],
                                  fitness_scores: List[float],
                                  population_size: int,
                                  generation: int,
                                  diversity: float) -> List[List[float]]:
    """
    Perform adaptive tournament selection with dynamic tournament size.
    Tournament size adapts based on generation and population diversity.
    """
    selected_parents = []

    # Determine tournament size based on generation and diversity
    if generation <= 10 or diversity > 0.1:
        # Early generations or high diversity: larger tournament for stronger selection pressure
        tournament_size = min(9, max(5, population_size // 3))
    elif generation >= 50 or diversity < 0.05:
        # Later generations or low diversity: smaller tournament to preserve diversity
        tournament_size = min(4, max(2, population_size // 6))
    else:
        # Middle generations: moderate tournament size
        tournament_size = 5

    # Elitism: keep the top performer
    elite_idx = np.argmax(fitness_scores)
    selected_parents.append(population[elite_idx].copy())

    # Tournament selection for rest
    for _ in range(population_size - 1):  # -1 because we already added elite
        tournament_indices = random.sample(range(population_size), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected_parents.append(population[winner_idx].copy())

    return selected_parents

def optimize_step_function_evolutionary(max_time_seconds=170) -> List[float]:
    """
    Evolutionary optimization to find optimal step function that maximizes 1/C1.
    """
    start_time = time.time()
    _evaluator.clear_cache()

    # Initialize population with more diverse strategies
    population_size = 30
    population = generate_diverse_population(population_size, (100, 1000))

    best_sequence = None
    best_inv_c1 = 0.0

    generation = 0
    stagnation_count = 0
    max_stagnation = 30
    diversity_threshold = 0.1  # Minimum diversity to maintain exploration

    while time.time() - start_time < max_time_seconds and stagnation_count < max_stagnation:
        generation += 1

        # Evaluate fitness (1/C1) of each individual
        fitness_scores = []
        for seq in population:
            _, inv_c1 = compute_autocorrelation_constant(seq)
            fitness_scores.append(inv_c1)

        # Track best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_inv_c1 = fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Calculate population diversity
        if len(fitness_scores) > 1:
            diversity = np.std(fitness_scores) / (np.mean(fitness_scores) + 1e-10)
        else:
            diversity = 1.0

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

        # Diversity maintenance: if diversity is too low, inject new variation
        if diversity < diversity_threshold and generation > 5:
            # Introduce some random sequences to increase diversity
            num_new = population_size // 6
            for _ in range(num_new):
                new_seq = generate_random_valid_sequence((100, 1000), 'step')
                population[random.randint(0, len(population)-1)] = new_seq

        # Perform adaptive tournament selection
        selected_parents = adaptive_tournament_selection(population, fitness_scores,
                                                        population_size, generation, diversity)

        # Create new population through crossover and mutation
        new_population = [best_sequence.copy()]  # Elitism: keep best individual

        while len(new_population) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation with adaptive rate
            child = mutate_sequence(child, generation, population_size, mutation_strength=0.2)

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