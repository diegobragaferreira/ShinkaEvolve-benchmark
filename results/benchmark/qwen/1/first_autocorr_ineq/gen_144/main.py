# EVOLVE-BLOCK-START

import numpy as np
from scipy.fft import fft, ifft
import random
import time
from typing import List, Tuple
import copy
from joblib import Parallel, delayed
from numba import jit

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
                conv = ifft(fft(sequence, 2 * n - 1) * fft(sequence, 2 * n - 1).conj()).real
            except Exception:
                # Fallback to JIT for large sequences if FFT fails
                conv = fast_convolve_jit(sequence, sequence)
        else:
            conv = fast_convolve_jit(sequence, sequence)
        max_conv = np.max(conv[:n])
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

def generate_structured_sequence(n: int) -> List[float]:
    """Generate a structured sequence that's likely to perform well."""
    # Create a sequence with exponential decay to reduce autocorrelation
    sequence = []
    for i in range(n):
        # Exponential decay with some noise to break symmetry
        base_val = max(0.01, 100 * np.exp(-i * 0.05))
        noise = random.uniform(0.9, 1.1)
        sequence.append(base_val * noise)
    return sequence

def generate_memory_based_sequence(elite_sequences: List[List[float]], n: int) -> List[float]:
    """Generate a sequence based on learned patterns from elite sequences."""
    if not elite_sequences:
        return generate_structured_sequence(n)

    # Take the average of elite sequences to form a base
    avg_sequence = np.mean(elite_sequences, axis=0)
    # Add some noise to prevent overfitting
    noise = [random.uniform(-0.1, 0.1) for _ in range(n)]
    base_seq = [max(0.01, val * (1 + noise[i])) for i, val in enumerate(avg_sequence)]
    return base_seq

def generate_population(population_size: int, min_n: int = 50, max_n: int = 1000,
                       elite_sequences: List[List[float]] = None) -> List[List[float]]:
    """Generate diverse initial population with structured sequences."""
    population = []
    for _ in range(population_size):
        n = random.randint(min_n, max_n)
        # Use a mix of structured and memory-based sequences
        if elite_sequences and random.random() < 0.3:
            # Memory-based sequence
            individual = generate_memory_based_sequence(elite_sequences, n)
        elif random.random() < 0.7:
            # Structured sequence
            individual = generate_structured_sequence(n)
        else:
            # Random sequence
            individual = [random.uniform(0.1, 100) for _ in range(n)]
        population.append(individual)
    return population

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply mutation to sequence with multiplicative Gaussian perturbation."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply multiplicative Gaussian perturbation
            perturbation = random.gauss(1, 0.1)
            mutated[i] *= abs(perturbation)  # Ensure non-negative
            mutated[i] = max(0.01, mutated[i])
    return mutated

def crossover_sequences(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    min_len = min(len(parent1), len(parent2))
    crossover_point = random.randint(1, min_len - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def local_search_refinement(sequence: List[float], max_iterations: int = 10) -> List[float]:
    """Apply local search refinement to improve sequence."""
    best_seq = sequence.copy()
    best_fitness = compute_autocorrelation_constant(best_seq)[1]  # Get 1/C1

    for _ in range(max_iterations):
        # Try small perturbations
        mutant = mutate_sequence(best_seq, 0.05)
        mutant_fitness = compute_autocorrelation_constant(mutant)[1]  # Get 1/C1

        if mutant_fitness > best_fitness:
            best_seq = mutant
            best_fitness = mutant_fitness

        # Try coordinate-wise optimization
        try:
            # Simple hill climbing approach
            new_seq = best_seq.copy()
            for i in range(len(new_seq)):
                # Try small adjustments
                old_val = new_seq[i]
                test_vals = [old_val * 0.9, old_val, old_val * 1.1]
                best_test = old_val
                best_test_fitness = compute_autocorrelation_constant(new_seq)[1]  # Get 1/C1

                for test_val in test_vals:
                    test_seq = new_seq.copy()
                    test_seq[i] = max(0.01, test_val)
                    test_fitness = compute_autocorrelation_constant(test_seq)[1]  # Get 1/C1
                    if test_fitness > best_test_fitness:
                        best_test = test_val
                        best_test_fitness = test_fitness

                new_seq[i] = best_test

            test_fitness = compute_autocorrelation_constant(new_seq)[1]  # Get 1/C1
            if test_fitness > best_fitness:
                best_seq = new_seq
                best_fitness = test_fitness

        except Exception:
            pass  # Skip if optimization fails

    return best_seq

def adaptive_mutation_rate(population_fitnesses: List[float]) -> float:
    """Calculate adaptive mutation rate based on population diversity."""
    if len(population_fitnesses) < 2:
        return 0.1

    std_dev = np.std(population_fitnesses)
    avg_fitness = np.mean(population_fitnesses)

    # Higher diversity = higher mutation rate
    if avg_fitness > 0:
        mutation_rate = min(0.3, max(0.01, 0.1 + std_dev / avg_fitness))
    else:
        mutation_rate = 0.1

    return mutation_rate

def adaptive_tournament_selection(population: List[List[float]],
                                 fitness_scores: List[float],
                                 diversity: float) -> List[float]:
    """Perform adaptive tournament selection based on population diversity."""
    # Adjust tournament size based on diversity
    if diversity > 0.1:
        tournament_size = 7
    elif diversity < 0.05:
        tournament_size = 3
    else:
        tournament_size = 5

    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]

    # Select the best individual from tournament
    best_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[best_idx]

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence.
    Uses evolutionary optimization with adaptive selection and local search.
    """
    start_time = time.time()
    max_time = 175  # Leave some time for cleanup
    _evaluator.clear_cache()

    # Configuration
    population_size = 50
    generations = 100
    max_stagnation = 20
    elite_size = 5
    elite_history = []  # Track elite sequences across generations

    # Initialize population with structured sequences
    population = generate_population(population_size)

    best_solution = None
    best_fitness = 0.0
    stagnation_counter = 0
    fitness_history = []

    for generation in range(generations):
        # Check time limit
        if time.time() - start_time > max_time:
            break

        # Evaluate fitness for all individuals in parallel
        fitness_scores = Parallel(n_jobs=-1)(
            delayed(compute_autocorrelation_constant)(individual)[1] for individual in population
        )

        # Track best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        fitness_history.append(best_fitness)

        # Check for stagnation using multi-generational trend analysis
        if len(fitness_history) > 10:
            recent_improvement = np.mean(fitness_history[-10:]) - np.mean(fitness_history[:-10])
            if recent_improvement < 0.0001:
                stagnation_counter += 1
                if stagnation_counter >= max_stagnation:
                    # Reset with new diverse population
                    population = generate_population(population_size, elite_sequences=elite_history)
                    stagnation_counter = 0
        else:
            stagnation_counter = 0

        # Calculate adaptive mutation rate
        mutation_rate = adaptive_mutation_rate(fitness_scores)

        # Selection: keep top individuals
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_size]
        elite = [population[i] for i in sorted_indices]
        elite_history.append(copy.deepcopy(elite))

        # Apply local search to elite members
        refined_elite = []
        for ind in elite:
            refined = local_search_refinement(ind)
            refined_elite.append(refined)
        elite = refined_elite

        # Create new population through selection, crossover, and mutation
        new_population = elite.copy()

        while len(new_population) < population_size:
            # Adaptive tournament selection
            parents = [adaptive_tournament_selection(population, fitness_scores,
                                                   np.std(fitness_scores) / max(1e-10, np.mean(fitness_scores)))
                      for _ in range(2)]
            child = crossover_sequences(parents[0], parents[1])
            mutated_child = mutate_sequence(child, mutation_rate)
            new_population.append(mutated_child)

        population = new_population

    # Final local search on best solution
    if best_solution is not None:
        best_solution = local_search_refinement(best_solution, 10)

    return best_solution if best_solution is not None else generate_structured_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")