# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple
import copy

# Constants
MAX_TIME_SECONDS = 180
MAX_MEMORY_GB = 5
MIN_SEQUENCE_LENGTH = 10
MAX_SEQUENCE_LENGTH = 1000
BENCHMARK_THRESHOLD = 1.5031
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 5
ADAPTIVE_THRESHOLD = 0.01

@numba.jit(nopython=True)
def compute_convolution_fast(a):
    """
    Compute the convolution of a sequence with itself using FFT for large arrays
    to improve performance.
    """
    n = len(a)
    if n < 100:
        # Direct computation for small arrays
        b = np.zeros(2*n - 1)
        for i in range(n):
            for j in range(n):
                b[i+j] += a[i] * a[j]
        return b
    else:
        # Use FFT for large arrays
        b = fftconvolve(a, a, mode='full')
        return b[:2*n - 1]

def compute_c1(sequence):
    """
    Compute C1 value for a given sequence.
    """
    n = len(sequence)
    if n < 1:
        return float('inf')

    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')

    conv = compute_convolution_fast(sequence)
    max_conv = np.max(conv)

    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def compute_inv_c1(sequence):
    """
    Compute inverse of C1 value (the objective to maximize).
    """
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 <= 0:
        return 0.0
    return 1.0 / c1

def generate_random_sequence(length: int) -> List[float]:
    """Generate a random sequence with specified length."""
    return [random.uniform(0, 100) for _ in range(length)]

def mutate_individual(individual: List[float], mutation_rate: float = MUTATION_RATE) -> List[float]:
    """Mutate an individual with adaptive mutation rates."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply different mutation strategies based on context
            if random.random() < 0.5:
                # Gaussian perturbation
                mutated[i] = max(0, mutated[i] + random.gauss(0, 0.1 * mutated[i] + 0.01))
            else:
                # Uniform mutation
                mutated[i] = random.uniform(0, 100)
    return mutated

def crossover_individuals(parent1: List[float], parent2: List[float],
                         crossover_rate: float = CROSSOVER_RATE) -> Tuple[List[float], List[float]]:
    """Perform crossover between two individuals."""
    if random.random() > crossover_rate or len(parent1) != len(parent2):
        return parent1.copy(), parent2.copy()

    # Single-point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def fitness_function(sequence: List[float]) -> float:
    """Calculate fitness (inverse C1) of a sequence."""
    return compute_inv_c1(sequence)

def adaptive_population_size(current_gen: int, best_fitness: float, prev_fitness: float) -> int:
    """Adaptively adjust population size based on convergence."""
    if abs(best_fitness - prev_fitness) < ADAPTIVE_THRESHOLD and current_gen > 10:
        return max(50, POPULATION_SIZE // 2)  # Reduce population size
    else:
        return POPULATION_SIZE

def genetic_algorithm_search() -> List[float]:
    """Main genetic algorithm implementation."""
    start_time = time.time()

    # Initialize population
    population = [generate_random_sequence(random.randint(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))
                  for _ in range(POPULATION_SIZE)]

    best_sequence = None
    best_fitness = 0.0
    prev_fitness = 0.0
    stagnation_count = 0

    for generation in range(GENERATIONS):
        if time.time() - start_time > MAX_TIME_SECONDS:
            break

        # Calculate fitness for all individuals
        fitness_scores = [fitness_function(individual) for individual in population]

        # Update best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_sequence = population[max_fitness_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Check for stagnation
        if stagnation_count > 10:
            # Restart with new random population
            population = [generate_random_sequence(random.randint(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))
                          for _ in range(POPULATION_SIZE)]
            stagnation_count = 0
            continue

        # Adaptive population sizing
        current_pop_size = adaptive_population_size(generation, best_fitness, prev_fitness)

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Keep elite individuals
        elite = sorted_population[:ELITISM_COUNT]

        # Create next generation
        next_generation = elite.copy()

        # Generate offspring through selection, crossover, and mutation
        while len(next_generation) < current_pop_size:
            # Tournament selection
            tournament_size = 3
            selected_parents = []
            for _ in range(2):
                tournament_indices = random.sample(range(len(sorted_population)), tournament_size)
                tournament_fitness = [sorted_fitness[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                selected_parents.append(sorted_population[winner_index])

            # Crossover
            child1, child2 = crossover_individuals(selected_parents[0], selected_parents[1])

            # Mutation
            child1 = mutate_individual(child1)
            child2 = mutate_individual(child2)

            next_generation.extend([child1, child2])

        # Trim to correct population size
        population = next_generation[:current_pop_size]
        prev_fitness = best_fitness

    return best_sequence if best_sequence is not None else generate_random_sequence(50)

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    try:
        # Run genetic algorithm search
        best_sequence = genetic_algorithm_search()

        # Ensure minimum length and validity
        if len(best_sequence) < MIN_SEQUENCE_LENGTH:
            best_sequence.extend([random.uniform(0, 100) for _ in range(MIN_SEQUENCE_LENGTH - len(best_sequence))])

        # Final validation
        if compute_c1(best_sequence) == float('inf') or compute_inv_c1(best_sequence) <= 0:
            # Fallback to random sequence if invalid
            best_sequence = generate_random_sequence(random.randint(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))

        return best_sequence
    except Exception as e:
        # Fallback to simple random generation on error
        return generate_random_sequence(random.randint(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")