# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use FFT for efficiency when sequence is large
    if n > 100:
        b = convolve_fft(a, a)
    else:
        b = convolve_direct(a, a)

    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

def fitness_function(sequence: List[float]) -> float:
    """Evaluate fitness of a sequence based on 1/C1."""
    _, inv_c1 = compute_c1_constant(sequence)
    return inv_c1

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply random mutation to a sequence."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.2 * mutated[i]))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform uniform crossover between two sequences."""
    min_len = min(len(seq1), len(seq2))
    child = []

    # Uniform crossover
    for i in range(min_len):
        if random.random() < 0.5:
            child.append(seq1[i])
        else:
            child.append(seq2[i])

    # Append extra elements from longer sequence
    if len(seq1) > len(seq2):
        child.extend(seq1[min_len:])
    elif len(seq2) > len(seq1):
        child.extend(seq2[min_len:])

    return child

def tournament_selection(population: List[List[float]], fitnesses: List[float],
                         tournament_size: int = 3) -> List[float]:
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def genetic_autocorrelation_optimizer(
    max_time_seconds: int = 180,
    pop_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    elite_size: int = 5
) -> List[float]:
    """
    Genetic Algorithm for optimizing step function to maximize 1/C1.
    """
    start_time = time.time()

    # Initialize population
    population = []
    for _ in range(pop_size):
        n = random.randint(100, 1000)
        individual = [random.random() * 100 for _ in range(n)]
        population.append(individual)

    best_sequence = None
    best_fitness = -float('inf')

    for gen in range(generations):
        if time.time() - start_time > max_time_seconds:
            break

        # Evaluate fitness for all individuals
        fitness_scores = [fitness_function(individual) for individual in population]

        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_sequence = population[max_fitness_idx].copy()

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Create new population
        new_population = []

        # Elitism: keep top individuals
        for i in range(elite_size):
            new_population.append(sorted_population[i].copy())

        # Generate offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection for parents
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation
            child = mutate_sequence(child, mutation_rate)

            # Random initialization for diversity
            if random.random() < 0.1:
                n = random.randint(100, 1000)
                child = [random.random() * 100 for _ in range(n)]

            new_population.append(child)

        population = new_population

    return best_sequence if best_sequence is not None else [1.0]

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    random.seed(42)
    np.random.seed(42)

    return genetic_autocorrelation_optimizer()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")