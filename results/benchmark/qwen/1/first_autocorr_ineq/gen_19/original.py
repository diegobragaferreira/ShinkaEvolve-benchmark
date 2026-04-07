# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
from typing import List, Tuple
import time

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Computes the autocorrelation constant C1 and its reciprocal 1/C1.

    Args:
        sequence: List of non-negative real numbers representing step heights

    Returns:
        Tuple of (C1, 1/C1) where C1 = 2*n*max(convolution) / (sum(sequence))^2
    """
    if not sequence or sum(sequence) < 0.01:
        return float('inf'), 0.0

    n = len(sequence)
    # Use FFT-based convolution for efficiency
    conv = fftconvolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    sum_seq = sum(sequence)

    if sum_seq == 0:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

def generate_random_valid_sequence(length_range=(50, 500)) -> List[float]:
    """Generate a random valid sequence within specified length range."""
    n = random.randint(*length_range)
    sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
    return sequence

def mutate_sequence(sequence: List[float], mutation_rate=0.1) -> List[float]:
    """Apply mutation to a sequence with specified rate."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate by adding small random value
            mutated[i] = max(0.0, mutated[i] + random.gauss(0, 0.5))
    return mutated

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if seq1 else seq2

    # Single-point crossover
    crossover_point = random.randint(1, min_len - 1)
    child = seq1[:crossover_point] + seq2[crossover_point:]
    return child

def optimize_step_function_evolutionary(max_time_seconds=170) -> List[float]:
    """
    Evolutionary optimization to find optimal step function that maximizes 1/C1.
    """
    start_time = time.time()

    # Initialize population
    population_size = 20
    population = [generate_random_valid_sequence() for _ in range(population_size)]

    best_sequence = None
    best_inv_c1 = 0.0

    generation = 0
    stagnation_count = 0
    max_stagnation = 50

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

        # Selection (tournament selection)
        selected_parents = []
        tournament_size = 3
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner_idx].copy())

        # Create new population through crossover and mutation
        new_population = []

        # Elitism: keep best individual
        new_population.append(best_sequence.copy())

        while len(new_population) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation
            child = mutate_sequence(child, mutation_rate=0.1)

            # Ensure minimum positive value
            child = [max(0.01, x) for x in child]

            new_population.append(child)

        population = new_population[:population_size]

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