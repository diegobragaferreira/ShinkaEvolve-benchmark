# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import time
import random
from typing import List, Tuple
from numba import jit
import multiprocessing as mp
from functools import partial

@jit(nopython=True)
def compute_autocorrelation_fast(sequence: np.ndarray) -> Tuple[float, float]:
    """
    Fast computation of autocorrelation constant C₁ using numba acceleration.
    Returns (C1_value, max_autocorr)
    """
    n = len(sequence)
    if n == 0:
        return float('inf'), 0

    # Compute autoconvolution manually for speed
    autocorr = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            autocorr[i+j] += sequence[i] * sequence[j]

    max_autocorr = np.max(autocorr[n-1:])
    sum_sq = np.sum(sequence)**2

    if sum_sq == 0:
        return float('inf'), max_autocorr

    C1 = 2 * n * max_autocorr / sum_sq
    return C1, max_autocorr

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    Returns (C1_value, max_autocorr)
    """
    if len(sequence) == 0:
        return float('inf'), 0

    # Convert to numpy array for efficient computation
    seq_np = np.array(sequence)

    # For small sequences, use direct computation
    if len(sequence) <= 1000:
        C1, max_autocorr = compute_autocorrelation_fast(seq_np)
        return C1, max_autocorr
    else:
        # For large sequences, use FFT-based convolution
        autocorr = fftconvolve(seq_np, seq_np[::-1], mode='full')
        autocorr = autocorr[len(sequence)-1:]
        max_autocorr = np.max(autocorr)

        sum_sq = np.sum(seq_np)**2
        if sum_sq == 0:
            return float('inf'), max_autocorr

        C1 = 2 * len(sequence) * max_autocorr / sum_sq
        return C1, max_autocorr

def evaluate_fitness(sequence: List[float]) -> float:
    """
    Evaluates fitness of a sequence (negative of 1/C₁ to maximize 1/C₁).
    """
    C1, _ = compute_autocorrelation_constant(sequence)
    if C1 == float('inf'):
        return float('-inf')  # Penalize invalid sequences
    return -1.0 / C1

def generate_random_sequence(n: int) -> List[float]:
    """
    Generate a valid random sequence with proper normalization.
    """
    # Generate random sequence
    seq = np.random.rand(n)
    # Normalize so sum is at least 0.01
    seq = seq * (0.01 / (np.sum(seq) + 1e-10))
    return seq.tolist()

def adaptive_mutation(parent: List[float], diversity: float) -> List[float]:
    """
    Apply adaptive mutation based on population diversity.
    """
    child = parent.copy()
    mutation_rate = max(0.01, min(0.5, 0.1 + 0.4 * (1 - diversity)))

    # Mutate random elements
    n = len(child)
    mutations = int(mutation_rate * n)
    for _ in range(mutations):
        idx = np.random.randint(0, n)
        # Small perturbation
        child[idx] *= np.random.normal(1.0, 0.1)
        # Clamp to reasonable bounds
        child[idx] = max(0, min(1000, child[idx]))

    return child

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """
    Single-point crossover between two sequences.
    """
    n1, n2 = len(parent1), len(parent2)
    n = min(n1, n2)

    if n == 0:
        return []

    # Random crossover point
    crossover_point = np.random.randint(1, n)

    # Create offspring
    child = parent1[:crossover_point] + parent2[crossover_point:]

    # Extend if needed
    if n1 > n:
        child.extend(parent1[n:])
    elif n2 > n:
        child.extend(parent2[n:])

    return child

def adaptive_evolutionary_search(max_time_seconds: int = 180) -> List[float]:
    """
    Adaptive evolutionary search for optimal sequence.
    """
    start_time = time.time()

    # Parameters
    population_size = 50
    max_generations = 1000
    elite_size = 5
    min_sequence_length = 100
    max_sequence_length = 2000
    stagnation_threshold = 50

    # Initialize population
    population = []
    for _ in range(population_size):
        n = np.random.randint(min_sequence_length, max_sequence_length)
        seq = generate_random_sequence(n)
        population.append(seq)

    best_fitness = float('-inf')
    best_sequence = None
    generation = 0
    stagnation_count = 0

    while time.time() - start_time < max_time_seconds and generation < max_generations:
        # Evaluate fitness for all individuals
        fitness_scores = []
        for seq in population:
            fitness = evaluate_fitness(seq)
            fitness_scores.append(fitness)

        # Sort by fitness (descending order)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Track best individual
        current_best_fitness = sorted_fitness[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_sequence = sorted_population[0].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Check for stagnation
        if stagnation_count > stagnation_threshold:
            # Introduce more diversity
            for i in range(elite_size, population_size):
                # Create new individuals by mutating elites
                parent_idx = np.random.randint(0, elite_size)
                population[i] = adaptive_mutation(sorted_population[parent_idx],
                                                 0.5)  # Dummy diversity for now
            stagnation_count = 0

        # Create new population
        new_population = sorted_population[:elite_size]  # Keep elites

        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = np.random.randint(0, elite_size * 2)
            parent2_idx = np.random.randint(0, elite_size * 2)

            parent1 = sorted_population[parent1_idx]
            parent2 = sorted_population[parent2_idx]

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            if np.random.rand() < 0.8:  # 80% mutation probability
                child = adaptive_mutation(child, 0.5)

            # Ensure valid range
            child = [max(0, min(1000, x)) for x in child]

            new_population.append(child)

        population = new_population
        generation += 1

    return best_sequence if best_sequence is not None else generate_random_sequence(1000)

def search_for_best_sequence() -> List[float]:
    """
    Main function to find the best coefficient sequence using adaptive evolutionary optimization.
    """
    try:
        # Run adaptive evolutionary search
        best_sequence = adaptive_evolutionary_search()
        return best_sequence
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple random sequence
        return generate_random_sequence(1000)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")