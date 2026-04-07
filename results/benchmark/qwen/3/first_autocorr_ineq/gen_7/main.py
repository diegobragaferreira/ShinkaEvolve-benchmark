# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import convolve as fft_convolve
import random
import time

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autocorrelation_constant(sequence):
    """Computes C₁ for a given sequence"""
    if len(sequence) == 0:
        return float('inf')

    # Compute autoconvolution using FFT for efficiency
    autoconv = fft_convolve(sequence, sequence, mode='full')
    # Take the middle part which corresponds to actual convolution
    autoconv = autoconv[len(sequence)-1:2*len(sequence)-1]

    max_conv = np.max(autoconv)
    sum_seq = np.sum(sequence)

    if sum_seq < 0.01:
        return float('inf')  # Invalid sequence

    C1 = 2 * len(sequence) * max_conv / (sum_seq ** 2)
    return C1

def compute_inverse_C1(sequence):
    """Computes 1/C₁ for a given sequence"""
    C1 = compute_autocorrelation_constant(sequence)
    if C1 == float('inf'):
        return 0.0  # Invalid sequence gets 0 score
    return 1.0 / C1

def benchmark_ratio(sequence):
    """Computes how much we beat the benchmark"""
    C1 = compute_autocorrelation_constant(sequence)
    if C1 == float('inf'):
        return 0.0
    return 1.5031 / C1

def combined_score(sequence):
    """Compute the combined score to maximize"""
    inv_C1 = compute_inverse_C1(sequence)
    ratio = benchmark_ratio(sequence)
    return inv_C1 * ratio  # We want both high inverse C1 and beating benchmark

def generate_random_step_function(n_steps=None):
    """Generate a random step function with specified number of steps"""
    if n_steps is None:
        n_steps = random.randint(50, 1000)  # Vary sequence length

    # Generate random heights in [0, 1000] range
    heights = np.random.uniform(0, 1000, n_steps)
    # Ensure sum is at least 0.01
    if np.sum(heights) < 0.01:
        heights[0] = 0.01
    return heights.tolist()

def mutate_sequence(sequence, mutation_rate=0.1):
    """Create a mutated version of the sequence"""
    new_sequence = sequence.copy()
    n = len(new_sequence)

    # Randomly change some elements
    for i in range(n):
        if random.random() < mutation_rate:
            # Apply small perturbation or random value
            if random.random() < 0.5:
                # Small perturbation
                new_sequence[i] = max(0, new_sequence[i] + random.gauss(0, 10))
            else:
                # Random value in valid range
                new_sequence[i] = random.uniform(0, 1000)

    # Ensure minimum sum requirement
    if np.sum(new_sequence) < 0.01:
        new_sequence[0] = max(0.01, new_sequence[0] + 0.01)

    return new_sequence

def crossover_sequences(seq1, seq2):
    """Create offspring from two parent sequences"""
    n1, n2 = len(seq1), len(seq2)
    min_len = min(n1, n2)

    # Create new sequence by combining parents
    child = []
    for i in range(max(n1, n2)):
        if i < min_len:
            # Blend genes from both parents
            child.append((seq1[i] + seq2[i]) / 2.0)
        elif i < n1:
            child.append(seq1[i])
        else:
            child.append(seq2[i])

    return child

def evolutionary_search(max_time_seconds=180):
    """Main evolutionary algorithm to search for best sequence"""
    start_time = time.time()

    # Initialize population
    population_size = 50
    population = [generate_random_step_function() for _ in range(population_size)]

    best_individual = None
    best_score = -float('inf')

    generation = 0
    stagnation_counter = 0
    max_stagnation = 20

    while time.time() - start_time < max_time_seconds:
        generation += 1

        # Evaluate fitness of individuals
        fitness_scores = []
        for individual in population:
            score = combined_score(individual)
            fitness_scores.append(score)

            # Update best individual
            if score > best_score:
                best_score = score
                best_individual = individual.copy()
                stagnation_counter = 0

        stagnation_counter += 1
        if stagnation_counter > max_stagnation:
            # Reset if no improvement for too long
            population = [generate_random_step_function() for _ in range(population_size)]
            stagnation_counter = 0

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_individuals = [population[i] for i in sorted_indices[:population_size//2]]

        # Generate new population through selection, crossover, and mutation
        new_population = []
        for _ in range(population_size):
            # Select parents
            parent1 = random.choice(top_individuals)
            parent2 = random.choice(top_individuals)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation
            child = mutate_sequence(child)

            new_population.append(child)

        population = new_population

        # Occasionally add new random individuals to maintain diversity
        if generation % 5 == 0:
            for i in range(5):
                population[random.randint(0, population_size-1)] = generate_random_step_function()

    return best_individual if best_individual is not None else generate_random_step_function()

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    return evolutionary_search()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")