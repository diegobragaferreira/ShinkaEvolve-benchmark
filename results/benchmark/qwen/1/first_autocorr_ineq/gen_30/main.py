# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from multiprocessing import Pool
import random
import math

def convolve_fft(a, b):
    """Compute convolution using FFT for better performance."""
    n = len(a)
    # Pad to length 2*n - 1 for full convolution
    padded_length = 2 * n - 1
    fa = fft(a, padded_length)
    fb = fft(b, padded_length)
    result = ifft(fa * fb.conj()).real
    # Return only the valid convolution part
    return result[:n]

def compute_c1(sequence):
    """Compute the C1 constant for a given sequence."""
    n = len(sequence)
    if n == 0:
        return float('inf')

    sum_a = np.sum(sequence)
    if sum_a < 1e-10:
        return float('inf')

    # Compute convolution using FFT
    conv = convolve_fft(sequence, sequence)
    max_conv = np.max(conv)

    # Compute C1 = 2n * max(conv) / (sum(a))^2
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def evaluate_fitness(sequence):
    """Evaluate fitness as inverse of C1 (higher is better)"""
    c1 = compute_c1(sequence)
    if c1 == float('inf'):
        return 0.0
    return 1.0 / c1

def generate_sine_wave_sequence(n):
    """Generate a structured sequence based on sine wave."""
    return [abs(math.sin(i * 0.5) * 100) + 1 for i in range(n)]

def generate_random_valid_sequence(n):
    """Generate a random valid sequence with some structure."""
    # Start with a sine wave pattern to provide good initial structure
    base_seq = generate_sine_wave_sequence(n)
    # Add some noise to make it less predictable
    noise_factor = 0.1
    return [max(0.01, x * (1 + random.uniform(-noise_factor, noise_factor)))
            for x in base_seq]

def mutate_sequence(sequence, mutation_rate):
    """Apply mutation to sequence with adaptive rate."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply Gaussian perturbation
            mutated[i] *= random.uniform(0.8, 1.2)
            mutated[i] = max(0.01, mutated[i])  # Ensure non-negative
    return mutated

def crossover_sequences(parent1, parent2):
    """Perform crossover between two sequences."""
    if len(parent1) != len(parent2):
        # If lengths differ, use the shorter one
        min_len = min(len(parent1), len(parent2))
        parent1 = parent1[:min_len]
        parent2 = parent2[:min_len]

    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def adaptive_mutation_rate(population_fitnesses):
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

def search_for_best_sequence() -> list[float]:
    """Evolutionary optimization to find the best sequence."""
    # Configuration
    population_size = 50
    generations = 100
    max_stagnation = 20
    elite_size = 5

    # Initialize population with structured sequences
    population = [
        generate_random_valid_sequence(random.randint(50, 500))
        for _ in range(population_size)
    ]

    best_solution = None
    best_fitness = 0.0
    stagnation_counter = 0

    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual.copy()

        # Check for stagnation
        if generation > 0 and abs(fitness_scores[-1] - fitness_scores[-2]) < 1e-6:
            stagnation_counter += 1
            if stagnation_counter >= max_stagnation:
                # Reset with new diverse population
                population = [
                    generate_random_valid_sequence(random.randint(50, 500))
                    for _ in range(population_size)
                ]
                stagnation_counter = 0
        else:
            stagnation_counter = 0

        # Calculate adaptive mutation rate
        mutation_rate = adaptive_mutation_rate(fitness_scores)

        # Selection: keep top individuals
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_size]
        elite = [population[i] for i in sorted_indices]

        # Create new population through selection, crossover, and mutation
        new_population = elite.copy()

        while len(new_population) < population_size:
            # Tournament selection
            parents = random.sample(elite, 2)
            child = crossover_sequences(parents[0], parents[1])
            mutated_child = mutate_sequence(child, mutation_rate)
            new_population.append(mutated_child)

        population = new_population

    return best_solution if best_solution is not None else generate_random_valid_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")