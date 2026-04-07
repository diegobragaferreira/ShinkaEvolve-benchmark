# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
import time
import copy

def compute_c1(sequence):
    """Compute C₁ for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    # Compute autoconvolution using FFT for efficiency
    a = np.array(sequence)
    conv = signal.fftconvolve(a, a, mode='full')
    conv = conv[len(a)-1:]  # Take the relevant part

    # Max convolution value
    max_conv = np.max(conv)

    # Sum of sequence squared
    sum_sq = np.sum(a)**2

    if sum_sq == 0:
        return float('inf')

    # Compute C₁
    c1 = 2 * len(a) * max_conv / sum_sq

    return c1

def compute_inv_c1(sequence):
    """Compute 1/C₁ for a given sequence."""
    c1 = compute_c1(sequence)
    if c1 == 0:
        return 0
    return 1.0 / c1

def generate_random_valid_sequence(min_length=10, max_length=1000, max_height=1000):
    """Generate a random valid sequence with specified constraints."""
    n = random.randint(min_length, max_length)
    # Generate random heights in [0, max_height]
    sequence = [random.uniform(0, max_height) for _ in range(n)]
    return sequence

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    # Uniform crossover
    child = []
    for i in range(min(len(parent1), len(parent2))):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])

    # Add remaining elements from longer parent
    if len(parent1) > len(parent2):
        child.extend(parent1[len(parent2):])
    elif len(parent2) > len(parent1):
        child.extend(parent2[len(parent1):])

    return child

def mutate(sequence, mutation_rate=0.1, max_height=1000):
    """Mutate a sequence."""
    mutated = copy.deepcopy(sequence)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = random.uniform(0, max_height)
    return mutated

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return copy.deepcopy(population[winner_index])

def evolutionary_optimization(max_generations=100, population_size=50,
                            mutation_rate=0.1, elite_size=5):
    """Main evolutionary optimization loop."""
    # Initialize population
    population = [generate_random_valid_sequence()
                  for _ in range(population_size)]

    best_score = 0
    best_individual = None
    stagnation_counter = 0
    max_stagnation = 20

    start_time = time.time()

    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitnesses = []
        for individual in population:
            inv_c1 = compute_inv_c1(individual)
            # Only consider sequences with sufficient sum
            if np.sum(individual) > 0.01:
                fitnesses.append(inv_c1)
            else:
                fitnesses.append(0)

        # Track best individual
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > best_score:
            best_score = fitnesses[best_idx]
            best_individual = copy.deepcopy(population[best_idx])
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Check for stagnation
        if stagnation_counter >= max_stagnation:
            break

        # Create new population
        new_population = []

        # Elitism: keep the best individuals
        sorted_indices = np.argsort(fitnesses)[::-1][:elite_size]
        for idx in sorted_indices:
            new_population.append(copy.deepcopy(population[idx]))

        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate(child, mutation_rate)

            new_population.append(child)

        population = new_population[:population_size]

        # Check time limit
        if time.time() - start_time > 170:  # Leave some buffer
            break

    return best_individual, best_score

def search_for_best_sequence():
    """Function to search for the best coefficient sequence using evolutionary approach."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Run evolutionary optimization
    best_sequence, best_score = evolutionary_optimization(
        max_generations=100,
        population_size=50,
        mutation_rate=0.1,
        elite_size=5
    )

    # Ensure we have a valid sequence
    if best_sequence is None:
        best_sequence = generate_random_valid_sequence()

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")