# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time
from typing import List, Tuple
from scipy.optimize import differential_evolution, minimize
import copy
import math

def convolve_fft(a: List[float], b: List[float]) -> List[float]:
    """
    Compute convolution using FFT for better performance.
    Returns the convolution of a and b.
    """
    n = len(a)
    if n == 0:
        return []

    # Pad to length 2*n - 1 for full convolution
    padded_length = 2 * n - 1
    fa = fft(a, padded_length)
    fb = fft(b, padded_length)
    result = ifft(fa * fb.conj()).real
    # Return only the valid convolution part
    return result[:n].tolist()

def compute_c1(sequence: List[float]) -> float:
    """
    Compute the C1 constant for a given sequence.
    C1 = 2n * max(auto_correlation) / (sum(sequence))^2
    """
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

def evaluate_fitness(sequence: List[float]) -> float:
    """
    Evaluate fitness as inverse of C1 (higher is better).
    Returns 0.0 if invalid sequence.
    """
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 <= 0:
        return 0.0
    return 1.0 / c1

def calculate_autocorrelation_variance(sequence: List[float]) -> float:
    """
    Calculate the variance of autocorrelation values to penalize erratic profiles.
    """
    n = len(sequence)
    if n == 0:
        return 0.0

    conv = convolve_fft(sequence, sequence)
    return np.var(conv)

def generate_structured_sequence(n: int) -> List[float]:
    """
    Generate a structured sequence that's likely to perform well.
    Uses exponential decay pattern to reduce autocorrelation peaks.
    """
    sequence = []
    for i in range(n):
        # Exponential decay with some noise to break symmetry
        base_val = max(0.01, 100 * np.exp(-i * 0.05))
        noise = random.uniform(0.9, 1.1)
        sequence.append(base_val * noise)
    return sequence

def generate_memory_based_sequence(elite_sequences: List[List[float]], n: int) -> List[float]:
    """
    Generate a sequence based on learned patterns from elite sequences.
    """
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
    """
    Generate diverse initial population with structured sequences.
    """
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
    """
    Apply mutation to sequence with multiplicative Gaussian perturbation.
    """
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply multiplicative Gaussian perturbation
            perturbation = random.gauss(1, 0.1)
            mutated[i] *= abs(perturbation)  # Ensure non-negative
            mutated[i] = max(0.01, mutated[i])
    return mutated

def crossover_sequences(parent1: List[float], parent2: List[float]) -> List[float]:
    """
    Perform crossover between two sequences.
    """
    min_len = min(len(parent1), len(parent2))
    crossover_point = random.randint(1, min_len - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def local_search_refinement(sequence: List[float], max_iterations: int = 10) -> List[float]:
    """
    Apply local search refinement to improve sequence.
    Uses a combination of gradient-free optimization and random perturbations.
    """
    best_seq = sequence.copy()
    best_fitness = evaluate_fitness(best_seq)

    for _ in range(max_iterations):
        # Try small perturbations
        mutant = mutate_sequence(best_seq, 0.05)
        mutant_fitness = evaluate_fitness(mutant)

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
                best_test_fitness = evaluate_fitness(new_seq)

                for test_val in test_vals:
                    test_seq = new_seq.copy()
                    test_seq[i] = max(0.01, test_val)
                    test_fitness = evaluate_fitness(test_seq)
                    if test_fitness > best_test_fitness:
                        best_test = test_val
                        best_test_fitness = test_fitness

                new_seq[i] = best_test

            test_fitness = evaluate_fitness(new_seq)
            if test_fitness > best_fitness:
                best_seq = new_seq
                best_fitness = test_fitness

        except Exception:
            pass  # Skip if optimization fails

    return best_seq

def adaptive_mutation_rate(population_fitnesses: List[float]) -> float:
    """
    Calculate adaptive mutation rate based on population diversity.
    """
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

def multi_start_local_search(sequence: List[float], num_starts: int = 5) -> List[float]:
    """
    Perform multi-start local search to find better local optima.
    """
    best_seq = sequence.copy()
    best_fitness = evaluate_fitness(best_seq)

    for _ in range(num_starts):
        # Perturb the sequence slightly
        perturbed = mutate_sequence(sequence, 0.05)
        refined = local_search_refinement(perturbed, 10)
        refined_fitness = evaluate_fitness(refined)
        
        if refined_fitness > best_fitness:
            best_seq = refined
            best_fitness = refined_fitness

    return best_seq

def adaptive_tournament_selection(population: List[List[float]], 
                                 fitness_scores: List[float], 
                                 diversity: float) -> List[float]:
    """
    Perform adaptive tournament selection based on population diversity.
    """
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
    Improved evolutionary optimization to find the best sequence.
    Uses FFT-based convolution, adaptive selection, and local search.
    """
    start_time = time.time()
    max_time = 175  # Leave some time for cleanup

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

        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual.copy()

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

        # Apply multi-start local search to elite members
        refined_elite = []
        for ind in elite:
            refined = multi_start_local_search(ind)
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

    # Final multi-start local search on best solution
    if best_solution is not None:
        best_solution = multi_start_local_search(best_solution, 10)

    return best_solution if best_solution is not None else generate_structured_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")