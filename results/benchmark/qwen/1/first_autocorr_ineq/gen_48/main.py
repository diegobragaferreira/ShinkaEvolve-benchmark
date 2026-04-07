# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution
import time
import random
from numba import jit

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@jit(nopython=True)
def compute_autocorrelation_fast(sequence: np.ndarray) -> float:
    """
    Fast computation of autocorrelation constant C₁ using numba acceleration.
    """
    n = len(sequence)
    if n == 0:
        return float('inf')

    # Compute autoconvolution manually for speed
    autocorr = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            autocorr[i+j] += sequence[i] * sequence[j]

    max_autocorr = np.max(autocorr[n-1:])
    sum_sq = np.sum(sequence)**2

    if sum_sq == 0:
        return float('inf')

    C1 = 2 * n * max_autocorr / sum_sq
    return C1

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    """
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return float('inf')

    # Use fast computation for small sequences
    if len(sequence) <= 1000:
        return compute_autocorrelation_fast(np.array(sequence))

    # Compute autocorrelation using FFT for efficiency
    autocorr = fftconvolve(sequence, sequence[::-1], mode='full')
    # Take the second half which corresponds to the actual autocorrelation
    autocorr = autocorr[len(sequence)-1:]
    max_autocorr = np.max(autocorr)

    sum_sq = np.sum(sequence)**2
    if sum_sq == 0:
        return float('inf')

    # C₁ = 2n * max(autocorr) / (sum(sequence))^2
    C1 = 2 * len(sequence) * max_autocorr / sum_sq

    return C1

def objective_function(sequence):
    """
    Objective function to maximize (inverse of C₁).
    """
    C1 = compute_autocorrelation_constant(sequence)
    if C1 == float('inf'):
        return 0.0  # Invalid sequences get penalized
    return 1.0 / C1  # Return 1/C1 as the objective

def generate_structured_sequence(n):
    """
    Generate a structured sequence using sine waves for better initialization.
    """
    sequence = []
    for i in range(n):
        # Sine wave with some randomness
        sine_component = abs(np.sin(np.pi * i / n))
        noise_component = np.random.random() * 0.1
        sequence.append(sine_component + noise_component)
    return sequence

def generate_random_sequence(n):
    """
    Generate a random sequence with normalization.
    """
    seq = np.random.rand(n)
    # Normalize to ensure sum > 0.01
    seq = seq * (0.01 / (np.sum(seq) + 1e-10))
    return seq.tolist()

def adaptive_mutation(parent, diversity_factor):
    """
    Apply adaptive mutation based on population diversity.
    """
    child = parent.copy()
    mutation_rate = max(0.01, min(0.5, 0.2 + 0.3 * (1 - diversity_factor)))

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

def crossover(parent1, parent2):
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

def adaptive_evolutionary_search(max_time_seconds=180):
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

    # Initialize population with structured sequences
    population = []
    for _ in range(population_size):
        n = np.random.randint(min_sequence_length, max_sequence_length)
        # Use structured sequences for better initial quality
        seq = generate_structured_sequence(n)
        population.append(seq)

    best_fitness = 0.0
    best_sequence = None
    generation = 0
    stagnation_count = 0

    while time.time() - start_time < max_time_seconds and generation < max_generations:
        # Evaluate fitness for all individuals
        fitness_scores = []
        for seq in population:
            fitness = objective_function(seq)
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
                population[i] = adaptive_mutation(sorted_population[parent_idx], 0.5)
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

def get_gradient_estimate(sequence, epsilon=1e-4):
    """
    Estimate gradient using finite differences.
    """
    n = len(sequence)
    if n == 0:
        return None

    grad = []
    for i in range(n):
        # Create perturbed sequences
        seq_plus = sequence.copy()
        seq_minus = sequence.copy()

        seq_plus[i] += epsilon
        seq_minus[i] -= epsilon

        # Evaluate both and estimate derivative
        val_plus = objective_function(seq_plus)
        val_minus = objective_function(seq_minus)

        grad_i = (val_plus - val_minus) / (2 * epsilon)
        grad.append(grad_i)

    return np.array(grad)

def refine_with_gradient_descent(sequence, max_iter=100, learning_rate=0.01):
    """
    Refine the sequence using gradient descent.
    """
    current_seq = sequence.copy()
    for _ in range(max_iter):
        grad = get_gradient_estimate(current_seq)
        if grad is None:
            break

        # Gradient ascent (since we're maximizing)
        current_seq = [max(0, x + learning_rate * g) for x, g in zip(current_seq, grad)]

        # Normalize to keep sum reasonable
        sum_seq = np.sum(current_seq)
        if sum_seq > 0.01:
            current_seq = [x * 0.01 / sum_seq for x in current_seq]

    return current_seq

def search_for_best_sequence():
    """
    Main function to find the best coefficient sequence using adaptive evolutionary optimization.
    """
    try:
        # Run adaptive evolutionary search
        best_sequence = adaptive_evolutionary_search()

        # Apply local refinement
        refined_sequence = refine_with_gradient_descent(best_sequence)

        # Evaluate final result
        final_fitness = objective_function(refined_sequence)

        # If refinement improved the result, use it
        if final_fitness > objective_function(best_sequence):
            return refined_sequence
        else:
            return best_sequence
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple structured sequence
        return generate_structured_sequence(1000)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")