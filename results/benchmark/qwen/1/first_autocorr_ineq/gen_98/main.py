# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from typing import List, Optional
import random
import time

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_c1(sequence: List[float]) -> float:
    """Compute C₁ constant for the given sequence."""
    n = len(sequence)
    if n == 0:
        return float('inf')

    # Use FFT for efficient convolution
    padded_seq = np.pad(sequence, (0, n - 1), 'constant')
    conv_result = np.real(ifft(fft(padded_seq) * np.conj(fft(sequence))))
    max_conv = np.max(conv_result[:2*n-1])

    # Calculate C₁
    sum_sq = np.sum(sequence) ** 2
    if sum_sq == 0:
        return float('inf')

    c1 = 2 * n * max_conv / sum_sq
    return c1

def compute_inv_c1(sequence: List[float]) -> float:
    """Compute 1/C₁ for the given sequence."""
    c1 = compute_c1(sequence)
    if c1 == 0:
        return float('inf')
    return 1.0 / c1

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)

    # Avoid division by zero
    if sum_sequence < 1e-10:
        return None

    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Use FFT for faster convolution
    try:
        conv_result = np.real(ifft(fft(normalized_sequence, 2*n-1) *
                                   np.conj(fft(normalized_sequence, 2*n-1))))
        rhs = np.max(conv_result[:2*n-1])  # Only consider the actual convolution results
    except Exception as e:
        print(f"Error during FFT convolution: {e}")
        return None

    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None

    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Precompute the convolution constraint matrix using explicit loop
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            print('LP optimization failed:', result.message)
            return None
    except Exception as e:
        print(f'LP optimization error: {e}')
        return None

def adaptive_tournament_selection(population: List[List[float]],
                                fitness_scores: List[float],
                                generation: int,
                                population_size: int) -> List[float]:
    """Perform adaptive tournament selection based on diversity and generation."""

    # Determine tournament size based on generation and population diversity
    if generation <= 20:  # Early generations
        tournament_size = min(9, max(5, population_size // 4))
    elif generation >= 50:  # Later generations
        tournament_size = min(4, max(3, population_size // 8))
    else:  # Middle generations
        tournament_size = 5

    # Perform tournament selection
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]

    return population[winner_idx].copy()

def generate_structured_sequence(length: int) -> List[float]:
    """Generate a more structured sequence that potentially performs better."""
    # Create a sequence with some inherent structure
    sequence = []

    # Mix of exponential decay and step-like patterns
    for i in range(length):
        # Exponential decay component
        exp_component = 100 * np.exp(-i * 0.01)
        # Add some periodic variations
        period_component = 10 * np.sin(i * 0.2) * np.cos(i * 0.05)
        # Combine components
        val = max(0.01, exp_component + period_component)
        sequence.append(val)

    return sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply mutation to a sequence."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply Gaussian noise
            mutated[i] = max(0, mutated[i] + np.random.normal(0, mutated[i] * 0.1))
    return mutated

def evolve_population(population: List[List[float]],
                     fitness_scores: List[float],
                     generation: int,
                     elite_count: int = 2) -> List[List[float]]:
    """Evolve the population for one generation."""
    # Sort by fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elites = [population[i] for i in sorted_indices[:elite_count]]

    # Select parents and create offspring
    new_population = elites.copy()

    # Fill remaining slots with offspring
    while len(new_population) < len(population):
        parent = adaptive_tournament_selection(population, fitness_scores, generation, len(population))
        offspring = mutate_sequence(parent)
        new_population.append(offspring)

    return new_population[:len(population)]

def search_for_best_sequence(max_generations: int = 100) -> list[float]:
    """Function to search for the best coefficient sequence using evolutionary algorithm."""
    # Parameters
    population_size = 20
    elite_count = 2
    max_stagnation = 20

    # Initialize population
    population = []
    for _ in range(population_size):
        n = np.random.randint(100, 1000)
        sequence = generate_structured_sequence(n)
        population.append(sequence)

    # Evaluate initial population
    fitness_scores = [compute_inv_c1(seq) for seq in population]

    # Track best
    best_idx = np.argmax(fitness_scores)
    best_sequence = population[best_idx].copy()
    best_inv_c1 = fitness_scores[best_idx]
    stagnation_counter = 0

    # Evolution loop
    start_time = time.time()
    for generation in range(max_generations):
        # Early stopping if benchmark is beaten
        if best_inv_c1 > 1.0 / 1.5031:
            print(f"Beaten benchmark at generation {generation}!")
            break

        # Evolve population
        population = evolve_population(population, fitness_scores, generation, elite_count)

        # Evaluate new population
        fitness_scores = [compute_inv_c1(seq) for seq in population]

        # Update best
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_inv_c1:
            best_idx = current_best_idx
            best_sequence = population[best_idx].copy()
            best_inv_c1 = fitness_scores[best_idx]
            stagnation_counter = 0
            print(f"Generation {generation}: New best inv_C1: {best_inv_c1:.6f}")
        else:
            stagnation_counter += 1

        # Check for stagnation
        if stagnation_counter >= max_stagnation:
            print(f"Stagnation detected at generation {generation}, restarting with new population")
            # Restart with new random sequences
            population = []
            for _ in range(population_size):
                n = np.random.randint(100, 1000)
                sequence = generate_structured_sequence(n)
                population.append(sequence)
            fitness_scores = [compute_inv_c1(seq) for seq in population]
            best_idx = np.argmax(fitness_scores)
            best_sequence = population[best_idx].copy()
            best_inv_c1 = fitness_scores[best_idx]
            stagnation_counter = 0

        # Timeout check
        if time.time() - start_time > 170:  # Leave 10 seconds for cleanup
            print("Timeout reached")
            break

    # Final refinement of best sequence
    refined_sequence = get_good_direction_to_move_into(best_sequence)
    if refined_sequence is not None:
        final_inv_c1 = compute_inv_c1(refined_sequence)
        if final_inv_c1 > best_inv_c1:
            best_sequence = refined_sequence
            best_inv_c1 = final_inv_c1
            print(f"Final refinement improved to inv_C1: {best_inv_c1:.6f}")

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")