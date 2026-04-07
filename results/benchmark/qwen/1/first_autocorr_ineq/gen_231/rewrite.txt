# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from typing import List, Optional
import random
import time
from scipy.optimize import minimize

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autocorrelation_constant(sequence: List[float]) -> tuple[float, float]:
    """Computes the autocorrelation constant C1 and its reciprocal 1/C1."""
    if not sequence or sum(sequence) < 0.01:
        return (float('inf'), 0.0)

    n = len(sequence)
    # Use FFT-based convolution for efficiency O(n log n)
    conv = np.convolve(sequence, sequence, mode='full')
    max_conv = np.max(conv)
    sum_seq = sum(sequence)

    if sum_seq == 0:
        return (float('inf'), 0.0)

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return (c1, inv_c1)

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

def local_refinement(sequence: List[float], max_iter: int = 100) -> List[float]:
    """
    Apply local refinement to improve the sequence using gradient-based optimization.
    """
    # Convert to numpy array for easier manipulation
    x0 = np.array(sequence)
    n = len(x0)

    # Define objective function to minimize (negative of 1/C1)
    def objective(x):
        # Ensure non-negativity and avoid near-zero values
        x = np.maximum(x, 1e-6)
        c1, _ = compute_autocorrelation_constant(x.tolist())
        return -1.0 / c1 if c1 > 0 else 1e6

    # Define bounds
    bounds = [(1e-6, 1000.0) for _ in range(n)]

    # Use L-BFGS-B for local optimization
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter})
        if result.success:
            refined = np.maximum(result.x, 1e-6).tolist()
            return refined
    except Exception as e:
        print(f"Local refinement error: {e}")

    return sequence

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

def generate_step_based_sequence(length: int) -> List[float]:
    """Generate a step-based sequence pattern."""
    # Create a few prominent steps
    sequence = [0.0] * length
    num_steps = max(2, min(20, length // 10))
    step_positions = sorted(random.sample(range(length), num_steps))

    for i, pos in enumerate(step_positions):
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = length
        height = random.uniform(10.0, 100.0)
        sequence[pos:end_pos] = [height] * (end_pos - pos)

    return sequence

def generate_population(population_size: int, length_range=(100, 1000)) -> List[List[float]]:
    """Generate a diverse initial population."""
    population = []

    # Generate sequences using different methods
    for _ in range(population_size // 3):
        n = random.randint(*length_range)
        population.append(generate_structured_sequence(n))

    for _ in range(population_size // 3):
        n = random.randint(*length_range)
        population.append(generate_step_based_sequence(n))

    # Fill remaining with standard random sequences
    while len(population) < population_size:
        n = random.randint(*length_range)
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
        population.append(sequence)

    return population

def compute_convolution_fft(sequence: np.ndarray) -> np.ndarray:
    """
    Computes the autoconvolution of a sequence using FFT for efficiency.
    Returns the convolution result up to the valid length.
    """
    n = len(sequence)
    padded_len = 2 * n - 1
    padded_seq = np.pad(sequence, (0, padded_len - n), 'constant')
    fft_seq = fft(padded_seq)
    conv_result = ifft(fft_seq * np.conj(fft_seq))
    return np.real(conv_result[:padded_len])

def evaluate_fitness(sequence: List[float]) -> tuple[float, float]:
    """
    Evaluates the fitness of a sequence by computing C₁.
    Returns (inverse_C1, C1) tuple.
    """
    if len(sequence) == 0:
        return 0.0, float('inf')

    a = np.array(sequence)
    sum_a = np.sum(a)

    # Avoid division by zero
    if sum_a < 1e-10:
        return 0.0, float('inf')

    # Compute autoconvolution
    b = compute_convolution_fft(a)
    max_b = np.max(b)

    # Compute C1
    n = len(sequence)
    c1 = (2 * n * max_b) / (sum_a ** 2)

    # Return inverse for maximization
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return inv_c1, c1

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Start with a diverse population
    population_size = 20
    population = generate_population(population_size)
    best_sequence = None
    best_inv_c1 = 0.0
    stagnation_count = 0
    max_stagnation = 20
    max_generations = 100
    time_limit = 180  # seconds
    start_time = time.time()

    # Main evolutionary loop
    for generation in range(max_generations):
        if time.time() - start_time > time_limit:
            break

        # Evaluate fitness for all individuals
        fitness_scores = []
        for seq in population:
            inv_c1, _ = evaluate_fitness(seq)
            fitness_scores.append(inv_c1)

        # Update best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_inv_c1 = fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Early termination if we've achieved good performance
        if best_inv_c1 > 0.6653 and stagnation_count > 10:
            break

        # Local refinement for top performers
        top_performers = sorted(range(len(population)),
                               key=lambda i: fitness_scores[i], reverse=True)[:5]
        for idx in top_performers:
            refined = local_refinement(population[idx])
            inv_c1, _ = evaluate_fitness(refined)
            if inv_c1 > fitness_scores[idx]:
                population[idx] = refined
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = refined

        # Selection and reproduction
        selected_parents = []

        # Tournament selection for better exploration
        for _ in range(population_size // 2):
            selected_parents.append(adaptive_tournament_selection(population, fitness_scores, generation, population_size))

        # Elitism: keep the best individual
        selected_parents.insert(0, best_sequence.copy())

        # Create new population
        new_population = [best_sequence.copy()]

        # Crossover and mutation
        while len(new_population) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Uniform crossover
            child = []
            for i in range(min(len(parent1), len(parent2))):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])

            # Mutation with adaptive rate
            mutation_rate = max(0.05, 0.3 * (1 - generation / max_generations))
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    child[i] = max(0.01, child[i] + random.gauss(0, 1.0))

            new_population.append(child)

        population = new_population[:population_size]

        # Debug output every few generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best inv_C1 = {best_inv_c1:.4f}")

    # Final refinement of the best sequence
    if best_sequence is not None:
        refined = local_refinement(best_sequence)
        inv_c1, _ = evaluate_fitness(refined)
        if inv_c1 > best_inv_c1:
            best_sequence = refined

    return best_sequence if best_sequence is not None else generate_structured_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")