# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import convolve as fft_convolve
import random
import time

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1(sequence):
    """Compute C1 value for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')

    # Use FFT for faster convolution
    conv_result = fft_convolve(sequence, sequence, mode='full')[:len(sequence)]
    max_conv = np.max(conv_result)

    # C1 = 2n * max(b) / (sum(a))^2
    n = len(sequence)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    return c1

def compute_inv_c1(sequence):
    """Compute 1/C1 value for a given sequence."""
    c1 = compute_c1(sequence)
    if c1 == float('inf'):
        return 0.0
    return 1.0 / c1

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    try:
        n = len(sequence)
        if n == 0:
            return None

        sum_sequence = np.sum(sequence)
        if sum_sequence < 0.01:
            return None

        # Normalize sequence
        normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence

        # Compute convolution maximum using FFT for efficiency
        conv_result = fft_convolve(normalized_sequence, normalized_sequence, mode='full')[:len(normalized_sequence)]
        rhs = np.max(conv_result)

        g_fun = solve_convolution_lp(normalized_sequence, rhs)
        if g_fun is None:
            # Fallback to random perturbation
            return [max(0, x + random.uniform(-0.1, 0.1)) for x in sequence]

        sum_g_fun = np.sum(g_fun)
        if sum_g_fun < 0.01:
            return [max(0, x + random.uniform(-0.1, 0.1)) for x in sequence]

        normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g_fun

        # Adaptive step size
        iteration = getattr(get_good_direction_to_move_into, 'iteration', 0)
        t = 0.01 * np.exp(-iteration / 100)
        get_good_direction_to_move_into.iteration = iteration + 1

        new_sequence = [
            max(0, (1 - t) * x + t * y) for x, y in zip(sequence, normalized_g_fun)
        ]
        return new_sequence
    except Exception as e:
        print(f"Error in get_good_direction_to_move_into: {e}")
        return None

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    try:
        n = len(f_sequence)
        if n == 0:
            return None

        # Use FFT-based constraint matrix to make it more efficient
        c = -np.ones(n)
        a_ub = []
        b_ub = []

        # Build constraint matrix using FFT-like approach
        for k in range(2 * n - 1):
            row = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_sequence[i]
            a_ub.append(row)
            b_ub.append(rhs)

        # Add non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        # Try to solve with different methods if default fails
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            g_sequence = result.x
            # Ensure non-negative values
            g_sequence = np.maximum(g_sequence, 0)
            return g_sequence
        else:
            # Try with different solver options
            try:
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='interior-point')
                if result.success:
                    g_sequence = result.x
                    g_sequence = np.maximum(g_sequence, 0)
                    return g_sequence
            except:
                pass

            # Fallback to simple approach
            return np.maximum(np.random.rand(n) * rhs, 0)

    except Exception as e:
        print(f"Error in solve_convolution_lp: {e}")
        return None

def initialize_sequence():
    """Initialize a good starting sequence."""
    # Try symmetric initialization first
    n = random.randint(100, 1000)
    # Create a symmetric sequence with decreasing values
    sequence = []
    for i in range(n):
        # Generate values that decrease from center
        val = max(0, 1.0 - abs(i - n//2) / (n//2))
        sequence.append(val)

    # Add some randomness
    sequence = [max(0, x + random.uniform(-0.1, 0.1)) for x in sequence]
    return sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Initial population
    population_size = 20
    population = []

    # Initialize diverse starting points
    for _ in range(population_size):
        sequence = initialize_sequence()
        population.append(sequence)

    best_sequence = None
    best_inv_c1 = -float('inf')
    max_iterations = 1000
    convergence_threshold = 1e-5

    start_time = time.time()

    for iteration in range(max_iterations):
        if time.time() - start_time > 170:  # Leave 10 seconds for cleanup
            break

        # Evaluate population
        fitness_scores = []
        for seq in population:
            inv_c1 = compute_inv_c1(seq)
            fitness_scores.append(inv_c1)

        # Find best in current population
        current_best_idx = np.argmax(fitness_scores)
        current_best_sequence = population[current_best_idx]
        current_best_inv_c1 = fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = current_best_sequence.copy()

        # Elitism - preserve top performers
        elite_count = max(1, population_size // 5)
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
        elite_sequences = [population[i] for i in sorted_indices]

        # Generate new population
        new_population = elite_sequences.copy()

        # Fill rest with mutated versions of elite sequences
        while len(new_population) < population_size:
            parent = random.choice(elite_sequences)
            child = get_good_direction_to_move_into(parent)
            if child is not None:
                # Add some noise to maintain diversity
                noise_factor = 0.05
                child = [max(0, x + random.uniform(-noise_factor, noise_factor)) for x in child]
                new_population.append(child)
            else:
                # Fallback to random initialization
                new_population.append(initialize_sequence())

        population = new_population

        # Check for convergence
        if iteration > 10 and abs(best_inv_c1 - compute_inv_c1(best_sequence)) < convergence_threshold:
            break

    # Final refinement of best sequence
    if best_sequence is not None:
        refined_sequence = get_good_direction_to_move_into(best_sequence)
        if refined_sequence is not None:
            final_inv_c1 = compute_inv_c1(refined_sequence)
            if final_inv_c1 > best_inv_c1:
                best_sequence = refined_sequence

    return best_sequence if best_sequence is not None else initialize_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")