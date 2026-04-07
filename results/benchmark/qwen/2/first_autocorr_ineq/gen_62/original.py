# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
import random

def compute_convolution_fft(seq):
    """Compute the autoconvolution using FFT for efficiency."""
    n = len(seq)
    padded_seq = np.pad(seq, (0, n-1), mode='constant')
    conv_result = np.fft.ifft(np.fft.fft(padded_seq) * np.conj(np.fft.fft(padded_seq)))
    return np.real(conv_result[:n])

def calculate_c1(sequence):
    """Calculate the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    sequence = np.array(sequence)
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')

    conv = compute_convolution_fft(sequence)
    max_b = np.max(conv)
    n = len(sequence)

    # Avoid division by zero or very small numbers
    if max_b <= 1e-12:
        return float('inf')

    c1 = (2 * n * max_b) / (sum_a ** 2)
    return c1

def evaluate_fitness(sequence):
    """Evaluate the inverse of C1 as fitness (we want to maximize 1/C1)."""
    c1 = calculate_c1(sequence)
    if c1 == float('inf') or c1 > 10000:
        return 0.0  # Penalty for invalid sequences
    return 1.0 / c1

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    # Normalize sequence for better numerical properties
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute the convolution to determine RHS
    conv = compute_convolution_fft(normalized_sequence)
    rhs = np.max(conv)

    # If RHS is too small, we can't proceed meaningfully
    if rhs <= 1e-12:
        return None

    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None

    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None

    # Normalize the result for consistency
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

    # Adaptive step size based on problem complexity
    t = min(0.05, 1.0 / (n + 1))
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    if n == 0:
        return None

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Build convolution constraint matrix efficiently
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

    # Use a more robust solver configuration
    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub,
                                options={'maxiter': 1000, 'tol': 1e-8})
    except:
        return None

    if result.success:
        g_sequence = result.x
        # Clip negative values that might arise from numerical errors
        g_sequence = np.clip(g_sequence, 0, None)
        return g_sequence
    else:
        return None

def initialize_good_sequence():
    """Initialize sequence with known good patterns."""
    # Try some known good patterns that often perform well
    patterns = [
        # Simple uniform pattern
        [1.0] * 100,
        # Alternating pattern
        [1.0, 0.0] * 50,
        # Exponential decay
        [1.0 / (i + 1) for i in range(100)],
        # Gaussian-like decay
        [np.exp(-i**2 / 200.0) for i in range(100)]
    ]

    # Choose randomly from patterns
    pattern = random.choice(patterns)
    # Add some noise to avoid local optima
    noise_level = 0.1
    noisy_pattern = [max(0.0, x + random.uniform(-noise_level, noise_level) * x)
                     for x in pattern]
    return noisy_pattern

def evolutionary_mutation(sequence, mutation_rate=0.05):
    """Apply evolutionary mutation to a sequence."""
    new_sequence = sequence.copy()

    # Mutate each element with small probability
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Apply multiplicative perturbation to preserve non-negativity
            factor = random.uniform(0.9, 1.1)
            new_sequence[i] = max(0.0, new_sequence[i] * factor)

    # Also occasionally add/subtract elements to vary sequence length
    if random.random() < 0.1 and len(new_sequence) > 10:
        idx = random.randint(0, len(new_sequence) - 1)
        new_sequence[idx] = max(0.0, new_sequence[idx] - random.uniform(0, 10))

    return new_sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Start with a good initialization
    best_sequence = initialize_good_sequence()
    best_fitness = evaluate_fitness(best_sequence)

    # Track diversity to avoid stagnation
    last_improvement = 0
    stagnant_count = 0

    # Main optimization loop
    for iteration in range(2000):
        # Alternate between local and global search
        if iteration % 10 == 0:
            # Global search with evolutionary mutation
            mutated_sequence = evolutionary_mutation(best_sequence)
            mutated_fitness = evaluate_fitness(mutated_sequence)

            if mutated_fitness > best_fitness:
                best_sequence = mutated_sequence
                best_fitness = mutated_fitness
                last_improvement = iteration
                stagnant_count = 0
            else:
                stagnant_count += 1

        else:
            # Local search using LP direction finding
            h_function = get_good_direction_to_move_into(best_sequence)
            if h_function is not None:
                h_fitness = evaluate_fitness(h_function)
                if h_fitness > best_fitness:
                    best_sequence = h_function
                    best_fitness = h_fitness
                    last_improvement = iteration
                    stagnant_count = 0
                elif stagnant_count > 50:
                    # If stuck for too long, restart with a new random sequence
                    best_sequence = initialize_good_sequence()
                    best_fitness = evaluate_fitness(best_sequence)
                    stagnant_count = 0

        # Early stopping criterion
        if iteration - last_improvement > 100:
            break

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")