# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time

def compute_convolution_fft(seq):
    """Compute the autoconvolution using FFT for efficiency."""
    n = len(seq)
    padded_seq = np.pad(seq, (0, n-1), mode='constant')
    conv_result = ifft(fft(padded_seq) * np.conj(fft(padded_seq)))
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
    if c1 == float('inf'):
        return 0.0  # Penalty for invalid sequences
    return 1.0 / c1

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
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
            return None
    except:
        return None

def get_good_direction_to_move_into(
    sequence: list[float],
    iteration: int = 0,
    max_iterations: int = 1000
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    if sum_sequence < 0.01:
        return None

    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    conv = compute_convolution_fft(normalized_sequence)
    rhs = np.max(conv)

    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        # Fallback to simple gradient descent on a smaller scale
        t = 0.001
        new_sequence = [(1 - t) * x + t * (x + np.random.normal(0, 0.01)) for x in sequence]
        return new_sequence

    sum_g = np.sum(g_fun)
    if sum_g < 0.01:
        return None

    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]

    # Adaptive step size based on iteration
    t = max(0.001, 0.1 * (1 - iteration / max_iterations))

    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]

    # Ensure non-negativity
    new_sequence = [max(0.0, x) for x in new_sequence]
    return new_sequence

def generate_initial_sequence():
    """Generate a better initial sequence to start optimization."""
    n = random.randint(100, 1000)
    # Try to generate a sequence that's likely to perform well
    # Start with some random structure that's known to work
    sequence = []
    for i in range(n):
        # Mix of different values to avoid trivial solutions
        if i % 5 == 0:
            sequence.append(random.uniform(100, 1000))
        else:
            sequence.append(random.uniform(0, 100))
    return sequence

def search_for_best_sequence(max_time_seconds=170) -> list[float]:
    """Function to search for the best coefficient sequence."""
    start_time = time.time()

    # Initialize with a good starting sequence
    best_sequence = generate_initial_sequence()
    best_fitness = evaluate_fitness(best_sequence)

    # Track elite sequences for preservation
    elite_sequences = [best_sequence.copy()]

    # Iteration counter for adaptive step sizing
    iteration = 0
    max_iterations = 1000

    while time.time() - start_time < max_time_seconds and iteration < max_iterations:
        iteration += 1

        # Periodically evaluate and maintain elites
        if iteration % 10 == 0:
            # Sort elites by fitness and keep top performers
            elite_fitnesses = [evaluate_fitness(s) for s in elite_sequences]
            sorted_indices = np.argsort(elite_fitnesses)[::-1]  # Descending order
            elite_sequences = [elite_sequences[i] for i in sorted_indices[:10]]

        h_function = get_good_direction_to_move_into(best_sequence, iteration, max_iterations, elite_sequences)

        if h_function is not None:
            candidate_sequence = h_function
            candidate_fitness = evaluate_fitness(candidate_sequence)

            # Keep elite sequences
            if candidate_fitness > best_fitness:
                best_sequence = candidate_sequence
                best_fitness = candidate_fitness

                # Keep top performers
                elite_sequences.append(candidate_sequence)
                if len(elite_sequences) > 10:
                    # Keep only the best elites
                    elite_fitnesses = [evaluate_fitness(s) for s in elite_sequences]
                    top_indices = np.argsort(elite_fitnesses)[-5:]  # Keep top 5
                    elite_sequences = [elite_sequences[i] for i in top_indices]
        else:
            # Fallback - slightly perturb current sequence
            new_sequence = []
            for x in best_sequence:
                new_sequence.append(max(0.0, x + random.uniform(-1, 1)))
            if evaluate_fitness(new_sequence) > best_fitness:
                best_sequence = new_sequence

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")