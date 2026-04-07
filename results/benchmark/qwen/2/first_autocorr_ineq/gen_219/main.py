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

    # Use FFT-based convolution for efficiency
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

def get_gradient_direction(sequence):
    """Compute gradient direction for sequence optimization."""
    n = len(sequence)
    if n == 0:
        return None

    # Compute current convolution
    conv = compute_convolution_fft(sequence)
    max_b = np.max(conv)
    sum_a = np.sum(sequence)

    if sum_a < 0.01 or max_b <= 1e-12:
        return None

    # Compute gradient w.r.t. each element of sequence
    grad = np.zeros(n)
    for i in range(n):
        # Approximate gradient using finite differences
        eps = 1e-6
        seq_plus = sequence.copy()
        seq_minus = sequence.copy()
        seq_plus[i] += eps
        seq_minus[i] -= eps

        conv_plus = compute_convolution_fft(seq_plus)
        conv_minus = compute_convolution_fft(seq_minus)

        d_max_plus = np.max(conv_plus)
        d_max_minus = np.max(conv_minus)

        grad[i] = (d_max_plus - d_max_minus) / (2 * eps)

    # Normalize gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-12:
        grad = grad / grad_norm

    return grad

def get_good_direction_to_move_into(
    sequence: list[float],
    iteration: int = 0,
    max_iterations: int = 1000
) -> list[float] | None:
    """Returns the direction to move into the sequence using gradient approach."""
    n = len(sequence)
    if n == 0:
        return None

    # Get gradient direction
    grad = get_gradient_direction(sequence)
    if grad is None:
        # Fallback to random perturbation
        t = 0.01
        new_sequence = [(1 - t) * x + t * (x + np.random.normal(0, 0.01)) for x in sequence]
        return new_sequence

    # Use adaptive step size
    current_fitness = evaluate_fitness(sequence)
    t = min(0.1, max(0.001, 1.0 / (1.0 + current_fitness)))

    # Move along negative gradient direction (steepest descent)
    new_sequence = [
        max(0.0, x - t * g) for x, g in zip(sequence, grad)
    ]

    # Ensure minimum positive value
    if np.sum(new_sequence) < 0.01:
        new_sequence = [max(0.1, x) for x in new_sequence]

    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS using FFT-based constraint generation."""
    n = len(f_sequence)
    c = -np.ones(n)

    # Use FFT-based approach to generate constraints more efficiently
    # Create convolution constraints without explicitly forming full Toeplitz matrix
    a_ub = []
    b_ub = []

    # The key insight is that convolution constraint can be expressed using FFT
    # But since we're doing linear programming, we still need to represent it as matrix
    # So we'll use a smarter way to generate the constraints

    # Generate constraints using a more numerically stable approach
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
        # Use 'highs' method for better numerical stability
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except Exception as e:
        # Fallback to a simple heuristic approach in case of LP failure
        return None

def generate_initial_sequence():
    """Generate a better initial sequence to start optimization."""
    n = random.randint(100, 1000)
    # Start with a sequence that balances high and low values
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
    elite_sequences = [best_sequence]

    # Iteration counter for adaptive step sizing
    iteration = 0
    max_iterations = 1000

    while time.time() - start_time < max_time_seconds and iteration < max_iterations:
        iteration += 1

        # Occasionally use gradient-based direction
        if np.random.random() < 0.3:
            h_function = get_good_direction_to_move_into(best_sequence, iteration, max_iterations)
            if h_function is not None:
                # Simple acceptance criterion
                new_fitness = evaluate_fitness(h_function)
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_sequence = h_function.copy()

                    # Keep top performers
                    elite_sequences.append(best_sequence)
                    if len(elite_sequences) > 10:
                        # Keep only the best elites
                        elite_fitnesses = [evaluate_fitness(s) for s in elite_sequences]
                        top_indices = np.argsort(elite_fitnesses)[-5:]  # Keep top 5
                        elite_sequences = [elite_sequences[i] for i in top_indices]
        else:
            # Random mutation
            new_sequence = []
            for x in best_sequence:
                # Add a small random perturbation
                delta = np.random.normal(0, 0.1 * np.mean(best_sequence) if np.mean(best_sequence) > 0 else 0.1)
                new_sequence.append(max(0.0, x + delta))

            new_fitness = evaluate_fitness(new_sequence)
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_sequence = new_sequence

                # Keep top performers
                elite_sequences.append(best_sequence)
                if len(elite_sequences) > 10:
                    # Keep only the best elites
                    elite_fitnesses = [evaluate_fitness(s) for s in elite_sequences]
                    top_indices = np.argsort(elite_fitnesses)[-5:]  # Keep top 5
                    elite_sequences = [elite_sequences[i] for i in top_indices]

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")