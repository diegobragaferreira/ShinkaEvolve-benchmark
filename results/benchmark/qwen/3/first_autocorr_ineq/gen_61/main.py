# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time

def compute_convolution_fft(seq):
    """Compute convolution using FFT for better performance."""
    n = len(seq)
    # Pad to next power of 2 for efficient FFT
    padded_len = 1 << (n - 1).bit_length()
    padded_seq = np.pad(seq, (0, padded_len - n), 'constant')

    # FFT-based convolution
    fft_seq = fft(padded_seq)
    conv_fft = fft_seq * fft_seq.conj()
    conv_result = ifft(conv_fft).real[:2*n-1]

    return conv_result

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    """
    if len(sequence) == 0:
        return float('inf'), 0.0

    a = np.array(sequence, dtype=np.float64)
    n = len(a)

    # Compute convolution using FFT for efficiency
    conv = compute_convolution_fft(a)
    max_b = np.max(conv)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    C1 = 2 * n * max_b / (sum_a ** 2)
    inv_C1 = 1 / C1

    return C1, inv_C1

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP with improved numerical stability and performance."""
    n = len(f_sequence)
    if n == 0:
        return None

    try:
        # Efficiently construct constraint matrix
        # Using a sparse representation for large n
        a_ub = []
        b_ub = []

        # Generate convolution constraints
        for k in range(2 * n - 1):
            row = np.zeros(n)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    row[j] = f_sequence[i]
            a_ub.append(row)
            b_ub.append(rhs)

        # Non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        # Objective: minimize sum of variables (which corresponds to maximizing 1/C1)
        c = np.ones(n)

        # Solve with bounds
        bounds = [(0, 1000) for _ in range(n)]  # Clip values to reasonable bounds

        # Use 'highs' solver for better performance
        result = optimize.linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'time_limit': 5}
        )

        if result.success:
            return result.x
        else:
            return None

    except Exception as e:
        return None

def get_good_direction_to_move_into(sequence):
    """
    Returns the direction to move into the sequence using LP optimization.
    """
    n = len(sequence)
    if n == 0:
        return None

    sum_sequence = np.sum(sequence)
    # Prevent division by zero or near-zero sums
    if sum_sequence < 1e-10:
        return None

    # Normalize sequence to have a specific scale
    normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence

    try:
        # Compute maximum convolution value using FFT
        conv = compute_convolution_fft(normalized_sequence)
        rhs = np.max(conv)

        # Solve LP optimization
        g_fun = solve_convolution_lp(normalized_sequence, rhs)

        if g_fun is None or np.any(np.isnan(g_fun)):
            return None

        # Normalize the resulting sequence
        sum_g = np.sum(g_fun)
        if sum_g < 1e-8:
            return None

        normalized_g_fun = g_fun * np.sqrt(2 * n) / max(sum_g, 1e-10)

        # Apply small perturbation to maintain diversity
        t = 0.02  # Slightly increased perturbation for better exploration
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
        ]

        # Ensure non-negativity
        new_sequence = [max(0, x) for x in new_sequence]

        return new_sequence

    except Exception as e:
        return None

def adaptive_search_step(current_sequence, max_iter=50):
    """
    Performs adaptive search to improve the sequence.
    """
    # Try LP-based direction first
    lp_direction = get_good_direction_to_move_into(current_sequence)
    if lp_direction is not None:
        current_sequence = lp_direction

    # Use local optimization to further refine
    n = len(current_sequence)
    if n == 0:
        return current_sequence

    # Objective function to minimize
    def objective(x):
        _, inv_C1 = compute_autocorrelation_constant(x)
        return -inv_C1  # We want to maximize 1/C1, so minimize -1/C1

    bounds = [(0, 1000) for _ in range(n)]
    
    try:
        # Use differential evolution for global search
        result = optimize.differential_evolution(objective, bounds, maxiter=max_iter, seed=42)
        if result.success:
            return result.x.tolist()
    except Exception:
        pass

    # Fallback to random perturbation
    new_sequence = current_sequence.copy()
    idx = np.random.randint(0, n)
    new_sequence[idx] = max(0, new_sequence[idx] + np.random.normal(0, 0.05))
    
    return new_sequence

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence.
    """
    np.random.seed(int(time.time()) % 1000000)
    best_sequence = []
    best_inv_C1 = 0.0
    best_C1 = float('inf')

    # Try multiple initialization strategies
    for _ in range(10):
        # Different initialization strategies
        strategy = np.random.choice(['random', 'geometric', 'spike'])

        if strategy == 'random':
            n = np.random.randint(100, 1000)
            sequence = np.random.uniform(0.1, 1.0, n).tolist()
        elif strategy == 'geometric':
            n = np.random.randint(100, 1000)
            sequence = [0.9 ** i for i in range(n)]
        else:  # spike
            n = np.random.randint(100, 1000)
            sequence = [0.0] * n
            spike_idx = np.random.randint(0, n)
            sequence[spike_idx] = 1.0

        # Improve the sequence
        improved_sequence = adaptive_search_step(sequence, max_iter=30)

        # Check if it's better
        c1, inv_c1 = compute_autocorrelation_constant(improved_sequence)
        if inv_c1 > best_inv_C1:
            best_inv_C1 = inv_c1
            best_sequence = improved_sequence
            best_C1 = c1

    # Final optimization
    if best_sequence:
        final_sequence = adaptive_search_step(best_sequence, max_iter=100)
        c1, inv_c1 = compute_autocorrelation_constant(final_sequence)
        if inv_c1 > best_inv_C1:
            return final_sequence

    # Fallback to a simple sequence
    if not best_sequence:
        best_sequence = [1.0] * 100

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
