# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import math
import random

def compute_c1_constant(sequence):
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use FFT for efficiency when sequence is large
    if n > 100:
        b = fftconvolve(a, a, mode='full')[:2*n-1]
    else:
        b = np.convolve(a, a, mode='full')[:2*n-1]

    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence with adaptive parameters."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)

    # Prevent division by zero
    if sum_sequence < 1e-10:
        return None

    # Normalize with adaptive factor
    adaptive_factor = np.sqrt(2 * n)
    normalized_sequence = [x * adaptive_factor / sum_sequence for x in sequence]

    # Use FFT for large sequences, direct convolution for small ones
    if n > 100:
        conv_result = fftconvolve(normalized_sequence, normalized_sequence, mode='full')[:2*n-1]
    else:
        conv_result = np.convolve(normalized_sequence, normalized_sequence, mode='full')[:2*n-1]

    rhs = np.max(conv_result)

    # Try solving LP with adaptive constraints
    g_fun = solve_convolution_lp(normalized_sequence, rhs, n)

    if g_fun is None:
        # Fallback: try with modified RHS
        rhs_fallback = rhs * 1.1
        g_fun = solve_convolution_lp(normalized_sequence, rhs_fallback, n)

    if g_fun is None:
        # Final fallback: simple gradient ascent
        t = 0.01
        new_sequence = [(1 - t) * x + t * max(x, 1e-6) for x in sequence]
        return new_sequence

    # Normalize g_fun
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    normalized_g_fun = [x * adaptive_factor / sum_g_fun for x in g_fun]

    # Adaptive step-size with curvature awareness
    t = min(0.05, 0.01 + 0.005 * np.log(n + 1))

    # Curvature-aware adjustment
    if n > 50:
        # Estimate the curvature effect for better step size selection
        current_c1, _ = compute_c1_constant(sequence)
        curvature_adjustment = max(0.0, 1.0 - min(1.0, current_c1 / 1.5))
        t *= (0.5 + 0.5 * curvature_adjustment)  # Adjust based on curvature

    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs, n):
    """Solves the convolution LP for a given sequence and RHS with advanced constraint handling."""
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Create constraint matrix
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

    # Adaptive tolerance for optimization and method selection
    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'time_limit': 10})
    except:
        # Fallback to 'simplex' method if 'highs' fails or times out
        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex')
        except:
            return None

    if result.success:
        g_sequence = result.x

        # Ensure non-negativity and reasonable values
        g_sequence = np.maximum(g_sequence, 0)
        if np.sum(g_sequence) < 1e-10:
            return None

        return g_sequence
    else:
        return None

def adaptive_sequence_length(sequence, target_ratio=0.8):
    """Dynamically adjust sequence length based on convergence properties."""
    n = len(sequence)

    if n > 1000:
        # Keep top 80% of the sequence values
        top_indices = np.argsort(sequence)[-int(n*target_ratio):]
        new_sequence = [sequence[i] for i in sorted(top_indices)]
        return new_sequence

    elif n < 50:
        # Expand with copies and slight mutations
        expanded = sequence.copy()
        for i in range(10):
            idx = random.randint(0, n-1)
            expanded.append(expanded[idx] * (1 + random.uniform(-0.2, 0.2)))
        return expanded

    return sequence

def generate_initial_sequence():
    """Generate a better initialized sequence with more structure."""
    n = max(50, int(random.uniform(100, 500)))
    # Mix of uniform and Gaussian distributions
    if random.random() < 0.5:
        return [random.uniform(0.1, 100.0) for _ in range(n)]
    else:
        # Use exponential distribution for skewness
        sequence = [random.expovariate(0.1) for _ in range(n)]
        # Normalize to prevent extreme values
        max_val = max(sequence)
        return [x * 100.0 / max_val if max_val > 0 else 1.0 for x in sequence]

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence with enhanced evolutionary approach."""
    # Initialize with a better sequence
    best_sequence = generate_initial_sequence()

    # Evolve for several iterations with adaptive strategy
    for iteration in range(50):  # Increased iterations for better exploration
        h_function = get_good_direction_to_move_into(best_sequence)
        if h_function is not None:
            best_sequence = h_function
        else:
            # If evolution fails, try to recover with a simple mutation
            index = np.random.randint(len(best_sequence))
            best_sequence[index] = max(0.01, best_sequence[index] * (1 + np.random.normal(0, 0.1)))

        # Occasionally adjust sequence length
        if iteration % 10 == 0:
            best_sequence = adaptive_sequence_length(best_sequence)

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")