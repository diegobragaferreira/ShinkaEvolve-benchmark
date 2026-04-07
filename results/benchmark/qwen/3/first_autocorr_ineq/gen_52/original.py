# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)

    # Avoid division by very small sums
    if sum_sequence < 1e-10:
        return None

    # Normalize sequence
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]

    # Compute convolution using FFT for efficiency
    conv_result = np.convolve(normalized_sequence, normalized_sequence, mode='full')
    rhs = np.max(conv_result)

    # Solve optimized LP problem
    g_fun = solve_convolution_lp(normalized_sequence, rhs)

    if g_fun is None:
        return None

    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    # Normalize the solution
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

    # Perform adaptive gradient update with damping
    t = 0.01 * (1.0 + np.random.rand() * 0.5)  # Adaptive step size
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]

    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)

    # Use FFT-based convolution for large sequences
    if n > 100:
        # For large sequences, use a sparse constraint approach
        # Generate a subset of convolution constraints
        num_constraints = min(2*n - 1, 1000)
        indices = sorted(random.sample(range(2*n - 1), num_constraints))
    else:
        indices = list(range(2*n - 1))

    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Generate convolution constraints efficiently
    for k in indices:
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

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
    except:
        return None

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        return None

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Initialize with a diverse set of random values
    n = random.randint(100, 1000)
    best_sequence = [random.uniform(0.1, 1.0) for _ in range(n)]

    # Apply multiple iterations of optimization
    for _ in range(20):  # More iterations for better convergence
        h_function = get_good_direction_to_move_into(best_sequence)
        if h_function is not None:
            best_sequence = h_function
        else:
            # Fallback: slightly modify the sequence
            idx = random.randint(0, len(best_sequence)-1)
            best_sequence[idx] = (best_sequence[idx] + random.uniform(-0.1, 0.1)) % 1.0

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")