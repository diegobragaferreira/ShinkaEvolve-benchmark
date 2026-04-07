# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import math
import random
from typing import List, Optional

# Library of historically successful sequences
HISTORICAL_SEQUENCES = [
    [1.0] * 100,  # Uniform sequence
    [1.0, 0.5] * 50,  # Alternating sequence
    [0.1, 0.2, 0.3, 0.4, 0.5] * 20,  # Increasing sequence
    [0.5 ** i for i in range(100)],  # Geometric decay
    [1.0 if i % 2 == 0 else 0.1 for i in range(100)],  # Sparse sequence
]

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')[:2*len(a)-1]

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')[:2*len(a)-1]

def compute_c1_constant(sequence: List[float]) -> tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use FFT for efficiency when sequence is large
    if n > 100:
        b = convolve_fft(a, a)
    else:
        b = convolve_direct(a, a)

    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

def solve_convolution_lp(f_sequence: List[float], rhs: float, n: int) -> Optional[List[float]]:
    """Solves the convolution LP for a given sequence and RHS with robust fallbacks."""
    try:
        c = -np.ones(n)
        a_ub = []
        b_ub = []

        # Build constraint matrix for convolution constraints
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

        # Solve the linear program with multiple methods as fallback
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'maxiter': 1000})

        if not result.success:
            # Try different method if highs fails
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='revised simplex', options={'maxiter': 1000})

        if result.success:
            return result.x.tolist()
        else:
            return None

    except Exception:
        return None

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
        conv_result = fftconvolve(normalized_sequence, normalized_sequence, mode='full')
    else:
        conv_result = np.convolve(normalized_sequence, normalized_sequence)

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
    current_c1, _ = compute_c1_constant(sequence)
    t = min(0.1, max(0.01, 0.05 * (1.0 - min(1.0, current_c1 / 1.5))))

    # Add diversity with Gaussian noise
    noise = [np.random.normal(0, 0.01) for _ in range(n)]
    new_sequence = [
        (1 - t) * x + t * y + noise[i] for i, (x, y) in enumerate(zip(sequence, normalized_g_fun))
    ]
    return new_sequence

def initialize_sequence():
    """Initialize a sequence using historical sampling combined with randomness."""
    # 70% chance to use historical sequence, 30% chance for random
    if random.random() < 0.7 and HISTORICAL_SEQUENCES:
        # Sample from historical sequences with slight perturbation
        historical = random.choice(HISTORICAL_SEQUENCES)
        # Apply small random perturbation
        perturbed = [max(0.0, x + random.gauss(0, 0.1 * x)) for x in historical]
        return perturbed
    else:
        # Purely random initialization
        n = random.randint(50, 500)
        return [random.random() * 100 for _ in range(n)]

def adaptive_sequence_length(sequence: List[float]) -> List[float]:
    """Dynamically adjust sequence length based on observed convergence properties."""
    n = len(sequence)

    # If sequence is too long, consider truncation
    if n > 1000:
        # Keep top 50% of the sequence values
        top_indices = np.argsort(sequence)[-n//2:]
        new_sequence = [sequence[i] for i in sorted(top_indices)]
        return new_sequence

    # If sequence is too short, consider expansion
    if n < 50:
        # Expand with copies and slight mutations
        expanded = sequence.copy()
        for i in range(10):
            idx = random.randint(0, n-1)
            expanded.append(expanded[idx] * (1 + random.uniform(-0.2, 0.2)))
        return expanded

    return sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence with adaptive search."""
    # Initialize with a diverse sequence
    best_sequence = initialize_sequence()

    # Evolve for several iterations
    for iteration in range(50):
        h_function = get_good_direction_to_move_into(best_sequence)
        if h_function is not None:
            best_sequence = h_function
        else:
            # If evolution fails, try to recover with a simple mutation
            index = np.random.randint(len(best_sequence))
            best_sequence[index] = max(0.01, best_sequence[index] * (1 + np.random.normal(0, 0.1)))

        # Occasionally adapt sequence length
        if iteration % 10 == 0:
            best_sequence = adaptive_sequence_length(best_sequence)

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")