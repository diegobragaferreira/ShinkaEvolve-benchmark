# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from typing import List, Optional, Tuple
import random
import time

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

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

def evaluate_fitness(sequence: List[float]) -> Tuple[float, float]:
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

def solve_convolution_lp(f_sequence: np.ndarray, rhs: float) -> Optional[np.ndarray]:
    """
    Solves the convolution linear program for a given sequence and RHS.
    """
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Construct constraint matrix for convolution
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

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
        if result.success:
            return result.x
        else:
            return None
    except Exception:
        return None

def get_good_direction_to_move_into(sequence: List[float]) -> Optional[List[float]]:
    """
    Determines a better direction to move towards based on current sequence.
    Returns a new sequence if successful, None otherwise.
    """
    n = len(sequence)
    if n == 0:
        return None

    a = np.array(sequence)
    sum_a = np.sum(a)

    # Avoid division by zero
    if sum_a < 1e-10:
        return None

    # Normalize
    normalized_a = a * np.sqrt(2 * n) / sum_a

    try:
        # Compute convolution using FFT
        conv_result = compute_convolution_fft(normalized_a)
        rhs = np.max(conv_result)
    except Exception:
        return None

    # Solve LP
    g_fun = solve_convolution_lp(normalized_a, rhs)
    if g_fun is None:
        return None

    # Check normalization
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    # Re-normalize
    normalized_g_fun = g_fun * np.sqrt(2 * n) / sum_g_fun

    # Blend with original
    t = 0.01
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    return new_sequence

def generate_structured_sequence(length: int) -> List[float]:
    """
    Generates a structured sequence to provide better initial points.
    """
    sequence = []
    for i in range(length):
        exp_component = 100 * np.exp(-i * 0.01)
        period_component = 10 * np.sin(i * 0.2) * np.cos(i * 0.05)
        val = max(0.01, exp_component + period_component)
        sequence.append(val)
    return sequence

def search_for_best_sequence() -> List[float]:
    """
    Main function to find the best sequence.
    """
    # Initialize with a structured sequence
    n = np.random.randint(100, 1000)
    best_sequence = generate_structured_sequence(n)

    # Try to improve the sequence
    improved_sequence = get_good_direction_to_move_into(best_sequence)
    if improved_sequence is not None:
        best_sequence = improved_sequence

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")