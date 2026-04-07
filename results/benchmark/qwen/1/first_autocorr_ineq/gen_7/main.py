# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time

def convolve_fft(a, b):
    """Compute convolution using FFT for better performance."""
    n = len(a)
    # Pad to length 2*n - 1 for full convolution
    padded_length = 2 * n - 1
    fa = fft(a, padded_length)
    fb = fft(b, padded_length)
    result = ifft(fa * fb.conj()).real
    # Return only the valid convolution part
    return result[:n]

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)

    # Avoid division by zero
    if sum_sequence < 1e-10:
        return None

    # Normalize sequence properly
    normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence

    # Use FFT-based convolution for efficiency
    conv_result = convolve_fft(normalized_sequence, normalized_sequence)
    rhs = np.max(conv_result)

    # Solve the linear program
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None

    # Normalize the solution
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g_fun

    # Apply small perturbation to move towards better solution
    t = 0.01
    new_sequence = (1 - t) * np.array(sequence) + t * normalized_g_fun

    return new_sequence.tolist()

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Build convolution constraints using FFT approach for better efficiency
    # We'll create constraints that ensure the convolution doesn't exceed rhs
    # This involves creating the convolution matrix efficiently

    # Generate convolution constraints manually for better control
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

    # Use a more robust solver
    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
    except Exception:
        return None

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        return None

def local_search_refinement(sequence, max_iterations=10):
    """Apply local search to refine the sequence."""
    current_seq = np.array(sequence).copy()
    n = len(current_seq)

    for _ in range(max_iterations):
        # Try small modifications to improve the solution
        modified = False
        for i in range(n):
            # Try slightly increasing/decreasing element
            test_seq = current_seq.copy()
            delta = 0.001
            test_seq[i] = min(1000, max(0, test_seq[i] + delta))

            # Check if improvement
            if is_better_solution(test_seq, current_seq):
                current_seq = test_seq
                modified = True

        if not modified:
            break

    return current_seq.tolist()

def is_better_solution(new_seq, old_seq):
    """Check if new sequence gives better C1 value."""
    if np.sum(new_seq) < 0.01:
        return False

    # Compute C1 for both sequences
    conv_new = np.convolve(new_seq, new_seq)
    max_conv_new = np.max(conv_new)
    sum_new = np.sum(new_seq)

    conv_old = np.convolve(old_seq, old_seq)
    max_conv_old = np.max(conv_old)
    sum_old = np.sum(old_seq)

    # Smaller max_conv_sum_ratio means better solution
    ratio_new = max_conv_new / (sum_new * sum_new)
    ratio_old = max_conv_old / (sum_old * sum_old)

    return ratio_new < ratio_old

def compute_inv_c1(sequence):
    """Compute 1/C1 for a given sequence."""
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return 0.0

    conv = np.convolve(sequence, sequence)
    max_conv = np.max(conv)
    sum_seq = np.sum(sequence)

    if max_conv <= 0 or sum_seq <= 0:
        return 0.0

    c1 = 2 * len(sequence) * max_conv / (sum_seq * sum_seq)
    return 1.0 / c1 if c1 > 0 else 0.0

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    # Start with a good initialization
    n = np.random.randint(100, 1000)
    best_sequence = [np.random.random() * 100 for _ in range(n)]

    # Apply local search to start
    best_sequence = local_search_refinement(best_sequence, 5)

    # Iterative improvement
    best_score = compute_inv_c1(best_sequence)
    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        h_function = get_good_direction_to_move_into(best_sequence)
        if h_function is None:
            # If optimization fails, try a simple perturbation
            idx = np.random.randint(len(best_sequence))
            best_sequence[idx] = min(1000, max(0, best_sequence[idx] + np.random.rand() * 10))
        else:
            # Accept the improved sequence
            best_sequence = h_function

        # Local refinement
        best_sequence = local_search_refinement(best_sequence, 10)

        # Update score
        current_score = compute_inv_c1(best_sequence)
        if current_score > best_score:
            best_score = current_score
            iteration = 0  # Reset iteration counter if improvement
        else:
            iteration += 1

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")