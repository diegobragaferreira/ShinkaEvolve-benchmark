# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1(sequence):
    """Compute the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    # Use FFT-based convolution for efficiency
    conv = fftconvolve(sequence, sequence, mode='full')
    # Take only the relevant part of convolution
    max_conv = np.max(conv[len(sequence)-1:])  # From index n-1 onwards

    # Normalize and compute C1
    sum_sq = np.sum(sequence)**2
    if sum_sq == 0:
        return float('inf')

    c1 = (2 * len(sequence) * max_conv) / sum_sq
    return c1

def compute_inv_c1(sequence):
    """Compute inverse of C1 (what we want to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == 0 or np.isinf(c1):
        return 0
    return 1.0 / c1

def get_good_direction_to_move_into(
    sequence: list[float],
) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    try:
        n = len(sequence)
        if n < 1:
            return None

        sum_sequence = np.sum(sequence)
        if sum_sequence < 1e-10:
            return None

        # Normalize sequence with sqrt(2*n) scaling factor
        normalized_sequence = np.array(sequence) * np.sqrt(2 * n) / sum_sequence

        # Compute convolution using FFT for efficiency
        conv_result = fftconvolve(normalized_sequence, normalized_sequence, mode='full')
        rhs = np.max(conv_result)

        # Solve the LP problem
        g_fun = solve_convolution_lp(normalized_sequence, rhs)

        if g_fun is None:
            return None

        sum_g = np.sum(g_fun)
        if sum_g < 1e-10:
            return None

        normalized_g_fun = np.array(g_fun) * np.sqrt(2 * n) / sum_g

        return normalized_g_fun.tolist()
    except Exception:
        return None

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    try:
        n = len(f_sequence)
        if n < 1:
            return None

        # Sample key convolution indices instead of generating all 2*n-1 constraints
        # This dramatically reduces memory usage and computation time
        max_constraints = min(5000, 2 * n)  # Cap at reasonable number
        constraint_indices = np.linspace(0, 2 * n - 2, min(max_constraints, 2 * n - 1), dtype=int)

        # Sort indices to help with memory access patterns
        constraint_indices = np.sort(constraint_indices)

        # Precompute convolution constraints efficiently
        a_ub = np.zeros((len(constraint_indices), n))
        b_ub = np.zeros(len(constraint_indices))

        for idx, k in enumerate(constraint_indices):
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    a_ub[idx, j] = f_sequence[i]
            b_ub[idx] = rhs

        # Add non-negativity constraints
        a_ub_nonneg = -np.eye(n)
        b_ub_nonneg = np.zeros(n)

        a_ub = np.vstack([a_ub, a_ub_nonneg])
        b_ub = np.hstack([b_ub, b_ub_nonneg])

        # Define objective function (minimize -sum x)
        c = -np.ones(n)

        # Solve with error handling
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            return result.x
        else:
            # Try alternative solver
            try:
                result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='interior-point')
                if result.success:
                    return result.x
            except:
                pass
            return None

    except Exception:
        return None

def create_structured_sequence(n):
    """Create a structured sequence for better starting points."""
    # Combines exponential decay, uniform, and random components
    exp_decay = np.exp(-np.linspace(0, 3, n))
    uniform = np.ones(n)
    random_part = np.random.rand(n)

    # Blend patterns with weights
    base_seq = 0.5 * exp_decay + 0.3 * uniform + 0.2 * random_part

    # Scale and normalize
    base_seq = base_seq / np.sum(base_seq) * 10

    # Clip to valid range
    base_seq = np.clip(base_seq, 0, 1000)

    return base_seq.tolist()

def refine_with_convolution_analysis(sequence, max_iterations=10):
    """Refine sequence by analyzing convolution peaks and adjusting accordingly."""
    seq = np.array(sequence)
    n = len(seq)

    for _ in range(max_iterations):
        # Compute convolution
        conv = fftconvolve(seq, seq, mode='full')
        conv_part = conv[n-1:]
        max_conv = np.max(conv_part)

        # Find indices where convolution peaks
        max_indices = np.where(conv_part >= 0.9 * max_conv)[0]

        # Adjust elements that contribute to peaks
        new_seq = seq.copy()
        for idx in max_indices[:min(3, len(max_indices))]:
            for offset in [-2, -1, 0, 1, 2]:
                pos = idx + offset
                if 0 <= pos < n:
                    new_seq[pos] *= 0.98  # Slight reduction

        seq = np.maximum(new_seq, 0)

        # Early stopping if sequence stabilizes
        if np.allclose(seq, new_seq, rtol=1e-6):
            break

    return seq.tolist()

def adaptive_gradient_update(sequence, iteration, max_iter):
    """Perform gradient update with adaptive step size."""
    # Adaptive learning rate with exponential decay
    t = 0.01 * np.exp(-iteration / max_iter * 5)

    # Get direction
    direction = get_good_direction_to_move_into(sequence)
    if direction is None:
        # Fallback to small random perturbation if direction fails
        return np.maximum(np.array(sequence) + np.random.normal(0, 0.001, len(sequence)), 0).tolist()

    # Mix with current sequence
    new_sequence = (1 - t) * np.array(sequence) + t * np.array(direction)
    return new_sequence.tolist()

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    best_inv_c1 = 0
    best_sequence = None

    # Time tracking for budget management
    start_time = time.time()
    max_time = 170  # Leave 10 seconds for cleanup

    # Multiple attempts to find good sequences
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break

        # Create diverse initial sequences
        n = random.randint(100, 1000)
        initial_seq = create_structured_sequence(n)

        # Apply adaptive optimization
        current_sequence = initial_seq.copy()
        current_inv_c1 = compute_inv_c1(current_sequence)

        # Optimization loop
        for iter_num in range(50):  # Reduced iterations to save time
            if time.time() - start_time > max_time:
                break

            # Adaptive gradient update
            updated_sequence = adaptive_gradient_update(current_sequence, iter_num, 50)

            # Ensure valid sequence
            updated_sequence = np.clip(updated_sequence, 0, 1000)
            if np.sum(updated_sequence) < 0.01:
                updated_sequence[0] = 0.1

            # Refine with convolution analysis
            refined_sequence = refine_with_convolution_analysis(updated_sequence)

            # Evaluate
            new_inv_c1 = compute_inv_c1(refined_sequence)

            # Update best if improved
            if new_inv_c1 > current_inv_c1:
                current_sequence = refined_sequence
                current_inv_c1 = new_inv_c1

                if current_inv_c1 > best_inv_c1:
                    best_inv_c1 = current_inv_c1
                    best_sequence = current_sequence[:]

        # Final refinement round
        if best_sequence is not None:
            final_refinement = refine_with_convolution_analysis(best_sequence)
            final_inv_c1 = compute_inv_c1(final_refinement)
            if final_inv_c1 > best_inv_c1:
                best_sequence = final_refinement

    # Fallback if nothing found
    if best_sequence is None:
        # Return a random good initial sequence
        n = random.randint(100, 500)
        best_sequence = create_structured_sequence(n)

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")