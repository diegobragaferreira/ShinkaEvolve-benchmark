# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import random
import time

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence using a hybrid evolutionary-gradient approach."""
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

    # Solve optimized LP problem with variable constraint sampling
    g_fun = solve_convolution_lp_adaptive(normalized_sequence, rhs, n)

    if g_fun is None:
        return None

    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None

    # Normalize the solution
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]

    # Perform adaptive gradient update with damping
    t = 0.01 * (1.0 + np.random.rand() * 0.5)
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]

    # Ensure all values remain within bounds
    new_sequence = [max(0, min(1000, x)) for x in new_sequence]

    return new_sequence

def solve_convolution_lp_adaptive(f_sequence, rhs, n):
    """Solves the convolution LP with adaptive constraint sampling and relaxation."""
    # Determine a subset of convolution constraints to use, tailored to the sequence size
    if n > 200:
        # Select constraints around the peak of the convolution (more informative)
        peak_position = int(len(f_sequence) / 2)
        num_constraints = min(2*n - 1, 1000)
        indices = []
        # Take indices near the center to reflect dominant convolution behavior
        for i in range(max(0, peak_position - num_constraints//4), 
                       min(2*n - 1, peak_position + num_constraints//4)):
            indices.append(i)
        # Fill up remaining indices with random samples if not enough
        while len(indices) < num_constraints:
            indices.append(random.randint(0, 2*n - 2))
        indices = sorted(set(indices))[:num_constraints]
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
        # Solve with relaxation factor to allow exploration
        relaxation_factor = 1.0
        modified_b_ub = [b * relaxation_factor for b in b_ub]
        result = optimize.linprog(c, A_ub=a_ub, b_ub=modified_b_ub, method='highs')
    except:
        return None

    if result.success:
        g_sequence = result.x
        # Ensure non-negativity due to numerical errors
        g_sequence = np.maximum(g_sequence, 0)
        return g_sequence
    else:
        return None

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence using evolutionary enhancement."""
    # Initialize with a diverse set of random values
    n = random.randint(100, 1000)
    best_sequence = [random.uniform(0.1, 1.0) for _ in range(n)]
    
    # Store best sequences found to maintain diversity
    best_sequences = [best_sequence[:]]
    
    # Apply multiple iterations of optimization
    for i in range(30):
        # Occasionally reproduce from best sequences to maintain diversity
        if i % 5 == 0 and len(best_sequences) > 1:
            rep_idx = random.randint(0, len(best_sequences) - 1)
            best_sequence = best_sequences[rep_idx][:]

        h_function = get_good_direction_to_move_into(best_sequence)
        if h_function is not None:
            best_sequence = h_function
            # Store this as a potential best solution
            if len(best_sequences) < 5:
                best_sequences.append(best_sequence[:])
            else:
                # Replace the oldest entry to keep population compact
                best_sequences.pop(0)
                best_sequences.append(best_sequence[:])
        else:
            # Fallback: slightly modify the sequence
            idx = random.randint(0, len(best_sequence)-1)
            best_sequence[idx] = max(0, best_sequence[idx] + random.uniform(-0.1, 0.1))
    
    # Return the best sequence found
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
