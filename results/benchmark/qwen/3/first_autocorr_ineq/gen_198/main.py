# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time
import random

def compute_convolution_fft(seq):
    """Compute convolution using FFT for better performance."""
    n = len(seq)
    if n == 0:
        return np.array([])
    
    # Pad to next power of 2 for efficient FFT
    padded_len = 1 << (n - 1).bit_length()
    padded_seq = np.pad(seq, (0, padded_len - n), 'constant')

    # FFT-based convolution
    fft_seq = fft(padded_seq)
    conv_fft = fft_seq * fft_seq.conj()
    conv_result = ifft(conv_fft).real[:2*n-1]

    return conv_result

def compute_c1_constant(sequence):
    """Compute C1 constant for the given sequence."""
    if len(sequence) == 0:
        return float('inf')
    
    # Compute convolution
    conv = compute_convolution_fft(sequence)
    max_conv = np.max(conv)
    sum_sq = np.sum(sequence)**2

    if sum_sq < 1e-12:  # Avoid division by zero
        return float('inf')

    c1 = 2 * len(sequence) * max_conv / sum_sq
    return c1

def compute_gradient_c1(sequence):
    """Approximate gradient of C1 constant with respect to sequence elements."""
    n = len(sequence)
    if n == 0:
        return np.zeros(n)
    
    # Compute convolution
    conv = compute_convolution_fft(sequence)
    max_conv_idx = np.argmax(conv)
    sum_seq = np.sum(sequence)
    sum_sq = sum_seq**2

    if sum_sq < 1e-12:
        return np.zeros(n)

    # Compute derivative of C1 w.r.t. each element
    grad = np.zeros(n)
    
    # Gradient calculation based on chain rule and product rule
    # dC1/dx_i = 2*n * (d(max_conv)/dx_i * sum_sq - max_conv * d(sum_sq)/dx_i) / sum_sq^2
    for i in range(n):
        # d(sum_sq)/dx_i = 2 * sum_seq
        d_sum_sq_dx = 2 * sum_seq
        
        # For convolution term, we consider contribution of x_i to max_conv
        # This is a simplified first-order approximation
        d_max_conv_dx = 0.0
        for j in range(n):
            k = i + j  # convolution index
            if 0 <= k < 2*n - 1:
                # Contribution to convolution at index k
                if k == max_conv_idx:
                    d_max_conv_dx += sequence[j]  # Simple approximation
        
        grad[i] = 2 * n * (d_max_conv_dx * sum_sq - max_conv * d_sum_sq_dx) / (sum_sq**2)
    
    return grad

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence with improved optimization."""
    if len(sequence) == 0:
        return None

    n = len(sequence)
    sum_sequence = np.sum(sequence)

    # Skip if sum is too small
    if sum_sequence < 1e-8:
        return None

    # Normalize sequence
    normalized_sequence = np.array(sequence) / sum_sequence

    try:
        # Compute maximum convolution value
        conv = compute_convolution_fft(normalized_sequence)
        max_conv = np.max(conv)

        # Solve LP with better constraint handling
        g_fun = solve_convolution_lp(normalized_sequence, max_conv)

        if g_fun is None or np.any(np.isnan(g_fun)):
            return None

        # Normalize the resulting sequence
        sum_g = np.sum(g_fun)
        if sum_g < 1e-8:
            return None

        normalized_g_fun = g_fun / sum_g

        # Apply gradient-based update to enhance the sequence
        grad = compute_gradient_c1(normalized_g_fun)
        step_size = 0.01 * np.linalg.norm(grad) if np.linalg.norm(grad) > 0 else 0.01
        
        # Gradient-based adjustment
        adjusted_sequence = normalized_g_fun - step_size * grad
        adjusted_sequence = np.maximum(adjusted_sequence, 0)  # Ensure non-negativity

        # Combine with previous direction
        t = 0.01
        new_sequence = [
            (1 - t) * x + t * y for x, y in zip(normalized_sequence, adjusted_sequence)
        ]

        # Ensure non-negativity
        new_sequence = [max(0, x) for x in new_sequence]

        return new_sequence

    except Exception as e:
        # Return None on any error
        return None

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP with improved numerical stability."""
    n = len(f_sequence)
    if n == 0:
        return None

    try:
        # Create constraint matrix efficiently
        # For each convolution index k, create constraint: sum_{i+j=k} f[i] * x[j] <= rhs
        # We'll build a sparse constraint matrix

        # Using a more stable approach with explicit constraint generation
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
        c = np.ones(n)  # Minimize sum(x) to maximize 1/C1

        # Add small regularization term to prevent numerical issues
        c = c + 1e-10

        # Solve with bounds
        bounds = [(0, None) for _ in range(n)]

        result = optimize.linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'disp': False}
        )

        if result.success:
            return result.x
        else:
            return None

    except Exception as e:
        return None

def initialize_sequence():
    """Generate an initial sequence with good structure for fast convergence."""
    # Use a mix of random and structured sequences
    n = random.randint(100, 1000)
    
    # Choose initialization type
    choice = random.random()
    
    if choice < 0.3:
        # Heavy-tail distribution like power law
        sequence = [random.expovariate(0.1) for _ in range(n)]
        # Normalize to prevent extreme values
        max_val = max(sequence)
        sequence = [x * 100.0 / max_val if max_val > 0 else 1.0 for x in sequence]
    elif choice < 0.6:
        # Uniform with peaks
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
        # Add a few peaks
        for i in range(min(10, n//20)):
            peak_pos = random.randint(0, n-1)
            sequence[peak_pos] = random.uniform(100.0, 1000.0)
    else:
        # Mixed distribution
        sequence = []
        for i in range(n):
            if random.random() < 0.7:
                sequence.append(random.uniform(0.1, 10.0))
            else:
                sequence.append(random.uniform(50.0, 100.0))
                
    return sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence with improved algorithm."""
    np.random.seed(int(time.time()) % 1000000)
    
    # Initialize with good starting point
    best_sequence = initialize_sequence()
    best_c1 = compute_c1_constant(best_sequence)
    
    # Iteratively improve the sequence
    for iteration in range(100):
        improved_sequence = get_good_direction_to_move_into(best_sequence)
        
        if improved_sequence is None:
            break
            
        new_c1 = compute_c1_constant(improved_sequence)
        
        # Accept improvement only if it leads to better C1 (smaller C1 means better upper bound)
        if new_c1 < best_c1:
            best_sequence = improved_sequence
            best_c1 = new_c1
        else:
            # Sometimes accept worse solutions to escape local minima
            if random.random() < 0.05:
                best_sequence = improved_sequence
                best_c1 = new_c1
                
        # Early stopping if improvement is minimal
        if abs(new_c1 - best_c1) < 1e-8:
            break
    
    # Final validation
    final_c1 = compute_c1_constant(best_sequence)
    if final_c1 < 100:  # Filter out bad candidates
        return best_sequence
    else:
        # Fall back to a reasonable default
        return [1.0] * 100

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
