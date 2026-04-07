# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import math
import random

def compute_c1(sequence):
    """Computes the C1 autocorrelation constant for a sequence."""
    n = len(sequence)
    if n < 1:
        return float('inf')
    
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')
        
    # Use FFT for large sequences, direct for small ones
    if n > 100:
        conv = fftconvolve(sequence, sequence, mode='full')
    else:
        conv = np.convolve(sequence, sequence)
    
    max_conv = np.max(conv)
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def compute_inv_c1(sequence):
    """Computes the inverse of C1 (objective to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 <= 0:
        return 0.0
    return 1.0 / c1

def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence using adaptive gradient."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    
    if sum_sequence < 1e-10:
        return None
    
    # Normalize sequence
    normalized_sequence = [x / sum_sequence for x in sequence]
    
    # Compute convolution
    if n > 100:
        conv_result = fftconvolve(normalized_sequence, normalized_sequence, mode='full')
        conv_result = conv_result[:2*n - 1]
    else:
        conv_result = np.convolve(normalized_sequence, normalized_sequence)
    
    rhs = np.max(conv_result)
    
    # Attempt to solve constrained optimization problem
    try:
        g_fun = solve_convolution_lp(normalized_sequence, rhs, n)
    except Exception:
        g_fun = None
        
    if g_fun is None:
        # Fallback: simple gradient ascent
        t = 0.01
        new_sequence = [(1 - t) * x + t * max(x, 1e-6) for x in sequence]
        return new_sequence

    # Normalize the solution
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None
        
    normalized_g_fun = [x / sum_g_fun for x in g_fun]
    
    # Adaptive step-size based on sequence size
    t = min(0.05, 0.01 + 0.005 * math.log(n + 1))
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs, n):
    """Solves the convolution LP for a given sequence and RHS."""
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    
    # Generate convolution constraints
    if n > 100:
        f_conv = fftconvolve(f_sequence, f_sequence, mode='full')
        f_conv = f_conv[:2*n-1]
    else:
        f_conv = np.convolve(f_sequence, f_sequence)
    
    # Build constraint matrix
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

    # Solve using high quality solver
    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
    except:
        # Fallback to simplex if highs fails
        try:
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex')
        except:
            raise Exception("Linear programming solver failed")

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        raise Exception("Linear programming did not converge")

def search_for_best_sequence() -> list[float]:
    """Main function to find the best coefficient sequence."""
    # Use log-scaled length generation for adaptiveness
    n = max(10, int(math.log(1000) * 50))
    best_sequence = np.random.rand(n).tolist()
    
    # Ensure initial positivity
    for i in range(len(best_sequence)):
        if best_sequence[i] < 0.01:
            best_sequence[i] = 0.01
    
    # Apply iterative improvement
    for iteration in range(20):
        direction = get_good_direction_to_move_into(best_sequence)
        if direction is not None:
            best_sequence = direction
        else:
            # Recovery step: add some randomness and restart
            index = random.randint(0, len(best_sequence) - 1)
            best_sequence[index] = max(0.01, best_sequence[index] * (1 + random.gauss(0, 0.1)))
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
