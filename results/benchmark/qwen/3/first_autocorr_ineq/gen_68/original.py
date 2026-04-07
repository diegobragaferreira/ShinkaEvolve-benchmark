# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
import random
from typing import List, Optional
import time

def get_good_direction_to_move_into(
    sequence: List[float],
) -> Optional[List[float]]:
    """Returns the direction to move into the sequence using adaptive strategy."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    
    if sum_sequence < 1e-10:
        return None
        
    # Normalize sequence
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    
    # Compute convolution using fast method
    conv_result = np.convolve(normalized_sequence, normalized_sequence, mode='full')
    rhs = np.max(conv_result)
    
    # Apply adaptive LP solving with reduced constraints
    g_fun = solve_convolution_lp_adaptive(normalized_sequence, rhs, n)
    
    if g_fun is None:
        return None
        
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None
        
    # Normalize the solution
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]
    
    # Use variable learning rate based on convergence
    t = 0.01 * (1.0 + np.random.rand() * 0.5)
    
    # Apply diversity-guided mutation
    new_sequence = [
        (1 - t) * x + t * y + 0.001 * random.gauss(0, 1) for x, y in zip(sequence, normalized_g_fun)
    ]
    
    # Clip values to ensure feasibility
    new_sequence = [max(0, min(1000, x)) for x in new_sequence]
    
    return new_sequence

def solve_convolution_lp_adaptive(f_sequence, rhs, n):
    """Adaptive LP solver that samples constraints intelligently."""
    # Sample a fraction of constraints for large sequences to reduce computation
    if n > 200:
        num_constraints = min(2*n - 1, 500)
        indices = sorted(random.sample(range(2*n - 1), num_constraints))
    else:
        indices = list(range(2*n - 1))
        
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    
    # Build constraints efficiently
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
        # Try with different methods for better robustness
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs', options={'maxiter': 500})
        if not result.success:
            # Fallback to basic method
            result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='simplex', options={'maxiter': 500})
    except Exception:
        return None
        
    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        return None

def search_for_best_sequence() -> List[float]:
    """Search for the best coefficient sequence with adaptive initialization."""
    # Initialize with varied lengths and distributions
    n = random.randint(50, 500)
    best_sequence = [random.uniform(0.1, 1.0) for _ in range(n)]
    
    # Multiple refinement iterations
    for _ in range(30):
        h_function = get_good_direction_to_move_into(best_sequence)
        if h_function is not None:
            best_sequence = h_function
        else:
            # Random perturbation if optimization fails
            idx = random.randint(0, len(best_sequence)-1)
            best_sequence[idx] = max(0, best_sequence[idx] + random.uniform(-0.5, 0.5))
    
    # Final cleanup and normalization
    total = sum(best_sequence)
    if total > 0.01:
        best_sequence = [x / total * 100 for x in best_sequence]
    else:
        best_sequence = [1.0 for _ in best_sequence]  # fallback
        
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
