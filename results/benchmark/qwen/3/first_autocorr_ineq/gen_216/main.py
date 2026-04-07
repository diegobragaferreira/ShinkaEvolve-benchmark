# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import random
import time
from cvxpy import *
import cvxpy as cp

# Suppress scientific notation for cleaner output
np.set_printoptions(suppress=True)

# Historical sequences for seeding the algorithm
HISTORICAL_SEQUENCES = [
    [1.0] * 100,
    [1.0] * 50 + [0.0] * 50,
    [1.0, 0.0] * 50,
    [0.0] * 25 + [1.0] * 50 + [0.0] * 25,
    [1.0, 1.0, 0.0, 0.0] * 25,
    [1.0, 0.5, 0.25, 0.125] * 25,
    [1.0, 0.8, 0.6, 0.4, 0.2] * 20,
    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] * 10,
]

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')

def assess_numerical_stability(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Assess whether FFT convolution would be numerically stable for given inputs.
    Returns True if using FFT is likely to be stable, False otherwise.
    """
    n = len(a)

    # For very small sequences, prefer direct convolution
    if n < 30:
        return False

    # For larger sequences, perform detailed stability analysis
    try:
        # Direct convolution for reference
        direct_conv = convolve_direct(a, b)

        # FFT convolution
        fft_conv = convolve_fft(a, b)

        # Compute relative error between methods
        direct_max = np.max(np.abs(direct_conv))
        fft_max = np.max(np.abs(fft_conv))

        # If either is zero, consider stable
        if direct_max == 0 and fft_max == 0:
            return True

        # Relative error metric
        if direct_max > 0:
            rel_error = np.mean(np.abs(direct_conv - fft_conv)) / direct_max
        else:
            rel_error = np.mean(np.abs(direct_conv - fft_conv)) / (fft_max + 1e-15)

        # Also consider the condition number of the discrete convolution matrix
        # For simplicity, we use a proxy based on the ratio of max to min values
        conv_ratio = np.max(np.abs(direct_conv)) / (np.min(np.abs(direct_conv)) + 1e-15)

        # If error is small and ratio is reasonable, FFT is acceptable
        return rel_error < 0.05 and conv_ratio < 10000

    except Exception:
        # Default to direct method if anything goes wrong
        return False

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Determine best convolution method based on stability assessment
    use_fft = n > 100 and assess_numerical_stability(a, a)

    if use_fft:
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

def solve_convex_optimization(sequence_length: int) -> List[float]:
    """Solve the convex optimization problem to find a sequence that maximizes 1/C1."""
    # Create variables
    a = cp.Variable(sequence_length, nonneg=True)
    
    # Create the objective: minimize max convolution value
    # We'll use a quadratic formulation to model the constraint that max convolution is bounded
    # The constraint is: a * a <= max_val (in a suitable sense)  
    # This is a simplified approach; we'll focus on finding a sequence that improves performance
    
    # Normalize the sequence to have sum = 1 for easier handling
    sum_constraint = cp.sum(a) >= 0.01  # Ensure sufficient mass
    
    # Formulate a quadratic problem to maximize 1/C1
    # This is a conceptual approach where we aim to minimize the effect of large values
    # in the convolution by constraining the overall structure
    
    # Objective: minimize max convolution value, which is equivalent to maximizing 1/C1
    # We'll use a heuristic by minimizing a measure of concentration in the convolution
    
    # Since we're dealing with a complex constraint, we'll use a simpler approach
    # to construct a sequence that performs well empirically
    # We'll construct a 'good' sequence based on known patterns
    # and then slightly refine it using convex optimization
    
    # Heuristic construction of a candidate sequence
    candidate_sequence = []
    for i in range(sequence_length):
        # Use a pattern based on decreasing weights for better performance
        weight = 1.0 / (1.0 + i * 0.05)
        candidate_sequence.append(max(0, weight))
    
    # Normalize to ensure sum is reasonable
    total = sum(candidate_sequence)
    if total > 0.01:
        candidate_sequence = [x / total * 100 for x in candidate_sequence]
    
    # Initial guess from historical sequences if available
    if sequence_length <= 1000:
        for hist_seq in HISTORICAL_SEQUENCES:
            if len(hist_seq) == sequence_length:
                candidate_sequence = hist_seq.copy()
                break
    
    # Refine using CVXPY to optimize certain aspects of the sequence
    # Here we'll use the fact that we want to minimize the maximum convolution value
    
    # Set initial value
    a.value = np.array(candidate_sequence)
    
    # Simple convex constraint: sum of elements is bounded
    constraints = [cp.sum(a) <= 1000, cp.sum(a) >= 0.01]
    
    # Objective: minimize a weighted version of the convolution
    # This is a proxy for minimizing the maximum convolution value
    # We compute a rough estimate and penalize high values
    
    # Define a simple quadratic optimization problem to improve the sequence
    # We're essentially solving an unconstrained optimization problem with constraints
    
    # Simplified approach: just use the known good sequence as a starting point
    # and slightly perturb it to increase 1/C1
    
    # This will be done by iteratively improving with a simple convex subproblem
    # We'll make a few adjustments to the candidate sequence using convex optimization
    
    # Create a simple quadratic cost that promotes spreading out the values
    # to reduce the maximum convolution
    objective = cp.Minimize(cp.sum_squares(a) + 0.001 * cp.sum(a))
    
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return a.value.tolist()
    except:
        # Fallback to the heuristic sequence
        pass
    
    # If convex optimization fails, return the heuristic sequence
    return candidate_sequence

def adaptive_search_for_best_sequence(
    max_time_seconds: int = 180,
    initial_length_range: tuple = (100, 500),
    attempts_per_length: int = 5
) -> list:
    """Main function to search for the best coefficient sequence using convex optimization."""
    start_time = time.time()
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Try several sequence lengths to find the optimal one
    trial_lengths = [int(l) for l in np.linspace(100, 1000, 10)]
    
    for length in trial_lengths:
        # Limit attempts per length to prevent excessive computation
        if time.time() - start_time > max_time_seconds:
            break
            
        for attempt in range(attempts_per_length):
            if time.time() - start_time > max_time_seconds:
                break
                
            # Get a candidate sequence using convex optimization
            candidate_sequence = solve_convex_optimization(length)
            
            # Compute its C1 constant
            _, inv_c1 = compute_c1_constant(candidate_sequence)
            
            # Update best if this is better
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = candidate_sequence[:]
                
    return best_sequence if best_sequence is not None else [1.0]

def search_for_best_sequence() -> list:
    """Function to search for the best coefficient sequence."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    return adaptive_search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")