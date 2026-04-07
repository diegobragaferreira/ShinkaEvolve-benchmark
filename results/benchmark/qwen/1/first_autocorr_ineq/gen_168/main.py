# EVOLVE-BLOCK-START
import numpy as np
from scipy.fft import fft, ifft
import cvxpy as cp
import random
import time
from typing import List, Tuple
import math

def convolve_fft(a: List[float], b: List[float]) -> List[float]:
    """Compute convolution using FFT for better performance."""
    n = len(a)
    if n == 0:
        return []

    # Pad to length 2*n - 1 for full convolution
    padded_length = 2 * n - 1
    fa = fft(a, padded_length)
    fb = fft(b, padded_length)
    result = ifft(fa * fb.conj()).real
    # Return only the valid convolution part
    return result[:n].tolist()

def compute_c1(sequence: List[float]) -> float:
    """Compute the C1 constant for a given sequence."""
    n = len(sequence)
    if n == 0:
        return float('inf')

    sum_a = np.sum(sequence)
    if sum_a < 1e-10:
        return float('inf')

    # Compute convolution using FFT
    conv = convolve_fft(sequence, sequence)
    max_conv = np.max(conv)

    # Compute C1 = 2n * max(conv) / (sum(a))^2
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def evaluate_fitness(sequence: List[float]) -> float:
    """Evaluate fitness as inverse of C1 (higher is better)."""
    c1 = compute_c1(sequence)
    if c1 == float('inf') or c1 <= 0:
        return 0.0
    return 1.0 / c1

def generate_structured_sequence(n: int) -> List[float]:
    """Generate a structured sequence that's likely to perform well."""
    # Create a sequence with exponential decay to reduce autocorrelation
    sequence = []
    for i in range(n):
        # Exponential decay with some noise to break symmetry
        base_val = max(0.01, 100 * np.exp(-i * 0.05))
        noise = random.uniform(0.9, 1.1)
        sequence.append(base_val * noise)
    return sequence

def solve_convex_optimization(n: int, max_steps: int = 50) -> List[float]:
    """
    Use convex optimization to directly solve for the optimal step function.
    Minimizes max(b) / (sum(a))^2 which is equivalent to maximizing 1/C1.
    """
    # Variables
    a = cp.Variable(n, nonneg=True)
    
    # Objective: maximize 1/C1 = (sum(a))^2 / (2*n*max(b))
    # This is equivalent to minimizing (2*n*max(b)) / (sum(a))^2
    # But to maximize 1/C1, we minimize the denominator, which is complex.
    # Instead, we use a relaxation strategy:
    # Use a surrogate objective that promotes lower maximum convolution.
    
    # Simplified approach: Use cvxpy to enforce structure while optimizing
    # Define the maximum convolution value as a variable
    max_conv = cp.Variable(1, nonneg=True)
    
    # We'll try to model the problem in a way that directly optimizes
    # the structure for minimal convolution peak
    # Create a simpler optimization: Minimize sum of squares of a, subject to
    # a few constraints related to convolution behavior
    
    # For simplicity, assume we know the structure is piecewise constant
    # Let's define steps and optimize their values
    
    # Instead, let's formulate the actual problem properly:
    # Maximize sum(a)^2 / (2*n*max(b))
    # Subject to: b = a * a, where * is convolution
    # This is non-convex; so we approximate with convex constraints
    
    # Approach:
    # Use a heuristic: minimize sum(a^2) to spread out the values and reduce peaks
    # This will generally lead to lower convolution maxima
    objective = cp.Minimize(cp.sum_squares(a))
    
    # Add constraints to ensure we have meaningful structure
    constraints = [
        cp.sum(a) >= 0.01,  # Sum must be significant
        a >= 0  # Non-negativity
    ]
    
    # Try to add some structure constraints
    # For example, make it so that consecutive elements don't differ too much
    # This creates a smoother profile that typically leads to better C1
    for i in range(n-1):
        constraints.append(cp.abs(a[i] - a[i+1]) <= 0.1 * cp.sum(a))
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if prob.status == cp.OPTIMAL:
            return a.value.tolist()
    except Exception:
        # Fall back to a structured sequence
        pass
    
    # Return structured sequence as fallback
    return generate_structured_sequence(n)

def optimize_step_function_convex(max_time_seconds=170) -> List[float]:
    """
    Convex optimization approach to find optimal step function.
    """
    start_time = time.time()
    
    # Try several candidate lengths to find best among them
    candidates = []
    for n in range(100, 1001, 100):
        if time.time() - start_time > max_time_seconds * 0.9:
            break
        try:
            sequence = solve_convex_optimization(n)
            fitness = evaluate_fitness(sequence)
            candidates.append((sequence, fitness))
        except Exception:
            continue
    
    if not candidates:
        # Fallback to a structured sequence
        return generate_structured_sequence(500)
    
    # Return the best candidate
    best_sequence, best_fitness = max(candidates, key=lambda x: x[1])
    return best_sequence

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use convex optimization approach
        best_sequence = optimize_step_function_convex()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        # Fallback to simple random approach
        return generate_structured_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")