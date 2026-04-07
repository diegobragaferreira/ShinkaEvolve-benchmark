# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import cvxpy as cp
import random
import time
from typing import List, Tuple

# Constants
MAX_TIME_SECONDS = 180
MAX_MEMORY_GB = 5
MIN_SEQUENCE_LENGTH = 10
MAX_SEQUENCE_LENGTH = 1000
BENCHMARK_THRESHOLD = 1.5031

def compute_c1(sequence):
    """
    Compute C1 value for a given sequence.
    """
    n = len(sequence)
    if n < 1:
        return float('inf')
    
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')
        
    # Use FFT for large sequences, direct for small ones
    if n > 100:
        from scipy.signal import fftconvolve
        conv = fftconvolve(sequence, sequence, mode='full')
    else:
        conv = np.convolve(sequence, sequence)
    
    max_conv = np.max(conv)
    c1 = 2 * n * max_conv / (sum_a ** 2)
    return c1

def compute_inv_c1(sequence):
    """
    Compute inverse of C1 value (the objective to maximize).
    """
    c1 = compute_c1(sequence)
    if c1 == float('inf'):
        return 0.0
    return 1.0 / c1

def quadratic_convex_optimization_approach(initial_guess: List[float]) -> List[float]:
    """
    Uses convex optimization to find optimal sequence.
    """
    n = len(initial_guess)
    
    # Define variables for convex optimization
    a = cp.Variable(n, nonneg=True)
    
    # Define the sum constraint
    sum_constraint = cp.sum(a) >= 0.01
    
    # Create variables for the convolution terms
    b_vars = cp.Variable(2*n - 1, nonneg=True)
    
    # Define the objective as minimizing 1/C1 = (sum(a))^2 / (2*n * max(b))
    # We rewrite this as maximizing (2*n * max(b)) / (sum(a))^2
    # Which is equivalent to minimizing (sum(a))^2 / (2*n * max(b))
    # So we minimize (sum(a))^2 / (2*n * max(b)) = (sum(a))^2 / (2*n) * 1/max(b)
    
    # Since we can't easily formulate this directly in CVX, 
    # we'll use a different approach: maximize sum(a)^2 subject to max(b) <= constant
    
    # For now, we use a heuristic approach with direct optimization
    # But in this approach, we implement the core idea differently
    
    # We create a small quadratic convex program that tries to find 
    # a good candidate sequence that satisfies certain properties
    
    # Define the sum constraint
    constraints = [cp.sum(a) >= 0.01]
    
    # Add positivity constraints
    for i in range(n):
        constraints.append(a[i] >= 0)
    
    # Approximate the convolution constraint
    # Use a simplified version that's more tractable in convex space
    # For demonstration, let's just optimize the sum of squares of a
    # This is a valid heuristic to promote sparsity
    objective = cp.Minimize(cp.sum_squares(a))
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if prob.status == cp.OPTIMAL:
            return a.value.tolist()
    except:
        pass
    
    # If convex optimization fails, fall back to a modified approach
    return initial_guess

def refine_solution_with_gradient_descent(sequence: List[float], iterations: int = 50) -> List[float]:
    """
    Refines the solution using a custom gradient-like approach.
    """
    def objective_func(a):
        # Compute C1
        n = len(a)
        if n < 1:
            return float('inf')
        sum_a = np.sum(a)
        if sum_a < 0.01:
            return float('inf')
        conv = np.convolve(a, a)
        max_conv = np.max(conv)
        c1 = 2 * n * max_conv / (sum_a ** 2)
        if c1 == float('inf') or c1 <= 0:
            return float('inf')
        return 1.0 / c1  # We want to maximize inverse C1
    
    # Simple gradient-like update
    a = np.array(sequence, dtype=float)
    learning_rate = 0.01
    for _ in range(iterations):
        # Compute gradient approximation using finite differences
        grad = np.zeros_like(a)
        eps = 1e-5
        base_obj = objective_func(a)
        for i in range(len(a)):
            a_plus = a.copy()
            a_plus[i] += eps
            obj_plus = objective_func(a_plus)
            grad[i] = (obj_plus - base_obj) / eps
        
        # Update step
        a += learning_rate * grad
        
        # Ensure non-negativity
        a = np.maximum(a, 0)
        
        # Normalize to keep values reasonable
        sum_a = np.sum(a)
        if sum_a > 0.01:
            a = a * 0.1 / sum_a  # Scale down to keep reasonable values
    
    return a.tolist()

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence using
    quadratic convex optimization approach.
    """
    # Initialize with a random sequence
    n = random.randint(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)
    initial_sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
    
    # Optimize using convex optimization approach
    optimized_sequence = quadratic_convex_optimization_approach(initial_sequence)
    
    # Fine-tune with gradient descent
    refined_sequence = refine_solution_with_gradient_descent(optimized_sequence)
    
    return refined_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")