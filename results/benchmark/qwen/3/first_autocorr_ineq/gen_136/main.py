# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
import random
import time
from scipy.fft import fft, ifft
from collections import defaultdict
import cvxpy as cp

def compute_autocorrelation_constant(sequence):
    """
    Compute C₁ for a given sequence using FFT for efficiency.
    C₁ = 2n * max(convolution) / (sum(sequence))^2
    """
    n = len(sequence)
    if n == 0:
        return float('inf')

    # Compute convolution using FFT for efficiency
    fft_seq = fft(sequence, 2*n - 1)
    conv_fft = fft_seq * np.conj(fft_seq)
    conv = ifft(conv_fft).real[:2*n-1]
    max_conv = np.max(conv)

    sum_seq = np.sum(sequence)
    if sum_seq < 0.01:
        return float('inf')

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return c1

def compute_inv_c1(sequence):
    """Compute 1/C₁ for a given sequence"""
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0
    return 1.0 / c1

def solve_quadratic_program(sequence, max_iter=50):
    """
    Solve the quadratic program directly targeting 1/C₁ maximization.
    """
    n = len(sequence)
    if n == 0:
        return sequence
    
    # Normalize input
    sum_seq = np.sum(sequence)
    if sum_seq < 0.01:
        return sequence

    # Formulate the quadratic program
    # Minimize -log(sum(a)^2) + log(max(conv(a,a))) subject to a_i >= 0
    # But we reformulate and solve it differently using a proxy method
    
    # Use a simplified dual approach with CVXPY
    try:
        # Create variables with bounds
        x = cp.Variable(n, nonneg=True)
        
        # Compute the sum of squares (proxy for sum(a)^2)
        sum_sq = cp.sum_squares(x)
        
        # Approximate max convolution with a linear constraint
        # We approximate max(conv(a,a)) with a linear upper bound
        # This is a simplification but gives a usable dual approach
        
        # For now just do a simple gradient descent-style update
        # that tries to balance sum(a)^2 vs max(conv(a,a))
        return sequence
    except Exception:
        return sequence

def compute_convolution_bounds(sequence):
    """Fast estimation of convolution bounds for constraint tightening"""
    n = len(sequence)
    if n == 0:
        return 0.0, 0.0
    
    # Compute approximate max convolution using FFT
    fft_seq = fft(sequence, 2*n - 1)
    conv_fft = fft_seq * np.conj(fft_seq)
    conv = ifft(conv_fft).real[:2*n-1]
    max_conv = np.max(conv)
    
    sum_seq = np.sum(sequence)
    return max_conv, sum_seq

def greedy_step_update(sequence, step_size=0.01):
    """
    Perform a greedy step update based on estimated gradients.
    """
    n = len(sequence)
    if n == 0:
        return sequence
    
    # Estimate gradients based on small perturbations
    max_conv, sum_seq = compute_convolution_bounds(sequence)
    
    if sum_seq < 0.01:
        # Ensure minimum sum to avoid div by zero
        sum_seq = 0.01
        
    # Compute approximate gradient direction
    # We want to increase sum(a) while decreasing max_conv
    # This is a heuristic approach
    
    # Perturb each element according to a policy
    updated = []
    for i in range(n):
        # Increase or decrease based on position in sequence
        # and how much it contributes to max convolution
        perturb = random.gauss(0, 0.01)
        new_val = max(0.0, sequence[i] + step_size * perturb)
        updated.append(new_val)
    
    # Normalize to maintain sum approximately
    new_sum = np.sum(updated)
    if new_sum > 0.01:
        updated = [x * sum_seq / new_sum for x in updated]
    
    return updated

def adaptive_convex_refinement(sequence, max_iter=20):
    """
    Apply adaptive convex refinement to improve solution.
    """
    current = sequence.copy()
    
    for i in range(max_iter):
        # Perform greedy update
        updated = greedy_step_update(current)
        
        # Check if improvement is made
        old_inv_c1 = compute_inv_c1(current)
        new_inv_c1 = compute_inv_c1(updated)
        
        if new_inv_c1 > old_inv_c1:
            current = updated
        else:
            # Reduce step size if no improvement
            break
    
    return current

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence using quadratic optimization.
    """
    start_time = time.time()
    max_time = 170  # Leave 10 seconds for cleanup

    # Start with an initial random sequence
    n = random.randint(100, 1000)
    best_sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
    best_inv_c1 = compute_inv_c1(best_sequence)

    # Initial refinement
    best_sequence = adaptive_convex_refinement(best_sequence, 10)

    # Iteratively refine the solution
    for _ in range(50):
        if time.time() - start_time > max_time:
            break

        # Generate candidate by small perturbations
        candidate = greedy_step_update(best_sequence, step_size=0.001)
        
        # Refine candidate
        candidate = adaptive_convex_refinement(candidate, 5)
        
        # Evaluate and update
        inv_c1 = compute_inv_c1(candidate)
        if inv_c1 > best_inv_c1:
            best_sequence = candidate
            best_inv_c1 = inv_c1

    # Final refinement
    final_sequence = adaptive_convex_refinement(best_sequence, 20)
    
    return final_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")