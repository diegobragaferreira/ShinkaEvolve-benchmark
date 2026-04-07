# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from cvxpy import *
import random
import time
import cvxpy as cp

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autocorrelation_constant(sequence):
    """Compute the autocorrelation constant C₁ for a sequence using efficient convolution."""
    n = len(sequence)
    if n == 0:
        return float('inf')
        
    sum_seq = np.sum(sequence)
    if sum_seq < 1e-10:
        return float('inf')
    
    # Use FFT for fast convolution
    padded_seq = np.pad(sequence, (0, n-1), 'constant', constant_values=0)
    conv_result = np.real(np.fft.ifft(np.fft.fft(padded_seq) * np.conj(np.fft.fft(padded_seq))))
    
    max_conv = np.max(conv_result[:2*n-1])
    
    # Compute C₁ = 2n * max(b) / (sum(a))^2
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    
    return c1

def evaluate_fitness(sequence):
    """Evaluate the fitness of a sequence (inverse of C₁)."""
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0
    return 1.0 / c1

def quadratic_convex_optimize(n):
    """
    Formulate and solve the problem as a convex optimization to find optimal step heights.
    This approach models the maximization of 1/C₁ as a convex optimization problem.
    """
    # Variables
    a = Variable(n, nonneg=True)
    
    # Objective function: Minimize the ratio max(b) / (sum(a))^2
    # This is equivalent to maximizing 1/C₁ = (sum(a))^2 / (2n * max(b))
    # In convex form we work with sum(a)^2 and max(b) separately
    
    # For simplicity, we use a heuristic approach that approximates the problem:
    # We approximate the maximum convolution value using a convex upper bound
    # and minimize the inverse objective
    
    # Create symbolic variables for the sum and max_conv
    sum_a = sum(a)
    
    # Simplified convex approximation approach:
    # We'll solve a related convex problem that tries to balance sum(a) vs max convolution
    
    # Objective: (sum(a))^2 / (max_conv) should be maximized
    # Which means we want to maximize (sum(a))^2 subject to max_conv bounded.
    # Let's simplify to a proxy: minimize (sum(a))^2 + lambda * max_conv
    
    # But in practice, we directly compute the objective with a heuristic:
    
    # Sample a few initial candidate sequences to understand the landscape
    candidates = []
    for _ in range(10):
        # Create a sequence with varying step heights
        seq = np.random.rand(n)
        seq = seq / np.sum(seq) * 100  # Scale appropriately
        candidates.append(seq)
    
    # Find the one that gives the best performance
    best_seq = None
    best_fitness = -np.inf
    
    for seq in candidates:
        fit = evaluate_fitness(seq)
        if fit > best_fitness:
            best_fitness = fit
            best_seq = seq
    
    return best_seq if best_seq is not None else [1.0] * n

def get_good_direction_to_move_into(sequence):
    """Returns an improved direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    
    # Avoid division by zero
    if sum_sequence < 1e-10:
        return None
        
    # Create a simple gradient update for the objective
    # Use finite differences to estimate gradient for 1/C1
    eps = 1e-5
    base_fitness = evaluate_fitness(sequence)
    
    # Estimate gradient per element
    grad = np.zeros(n)
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = eps
        seq_plus = np.array(sequence) + delta
        seq_minus = np.array(sequence) - delta
        seq_plus = np.maximum(seq_plus, 0)
        seq_minus = np.maximum(seq_minus, 0)
        
        fitness_plus = evaluate_fitness(seq_plus)
        fitness_minus = evaluate_fitness(seq_minus)
        
        grad[i] = (fitness_plus - fitness_minus) / (2 * eps)
    
    # Perform gradient ascent step
    step_size = 0.1
    new_sequence = np.array(sequence) + step_size * grad
    new_sequence = np.maximum(new_sequence, 0)  # Keep non-negative
    
    return new_sequence.tolist()

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    start_time = time.time()
    
    best_sequence = None
    best_fitness = 0.0
    
    # Try different lengths to see if we can get better results
    lengths_to_try = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Use the convex optimization approach for different sizes
    for length in lengths_to_try:
        if time.time() - start_time > 170:
            break
            
        # Try to find an optimal sequence
        try:
            sequence = quadratic_convex_optimize(length)
            
            # Enhance with local refinement
            enhanced_sequence = get_good_direction_to_move_into(sequence)
            if enhanced_sequence is not None:
                enhanced_fitness = evaluate_fitness(enhanced_sequence)
                if enhanced_fitness > best_fitness:
                    best_fitness = enhanced_fitness
                    best_sequence = enhanced_sequence
            else:
                # Fallback to original if enhancement fails
                seq_fitness = evaluate_fitness(sequence)
                if seq_fitness > best_fitness:
                    best_fitness = seq_fitness
                    best_sequence = sequence
                    
        except Exception as e:
            continue
            
    # Final fallback
    if best_sequence is None:
        n = random.randint(100, 1000)
        best_sequence = [random.uniform(0.01, 1000.0) for _ in range(n)]
        
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")