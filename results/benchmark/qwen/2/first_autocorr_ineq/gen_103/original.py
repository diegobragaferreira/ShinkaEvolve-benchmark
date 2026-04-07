# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import random
import time
import math

def compute_c1_constant(sequence):
    """Computes the C1 constant for a given sequence."""
    a = np.array(sequence)
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')
    
    conv = fftconvolve(a, a, mode='full')[:len(a)*2-1]
    max_conv = np.max(conv)
    n = len(a)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    return c1

def evaluate_sequence(sequence):
    """Evaluates a sequence and returns 1/C1 (the objective to maximize)."""
    try:
        c1 = compute_c1_constant(sequence)
        if c1 == float('inf'):
            return 0.0
        return 1.0 / c1
    except Exception:
        return 0.0

def generate_random_sequence(length=None, min_length=10, max_length=1000):
    """Generate a random sequence with specified or random length."""
    if length is None:
        length = random.randint(min_length, max_length)
    
    sequence = [random.uniform(0, 1000) for _ in range(length)]
    if all(x == 0 for x in sequence):
        sequence[0] = 1.0
    return sequence

def create_step_function_from_convolution(sequence):
    """Create a step function representation that approximates the convolution behavior."""
    a = np.array(sequence)
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return None
    
    conv = fftconvolve(a, a, mode='full')[:len(a)*2-1]
    max_conv = np.max(conv)
    
    # Normalize the convolution to approximate a step function
    normalized_conv = conv / max_conv if max_conv != 0 else conv
    return normalized_conv.tolist()

def quadratic_optimization_approach(initial_sequence, max_iter=1000):
    """Use quadratic programming to directly optimize the 1/C1 objective."""
    # Set up initial parameters
    n = len(initial_sequence)
    x0 = np.array(initial_sequence)
    
    # Define the objective function to minimize (negative of 1/C1)
    def objective(x):
        # Ensure non-negativity
        x = np.maximum(x, 0)
        
        # Compute the convolution
        conv = fftconvolve(x, x, mode='full')[:len(x)*2-1]
        
        max_conv = np.max(conv)
        sum_x = np.sum(x)
        
        if sum_x < 0.01:
            return 1e6  # Penalize invalid sequences
        
        c1 = (2 * len(x) * max_conv) / (sum_x ** 2)
        return -1.0 / c1  # Negative because we want to maximize 1/C1
    
    # Define the constraints
    def constraint_positive(x):
        return x
    
    # Use L-BFGS-B optimizer which handles bounds well
    bounds = [(0, 1000) for _ in range(n)]
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-9})
    
    if result.success:
        return result.x.tolist()
    else:
        return initial_sequence

def local_improvement_search(initial_sequence, max_iter=100):
    """Improve a sequence using local search around it."""
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence(current_sequence)

    for _ in range(max_iter):
        # Try mutating the current sequence
        mutated = [x * random.uniform(0.8, 1.2) for x in current_sequence]
        # Clip to valid range [0, 1000]
        mutated = [max(0, min(1000, x)) for x in mutated]
        mutated_fitness = evaluate_sequence(mutated)

        if mutated_fitness > current_fitness:
            current_sequence = mutated
            current_fitness = mutated_fitness

        # Try quadratic optimization
        optimized = quadratic_optimization_approach(current_sequence, 100)
        optimized_fitness = evaluate_sequence(optimized)
        
        if optimized_fitness > current_fitness:
            current_sequence = optimized
            current_fitness = optimized_fitness

    return current_sequence, current_fitness

def search_for_best_sequence():
    """Main search function to find the best sequence."""
    best_sequence = None
    best_inv_c1 = 0.0

    # Try multiple random starting points
    for attempt in range(5):
        # Random initialization
        initial_sequence = generate_random_sequence()
        
        # Quadratic optimization approach
        optimized_seq = quadratic_optimization_approach(initial_sequence)
        optimized_fitness = evaluate_sequence(optimized_seq)
        
        if optimized_fitness > best_inv_c1:
            best_inv_c1 = optimized_fitness
            best_sequence = optimized_seq

        # Also try local improvement
        improved_seq, improved_fitness = local_improvement_search(initial_sequence, 100)
        if improved_fitness > best_inv_c1:
            best_inv_c1 = improved_fitness
            best_sequence = improved_seq

    # Final local optimization on the best found sequence
    if best_sequence is not None:
        final_seq, final_fitness = local_improvement_search(best_sequence, 500)
        if final_fitness > best_inv_c1:
            best_inv_c1 = final_fitness
            best_sequence = final_seq

    return best_sequence if best_sequence is not None else generate_random_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")