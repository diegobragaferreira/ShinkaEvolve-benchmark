# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import time
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    Returns 1/C₁ which we want to maximize.
    """
    if len(sequence) == 0 or np.sum(sequence) < 0.01:
        return 0.0
    
    # Compute convolution using FFT for efficiency
    conv = fftconvolve(sequence, sequence, mode='full')
    # Take the maximum of the convolution (excluding the zero-padding)
    max_conv = np.max(conv[len(sequence)-1:])
    
    # Calculate C₁ = 2*n*max(b) / (sum(a))^2
    sum_a = np.sum(sequence)
    n = len(sequence)
    
    if sum_a == 0:
        return 0.0
        
    C1 = 2 * n * max_conv / (sum_a ** 2)
    return 1 / C1  # Return reciprocal for maximization

def quadratic_programming_optimization():
    """
    Uses quadratic programming to find optimal step function.
    Reformulates the problem to directly optimize the inverse autocorrelation constant.
    """
    # Start with a good initial guess
    n = random.randint(200, 800)
    initial_sequence = np.random.rand(n) * 100  # Random sequence with some scale
    
    # Add a structured component to encourage good solutions
    # Use a decaying pattern that balances mass concentration and convolution
    decay_pattern = np.exp(-np.arange(n) / (n/5))
    initial_sequence = initial_sequence * 0.3 + decay_pattern * 0.7
    
    # Normalize to ensure meaningful magnitude
    sum_seq = np.sum(initial_sequence)
    if sum_seq > 0:
        initial_sequence = initial_sequence / sum_seq * 100
    
    # Objective: maximize 1/C₁, which is equivalent to minimizing C₁
    # We'll use a constrained optimization approach with a penalty method
    def objective(x):
        # Ensure non-negativity
        x = np.maximum(x, 0)
        # Normalize
        sum_x = np.sum(x)
        if sum_x < 1e-6:
            return 1e10
        x = x / sum_x
        # Compute C1
        c1 = compute_autocorrelation_constant(x)
        if c1 <= 0:
            return 1e10
        # Minimize negative of 1/C1 (since we want to maximize 1/C1)
        return -1.0 / c1
    
    # Constraints
    # For now, we only enforce non-negativity
    constraints = [{'type': 'ineq', 'fun': lambda x: x}]  # x >= 0
    
    # Bounds for each variable
    bounds = [(0, 1000) for _ in range(n)]
    
    # Solve using L-BFGS with bounds (more suitable for this problem)
    result = minimize(objective, initial_sequence, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 500, 'ftol': 1e-9})
    
    if result.success:
        optimized_sequence = result.x
        # Ensure normalization
        sum_opt = np.sum(optimized_sequence)
        if sum_opt > 0:
            optimized_sequence = optimized_sequence / sum_opt * 100
    else:
        # Return initial sequence if optimization failed
        optimized_sequence = initial_sequence
    
    return optimized_sequence

def search_for_best_sequence() -> list[float]:
    """
    Main function to search for the best coefficient sequence.
    Uses quadratic programming approach.
    """
    start_time = time.time()
    max_time = 170  # Leave 10 seconds for cleanup
    
    best_score = 0.0
    best_sequence = None
    
    # Try multiple restarts to escape local optima
    for attempt in range(5):
        if time.time() - start_time > max_time:
            break
            
        try:
            sequence = quadratic_programming_optimization()
            score = compute_autocorrelation_constant(sequence)
            
            if score > best_score:
                best_score = score
                best_sequence = sequence.copy()
                
                # Check if we beat the benchmark
                if score > 1.5031:
                    print(f"BEAT BENCHMARK! Score: {score:.6f}")
                    break
        except Exception as e:
            continue
    
    # Final fallback if nothing worked
    if best_sequence is None:
        n = 200
        best_sequence = np.random.rand(n) * 100
        best_sequence = best_sequence / np.sum(best_sequence) * 100
    
    # Ensure it meets requirements
    if np.sum(best_sequence) < 0.01:
        best_sequence[0] += 0.01
    
    # Clip values to [0, 1000] for practicality
    best_sequence = np.clip(best_sequence, 0, 1000).tolist()
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f}s with score: {best_score:.6f}")
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")