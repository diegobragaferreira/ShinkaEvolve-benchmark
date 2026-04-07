# EVOLVE-BLOCK-START

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution
import random
import time

np.random.seed(42)
random.seed(42)

def compute_autocorrelation_constant(sequence):
    """
    Computes the autocorrelation constant C₁ for a given sequence.
    
    Parameters:
        sequence (list): List of non-negative real numbers representing step heights
    
    Returns:
        tuple: (C1_value, inv_C1_value) where inv_C1_value = 1/C1_value
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    a = np.array(sequence)
    n = len(a)
    
    # Compute convolution using FFT for efficiency
    b = fftconvolve(a, a, mode='full')
    b = b[n-1:2*n-1]  # Take only the relevant part of convolution
    
    max_b = np.max(b)
    sum_a = np.sum(a)
    
    if sum_a < 0.01:
        return float('inf'), 0.0
    
    C1 = 2 * n * max_b / (sum_a ** 2)
    inv_C1 = 1 / C1
    
    return C1, inv_C1

def generate_initial_sequences():
    """
    Generates a set of initial sequences with varying characteristics.
    """
    sequences = []
    
    # Generate sequences with different patterns
    for _ in range(5):
        n = np.random.randint(100, 1000)
        # Random uniform distribution
        seq = np.random.uniform(0, 1, n).tolist()
        sequences.append(seq)
        
    # Add some structured sequences
    for _ in range(3):
        n = np.random.randint(100, 1000)
        # Geometric decay pattern
        decay_factor = np.random.uniform(0.5, 0.9)
        seq = [decay_factor ** i for i in range(n)]
        sequences.append(seq)
        
    # Add some step functions
    for _ in range(3):
        n = np.random.randint(100, 1000)
        seq = [1.0] * (n // 2) + [0.0] * (n - n // 2)
        sequences.append(seq)
    
    return sequences

def adaptive_search_step(current_sequence, max_iterations=100):
    """
    Performs an adaptive search step to improve the sequence.
    """
    n = len(current_sequence)
    if n == 0:
        return current_sequence
        
    # Perform a local optimization step
    bounds = [(0, 1000) for _ in range(n)]
    
    def objective(x):
        _, inv_C1 = compute_autocorrelation_constant(x)
        return -inv_C1  # We want to maximize 1/C1, so we minimize -1/C1
    
    try:
        result = differential_evolution(objective, bounds, maxiter=50, seed=42)
        if result.success:
            return result.x.tolist()
    except Exception:
        pass
    
    # If optimization fails, make a simple perturbation
    new_sequence = current_sequence.copy()
    idx = np.random.randint(0, n)
    new_sequence[idx] = max(0, new_sequence[idx] + np.random.normal(0, 0.1))
    
    return new_sequence

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence.
    """
    start_time = time.time()
    max_time = 180  # seconds
    
    best_inv_C1 = 0.0
    best_sequence = None
    best_C1 = float('inf')
    
    # Initialize with multiple promising sequences
    initial_sequences = generate_initial_sequences()
    
    for i, init_seq in enumerate(initial_sequences):
        if time.time() - start_time > max_time - 5:
            break
            
        current_seq = init_seq.copy()
        current_C1, current_inv_C1 = compute_autocorrelation_constant(current_seq)
        
        if current_inv_C1 > best_inv_C1:
            best_inv_C1 = current_inv_C1
            best_sequence = current_seq.copy()
            best_C1 = current_C1
            
        # Adaptive search
        for iter_count in range(100):
            if time.time() - start_time > max_time - 5:
                break
                
            new_seq = adaptive_search_step(current_seq)
            new_C1, new_inv_C1 = compute_autocorrelation_constant(new_seq)
            
            if new_inv_C1 > current_inv_C1:
                current_seq = new_seq
                current_C1 = new_C1
                current_inv_C1 = new_inv_C1
                
                if current_inv_C1 > best_inv_C1:
                    best_inv_C1 = current_inv_C1
                    best_sequence = current_seq.copy()
                    best_C1 = current_C1
                    
    # Final refinement of the best found sequence
    if best_sequence is not None:
        bounds = [(0, 1000) for _ in range(len(best_sequence))]
        
        def objective(x):
            _, inv_C1 = compute_autocorrelation_constant(x)
            return -inv_C1
            
        try:
            result = differential_evolution(objective, bounds, maxiter=100, seed=42)
            if result.success:
                final_C1, final_inv_C1 = compute_autocorrelation_constant(result.x)
                if final_inv_C1 > best_inv_C1:
                    return result.x.tolist()
        except Exception:
            pass
    
    return best_sequence if best_sequence is not None else [1.0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
