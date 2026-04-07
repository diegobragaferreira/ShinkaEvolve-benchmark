# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution."""
    return fftconvolve(a, b, mode='full')[:2*len(a)-1]

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    b = convolve_fft(a, a)
    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

def greedy_construct_sequence(n: int) -> List[float]:
    """
    Construct a sequence greedily to minimize the C1 constant.
    Uses a geometric progression pattern that empirically performs well.
    """
    # Start with a geometric decay pattern
    base_ratio = 0.95
    sequence = [1000 * (base_ratio ** i) for i in range(n)]
    
    # Normalize to ensure reasonable sum
    total = sum(sequence)
    if total > 0:
        sequence = [x / total * n * 100 for x in sequence]
    
    # Apply a slight modification to enhance performance
    # Use a more refined decay factor for better balance
    base_ratio = 0.92
    refined_sequence = [1000 * (base_ratio ** i) for i in range(n)]
    refined_total = sum(refined_sequence)
    if refined_total > 0:
        refined_sequence = [x / refined_total * n * 100 for x in refined_sequence]
    
    return refined_sequence

def local_improve_sequence(sequence: List[float], iterations: int = 10) -> List[float]:
    """
    Apply local improvements to the sequence using gradient-like adjustments.
    """
    current = np.array(sequence, dtype=float)
    n = len(current)
    
    for _ in range(iterations):
        # Calculate convolution
        conv = convolve_fft(current, current)
        max_conv = np.max(conv)
        sum_current = np.sum(current)
        
        if sum_current < 1e-10:
            break
            
        # Simple gradient approximation
        eps = 1e-5
        grad = np.zeros_like(current)
        for i in range(n):
            current_eps = current.copy()
            current_eps[i] += eps
            conv_eps = convolve_fft(current_eps, current_eps)
            max_conv_eps = np.max(conv_eps)
            grad[i] = (max_conv_eps - max_conv) / eps
            
        # Update using gradient ascent (but bounded to keep values reasonable)
        learning_rate = 0.01
        current += learning_rate * grad
        current = np.maximum(current, 0)  # Ensure non-negativity
        
    return current.tolist()

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    best_inv_c1 = 0.0
    best_sequence = None
    start_time = time.time()
    max_time = 180
    
    # Try multiple sequence lengths to find the best
    lengths_to_try = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    for n in lengths_to_try:
        if time.time() - start_time > max_time:
            break
            
        # Greedily construct sequence
        sequence = greedy_construct_sequence(n)
        
        # Improve locally
        improved_sequence = local_improve_sequence(sequence, 20)
        
        # Evaluate
        _, inv_c1 = compute_c1_constant(improved_sequence)
        
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = improved_sequence.copy()
            
        # Early stopping if we're getting close to benchmark
        if best_inv_c1 > 0.6654:  # 1/1.5031
            break
    
    # Final refinement if needed
    if best_sequence is not None:
        final_sequence = local_improve_sequence(best_sequence, 10)
        _, final_inv_c1 = compute_c1_constant(final_sequence)
        if final_inv_c1 > best_inv_c1:
            best_sequence = final_sequence
    
    # Fallback if nothing was found
    if best_sequence is None:
        best_sequence = [1.0] * 100
        
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")