# EVOLVE-BLOCK-START
import numpy as np
from cvxpy import *
import random
import time
from typing import List, Tuple, Optional

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')[:2*len(a)-1]

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use direct convolution for small sequences to ensure accuracy
    if n <= 100:
        b = convolve_direct(a, a)
    else:
        # For larger sequences, use FFT if available
        try:
            from scipy.signal import fftconvolve
            b = fftconvolve(a, a, mode='full')[:2*n-1]
        except:
            b = convolve_direct(a, a)

    max_conv = np.max(b)
    sum_a = np.sum(a)

    if sum_a < 0.01:
        return float('inf'), 0.0

    c1 = 2 * n * max_conv / (sum_a ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    return c1, inv_c1

def solve_convolution_quadratic_optimization(sequence: List[float]) -> Optional[List[float]]:
    """Solve the convolution optimization problem using disciplined convex programming."""
    try:
        n = len(sequence)
        if n < 10:
            return None
            
        # Define variables
        x = Variable(n, nonneg=True)
        
        # Objective: maximize 1/C1 = (sum(x))^2 / (2*n*max(conv(x,x)))
        # This is equivalent to minimizing (2*n*max(conv(x,x))) / (sum(x))^2
        # Which we can approximate by minimizing sum(x)^2 / max(conv(x,x)) for practical purposes
        
        # We'll work with a simplified quadratic relaxation of the constraint
        # For now, let's just try to find a point that satisfies basic constraints
        # Let's use a simpler approach: we try to minimize sum(x)^2 subject to max(conv(x,x)) <= 1
        # This approximates maximizing 1/C1
        
        # Create constraints manually
        # For a true convex formulation, we'd express the convolution constraint properly
        # But because of the complexity of convolution in CVXPY, we take a heuristic approach:
        # We use the fact that we want to spread the mass to reduce max convolution value
        
        # Heuristic: Try to make sequence as uniform as possible while maintaining positivity
        # This often leads to better convolution properties due to spreading out mass
        target_sum = sum(sequence)
        uniform_x = [target_sum / n] * n
        
        # Add slight variation to avoid local minima
        for i in range(n):
            uniform_x[i] *= (1 + 0.1 * (i % 3 - 1))  # Vary slightly to avoid plateaus
            
        # Ensure all values are positive
        uniform_x = [max(1e-6, val) for val in uniform_x]
        
        # Normalize to preserve sum
        actual_sum = sum(uniform_x)
        scaled_x = [val * target_sum / actual_sum for val in uniform_x]
        
        return scaled_x
        
    except Exception:
        return None

def improve_sequence_quadratically(sequence: List[float]) -> Optional[List[float]]:
    """Use a quadratic optimization inspired approach to improve the sequence."""
    try:
        n = len(sequence)
        if n < 10:
            return None
            
        # For simplicity and efficiency, apply a gradient-like update based on 
        # local properties of the sequence. This involves computing a local 
        # measure of the convolution to guide a descent step.
        
        # First, compute the current convolution to understand the structure
        a = np.array(sequence)
        if n <= 100:
            b = convolve_direct(a, a)
        else:
            try:
                from scipy.signal import fftconvolve
                b = fftconvolve(a, a, mode='full')[:2*n-1]
            except:
                b = convolve_direct(a, a)
                
        max_conv = np.max(b)
        sum_a = np.sum(a)
        
        if sum_a < 0.01:
            return None
            
        # Create a simple gradient-based update that tries to reduce max(b)
        # by adjusting the sequence to distribute the mass more evenly
        # We'll use a smoothing operator in the spirit of quadratic optimization
        
        # Compute a smoothed version of the sequence
        smoothed = np.copy(a)
        if n > 10:
            # Apply a simple averaging filter to smooth the sequence
            for _ in range(5):  # Multiple passes for stronger smoothing
                new_smoothed = np.copy(smoothed)
                for i in range(n):
                    neighbors = []
                    for di in [-2, -1, 0, 1, 2]:
                        if 0 <= i + di < n:
                            neighbors.append(smoothed[i + di])
                    if neighbors:
                        new_smoothed[i] = np.mean(neighbors)
                smoothed = new_smoothed
                
        # Scale the smoothed version to preserve sum
        smoothed_sum = np.sum(smoothed)
        if smoothed_sum > 1e-10:
            smoothed = smoothed * sum_a / smoothed_sum
            
        # Clip values to stay within bounds
        smoothed = np.clip(smoothed, 0, 1000)
        
        # Return as list
        return smoothed.tolist()
        
    except Exception:
        return None

def adaptive_search_for_best_sequence(
    max_time_seconds: int = 180,
    initial_length_range: Tuple[int, int] = (100, 500),
    attempts: int = 100  # Increase attempts for better exploration
) -> List[float]:
    """Main function to search for the best coefficient sequence using quadratic optimization-inspired approach."""
    start_time = time.time()
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Start with several random sequences and iteratively improve them
    for attempt in range(attempts):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Generate a random sequence
        n = random.randint(*initial_length_range)
        sequence = [random.random() * 100 for _ in range(n)]
        
        # Try to improve the sequence
        improved_sequence = improve_sequence_quadratically(sequence)
        
        if improved_sequence is not None:
            # Evaluate the improved sequence
            _, inv_c1 = compute_c1_constant(improved_sequence)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = improved_sequence.copy()
                
        # Also try some deterministic sequences that often perform well
        if attempt % 10 == 0:  # Every 10th attempt, try a deterministic sequence
            n = random.randint(*initial_length_range)
            # Try a sequence with exponential decay (often good for autocorrelation)
            sequence = [0.9 ** i for i in range(n)]
            # Normalize
            s = sum(sequence)
            if s > 0:
                sequence = [x / s for x in sequence]
                
            improved_sequence = improve_sequence_quadratically(sequence)
            if improved_sequence is not None:
                _, inv_c1 = compute_c1_constant(improved_sequence)
                if inv_c1 > best_inv_c1:
                    best_inv_c1 = inv_c1
                    best_sequence = improved_sequence.copy()

    # Final polishing using a few gradient steps
    if best_sequence is not None:
        for _ in range(20):
            polished_sequence = improve_sequence_quadratically(best_sequence)
            if polished_sequence is None:
                break
                
            _, inv_c1_new = compute_c1_constant(polished_sequence)
            _, inv_c1_old = compute_c1_constant(best_sequence)
            
            if inv_c1_new > inv_c1_old:
                best_sequence = polished_sequence
            else:
                break

    return best_sequence if best_sequence is not None else [1.0]

def search_for_best_sequence() -> List[float]:
    """Function to search for the best coefficient sequence."""
    return adaptive_search_for_best_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")