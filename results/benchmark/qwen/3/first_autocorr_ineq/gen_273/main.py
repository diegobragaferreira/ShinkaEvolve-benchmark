# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
from typing import List, Tuple, Optional
import warnings

# Suppress scientific notation for cleaner output
np.set_printoptions(suppress=True)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def convolve_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Efficient FFT-based convolution for large sequences."""
    return fftconvolve(a, b, mode='full')[:2*len(a)-1]

def convolve_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct convolution for small sequences."""
    return np.convolve(a, b, mode='full')[:2*len(a)-1]

def adaptive_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Adaptive convolution method selection based on sequence properties."""
    n = len(a)

    # For very small sequences, always use direct method
    if n < 50:
        return convolve_direct(a, b)

    # For sequences with very low variance, prefer direct
    if np.std(a) < 1e-3 or np.std(b) < 1e-3:
        return convolve_direct(a, b)

    # Prefer FFT for larger sequences
    return convolve_fft(a, b)

def compute_c1_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 constant and 1/C1 value for a given sequence."""
    a = np.array(sequence)
    n = len(a)

    # Use FFT for efficiency when sequence is large and stable
    if n > 100:
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

def compute_gradient_estimate(a: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute a numerical gradient estimate of how changes in step heights affect max convolution.
    This is done by computing how small perturbations in each step height change the convolution maximum.
    """
    n = len(a)
    grad = np.zeros(n)
    base_conv = adaptive_convolve(a, a)
    base_max = np.max(base_conv)

    for i in range(n):
        # Perturb step i
        a_plus = a.copy()
        a_plus[i] += epsilon
        a_minus = a.copy()
        a_minus[i] -= epsilon

        # Compute convolution for both perturbed sequences
        conv_plus = adaptive_convolve(a_plus, a_plus)
        conv_minus = adaptive_convolve(a_minus, a_minus)
        
        # Estimate gradient using central difference
        max_plus = np.max(conv_plus)
        max_minus = np.max(conv_minus)
        
        grad[i] = (max_plus - max_minus) / (2 * epsilon)
    
    return grad

def project_onto_feasible_set(a: np.ndarray) -> np.ndarray:
    """Project sequence onto feasible set with non-negativity and bounded values."""
    return np.clip(a, 0, 1000)

def gradient_step(a: np.ndarray, lr: float, use_curvature: bool = True) -> np.ndarray:
    """Perform a gradient step with optional curvature-aware learning rate."""
    # Compute gradient estimate
    grad = compute_gradient_estimate(a)

    # Adjust learning rate based on curvature (simplified approach)
    if use_curvature:
        # Estimate effective curvature from gradient changes
        curvature = np.linalg.norm(grad)  # Simplified curvature measure
        # Reduce learning rate when curvature is high (flatter region)
        lr_adjusted = lr / (1.0 + 0.1 * curvature)
    else:
        lr_adjusted = lr
    
    # Perform gradient update
    updated = a - lr_adjusted * grad
    
    # Project back to feasible set
    projected = project_onto_feasible_set(updated)
    
    # Ensure minimum sum constraint
    if np.sum(projected) < 0.01:
        # If sum is too small, scale up uniformly
        scale = 0.01 / (np.sum(projected) + 1e-10)
        projected = projected * scale
    
    return projected

def adaptive_gradient_optimization(initial_sequence: List[float], 
                                  max_iterations: int = 100, 
                                  initial_lr: float = 0.1) -> List[float]:
    """Perform gradient-based optimization with adaptive learning rates."""
    a = np.array(initial_sequence).astype(float)
    lr = initial_lr
    
    for iteration in range(max_iterations):
        # Compute current C1 value
        _, inv_c1 = compute_c1_constant(a.tolist())
        
        # Perform gradient step
        a_new = gradient_step(a, lr, use_curvature=True)
        
        # Compute new C1 value
        _, inv_c1_new = compute_c1_constant(a_new.tolist())
        
        # Accept the move if it improves the objective
        if inv_c1_new > inv_c1:
            a = a_new
            # Adaptively adjust learning rate
            lr = min(1.0, lr * 1.05)  # Gradually increase LR if successful
        else:
            # Decrease learning rate if no improvement
            lr = max(1e-6, lr * 0.95)
            
    return a.tolist()

def search_for_best_sequence() -> List[float]:
    """Main search function that finds the optimal sequence using gradient-based optimization."""
    # Start with a variety of initial sequences
    initial_sequences = []
    
    # Add some structured sequences from the literature
    initial_sequences.append([1.0] * 100)
    initial_sequences.append([1.0] * 50 + [0.0] * 50)
    initial_sequences.append([1.0, 0.0] * 50)
    
    # Add random sequences
    for _ in range(5):
        n = random.randint(100, 500)
        seq = [random.random() * 100 for _ in range(n)]
        initial_sequences.append(seq)
    
    best_sequence = None
    best_inv_c1 = 0.0
    
    # Iterate through initial sequences
    for seq in initial_sequences:
        # Apply gradient-based optimization
        optimized = adaptive_gradient_optimization(seq, max_iterations=50)
        
        # Evaluate the result
        _, inv_c1 = compute_c1_constant(optimized)
        
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = optimized
            
    # Fine-tune with additional rounds if needed
    if best_sequence is not None:
        final_sequence = adaptive_gradient_optimization(best_sequence, max_iterations=50)
        _, inv_c1 = compute_c1_constant(final_sequence)
        
        if inv_c1 > best_inv_c1:
            best_sequence = final_sequence
    
    # If no good sequence found, return a default
    return best_sequence if best_sequence is not None else [1.0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")