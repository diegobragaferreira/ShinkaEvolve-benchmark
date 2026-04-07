# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
import random
import time
from functools import partial

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def compute_c1(sequence):
    """Compute the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    # Use FFT-based convolution for efficiency
    conv = fftconvolve(sequence, sequence, mode='full')
    # Take only the relevant part of convolution
    max_conv = np.max(conv[len(sequence)-1:])  # From index n-1 onwards

    # Normalize and compute C1
    sum_sq = np.sum(sequence)**2
    if sum_sq == 0:
        return float('inf')

    c1 = (2 * len(sequence) * max_conv) / sum_sq
    return c1

def compute_inv_c1(sequence):
    """Compute inverse of C1 (what we want to maximize)."""
    c1 = compute_c1(sequence)
    if c1 == 0 or np.isinf(c1):
        return 0
    return 1.0 / c1

def compute_convolution_constraints(sequence):
    """Compute the convolution constraints matrix for LP solving."""
    n = len(sequence)
    if n < 1:
        return np.array([]), np.array([])
    
    # Normalize sequence
    sum_seq = np.sum(sequence)
    if sum_seq < 1e-10:
        return np.array([]), np.array([])
    
    norm_seq = sequence / sum_seq
    
    # Precompute convolution constraints efficiently
    num_constraints = 2 * n - 1
    a_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)

    # Generate convolution constraints using optimized loop
    for k in range(num_constraints):
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                a_ub[k, j] = norm_seq[i]
        b_ub[k] = 1.0  # RHS is set to 1 for normalization purposes
    
    # Add non-negativity constraints
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])
    
    return a_ub, b_ub

def compute_grad_conv(sequence):
    """Compute the gradient of max convolution with respect to sequence elements."""
    n = len(sequence)
    if n < 1:
        return np.zeros(n)
    
    sum_seq = np.sum(sequence)
    if sum_seq < 1e-10:
        return np.zeros(n)
    
    # Normalize
    norm_seq = sequence / sum_seq
    
    # Compute convolution
    conv = fftconvolve(norm_seq, norm_seq, mode='full')
    max_conv = np.max(conv)
    
    # Compute gradient of max convolution w.r.t. normalized sequence
    # This is a simplified gradient calculation, actual implementation would be more complex
    # Here we approximate the gradient based on how changes in elements affect the maximum
    grad = np.zeros(n)
    
    # For a full gradient, one would need to compute the derivative of convolution
    # and identify the contribution of each element to the maximum convolution value
    # This is a simplified heuristic gradient for demonstration
    for i in range(n):
        # Approximate contribution of element i to the maximum convolution
        # This is a placeholder for more sophisticated gradient computation
        grad[i] = 1.0 / (n + 1)  # Uniform approximation
    
    return grad

def direct_convolution_optimization(sequence):
    """Direct optimization in convolution space."""
    n = len(sequence)
    if n < 1:
        return sequence
    
    # Initialize with structured sequence
    init_seq = create_structured_sequence(n)
    
    # Objective function: negative inverse C1 (to minimize)
    def obj_func(x):
        return -compute_inv_c1(x)
    
    # Gradient function (approximate)
    def grad_func(x):
        # Compute approximate gradient
        return -compute_grad_conv(x)
    
    # Use L-BFGS-B with bounds for optimization
    bounds = [(0, 1000) for _ in range(n)]
    
    try:
        result = optimize.minimize(
            obj_func, 
            init_seq, 
            method='L-BFGS-B',
            jac=grad_func,
            bounds=bounds,
            options={'maxiter': 500}
        )
        
        if result.success:
            return result.x.tolist()
    except:
        pass
    
    # Fallback to simple gradient ascent
    current_seq = np.array(init_seq)
    lr = 0.001
    for _ in range(100):
        grad = compute_grad_conv(current_seq)
        current_seq += lr * grad
        current_seq = np.clip(current_seq, 0, 1000)
        
        # Stop if improvement is negligible
        if np.linalg.norm(grad) < 1e-6:
            break
            
    return current_seq.tolist()

def create_structured_sequence(n):
    """Create a structured sequence for better starting points."""
    # Combines exponential decay, uniform, and random components
    exp_decay = np.exp(-np.linspace(0, 3, n))
    uniform = np.ones(n)
    random_part = np.random.rand(n)

    # Blend patterns with weights
    base_seq = 0.5 * exp_decay + 0.3 * uniform + 0.2 * random_part

    # Scale and normalize
    base_seq = base_seq / np.sum(base_seq) * 10

    # Clip to valid range
    base_seq = np.clip(base_seq, 0, 1000)

    return base_seq.tolist()

def refine_with_convolution_analysis(sequence, max_iterations=10):
    """Refine sequence by analyzing convolution peaks and adjusting accordingly."""
    seq = np.array(sequence)
    n = len(seq)

    for _ in range(max_iterations):
        # Compute convolution
        conv = fftconvolve(seq, seq, mode='full')
        conv_part = conv[n-1:]
        max_conv = np.max(conv_part)

        # Find indices where convolution peaks
        max_indices = np.where(conv_part >= 0.9 * max_conv)[0]

        # Adjust elements that contribute to peaks
        new_seq = seq.copy()
        for idx in max_indices[:min(3, len(max_indices))]:
            for offset in [-2, -1, 0, 1, 2]:
                pos = idx + offset
                if 0 <= pos < n:
                    new_seq[pos] *= 0.98  # Slight reduction

        seq = np.maximum(new_seq, 0)

        # Early stopping if sequence stabilizes
        if np.allclose(seq, new_seq, rtol=1e-6):
            break

    return seq.tolist()

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    best_inv_c1 = 0
    best_sequence = None

    # Time tracking for budget management
    start_time = time.time()
    max_time = 170  # Leave 10 seconds for cleanup

    # Multiple attempts to find good sequences
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break

        # Create diverse initial sequences
        n = random.randint(100, 1000)
        initial_seq = create_structured_sequence(n)

        # Apply direct optimization in convolution space
        optimized_sequence = direct_convolution_optimization(initial_seq)
        
        # Further refinement
        final_sequence = refine_with_convolution_analysis(optimized_sequence)
        
        # Evaluate
        new_inv_c1 = compute_inv_c1(final_sequence)

        # Update best if improved
        if new_inv_c1 > best_inv_c1:
            best_inv_c1 = new_inv_c1
            best_sequence = final_sequence[:]

    # Final refinement
    if best_sequence is not None:
        final_refinement = refine_with_convolution_analysis(best_sequence)
        final_inv_c1 = compute_inv_c1(final_refinement)
        if final_inv_c1 > best_inv_c1:
            best_sequence = final_refinement

    # Fallback if nothing found
    if best_sequence is None:
        # Return a random good initial sequence
        n = random.randint(100, 500)
        best_sequence = create_structured_sequence(n)

    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")