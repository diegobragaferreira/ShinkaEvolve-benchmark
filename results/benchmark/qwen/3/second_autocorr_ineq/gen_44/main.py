# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

@njit
def compute_convolution_norms_numba(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    JIT compiled version for improved performance with proper trapezoidal integration.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using proper convolution
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Compute autoconvolution - efficient loop
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j] * dx

    # Compute norms using trapezoidal rule for integration
    # ||g||₂² using trapezoidal integration of g²
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

@njit
def compute_c2_numba(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞) - JIT compiled version"""
    g2_sq, g1, ginf = compute_convolution_norms_numba(f_values)

    if g1 == 0 or ginf == 0:
        return 0.0

    return g2_sq / (g1 * ginf)

def compute_c2(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    return compute_c2_numba(f_values)

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Use a fixed number of steps for reproducibility and efficiency
    n_steps = 1000  # Increased from 200 to improve resolution

    # Start with a sophisticated initial construction based on mathematical insights
    # Create a symmetric function that promotes flat convolution profiles
    # This approach tries to avoid sharp peaks that reduce C₂
    
    # Create a smooth, symmetric pattern that encourages uniform convolution
    f_values = []
    midpoint = n_steps // 2
    
    # Base construction with gradual rise and fall
    for i in range(n_steps):
        if i <= midpoint:
            # Rising part - starts from 0
            f_values.append(i / midpoint)
        else:
            # Falling part - decreases back to 0
            f_values.append((n_steps - i) / (n_steps - midpoint))
    
    # Normalize the function so its integral is reasonable
    total_area = sum(f_values) * (0.5 / n_steps)
    if total_area > 0:
        f_values = [x / total_area for x in f_values]
    
    # Apply some noise to escape local optima
    for i in range(len(f_values)):
        noise_factor = 0.05
        noise = np.random.normal(0, noise_factor * f_values[i])
        f_values[i] = max(0, f_values[i] + noise)
    
    # Refine using adaptive local search 
    best_f = f_values.copy()
    best_c2 = compute_c2(best_f)
    
    # Adaptive local search parameters
    max_iterations = 500  # Reduced to ensure timelimit compliance
    improvement_threshold = 0.0001  # Minimum meaningful improvement
    patience = 20  # Stop if no improvement for this many iterations
    
    # Track recent improvements for adaptive behavior
    recent_improvements = []
    current_patience = 0
    
    # Perform adaptive local optimization
    for iteration in range(max_iterations):
        test_f = best_f.copy()
        
        # Modify a few random positions with adaptive strategy
        num_modifications = max(1, min(50, len(test_f) // 20))  # Dynamic modification count
        
        if len(recent_improvements) > 5:
            avg_improvement = np.mean(recent_improvements[-5:])
            # Reduce modifications if progress is slow
            if avg_improvement < improvement_threshold * 0.1:
                num_modifications = max(1, num_modifications // 2)
        
        mod_indices = np.random.choice(len(test_f), num_modifications, replace=False)
        for idx in mod_indices:
            # Add small random change with bounded support
            change = np.random.normal(0, 0.05 * best_f[idx])
            test_f[idx] = max(0, test_f[idx] + change)
        
        # Evaluate and accept improvement
        test_c2 = compute_c2(test_f)
        improvement = test_c2 - best_c2
        
        if test_c2 > best_c2:
            best_c2 = test_c2
            best_f = test_f
            recent_improvements.append(improvement)
            current_patience = 0
        else:
            current_patience += 1
            recent_improvements.append(improvement)
        
        # Early stopping if no significant improvement for several iterations
        if current_patience >= patience:
            break

    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")