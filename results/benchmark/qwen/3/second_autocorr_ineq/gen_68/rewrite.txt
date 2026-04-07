# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import time
from numba import jit, prange

# Set random seeds for reproducibility
np.random.seed(42)
np.random.seed(42)

@jit(nopython=True, parallel=True)
def compute_autoconvolution_kernel(f_vals):
    """
    Precompute autoconvolution kernel and perform convolution efficiently
    """
    n = len(f_vals)
    # Autoconvolution size is 2*n - 1
    g_size = 2 * n - 1
    g_vals = np.zeros(g_size, dtype=np.float64)

    # Compute convolution directly with numba optimization
    for i in prange(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_size:
                g_vals[idx] += f_vals[i] * f_vals[j]

    return g_vals

@jit(nopython=True)
def compute_trapezoidal_norms(g_vals, dx):
    """
    Compute trapezoidal norms efficiently
    """
    n = len(g_vals)

    # For L2 norm squared using trapezoidal rule
    l2_norm_sq = 0.0
    if n >= 2:
        # Trapezoidal rule: sum of (y[i]^2 + y[i+1]^2)/2 * dx
        for i in range(n-1):
            l2_norm_sq += (g_vals[i] * g_vals[i] + g_vals[i+1] * g_vals[i+1]) * dx / 2.0
    elif n == 1:
        l2_norm_sq = g_vals[0] * g_vals[0] * dx

    # For L1 norm using trapezoidal rule (average of adjacent heights * dx)
    l1_norm = 0.0
    if n >= 2:
        for i in range(n-1):
            l1_norm += (abs(g_vals[i]) + abs(g_vals[i+1])) * dx / 2.0
    elif n == 1:
        l1_norm = abs(g_vals[0]) * dx

    # Infinity norm
    linf_norm = 0.0
    for i in range(n):
        val = abs(g_vals[i])
        if val > linf_norm:
            linf_norm = val

    return l1_norm, l2_norm_sq, linf_norm

def compute_autoconvolution_norms(f_values: List[float]):
    """
    Compute the three norms needed for C₂ calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    if not f_values:
        return 0.0, 0.0, 0.0

    # Create step function on [-1/4, 1/4]
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    dx = 0.5 / n_steps  # Step size

    # Create piecewise constant function from step heights
    f = np.array(f_values, dtype=np.float64)

    # Compute autoconvolution g = f * f using numba optimized version
    g = compute_autoconvolution_kernel(f)

    # Adjust indices for the correct domain
    # Result has length 2*n_steps - 1
    g_len = len(g)

    # Extract the central region corresponding to [-1/4, 1/4]
    # This takes the middle n_steps elements of the full convolution
    central_start = (g_len - n_steps) // 2
    central_end = central_start + n_steps
    g_centered = g[central_start:central_end]

    # Compute norms using trapezoidal integration
    g_abs = np.abs(g_centered)

    # Compute norms using trapezoidal rules for more accurate integration
    norm_1, norm_2_sq, norm_inf = compute_trapezoidal_norms(g_abs, dx)

    return norm_2_sq, norm_1, norm_inf

def evaluate_c2_with_penalty(f_values: List[float], penalty_weight: float = 1e4):
    """
    Evaluate C₂ with penalty for negative values
    """
    try:
        # Convert individual to list of floats and ensure non-negative
        f_values = [max(0.0, float(x)) for x in f_values]
        
        # Compute the norms
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return 0.0

        # Calculate C₂
        c2 = norm_2_sq / (norm_1 * norm_inf)
        
        # Add penalty for any negative values (though they should be clipped already)
        penalty = 0.0
        for val in f_values:
            if val < 0:
                penalty += penalty_weight * abs(val)
        
        return max(0.0, c2 - penalty)
    except Exception:
        return 0.0

def construct_function() -> List[float]:
    """
    Function to construct step-function with high C₂ value using adaptive gradient-based optimization.
    """
    # Time limit
    TIME_LIMIT = 85  # seconds
    start_time = time.time()
    
    # Initial parameters for optimization
    max_iterations = 2000
    initial_lr = 0.1
    min_lr = 1e-6
    patience = 20
    patience_counter = 0
    best_score = 0.0
    best_solution = None

    # Generate initial pattern based on mathematical intuition
    # Start with alternating high/low pattern and geometric progression
    n_steps = 500  # Fixed for consistent comparison
    initial_pattern = []
    for i in range(n_steps):
        if i % 3 == 0:
            initial_pattern.append(1.0)
        elif i % 3 == 1:
            initial_pattern.append(0.5)
        else:  # i % 3 == 2
            initial_pattern.append(0.1)
    
    # Add some random variation to break symmetry
    for i in range(len(initial_pattern)):
        if np.random.random() < 0.1:
            initial_pattern[i] *= np.random.uniform(0.8, 1.2)
    
    # Ensure non-negative
    current_solution = [max(0.0, x) for x in initial_pattern]
    
    # Adaptive learning rate
    lr = initial_lr
    last_improvement = 0
    
    # Optimization loop - run for maximum iterations or until time limit
    for iteration in range(max_iterations):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Evaluate current solution
        current_score = evaluate_c2_with_penalty(current_solution)
        
        if current_score > best_score:
            best_score = current_score
            best_solution = current_solution.copy()
            patience_counter = 0
            last_improvement = iteration
        else:
            patience_counter += 1
            
        # If no improvement for several iterations, reduce learning rate
        if patience_counter >= patience:
            lr = max(min_lr, lr * 0.5)
            patience_counter = 0
            
        # Simple gradient estimation using finite differences
        epsilon = 1e-4
        gradient = []
        
        for i in range(len(current_solution)):
            # Perturb dimension i
            perturbed_plus = current_solution.copy()
            perturbed_minus = current_solution.copy()
            
            perturbed_plus[i] = max(0.0, current_solution[i] + epsilon)
            perturbed_minus[i] = max(0.0, current_solution[i] - epsilon)
            
            # Estimate gradient component
            score_plus = evaluate_c2_with_penalty(perturbed_plus)
            score_minus = evaluate_c2_with_penalty(perturbed_minus)
            
            grad_i = (score_plus - score_minus) / (2 * epsilon)
            gradient.append(grad_i)
        
        # Update solution with adaptive gradient descent
        updated_solution = []
        for i in range(len(current_solution)):
            # Apply gradient with learning rate
            new_val = current_solution[i] - lr * gradient[i]
            updated_solution.append(max(0.0, new_val))
            
        current_solution = updated_solution
        
        # Occasionally add some noise to escape local minima
        if iteration % 50 == 0 and iteration > 0:
            for i in range(len(current_solution)):
                if np.random.random() < 0.1:
                    current_solution[i] *= np.random.uniform(0.9, 1.1)
    
    # Return best solution found
    if best_solution is not None:
        return [float(x) for x in best_solution]
    
    # Fallback: return initial pattern
    return [float(x) for x in initial_pattern]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")