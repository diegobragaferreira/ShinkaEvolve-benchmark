# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
import math

@njit
def compute_convolution_norms_numba(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    JIT compiled version for improved performance.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using piecewise linear integration
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Compute autoconvolution - JIT compiled loop
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j]

    # Compute norms using trapezoidal-like integration
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

@njit
def construct_initial_pattern(n_steps):
    """Construct an intelligent initial pattern for better convergence"""
    # Use a combination of geometric and sinusoidal patterns
    # to create a function that should promote flatter convolution
    
    f_values = np.zeros(n_steps)
    
    # Create a bell-shaped pattern centered in the middle
    center = n_steps // 2
    width = n_steps // 4
    
    # Gaussian-like envelope to create a smooth, flat convolution profile
    for i in range(n_steps):
        distance_from_center = abs(i - center)
        if distance_from_center < width:
            # Smooth bell shape
            f_values[i] = np.exp(-0.5 * (distance_from_center / (width/2))**2)
        else:
            # Slightly decayed tails
            f_values[i] = 0.1 * np.exp(-0.5 * (distance_from_center / (width/2))**2)
    
    # Normalize to avoid overly large values
    total = np.sum(f_values)
    if total > 0:
        f_values = f_values / total * 100
    
    return f_values

def compute_convolution_norms(f_values, domain_length=0.5):
    """Wrapper function for norm computation"""
    return compute_convolution_norms_numba(f_values, domain_length)

def compute_c2(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    return compute_c2_numba(f_values)

def adaptive_local_search(initial_f, max_iterations=50):
    """Perform adaptive local search to improve the solution"""
    best_f = initial_f.copy()
    best_c2 = compute_c2(best_f)
    
    # Track recent improvements
    recent_improvements = []
    patience = 0
    max_patience = 15
    
    for iteration in range(max_iterations):
        test_f = best_f.copy()
        
        # Determine number of modifications based on progress
        if len(recent_improvements) < 5:
            num_modifications = max(1, len(test_f) // 20)
        else:
            avg_improvement = np.mean(recent_improvements[-5:])
            if avg_improvement < 1e-6:
                num_modifications = max(1, len(test_f) // 40)
            else:
                num_modifications = max(1, len(test_f) // 10)
        
        # Apply random modifications
        mod_indices = np.random.choice(len(test_f), num_modifications, replace=False)
        for idx in mod_indices:
            # Small random change with bounded support
            change_factor = np.random.normal(0, 0.1)
            new_value = best_f[idx] * (1 + change_factor)
            test_f[idx] = max(0, new_value)  # Ensure non-negativity
        
        # Evaluate and accept improvement
        test_c2 = compute_c2(test_f)
        improvement = test_c2 - best_c2
        
        if test_c2 > best_c2:
            best_c2 = test_c2
            best_f = test_f
            recent_improvements.append(improvement)
            patience = 0
        else:
            recent_improvements.append(improvement)
            patience += 1
            
        # Early stopping if no improvement for too long
        if patience >= max_patience:
            break
    
    return best_f, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary approach."""
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Use a moderate number of steps for balance between resolution and speed
    n_steps = 2000
    
    # Construct initial pattern
    f_values = construct_initial_pattern(n_steps).tolist()
    
    # Try optimization with differential evolution
    try:
        # Define bounds for each variable
        bounds = [(0, 100) for _ in range(n_steps)]
        
        # Run differential evolution with limited iterations
        result = differential_evolution(
            lambda x: -compute_c2(x),  # Minimize negative C2
            bounds,
            maxiter=20,  # Limited iterations due to time constraints
            popsize=10,  # Smaller population for faster execution
            seed=42,
            disp=False
        )
        
        # Use the result if it improved upon our initial function
        if result.success and -result.fun > compute_c2(f_values):
            f_values = result.x.tolist()
            
    except Exception:
        # If differential evolution fails, proceed with local search
        pass
    
    # Perform adaptive local search to fine-tune the solution
    refined_f, final_c2 = adaptive_local_search(f_values, max_iterations=100)
    
    return refined_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")