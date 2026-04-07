# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def compute_convolution_norms(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    Uses piecewise linear integration with trapezoidal approach for ||g||₂².
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using discrete computation
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Compute autoconvolution using direct convolution sum
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j] * dx

    # Compute norms using piecewise linear integration approach
    # For ||g||₂² using trapezoidal-like formula: (dx/3)(g₀² + g₀g₁ + g₁²)
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

def compute_c2(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    g2_sq, g1, ginf = compute_convolution_norms(f_values)

    if g1 == 0 or ginf == 0:
        return 0.0

    return g2_sq / (g1 * ginf)

def gaussian_bump(x, center, width, height):
    """Generate a Gaussian-shaped bump"""
    return height * np.exp(-0.5 * ((x - center) / width)**2)

def construct_geometric_initial_function(n_steps):
    """Construct initial function using geometric principles"""
    # Create a smooth, bell-shaped function that encourages flat convolution profiles
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Generate multiple overlapping Gaussian bumps with different parameters
    # to create a function that promotes uniformity in convolution
    f_values = np.zeros(n_steps)
    
    # Base Gaussian bump at center
    f_values += gaussian_bump(x, 0.0, 0.1, 1.0)
    
    # Additional bumps to promote flat convolution behavior
    f_values += 0.5 * gaussian_bump(x, -0.1, 0.05, 0.8)
    f_values += 0.3 * gaussian_bump(x, 0.1, 0.07, 0.6)
    
    # Ensure non-negativity and normalize
    f_values = np.maximum(f_values, 0)
    
    # Normalize to reasonable scale
    total_area = np.sum(f_values) * (0.5 / n_steps)
    if total_area > 0:
        f_values = f_values / total_area
    
    return f_values.tolist()

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using gradient-free optimization"""
    # Set up optimization parameters
    n_steps = 200
    
    # Create initial function using geometric construction
    initial_f = construct_geometric_initial_function(n_steps)
    
    # Convert to numpy array for optimization
    x0 = np.array(initial_f)
    
    # Define bounds (non-negative)
    bounds = [(0, None) for _ in range(n_steps)]
    
    # Optimization using L-BFGS-B with multiple restarts
    best_f = x0.copy()
    best_c2 = compute_c2(best_f.tolist())
    
    # Multiple restarts with different starting points
    for restart in range(5):
        # Slightly perturb original solution for different restart
        if restart > 0:
            np.random.seed(restart)
            perturbed_x0 = x0 * (1 + np.random.normal(0, 0.1, n_steps))
            perturbed_x0 = np.maximum(perturbed_x0, 0)  # Ensure non-negativity
            current_x0 = perturbed_x0
        else:
            current_x0 = x0
            
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                lambda x: -compute_c2(x.tolist()),  # Minimize negative C2 (maximize C2)
                current_x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6},
                callback=None
            )
            
            if result.success:
                optimized_f = np.maximum(result.x, 0)  # Ensure non-negativity
                optimized_c2 = compute_c2(optimized_f.tolist())
                
                if optimized_c2 > best_c2:
                    best_c2 = optimized_c2
                    best_f = optimized_f
                    
        except Exception as e:
            # If optimization fails, continue with existing best
            continue
    
    # Final local refinement using coordinate-wise optimization
    refined_f = best_f.copy()
    old_c2 = best_c2
    
    # Try coordinate-wise improvements
    for coord_iter in range(20):
        improved = False
        for i in range(len(refined_f)):
            # Try small perturbations
            original_value = refined_f[i]
            step_sizes = [0.01, 0.05, 0.1]
            
            for step in step_sizes:
                # Try increasing and decreasing
                for direction in [1, -1]:
                    test_f = refined_f.copy()
                    new_val = original_value + direction * step
                    test_f[i] = max(0, new_val)
                    
                    new_c2 = compute_c2(test_f.tolist())
                    if new_c2 > old_c2:
                        refined_f = test_f
                        old_c2 = new_c2
                        improved = True
                        
        if not improved:
            break
            
    return refined_f.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")