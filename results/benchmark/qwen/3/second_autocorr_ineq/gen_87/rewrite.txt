# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
from typing import List, Tuple
import math
import time

@njit
def compute_convolution_norms_optimized(f_values: np.ndarray) -> Tuple[float, float, float]:
    """
    Optimized computation of autoconvolution norms using specialized piecewise integration
    tailored for step function convolution.
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Precompute step width for normalized domain [-1/4, 1/4]
    dx = 0.5 / n
    
    # Compute autoconvolution g = f * f using optimized nested loops
    # Result has size 2*n-1 (due to convolution)
    g_size = 2 * n - 1
    g = np.zeros(g_size)
    
    # Optimized convolution computation with proper indexing
    for i in range(n):
        for j in range(n):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j] * dx
    
    # Compute norms using specialized piecewise integration
    # For ||g||₂²: use trapezoidal rule for g² with proper weighting
    g2_sq = 0.0
    for i in range(g_size - 1):
        # Use 1/3 * (y₁² + y₁y₂ + y₂²) * dx for each segment
        y1, y2 = g[i], g[i+1]
        g2_sq += (y1*y1 + y1*y2 + y2*y2) * dx / 3.0
    
    # ||g||₁ = sum(|g_i| * dx)  
    g1 = np.sum(np.abs(g)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

@njit
def compute_c2_optimized(f_values: np.ndarray) -> float:
    """Optimized C2 computation using JIT compilation"""
    g2_sq, g1, ginf = compute_convolution_norms_optimized(f_values)
    
    if g1 <= 1e-15 or ginf <= 1e-15:
        return 0.0
    
    return g2_sq / (g1 * ginf)

def construct_geometric_initial_function(n_steps: int) -> List[float]:
    """
    Create an initial function using geometric progression that tends to produce
    favorable autoconvolution properties. This exploits mathematical insights
    about step function behavior rather than random search.
    """
    # Use geometric progression that starts high and tapers off
    # This pattern often produces flatter autoconvolutions
    base_height = 1.0
    geometric_factor = 0.95
    
    # Create geometrically decreasing sequence
    f_values = []
    for i in range(n_steps):
        # Use geometric decay with some oscillation for better spread
        factor = geometric_factor ** (i / max(1, n_steps / 3))
        height = base_height * factor
        
        # Add some variation to avoid perfect symmetry
        if i % 7 == 0:
            height *= 1.2
        elif i % 5 == 0:
            height *= 0.8
            
        f_values.append(max(0.0, height))
    
    # Normalize to reasonable scale
    total_area = sum(f_values) * (0.5 / n_steps)
    if total_area > 0:
        f_values = [x / total_area * 2.0 for x in f_values]
        
    return f_values

def adaptive_gradient_refinement(initial_f: List[float], max_iterations: int = 100) -> List[float]:
    """
    Apply adaptive gradient-based refinement to improve the initial function.
    Uses finite differences to estimate gradients and improve C2.
    """
    f_current = np.array(initial_f, dtype=np.float64)
    n = len(f_current)
    
    if n == 0:
        return initial_f
    
    # Learning rate that adapts based on progress
    learning_rate_base = 0.05
    tolerance = 1e-6
    patience_counter = 0
    best_score = compute_c2_optimized(f_current)
    best_f = f_current.copy()
    
    # Gradient descent with adaptive learning rate
    for iteration in range(max_iterations):
        current_score = compute_c2_optimized(f_current)
        
        # Compute finite difference gradients
        grad = np.zeros(n)
        epsilon = 1e-4
        
        for i in range(n):
            # Perturb dimension i
            f_plus = f_current.copy()
            f_minus = f_current.copy()
            
            f_plus[i] = max(0.0, f_current[i] + epsilon)
            f_minus[i] = max(0.0, f_current[i] - epsilon)
            
            score_plus = compute_c2_optimized(f_plus)
            score_minus = compute_c2_optimized(f_minus)
            
            grad[i] = (score_plus - score_minus) / (2 * epsilon)
        
        # Update with gradient
        # Adaptive learning rate based on gradient magnitude
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            adaptive_lr = learning_rate_base / (1.0 + grad_norm)
        else:
            adaptive_lr = learning_rate_base
            
        # Apply update
        f_new = f_current - adaptive_lr * grad
        
        # Ensure non-negativity
        f_new = np.maximum(f_new, 0.0)
        
        new_score = compute_c2_optimized(f_new)
        
        # Accept improvement
        if new_score > current_score:
            f_current = f_new
            if new_score > best_score:
                best_score = new_score
                best_f = f_current.copy()
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            patience_counter += 1
            
        # Early stopping if no improvement for several iterations
        if patience_counter > 15:
            break
    
    return best_f.tolist()

def dynamic_resolution_optimization(initial_f: List[float]) -> List[float]:
    """
    Dynamically adjust the resolution of the step function to optimize C2.
    Starts with moderate resolution and can increase if needed.
    """
    f_best = initial_f.copy()
    best_c2 = compute_c2_optimized(np.array(f_best))
    
    # Try different resolutions around the initial one
    base_resolutions = [500, 750, 1000, 1250, 1500]
    
    for res in base_resolutions:
        # Create a smoother version of the function at higher resolution
        f_high_res = []
        n_orig = len(f_best)
        
        # Interpolate to higher resolution
        for i in range(res):
            # Map index to original domain
            orig_idx = (i * (n_orig - 1)) / (res - 1) if res > 1 else 0
            left_idx = int(math.floor(orig_idx))
            right_idx = min(left_idx + 1, n_orig - 1)
            frac = orig_idx - left_idx
            
            # Linear interpolation
            if left_idx < n_orig:
                val = f_best[left_idx] * (1 - frac) + f_best[right_idx] * frac
                f_high_res.append(val)
            else:
                f_high_res.append(f_best[-1] if f_best else 0)
        
        # Refine at higher resolution
        refined = adaptive_gradient_refinement(f_high_res, max_iterations=30)
        refined_c2 = compute_c2_optimized(np.array(refined))
        
        if refined_c2 > best_c2:
            best_c2 = refined_c2
            f_best = refined
    
    return f_best

def construct_function() -> List[float]:
    """
    Main function that constructs a high-C2 step function using the novel approach.
    """
    # Set deterministic seeds
    np.random.seed(42)
    import random
    random.seed(42)
    
    # Start with geometric initialization - this is the key innovation
    n_steps = 1000  # Fixed resolution for reproducibility and efficiency
    initial_f = construct_geometric_initial_function(n_steps)
    
    # Apply gradient-based refinement
    refined_f = adaptive_gradient_refinement(initial_f, max_iterations=80)
    
    # Apply dynamic resolution optimization
    final_f = dynamic_resolution_optimization(refined_f)
    
    # Final local optimization using gradient refinement
    result = adaptive_gradient_refinement(final_f, max_iterations=20)
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")