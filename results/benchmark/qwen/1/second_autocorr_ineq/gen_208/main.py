# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import List
import numba
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# JIT compiled functions for performance
@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
    """
    Fast computation of autoconvolution norms using Numba JIT compilation
    """
    n = len(f_values)
    
    # Compute autoconvolution g = f * f using discrete convolution
    # The resulting g will have length 2*n - 1 where n is the length of f
    g_length = 2 * n - 1
    g = np.zeros(g_length)
    
    # Manual convolution loop for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Compute the norms
    # ||g||₂² = sum(g[i]²) using proper piecewise integration
    norm_g_2_squared = 0.0
    
    # For piecewise linear integration, we use trapezoidal-like approach:
    # for consecutive pairs of points (y1, y2) with unit spacing:
    # integral of y^2 ≈ (1/3)(y1^2 + y1*y2 + y2^2)
    for i in range(g_length - 1):
        y1 = g[i]
        y2 = g[i + 1]
        norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0
    
    # ||g||₁ = sum(|g[i]|)
    norm_g_1 = 0.0
    for i in range(g_length):
        norm_g_1 += abs(g[i])
    
    # ||g||∞ = max(|g[i]|)
    norm_g_inf = 0.0
    for i in range(g_length):
        abs_g = abs(g[i])
        if abs_g > norm_g_inf:
            norm_g_inf = abs_g
    
    return norm_g_2_squared, norm_g_1, norm_g_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the norms ||g||₂², ||g||₁, and ||g||∞ for the autoconvolution g = f*f
    Using proper piecewise linear integration for ||g||₂² as specified in requirements
    """
    f = np.array(f_values)
    
    # Use the fast JIT-compiled version
    norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms_fast(f)
    
    return norm_g_2_squared, norm_g_1, norm_g_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function
    """
    try:
        norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_g_1 <= 1e-12 or norm_g_inf <= 1e-12:
            return 0.0

        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception as e:
        # Fallback in case of any numerical issues
        return 0.0

def generate_multi_scale_initialization(n_steps: int, coarse_size: int = 100) -> List[float]:
    """
    Generate initial configuration using multi-scale sampling approach
    """
    # Start with coarse resolution to understand the landscape
    coarse_f = np.zeros(coarse_size)
    
    # Create a balanced alternating pattern with some randomness
    segment_size = max(1, coarse_size // 12)
    for i in range(0, coarse_size, segment_size):
        end_idx = min(i + segment_size, coarse_size)
        if (i // segment_size) % 2 == 0:
            # High region with variation
            coarse_f[i:end_idx] = 0.7 + np.random.random(end_idx - i) * 0.25
        else:
            # Low region with variation
            coarse_f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.15

    # Interpolate to full resolution
    coarse_x = np.linspace(-1, 1, coarse_size)
    fine_x = np.linspace(-1, 1, n_steps)
    
    # Use spline interpolation for smooth transition
    from scipy.interpolate import interp1d
    interpolate_func = interp1d(coarse_x, coarse_f, kind='linear', fill_value='extrapolate')
    f = interpolate_func(fine_x)
    
    # Add Gaussian-shaped smoothing to reduce artifacts and add structure
    gaussian_width = 0.2 + np.random.random() * 0.15
    gaussian = np.exp(-0.5 * (fine_x / gaussian_width)**2)
    f = f * gaussian * 0.6 + gaussian * 0.4
    
    # Add some additional structure
    n_peaks = 2 + np.random.randint(0, 3)
    for _ in range(n_peaks):
        peak_pos = np.random.randint(0, n_steps)
        peak_width = max(1, n_steps // 15 + np.random.randint(-2, 3))
        start = max(0, peak_pos - peak_width // 2)
        end = min(n_steps, peak_pos + peak_width // 2)
        peak_height = 0.3 + np.random.random() * 0.4
        f[start:end] = np.maximum(f[start:end], peak_height * 0.7 + np.random.random(end - start) * 0.3)
    
    # Ensure non-negativity and normalize
    f = np.clip(f, 0, None)
    if np.sum(f) > 0:
        f = f / np.sum(f)
    
    return f.tolist()

def adaptive_gradient_optimization(initial_f: List[float], max_iter: int = 30) -> List[float]:
    """
    Perform adaptive gradient-based optimization with momentum and learning rate scheduling
    """
    f_start = np.array(initial_f)
    n_steps = len(f_start)
    
    # Use scipy's L-BFGS-B as it provides good control over constraints and is robust
    def objective(x):
        return -evaluate_c2(x.tolist())
    
    def gradient(x):
        # Convert to numpy array for gradient computation
        x_array = np.array(x)
        # Simple finite difference approximation for gradient
        eps = 1e-6
        grad_vals = np.zeros_like(x_array)
        
        for i in range(len(x_array)):
            x_plus = x_array.copy()
            x_minus = x_array.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad_vals[i] = (evaluate_c2(x_plus.tolist()) - evaluate_c2(x_minus.tolist())) / (2 * eps)
        
        return -grad_vals  # Negative because we're minimizing negative C2
    
    try:
        # Define bounds
        bounds = [(0, 1.0) for _ in range(n_steps)]
        
        # Run L-BFGS-B optimization
        result = minimize(
            objective,
            f_start,
            method='L-BFGS-B',
            bounds=bounds,
            jac=gradient,
            options={'maxiter': max_iter, 'gtol': 1e-6},
            callback=None
        )
        
        if result.success:
            optimized_f = np.maximum(result.x, 0)
            if np.sum(optimized_f) > 0:
                optimized_f = optimized_f / np.sum(optimized_f)
            return optimized_f.tolist()
    except Exception as e:
        print(f"Gradient optimization failed: {e}")
        
    return initial_f

def advanced_refinement_strategy(n_steps: int) -> List[float]:
    """
    Apply a multi-phase refinement strategy to maximize C2
    """
    # Phase 1: Multi-scale initialization with good structural properties
    initial_f = generate_multi_scale_initialization(n_steps)
    
    # Phase 2: Gradient-based refinement
    refined_f = adaptive_gradient_optimization(initial_f, max_iter=25)
    
    # Phase 3: Post-hoc polishing with local search
    try:
        # Try a few local perturbations to see if we can improve further
        best_f = refined_f.copy()
        best_c2 = evaluate_c2(best_f)
        
        # Small random perturbations
        for _ in range(15):
            perturbed_f = np.array(best_f) + np.random.normal(0, 0.01, n_steps)
            perturbed_f = np.clip(perturbed_f, 0, None)
            if np.sum(perturbed_f) > 0:
                perturbed_f = perturbed_f / np.sum(perturbed_f)
                c2_new = evaluate_c2(perturbed_f.tolist())
                if c2_new > best_c2:
                    best_c2 = c2_new
                    best_f = perturbed_f.tolist()
        
        refined_f = best_f
        
    except Exception as e:
        pass  # If refinement fails, continue with current best
    
    return refined_f

def evolutionary_optimization() -> List[float]:
    """
    Use evolutionary algorithm to optimize step function with enhanced parameters
    """
    n_steps = 500  # Reasonable size for exploration
    
    # Define bounds for each parameter (step height)
    bounds = [(0, 1.0) for _ in range(n_steps)]
    
    def objective(x):
        # Return negative because we want to maximize C2
        return -evaluate_c2(x.tolist())
    
    # Use differential evolution for global optimization with improved parameters
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=25,  # Reduced iterations for faster execution
            popsize=15,   # Slightly larger population for better exploration
            seed=42,
            disp=False
        )
        
        if result.success:
            optimized_f = np.maximum(result.x, 0)
            # Normalize to ensure good scaling
            if np.sum(optimized_f) > 0:
                optimized_f = optimized_f / np.sum(optimized_f)
            
            # Post-process with local optimization to refine the solution
            try:
                # Local refinement with L-BFGS
                def local_objective(f_vals):
                    return -evaluate_c2(f_vals.tolist())
                
                bounds_local = [(0, 1.0) for _ in range(n_steps)]
                local_result = minimize(
                    local_objective,
                    optimized_f,
                    method='L-BFGS-B',
                    bounds=bounds_local,
                    options={'maxiter': 15}
                )
                
                if local_result.success:
                    refined_f = np.maximum(local_result.x, 0)
                    if np.sum(refined_f) > 0:
                        refined_f = refined_f / np.sum(refined_f)
                    return refined_f.tolist()
            except:
                pass
                
            return optimized_f.tolist()
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Return default if optimization fails
    return [1.0/n_steps] * n_steps

def sophisticated_initialization() -> List[float]:
    """
    Generate a sophisticated initial configuration based on mathematical intuition
    """
    n_steps = 500
    
    # Create a step function that tries to balance flatness with sufficient mass
    # Based on mathematical insights: we want to create a function that when convolved 
    # produces a relatively flat profile but with enough energy to achieve high C2
    
    # Start with alternating high/low regions with some randomness
    f = np.zeros(n_steps)
    
    # First create a base alternating pattern with some randomness
    segment_size = max(1, n_steps // 10)
    for i in range(0, n_steps, segment_size):
        end_idx = min(i + segment_size, n_steps)
        if (i // segment_size) % 2 == 0:
            # High region
            f[i:end_idx] = 0.8 + np.random.random(end_idx - i) * 0.2
        else:
            # Low region
            f[i:end_idx] = 0.1 + np.random.random(end_idx - i) * 0.1
    
    # Add Gaussian-based smoothing for more natural transitions
    x = np.linspace(-1, 1, n_steps)
    gaussian = np.exp(-0.5 * (x / 0.25)**2)
    f = f * gaussian * 0.6 + gaussian * 0.4
    
    # Add some noise to break symmetry
    noise = np.random.normal(0, 0.02, n_steps)
    f = f + noise
    
    # Ensure non-negativity
    f = np.clip(f, 0, None)
    
    # Normalize
    if np.sum(f) > 0:
        f = f / np.sum(f)
    
    return f.tolist()

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value using improved multi-stage optimization
    """
    try:
        # Try sophisticated initialization first
        initial_f = sophisticated_initialization()
        c2_initial = evaluate_c2(initial_f)
        
        # Try advanced multi-scale refinement
        multi_scale_f = advanced_refinement_strategy(500)
        c2_multi_scale = evaluate_c2(multi_scale_f)
        
        # Then run evolutionary optimization
        evolutionary_f = evolutionary_optimization()
        c2_evolutionary = evaluate_c2(evolutionary_f)
        
        # Compare results and return the best
        best_c2 = max(c2_initial, c2_multi_scale, c2_evolutionary)
        
        if best_c2 == c2_multi_scale:
            return multi_scale_f
        elif best_c2 == c2_evolutionary:
            return evolutionary_f
        else:
            return initial_f
            
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple initialization
        n_steps = 500
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")