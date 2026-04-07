# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import time
from scipy.optimize import differential_evolution
import random

@njit
def compute_autoconvolution_manual(f_values):
    """
    Manual computation of autoconvolution for better performance with Numba
    """
    n = len(f_values)
    if n == 0:
        return np.array([])
    
    # Allocate result array for autoconvolution
    g = np.zeros(2 * n - 1)
    
    # Manual convolution computation
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    return g

@njit
def compute_autoconvolution_norms_fast(f_values):
    """
    Fast computation of autoconvolution norms with Numba optimization
    """
    # Convert to numpy array and ensure non-negative values
    f = np.array(f_values, dtype=np.float64)
    f = np.maximum(f, 0)  # Clip negative values to 0
    
    if len(f) == 0:
        return 0.0, 0.0, 0.0
    
    # Create step function on [-1/4, 1/4]
    step_width = 0.5 / len(f)
    
    # Compute autoconvolution manually
    g_full = compute_autoconvolution_manual(f)
    
    # Trim to match [-1/4, 1/4] interval
    half_len = len(f)
    g_center = len(g_full) // 2
    g_trimmed = g_full[g_center - half_len : g_center + half_len]
    
    # Compute norms
    # ||g||_2^2 using trapezoidal rule for piecewise linear integration
    g_abs = np.abs(g_trimmed)
    if len(g_abs) < 2:
        norm_2_squared = 0.0
    else:
        # Trapezoidal integration formula for piecewise linear segments
        # Each segment contributes (width/3)*(y1^2 + y1*y2 + y2^2)
        widths = np.full(len(g_abs)-1, step_width)
        y1 = g_abs[:-1]
        y2 = g_abs[1:]
        norm_2_squared = np.sum(widths * (y1**2 + y1*y2 + y2**2) / 3.0)
    
    # ||g||_1 = sum of absolute values divided by number of elements for normalization
    norm_1 = np.sum(np.abs(g_trimmed)) / (len(g_trimmed) + 1) if len(g_trimmed) > 0 else 1e-12
    
    # ||g||_∞ = max absolute value
    norm_inf = np.max(np.abs(g_trimmed)) if len(g_trimmed) > 0 else 1e-12
    
    return norm_2_squared, norm_1, norm_inf

def evaluate_c2_fast(f_values):
    """
    Fast evaluation of C2 with optimized norm computation
    """
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms_fast(f_values)
    
    # Prevent division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    c2 = norm_2_squared / (norm_1 * norm_inf)
    return c2

def sophisticated_initialization(n_steps):
    """
    Create initial population with sophisticated approaches:
    1. Alternating high/low pattern with randomization
    2. Multi-peak Gaussian pattern
    3. Smooth sinusoidal pattern
    4. Random distribution with mixed values
    """
    # Approach 1: Alternating pattern with randomness
    pattern1 = []
    for i in range(n_steps):
        base_val = 1.0 if i % 2 == 0 else 0.1
        # Add slight randomness
        noise = np.random.uniform(0.8, 1.2)
        pattern1.append(max(0.0, base_val * noise))
    
    # Approach 2: Multi-peak Gaussian pattern
    x = np.linspace(-1, 1, n_steps)
    peak1 = np.exp(-((x - 0.3)**2) / 0.1) * 1.5
    peak2 = np.exp(-((x + 0.3)**2) / 0.1) * 1.5
    peak3 = np.exp(-(x**2) / 0.2) * 0.8
    pattern2 = np.maximum(np.maximum(peak1, peak2), peak3)
    
    # Approach 3: Sinusoidal pattern
    pattern3 = 0.5 + 0.5 * np.sin(4 * np.pi * x)  # Smooth oscillation
    pattern3 = pattern3 * 1.5  # Scale up
    
    # Approach 4: Random with mixed high/low values
    pattern4 = []
    for i in range(n_steps):
        if np.random.random() < 0.3:  # 30% chance for high value
            pattern4.append(np.random.uniform(1.0, 2.0))
        else:
            pattern4.append(np.random.uniform(0.0, 0.5))
    
    # Return the best one based on initial evaluation
    patterns = [pattern1, pattern2.tolist(), pattern3.tolist(), pattern4]
    best_pattern = pattern4  # Default fallback
    best_score = -1.0
    
    for p in patterns:
        score = evaluate_c2_fast(p)
        if score > best_score:
            best_score = score
            best_pattern = p
    
    return best_pattern

def evolutionary_optimization(max_time=80):
    """
    Main evolutionary optimization routine with adaptive parameters and better strategies
    """
    start_time = time.time()
    
    # Parameters for optimization - dynamic sizing based on time available
    n_steps = np.random.randint(500, 3000)  # Variable length within reasonable bounds
    
    # Initialize with sophisticated approach
    initial_individual = sophisticated_initialization(n_steps)
    
    # Define bounds for each variable (non-negative)
    bounds = [(0.0, 10.0)] * n_steps
    
    # Set up problem with objective function - maximize C2 by minimizing negative C2
    def objective(x):
        return -evaluate_c2_fast(x)
    
    try:
        # Phase 1: Differential Evolution with adaptive population size
        popsize = min(50, max(10, n_steps // 50))
        maxiter = min(100, max(30, n_steps // 100))
        
        # Use a slightly smaller population for larger problems to save time
        if n_steps > 2000:
            popsize = max(20, n_steps // 100)
            maxiter = min(50, max(20, n_steps // 200))
        
        # Run optimization with better parameters
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            disp=False,
            atol=1e-6,
            rtol=1e-6
        )
        
        # Check for early termination
        if time.time() - start_time > max_time * 0.9:
            return result.x.tolist()
            
        # Return the best solution found
        return result.x.tolist()
        
    except Exception as e:
        # Fallback to simpler optimization
        try:
            # Start with the best initial solution
            best_f = initial_individual
            best_c2 = evaluate_c2_fast(best_f)
            
            # Perform a few rounds of hill climbing
            for _ in range(50):
                if time.time() - start_time > max_time * 0.9:
                    break
                    
                # Create neighbor by slightly perturbing
                trial = []
                for val in best_f:
                    # Perturb by up to 20%
                    perturbation = np.random.uniform(0.8, 1.2)
                    new_val = max(0.0, val * perturbation)
                    trial.append(new_val)
                
                c2 = evaluate_c2_fast(trial)
                if c2 > best_c2:
                    best_c2 = c2
                    best_f = trial
            
            return best_f
            
        except Exception:
            # Final fallback
            return sophisticated_initialization(n_steps)

def construct_function() -> list[float]:
    """
    Main function to construct optimized step-function for high C2 value
    """
    # Allow some time budget for computation
    start_time = time.time()
    
    try:
        # Run evolutionary optimization
        f_values = evolutionary_optimization(80)
        
        # Final validation and cleanup
        f_values = np.array(f_values)
        f_values = np.maximum(f_values, 0)  # Ensure non-negative
        f_values = f_values.tolist()
        
        # If too long, truncate to reasonable size
        if len(f_values) > 5000:
            f_values = f_values[:5000]
        
        # Ensure minimum length for proper computation
        if len(f_values) < 100:
            f_values = f_values + [1.0] * (100 - len(f_values))
        
        return f_values
        
    except Exception as e:
        # Return a fallback solution in case of any failure
        n_steps = 500
        return [1.0] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
