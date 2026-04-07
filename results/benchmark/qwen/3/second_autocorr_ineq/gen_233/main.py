# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import time
import numba
from numba import jit
import random
from typing import List, Tuple

# Constants
DOMAIN = [-0.25, 0.25]
N_MIN, N_MAX = 100, 2000
MAX_TIME_SECONDS = 85

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using numba-optimized approach"""
    n = len(f_vals)
    g = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    return g

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute norms efficiently using numba"""
    n = len(g_vals)
    l2_sq = 0.0
    l1 = 0.0
    l_inf = 0.0
    for i in range(n):
        val = g_vals[i]
        l2_sq += val * val
        l1 += abs(val)
        if abs(val) > l_inf:
            l_inf = abs(val)
    return l2_sq, l1, l_inf

def compute_c2(f_vals):
    """Compute C2 value for given function values"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)
        
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)
        
        # Compute norms
        l2_sq, l1, l_inf = compute_norms_numba(g_vals)
        
        # Avoid division by zero
        if l1 <= 1e-12 or l_inf <= 1e-12:
            return 0.0
        
        # Compute C2
        c2 = l2_sq / (l1 * l_inf)
        return c2
    except Exception as e:
        return 0.0

def generate_convolution_optimized_initial_function(n):
    """
    Generate initial function that's specifically designed to produce 
    favorable convolution properties for maximizing C2.
    """
    # Strategy: Create a function that promotes uniform convolution behavior
    # by combining multiple components that contribute to flat, well-distributed g
    
    f_vals = np.zeros(n)
    
    # 1. Create base smooth structure with controlled variation
    x = np.linspace(-0.25, 0.25, n)
    
    # Multiple frequency components that blend well
    base = np.exp(-0.5 * (x / 0.1) ** 2)  # Main Gaussian peak
    mod1 = 0.3 * np.exp(-0.5 * ((x - 0.1) / 0.08) ** 2)  # Shifted peak
    mod2 = 0.2 * np.exp(-0.5 * ((x + 0.1) / 0.08) ** 2)  # Opposite peak
    
    # Combine with some sinusoidal modulation for complexity
    wave = 0.1 * np.sin(20 * np.pi * x) + 0.1 * np.cos(15 * np.pi * x)
    
    # Construct final pattern
    combined = base + mod1 + mod2 + wave + 0.1  # Add offset to avoid zeros
    
    # Ensure all values are positive and normalize
    combined = np.maximum(combined, 0)
    
    # Normalize to reasonable total area
    total = np.sum(combined)
    if total > 0:
        combined = combined / total * 5.0
    
    return combined.tolist()

def generate_balanced_initial_function(n):
    """
    Generate a balanced function with structured peaks that avoid extreme
    convolution spikes while promoting good C2 values.
    """
    f_vals = np.zeros(n)
    
    # Create multiple smaller peaks to avoid single large convolution peaks
    num_peaks = max(3, n // 50)
    
    for peak_idx in range(num_peaks):
        # Distribute peaks evenly but add some randomness
        position = int((peak_idx + 1) * (n / (num_peaks + 1)))
        height = 1.0 + random.random() * 2.0
        spread = max(1, n // 40)
        
        # Apply triangular kernel to each peak
        for i in range(max(0, position - spread), min(n, position + spread + 1)):
            distance = abs(i - position)
            kernel = max(0, 1 - distance / spread)
            f_vals[i] += height * kernel
    
    # Add some additional smoothing
    smoothed = np.zeros(n)
    for i in range(n):
        if i == 0:
            smoothed[i] = f_vals[i]
        elif i == n - 1:
            smoothed[i] = f_vals[i]
        else:
            smoothed[i] = 0.3 * f_vals[i-1] + 0.4 * f_vals[i] + 0.3 * f_vals[i+1]
    
    # Ensure non-negativity and normalize
    smoothed = np.maximum(smoothed, 0)
    total = np.sum(smoothed)
    if total > 0:
        smoothed = smoothed / total * 5.0
    
    return smoothed.tolist()

def generate_adaptive_initial_function(n):
    """
    Generate initial function with adaptive structure based on problem dimensionality
    """
    if n < 200:
        # For small dimensions, use a more concentrated structure
        return generate_balanced_initial_function(n)
    else:
        # For larger dimensions, use convolution-optimized structure
        return generate_convolution_optimized_initial_function(n)

def adaptive_local_search(f_vals, max_iterations=50):
    """Perform adaptive coordinate-wise local search to refine the solution"""
    current_f_vals = f_vals.copy()
    current_c2 = compute_c2(current_f_vals)
    
    # Different step sizes for different phases
    step_phases = [
        (0.1, 10),   # Large steps initially
        (0.05, 20),  # Medium steps
        (0.01, 20)   # Fine tuning
    ]
    
    iteration = 0
    for step_size, phase_iterations in step_phases:
        if iteration >= max_iterations:
            break
            
        for _ in range(min(phase_iterations, max_iterations - iteration)):
            if iteration >= max_iterations:
                break
                
            improved = False
            # Try small perturbations to each element
            for i in range(len(current_f_vals)):
                original_val = current_f_vals[i]
                
                # Try both directions and step sizes
                for direction in [-1, 1]:
                    for step in [step_size, step_size * 0.5]:
                        if iteration >= max_iterations:
                            break
                            
                        test_val = original_val + direction * step
                        if test_val >= 0:
                            test_vals = current_f_vals.copy()
                            test_vals[i] = test_val
                            new_c2 = compute_c2(test_vals)
                            
                            if new_c2 > current_c2:
                                current_c2 = new_c2
                                current_f_vals = test_vals
                                improved = True
                                iteration += 1
                                break
                    if improved:
                        break
                        
            iteration += 1
            if not improved:
                break
                
    return current_f_vals, current_c2

def optimization_pipeline():
    """Main optimization pipeline that tries multiple approaches"""
    best_c2 = 0.0
    best_f_vals = None
    
    # Multi-initialization strategy with diverse approaches
    initial_strategies = [
        lambda n: np.ones(n) * 0.5,  # Uniform function
        generate_convolution_optimized_initial_function,
        generate_balanced_initial_function,
        generate_adaptive_initial_function,
        lambda n: np.random.exponential(1, n)  # Exponential random
    ]
    
    # Try different configurations with different initialization strategies
    for strategy in initial_strategies:
        for attempt in range(15):  # 15 attempts per strategy
            if attempt * 3 > MAX_TIME_SECONDS - 10:  # Time budget check
                break
                
            # Randomly sample number of steps
            n_steps = np.random.randint(N_MIN, N_MAX + 1)
            
            # Generate candidate with specific strategy
            try:
                f_vals = strategy(n_steps)
                
                # Evaluate
                c2 = compute_c2(f_vals)
                
                if c2 > best_c2:
                    best_c2 = c2
                    best_f_vals = f_vals.copy()
            except Exception:
                continue
    
    # If we found something better than baseline, let's refine it further
    if best_f_vals is not None and best_c2 > 0:
        # First apply adaptive local search to refine
        refined_f_vals, refined_c2 = adaptive_local_search(best_f_vals, max_iterations=30)
        
        if refined_c2 > best_c2:
            best_c2 = refined_c2
            best_f_vals = refined_f_vals
        
        # Then try differential evolution for final tuning if time allows
        n_steps = len(best_f_vals)
        if n_steps < 1000:  # Only for smaller problems to avoid memory issues
            try:
                def objective(x):
                    # Ensure non-negative values and reasonable scaling
                    f_vals = np.abs(x) * 5.0  # Scale to reasonable range
                    return -compute_c2(f_vals)  # Negative because we minimize
                
                bounds = [(0, 10) for _ in range(n_steps)]
                result = differential_evolution(
                    objective, 
                    bounds, 
                    maxiter=30, 
                    popsize=10, 
                    seed=42,
                    disp=False
                )
                
                if result.success:
                    final_f_vals = np.abs(result.x) * 5.0
                    final_c2 = compute_c2(final_f_vals)
                    
                    if final_c2 > best_c2:
                        best_c2 = final_c2
                        best_f_vals = final_f_vals
            except Exception:
                pass  # Fall back to previous best if optimization fails
    
    return best_f_vals.tolist() if best_f_vals is not None else [0.5]*100

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Execute the optimization pipeline
    result = optimization_pipeline()
    
    # Limit execution time
    end_time = time.time()
    if end_time - start_time > MAX_TIME_SECONDS - 5:  # Leave buffer
        return result
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
