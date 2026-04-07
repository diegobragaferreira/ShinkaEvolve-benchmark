# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
import random
import time
from typing import List, Tuple
import numba
from numba import jit, prange
import math

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True, parallel=True)
def compute_autoconvolution_fast_parallel(f_vals):
    """Fast parallel numba-based autoconvolution computation"""
    n = len(f_vals)
    g = np.zeros(2*n - 1)
    
    # Parallel computation of convolution
    for i in prange(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    
    return g

@jit(nopython=True)
def compute_convolution_norms_fast(g_vals):
    """Fast computation of convolution norms"""
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

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values):
    """
    Fast computation of autoconvolution norms with numba JIT
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Ensure non-negative values
    f_array = np.array(f_values)
    f_array = np.maximum(f_array, 0.0)
    
    # Compute autoconvolution
    g_vals = compute_autoconvolution_fast_parallel(f_array)
    
    # Compute norms
    l2_sq, l1, l_inf = compute_convolution_norms_fast(g_vals)
    
    # Handle edge cases
    if l1 <= 1e-15:
        l1 = 1e-15
    if l_inf <= 1e-15:
        l_inf = 1e-15
        
    return l2_sq, l1, l_inf

def calculate_c2(f_values: List[float]) -> float:
    """
    Calculate the C2 constant from the step function values.
    """
    try:
        g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms_fast(f_values)
        c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
        return c2
    except:
        return 0.0

def construct_geometric_initial_function(n: int) -> List[float]:
    """
    Create enhanced initial function with multi-scale geometric patterns
    """
    # Create a sophisticated multi-scale geometric pattern for better C2 performance
    f_vals = []
    
    # Scale 1: Main bell-shaped pattern
    x = np.linspace(0, 1, n)
    bell_pattern = 0.8 * np.exp(-((x - 0.3)**2) / (2 * 0.1**2)) + \
                   0.6 * np.exp(-((x - 0.7)**2) / (2 * 0.1**2)) + \
                   0.4 * np.exp(-((x - 0.5)**2) / (2 * 0.15**2))
    
    # Scale 2: Medium scale sinusoidal variation
    sin_pattern = 0.2 * np.sin(12 * np.pi * x) + 0.1 * np.cos(24 * np.pi * x)
    
    # Scale 3: Fine scale noise
    noise_pattern = 0.05 * np.random.normal(0, 1, n)
    
    # Combine all patterns
    combined = bell_pattern + sin_pattern + noise_pattern
    
    # Normalize and clip to [0, 1]
    combined = np.clip(combined, 0, 1)
    
    # Convert to list
    f_vals = combined.tolist()
    
    # Add occasional strong peaks for convolution enhancement
    num_peaks = max(1, n // 15)
    for _ in range(num_peaks):
        peak_pos = random.randint(0, n-1)
        f_vals[peak_pos] = max(f_vals[peak_pos], 1.5 * random.random())
    
    return f_vals

def adaptive_local_search(start_point: List[float], max_evals: int = 1000) -> Tuple[List[float], float]:
    """
    Enhanced adaptive local search with improved gradient estimation and smarter refinement
    """
    # Convert to numpy for easier manipulation
    x0 = np.array(start_point)
    n = len(x0)
    
    # First try: Nelder-Mead with aggressive parameters
    try:
        res_nm = minimize(
            lambda x: -calculate_c2(x.tolist()), 
            x0, 
            method='Nelder-Mead',
            options={'maxfev': max_evals//4, 'disp': False, 'adaptive': True, 'initial_simplex': None}
        )
        if not res_nm.success:
            raise Exception("Nelder-Mead failed")
        best_x = res_nm.x
        best_c2 = -res_nm.fun
    except:
        # Fallback to simpler approach
        best_x = x0
        best_c2 = calculate_c2(x0.tolist())
    
    # Second try: Powell method for better convergence
    try:
        res_powell = minimize(
            lambda x: -calculate_c2(x.tolist()),
            best_x,
            method='Powell',
            options={'maxfev': max_evals//4, 'disp': False}
        )
        if res_powell.success:
            test_c2 = -res_powell.fun
            if test_c2 > best_c2:
                best_x = res_powell.x
                best_c2 = test_c2
    except:
        pass
    
    # Third try: Smart coordinate-wise refinement with better gradient estimation
    try:
        # Implement advanced coordinate-wise refinement
        current_x = best_x.copy()
        current_c2 = best_c2
        prev_c2 = 0
        iterations = 0
        tolerance = 1e-8
        max_iterations = 30
        
        # Cache function evaluations to avoid recomputation
        cached_evals = {}
        
        def cached_c2(x_tuple):
            if x_tuple in cached_evals:
                return cached_evals[x_tuple]
            result = calculate_c2(list(x_tuple))
            cached_evals[x_tuple] = result
            return result
        
        while abs(current_c2 - prev_c2) > tolerance and iterations < max_iterations:
            prev_c2 = current_c2
            improved = False
            
            # Try to find a better direction for each dimension using more sophisticated approach
            for i in range(n):
                # Estimate gradient using central difference with multiple step sizes
                eps = 1e-5
                
                # Try increasing and decreasing the parameter
                temp_x_plus = current_x.copy()
                temp_x_plus[i] = max(0, temp_x_plus[i] + eps)
                c2_plus = cached_c2(tuple(temp_x_plus))
                
                temp_x_minus = current_x.copy()
                temp_x_minus[i] = max(0, temp_x_minus[i] - eps)
                c2_minus = cached_c2(tuple(temp_x_minus))
                
                # Estimate gradient
                grad_estimate = (c2_plus - c2_minus) / (2 * eps)
                
                # If gradient is positive (increasing), move in that direction
                if grad_estimate > 0:
                    # Adaptive step sizing based on gradient magnitude
                    step_size = 0.02 * (1.0 + abs(grad_estimate) * 0.1)
                    new_x = current_x.copy()
                    new_x[i] = max(0, current_x[i] + step_size * grad_estimate)
                    new_c2 = cached_c2(tuple(new_x))
                    
                    if new_c2 > current_c2:
                        current_x = new_x
                        current_c2 = new_c2
                        improved = True
                        
            if not improved:
                # If no improvement, try more systematic random perturbations
                for i in range(min(3, n)):  # Try up to 3 dimensions
                    if random.random() < 0.4:  # 40% chance to perturb
                        idx = random.randint(0, n-1)
                        # Use exponential distribution for step size
                        delta = np.random.exponential(0.02) * (1.0 if random.random() > 0.5 else -1.0)
                        current_x[idx] = max(0, current_x[idx] + delta)
                        current_c2 = cached_c2(tuple(current_x))
                        improved = True
                        
            iterations += 1
            
        if current_c2 > best_c2:
            best_x = current_x
            best_c2 = current_c2
    except:
        pass
    
    return best_x.tolist(), best_c2

def multi_start_optimization(max_starts: int = 30, max_evals_per_start: int = 800) -> Tuple[List[float], float]:
    """
    Multi-start optimization with sophisticated initialization and smart refinement
    """
    best_overall_c2 = 0.0
    best_overall_solution = None
    
    # Track good solutions for adaptive sampling  
    good_solutions = []
    good_c2_scores = []
    
    # Start with diverse geometric patterns
    for i in range(max_starts):
        # Generate different initial patterns with varying characteristics
        n = random.randint(200, 800)  # Vary the length for exploration
        
        # Strategy 1: Enhanced geometric pattern
        if i % 6 == 0:
            initial_solution = construct_geometric_initial_function(n)
        # Strategy 2: Sine wave pattern with variation
        elif i % 6 == 1:
            x = np.linspace(0, 1, n)
            initial_solution = [0.5 + 0.3 * np.sin(6 * np.pi * x[j]) + 
                              0.1 * np.sin(12 * np.pi * x[j]) + 
                              random.gauss(0, 0.05) for j in range(n)]
            initial_solution = [max(0, x) for x in initial_solution]
        # Strategy 3: Alternating pattern with peaks
        elif i % 6 == 2:
            initial_solution = [random.uniform(0.6, 1.0) if j % 2 == 0 
                              else random.uniform(0.1, 0.3) for j in range(n)]
        # Strategy 4: Gaussian-based with variance
        elif i % 6 == 3:
            initial_solution = [abs(random.gauss(0.5, 0.1)) for _ in range(n)]
        # Strategy 5: Multi-peak pattern
        elif i % 6 == 4:
            x = np.linspace(0, 1, n)
            initial_solution = [0.3 * np.exp(-((x[j] - 0.2)**2) / (2 * 0.05**2)) + 
                              0.4 * np.exp(-((x[j] - 0.6)**2) / (2 * 0.05**2)) + 
                              0.3 * np.exp(-((x[j] - 0.9)**2) / (2 * 0.05**2)) + 
                              random.gauss(0, 0.02) for j in range(n)]
            initial_solution = [max(0, x) for x in initial_solution]
        # Strategy 6: Randomized smooth pattern
        else:
            initial_solution = [random.random() for _ in range(n)]
        
        # Local optimization from this starting point
        solution, c2 = adaptive_local_search(initial_solution, max_evals_per_start)
        
        if c2 > best_overall_c2:
            best_overall_c2 = c2
            best_overall_solution = solution.copy()
            
        # Keep track of good solutions for reseeding
        if c2 > best_overall_c2 * 0.85:  # Keep top 15% solutions
            good_solutions.append(solution.copy())
            good_c2_scores.append(c2)
    
    # Further refine top solutions more aggressively
    top_count = min(4, len(good_solutions))
    for i, sol in enumerate(good_solutions[:top_count]):  # Refine top few
        if i < 4:  # Limit additional refinements
            refined_sol, refined_c2 = adaptive_local_search(sol, max_evals_per_start//3)
            if refined_c2 > best_overall_c2:
                best_overall_c2 = refined_c2
                best_overall_solution = refined_sol.copy()
    
    return best_overall_solution, best_overall_c2

def construct_function() -> List[float]:
    """
    Function to construct step-function with high C2 value.
    Uses enhanced gradient-free optimization approach with smart refinement.
    """
    start_time = time.time()
    
    # Multi-start optimization with enhanced search
    try:
        final_solution, final_c2 = multi_start_optimization(
            max_starts=25, 
            max_evals_per_start=700
        )
        
        # Ensure we return a valid solution
        if final_solution is not None:
            # Final validation check
            validate_c2 = calculate_c2(final_solution)
            if validate_c2 > final_c2:
                final_c2 = validate_c2
        else:
            # Fallback to enhanced geometric pattern
            final_solution = construct_geometric_initial_function(500)
            final_c2 = calculate_c2(final_solution)
            
        # Make sure we have a reasonable minimum
        if final_c2 < 0.1:
            final_solution = construct_geometric_initial_function(500)
            final_c2 = calculate_c2(final_solution)
            
        end_time = time.time()
        return final_solution
        
    except Exception as e:
        # Final fallback
        print(f"Error occurred: {e}")
        return construct_geometric_initial_function(500)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")