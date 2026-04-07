# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import random
import time
from typing import List, Tuple
import numba
from numba import jit

@jit(nopython=True)
def compute_autoconvolution_fast(f_vals):
    """Fast numba-based autoconvolution computation"""
    n = len(f_vals)
    g = np.zeros(2*n - 1)
    
    for i in range(n):
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

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation with proper numerical handling
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Ensure non-negative values
    f_array = np.array(f_values)
    f_array = np.maximum(f_array, 0.0)
    
    # Compute autoconvolution
    g_vals = compute_autoconvolution_fast(f_array)
    
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
        g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
        return c2
    except:
        return 0.0

def construct_geometric_initial_function(n: int) -> List[float]:
    """
    Create initial function with geometric patterns that are likely to produce good C2 values
    """
    # Create a pattern that balances high peaks with spread-out structure
    f_vals = []
    
    # Base pattern: alternating high/low with some smooth transitions
    for i in range(n):
        # Create a base geometric structure
        x = i / (n - 1) if n > 1 else 0.5
        # Add some smooth variation around a geometric base
        base_val = 0.5 + 0.5 * np.sin(4 * np.pi * x)
        # Add some randomization but keep values positive
        noise = 0.1 * (np.random.random() - 0.5)
        val = max(0, base_val + noise)
        f_vals.append(val)
    
    # Occasionally add a few strong peaks for convolution enhancement
    num_peaks = max(1, n // 20)
    for _ in range(num_peaks):
        peak_pos = random.randint(0, n-1)
        f_vals[peak_pos] = max(f_vals[peak_pos], 2.0 * random.random())
    
    return f_vals

def adaptive_local_search(start_point: List[float], max_evals: int = 1000) -> Tuple[List[float], float]:
    """
    Adaptive local search using multiple optimization strategies with finite difference gradient estimation
    """
    # Convert to numpy for easier manipulation
    x0 = np.array(start_point)
    n = len(x0)
    
    # First try: Nelder-Mead with adaptive parameters
    try:
        res_nm = minimize(
            lambda x: -calculate_c2(x.tolist()), 
            x0, 
            method='Nelder-Mead',
            options={'maxfev': max_evals//3, 'disp': False, 'adaptive': True}
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
            options={'maxfev': max_evals//3, 'disp': False}
        )
        if res_powell.success:
            test_c2 = -res_powell.fun
            if test_c2 > best_c2:
                best_x = res_powell.x
                best_c2 = test_c2
    except:
        pass
    
    # Third try: Coordinate-wise refinement with gradient estimation
    try:
        # Implement a smart coordinate-wise gradient estimation and refinement
        current_x = best_x.copy()
        current_c2 = best_c2
        prev_c2 = 0
        iterations = 0
        tolerance = 1e-8
        max_iterations = 50
        
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
            
            # Try to find a better direction for each dimension
            for i in range(n):
                # Estimate gradient using finite differences
                eps = 1e-4
                
                # Try increasing the parameter
                temp_x_plus = current_x.copy()
                temp_x_plus[i] = max(0, temp_x_plus[i] + eps)
                c2_plus = cached_c2(tuple(temp_x_plus))
                
                # Try decreasing the parameter  
                temp_x_minus = current_x.copy()
                temp_x_minus[i] = max(0, temp_x_minus[i] - eps)
                c2_minus = cached_c2(tuple(temp_x_minus))
                
                # Estimate gradient
                grad_estimate = (c2_plus - c2_minus) / (2 * eps)
                
                # If gradient is positive (increasing), move in that direction
                if grad_estimate > 0:
                    # Move along the estimated gradient direction
                    step_size = 0.01 * (1.0 + random.random() * 0.5)  # Adaptive step size
                    new_x = current_x.copy()
                    new_x[i] = max(0, current_x[i] + step_size * grad_estimate)
                    new_c2 = cached_c2(tuple(new_x))
                    
                    if new_c2 > current_c2:
                        current_x = new_x
                        current_c2 = new_c2
                        improved = True
                        
            if not improved:
                # If no improvement, try small random perturbations
                for i in range(min(5, n)):  # Try up to 5 dimensions
                    if random.random() < 0.3:  # 30% chance to perturb
                        idx = random.randint(0, n-1)
                        delta = (random.random() - 0.5) * 0.05
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

def multi_start_optimization(max_starts: int = 25, max_evals_per_start: int = 1000) -> Tuple[List[float], float]:
    """
    Multi-start optimization with diverse initialization strategies and adaptive refinement
    """
    best_overall_c2 = 0.0
    best_overall_solution = None
    
    # Track good solutions for adaptive sampling  
    good_solutions = []
    good_c2_scores = []
    
    # Start with diverse geometric patterns
    for i in range(max_starts):
        # Generate different initial patterns with varying characteristics
        n = random.randint(150, 900)  # Vary the length for exploration
        
        # Strategy 1: Geometric pattern
        if i % 4 == 0:
            initial_solution = construct_geometric_initial_function(n)
        # Strategy 2: Sine wave pattern
        elif i % 4 == 1:
            initial_solution = [0.5 + 0.3 * np.sin(2 * np.pi * j / n) + random.gauss(0, 0.05) 
                              for j in range(n)]
            initial_solution = [max(0, x) for x in initial_solution]
        # Strategy 3: Alternating pattern
        elif i % 4 == 2:
            initial_solution = [random.uniform(0.7, 1.0) if j % 2 == 0 
                              else random.uniform(0.1, 0.3) for j in range(n)]
        # Strategy 4: Gaussian-based
        else:
            initial_solution = [abs(random.gauss(0.5, 0.15)) for _ in range(n)]
        
        # Local optimization from this starting point
        solution, c2 = adaptive_local_search(initial_solution, max_evals_per_start)
        
        if c2 > best_overall_c2:
            best_overall_c2 = c2
            best_overall_solution = solution.copy()
            
        # Keep track of good solutions for reseeding
        if c2 > best_overall_c2 * 0.9:  # Keep top 10% solutions
            good_solutions.append(solution.copy())
            good_c2_scores.append(c2)
    
    # Further refine top solutions
    top_count = min(5, len(good_solutions))
    for i, sol in enumerate(good_solutions[:top_count]):  # Refine top few
        if i < 5:  # Limit additional refinements
            refined_sol, refined_c2 = adaptive_local_search(sol, max_evals_per_start//2)
            if refined_c2 > best_overall_c2:
                best_overall_c2 = refined_c2
                best_overall_solution = refined_sol.copy()
    
    return best_overall_solution, best_overall_c2

def construct_function() -> List[float]:
    """
    Function to construct step-function with high C2 value.
    Uses gradient-free optimization approach with adaptive refinement.
    """
    start_time = time.time()
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Multi-start optimization with adaptive search
    try:
        final_solution, final_c2 = multi_start_optimization(
            max_starts=20, 
            max_evals_per_start=800
        )
        
        # Ensure we return a valid solution
        if final_solution is not None:
            # Final validation check
            validate_c2 = calculate_c2(final_solution)
            if validate_c2 > final_c2:
                final_c2 = validate_c2
        else:
            # Fallback to simple geometric pattern
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