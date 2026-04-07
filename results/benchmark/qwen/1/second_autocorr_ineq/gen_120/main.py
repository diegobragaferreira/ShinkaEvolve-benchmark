# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
from numba import jit, prange
import numba
from scipy.optimize import minimize

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation"""
    n = len(f_vals)
    # Convolution result has length 2*n-1
    g_len = 2 * n - 1
    g = np.zeros(g_len)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation with improved integration"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (max absolute value)
    for i in range(len(g_vals)):
        if abs(g_vals[i]) > g_max:
            g_max = abs(g_vals[i])

    # Compute L2^2 norm using improved trapezoidal integration
    # For discrete convolution on [-1/2, 1/2] with len(g_vals) samples
    # Step width h = 1.0 / (len(g_vals) - 1)  
    if len(g_vals) >= 2:
        # Trapezoidal rule: h * (y0^2 + 2*y1^2 + ... + yn-1^2)/2
        # But we use the more accurate piece-wise quadratic integration
        g_l2_sq = 0.0
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        
        # Using trapezoidal rule for integral of g^2
        # Sum of (h/2) * (g[i]^2 + g[i+1]^2) for all adjacent pairs
        for i in range(len(g_vals) - 1):
            g_l2_sq += (h/2) * (g_vals[i]*g_vals[i] + g_vals[i+1]*g_vals[i+1])
        
        # For exact quadratic integration over piecewise linear segments:
        # We can also compute this using the more accurate formula for piecewise integration
        # But the basic trapezoidal rule approach works well here
        pass  # Already computed above correctly
        
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # Compute C2
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def compute_c2_for_params(params):
    """Wrapper function for optimization with error handling"""
    try:
        # Ensure non-negative values
        f_vals = np.clip(params, 0, None)
        
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)
        
        # Compute C2
        c2 = compute_c2_numba(g_vals)
        
        return c2
    except Exception as e:
        return 0.0

def advanced_sophisticated_initialization(dim):
    """Create an advanced initial step function with multi-scale pattern generation"""
    # Create base pattern using multi-scale approach
    init_params = []
    
    # Scale 1: Base sinusoidal pattern
    base_pattern = []
    for i in range(dim):
        base_val = 0.5 + 0.3 * np.sin(i * 0.5)
        base_pattern.append(base_val)
    
    # Scale 2: Add peak valleys for structure
    peak_valley_pattern = []
    for i in range(dim):
        if i % 5 == 0:
            peak_valley_pattern.append(1.0)
        elif i % 5 == 2:
            peak_valley_pattern.append(0.2)
        else:
            peak_valley_pattern.append(0.5)
    
    # Scale 3: Random variation to ensure diversity
    random_pattern = np.random.random(dim) * 0.4 + 0.3
    
    # Combine patterns with weights
    combined_pattern = []
    for i in range(dim):
        combined_val = 0.4 * base_pattern[i] + 0.3 * peak_valley_pattern[i] + 0.3 * random_pattern[i]
        combined_pattern.append(combined_val)
    
    # Add controlled noise from Sobol sequence
    try:
        sampler = qmc.Sobol(d=dim, seed=42)
        sobol_points = sampler.random(n=100)
    except:
        sobol_points = np.random.random((100, dim))
    
    # Apply Sobol noise to create more diverse initialization
    final_pattern = []
    for i in range(dim):
        noise = sobol_points[i % 100][0] * 0.15 if i < 100 else np.random.random() * 0.15
        val = max(0, combined_pattern[i] + noise - 0.075)
        final_pattern.append(val)
    
    return final_pattern

def adaptive_differential_evolution(x0, bounds, initial_popsize, maxiter):
    """Run differential evolution with adaptive parameters based on convergence"""
    # Start with small population for exploration
    popsize = initial_popsize
    
    # Parameters for differential evolution
    de_params = {
        'mutation': (0.5, 1),
        'recombination': 0.7,
        'popsize': popsize,
        'maxiter': maxiter,
        'seed': 42,
        'tol': 1e-6,
        'init': 'latinhypercube',
        'disp': False
    }
    
    # Run optimization
    result = differential_evolution(
        lambda x: -compute_c2_for_params(x),
        bounds,
        **de_params
    )
    
    return result.x

def advanced_local_search(initial_params, max_iter=50):
    """Advanced local search that incorporates convolution structure knowledge"""
    def objective(x):
        return -compute_c2_for_params(x)
    
    # Try multiple local search methods
    best_params = initial_params.copy()
    best_c2 = compute_c2_for_params(best_params)
    
    # Method 1: L-BFGS-B
    try:
        result = minimize(
            objective, 
            best_params, 
            method='L-BFGS-B', 
            bounds=[(0, 10) for _ in range(len(initial_params))],
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        if result.success:
            params = result.x
            c2 = compute_c2_for_params(params)
            if c2 > best_c2:
                best_params = params
                best_c2 = c2
    except:
        pass
    
    # Method 2: Local perturbation to escape local minima
    try:
        # Add stochastic perturbation to avoid local optima
        np.random.seed(42)
        perturbed_params = best_params + np.random.normal(0, 0.05, len(best_params))
        perturbed_params = np.clip(perturbed_params, 0, None)
        c2 = compute_c2_for_params(perturbed_params)
        if c2 > best_c2:
            best_params = perturbed_params
            best_c2 = c2
    except:
        pass
    
    return best_params

def multi_scale_evolutionary_optimization():
    """Perform multi-scale evolutionary optimization with adaptive parameters"""
    start_time = time.time()
    
    # Multi-scale approach: start with coarse optimization, then refine
    # Scale 1: Coarse search with small population
    coarse_dim = np.random.randint(200, 400)
    coarse_bounds = [(0, 10)] * coarse_dim
    coarse_x0 = advanced_sophisticated_initialization(coarse_dim)
    
    coarse_result = adaptive_differential_evolution(
        coarse_x0, 
        coarse_bounds, 
        initial_popsize=10, 
        maxiter=30
    )
    
    # Scale 2: Medium search with moderate population
    medium_dim = np.random.randint(400, 700)
    medium_bounds = [(0, 10)] * medium_dim
    medium_x0 = advanced_sophisticated_initialization(medium_dim)
    
    medium_result = adaptive_differential_evolution(
        medium_x0, 
        medium_bounds, 
        initial_popsize=15, 
        maxiter=50
    )
    
    # Scale 3: Fine search with larger population
    fine_dim = np.random.randint(700, 1000)
    fine_bounds = [(0, 10)] * fine_dim
    fine_x0 = advanced_sophisticated_initialization(fine_dim)
    
    fine_result = adaptive_differential_evolution(
        fine_x0, 
        fine_bounds, 
        initial_popsize=20, 
        maxiter=70
    )
    
    # Evaluate all results and select best
    results = [
        (coarse_result, compute_c2_for_params(coarse_result)),
        (medium_result, compute_c2_for_params(medium_result)),
        (fine_result, compute_c2_for_params(fine_result))
    ]
    
    # Refine the best initial result with local search
    best_params, best_c2 = max(results, key=lambda x: x[1])
    
    # Apply advanced local search for final improvement
    refined_params = advanced_local_search(best_params, max_iter=100)
    refined_c2 = compute_c2_for_params(refined_params)
    
    if refined_c2 > best_c2:
        return refined_params
    else:
        return best_params

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using multi-scale optimization."""
    start_time = time.time()

    # Multi-start approach with adaptive parameters
    best_c2 = -np.inf
    best_params = None

    # Try multiple optimization runs with different approaches
    for run in range(5):  # Run 5 different optimization attempts
        if time.time() - start_time > 85:  # Leave buffer for cleanup
            break
            
        try:
            # Run multi-scale evolutionary optimization
            params = multi_scale_evolutionary_optimization()
            
            # Compute actual C2 value
            if len(params) > 0:
                c2 = compute_c2_for_params(params)
                
                if c2 > best_c2:
                    best_c2 = c2
                    best_params = params.copy()
        except Exception as e:
            continue

    # If no valid parameters found, return default
    if best_params is None:
        return [0.5] * 100

    # Final check and conversion to list
    final_f_vals = np.clip(best_params, 0, None)
    return final_f_vals.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
