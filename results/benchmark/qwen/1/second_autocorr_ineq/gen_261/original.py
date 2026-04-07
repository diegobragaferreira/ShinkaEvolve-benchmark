# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc
import time
from numba import jit, prange
import numba
from sklearn.cluster import KMeans
import warnings

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
    """Compute C2 value using fast Numba implementation with proper integration"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms using trapezoidal integration for L2^2
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

    # Compute L2^2 norm using trapezoidal integration
    if len(g_vals) >= 2:
        g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
        for i in range(1, len(g_vals)-1):
            g_l2_sq += 2 * g_vals[i] * g_vals[i]
        # Correct step width for domain [-1/2, 1/2] with len(g_vals) points
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # Compute C2 with improved numerical stability
    epsilon = 1e-16
    if g_l1 > epsilon and g_max > epsilon:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute L1, L2^2, and L-infinity norms efficiently"""
    n = len(g_vals)

    # L1 norm approximation (sum of absolute values)
    l1_norm = 0.0
    for i in range(n):
        l1_norm += abs(g_vals[i])

    # L2^2 norm (sum of squares)
    l2_sq_norm = 0.0
    for i in range(n):
        l2_sq_norm += g_vals[i] * g_vals[i]

    # L-infinity norm (maximum absolute value)
    linf_norm = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf_norm:
            linf_norm = abs_val

    return l1_norm, l2_sq_norm, linf_norm

def gaussian_pattern_initialization(dim):
    """Initialize with a Gaussian-shaped pattern"""
    x = np.linspace(-1.0, 1.0, dim)
    # Create two overlapping Gaussians with different widths
    g1 = np.exp(-x**2 / 0.1) * 0.8
    g2 = np.exp(-(x - 0.4)**2 / 0.2) * 0.6
    g3 = np.exp(-(x + 0.4)**2 / 0.2) * 0.6
    pattern = np.maximum(g1, np.maximum(g2, g3))
    return pattern.tolist()

def fractal_like_initialization(dim):
    """Initialize with fractal-like self-similar pattern"""
    pattern = []
    # Generate multiple scales of pattern to create complexity
    for i in range(dim):
        # Base pattern with multiple frequencies
        pos = i / dim
        base = 0.5 + 0.3 * np.sin(pos * np.pi * 8) 
        base += 0.2 * np.sin(pos * np.pi * 16)
        base += 0.1 * np.sin(pos * np.pi * 32)
        pattern.append(max(0, base))
    return pattern

def structured_initialization(dim):
    """Create a structured pattern with alternating peaks and valleys"""
    pattern = []
    for i in range(dim):
        if i % 5 == 0:
            pattern.append(1.0)
        elif i % 5 == 2:
            pattern.append(0.3)
        else:
            pattern.append(0.7)
    return pattern

def initialize_with_clustering_analysis(dim):
    """Initialize using clustering to find promising patterns"""
    # Generate multiple candidate patterns
    candidates = []
    candidates.append(gaussian_pattern_initialization(dim))
    candidates.append(fractal_like_initialization(dim))
    candidates.append(structured_initialization(dim))
    
    # Add some random variations
    np.random.seed(42)
    for _ in range(3):
        pattern = []
        for i in range(dim):
            pattern.append(0.5 + 0.3 * np.sin(i * 0.3) + np.random.normal(0, 0.1))
        candidates.append([max(0, p) for p in pattern])
    
    # Evaluate all candidates
    best_candidate = candidates[0]
    best_score = -1.0
    
    for candidate in candidates:
        try:
            score = compute_c2_numba(compute_autoconvolution_numba(candidate))
            if score > best_score:
                best_score = score
                best_candidate = candidate[:]
        except:
            continue
    
    return best_candidate

def adaptive_evolutionary_optimization(initial_dim, max_iter=150, adaptive=True):
    """Perform evolutionary optimization with adaptive parameters"""
    
    # Start with good initialization
    x0 = initialize_with_clustering_analysis(initial_dim)

    # Set bounds for optimization
    bounds = [(0, 10)] * len(x0)
    
    # Initial parameters
    popsize = min(20, initial_dim // 10 + 5)  # Adaptive population size
    mutation_range = (0.5, 0.9)
    recombination_rate = 0.7
    
    # Parameters for differential evolution
    de_params = {
        'mutation': mutation_range,
        'recombination': recombination_rate,
        'popsize': popsize,
        'maxiter': max_iter,
        'seed': 42,
        'tol': 1e-6,
        'init': 'latinhypercube',
        'disp': False
    }

    # Run optimization
    result = differential_evolution(
        lambda x: -compute_c2_numba(compute_autoconvolution_numba(x)),
        bounds,
        **de_params
    )

    return result.x

def convex_local_refinement(initial_params, max_iter=100):
    """Use convex optimization-based local refinement for better convergence"""
    
    def objective(x):
        return -compute_c2_numba(compute_autoconvolution_numba(x))
    
    try:
        # First use L-BFGS-B for smooth local optimization
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=[(0, 10) for _ in range(len(initial_params))],
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-9},
            tol=1e-9
        )
        return result.x
    except Exception as e:
        # Fallback to simpler method if L-BFGS fails
        try:
            # Use Nelder-Mead as fallback
            result = minimize(
                objective,
                initial_params,
                method='Nelder-Mead',
                options={'maxiter': max_iter//2, 'ftol': 1e-8, 'xtol': 1e-8}
            )
            return result.x
        except Exception:
            # Last resort: simple coordinate-wise improvement
            current_params = np.array(initial_params)
            for iteration in range(max_iter):
                improved = False
                for i in range(len(current_params)):
                    # Test small adjustments
                    test_params = current_params.copy()
                    adjustment = 0.01 * np.random.randn()
                    test_params[i] = max(0, current_params[i] + adjustment)
                    if compute_c2_numba(compute_autoconvolution_numba(test_params)) > \
                       compute_c2_numba(compute_autoconvolution_numba(current_params)):
                        current_params[i] = test_params[i]
                        improved = True
                if not improved:
                    break
            return current_params

def multi_stage_optimization():
    """Run multi-stage optimization pipeline with progressive refinement"""
    start_time = time.time()
    best_c2 = -np.inf
    best_params = None
    
    # Stage 1: Coarse optimization with large population
    configs = [(300, 80), (500, 100), (700, 120)]
    
    for dim, iter_count in configs:
        if time.time() - start_time > 80:  # Leave buffer for cleanup
            break
            
        try:
            # Multi-start with different seeds
            for seed in [42, 123, 456]:
                np.random.seed(seed)
                params = adaptive_evolutionary_optimization(dim, iter_count)
                
                # Local refinement
                refined_params = convex_local_refinement(params, 50)
                
                # Evaluate final result
                c2 = compute_c2_numba(compute_autoconvolution_numba(refined_params))
                
                if c2 > best_c2:
                    best_c2 = c2
                    best_params = refined_params.copy()
                    
        except Exception as e:
            continue
    
    # Stage 2: Fine-tuning with smaller population but more iterations
    if best_params is not None and time.time() - start_time < 85:
        try:
            # Use best parameters as starting point for fine-tuning
            fine_params = convex_local_refinement(best_params, 100)
            fine_c2 = compute_c2_numba(compute_autoconvolution_numba(fine_params))
            
            if fine_c2 > best_c2:
                best_c2 = fine_c2
                best_params = fine_params
        except Exception:
            pass
    
    # Stage 3: Additional adaptive refinement with different parameters
    if best_params is not None and time.time() - start_time < 88:
        try:
            # Try a different sampling approach
            adaptive_configs = [
                (600, 80),  # Different dimension and iterations
                (800, 60),
                (1000, 50)
            ]
            
            for dim, iter_count in adaptive_configs:
                if time.time() - start_time > 88:
                    break
                    
                np.random.seed(int(time.time()) % 10000)
                params = adaptive_evolutionary_optimization(dim, iter_count)
                refined_params = convex_local_refinement(params, 75)
                c2 = compute_c2_numba(compute_autoconvolution_numba(refined_params))
                
                if c2 > best_c2:
                    best_c2 = c2
                    best_params = refined_params.copy()
                    
        except Exception:
            pass
    
    return best_params, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    try:
        # Main optimization strategy
        best_params, best_c2 = multi_stage_optimization()
        
        # If we found a good solution, use it
        if best_params is not None and best_c2 > 0:
            return best_params
        
        # Fallback to deterministic approach
        size = 800
        params = adaptive_evolutionary_optimization(size, 100)
        params = convex_local_refinement(params, 50)
        c2_val = compute_c2_numba(compute_autoconvolution_numba(params))
        print(f"Best C2 found: {c2_val}")

        # Return the optimized values
        return params.tolist()

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Final fallback to structured initialization
        return initialize_with_clustering_analysis(1000)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")