# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
from numba import jit, prange
import numba

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
    # The convolution g_vals spans domain [-1/2, 1/2], so total width is 1.0
    # With len(g_vals) points, step width h = 1.0 / (len(g_vals) - 1) if len > 1
    if len(g_vals) >= 2:
        g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
        for i in range(1, len(g_vals)-1):
            g_l2_sq += 2 * g_vals[i] * g_vals[i]
        # Correct step width for domain [-1/2, 1/2] with len(g_vals) points
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # Compute C2
    if g_l1 > 1e-15 and g_max > 1e-15:
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

def objective_function(params):
    """Objective function to minimize (negative C2)"""
    try:
        # Clip negative values
        f_vals = np.clip(params, 0, None)

        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)

        # Compute C2
        c2 = compute_c2_numba(g_vals)

        # Return negative because we're minimizing
        return -c2
    except Exception as e:
        return 1e10  # Large penalty for invalid results

def sophisticated_initialization(dim):
    """Create a sophisticated initial step function using Sobol sequences and pattern recognition"""
    # Generate points using Sobol sequence for better space-filling
    try:
        sampler = qmc.Sobol(d=dim, seed=42)
        points = sampler.random(n=100)
    except:
        # Fallback to regular random if Sobol fails
        points = np.random.random((100, dim))

    # Create a pattern that has shown success in previous implementations
    init_params = []
    for i in range(dim):
        # Create structured pattern: alternating high/low with sinusoidal modulation
        pattern_val = 0.5 + 0.3 * np.sin(i * 0.7)
        # Add variation from Sobol sampling
        variation = points[i % 100][0] * 0.2 if i < 100 else np.random.random() * 0.2
        init_params.append(max(0, pattern_val + variation - 0.1))

    return init_params

def evolutionary_optimization(initial_dim):
    """Perform evolutionary optimization with adaptive parameters"""
    # Start with good initialization
    x0 = sophisticated_initialization(initial_dim)

    # Set bounds for optimization
    bounds = [(0, 10)] * len(x0)

    # Parameters for differential evolution
    de_params = {
        'mutation': (0.5, 1),
        'recombination': 0.7,
        'popsize': 15,
        'maxiter': 100,
        'seed': 42,
        'tol': 1e-6,
        'init': 'latinhypercube',
        'disp': False
    }

    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds,
        **de_params
    )

    return result.x

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Multi-start approach with different random seeds and dimensions
    best_c2 = -np.inf
    best_params = None

    # Try multiple optimizations with different configurations
    configs = [
        (500, 42),
        (700, 123),
        (900, 456),
        (1100, 789)
    ]

    # Also try a few random configurations
    for _ in range(3):
        dim = np.random.randint(500, 1200)
        seed = np.random.randint(1000, 9999)
        configs.append((dim, seed))

    for dim, seed in configs:
        if time.time() - start_time > 85:  # Leave buffer for cleanup
            break

        try:
            np.random.seed(seed)
            params = evolutionary_optimization(dim)

            # Compute actual C2 value
            f_vals = np.clip(params, 0, None)
            if len(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                c2 = compute_c2_numba(g_vals)

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