# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from numba import jit
import time

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using Numba for speed"""
    n = len(f_vals)
    # Create convolution result array
    g = np.zeros(2*n - 1)

    # Compute autoconvolution: g[k] = sum(f[i]*f[k-i])
    for i in range(n):
        for j in range(n):
            k = i + j
            g[k] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute norms efficiently with Numba"""
    n = len(g_vals)

    # L2 norm squared (using trapezoidal approximation for piecewise linear)
    l2_sq = 0.0
    for i in range(n-1):
        h = 1.0  # assuming unit spacing
        y1 = g_vals[i]
        y2 = g_vals[i+1]
        l2_sq += (h/3.0) * (y1*y1 + y1*y2 + y2*y2)

    # L1 norm
    l1 = 0.0
    for i in range(n):
        l1 += abs(g_vals[i])

    # L-infinity norm
    linf = 0.0
    for i in range(n):
        if abs(g_vals[i]) > linf:
            linf = abs(g_vals[i])

    return l2_sq, l1, linf

def evaluate_c2(f_vals):
    """Evaluate C2 for a given set of step heights"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        # Skip empty sequences
        if len(f_vals) == 0:
            return 0.0

        # Compute autoconvolution using Numba for speed
        g_vals = compute_autoconvolution_numba(f_vals)

        # Compute norms
        l2_squared, l1, l_inf = compute_norms_numba(g_vals)

        # Avoid division by zero
        if l1 <= 1e-15 or l_inf <= 1e-15:
            return 0.0

        # Compute C2
        c2 = l2_squared / (l1 * l_inf)
        return c2
    except Exception as e:
        return 0.0

def objective_function(x):
    """Objective function to minimize (negative C2)"""
    c2 = evaluate_c2(x)
    return -c2

def sophisticated_initialization(n_steps):
    """Create a sophisticated initial population with alternating pattern and Gaussian smoothing"""
    # Create alternating high/low regions with some randomness
    initial = []

    # Divide into segments for structured alternation
    segment_size = max(1, n_steps // 10)

    for i in range(n_steps):
        segment_idx = i // segment_size
        if segment_idx % 2 == 0:
            # High value region (with some randomness)
            base_val = 1.0 + np.random.random() * 0.5
        else:
            # Low value region (with some randomness)
            base_val = 0.1 + np.random.random() * 0.3

        # Apply Gaussian smoothing to avoid sharp transitions
        if len(initial) >= 2:
            # Smooth with previous values
            smooth_factor = 0.7
            base_val = smooth_factor * base_val + (1-smooth_factor) * 0.5 * (initial[-1] + initial[-2])

        initial.append(max(0, base_val))

    return np.array(initial)

def evolutionary_optimization():
    """Use enhanced differential evolution with multi-start and adaptive settings"""
    # Start with a larger initial size for better optimization
    n_steps = 1000

    # Define bounds for each parameter (step height)
    bounds = [(0.0, 2.0) for _ in range(n_steps)]

    # Multi-start approach for better exploration
    best_score = -np.inf
    best_solution = None

    # Try multiple random initializations
    for start_seed in [42, 123, 456]:
        # Use sophisticated initialization
        np.random.seed(start_seed)
        x0 = sophisticated_initialization(n_steps)

        # Adaptive population size based on problem dimensionality
        popsize = max(15, min(30, n_steps // 50))  # Adjust based on n_steps

        try:
            # Run differential evolution with adaptive settings
            result = differential_evolution(
                objective_function,
                bounds,
                seed=start_seed,
                maxiter=150,  # Increase iterations
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False,
                tol=1e-6,
                callback=None  # Remove callback to save time
            )

            # Check if this is better
            if -result.fun > best_score:
                best_score = -result.fun
                best_solution = result.x.copy()

        except Exception:
            continue  # Skip this run if it fails

    # If we didn't find a solution, fallback to a simple approach
    if best_solution is None:
        # Create a basic pattern
        best_solution = np.ones(n_steps) * 0.5

    return best_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using enhanced evolutionary optimization."""
    start_time = time.time()

    # Use enhanced evolutionary optimization to find optimal step heights
    optimized_params = evolutionary_optimization()

    # Clip negative values to zero
    f_values = np.maximum(optimized_params, 0).tolist()

    end_time = time.time()

    # Verify the result
    c2_value = evaluate_c2(optimized_params)

    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"C2 achieved: {c2_value}")

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")