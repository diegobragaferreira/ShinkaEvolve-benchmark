# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
from scipy import signal
from scipy.optimize import differential_evolution
import random

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals, dx):
    """
    Efficiently compute autoconvolution using Numba JIT compilation
    """
    n = len(f_vals)
    # Autoconvolution using discrete convolution formula
    # For step functions, we can use a more efficient approach
    g = np.zeros(2*n - 1)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < len(g):
                g[idx] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """
    Compute L1, L2^2, and L-infinity norms efficiently
    """
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

@jit(nopython=True)
def compute_c2_numba(f_vals, dx):
    """
    Compute C2 value using optimized numba functions
    """
    # Compute autoconvolution
    g_vals = compute_autoconvolution_numba(f_vals, dx)

    # Compute norms
    l1, l2_sq, linf = compute_norms_numba(g_vals)

    # Avoid division by zero
    if l1 <= 1e-15 or linf <= 1e-15:
        return 0.0

    # Return C2 value
    return l2_sq / (l1 * linf)

def evaluate_step_function(f_vals):
    """
    Evaluate a step function and return C2 value
    """
    try:
        # Ensure non-negative values
        f_vals = np.array([max(0.0, x) for x in f_vals])

        # Use a fixed dx for consistent spacing
        dx = 0.5 / len(f_vals) if len(f_vals) > 0 else 0.001

        # Compute C2 value
        c2 = compute_c2_numba(f_vals, dx)
        return c2
    except Exception as e:
        return 0.0

def sophisticated_initialization(n_steps):
    """
    Create a sophisticated initial step function
    """
    # Start with a simple pattern - combine high and low regions
    # This creates a good baseline for optimization
    base_pattern = []
    for i in range(n_steps):
        # Alternating pattern with some randomness
        if i % 2 == 0:
            base_pattern.append(max(0.0, 1.0 + np.random.normal(0, 0.1)))
        else:
            base_pattern.append(max(0.0, 0.1 + np.random.normal(0, 0.05)))

    # Apply gentle smoothing
    smoothed = []
    for i in range(len(base_pattern)):
        # Apply moving average for smoothing
        window_start = max(0, i - 1)
        window_end = min(len(base_pattern), i + 2)
        avg = np.mean(base_pattern[window_start:window_end])
        smoothed.append(avg)

    return smoothed

def evolutionary_optimization():
    """
    Use enhanced evolutionary algorithm with multi-start strategy for optimization
    """
    # Fixed parameters for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Use a reasonable range for number of steps
    min_steps = 200
    max_steps = 2000

    best_solution = None
    best_c2 = -1.0

    # Multi-start approach with different initialization strategies
    for start_iter in range(5):  # Try 5 different starting points
        # Determine number of steps for this run
        n_steps = np.random.randint(min_steps, max_steps)

        # Choose initialization strategy
        init_strategy = np.random.choice(['gaussian', 'alternating', 'peak_centered'])

        if init_strategy == 'gaussian':
            initial_solution = create_gaussian_initialization(n_steps)
        elif init_strategy == 'alternating':
            initial_solution = sophisticated_initialization(n_steps)
        else:  # peak_centered
            initial_solution = create_peak_centered_initialization(n_steps)

        # Define bounds for differential evolution (step heights between 0 and 5)
        bounds = [(0.0, 5.0)] * len(initial_solution)

        try:
            # Run differential evolution with increased parameters for better search
            result = differential_evolution(
                lambda x: -evaluate_step_function(x),  # Negative because we want to maximize
                bounds,
                maxiter=150,      # Increased iterations
                popsize=20,       # Increased population size
                seed=42 + start_iter,  # Different seed for each start
                strategy='best1bin',
                tol=1e-6,
                recombination=0.7,
                disp=False
            )

            if result.success:
                current_c2 = evaluate_step_function(result.x.tolist())
                if current_c2 > best_c2:
                    best_c2 = current_c2
                    best_solution = result.x.tolist()

        except Exception as e:
            continue  # Skip this iteration if it fails

    # If no good solution found, return default
    if best_solution is None:
        n_steps = 500
        best_solution = sophisticated_initialization(n_steps)

    return best_solution

def create_gaussian_initialization(n_steps):
    """Create a gaussian-like initial pattern"""
    x = np.linspace(-1, 1, n_steps)
    # Create multiple gaussian peaks
    gaussian1 = np.exp(-0.5 * ((x - 0.3) / 0.2)**2) * 0.8
    gaussian2 = np.exp(-0.5 * ((x + 0.3) / 0.2)**2) * 0.5
    base = gaussian1 + gaussian2 + 0.1

    # Normalize
    if np.sum(base) > 0:
        base = base / np.sum(base)

    return base.tolist()

def create_peak_centered_initialization(n_steps):
    """Create a peak-centered initial pattern"""
    f = np.zeros(n_steps)
    # Create a central peak with surrounding regions
    center = n_steps // 2
    width = max(1, n_steps // 6)

    # Central peak
    f[max(0, center-width//2):min(n_steps, center+width//2)] = 1.0

    # Surrounding areas with lower values
    for i in range(n_steps):
        dist_from_center = abs(i - center) / n_steps
        if dist_from_center < 0.4:
            f[i] *= (1 - dist_from_center * 2.5)
        else:
            f[i] *= 0.3

    # Add some noise
    noise = np.random.normal(0, 0.05, n_steps)
    f = f + noise

    # Ensure non-negativity
    f = np.clip(f, 0, None)

    # Normalize
    if np.sum(f) > 0:
        f = f / np.sum(f)

    return f.tolist()

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value.
    Uses evolutionary optimization to find better solutions than random initialization.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Run evolutionary optimization
    start_time = time.time()
    try:
        best_solution = evolutionary_optimization()
    except Exception as e:
        # Fallback to simple approach if evolution fails
        print(f"Evolution failed with error: {e}")
        best_solution = [1.0] * 100  # Default case

    end_time = time.time()
    eval_time = end_time - start_time

    # Ensure the solution is valid
    if not best_solution:
        best_solution = [1.0] * 100

    print(f"Eval time: {eval_time:.4f}s")
    print(f"Best C2 found: {evaluate_step_function(best_solution):.6f}")

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")