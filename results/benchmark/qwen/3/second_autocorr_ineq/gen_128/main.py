# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import time
from scipy.optimize import differential_evolution
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@njit
def compute_autoconvolution_norms_numba(f_values):
    """
    Compute the autoconvolution g = f*f and return its L2, L1, and L-infinity norms.
    Uses piecewise linear integration for L2 norm.
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Create convolution using discrete convolution (equivalent to autoconvolution)
    # The convolution will have length 2*n - 1
    g = np.zeros(2 * n - 1)

    # Compute autoconvolution manually for efficiency
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]

    # Compute norms
    # L2 norm squared
    l2_norm_squared = 0.0
    if len(g) >= 2:
        # Piecewise linear integration using trapezoidal rule approximation
        # For intervals, we use (h/3)(y1^2 + y1*y2 + y2^2) for each adjacent pair
        h = 1.0  # Since step size is normalized to 1 for simplicity
        for i in range(len(g) - 1):
            y1 = g[i]
            y2 = g[i+1]
            l2_norm_squared += (h/3.0) * (y1*y1 + y1*y2 + y2*y2)

    # L1 norm
    l1_norm = np.sum(np.abs(g)) / (len(g) + 1)  # Normalize by number of intervals

    # L-infinity norm
    l_inf_norm = np.max(np.abs(g))

    return l2_norm_squared, l1_norm, l_inf_norm

@njit
def calculate_c2_numba(l2_norm_squared, l1_norm, l_inf_norm):
    """Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    if l1_norm <= 1e-15 or l_inf_norm <= 1e-15:
        return 0.0
    return l2_norm_squared / (l1_norm * l_inf_norm)

def compute_c2(f_values):
    """Wrapper function to compute C2 from step function values"""
    try:
        l2, l1, l_inf = compute_autoconvolution_norms_numba(f_values)
        return calculate_c2_numba(l2, l1, l_inf)
    except Exception:
        return 0.0

def construct_geometric_initial_function(length=1000):
    """Construct an initial function with geometric patterns that tend to produce good C2 values."""
    # Create a function with alternating high and low regions
    # and some smooth transitions to avoid sharp discontinuities
    f_values = []
    for i in range(length):
        # Create a pattern that balances high and low values
        if i % 8 < 2:  # Two high peaks
            f_values.append(np.random.uniform(0.8, 1.0))
        elif i % 8 < 4:  # Two medium regions
            f_values.append(np.random.uniform(0.4, 0.8))
        elif i % 8 < 6:  # Two low regions
            f_values.append(np.random.uniform(0.1, 0.4))
        else:  # Two very low regions
            f_values.append(np.random.uniform(0.0, 0.2))

    # Add some smoothing to reduce numerical artifacts
    smoothed = []
    for i in range(len(f_values)):
        # Apply a simple moving average
        window_size = min(3, len(f_values))
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(f_values), i + window_size//2 + 1)
        avg = np.mean(f_values[start_idx:end_idx])
        smoothed.append(avg)

    return smoothed

def optimize_with_de(local_search=True):
    """Use differential evolution to optimize the function."""
    # Start with a good geometric initial function
    initial_f = construct_geometric_initial_function(1000)

    # Define bounds for each parameter (height of each step)
    bounds = [(0.0, 2.0) for _ in range(len(initial_f))]

    # Objective function for differential evolution
    def objective(params):
        # Clip negative values
        params = [max(0.0, p) for p in params]
        return -compute_c2(params)  # Negative because we want to maximize

    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False
    )

    # Return the best solution found
    optimized_params = [max(0.0, p) for p in result.x]
    return optimized_params

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Try optimization approach first
    try:
        # Use differential evolution to find a good solution
        f_values = optimize_with_de()

        # If we have time left, do a quick local refinement
        elapsed = time.time() - start_time
        if elapsed < 80:  # Still have time for refinement
            # Do some simple coordinate-wise refinement
            original_c2 = compute_c2(f_values)
            best_f = f_values.copy()
            best_c2 = original_c2

            # Try small perturbations to refine
            for _ in range(200):
                candidate = best_f.copy()
                # Perturb one element at a time
                idx = np.random.randint(len(candidate))
                candidate[idx] = max(0.0, candidate[idx] + np.random.normal(0, 0.05))

                candidate_c2 = compute_c2(candidate)
                if candidate_c2 > best_c2:
                    best_c2 = candidate_c2
                    best_f = candidate

            f_values = best_f
    except Exception as e:
        # Fallback to geometric initialization if optimization fails
        f_values = construct_geometric_initial_function(1000)

    # Ensure we don't exceed time limits
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for final processing
        # Return a reasonable solution
        return construct_geometric_initial_function(500)

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")