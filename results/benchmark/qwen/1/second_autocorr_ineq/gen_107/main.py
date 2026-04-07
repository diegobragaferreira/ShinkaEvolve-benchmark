# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
import time
from numba import njit
import random

@njit
def compute_autoconvolution_numba(f):
    """Compute autoconvolution g = f * f using numba JIT"""
    n = len(f)
    # Autoconvolution using discrete convolution
    g = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            g[i + j] += f[i] * f[j]

    # Trim to center portion (length n-1)
    offset = (n - 1) // 2
    g_trimmed = g[offset:(2*n-1)-offset]
    return g_trimmed

@njit
def compute_c2_numba(f):
    """Compute C2 value for given step function f using numba JIT"""
    if len(f) < 2:
        return 0.0

    # Compute autoconvolution
    g = compute_autoconvolution_numba(f)

    if len(g) == 0:
        return 0.0

    # Compute norms
    norm_l2_sq = 0.0
    norm_l1 = 0.0
    norm_inf = 0.0

    for i in range(len(g)):
        abs_g = abs(g[i])
        norm_l2_sq += abs_g * abs_g
        norm_l1 += abs_g
        if abs_g > norm_inf:
            norm_inf = abs_g

    # Avoid division by zero
    if norm_l1 < 1e-12 or norm_inf < 1e-12:
        return 0.0

    c2 = norm_l2_sq / (norm_l1 * norm_inf)
    return c2

def generate_multi_scale_initialization(dim):
    """Generate initial function with multiple scales for better exploration"""
    # Create a combination of different patterns
    init_params = np.zeros(dim)

    # Scale 1: Centered Gaussian pattern
    center = dim // 2
    sigma = dim / 6
    for i in range(dim):
        init_params[i] += 1.0 * np.exp(-0.5 * ((i - center) / sigma) ** 2)

    # Scale 2: Sinusoidal modulation
    for i in range(dim):
        init_params[i] += 0.3 * np.sin(2 * np.pi * i / (dim / 4))

    # Scale 3: Random component
    np.random.seed(42)
    rand_component = np.random.random(dim) * 0.2
    init_params += rand_component

    # Ensure non-negative values
    init_params = np.maximum(init_params, 0)

    # Normalize to reasonable range
    max_val = np.max(init_params)
    if max_val > 0:
        init_params = init_params / max_val * 1.5

    return init_params.tolist()

def adaptive_differential_evolution(dim, max_time=80):
    """Run differential evolution with adaptive population size"""
    start_time = time.time()

    # Adaptive population sizing based on problem dimension
    popsize = min(20, max(10, dim // 10))  # Start with smaller population for early search

    # Create initial population using multi-scale initialization
    bounds = [(0, 10) for _ in range(dim)]

    # Initial optimization with small population
    try:
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=min(100, 2000//popsize),
            popsize=popsize,
            seed=42,
            strategy='best1bin',
            disp=False
        )

        if not result.success:
            raise Exception("Differential evolution failed")

        best_x = result.x
        best_c2 = -objective_function(best_x)

        # Increase population size if early progress is good
        if best_c2 > 0.9:  # If already quite good, expand population
            popsize = min(30, max(popsize, 15))
            # Re-run with larger population
            try:
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=min(100, 2000//popsize),
                    popsize=popsize,
                    seed=42,
                    strategy='best1bin',
                    disp=False
                )

                if result.success:
                    final_x = result.x
                    final_c2 = -objective_function(final_x)
                    best_x = final_x if final_c2 > best_c2 else best_x
                    best_c2 = max(best_c2, final_c2)
            except:
                pass

        # Return the optimized parameters
        return best_x

    except Exception as e:
        # Fallback to simple initialization if something goes wrong
        return generate_multi_scale_initialization(dim)

def objective_function(x):
    """Objective function to minimize (negative C2)"""
    # x contains the step heights
    # Need to ensure non-negativity
    f = np.maximum(x, 0)
    c2 = compute_c2(f)
    return -c2  # Negative because we want to maximize

def stochastic_perturbation(params, perturbation_strength=0.05):
    """Add stochastic perturbation to prevent premature convergence"""
    # Add small random noise to parameters
    noise = np.random.normal(0, perturbation_strength, len(params))
    perturbed = params + noise

    # Ensure non-negativity and normalize
    perturbed = np.clip(perturbed, 0, None)

    # Normalize to preserve relative proportions
    max_val = np.max(perturbed)
    if max_val > 0:
        perturbed = perturbed / max_val * np.max(params)

    return perturbed

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using optimization."""
    # Set seed for reproducibility
    np.random.seed(42)
    global start_time
    start_time = time.time()

    # Try different configurations to find the best one
    best_c2 = 0.0
    best_f = []

    # Multiple attempts with different dimensions and strategies
    dimensions = [300, 500, 700, 1000]

    for dim in dimensions:
        if time.time() - start_time > 85:
            break

        try:
            # Strategy 1: Adaptive differential evolution
            params = adaptive_differential_evolution(dim, max_time=85)

            # Stochastic perturbation to escape local optima
            perturbed_params = stochastic_perturbation(params, 0.03)

            # Compute actual C2 value
            f_vals = np.clip(perturbed_params, 0, None)
            if len(f_vals) > 0:
                c2_value = compute_c2(f_vals)

                if c2_value > best_c2:
                    best_c2 = c2_value
                    best_f = perturbed_params.tolist()

        except Exception as e:
            continue

    # If nothing worked, fallback to simple construction
    if len(best_f) == 0:
        # Create a simple but reasonable step function
        n = 500
        best_f = [1.0] * n
        best_c2 = compute_c2(best_f)

    return best_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")