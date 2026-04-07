# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import time
import jax.numpy as jnp
from jax import jit as jax_jit, grad
import jax

# Global variables for time management
MAX_TIME_SECONDS = 85  # Leave 5 seconds buffer

def compute_autoconvolution_numpy(f_vals):
    """Compute autoconvolution using efficient NumPy convolution"""
    return np.convolve(f_vals, f_vals, mode='full')

def compute_norms_piecewise_linear(g_vals):
    """Compute norms using piecewise linear integration"""
    g_vals = np.array(g_vals)

    # L2 norm squared using trapezoidal-like integration
    l2_squared = 0.0
    if len(g_vals) >= 2:
        # For piecewise linear integration: integrate over intervals
        # Each interval contributes (h/3)(y1^2 + y1*y2 + y2^2) where h=1
        for i in range(len(g_vals)-1):
            y1 = g_vals[i]
            y2 = g_vals[i+1]
            l2_squared += (y1*y1 + y1*y2 + y2*y2) / 3.0

    # L1 norm (sum of absolute values divided by number of intervals)
    l1 = np.sum(np.abs(g_vals)) / (len(g_vals) + 1) if len(g_vals) + 1 > 0 else 0.0

    # L-infinity norm (maximum absolute value)
    l_inf = np.max(np.abs(g_vals))

    return l2_squared, l1, l_inf

@jax_jit
def evaluate_c2_jax(f_vals):
    """JAX version of C2 computation for improved performance"""
    try:
        # Ensure non-negative values
        f_vals = jnp.maximum(f_vals, 0.0)

        # Compute autoconvolution
        g_vals = jnp.convolve(f_vals, f_vals, mode='full')

        # Compute norms using JAX operations
        l2_squared = 0.0
        if len(g_vals) >= 2:
            # Trapezoidal-like integration
            for i in range(len(g_vals)-1):
                y1 = g_vals[i]
                y2 = g_vals[i+1]
                l2_squared += (y1*y1 + y1*y2 + y2*y2) / 3.0

        # L1 norm
        l1 = jnp.sum(jnp.abs(g_vals)) / (len(g_vals) + 1) if len(g_vals) + 1 > 0 else 0.0

        # L-infinity norm
        l_inf = jnp.max(jnp.abs(g_vals))

        # Avoid division by zero
        l1_safe = jnp.where(l1 <= 1e-15, 1e-15, l1)
        l_inf_safe = jnp.where(l_inf <= 1e-15, 1e-15, l_inf)

        # Compute C2
        c2 = l2_squared / (l1_safe * l_inf_safe)
        return c2
    except:
        return 0.0

def evaluate_c2(f_vals):
    """Evaluate C2 for a given set of step heights with fallback to numpy version"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        # Skip empty sequences
        if len(f_vals) == 0:
            return 0.0

        # Compute autoconvolution
        g_vals = compute_autoconvolution_numpy(f_vals)

        # Compute norms
        l2_squared, l1, l_inf = compute_norms_piecewise_linear(g_vals)

        # Avoid division by zero
        if l1 <= 1e-15 or l_inf <= 1e-15:
            return 0.0

        # Compute C2
        c2 = l2_squared / (l1 * l_inf)
        return c2
    except Exception as e:
        return 0.0

def sophisticated_initialization(n_steps):
    """Create a sophisticated initial population with multi-scale features"""
    # Create initial pattern with multiple scales
    initial = np.zeros(n_steps)

    # Large scale structure (fewer segments)
    scale1_size = max(1, n_steps // 8)
    for i in range(0, n_steps, scale1_size):
        # Alternate high/low values for large regions
        for j in range(i, min(i + scale1_size, n_steps)):
            if (i // scale1_size) % 2 == 0:
                initial[j] = 1.0 + np.random.random() * 0.5
            else:
                initial[j] = 0.1 + np.random.random() * 0.3

    # Medium scale structure (more segments)
    scale2_size = max(1, n_steps // 16)
    for i in range(0, n_steps, scale2_size):
        # Add some variation within medium segments
        for j in range(i, min(i + scale2_size, n_steps)):
            if np.random.random() < 0.3:  # 30% chance of modification
                if initial[j] > 0.5:
                    initial[j] = initial[j] * (0.8 + np.random.random() * 0.4)  # Slight reduction
                else:
                    initial[j] = initial[j] * (0.9 + np.random.random() * 0.3)  # Slight increase

    # Add some noise
    noise_level = 0.1
    initial += np.random.normal(0, noise_level, n_steps)

    # Ensure non-negative
    initial = np.maximum(initial, 0)

    return initial

def generate_multi_scale_initialization(n_steps):
    """Generate multiple diversified initial populations"""
    initial_populations = []

    # Generate several different initializations
    for i in range(3):
        np.random.seed(42 + i)  # Different seeds for variety
        population = sophisticated_initialization(n_steps)
        initial_populations.append(population)

    return initial_populations

def adaptive_differential_evolution(objective_func, bounds, n_steps, max_iter=None):
    """Adaptive differential evolution with dynamic popsize"""
    if max_iter is None:
        max_iter = 150

    # Adaptive population size: start small and increase if needed
    popsize = min(max(10, n_steps // 50), 20)

    # Multi-start
    best_x = None
    best_c2 = -np.inf

    # Store timing for early termination
    start_time = time.time()

    for run in range(3):  # Multiple runs
        if time.time() - start_time > MAX_TIME_SECONDS * 0.8:
            break

        # Generate diverse initial population
        initial_populations = generate_multi_scale_initialization(n_steps)

        # Try each initial population
        for i, x0 in enumerate(initial_populations):
            # Check time remaining
            if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
                break

            try:
                # Run differential evolution with this initial population
                result = differential_evolution(
                    objective_func,
                    bounds,
                    x0=x0,
                    seed=run * 10 + i,
                    maxiter=min(max_iter, int(50 + 30 * run)),  # Decreasing iterations in later runs
                    popsize=popsize,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False,
                    tol=1e-6
                )

                if -result.fun > best_c2:
                    best_c2 = -result.fun
                    best_x = result.x.copy()

            except Exception:
                continue

    return best_x if best_x is not None else np.array([0.5] * n_steps)

def advanced_refinement_strategy(x0, objective_func, bounds, n_steps):
    """Apply advanced refinement after global optimization"""
    # First do a few iterations with higher precision
    refined_result = differential_evolution(
        objective_func,
        bounds,
        x0=x0,
        seed=999,
        maxiter=50,
        popsize=15,
        mutation=(0.8, 1.0),
        recombination=0.9,
        disp=False,
        tol=1e-8
    )

    # Add some local gradient-based refinement if possible
    # Since we're dealing with discrete steps, we apply small perturbations
    try:
        current_x = refined_result.x.copy()
        current_c2 = -refined_result.fun

        # Try small random perturbations around current solution
        for _ in range(20):
            perturbed_x = current_x.copy()

            # Perturb a few random elements
            indices = np.random.choice(n_steps, size=max(1, n_steps // 10), replace=False)
            for idx in indices:
                # Small random change within bounds
                delta = np.random.normal(0, 0.05)
                perturbed_x[idx] = np.clip(perturbed_x[idx] + delta, 0.0, 1.0)

            new_c2 = -objective_func(perturbed_x)
            if new_c2 > current_c2:
                current_x = perturbed_x
                current_c2 = new_c2

        return current_x
    except Exception:
        return refined_result.x

def evolutionary_optimization():
    """Enhanced evolutionary optimization routine"""
    # Start with a reasonable initial size
    n_steps = 1000  # Increased from 500 for better resolution

    # Define bounds for each parameter (step height)
    bounds = [(0.0, 1.0) for _ in range(n_steps)]

    # Objective function (minimize negative C2)
    def objective_func(x):
        return -evaluate_c2(x)

    # Run adaptive differential evolution
    best_x = adaptive_differential_evolution(objective_func, bounds, n_steps)

    # Apply refinement
    refined_x = advanced_refinement_strategy(best_x, objective_func, bounds, n_steps)

    return refined_x

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using enhanced evolutionary optimization."""
    start_time = time.time()

    # Use enhanced evolutionary optimization to find optimal step heights
    optimized_params = evolutionary_optimization()

    # Ensure non-negative values
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