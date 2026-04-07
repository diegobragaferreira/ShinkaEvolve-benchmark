# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from numba import jit
import time
from sklearn.cluster import KMeans
import jax.numpy as jnp
import jax
from functools import partial

# Enable X64 precision for numerical stability
jax.config.update('jax_enable_x64', True)

@partial(jax.jit, static_argnums=(1,))
def compute_autoconvolution_jax(f_vals, n_steps):
    """Compute autoconvolution g = f*f using JAX for vectorized operations"""
    # Using JAX's convolution operation for better performance
    f = jnp.array(f_vals, dtype=jnp.float64)
    # Using full convolution mode
    g = jnp.convolve(f, f, mode='full')

    # Extract center portion
    middle_idx = n_steps - 1
    half_width = n_steps
    g_centered = g[middle_idx - half_width + 1 : middle_idx + half_width]

    return g_centered

@partial(jax.jit, static_argnums=(1,))
def compute_c2_jax(f_vals, n_steps):
    """Compute C2 using JAX for vectorized operations"""
    # Compute autoconvolution
    g_vals = compute_autoconvolution_jax(f_vals, n_steps)

    # Compute norms using JAX operations
    g_squared = g_vals ** 2
    g_abs = jnp.abs(g_vals)

    # ||g||₂² - sum of squares
    l2_squared = jnp.sum(g_squared)

    # ||g||₁ - sum of absolute values
    l1 = jnp.sum(g_abs)

    # ||g||∞ - maximum absolute value
    l_inf = jnp.max(g_abs)

    # Avoid division by zero
    l1_safe = jnp.where(l1 <= 1e-15, 1e-15, l1)
    l_inf_safe = jnp.where(l_inf <= 1e-15, 1e-15, l_inf)

    # Compute C2
    c2 = l2_squared / (l1_safe * l_inf_safe)

    return c2

@partial(jax.jit, static_argnums=(1,))
def compute_norms_jax(g_vals):
    """Compute norms using JAX vectorized operations"""
    # ||g||₂² - sum of squares
    l2_squared = jnp.sum(g_vals ** 2)

    # ||g||₁ - sum of absolute values
    l1 = jnp.sum(jnp.abs(g_vals))

    # ||g||∞ - maximum absolute value
    l_inf = jnp.max(jnp.abs(g_vals))

    return l2_squared, l1, l_inf

def evaluate_c2_jax(f_vals):
    """Evaluate C2 using JAX for vectorized computation"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        # Skip empty sequences
        if len(f_vals) == 0:
            return 0.0

        n_steps = len(f_vals)
        # Use JAX version for improved computation
        c2 = compute_c2_jax(f_vals, n_steps)
        return float(c2)
    except Exception as e:
        return 0.0

def compute_autoconvolution(f_vals):
    """Compute autoconvolution g = f*f using NumPy's convolve function for better efficiency"""
    # Using 'full' mode to get complete convolution result
    # This is equivalent to manual double sum but much faster and numerically stable
    g = np.convolve(f_vals, f_vals, mode='full')
    return g

def compute_norms(g_vals):
    """Compute L2, L1, and L-infinity norms correctly"""
    # L2 norm squared (using trapezoidal-like approximation)
    l2_squared = 0.0
    n = len(g_vals)
    if n >= 2:
        # For piecewise linear integration: integrate over intervals
        # Each interval contributes (h/3)(y1^2 + y1*y2 + y2^2) where h=1
        for i in range(n-1):
            y1 = g_vals[i]
            y2 = g_vals[i+1]
            l2_squared += (y1*y1 + y1*y2 + y2*y2) / 3.0

    # L1 norm (sum of absolute values - correctly computed)
    l1 = 0.0
    for val in g_vals:
        l1 += abs(val)

    # L-infinity norm (maximum absolute value)
    l_inf = np.max(np.abs(g_vals))

    return l2_squared, l1, l_inf

def evaluate_c2(f_vals):
    """Evaluate C2 for a given set of step heights"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        # Skip empty sequences
        if len(f_vals) == 0:
            return 0.0

        # Compute autoconvolution
        g_vals = compute_autoconvolution(f_vals)

        # Compute norms
        l2_squared, l1, l_inf = compute_norms(g_vals)

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

def objective_function_jax(x):
    """JAX-based objective function to minimize (negative C2)"""
    c2 = evaluate_c2_jax(x)
    return -c2

def sophisticated_initialization(n_steps):
    """Create a sophisticated initial population with pattern recognition"""
    # Create base pattern using sine wave modulation
    base_pattern = np.zeros(n_steps)
    for i in range(n_steps):
        # Sine wave pattern with varying frequencies
        freq = 0.05 + 0.1 * np.sin(i * 0.1)
        base_pattern[i] = 0.5 + 0.3 * np.sin(i * freq)

    # Add statistical clustering to create more structured patterns
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_input = base_pattern.reshape(-1, 1)
    clusters = kmeans.fit_predict(cluster_input)

    # Create clustered pattern
    clustered_pattern = np.zeros(n_steps)
    for i in range(n_steps):
        clustered_pattern[i] = base_pattern[i] * (0.8 + 0.4 * clusters[i] / 2.0)

    # Add noise for diversity
    noise = np.random.normal(0, 0.1, n_steps)
    init_params = clustered_pattern + noise

    # Ensure non-negative values
    init_params = np.clip(init_params, 0, None)

    return init_params

def generate_multi_scale_initialization(n_steps):
    """Generate initialization at multiple scales to enhance exploration"""
    # Create base initializations with different strategies
    initializations = []

    # Base initialization
    base_init = sophisticated_initialization(n_steps)
    initializations.append(base_init.copy())

    # Variation 1: Random shuffle with some preservation
    shuffled = base_init.copy()
    np.random.shuffle(shuffled)
    initializations.append(shuffled)

    # Variation 2: Smoothed version
    smoothed = base_init.copy()
    if len(smoothed) > 10:
        # Apply simple smoothing filter
        for i in range(1, len(smoothed)-1):
            smoothed[i] = 0.3 * smoothed[i-1] + 0.4 * smoothed[i] + 0.3 * smoothed[i+1]
    initializations.append(smoothed)

    # Variation 3: High-low alternating pattern from base
    alternating = np.zeros_like(base_init)
    for i in range(len(base_init)):
        if i % 2 == 0:
            alternating[i] = base_init[i] * (0.8 + 0.4 * np.random.random())
        else:
            alternating[i] = base_init[i] * (0.1 + 0.3 * np.random.random())
    initializations.append(alternating)

    # Variation 4: Multi-modal pattern
    multimodal = np.zeros_like(base_init)
    for i in range(len(base_init)):
        multimodal[i] = 0.3 + 0.4 * np.sin(i * 0.05) + 0.2 * np.sin(i * 0.2)
    initializations.append(multimodal)

    return initializations

def adaptive_differential_evolution(objective_func, bounds, n_steps, max_iter=None):
    """Adaptive differential evolution with dynamic parameters"""
    if max_iter is None:
        max_iter = 150

    # Adaptive population size based on dimensionality
    popsize = min(max(10, n_steps // 50), 20)

    # Multi-start with different seeds
    best_x = None
    best_c2 = -np.inf

    # Store timing for early termination
    start_time = time.time()

    # Multiple runs with different strategies
    seeds = [42, 123, 456, 789, 101]

    for seed in seeds:
        if time.time() - start_time > 85 * 0.8:
            break

        try:
            # Generate diverse initial population
            initial_populations = generate_multi_scale_initialization(n_steps)

            for i, x0 in enumerate(initial_populations):
                if time.time() - start_time > 85 * 0.9:
                    break

                try:
                    # Run differential evolution with this initial population
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        x0=x0,
                        seed=seed,
                        maxiter=min(max_iter, int(50 + 20 * (seed % 3))),  # Vary iterations per seed
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

        except Exception:
            continue

    return best_x if best_x is not None else np.array([0.5] * n_steps)

def improved_evolutionary_optimization():
    """Use improved differential evolution with adaptive parameters"""
    # Start with a larger initial size for better resolution
    n_steps = 1000

    # Define bounds for each parameter (step height)
    bounds = [(0.0, 2.0) for _ in range(n_steps)]

    # Multi-start with adaptive differential evolution
    best_x = adaptive_differential_evolution(objective_function, bounds, n_steps)

    return best_x

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using enhanced evolutionary optimization."""
    start_time = time.time()

    # Use improved evolutionary optimization to find optimal step heights
    optimized_params = improved_evolutionary_optimization()

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