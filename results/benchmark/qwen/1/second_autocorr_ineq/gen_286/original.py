# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
from numba import jit
import time
from sklearn.cluster import KMeans

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Fast Numba-based autoconvolution computation"""
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
    """Fast Numba-based C2 computation with proper numerical integration"""
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

    # For L2 norm squared using proper trapezoidal integration
    if len(g_vals) >= 2:
        # Use trapezoidal rule for integration of g^2
        # Apply trapezoidal rule: sum of (h/2)*(g[i]^2 + g[i+1]^2) 
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq = 0.0
        for i in range(len(g_vals)):
            if i == 0 or i == len(g_vals) - 1:
                g_l2_sq += g_vals[i] * g_vals[i]
            else:
                g_l2_sq += 2.0 * g_vals[i] * g_vals[i]
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # Compute C2 with numerical stability checks
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def compute_autoconvolution_norms(f_values):
    """Compute the three norms needed for C2 calculation"""
    # Ensure non-negative values
    f = np.maximum(f_values, 0.0)
    
    # Compute autoconvolution g = f * f (discrete convolution)
    g = convolve(f, f, mode='full')
    
    # Extract the central portion that represents the main interval
    n = len(f)
    middle_idx = n - 1
    half_width = n
    
    # Take the central part of the convolution
    g_centered = g[middle_idx - half_width + 1 : middle_idx + half_width]
    
    # Compute the norms
    g_squared = g_centered ** 2
    g_abs = np.abs(g_centered)
    
    # ||g||₂² - integrate using trapezoidal rule manually for piecewise linear
    norm_g2_sq = np.sum(g_squared)
    
    # ||g||₁ - sum of absolute values
    norm_g1 = np.sum(g_abs)
    
    # ||g||∞ - maximum absolute value
    norm_ginf = np.max(g_abs)
    
    return norm_g2_sq, norm_g1, norm_ginf

def evaluate_c2(f_values):
    """Evaluate C2 for a given set of step heights"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_values, 0)

        # Skip empty sequences
        if len(f_vals) == 0:
            return 0.0

        # Compute autoconvolution using numba for speed
        g_vals = compute_autoconvolution_numba(f_vals)

        # Compute C2 using numba-optimized function
        c2 = compute_c2_numba(g_vals)
        
        return c2
    except Exception as e:
        return 0.0

def objective_function(x):
    """Objective function to minimize (negative C2)"""
    c2 = evaluate_c2(x)
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

def enhanced_evolutionary_optimization():
    """Use enhanced differential evolution with adaptive parameters"""
    # Start with a reasonable initial size
    n_steps = np.random.randint(500, 3000)

    # Define bounds for each parameter (step height)
    bounds = [(0.0, 3.0) for _ in range(n_steps)]

    # Multi-start with adaptive differential evolution
    best_x = adaptive_differential_evolution(objective_function, bounds, n_steps)

    return best_x

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using enhanced evolutionary optimization."""
    start_time = time.time()

    # Use enhanced evolutionary optimization to find optimal step heights
    optimized_params = enhanced_evolutionary_optimization()

    # Clip negative values to zero
    f_values = np.maximum(optimized_params, 0).tolist()

    end_time = time.time()

    # Verify the result with detailed computation
    c2_value = evaluate_c2(optimized_params)

    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"C2 achieved: {c2_value}")

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")