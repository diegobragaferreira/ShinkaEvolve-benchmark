# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import time
from numba import jit, prange
import numba
from sklearn.cluster import KMeans

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS
MAX_TIME_SECONDS = 85  # Leave 5 seconds buffer

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation with FFT for better performance"""
    n = len(f_vals)

    # For FFT-based convolution, we need power-of-2 lengths for efficiency
    # Pad to next power of 2 for better FFT performance
    padded_len = 1
    while padded_len < 2 * n - 1:
        padded_len <<= 1

    # Pad the input
    padded_f = np.zeros(padded_len)
    padded_f[:n] = f_vals

    # Use FFT-based convolution
    # We compute FFT of f, square it, then inverse FFT
    f_fft = np.fft.fft(padded_f)
    g_fft = f_fft * f_fft
    g = np.real(np.fft.ifft(g_fft))

    # Return only the valid portion (first 2*n-1 elements)
    return g[:2*n-1]

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation with proper integration"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms using trapezoidal-like integration for L2^2
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

    # Compute L2^2 norm correctly using trapezoidal-like integration
    if len(g_vals) >= 2:
        # For piecewise linear integration: integrate over intervals
        # Each interval contributes (h/3)(y1^2 + y1*y2 + y2^2) where h=1 in convolution domain
        for i in range(len(g_vals) - 1):
            y1 = g_vals[i]
            y2 = g_vals[i + 1]
            g_l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0

    # Compute C2
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

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
    """Create a sophisticated initial step function using pattern recognition and statistical methods"""
    # Create base pattern using sine wave modulation
    base_pattern = np.zeros(dim)
    for i in range(dim):
        # Sine wave pattern with varying frequencies
        freq = 0.05 + 0.1 * np.sin(i * 0.1)
        base_pattern[i] = 0.5 + 0.3 * np.sin(i * freq)

    # Add statistical clustering to create more structured patterns
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_input = base_pattern.reshape(-1, 1)
    clusters = kmeans.fit_predict(cluster_input)

    # Create clustered pattern
    clustered_pattern = np.zeros(dim)
    for i in range(dim):
        clustered_pattern[i] = base_pattern[i] * (0.8 + 0.4 * clusters[i] / 2.0)

    # Add noise for diversity
    noise = np.random.normal(0, 0.1, dim)
    init_params = clustered_pattern + noise

    # Ensure non-negative values
    init_params = np.clip(init_params, 0, None)

    return init_params

def generate_multi_scale_initialization(dim):
    """Generate initialization at multiple scales to enhance exploration"""
    # Create base initializations with different strategies
    initializations = []

    # Base initialization
    base_init = sophisticated_initialization(dim)
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
        if time.time() - start_time > MAX_TIME_SECONDS * 0.8:
            break

        try:
            # Generate diverse initial population
            initial_populations = generate_multi_scale_initialization(n_steps)

            for i, x0 in enumerate(initial_populations):
                if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
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

def simulated_annealing_refinement(initial_params, max_iter=100):
    """Apply simulated annealing inspired refinement to improve solution quality"""
    current_params = np.array(initial_params)
    best_params = current_params.copy()
    best_c2 = -np.inf

    # Annealing schedule
    temp = 1.0
    cooling_rate = 0.95

    for iteration in range(max_iter):
        if iteration > 0 and iteration % 10 == 0:
            temp *= cooling_rate

        # Perturb current solution
        perturbed = current_params + np.random.normal(0, temp * 0.05, len(current_params))
        perturbed = np.clip(perturbed, 0, None)

        # Evaluate perturbed solution
        try:
            f_vals = perturbed
            if len(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                c2 = compute_c2_numba(g_vals)

                if c2 > best_c2:
                    best_c2 = c2
                    best_params = perturbed.copy()

                # Accept worse solutions with probability proportional to temperature
                if c2 < best_c2:
                    accept_prob = np.exp((c2 - best_c2) / max(temp, 1e-8))
                    if np.random.random() < accept_prob:
                        current_params = perturbed.copy()
                else:
                    current_params = perturbed.copy()
        except Exception:
            continue

    return best_params

def adaptive_optimization_strategy():
    """Use adaptive optimization strategy with multiple phases"""
    # Start with a reasonable initial size based on performance expectations
    n_steps = 1000  # Increased for better resolution

    # Define bounds for each parameter (step height)
    bounds = [(0.0, 1.0) for _ in range(n_steps)]

    # Objective function (minimize negative C2)
    def objective_func(x):
        return -compute_c2_numba(compute_autoconvolution_numba(np.clip(x, 0, None)))

    # Phase 1: Global search with adaptive DE
    best_x = adaptive_differential_evolution(objective_func, bounds, n_steps)

    # Phase 2: Local refinement with simulated annealing
    refined_x = simulated_annealing_refinement(best_x, max_iter=100)

    return refined_x

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using adaptive optimization."""
    start_time = time.time()

    # Use adaptive optimization strategy to find optimal step heights
    optimized_params = adaptive_optimization_strategy()

    # Ensure non-negative values
    f_values = np.maximum(optimized_params, 0).tolist()

    end_time = time.time()

    # Verify the result
    try:
        g_vals = compute_autoconvolution_numba(np.clip(optimized_params, 0, None))
        c2_value = compute_c2_numba(g_vals)
    except:
        c2_value = 0.0

    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"C2 achieved: {c2_value}")

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")