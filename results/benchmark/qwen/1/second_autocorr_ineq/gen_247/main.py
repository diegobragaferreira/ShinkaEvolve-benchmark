# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from numba import jit
import time
from scipy.sparse import csr_matrix
import warnings

# Suppress any warnings from sparse operations
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """
    Compute autoconvolution using fast Numba implementation
    """
    n = len(f_vals)
    # Output length for convolution
    output_length = 2 * n - 1
    g = np.zeros(output_length)

    # Direct convolution implementation
    for i in range(n):
        for j in range(n):
            pos = i + j
            if 0 <= pos < output_length:
                g[pos] += f_vals[i] * f_vals[j]

    return g

def compute_autoconvolution_fft(f_vals):
    """
    Compute autoconvolution using FFT for better performance on large inputs
    """
    n = len(f_vals)
    # For FFT-based convolution, we pad to next power of 2 for efficiency
    padded_len = 1
    while padded_len < 2 * n - 1:
        padded_len <<= 1

    # Pad the input
    padded_f = np.zeros(padded_len)
    padded_f[:n] = f_vals

    # Use FFT-based convolution
    f_fft = np.fft.fft(padded_f)
    g_fft = f_fft * f_fft
    g = np.real(np.fft.ifft(g_fft))

    # Return only the valid portion
    return g[:2*n-1]

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute L2, L1, and L-infinity norms efficiently"""
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

    # L1 norm (sum of absolute values divided by number of intervals)
    l1 = 0.0
    for val in g_vals:
        l1 += abs(val)
    l1 /= (n + 1)  # Approximate integral via piecewise constant approximation

    # L-infinity norm (maximum absolute value)
    l_inf = 0.0
    for val in g_vals:
        abs_val = abs(val)
        if abs_val > l_inf:
            l_inf = abs_val

    return l2_squared, l1, l_inf

def evaluate_c2(f_vals):
    """Evaluate C2 for a given set of step heights"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        # Skip empty sequences
        if len(f_vals) == 0:
            return 0.0

        # Use FFT for large inputs, else use Numba
        # FFT is more efficient for large arrays (> 100 elements typically)
        if len(f_vals) > 100:
            g_vals = compute_autoconvolution_fft(f_vals)
        else:
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

def create_structure_aware_initialization(n_steps):
    """Create initial population that is aware of the structure that tends to produce better results"""
    # Create a structure that combines multiple peaks and valleys to encourage good autoconvolution
    initial = np.zeros(n_steps)

    # Create multiple regions with different patterns
    region_size = max(1, n_steps // 8)

    for i in range(0, n_steps, region_size):
        region_end = min(i + region_size, n_steps)

        # Within each region, create alternating high/low pattern
        for j in range(i, region_end):
            # Create some periodicity in the pattern
            pattern_pos = j % (region_size // 4) if region_size > 4 else 0
            if pattern_pos < (region_size // 4):
                # High values
                initial[j] = 0.8 + 0.2 * np.random.random()
            else:
                # Low values
                initial[j] = 0.1 + 0.1 * np.random.random()

    # Add some noise to avoid perfect patterns that might get stuck
    noise = np.random.normal(0, 0.05, n_steps)
    initial += noise
    initial = np.maximum(initial, 0)

    return initial

def create_multiscale_initialization(n_steps):
    """Create multiple initializations at different scales to improve exploration"""
    initializations = []

    # Base scaled initialization
    base_init = create_structure_aware_initialization(n_steps)
    initializations.append(base_init)

    # Coarse grained version
    coarse_size = max(10, n_steps // 10)
    coarse_init = np.zeros(coarse_size)
    for i in range(coarse_size):
        coarse_init[i] = 0.5 + 0.3 * np.sin(i * 0.5)
    # Expand to full size
    expanded_coarse = np.interp(np.linspace(0, coarse_size-1, n_steps),
                               np.arange(coarse_size), coarse_init)
    initializations.append(expanded_coarse)

    # Random pattern
    random_init = np.random.random(n_steps) * 0.8 + 0.1
    initializations.append(random_init)

    # Wavelet-inspired pattern
    wavelet_pattern = np.zeros(n_steps)
    for i in range(n_steps):
        wavelet_pattern[i] = 0.5 + 0.3 * np.sin(i * 0.1) * np.exp(-i / (n_steps * 0.5))
    initializations.append(wavelet_pattern)

    return initializations

def advanced_evolutionary_optimization():
    """Advanced evolutionary optimization with multi-scale initialization and adaptive parameters"""
    # Start with larger initial size for better resolution
    n_steps = 1000

    # Define bounds for each parameter (step height)
    bounds = [(0.0, 1.0) for _ in range(n_steps)]

    # Create multiple initializations
    initial_populations = create_multiscale_initialization(n_steps)

    best_c2 = -np.inf
    best_solution = None

    # Store timing for early termination
    start_time = time.time()
    time_limit = 85  # seconds

    # Try each initialization
    for i, x0 in enumerate(initial_populations):
        if time.time() - start_time > time_limit * 0.95:
            break

        try:
            # Adaptive population size based on iteration count
            popsize = min(max(15, n_steps // 50), 30)

            # Run differential evolution with this specific initialization
            result = differential_evolution(
                objective_function,
                bounds,
                x0=x0,
                seed=42 + i,
                maxiter=150,
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False,
                tol=1e-6,
                callback=None
            )

            if -result.fun > best_c2:
                best_c2 = -result.fun
                best_solution = result.x.copy()

        except Exception:
            continue

    # If we failed to find anything, return a default
    if best_solution is None:
        best_solution = np.ones(n_steps) * 0.5

    return best_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using advanced evolutionary optimization."""
    start_time = time.time()

    # Use advanced evolutionary optimization to find optimal step heights
    optimized_params = advanced_evolutionary_optimization()

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