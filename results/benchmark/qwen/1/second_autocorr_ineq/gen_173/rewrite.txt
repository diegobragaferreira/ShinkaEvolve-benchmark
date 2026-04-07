# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
import time
from numba import jit, prange
import numba
import jax.numpy as jnp
from jax import jit as jax_jit, grad
import jax
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS
MAX_TIME_SECONDS = 85

@jit(nopython=True, cache=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation with improved efficiency"""
    n = len(f_vals)
    if n == 0:
        return np.array([])
    
    # Convolution result has length 2*n-1
    g_len = 2 * n - 1
    g = np.zeros(g_len, dtype=np.float64)

    # Compute convolution manually for efficiency
    # Use prange for parallel execution if available
    for i in range(n):
        f_i = f_vals[i]
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_i * f_vals[j]

    return g

@jit(nopython=True, cache=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation with proper integration and numerical stability"""
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

    # Compute C2 with proper numerical stability
    epsilon = 1e-15
    if g_l1 > epsilon and g_max > epsilon:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def evaluate_c2_manual(f_vals):
    """Manual C2 evaluation with numerical stability and error handling"""
    # Ensure non-negative values
    f_vals = np.maximum(f_vals, 0)

    if len(f_vals) == 0:
        return 0.0

    try:
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)

        # Compute C2
        c2 = compute_c2_numba(g_vals)
        return c2
    except Exception:
        return 0.0

@jax_jit
def evaluate_c2_jax(f_vals):
    """JAX-based C2 computation for gradient-based optimization with improved stability"""
    try:
        # Ensure non-negative values
        f_vals = jnp.maximum(f_vals, 0.0)

        # Compute autoconvolution using JAX's convolution
        g_vals = jnp.convolve(f_vals, f_vals, mode='full')

        # Compute norms using JAX operations
        # L2^2 norm using piecewise linear integration
        l2_squared = 0.0
        if len(g_vals) >= 2:
            for i in range(len(g_vals)-1):
                y1 = g_vals[i]
                y2 = g_vals[i+1]
                l2_squared += (y1*y1 + y1*y2 + y2*y2) / 3.0

        # L1 norm (piecewise constant approximation)
        l1 = jnp.sum(jnp.abs(g_vals)) / (len(g_vals) + 1) if len(g_vals) + 1 > 0 else 0.0

        # L-infinity norm
        l_inf = jnp.max(jnp.abs(g_vals))

        # Avoid division by zero with safer thresholds
        epsilon = 1e-15
        l1_safe = jnp.where(l1 <= epsilon, epsilon, l1)
        l_inf_safe = jnp.where(l_inf <= epsilon, epsilon, l_inf)

        # Compute C2
        c2 = l2_squared / (l1_safe * l_inf_safe)
        return c2
    except Exception:
        return 0.0

@partial(jax_jit, static_argnums=(0,))
def compute_gradients_jax_cached(f_vals, num_steps):
    """Compute gradients of C2 w.r.t. input using JAX with caching"""
    # Create a wrapper for jax.grad
    def c2_wrapper(f_vals_vec):
        f_vals = f_vals_vec.reshape(num_steps)
        return evaluate_c2_jax(f_vals)
    
    # Compute gradients
    grad_fn = jax.grad(c2_wrapper)
    gradients = grad_fn(jnp.array(f_vals))
    return gradients

def adaptive_gradient_optimization(initial_params):
    """Adaptive gradient-based optimization approach with improved stability"""
    # Convert to JAX array for gradient computation
    x0 = jnp.array(initial_params)
    
    # Adaptive learning rate and iterations
    learning_rate = 0.01
    max_iter = 150
    
    # Track best solution
    best_x = x0
    best_c2 = evaluate_c2_jax(x0)
    
    # Store history for convergence monitoring
    history = [best_c2]
    
    # Use early stopping criteria
    patience = 0
    best_improvement = 0
    min_improvement = 1e-6
    
    for iteration in range(max_iter):
        # Check if we've been running too long
        if iteration > 0 and iteration % 10 == 0:
            # Check time limit
            if time.time() > (time.time() - time.time() + MAX_TIME_SECONDS * 0.95):
                break
                
        # Compute gradients
        try:
            grads = compute_gradients_jax_cached(x0, len(initial_params))
            
            # Update parameters with gradient ascent (since we want to maximize C2)
            x_new = x0 + learning_rate * grads
            
            # Project back to feasible space [0, 1]
            x_new = jnp.clip(x_new, 0.0, 1.0)
            
            # Evaluate new solution
            new_c2 = evaluate_c2_jax(x_new)
            
            # Accept improvement
            if new_c2 > best_c2:
                best_c2 = new_c2
                best_x = x_new
                history.append(best_c2)
                best_improvement = 0
            else:
                best_improvement += 1
            
            # Update for next iteration
            x0 = x_new
            
            # Reduce learning rate over time
            learning_rate *= 0.995
            
            # Early stopping based on minimal improvement
            if best_improvement > 10:
                break
                
        except Exception as e:
            # If gradient computation fails, fall back to differential evolution
            break
            
    return np.array(best_x)

def generate_harmonic_initialization(n_steps):
    """Generate initial step function using advanced harmonic patterns that are known to work well"""
    # Create a combination of harmonics that produce good autoconvolution properties
    initial = np.zeros(n_steps)
    
    # Base pattern: combined sine and cosine waves with multiple frequencies
    frequencies = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    amplitudes = [0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    
    # Generate base signal with multiple harmonics
    for i in range(n_steps):
        pattern_sum = 0.0
        for freq, amp in zip(frequencies, amplitudes):
            pattern_sum += amp * np.cos(i * freq) + amp * np.sin(i * freq * 1.3)
        initial[i] = max(0, 0.2 + 0.4 * pattern_sum)
    
    # Add some structured noise for diversity
    noise = np.random.normal(0, 0.03, n_steps)
    initial += noise
    
    # Ensure non-negative
    initial = np.maximum(initial, 0)
    
    # Apply smoothing to reduce sharp transitions
    if n_steps > 10:
        smoothed = initial.copy()
        for i in range(1, n_steps-1):
            smoothed[i] = 0.2 * initial[i-1] + 0.6 * initial[i] + 0.2 * initial[i+1]
        initial = smoothed
    
    # Apply a final refinement to make sure it's reasonable
    initial = np.clip(initial, 0, 1.0)
    
    return initial

def generate_multiscale_initialization(n_steps):
    """Generate multiple diverse initializations for better exploration"""
    initializations = []
    
    # Harmonic pattern
    np.random.seed(42)
    initializations.append(generate_harmonic_initialization(n_steps))
    
    # Alternating pattern
    np.random.seed(123)
    alternating = np.zeros(n_steps)
    for i in range(n_steps):
        if i % 2 == 0:
            alternating[i] = np.random.uniform(0.7, 1.0)
        else:
            alternating[i] = np.random.uniform(0.0, 0.3)
    initializations.append(alternating)
    
    # Gaussian pattern
    np.random.seed(456)
    gaussian_pattern = np.random.normal(0.5, 0.15, n_steps)
    gaussian_pattern = np.clip(gaussian_pattern, 0, 1.0)
    initializations.append(gaussian_pattern)
    
    # Spike pattern
    spike_pattern = np.zeros(n_steps)
    for i in range(0, n_steps, 50):
        if i + 10 < n_steps:
            spike_pattern[i:i+10] = np.random.uniform(0.8, 1.0)
    initializations.append(spike_pattern)
    
    return initializations

def smart_hybrid_optimization_strategy():
    """Smart hybrid approach combining global and local optimization with improved efficiency"""
    # Start with a reasonable initial size
    n_steps = 1000
    
    # Phase 1: Multi-start global search with diverse initializations
    best_c2 = -np.inf
    best_params = None
    start_time = time.time()
    
    # Generate multiple initializations
    initializations = generate_multiscale_initialization(n_steps)
    
    # Run evolutionary optimization on each initialization
    for i, x0 in enumerate(initializations):
        if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
            break
            
        try:
            # Use differential evolution with adaptive parameters
            bounds = [(0.0, 1.0) for _ in range(n_steps)]
            
            # Dynamic population size based on problem complexity
            popsize = max(15, min(25, n_steps // 40))
            
            result = differential_evolution(
                lambda x: -evaluate_c2_manual(x),
                bounds,
                x0=x0,
                seed=i + 42,
                maxiter=80,  # Reduced iterations to save time
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False,
                tol=1e-6
            )
            
            c2 = -result.fun
            if c2 > best_c2:
                best_c2 = c2
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Phase 2: Local refinement using gradient-based optimization
    if best_params is not None and time.time() - start_time < MAX_TIME_SECONDS * 0.95:
        try:
            refined_params = adaptive_gradient_optimization(best_params)
            refined_c2 = evaluate_c2_manual(refined_params)
            
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_params = refined_params
        except Exception:
            pass
    
    # Final fallback if nothing was found
    if best_params is None:
        # Create a reasonable fallback pattern
        best_params = np.ones(n_steps) * 0.5
    
    return best_params

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using improved hybrid optimization."""
    start_time = time.time()
    
    # Use improved hybrid optimization strategy
    optimized_params = smart_hybrid_optimization_strategy()
    
    # Ensure non-negative values and convert to list
    f_values = np.maximum(optimized_params, 0).tolist()
    
    end_time = time.time()
    
    # Final verification of C2 value
    try:
        final_c2 = evaluate_c2_manual(optimized_params)
    except:
        final_c2 = 0.0
    
    total_time = end_time - start_time
    print(f"Optimization completed in {total_time:.2f} seconds")
    print(f"C2 achieved: {final_c2}")
    
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")