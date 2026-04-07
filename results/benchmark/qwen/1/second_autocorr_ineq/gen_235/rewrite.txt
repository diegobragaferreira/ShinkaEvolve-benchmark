# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
import time
from numba import jit
import jax.numpy as jnp
from jax import jit as jax_jit, grad
import jax
from functools import partial
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation"""
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
def compute_c2_norms_numba(g_vals):
    """Compute C2 norms using fast Numba implementation with proper integration"""
    if len(g_vals) == 0:
        return 0.0, 0.0, 0.0

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

    return g_l2_sq, g_l1, g_max

def evaluate_c2_manual(f_vals):
    """Manual C2 evaluation with numerical stability"""
    # Ensure non-negative values
    f_vals = np.maximum(f_vals, 0)

    if len(f_vals) == 0:
        return 0.0

    # Compute autoconvolution
    g_vals = compute_autoconvolution_numba(f_vals)

    # Compute norms
    l2_squared, l1, l_inf = compute_c2_norms_numba(g_vals)

    # Compute C2 with numerical stability
    epsilon = 1e-15
    if l1 > epsilon and l_inf > epsilon:
        c2 = l2_squared / (l1 * l_inf)
    else:
        c2 = 0.0
        
    return c2

@jax_jit
def evaluate_c2_jax(f_vals):
    """JAX-based C2 computation for gradient-based optimization"""
    try:
        # Ensure non-negative values with JAX operations
        f_vals = jnp.maximum(f_vals, 0.0)

        # Compute autoconvolution using JAX's convolution
        g_vals = jnp.convolve(f_vals, f_vals, mode='full')

        # Compute norms using JAX operations
        # L2^2 norm using trapezoidal-like integration  
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

        # Avoid division by zero
        l1_safe = jnp.where(l1 <= 1e-15, 1e-15, l1)
        l_inf_safe = jnp.where(l_inf <= 1e-15, 1e-15, l_inf)

        # Compute C2
        c2 = l2_squared / (l1_safe * l_inf_safe)
        return c2
    except:
        return 0.0

@partial(jax_jit, static_argnums=(0,))
def compute_gradients_jax(f_vals, num_steps):
    """Compute gradients of C2 w.r.t. input using JAX automatic differentiation"""
    # Create a wrapper for jax.grad
    def c2_wrapper(f_vals_vec):
        f_vals = f_vals_vec.reshape(num_steps)
        return evaluate_c2_jax(f_vals)
    
    # Compute gradients using automatic differentiation
    grad_fn = jax.grad(c2_wrapper)
    gradients = grad_fn(jnp.array(f_vals))
    return gradients

def gradient_based_optimization(initial_params, max_iter=200):
    """Gradient-based optimization using JAX automatic differentiation"""
    # Convert to JAX array for gradient computation
    x0 = jnp.array(initial_params)
    
    # Adaptive learning rate and iterations
    learning_rate = 0.01
    best_x = x0
    best_c2 = evaluate_c2_jax(x0)
    
    # Store history for convergence monitoring
    history = [best_c2]
    
    for iteration in range(max_iter):
        try:
            # Compute gradients
            grads = compute_gradients_jax(x0, len(initial_params))
            
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
                
            # Update for next iteration
            x0 = x_new
            
            # Reduce learning rate over time (adaptive decay)
            if iteration > 0 and iteration % 50 == 0:
                learning_rate *= 0.95
                
        except Exception as e:
            # If gradient computation fails, stop optimization
            break
            
    return np.array(best_x)

def generate_harmonic_initialization(n_steps):
    """Generate sophisticated harmonic initializations that tend to perform well"""
    # Create a sophisticated combination of harmonics
    initial = np.zeros(n_steps)
    
    # Create base with multiple frequencies and phases
    freqs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # Different frequencies
    amplitudes = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]  # Amplitudes for each frequency
    phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]  # Phases
    
    for i in range(n_steps):
        # Sum of cosines with different frequencies and phases
        pattern_sum = 0.0
        for freq, amp, phase in zip(freqs, amplitudes, phases):
            pattern_sum += amp * np.cos(i * freq + phase)
        # Add a decaying envelope to concentrate energy
        envelope = np.exp(-i / (n_steps * 0.3))
        initial[i] = max(0, 0.2 + 0.6 * pattern_sum * envelope)
    
    # Add some noise for diversity
    noise = np.random.normal(0, 0.05, n_steps)
    initial += noise
    
    # Apply smoothing to create less sharp transitions
    if n_steps > 10:
        smoothed = initial.copy()
        for i in range(1, n_steps-1):
            smoothed[i] = 0.3 * initial[i-1] + 0.4 * initial[i] + 0.3 * initial[i+1]
        initial = smoothed
    
    # Ensure non-negative values
    initial = np.maximum(initial, 0)
    
    return initial

def create_multi_scale_initializations(n_steps):
    """Create multiple diverse initializations at different scales"""
    initializations = []
    
    # Base harmonic initialization
    np.random.seed(42)
    initializations.append(generate_harmonic_initialization(n_steps))
    
    # Alternative pattern: alternating high/low with sinusoidal modulation
    alt_pattern = np.zeros(n_steps)
    for i in range(n_steps):
        # Create alternating pattern with sinusoidal variation
        if i % 4 == 0:
            alt_pattern[i] = 0.8 + 0.2 * np.sin(i * 0.2)
        elif i % 4 == 1:
            alt_pattern[i] = 0.6 + 0.2 * np.cos(i * 0.3)
        elif i % 4 == 2:
            alt_pattern[i] = 0.4 + 0.1 * np.sin(i * 0.1)
        else:
            alt_pattern[i] = 0.2 + 0.1 * np.cos(i * 0.4)
    initializations.append(alt_pattern)
    
    # Random pattern with structured elements
    structured_random = np.random.random(n_steps) * 0.7 + 0.2
    # Add some periodicity
    for i in range(n_steps):
        structured_random[i] += 0.1 * np.sin(i * 0.1)
    initializations.append(structured_random)
    
    # Smoothed version of random
    smoothed_random = np.random.random(n_steps)
    if n_steps > 10:
        for i in range(1, n_steps-1):
            smoothed_random[i] = 0.3 * smoothed_random[i-1] + 0.4 * smoothed_random[i] + 0.3 * smoothed_random[i+1]
    initializations.append(smoothed_random)
    
    return initializations

def advanced_multi_stage_optimization():
    """Multi-stage optimization combining global search and local refinement"""
    n_steps = 1000
    
    # Multi-scale initialization strategy
    initial_populations = create_multi_scale_initializations(n_steps)
    
    # Stage 1: Multi-start differential evolution with diverse initializations
    best_c2 = -np.inf
    best_solution = None
    
    start_time = time.time()
    time_limit = 85  # seconds
    
    # Try different initializations
    for i, x0 in enumerate(initial_populations):
        if time.time() - start_time > time_limit * 0.9:
            break
            
        try:
            # Differential evolution with adaptive population size
            bounds = [(0.0, 1.0) for _ in range(n_steps)]
            # Adaptive population size based on dimensionality
            popsize = max(10, min(25, n_steps // 40))
            
            # Run DE with this specific initialization
            result = differential_evolution(
                lambda x: -evaluate_c2_manual(x),
                bounds,
                x0=x0,
                seed=42 + i,
                maxiter=100,
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False,
                tol=1e-6
            )
            
            # Check if this is better than our current best
            if -result.fun > best_c2:
                best_c2 = -result.fun
                best_solution = result.x.copy()
                
        except Exception:
            continue
    
    # Stage 2: Local refinement using gradient-based optimization
    if best_solution is not None and time.time() - start_time < time_limit * 0.95:
        try:
            # Apply gradient-based optimization using JAX for better convergence
            refined_params = gradient_based_optimization(best_solution, max_iter=150)
            refined_c2 = evaluate_c2_manual(refined_params)
            
            # Check if refinement improved the result
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_solution = refined_params
        except Exception:
            pass
    
    # Return best solution found
    return best_solution if best_solution is not None else np.ones(n_steps) * 0.5

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using advanced multi-stage optimization."""
    start_time = time.time()

    # Use advanced multi-stage optimization
    optimized_params = advanced_multi_stage_optimization()

    # Ensure non-negative values
    f_values = np.maximum(optimized_params, 0).tolist()

    end_time = time.time()

    # Verify the result
    c2_value = evaluate_c2_manual(optimized_params)

    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"C2 achieved: {c2_value}")

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")