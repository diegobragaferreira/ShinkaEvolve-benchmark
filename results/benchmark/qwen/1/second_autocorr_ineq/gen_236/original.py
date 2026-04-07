# EVOLVE-BLOCK-START
import numpy as np
import time
from numba import jit, prange
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
from jax.scipy.signal import convolve as jax_convolve
import math
from scipy.optimize import differential_evolution
from collections import deque

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation"""
    n = len(f_vals)
    if n == 0:
        return np.array([])

    # For autoconvolution f*f, we want the central portion of the full convolution
    g_len = 2 * n - 1
    g = np.zeros(g_len)

    # Compute convolution manually for efficiency
    for i in range(n):
        f_i = f_vals[i]
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_i * f_vals[j]

    # Take the central portion (for autoconvolution)
    center_start = (n - 1) // 2
    center_end = center_start + (n - 1)
    if center_end <= len(g):
        g_trimmed = g[center_start:center_end]
    else:
        g_trimmed = g[center_start:]

    return g_trimmed

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L2 norm squared (piecewise quadratic integration - more accurate)
    for i in range(len(g_vals) - 1):
        val1 = g_vals[i]
        val2 = g_vals[i+1]
        # Trapezoidal rule with quadratic approximation: ∫(a*x+b)^2 dx = (1/3)*[(a*x2+b)^3 - (a*x1+b)^3] / a
        # But in our case, for linear segments, this becomes: (h/3)(y1^2 + y1*y2 + y2^2)
        h = STEP_WIDTH
        g_l2_sq += (h/3) * (val1*val1 + val1*val2 + val2*val2)

    # For L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (max absolute value)
    for i in range(len(g_vals)):
        if abs(g_vals[i]) > g_max:
            g_max = abs(g_vals[i])

    # Compute C2
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def sophisticated_multi_scale_initialization(dim):
    """Create initial parameters using multi-scale patterns for better exploration"""
    # Create several different patterns and select the best
    patterns = []
    
    # Pattern 1: Multi-peak Gaussian structure
    x = np.linspace(0, 1, dim)
    pattern1 = np.zeros(dim)
    for peak_idx in range(3):
        center = 0.2 + 0.3 * peak_idx
        width = 0.1 + 0.05 * np.random.random()
        height = 0.8 + 0.4 * np.random.random()
        pattern1 += height * np.exp(-((x - center)**2) / (2 * width**2))
    patterns.append(pattern1.tolist())
    
    # Pattern 2: Alternating high/low pattern
    pattern2 = []
    for i in range(dim):
        if i % 3 == 0:
            pattern2.append(1.5 + 0.3 * np.random.random())
        elif i % 3 == 1:
            pattern2.append(0.3 + 0.2 * np.random.random())
        else:
            pattern2.append(0.8 + 0.2 * np.random.random())
    patterns.append(pattern2)
    
    # Pattern 3: Sine wave modulation with decay
    pattern3 = []
    for i in range(dim):
        pos = i / (dim - 1) if dim > 1 else 0.5
        # Create a pattern with multiple frequencies and decay
        freq_component = 0.5 * np.sin(16 * np.pi * pos) + 0.3 * np.sin(8 * np.pi * pos)
        decay = np.exp(-5 * pos)  # Exponential decay
        val = 0.5 + 0.5 * freq_component * decay + 0.1 * np.random.random()
        pattern3.append(max(0.0, val))
    patterns.append(pattern3)
    
    # Pattern 4: Power law decay with oscillation
    pattern4 = []
    for i in range(dim):
        pos = i / (dim - 1) if dim > 1 else 0.5
        # Power law with oscillation
        power_val = pos**(0.5)  # Sub-linear decay
        oscillation = 0.2 * np.sin(12 * np.pi * pos)
        val = 0.8 * power_val + 0.2 * oscillation + 0.1
        pattern4.append(max(0.0, val))
    patterns.append(pattern4)
    
    # Select the best pattern by evaluating it
    best_pattern = patterns[0]
    best_score = -1e10
    
    for pattern in patterns:
        try:
            # Evaluate the pattern quickly
            f_vals = np.clip(pattern, 0, None)
            g_vals = compute_autoconvolution_numba(f_vals)
            c2 = compute_c2_numba(g_vals)
            if c2 > best_score:
                best_score = c2
                best_pattern = pattern
        except:
            continue
    
    return best_pattern

def jax_autograd_optimization(initial_params, max_iter=800):
    """Optimization using JAX automatic differentiation with adaptive parameters"""
    
    # Convert to JAX array
    x0 = jnp.array(initial_params)
    
    # Define the objective in JAX
    @jax_jit
    def jax_objective(params):
        # Convert back to numpy for our computations
        f_vals_np = np.array(params)
        f_vals_np = np.clip(f_vals_np, 0, None)
        
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals_np)
        
        # Compute C2
        c2 = compute_c2_numba(g_vals)
        
        return -c2  # Negative because we minimize
    
    # Get the gradient function
    grad_fn = grad(jax_objective)
    
    # Basic gradient descent with adaptive step size
    current_x = x0
    best_x = current_x
    best_value = jax_objective(current_x)
    
    # Adaptive learning rate
    learning_rate = 0.001
    momentum = 0.95  # Increased momentum for stable convergence
    velocity = jnp.zeros_like(current_x)
    
    # Convergence monitoring
    recent_losses = deque(maxlen=10)
    
    for i in range(max_iter):
        # Compute gradient
        grad_val = grad_fn(current_x)
        
        # Update with momentum
        velocity = momentum * velocity - learning_rate * grad_val
        current_x = current_x + velocity
        
        # Clip to non-negative
        current_x = jnp.maximum(current_x, 0)
        
        # Evaluate objective
        current_value = jax_objective(current_x)
        
        # Track recent losses for early stopping
        recent_losses.append(current_value)
        
        # Update best solution
        if current_value < best_value:
            best_value = current_value
            best_x = current_x
    
    return np.array(best_x)

def adaptive_evolutionary_strategy():
    """Improved evolutionary optimization with adaptive parameters and convergence detection"""
    # Try different initialization strategies with varying dimensions
    best_c2 = -np.inf
    best_params = None
    
    # Different initialization strategies
    strategies = [
        ("multiscale", lambda dim: sophisticated_multi_scale_initialization(dim)),
    ]
    
    # Set up timing
    start_time = time.time()
    time_limit = 85  # seconds
    
    # Iterative optimization with adaptive dimensionality
    iteration = 0
    max_iterations = 20
    
    while iteration < max_iterations and (time.time() - start_time < time_limit - 2):
        iteration += 1
        
        for strategy_name, init_func in strategies:
            for seed in [42 + iteration * 10 + s for s in range(3)]:
                try:
                    np.random.seed(seed)
                    
                    # Adaptive dimension for this iteration
                    base_dim = 500 + iteration * 100
                    dim = max(200, min(1500, base_dim + np.random.randint(-200, 200)))
                    
                    # Generate initial parameter
                    initial_params = init_func(dim)
                    
                    # Optimize using JAX auto-differentiation
                    optimized_params = jax_autograd_optimization(initial_params, max_iter=300)
                    
                    # Evaluate final result
                    f_vals = np.clip(optimized_params, 0, None)
                    g_vals = compute_autoconvolution_numba(f_vals)
                    c2 = compute_c2_numba(g_vals)
                    
                    if c2 > best_c2:
                        best_c2 = c2
                        best_params = optimized_params.copy()
                        
                except Exception as e:
                    continue
                    
                # Check time limit
                if time.time() - start_time > time_limit - 2:
                    break
        
        # Early stopping condition - if no significant improvement for several iterations
        if iteration > 5:
            # Simple check: if last few iterations didn't improve much
            pass  # Continue for now based on time limit
            
    return best_params, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Main optimization loop
    best_c2 = -np.inf
    best_params = None

    # Try multiple optimization attempts
    for attempt in range(3):
        try:
            # Use our adaptive evolutionary approach
            params, c2 = adaptive_evolutionary_strategy()

            if params is not None and c2 > best_c2:
                best_c2 = c2
                best_params = params.copy()
        except Exception as e:
            continue

        # Early exit if we've been running too long
        if time.time() - start_time > 85:  # Leave buffer for cleanup
            break

    # If no valid parameters found, return default
    if best_params is None:
        return [0.5] * 100

    # Final check and conversion to list
    final_f_vals = np.clip(best_params, 0, None)
    return final_f_vals.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")