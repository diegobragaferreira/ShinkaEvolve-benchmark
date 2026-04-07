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

    # For L2 norm squared (trapezoidal integration)
    for i in range(len(g_vals) - 1):
        val1 = g_vals[i]
        val2 = g_vals[i+1]
        # Trapezoidal rule with quadratic approximation
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

# Custom HMC implementation using JAX for automatic differentiation
class HamiltonianMC:
    def __init__(self, target_log_prob_fn, step_size=0.01, num_leapfrog_steps=10):
        self.target_log_prob_fn = target_log_prob_fn
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps

    @staticmethod
    @jax_jit
    def _leapfrog_update(position, momentum, grad_log_prob, step_size, num_steps):
        """Perform leapfrog integration for HMC"""
        # Initialize position and momentum
        new_position = position
        new_momentum = momentum

        # Perform leapfrog steps
        for _ in range(num_steps):
            # Half step for momentum
            new_momentum = new_momentum - 0.5 * step_size * grad_log_prob(new_position)

            # Full step for position
            new_position = new_position + step_size * new_momentum

            # Recompute gradient
            grad_log_prob_new = grad_log_prob(new_position)

            # Half step for momentum
            new_momentum = new_momentum - 0.5 * step_size * grad_log_prob_new

        return new_position, new_momentum

    def sample(self, initial_position, num_samples, rng_key):
        """Sample from the target distribution using HMC"""
        samples = []
        current_position = initial_position

        # Simple approach: perform a few leapfrog steps to get a better sample
        # We'll create a modified HMC that focuses on optimization rather than sampling
        return current_position

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

def sophisticated_geometric_initialization(dim):
    """Create an initialization that places steps geometrically to promote high convolution values"""
    # Create a pattern that starts with high values and tapers off
    init_params = []

    # Create a bell-curve like pattern centered in the middle
    center_point = 0.5
    decay_factor = 8.0
    peak_height = 1.0
    
    for i in range(dim):
        position = i / (dim - 1) if dim > 1 else 0.5
        # Gaussian-like shape with exponential decay
        gaussian_val = peak_height * np.exp(-decay_factor * (position - center_point)**2)
        
        # Add some structured variation for complexity
        struct_val = 0.1 * np.sin(8 * np.pi * position) + 0.15 * np.cos(6 * np.pi * position)
        final_val = max(0, gaussian_val + struct_val + 0.1)
        
        init_params.append(final_val)

    # Normalize to prevent extreme values
    if init_params:
        max_val = max(init_params)
        if max_val > 0:
            init_params = [x/max_val * 1.0 for x in init_params]

    return init_params

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
    
    # Pattern 3: Geometric pattern with exponential decay
    pattern3 = []
    for i in range(dim):
        pos = i / (dim - 1) if dim > 1 else 0.5
        val = np.exp(-4 * pos) * (0.5 + 0.5 * np.sin(6 * np.pi * pos))
        pattern3.append(max(0.0, val))
    patterns.append(pattern3)
    
    # Pattern 4: Sine wave modulation
    pattern4 = []
    for i in range(dim):
        pos = i / (dim - 1) if dim > 1 else 0.5
        val = 0.5 + 0.5 * np.sin(16 * np.pi * pos) + 0.1 * np.random.random()
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

def jax_autograd_optimization(initial_params, max_iter=1000):
    """Optimization using JAX automatic differentiation"""
    
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
    
    learning_rate = 0.001
    momentum = 0.9
    velocity = jnp.zeros_like(current_x)
    
    # Adaptive step size based on convergence
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
        
        # Track best solution
        if current_value < best_value:
            best_value = current_value
            best_x = current_x
    
    return np.array(best_x)

def evolutionary_optimization_strategy():
    """Improved evolutionary optimization with adaptive parameters"""
    # Try multiple initialization strategies with different dimensions
    best_c2 = -np.inf
    best_params = None
    
    # Different starting points with various initialization strategies
    strategies = [
        ("geometric", lambda dim: sophisticated_geometric_initialization(dim)),
        ("multiscale", lambda dim: sophisticated_multi_scale_initialization(dim)),
    ]
    
    # Try multiple random starts with different parameters
    for strategy_name, init_func in strategies:
        for start_seed in [42, 123, 234, 345, 456]:
            try:
                np.random.seed(start_seed)
                
                # Use adaptive sizing
                if strategy_name == "geometric":
                    dim = np.random.randint(400, 1000)
                else:
                    dim = np.random.randint(500, 1200)
                
                # Generate initial parameter
                initial_params = init_func(dim)
                
                # Optimize using JAX auto-differentiation
                optimized_params = jax_autograd_optimization(initial_params, max_iter=500)
                
                # Evaluate final result
                f_vals = np.clip(optimized_params, 0, None)
                g_vals = compute_autoconvolution_numba(f_vals)
                c2 = compute_c2_numba(g_vals)
                
                if c2 > best_c2:
                    best_c2 = c2
                    best_params = optimized_params.copy()
                    
            except Exception as e:
                continue
                
    return best_params, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Multi-start approach with different initialization strategies
    best_c2 = -np.inf
    best_params = None

    # Try multiple optimizations with different strategies
    for seed in [42, 123, 234, 345, 456, 567, 678, 789]:
        np.random.seed(seed)
        try:
            # Use our improved evolutionary approach
            params, c2 = evolutionary_optimization_strategy()

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