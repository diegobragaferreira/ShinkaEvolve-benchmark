# EVOLVE-BLOCK-START
import numpy as np
import time
from numba import jit, prange
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
from scipy.optimize import differential_evolution
import random

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

    # Compute convolution manually for efficiency - optimized inner loop
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
    """Compute C2 value using fast Numba implementation with optimized integration"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms with optimized accumulation
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L2 norm squared using piecewise quadratic approximation (more accurate than simple trapezoid)
    for i in range(len(g_vals) - 1):
        val1 = g_vals[i]
        val2 = g_vals[i+1]
        # Using trapezoidal rule with quadratic approximation for better accuracy
        # (h/3)(y1^2 + y1*y2 + y2^2)
        g_l2_sq += (STEP_WIDTH/3.0) * (val1*val1 + val1*val2 + val2*val2)

    # For L1 norm (sum of absolute values) - normalized
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (max absolute value)
    for i in range(len(g_vals)):
        abs_val = abs(g_vals[i])
        if abs_val > g_max:
            g_max = abs_val

    # Compute C2 with better numerical stability
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def sophisticated_multiscale_initialization(dim):
    """Create initial parameters using multi-scale patterns for better exploration"""
    # Create several different patterns and select the best
    patterns = []
    
    # Pattern 1: Multi-peak Gaussian structure
    x = np.linspace(0, 1, dim)
    pattern1 = np.zeros(dim)
    # Add 3 peaks with random positions and heights
    for peak_idx in range(3):
        center = 0.1 + 0.4 * np.random.random()  # Position between 0.1 and 0.5
        width = 0.05 + 0.05 * np.random.random()  # Width between 0.05 and 0.1
        height = 0.8 + 0.4 * np.random.random()   # Height between 0.8 and 1.2
        pattern1 += height * np.exp(-((x - center)**2) / (2 * width**2))
    patterns.append(pattern1.tolist())
    
    # Pattern 2: Alternating high/low pattern with more randomness
    pattern2 = []
    for i in range(dim):
        if i % 3 == 0:
            pattern2.append(1.2 + 0.5 * np.random.random())
        elif i % 3 == 1:
            pattern2.append(0.3 + 0.2 * np.random.random())
        else:
            pattern2.append(0.8 + 0.3 * np.random.random())
    patterns.append(pattern2)
    
    # Pattern 3: Geometric pattern with exponential decay
    pattern3 = []
    for i in range(dim):
        pos = i / (dim - 1) if dim > 1 else 0.5
        # Bell-shaped curve with sine modulation
        val = np.exp(-4 * pos) * (0.6 + 0.4 * np.sin(8 * np.pi * pos))
        pattern3.append(max(0.0, val))
    patterns.append(pattern3)
    
    # Pattern 4: Sine wave modulation with peak concentration
    pattern4 = []
    for i in range(dim):
        pos = i / (dim - 1) if dim > 1 else 0.5
        # Concentrated peaks near center
        val = 0.4 + 0.6 * np.sin(12 * np.pi * pos) + 0.1 * np.random.random()
        # Add Gaussian shaping for central concentration
        gauss = np.exp(-8 * (pos - 0.5)**2)
        val = max(0.0, val * gauss)
        pattern4.append(val)
    patterns.append(pattern4)
    
    # Pattern 5: Random with heavy-tailed distribution
    pattern5 = []
    for i in range(dim):
        # Heavy-tailed distribution
        r = np.random.random()
        if r < 0.7:
            pattern5.append(0.2 + 0.3 * np.random.random())
        else:
            pattern5.append(1.0 + 1.5 * np.random.random())
    patterns.append(pattern5)
    
    # Select the best pattern by evaluating it with quick Numba computation
    best_pattern = patterns[0]
    best_score = -1e10
    
    for pattern in patterns:
        try:
            # Evaluate the pattern quickly using fast numba version
            f_vals = np.clip(pattern, 0, None)
            g_vals = compute_autoconvolution_numba(f_vals)
            c2 = compute_c2_numba(g_vals)
            if c2 > best_score:
                best_score = c2
                best_pattern = pattern
        except:
            continue
    
    return best_pattern

def adaptive_gradient_optimization(initial_params, max_iter=300):
    """Optimization using JAX automatic differentiation with adaptive learning rate"""
    # Convert to JAX array
    x0 = jnp.array(initial_params)
    
    # Define the objective in JAX
    @jax_jit
    def jax_objective(params):
        # Convert back to numpy for our existing numba computations
        f_vals_np = np.array(params)
        f_vals_np = np.clip(f_vals_np, 0, None)
        
        # Compute autoconvolution using our optimized numba function
        g_vals = compute_autoconvolution_numba(f_vals_np)
        
        # Compute C2 using our optimized numba function
        c2 = compute_c2_numba(g_vals)
        
        return -c2  # Negative because we minimize
    
    # Get the exact gradient function using JAX automatic differentiation
    grad_fn = grad(jax_objective)
    
    # Basic gradient descent with adaptive step size and momentum
    current_x = x0
    best_x = current_x
    best_value = jax_objective(current_x)
    
    # Adaptive parameters
    momentum = 0.9
    velocity = jnp.zeros_like(current_x)
    learning_rate = 0.01
    
    # Reduced iterations for speed
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

def advanced_refinement_strategy(initial_params):
    """Advanced refinement with multiple techniques to escape local optima"""
    current_solution = np.array(initial_params).copy()
    current_c2 = evaluate_c2_fast(current_solution)
    
    # 1. Apply gradient-based refinement with multiple restarts
    try:
        # Try multiple refinements with different learning rates
        best_refined = current_solution
        best_refined_c2 = current_c2
        
        for lr in [0.01, 0.005, 0.001]:
            refined_solution = adaptive_gradient_optimization(current_solution, max_iter=150)
            refined_c2 = evaluate_c2_fast(refined_solution)
            
            if refined_c2 < best_refined_c2:
                best_refined_c2 = refined_c2
                best_refined = refined_solution
        
        current_solution = best_refined
        current_c2 = best_refined_c2
    except:
        pass
    
    # 2. Apply stochastic perturbation to prevent local optima trapping
    try:
        # Add controlled noise to escape local minima
        noise_std = 0.03 * np.std(current_solution) if np.std(current_solution) > 1e-10 else 0.01
        perturbed_solution = current_solution + np.random.normal(0, noise_std, len(current_solution))
        perturbed_solution = np.clip(perturbed_solution, 0, None)
        perturbed_c2 = evaluate_c2_fast(perturbed_solution)
        
        if perturbed_c2 < current_c2:
            current_solution = perturbed_solution
            current_c2 = perturbed_c2
    except:
        pass
    
    # 3. Run local search with neighbors
    try:
        # Generate several neighbor solutions and keep best
        best_neighbor = current_solution
        best_neighbor_c2 = current_c2
        
        for _ in range(5):
            # Create neighbor by small random perturbations
            neighbor_solution = current_solution + np.random.normal(0, 0.01, len(current_solution))
            neighbor_solution = np.clip(neighbor_solution, 0, None)
            neighbor_c2 = evaluate_c2_fast(neighbor_solution)
            
            if neighbor_c2 < best_neighbor_c2:
                best_neighbor_c2 = neighbor_c2
                best_neighbor = neighbor_solution
                
        current_solution = best_neighbor
        current_c2 = best_neighbor_c2
    except:
        pass
    
    return current_solution

@jit(nopython=True)
def evaluate_c2_fast(f_vals):
    """Fast evaluation of C2 for use in refinement strategies"""
    try:
        g_vals = compute_autoconvolution_numba(f_vals)
        return compute_c2_numba(g_vals)
    except:
        return 1e10  # Large penalty for invalid results

def evolutionary_optimization_strategy():
    """Improved evolutionary optimization focused on quality over quantity"""
    best_c2 = -np.inf
    best_params = None
    
    # Try multiple initialization strategies with different dimensions
    for seed in [42, 123, 234, 345, 456]:
        try:
            np.random.seed(seed)
            
            # Use our optimized multiscale initialization with adaptive sizing
            dim = np.random.randint(500, 1500)  # Wider range for diversity
            initial_params = sophisticated_multiscale_initialization(dim)
            
            # Optimize using gradient-based approach
            optimized_params = adaptive_gradient_optimization(initial_params, max_iter=200)
            
            # Apply advanced refinement
            refined_params = advanced_refinement_strategy(optimized_params)
            
            # Evaluate final result
            final_c2 = evaluate_c2_fast(refined_params)
            
            if final_c2 > best_c2:
                best_c2 = final_c2
                best_params = refined_params.copy()
        except Exception as e:
            continue
                
    return best_params, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Multi-start approach with different random seeds
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