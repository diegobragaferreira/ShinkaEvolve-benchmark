# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc
import time
from numba import jit, prange
import numba
from sklearn.cluster import KMeans
import warnings
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
import optuna

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

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

@jax_jit
def compute_autoconvolution_jax(f_vals):
    """Compute autoconvolution using JAX for better gradient computation"""
    # Using JAX's convolution with proper padding
    f = jnp.array(f_vals)
    # Use explicit convolution to match the mathematical definition
    # Padding to avoid circular effects
    pad_width = len(f) - 1
    padded_f = jnp.pad(f, (pad_width, pad_width), mode='constant', constant_values=0)
    # Manual convolution using JAX operations
    result = jnp.convolve(padded_f, f, mode='valid')
    return result

@jax_jit
def evaluate_c2_jax(f_vals):
    """Evaluate C2 using JAX for automatic differentiation"""
    g_vals = compute_autoconvolution_jax(f_vals)

    # Compute norms
    g_l2_sq = jnp.sum(g_vals ** 2)
    g_l1 = jnp.sum(jnp.abs(g_vals))
    g_max = jnp.max(jnp.abs(g_vals))

    # Avoid division by zero
    epsilon = 1e-16
    g_l1 = jnp.maximum(g_l1, epsilon)
    g_max = jnp.maximum(g_max, epsilon)

    c2 = g_l2_sq / (g_l1 * g_max)
    return c2

# Vectorized batch processing for efficiency
@jax.vmap
def compute_autoconvolution_batched(f_vals):
    """Vectorized autoconvolution computation for batches of functions"""
    return compute_autoconvolution_jax(f_vals)

@jax.vmap
def evaluate_c2_batched(f_vals):
    """Vectorized C2 computation for batches of functions"""
    return evaluate_c2_jax(f_vals)

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation with proper integration"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms using trapezoidal integration for L2^2
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

    # Compute L2^2 norm using trapezoidal integration
    if len(g_vals) >= 2:
        g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
        for i in range(1, len(g_vals)-1):
            g_l2_sq += 2 * g_vals[i] * g_vals[i]
        # Correct step width for domain [-1/2, 1/2] with len(g_vals) points
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # Compute C2 with improved numerical stability
    epsilon = 1e-16
    if g_l1 > epsilon and g_max > epsilon:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute L1, L2^2, and L-infinity norms efficiently"""
    n = len(g_vals)

    # L1 norm approximation (sum of absolute values)
    l1_norm = 0.0
    for i in range(n):
        l1_norm += abs(g_vals[i])

    # L2^2 norm (sum of squares)
    l2_sq_norm = 0.0
    for i in range(n):
        l2_sq_norm += g_vals[i] * g_vals[i]

    # L-infinity norm (maximum absolute value)
    linf_norm = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf_norm:
            linf_norm = abs_val

    return l1_norm, l2_sq_norm, linf_norm

def gaussian_pattern_initialization(dim):
    """Initialize with a Gaussian-shaped pattern"""
    x = np.linspace(-1.0, 1.0, dim)
    # Create two overlapping Gaussians with different widths
    g1 = np.exp(-x**2 / 0.1) * 0.8
    g2 = np.exp(-(x - 0.4)**2 / 0.2) * 0.6
    g3 = np.exp(-(x + 0.4)**2 / 0.2) * 0.6
    pattern = np.maximum(g1, np.maximum(g2, g3))
    return pattern.tolist()

def fractal_like_initialization(dim):
    """Initialize with fractal-like self-similar pattern"""
    pattern = []
    # Generate multiple scales of pattern to create complexity
    for i in range(dim):
        # Base pattern with multiple frequencies
        pos = i / dim
        base = 0.5 + 0.3 * np.sin(pos * np.pi * 8)
        base += 0.2 * np.sin(pos * np.pi * 16)
        base += 0.1 * np.sin(pos * np.pi * 32)
        pattern.append(max(0, base))
    return pattern

def structured_initialization(dim):
    """Create a structured pattern with alternating peaks and valleys"""
    pattern = []
    for i in range(dim):
        if i % 5 == 0:
            pattern.append(1.0)
        elif i % 5 == 2:
            pattern.append(0.3)
        else:
            pattern.append(0.7)
    return pattern

def generate_multi_scale_initialization(dim):
    """Generate multi-scale initial patterns to capture various structures"""
    # Multiple initialization strategies
    strategies = [
        gaussian_pattern_initialization,
        fractal_like_initialization,
        structured_initialization
    ]

    best_score = -1.0
    best_pattern = None

    for strategy in strategies:
        try:
            pattern = strategy(dim)
            # Evaluate with fast numba version
            score = compute_c2_numba(compute_autoconvolution_numba(pattern))
            if score > best_score:
                best_score = score
                best_pattern = pattern[:]
        except:
            continue

    # If no good pattern found, create a hybrid pattern
    if best_pattern is None:
        # Combine structured and random elements
        best_pattern = []
        for i in range(dim):
            # Mix of structured and random
            base = 0.5 + 0.3 * np.sin(i * 0.3)
            noise = np.random.normal(0, 0.1)
            val = base + noise
            best_pattern.append(max(0, val))

    return best_pattern

def adaptive_evolutionary_optimization(initial_dim, max_iter=150, adaptive=True):
    """Perform evolutionary optimization with adaptive parameters"""

    # Start with multi-scale initialization
    x0 = generate_multi_scale_initialization(initial_dim)

    # Set bounds for optimization
    bounds = [(0, 10)] * len(x0)

    # Adaptive population sizing strategy
    initial_popsize = min(10, initial_dim // 50 + 3)  # Start small for fast exploration
    max_popsize = min(25, initial_dim // 10 + 10)      # Maximum population size
    popsize = initial_popsize

    # Parameters for differential evolution
    de_params = {
        'mutation': (0.5, 0.9),
        'recombination': 0.7,
        'popsize': popsize,
        'maxiter': max_iter,
        'seed': 42,
        'tol': 1e-6,
        'init': 'latinhypercube',
        'disp': False
    }

    # Run optimization
    result = differential_evolution(
        lambda x: -compute_c2_numba(compute_autoconvolution_numba(x)),
        bounds,
        **de_params
    )

    return result.x

def adaptive_gradient_optimization(initial_params, max_iter=100):
    """Use gradient-based optimization with automatic differentiation"""
    def objective(x):
        return -evaluate_c2_jax(x)

    try:
        # Use L-BFGS-B for smooth local optimization with gradients
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=[(0, 10) for _ in range(len(initial_params))],
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-9},
            tol=1e-9
        )
        return result.x
    except Exception as e:
        # Fallback to simpler method if L-BFGS fails
        try:
            # Use Nelder-Mead as fallback
            result = minimize(
                objective,
                initial_params,
                method='Nelder-Mead',
                options={'maxiter': max_iter//2, 'ftol': 1e-8, 'xtol': 1e-8}
            )
            return result.x
        except Exception:
            # Last resort: simple coordinate-wise improvement
            current_params = np.array(initial_params)
            for iteration in range(max_iter):
                improved = False
                for i in range(len(current_params)):
                    # Test small adjustments
                    test_params = current_params.copy()
                    adjustment = 0.01 * np.random.randn()
                    test_params[i] = max(0, current_params[i] + adjustment)
                    if compute_c2_numba(compute_autoconvolution_numba(test_params)) > \
                       compute_c2_numba(compute_autoconvolution_numba(current_params)):
                        current_params[i] = test_params[i]
                        improved = True
                if not improved:
                    break
            return current_params

def advanced_refinement_strategy(initial_params, max_iter=100):
    """Advanced refinement using both gradient information and local search"""
    # First perform gradient-based refinement
    refined_params = adaptive_gradient_optimization(initial_params, max_iter//2)

    # Then do additional local search to fine-tune
    try:
        # Add controlled stochastic perturbations to escape local optima
        perturbed_params = refined_params + np.random.normal(0, 0.005, len(refined_params))  # Reduced perturbation scale
        perturbed_params = np.maximum(perturbed_params, 0)

        # Evaluate and compare
        orig_score = compute_c2_numba(compute_autoconvolution_numba(refined_params))
        perturbed_score = compute_c2_numba(compute_autoconvolution_numba(perturbed_params))

        if perturbed_score > orig_score:
            refined_params = perturbed_params

        # Apply stochastic perturbation before final refinement to further escape local optima
        final_perturbation = np.random.normal(0, 0.002, len(refined_params))
        final_params = np.maximum(refined_params + final_perturbation, 0)

        # Final local search around the best
        final_params = adaptive_gradient_optimization(final_params, max_iter//2)
        return final_params
    except:
        return refined_params

def multi_stage_optimization():
    """Run multi-stage optimization pipeline with progressive refinement"""
    start_time = time.time()
    best_c2 = -np.inf
    best_params = None

    # Stage 1: Coarse optimization with various population sizes
    configs = [(300, 80), (500, 100), (700, 120)]

    for dim, iter_count in configs:
        if time.time() - start_time > 80:  # Leave buffer for cleanup
            break

        try:
            # Multi-start with different seeds
            for seed in [42, 123, 456]:
                np.random.seed(seed)
                params = adaptive_evolutionary_optimization(dim, iter_count)

                # Advanced local refinement
                refined_params = advanced_refinement_strategy(params, 50)

                # Evaluate final result
                c2 = compute_c2_numba(compute_autoconvolution_numba(refined_params))

                if c2 > best_c2:
                    best_c2 = c2
                    best_params = refined_params.copy()

        except Exception as e:
            continue

    # Stage 2: Fine-tuning with smaller population but more iterations
    if best_params is not None and time.time() - start_time < 85:
        try:
            # Use best parameters as starting point for fine-tuning
            fine_params = advanced_refinement_strategy(best_params, 100)
            fine_c2 = compute_c2_numba(compute_autoconvolution_numba(fine_params))

            if fine_c2 > best_c2:
                best_c2 = fine_c2
                best_params = fine_params
        except Exception:
            pass

    # Stage 3: Multi-start optimization using optuna for global search
    if time.time() - start_time < 85:
        try:
            # Create a simple multi-start optuna-like approach
            for seed in [789, 999, 111]:
                np.random.seed(seed)
                # Generate random initial candidate
                rand_dim = np.random.randint(500, 1000)
                rand_params = generate_multi_scale_initialization(rand_dim)

                # Refine this candidate
                refined_rand = advanced_refinement_strategy(rand_params, 50)

                # Evaluate
                c2 = compute_c2_numba(compute_autoconvolution_numba(refined_rand))

                if c2 > best_c2:
                    best_c2 = c2
                    best_params = refined_rand.copy()
        except Exception:
            pass

    return best_params, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    try:
        # Main optimization strategy
        best_params, best_c2 = multi_stage_optimization()

        # If we found a good solution, use it
        if best_params is not None and best_c2 > 0:
            return best_params

        # Fallback to deterministic approach
        size = 800
        params = adaptive_evolutionary_optimization(size, 100)
        params = advanced_refinement_strategy(params, 50)
        c2_val = compute_c2_numba(compute_autoconvolution_numba(params))
        print(f"Best C2 found: {c2_val}")

        # Return the optimized values
        return params.tolist()

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Final fallback to structured initialization
        return generate_multi_scale_initialization(1000)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")