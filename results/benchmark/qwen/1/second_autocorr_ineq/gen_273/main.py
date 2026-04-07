# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc
import time
from numba import jit
import jax
import jax.numpy as jnp
from jax import grad, hessian
import optuna
from functools import partial

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation with improved integration"""
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

def compute_autoconvolution_fft(f_vals):
    """
    Compute autoconvolution using FFT for better numerical stability and performance.
    This is more efficient and numerically stable for large arrays.
    """
    import numpy as np
    from scipy.fft import fft, ifft

    # Ensure input is numpy array
    f_vals = np.asarray(f_vals)
    n = len(f_vals)

    # Pad to avoid circular convolution effects
    padded_length = 2 * n - 1
    f_padded = np.pad(f_vals, (0, padded_length - n), mode='constant', constant_values=0)

    # FFT-based convolution
    F = fft(f_padded)
    G_fft = F * F  # Point-wise multiplication in frequency domain
    g = ifft(G_fft).real[:padded_length]

    return g

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation with proper integration"""
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

    # For L2^2 norm using proper trapezoidal integration
    if len(g_vals) >= 2:
        # Trapezoidal rule: h * (y0^2 + 2*y1^2 + ... + yn-1^2)/2
        g_l2_sq = g_vals[0]*g_vals[0] + g_vals[-1]*g_vals[-1]
        for i in range(1, len(g_vals)-1):
            g_l2_sq += 2 * g_vals[i] * g_vals[i]
        h = 1.0 / (len(g_vals) - 1) if len(g_vals) > 1 else 0.001
        g_l2_sq *= h / 2.0
    elif len(g_vals) == 1:
        g_l2_sq = g_vals[0] * g_vals[0]

    # Compute C2
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def compute_c2_fft(f_vals):
    """Compute C2 value using FFT-based autoconvolution for better performance"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        # For small arrays, use the numba version for consistency
        # For larger arrays, use FFT for performance
        if len(f_vals) > 500:
            # Use FFT-based convolution
            g_vals = compute_autoconvolution_fft(f_vals)
        else:
            # Use numba version for small arrays
            g_vals = compute_autoconvolution_numba(f_vals)

        # Compute C2
        c2 = compute_c2_numba(g_vals)
        return c2

    except Exception as e:
        return 0.0

def evaluate_c2_jax(f_vals):
    """JAX-based evaluation of C2 for better gradient computation"""
    try:
        # Ensure non-negative values
        f_vals = jnp.maximum(f_vals, 0.0)

        # Compute autoconvolution
        n = len(f_vals)
        g_len = 2 * n - 1
        g = jnp.zeros(g_len)

        # Manual convolution (vectorized version)
        for i in range(n):
            for j in range(n):
                idx = i + j
                if 0 <= idx < g_len:
                    g = g.at[idx].add(f_vals[i] * f_vals[j])

        # Compute norms for C2
        g_l2_sq = jnp.sum(g**2)
        g_l1 = jnp.sum(jnp.abs(g))
        g_max = jnp.max(jnp.abs(g))

        # Avoid division by zero
        epsilon = 1e-16
        c2 = g_l2_sq / (jnp.maximum(g_l1, epsilon) * jnp.maximum(g_max, epsilon))

        return c2
    except Exception:
        return 0.0

def compute_gradient_jax(f_vals):
    """Compute gradient of C2 with respect to f_vals using JAX"""
    try:
        # Vectorize the function for gradient calculation
        grad_fn = grad(evaluate_c2_jax)
        return jnp.array(grad_fn(f_vals))
    except Exception:
        return jnp.zeros_like(f_vals)

def gaussian_pattern_initialization(dim):
    """Initialize with a Gaussian-shaped pattern"""
    x = np.linspace(-1.0, 1.0, dim)
    # Create two overlapping Gaussians with different widths
    g1 = np.exp(-x**2 / 0.1) * 0.8
    g2 = np.exp(-(x - 0.4)**2 / 0.2) * 0.6
    g3 = np.exp(-(x + 0.4)**2 / 0.2) * 0.6
    pattern = np.maximum(g1, np.maximum(g2, g3))
    return pattern.tolist()

def cosine_modulated_initialization(dim):
    """Initialize with cosine-modulated pattern"""
    pattern = []
    for i in range(dim):
        # Create alternating pattern with cosine modulation
        pos = i / dim * 2 - 1
        val = 0.5 + 0.4 * np.cos(pos * np.pi * 3) + 0.1 * np.sin(pos * np.pi * 7)
        pattern.append(max(0, val))
    return pattern

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

def latin_hypercube_initialization(dim):
    """Initialize using Latin Hypercube Sampling"""
    try:
        sampler = qmc.LatinHypercube(d=1, seed=42)
        samples = sampler.random(n=dim)
        # Scale to [0, 1] and add some variation
        pattern = [0.3 + 0.7 * s[0] + 0.1 * np.sin(i) for i, s in enumerate(samples)]
        return [max(0, p) for p in pattern]
    except:
        # Fallback to basic random
        return [0.3 + 0.7 * np.random.random() + 0.1 * np.sin(i) for i in range(dim)]

def mixed_strategy_initialization(dim):
    """Create an initialization combining multiple strategies"""
    strategies = [
        gaussian_pattern_initialization,
        cosine_modulated_initialization,
        fractal_like_initialization,
        latin_hypercube_initialization
    ]

    # Evaluate all strategies and select the best
    best_score = -1.0
    best_pattern = None

    for strategy in strategies:
        try:
            pattern = strategy(dim)
            score = compute_c2_numba(compute_autoconvolution_numba(pattern))
            if score > best_score:
                best_score = score
                best_pattern = pattern[:]
        except:
            continue

    if best_pattern is None:
        # Fallback to simple pattern
        return [0.5] * dim

    return best_pattern

def adaptive_population_evolution(initial_dim, max_iter=150):
    """Perform evolutionary optimization with adaptive population sizing"""
    # Start with multiple initialization strategies
    x0 = mixed_strategy_initialization(initial_dim)

    # Initialize parameters with dynamic scaling
    popsize = min(25, initial_dim // 10 + 10)  # Adaptive population size (larger)
    mutation_range = (0.7, 1.0)  # Wider mutation range for better exploration
    recombination_rate = 0.7

    # Set bounds for optimization
    bounds = [(0, 10)] * len(x0)

    # Run differential evolution with adaptive parameters
    result = differential_evolution(
        lambda x: -compute_c2_fft(x),
        bounds,
        maxiter=max_iter,
        popsize=popsize,
        seed=42,
        disp=False,
        tol=1e-6,
        mutation=mutation_range,
        recombination=recombination_rate,
        init='random'  # Use random initialization to avoid bias
    )

    return result.x

def advanced_local_search(initial_params, max_iter=100):
    """Advanced local search with stochastic perturbations"""
    def objective(x):
        return -compute_c2_fft(x)

    try:
        # First try Nelder-Mead for global refinement
        result1 = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': max_iter//3, 'ftol': 1e-8, 'xtol': 1e-8}
        )

        # Add stochastic perturbation to escape local optimum
        perturbed_params = result1.x + 0.01 * np.random.randn(len(result1.x))
        perturbed_params = np.maximum(perturbed_params, 0)

        # Then try L-BFGS-B for local fine-tuning
        result2 = minimize(
            objective,
            perturbed_params,
            method='L-BFGS-B',
            bounds=[(0, 10) for _ in range(len(initial_params))],
            options={'maxiter': max_iter//3, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )

        # Final stochastic perturbation for escaping plateaus
        final_params = result2.x + 0.005 * np.random.randn(len(result2.x))
        final_params = np.maximum(final_params, 0)

        return final_params
    except Exception:
        # Fallback to coordinate-wise optimization
        try:
            # Simple gradient descent-like approach for robustness
            current_params = np.array(initial_params)
            for _ in range(max_iter//2):
                new_params = current_params.copy()
                # Update each parameter slightly
                for i in range(len(current_params)):
                    test_params = current_params.copy()
                    test_params[i] = max(0, current_params[i] + 0.01 * np.random.randn())
                    if compute_c2_fft(test_params) > compute_c2_fft(current_params):
                        new_params[i] = test_params[i]
                current_params = new_params
            return current_params
        except:
            return initial_params

def multi_scale_optimization():
    """Run multi-scale optimization with progressive refinement"""
    start_time = time.time()
    best_c2 = -np.inf
    best_params = None

    # Multi-scale approach: start with coarser resolution
    scale_configs = [
        (200, 70),   # Coarse scale
        (400, 80),   # Medium scale
        (600, 90),   # Fine scale
        (800, 100),  # Finer scale
    ]

    # Add randomized configurations for diversity
    for _ in range(8):
        dim = np.random.randint(300, 900)
        iterations = np.random.randint(60, 120)
        scale_configs.append((dim, iterations))

    # Run optimization for each scale
    for n_steps, n_trials in scale_configs:
        if time.time() - start_time > 85:  # Leave buffer for cleanup
            break

        try:
            # Try different random seeds
            for seed in [42, 123, 456, 789]:
                np.random.seed(seed)
                # Run adaptive evolutionary optimization
                params = adaptive_population_evolution(n_steps, n_trials)

                # Apply advanced local refinement with stochastic perturbations
                refined_params = advanced_local_search(params, 60)

                # Evaluate final result
                c2 = compute_c2_numba(compute_autoconvolution_numba(refined_params))

                if c2 > best_c2:
                    best_c2 = c2
                    best_params = refined_params.copy()

        except Exception as e:
            continue

    # Final refinement if time permits
    if best_params is not None and time.time() - start_time < 85:
        try:
            # Additional fine-tuning
            final_params = advanced_local_search(best_params, 40)
            final_c2 = compute_c2_numba(compute_autoconvolution_numba(final_params))

            if final_c2 > best_c2:
                best_c2 = final_c2
                best_params = final_params
        except:
            pass

    return best_params, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    try:
        # Main optimization strategy
        best_params, best_c2 = multi_scale_optimization()

        # If we found a good solution, use it
        if best_params is not None and best_c2 > 0:
            return best_params

        # Fallback to deterministic approach
        size = 800
        best_f_vals = adaptive_population_evolution(size, 100)
        best_f_vals = advanced_local_search(best_f_vals, 50)
        c2_val = compute_c2_numba(compute_autoconvolution_numba(best_f_vals))
        print(f"Best C2 found: {c2_val}")

        # Return the optimized values
        return best_f_vals.tolist()

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Final fallback to structured initialization
        return mixed_strategy_initialization(1000)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")