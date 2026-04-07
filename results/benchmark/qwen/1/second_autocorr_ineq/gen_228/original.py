# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
import time
from typing import List, Tuple
import numba
from numba import jit
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
import warnings
warnings.filterwarnings('ignore')

# Global constants
MAX_EVALUATIONS = 10000
TIME_LIMIT_SECONDS = 85
DEFAULT_N_STEPS = 500

@jit(nopython=True)
def compute_autoconvolution_norms_numba(f_vals: np.ndarray) -> Tuple[float, float, float]:
    """
    Fast computation of autoconvolution norms using Numba JIT compilation
    """
    n = len(f_vals)

    # Compute autoconvolution g = f * f using discrete convolution
    g_length = 2 * n - 1
    g = np.zeros(g_length)

    # Manual convolution loop for speed
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    # Compute the norms using piecewise linear integration for ||g||₂²
    # Trapezoidal-like: (1/3)(y1² + y1*y2 + y2²) per segment
    norm_g_2_squared = 0.0
    for i in range(g_length - 1):
        y1 = g[i]
        y2 = g[i + 1]
        norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0

    # ||g||₁ = sum(|g[i]|) / (len(g) + 1) for normalization
    norm_g_1 = 0.0
    for i in range(g_length):
        norm_g_1 += abs(g[i])
    norm_g_1 = norm_g_1 / (g_length + 1)

    # ||g||∞ = max(|g[i]|)
    norm_g_inf = 0.0
    for i in range(g_length):
        abs_g = abs(g[i])
        if abs_g > norm_g_inf:
            norm_g_inf = abs_g

    return norm_g_2_squared, norm_g_1, norm_g_inf

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the norms ||g||₂², ||g||₁, and ||g||∞ for the autoconvolution g = f*f
    """
    f = np.array(f_values)
    norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms_numba(f)
    return norm_g_2_squared, norm_g_1, norm_g_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function
    """
    try:
        norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_g_1 <= 1e-12 or norm_g_inf <= 1e-12:
            return 0.0

        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception as e:
        return 0.0

@jax_jit
def compute_c2_jax(f_vals):
    """JAX version for automatic differentiation - simplified for gradient support"""
    f = jnp.array(f_vals)
    f = jnp.maximum(f, 0)  # Ensure non-negative

    if len(f_vals) == 0:
        return jnp.array(0.0)

    # Simple autoconvolution using JAX
    g_full = jnp.convolve(f, f, mode='full')

    # Take center portion corresponding to [-1/4, 1/4] interval
    half_len = len(f_vals)
    center_start = len(g_full) // 2 - half_len + 1
    center_end = len(g_full) // 2 + half_len - 1
    g = g_full[center_start:center_end]

    # Compute norms using JAX operations
    g_abs = jnp.abs(g)

    # L2 squared norm (sum of squares)
    norm_l2_sq = jnp.sum(g_abs * g_abs)

    # L1 norm (normalized)
    norm_l1 = jnp.sum(g_abs) / (len(g) + 1)

    # L-infinity norm
    norm_inf = jnp.max(g_abs)

    # Avoid division by zero
    safe_l1 = jnp.where(norm_l1 <= 1e-15, 1e-15, norm_l1)
    safe_inf = jnp.where(norm_inf <= 1e-15, 1e-15, norm_inf)

    return norm_l2_sq / (safe_l1 * safe_inf)

def compute_gradient_jax(f_vals):
    """Compute gradient of C2 score using JAX automatic differentiation"""
    try:
        # Ensure inputs are properly clipped
        f_array = jnp.array(jnp.clip(f_vals, 0, None))

        # Compute gradient using JAX automatic differentiation
        grad_func = grad(compute_c2_jax)
        gradients = grad_func(f_array)

        # Convert back to numpy array
        return np.array(gradients)
    except Exception as e:
        # Return zero gradients if computation fails
        return np.zeros_like(f_vals)

def generate_harmonic_initialization(n_steps: int) -> List[float]:
    """Generate initial step function based on harmonics and spectral properties"""
    # Create a pattern that is likely to produce good spectral characteristics
    f = np.zeros(n_steps)

    # Base frequency components
    x = np.linspace(-1, 1, n_steps)

    # Add fundamental and harmonic components to encourage favorable spectra
    # Fundamental frequency
    f += 0.5 + 0.3 * np.sin(2 * np.pi * x)

    # Second harmonic
    f += 0.2 * np.sin(4 * np.pi * x)

    # Third harmonic
    f += 0.1 * np.sin(6 * np.pi * x)

    # Add some random components to break symmetry
    f += 0.1 * np.random.random(n_steps)

    # Ensure non-negative values
    f = np.clip(f, 0, None)

    # Normalize
    if np.sum(f) > 0:
        f = f / np.sum(f)

    return f.tolist()

def generate_spectral_initialization(n_steps: int) -> List[float]:
    """Generate an initial function that encourages spectral flatness in g"""
    # Start with a multi-peak structure to promote complex convolution properties
    f = np.zeros(n_steps)

    # Create peaks at different locations to encourage rich spectral content
    peak_positions = [0.2, 0.4, 0.6, 0.8]  # Relative positions
    peak_heights = [0.8, 0.7, 0.9, 0.6]

    x = np.linspace(-1, 1, n_steps)

    for pos, height in zip(peak_positions, peak_heights):
        # Place Gaussian-like peaks
        sigma = 0.1
        mu = (pos - 0.5) * 2  # Map to [-1, 1]
        f += height * np.exp(-0.5 * ((x - mu) / sigma)**2)

    # Add a base component
    f += 0.2 * (1 + np.sin(3 * np.pi * x))

    # Add small random noise for exploration
    f += 0.05 * np.random.random(n_steps)

    # Ensure non-negative values
    f = np.clip(f, 0, None)

    # Normalize
    if np.sum(f) > 0:
        f = f / np.sum(f)

    return f.tolist()

def generate_multi_scale_initialization(n_steps: int) -> List[float]:
    """Generate diverse initializations at multiple scales"""
    # Sample from various patterns
    patterns = []

    # Pattern 1: Sine-wave like
    x = np.linspace(-1, 1, n_steps)
    pattern1 = 0.5 + 0.3 * np.sin(4 * np.pi * x) + 0.2 * np.cos(8 * np.pi * x)
    pattern1 = np.clip(pattern1, 0, None)
    pattern1 = pattern1 / np.sum(pattern1)
    patterns.append(pattern1.tolist())

    # Pattern 2: Multi-peak structure
    pattern2 = np.zeros(n_steps)
    for i in range(0, n_steps, max(1, n_steps // 5)):
        if i < n_steps:
            pattern2[i] = 1.0 + 0.2 * np.random.random()
    pattern2 = np.clip(pattern2, 0, None)
    pattern2 = pattern2 / np.sum(pattern2)
    patterns.append(pattern2.tolist())

    # Pattern 3: Harmonic combination
    pattern3 = generate_harmonic_initialization(n_steps)
    patterns.append(pattern3)

    # Pattern 4: Spectral optimized
    pattern4 = generate_spectral_initialization(n_steps)
    patterns.append(pattern4)

    # Select the best pattern based on initial evaluation
    best_pattern = patterns[0]
    best_score = -1.0

    for pattern in patterns:
        try:
            score = evaluate_c2(pattern)
            if score > best_score:
                best_score = score
                best_pattern = pattern[:]
        except:
            continue

    return best_pattern

def gradient_guided_evolution_step(current_params: List[float],
                                  max_steps: int = 100) -> List[float]:
    """
    Perform gradient-guided evolution step using JAX gradients
    """
    current_params = np.array(current_params, dtype=float)
    current_c2 = evaluate_c2(current_params)

    # Adaptive learning rate
    learning_rate = 0.1
    patience = 0
    best_c2 = current_c2
    best_params = current_params.copy()

    for iteration in range(max_steps):
        try:
            # Compute gradient using JAX for accuracy
            grad_vec = compute_gradient_jax(current_params)

            # Apply gradient step
            new_params = current_params + learning_rate * grad_vec

            # Ensure non-negativity
            new_params = np.maximum(new_params, 0)

            # Normalize
            if np.sum(new_params) > 0:
                new_params = new_params / np.sum(new_params)

        except Exception:
            # Fallback to simple random perturbation
            new_params = current_params + 0.01 * np.random.random(len(current_params))
            new_params = np.maximum(new_params, 0)
            if np.sum(new_params) > 0:
                new_params = new_params / np.sum(new_params)

        # Evaluate new solution
        new_c2 = evaluate_c2(new_params)

        if new_c2 > current_c2:
            current_params = new_params
            current_c2 = new_c2
            patience = 0

            if new_c2 > best_c2:
                best_c2 = new_c2
                best_params = current_params.copy()
        else:
            patience += 1
            if patience > 10:
                learning_rate *= 0.5
                patience = 0
                if learning_rate < 1e-6:
                    break

    return best_params.tolist()

def multi_scale_optimization(n_steps: int, max_time_seconds: float = 85.0) -> List[float]:
    """
    Multi-scale optimization combining different strategies
    """
    start_time = time.time()

    # Phase 1: Multi-start with various initializations
    best_c2 = -np.inf
    best_params = None

    # Try multiple diverse initializations
    for i in range(5):
        if time.time() - start_time > max_time_seconds * 0.7:
            break

        try:
            # Generate different types of initializations
            if i == 0:
                initial_params = generate_harmonic_initialization(n_steps)
            elif i == 1:
                initial_params = generate_spectral_initialization(n_steps)
            else:
                initial_params = generate_multi_scale_initialization(n_steps)

            # Apply gradient-guided evolution
            evolved_params = gradient_guided_evolution_step(initial_params, max_steps=100)

            # Evaluate final result
            final_c2 = evaluate_c2(evolved_params)

            if final_c2 > best_c2:
                best_c2 = final_c2
                best_params = evolved_params.copy()

        except Exception as e:
            continue

    # Phase 2: Refinement with local optimizer
    if best_params is not None and time.time() - start_time < max_time_seconds - 3.0:
        try:
            def objective(x):
                return -evaluate_c2(x)

            bounds = [(0, None) for _ in range(len(best_params))]
            result = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )

            if result.success:
                refined_params = np.maximum(result.x, 0)
                if np.sum(refined_params) > 0:
                    refined_params = refined_params / np.sum(refined_params)
                refined_c2 = evaluate_c2(refined_params)

                if refined_c2 > best_c2:
                    best_c2 = refined_c2
                    best_params = refined_params.tolist()

        except Exception:
            pass

    return best_params if best_params is not None else [1.0/n_steps] * n_steps

def construct_function() -> list[float]:
    """
    Main function to construct optimized step-function for high C2 value
    Uses hybrid gradient-guided evolution with multi-scale initialization
    """
    start_time = time.time()

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Try different problem sizes for better exploration
    problem_sizes = [300, 400, 500, 600, 700, 800]
    best_c2 = -np.inf
    best_params = None

    # Multi-start optimization
    for sz in problem_sizes:
        if time.time() - start_time > TIME_LIMIT_SECONDS * 0.9:
            break

        try:
            params = multi_scale_optimization(sz, max_time_seconds=TIME_LIMIT_SECONDS)
            c2 = evaluate_c2(params)

            if c2 > best_c2:
                best_c2 = c2
                best_params = params.copy()

        except Exception as e:
            continue

    # Final fallback if nothing worked
    if best_params is None:
        n_steps = 500
        return [1.0/n_steps] * n_steps

    return best_params

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")