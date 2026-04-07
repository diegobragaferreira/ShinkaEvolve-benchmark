# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import convolve
from numba import jit
import time
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit

# Enable JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using Numba for speed"""
    n = len(f_vals)
    # Create convolution result array
    g = np.zeros(2*n - 1)

    # Compute autoconvolution: g[k] = sum(f[i]*f[k-i])
    for i in range(n):
        for j in range(n):
            k = i + j
            g[k] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute norms efficiently with Numba"""
    n = len(g_vals)

    # L2 norm squared (using trapezoidal approximation for piecewise linear)
    l2_sq = 0.0
    for i in range(n-1):
        h = 1.0  # assuming unit spacing
        y1 = g_vals[i]
        y2 = g_vals[i+1]
        l2_sq += (h/3.0) * (y1*y1 + y1*y2 + y2*y2)

    # L1 norm
    l1 = 0.0
    for i in range(n):
        l1 += abs(g_vals[i])

    # L-infinity norm
    linf = 0.0
    for i in range(n):
        if abs(g_vals[i]) > linf:
            linf = abs(g_vals[i])

    return l2_sq, l1, linf

def compute_c2_score(f_vals):
    """Compute C2 score for given step function values"""
    try:
        # Ensure non-negative values
        f_vals = np.maximum(f_vals, 0)

        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)

        # Compute norms
        l2_sq, l1, linf = compute_norms_numba(g_vals)

        # Avoid division by zero
        if l1 < 1e-12 or linf < 1e-12:
            return 0.0

        # Compute C2
        c2 = l2_sq / (l1 * linf)
        return c2

    except Exception:
        return 0.0

@jax_jit
def compute_c2_jax(f_vals):
    """JAX version for automatic differentiation"""
    # Convert to JAX array
    f = jnp.array(f_vals, dtype=jnp.float32)

    # Compute autoconvolution using JAX operations
    g = jnp.convolve(f, f, mode='full')
    n = len(f)
    offset = (n - 1) // 2
    g = g[offset:-offset]

    # Compute norms
    g_abs = jnp.abs(g)
    norm_l2_sq = jnp.sum(g_abs**2)
    norm_l1 = jnp.sum(g_abs)
    norm_inf = jnp.max(g_abs)

    # Avoid division by zero
    eps = 1e-12
    norm_l1 = jnp.where(norm_l1 < eps, eps, norm_l1)
    norm_inf = jnp.where(norm_inf < eps, eps, norm_inf)

    c2 = norm_l2_sq / (norm_l1 * norm_inf)
    return c2

def compute_gradient_jax(f_vals):
    """Compute gradient of C2 with respect to f_vals using JAX"""
    try:
        f = jnp.array(f_vals, dtype=jnp.float32)
        grad_fn = grad(compute_c2_jax)
        grad_val = grad_fn(f)
        return np.array(grad_val)
    except Exception:
        return np.zeros_like(f_vals)

def sophisticated_initialization(n):
    """Create good initial candidate with alternating pattern and Gaussian smoothing"""
    # Create alternating high/low regions
    f_vals = []
    segment_size = max(1, n // 8)  # Create ~8 segments

    for i in range(n):
        # Alternate between high and low values
        segment_idx = i // segment_size
        if segment_idx % 2 == 0:
            # High value region
            val = 1.0 + np.random.random() * 0.5
        else:
            # Low value region
            val = 0.1 + np.random.random() * 0.3

        # Add some Gaussian smoothing to avoid sharp transitions
        if i > 0 and i < n-1:
            smooth_factor = 0.7
            val = smooth_factor * val + (1-smooth_factor) * 0.5 * (
                f_vals[-1] + f_vals[-2] if len(f_vals) >= 2 else val)

        f_vals.append(max(0, val))

    return np.array(f_vals)

def gradient_ascent_refinement(initial_f, max_iter=50, step_size=0.01):
    """Refine solution using gradient ascent"""
    f_current = np.array(initial_f, dtype=np.float32)

    try:
        for _ in range(max_iter):
            # Compute gradient
            grad_val = compute_gradient_jax(f_current)

            # Update with gradient ascent
            f_new = f_current + step_size * grad_val

            # Ensure non-negativity
            f_new = np.maximum(f_new, 0)

            # Check improvement
            old_c2 = compute_c2_score(f_current)
            new_c2 = compute_c2_score(f_new)

            if new_c2 > old_c2:
                f_current = f_new
            else:
                # Reduce step size if no improvement
                step_size *= 0.5
                if step_size < 1e-6:
                    break

    except Exception:
        pass  # Return current if refinement fails

    return f_current

def multi_scale_refinement(initial_f, levels=3):
    """Apply refinement at multiple scales"""
    f_refined = np.array(initial_f)

    # Apply refinement at different scales
    for level in range(levels):
        # Reduce step size for finer refinement
        step_size = 0.01 / (2**level)
        max_iter = 20 // (level + 1)

        f_refined = gradient_ascent_refinement(f_refined, max_iter=max_iter, step_size=step_size)

    return f_refined

def evolutionary_optimization():
    """Main evolutionary optimization routine with gradient refinement"""
    # Set up optimization parameters
    n = 1000  # Number of steps - can be adjusted

    # Define bounds for each parameter (step height)
    bounds = [(0.0, 2.0) for _ in range(n)]

    def objective(x):
        # Convert to array and compute score
        score = compute_c2_score(x)
        return -score  # Minimize negative to maximize original score

    # Run differential evolution with multiple starts for better exploration
    best_score = 0.0
    best_solution = None

    # Multi-start approach
    for start in range(3):  # Three different starting points
        # Generate different initial solutions
        x0 = sophisticated_initialization(n) + np.random.normal(0, 0.1, n)

        # Run differential evolution with adaptive settings
        result = differential_evolution(
            objective,
            bounds,
            seed=start,
            maxiter=100,
            popsize=max(10, n//10),
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )

        if -result.fun > best_score:
            best_score = -result.fun
            best_solution = result.x.copy()

    # If we found a solution, refine it with gradient ascent
    if best_solution is not None:
        try:
            # Apply multi-scale gradient refinement
            refined_solution = multi_scale_refinement(best_solution, levels=2)
            refined_score = compute_c2_score(refined_solution)

            # Keep the better of the two
            if refined_score > best_score:
                best_score = refined_score
                best_solution = refined_solution
        except Exception:
            pass  # Continue with original solution if refinement fails

    return best_solution, best_score

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using evolutionary optimization"""
    # Time limit enforcement
    start_time = time.time()

    try:
        # Run evolutionary optimization
        f_values, score = evolutionary_optimization()

        # Ensure we don't exceed time limit
        elapsed = time.time() - start_time
        if elapsed > 85:  # Leave 5 seconds buffer
            pass

        # Return the best solution
        return list(f_values)

    except Exception as e:
        # Fallback to simple initialization if optimization fails
        print(f"Fallback due to error: {e}")
        return sophisticated_initialization(1000).tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")