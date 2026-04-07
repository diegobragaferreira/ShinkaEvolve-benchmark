# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import jit, njit
import random
import time
from scipy.signal import convolve
import copy
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 150
MUTATION_RATE = 0.8
CROSSOVER_PROB = 0.7
NUM_STARTS = 8
MAX_EVALUATIONS = 15000

@njit
def compute_autoconvolution_manual(f_vals):
    """
    Manual computation of autoconvolution for better performance with Numba
    """
    n = len(f_vals)
    if n == 0:
        return np.array([])

    # Allocate result array for autoconvolution
    g = np.zeros(2 * n - 1)

    # Manual convolution computation
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    return g

@njit
def compute_autoconvolution_norms(f_vals):
    """
    Compute the three norms of the autoconvolution g = f*f
    Returns ||g||₂², ||g||₁, ||g||∞
    """
    # Ensure input is numpy array
    f = np.array(f_vals, dtype=np.float64)
    f = np.maximum(f, 0)  # Clip negative values to 0

    if len(f) == 0:
        return 0.0, 0.0, 0.0

    # Compute autoconvolution manually for better control and performance
    g_full = compute_autoconvolution_manual(f)

    # Trim to match proper interval [-1/4, 1/4]
    # This assumes we're working with the central region
    half_len = len(f)
    g_center = len(g_full) // 2
    g_trimmed = g_full[g_center - half_len : g_center + half_len]

    # Compute norms using trapezoidal integration for ||g||₂²
    if len(g_trimmed) < 2:
        norm_l2_sq = 0.0
    else:
        # Trapezoidal integration formula for piecewise linear segments
        # Each segment contributes (width/3)*(y1^2 + y1*y2 + y2^2)
        step_width = 0.5 / len(f)
        g_abs = np.abs(g_trimmed)
        widths = np.full(len(g_abs)-1, step_width)
        y1 = g_abs[:-1]
        y2 = g_abs[1:]
        norm_l2_sq = np.sum(widths * (y1**2 + y1*y2 + y2**2) / 3.0)

    # ||g||_1 = sum of absolute values normalized
    norm_l1 = np.sum(np.abs(g_trimmed)) / (len(g_trimmed) + 1) if len(g_trimmed) > 0 else 1e-15

    # ||g||_∞ = max absolute value
    norm_inf = np.max(np.abs(g_trimmed)) if len(g_trimmed) > 0 else 1e-15

    return norm_l2_sq, norm_l1, norm_inf

@njit
def compute_c2_score(f_vals):
    """
    Compute the C2 score: ||g||₂² / (||g||₁ · ||g||∞)
    """
    norm_l2_sq, norm_l1, norm_inf = compute_autoconvolution_norms(f_vals)

    # Avoid division by zero
    if norm_l1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    return norm_l2_sq / (norm_l1 * norm_inf)

# JAX version of the core computation for automatic differentiation
@jax_jit
def compute_autoconvolution_jax(f_vals):
    """JAX version of autoconvolution for gradient computation"""
    f = jnp.array(f_vals)
    f = jnp.maximum(f, 0)  # Clip negative values to 0
    if len(f) == 0:
        return jnp.array([])

    # Using JAX's convolution operation
    # This is a simplified approach - for exact matching, we'd need
    # to replicate the exact indexing logic from compute_autoconvolution_manual
    # But for gradient computation purposes, this works well
    g = jnp.convolve(f, f, mode='full')

    # Trim to center portion (this matches our manual computation)
    half_len = len(f)
    center_start = len(g) // 2 - half_len + 1
    center_end = len(g) // 2 + half_len - 1
    g_trimmed = g[center_start:center_end]

    return g_trimmed

@jax_jit
def compute_c2_jax(f_vals):
    """JAX version of C2 computation for gradient computation"""
    g = compute_autoconvolution_jax(f_vals)

    # Compute norms
    norm_l2_sq = jnp.sum(g * g)
    norm_l1 = jnp.sum(jnp.abs(g)) / (len(g) + 1) if len(g) > 0 else 1e-15
    norm_inf = jnp.max(jnp.abs(g)) if len(g) > 0 else 1e-15

    # Avoid division by zero
    return norm_l2_sq / (norm_l1 * norm_inf)

# Gradient computation function using JAX
def evaluate_c2_jax(params):
    """Evaluate C2 using JAX for accurate gradient computation"""
    try:
        # Clip negative values
        f_vals = np.clip(params, 0, None)

        # Convert to JAX array for gradient computation
        f_jax = jnp.array(f_vals)

        # Compute C2 using JAX version
        c2 = compute_c2_jax(f_jax)
        return float(c2)
    except Exception as e:
        return 0.0

def compute_c2_gradient_jax(params):
    """Compute gradient of C2 with respect to parameters using JAX"""
    try:
        # Clip negative values
        f_vals = np.clip(params, 0, None)

        # Convert to JAX array
        f_jax = jnp.array(f_vals)

        # Compute gradient using JAX automatic differentiation
        grad_func = grad(compute_c2_jax)
        gradients = grad_func(f_jax)

        return np.array(gradients)
    except Exception as e:
        return np.zeros_like(params)

def generate_multi_scale_initialization(n):
    """Generate initial population with multi-scale patterns for better exploration"""
    patterns = []

    # Pattern 1: Multi-peak Gaussian with varying scales
    x = np.linspace(-1, 1, n)
    pattern1 = np.zeros(n)
    for i in range(3):  # Three peaks
        center = -0.5 + i * 0.5
        width = 0.2 + random.random() * 0.3
        height = 0.8 + random.random() * 0.4
        pattern1 += height * np.exp(-((x - center)**2) / (2 * width**2))
    patterns.append(pattern1.tolist())

    # Pattern 2: Alternating pattern with high/low values
    pattern2 = []
    for i in range(n):
        base_val = 1.5 if i % 3 == 0 else 0.3 if i % 3 == 1 else 0.8
        pattern2.append(base_val + random.random() * 0.3)
    patterns.append(pattern2)

    # Pattern 3: Sinusoidal pattern with modulation
    pattern3 = []
    for i in range(n):
        x_pos = i / (n - 1) if n > 1 else 0.5
        base = 1.0 + 0.3 * np.sin(8 * np.pi * x_pos)
        mod = 0.2 * np.cos(12 * np.pi * x_pos)
        pattern3.append(max(0.0, base + mod))
    patterns.append(pattern3)

    # Pattern 4: Single dominant peak with decay
    pattern4 = []
    center = n // 2
    for i in range(n):
        distance = abs(i - center) / (n // 2)
        val = max(0.0, 1.0 * np.exp(-2 * distance**2))
        pattern4.append(val + 0.1 * random.random())
    patterns.append(pattern4)

    # Pattern 5: Random with heavy-tailed distribution
    pattern5 = []
    for i in range(n):
        # Heavy-tailed distribution
        r = random.random()
        if r < 0.7:
            pattern5.append(0.1 + 0.3 * random.random())
        else:
            pattern5.append(1.0 + 2.0 * random.random())
    patterns.append(pattern5)

    # Select the best performing pattern among these
    best_pattern = patterns[0]
    best_score = -1.0

    for p in patterns:
        try:
            score = compute_c2_score(p)
            if score > best_score:
                best_score = score
                best_pattern = p
        except:
            continue

    return best_pattern

def adaptive_evolutionary_optimization():
    """Enhanced evolutionary optimization with adaptive parameters"""
    best_c2 = 0.0
    best_f = None

    # Try multiple random starts with different initialization strategies
    for start_num in range(NUM_STARTS):
        # Vary population size and generation count based on start number
        pop_size = POPULATION_SIZE + (start_num % 3) * 20  # Vary population size
        gen_count = GENERATIONS + (start_num % 2) * 50     # Vary generations

        # Initialize population with diverse multi-scale patterns
        population = []

        for i in range(pop_size):
            # Adaptive sizing: smaller for early starts, larger for later
            if start_num < NUM_STARTS // 2:
                n = random.randint(300, 800)
            else:
                n = random.randint(500, 1200)

            individual = generate_multi_scale_initialization(n)
            # Ensure non-negative values
            individual = [max(0.0, x) for x in individual]
            population.append(individual)

        # Check if we have a valid population
        if len(population) == 0:
            continue

        # Differential evolution with custom bounds
        bounds = [(0.0, 3.0) for _ in range(len(population[0]))]

        def objective(x):
            # Convert to list of floats
            f_vals = [float(xi) for xi in x]
            try:
                c2 = compute_c2_score(f_vals)
                return -c2  # Negative because we maximize
            except:
                return 1e10  # Penalty for invalid solutions

        try:
            # Run differential evolution with adaptive parameters
            result = differential_evolution(
                objective,
                bounds,
                maxiter=gen_count,
                popsize=pop_size,
                mutation=MUTATION_RATE,
                recombination=CROSSOVER_PROB,
                seed=start_num,
                disp=False,
                atol=1e-6,
                rtol=1e-6
            )

            if result.success:
                final_solution = [max(0.0, float(x)) for x in result.x]
                final_c2 = compute_c2_score(final_solution)

                if final_c2 > best_c2:
                    best_c2 = final_c2
                    best_f = final_solution

        except Exception as e:
            continue

    # If we found a solution, apply advanced local refinement with JAX gradients
    if best_f is not None and best_c2 > 0:
        try:
            # Apply multiple refinement steps with different techniques
            current_solution = copy.deepcopy(best_f)
            current_c2 = best_c2

            # Enhanced refinement with JAX gradient information
            # First, try L-BFGS-B refinement using JAX gradients
            bounds = [(0.0, 3.0) for _ in range(len(current_solution))]

            def local_objective(x):
                # Ensure non-negativity
                x = [max(0.0, xi) for xi in x]
                return -compute_c2_score(x)

            # For JAX-based refinement, we'll use a simple gradient ascent approach
            # This is more reliable than trying to use scipy.optimize with JAX gradients
            # directly due to compatibility issues

            # Alternative: Apply gradient-based refinement with manual gradient
            # This might be more stable for hybrid optimization

            # Simple gradient ascent approach with JAX
            # We perform limited gradient updates based on JAX gradients
            max_iter = 50
            learning_rate = 0.001

            # Convert to jax array
            current_jax = jnp.array(current_solution)

            for i in range(max_iter):
                try:
                    # Compute current C2 score
                    current_c2_val = evaluate_c2_jax(current_solution)

                    # Compute gradients using JAX
                    grad_val = compute_c2_gradient_jax(current_solution)

                    # Update using gradient ascent
                    updated_solution = current_jax + learning_rate * grad_val
                    updated_solution = jnp.maximum(updated_solution, 0)  # Keep non-negative

                    # Convert back to list
                    updated_list = list(updated_solution)

                    # Evaluate new score
                    new_c2 = compute_c2_score(updated_list)

                    if new_c2 > current_c2_val:
                        current_solution = updated_list
                        current_c2 = new_c2
                    else:
                        # If no improvement, reduce learning rate
                        learning_rate *= 0.9
                        if learning_rate < 1e-6:
                            break

                except Exception as e:
                    break  # If something goes wrong, stop the gradient updates

            # Final check of improved solution
            final_c2 = compute_c2_score(current_solution)
            if final_c2 > best_c2:
                best_c2 = final_c2
                best_f = current_solution

        except Exception as e:
            pass

    return best_f if best_f is not None else generate_multi_scale_initialization(500)

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Run the enhanced optimization
    start_time = time.time()
    try:
        f_values = adaptive_evolutionary_optimization()
        elapsed_time = time.time() - start_time
        # Add safety check to ensure valid output
        if f_values is None or len(f_values) == 0:
            # Fallback to simple uniform distribution
            f_values = [0.5] * 200
    except:
        # Final fallback
        f_values = [0.5] * 200

    # Ensure we return a reasonable-sized list
    if len(f_values) < 50:
        f_values = f_values + [0.5] * (50 - len(f_values))
    elif len(f_values) > 10000:
        f_values = f_values[:10000]

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")