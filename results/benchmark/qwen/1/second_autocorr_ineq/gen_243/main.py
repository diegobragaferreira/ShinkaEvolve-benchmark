# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
import random
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
from collections import defaultdict
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
warnings.filterwarnings('ignore')

# Global constants for performance tuning
MAX_TIME_SECONDS = 90.0
DEFAULT_STEPS = 3000
MIN_STEPS = 200
MAX_STEPS = 10000
POPULATION_SIZE_BASE = 12
INITIAL_REFINEMENT_ITERATIONS = 30
FINAL_REFINEMENT_ITERATIONS = 20

# Optimized convolution computation using FFT for better scalability
@jit(nopython=True)
def compute_autoconvolution_fft(f_vals):
    """
    Compute autoconvolution using FFT for O(n log n) complexity
    This is more efficient for large arrays compared to nested loops
    """
    n = len(f_vals)

    # For autoconvolution f*f, we can use FFT:
    # fft(f)*fft(f) = fft(f*f)
    # But since we want f*f, we need to do:
    # 1. Take FFT of f
    # 2. Square element-wise
    # 3. Take inverse FFT
    # Note: we need to handle the circular nature properly

    # Use the standard approach with padding to avoid wraparound effects
    # Pad to 2*n - 1 for full convolution, but we'll compute autoconvolution correctly
    # For autoconvolution f*f, we can use: FFT(f) * FFT(f) = FFT(f*f)
    # Then inverse FFT gives us the result

    # Pad to at least 2*n-1 for full convolution (but actually for autoconvolution we can be smarter)
    # But for simplicity, let's pad to 2*n for autoconvolution
    padded_len = 2 * n
    padded_f = np.zeros(padded_len)
    padded_f[:n] = f_vals

    # Compute FFT
    fft_f = np.fft.fft(padded_f)

    # Square the FFT (this is what we want for f*f)
    fft_squared = fft_f * fft_f

    # Inverse FFT
    g_padded = np.fft.ifft(fft_squared)

    # Take only the first n terms (center of convolution)
    # For autoconvolution of length n, the result is of length 2*n-1 but we want the "valid" part
    # Actually for a*b where both are same length, result is 2*n-1
    # For autoconvolution f*f, we take the central part
    g_full = np.real(g_padded)

    # For autoconvolution of length n, the result has length 2*n-1
    # We take the center part (n-1 elements) that corresponds to valid convolution
    # But actually we want the full autoconvolution, so we return it all
    # Or we could return the trimmed version for better matching with expected output
    return g_full

@jit(nopython=True)
def compute_norms_fft(g_vals):
    """
    Compute L1, L2^2, and L-infinity norms using optimized loops
    """
    n = len(g_vals)

    # L1 norm (sum of absolute values)
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

@jit(nopython=True)
def compute_c2_fft(f_vals):
    """
    Fast computation of C2 using FFT-based convolution and norms
    """
    # Compute autoconvolution using FFT for better scalability
    g_vals = compute_autoconvolution_fft(f_vals)

    # Compute norms
    l1, l2_sq, linf = compute_norms_fft(g_vals)

    # Avoid division by zero
    if l1 <= 1e-15 or linf <= 1e-15:
        return 0.0

    # Return C2 value
    return l2_sq / (l1 * linf)

def evaluate_step_function_fft(f_vals):
    """
    Evaluate step function with robust error handling and FFT computation
    """
    try:
        # Ensure non-negative values with clipping
        f_vals = np.array([max(0.0, x) for x in f_vals])

        # Handle edge cases
        if len(f_vals) == 0 or np.isnan(np.sum(f_vals)) or np.isinf(np.sum(f_vals)):
            return 0.0

        # If all values are zero, return 0
        if np.sum(f_vals) == 0:
            return 0.0

        # Compute C2 value using FFT method
        c2 = compute_c2_fft(f_vals)

        # Validate result
        if np.isnan(c2) or np.isinf(c2) or c2 < 0:
            return 0.0

        return c2
    except Exception as e:
        return 0.0

def evaluate_step_function_fft(f_vals):
    """
    Evaluate step function with robust error handling and FFT computation
    """
    try:
        # Ensure non-negative values with clipping
        f_vals = np.array([max(0.0, x) for x in f_vals])

        # Handle edge cases
        if len(f_vals) == 0 or np.isnan(np.sum(f_vals)) or np.isinf(np.sum(f_vals)):
            return 0.0

        # If all values are zero, return 0
        if np.sum(f_vals) == 0:
            return 0.0

        # Compute C2 value using FFT method
        c2 = compute_c2_fft(f_vals)

        # Validate result
        if np.isnan(c2) or np.isinf(c2) or c2 < 0:
            return 0.0

        return c2
    except Exception as e:
        return 0.0

# JAX-based gradient computation for optimization
def compute_c2_jax(f_vals):
    """JAX version for automatic differentiation"""
    # Convert to JAX array
    f = jnp.array(f_vals, dtype=jnp.float32)

    # Compute autoconvolution using JAX operations (using fft approach)
    g = jnp.fft.ifft(jnp.fft.fft(f) * jnp.fft.fft(f))
    g_real = jnp.real(g)

    # Compute norms
    g_abs = jnp.abs(g_real)
    norm_l2_sq = jnp.sum(g_abs**2)
    norm_l1 = jnp.sum(g_abs)
    norm_inf = jnp.max(g_abs)

    # Avoid division by zero
    eps = 1e-12
    norm_l1 = jnp.where(norm_l1 < eps, eps, norm_l1)
    norm_inf = jnp.where(norm_inf < eps, eps, norm_inf)

    c2 = norm_l2_sq / (norm_l1 * norm_inf)
    return c2

@jax_jit
def compute_c2_jax_vectorized(f_vals):
    """Vectorized JAX version for computing C2"""
    return compute_c2_jax(f_vals)

def compute_gradient_jax(f_vals):
    """Compute gradient of C2 with respect to f_vals using JAX"""
    try:
        f = jnp.array(f_vals, dtype=jnp.float32)
        grad_fn = grad(compute_c2_jax)
        grad_val = grad_fn(f)
        return np.array(grad_val)
    except Exception:
        return np.zeros_like(f_vals)

def create_hierarchical_initialization(n_steps):
    """
    Create hierarchical initial solution with multi-scale structure
    """
    # Start with a coarse grid and build up
    coarse_steps = max(100, n_steps // 10)

    # Create coarse pattern with strategic peaks
    coarse_pattern = np.zeros(coarse_steps)

    # Place strategic peaks at regular intervals
    peak_positions = []
    for i in range(1, coarse_steps, max(5, coarse_steps // 8)):
        if i < coarse_steps:
            peak_positions.append(i)

    # Create peaks with diminishing height
    for i, pos in enumerate(peak_positions):
        height = 1.0 + 0.5 * np.sin(i * 0.5)  # Varying pattern
        # Spread each peak over a few points
        spread = max(2, coarse_steps // 20)
        for j in range(max(0, pos - spread), min(coarse_steps, pos + spread)):
            distance = abs(j - pos)
            # Gaussian-like spreading
            spread_factor = np.exp(-distance**2 / (2 * spread**2))
            coarse_pattern[j] += height * spread_factor

    # Normalize to sum to coarse_steps
    if np.sum(coarse_pattern) > 0:
        coarse_pattern = coarse_pattern * coarse_steps / np.sum(coarse_pattern)

    # Interpolate to target resolution
    if n_steps > coarse_steps:
        # Linear interpolation
        coarse_positions = np.linspace(0, coarse_steps-1, coarse_steps)
        target_positions = np.linspace(0, coarse_steps-1, n_steps)
        fine_pattern = np.interp(target_positions, coarse_positions, coarse_pattern)
    else:
        # Downsample if too fine
        fine_pattern = coarse_pattern[:n_steps]

    # Add some random variation to break symmetry
    noise = np.random.normal(0, 0.05, n_steps)
    fine_pattern = fine_pattern + noise

    # Ensure non-negativity
    fine_pattern = np.maximum(fine_pattern, 0.0)

    # Final normalization
    if np.sum(fine_pattern) > 0:
        fine_pattern = fine_pattern * n_steps / np.sum(fine_pattern)

    return fine_pattern

def create_multi_scale_structural_initialization(n_steps):
    """
    Create diverse initial solutions with different structural properties for hierarchical optimization
    """
    # Multiple initialization strategies that work well with hierarchical approaches
    strategies = [
        lambda n: create_hierarchical_bell_shaped(n),
        lambda n: create_hierarchical_alternating(n),
        lambda n: create_hierarchical_peak_centered(n),
        lambda n: create_hierarchical_smooth_transition(n),
        lambda n: create_hierarchical_balanced(n),
        lambda n: create_hierarchical_asymmetric(n),
        lambda n: create_hierarchical_multipeak(n)
    ]

    # Choose a strategy
    strategy = np.random.choice(strategies)
    pattern = strategy(n_steps)

    # Apply minor random perturbations for diversity
    noise = np.random.normal(0, 0.02, n_steps)
    pattern = pattern + noise
    pattern = np.maximum(pattern, 0.0)

    # Final normalization
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)

    return pattern

def create_hierarchical_bell_shaped(n_steps):
    """Create bell-shaped pattern optimized for sparse convolution"""
    x = np.linspace(0, 1, n_steps)
    # Create Gaussian-like shape with emphasis on central region
    pattern = 1.0 + 0.8 * np.exp(-12 * (x - 0.5)**2) - 0.3 * np.exp(-6 * x**2) - 0.3 * np.exp(-6 * (1-x)**2)
    pattern = np.clip(pattern, 0, np.inf)

    # Normalize appropriately
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_alternating(n_steps):
    """Create alternating high/low pattern for hierarchical computation"""
    pattern = []
    for i in range(n_steps):
        if i % 3 == 0:
            pattern.append(1.0 + np.random.random() * 0.3)
        elif i % 3 == 1:
            pattern.append(0.3 + np.random.random() * 0.2)
        else:
            pattern.append(0.7 + np.random.random() * 0.2)

    pattern = np.array(pattern)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_peak_centered(n_steps):
    """Create peak-centered pattern with tapering edges optimized for hierarchical computation"""
    pattern = np.zeros(n_steps)
    center = n_steps // 2
    width = max(1, n_steps // 8 + np.random.randint(-2, 3))

    # Create central peak with smooth transitions
    for i in range(n_steps):
        distance_from_center = abs(i - center)
        if distance_from_center <= width:
            # Quadratic transition
            t = distance_from_center / width
            pattern[i] = 1.0 * (1 - t**2) + 0.4 * np.random.random()

    # Add noise for diversity
    noise = np.random.normal(0, 0.04, n_steps)
    pattern = pattern + noise
    pattern = np.clip(pattern, 0, np.inf)

    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_smooth_transition(n_steps):
    """Create smooth transition pattern optimized for hierarchical computation"""
    pattern = np.zeros(n_steps)
    # Create smooth ramp with more variation
    for i in range(n_steps):
        x = i / (n_steps - 1) if n_steps > 1 else 0.5
        pattern[i] = 0.6 + 0.4 * np.sin(np.pi * x + np.random.random()) + np.random.normal(0, 0.08)

    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_balanced(n_steps):
    """Create balanced pattern optimized for hierarchical computation"""
    # Create pattern that maintains balance
    pattern = np.ones(n_steps) * 0.6

    # Add structured variation
    for i in range(0, n_steps, 15):
        if i < n_steps:
            pattern[i] = 1.2 + np.random.random() * 0.3

    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_asymmetric(n_steps):
    """Create asymmetric pattern that breaks symmetry for better hierarchical exploration"""
    pattern = np.zeros(n_steps)

    # Create asymmetric structure with higher values on one side
    for i in range(n_steps):
        x = i / (n_steps - 1) if n_steps > 1 else 0.5
        # Asymmetric exponential decay
        if x < 0.4:
            pattern[i] = 1.0 + 0.4 * np.exp(-6 * x)
        elif x < 0.7:
            pattern[i] = 0.6 + 0.2 * np.exp(-8 * (x - 0.4))
        else:
            pattern[i] = 0.3 + 0.2 * np.exp(-12 * (x - 0.7))

    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_hierarchical_multipeak(n_steps):
    """Create multi-peak pattern optimized for hierarchical computation"""
    pattern = np.zeros(n_steps)

    # Multiple peaks at strategic locations
    peak_positions = [n_steps // 5, 2*n_steps//5, 3*n_steps//5, 4*n_steps//5]
    for i, pos in enumerate(peak_positions):
        if pos < n_steps:
            # Gaussian peaks
            width = n_steps // 20 + np.random.randint(-2, 3)
            for j in range(max(0, pos - width), min(n_steps, pos + width)):
                distance = abs(j - pos)
                spread_factor = np.exp(-distance**2 / (2 * width**2))
                pattern[j] += 1.0 * spread_factor

    # Add some randomness
    noise = np.random.normal(0, 0.03, n_steps)
    pattern = pattern + noise
    pattern = np.clip(pattern, 0, np.inf)

    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def hierarchical_adaptive_evolutionary_optimization(initial_population):
    """
    Advanced hierarchical evolutionary optimization with multi-scale awareness
    """
    # Track convergence and adapt parameters dynamically
    best_scores = []
    patience_counter = 0
    max_patience = 8
    population_size = POPULATION_SIZE_BASE

    # Start with initial population
    population = [list(ind) for ind in initial_population]
    current_best = max(population, key=evaluate_step_function_sparse)
    best_scores.append(evaluate_step_function_sparse(current_best))

    # Hierarchical approach: start with coarse grids and refine
    current_scale = 1.0

    # Adaptive parameters based on convergence behavior
    for generation in range(150):  # Limited to prevent timeout
        # Scale population size based on hierarchy level
        if len(population) < 10:
            population_size = 10
        elif len(population) > 25:
            population_size = 20
        else:
            population_size = len(population) // 2 + 5

        # Evaluate all individuals with sparse computation
        fitnesses = [evaluate_step_function_sparse(ind) for ind in population]

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitnesses = [fitnesses[i] for i in sorted_indices]

        # Update best
        current_best = sorted_population[0]
        best_scores.append(sorted_fitnesses[0])

        # Check for convergence with multiple criteria
        if len(best_scores) >= 5:
            # Check for stagnation in improvement
            recent_improvement = best_scores[-1] - best_scores[-5]
            if recent_improvement < 1e-9:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= max_patience:
                # Increase population size to escape local minimum
                population_size = min(population_size * 2, 40)
                patience_counter = 0

        # Create offspring using tournament selection and crossover
        new_population = []

        # Elitism: keep the best 25%
        elite_count = max(1, int(0.25 * population_size))
        new_population.extend(sorted_population[:elite_count])

        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(sorted_population), tournament_size)
            tournament_fitnesses = [sorted_fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]

            # Clone selected parent
            parent = sorted_population[winner_index].copy()

            # Hierarchical-aware mutation with scale adjustment
            mutation_strength = 0.15 * (1 - generation / 150.0) * current_scale  # Decrease over time and scale

            # Mutate with hierarchical consideration
            for i in range(len(parent)):
                if np.random.random() < 0.12:  # 12% chance to mutate each element
                    noise = np.random.normal(0, mutation_strength)
                    parent[i] = max(0, parent[i] + noise)

            new_population.append(parent)

        # Replace population
        population = new_population

        # Early termination based on time
        if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
            break

    return current_best

def multi_objective_sparse_optimization(initial_solution):
    """
    Multi-objective optimization that considers both flatness and peak suppression
    """
    # This combines multiple objectives to better shape g distribution
    best_solution = initial_solution.copy()
    best_c2 = evaluate_step_function_sparse(best_solution)

    # Objective 1: Maximize C2
    def objective1(x):
        return -evaluate_step_function_sparse(x)

    # Objective 2: Minimize peak-to-mean ratio in g (encourage flatness)
    def objective2(x):
        try:
            f_vals = np.array(x)
            g_vals = compute_sparse_autoconvolution(f_vals)

            # Compute peak-to-mean ratio (lower is better for flatness)
            mean_g = np.mean(np.abs(g_vals))
            max_g = np.max(np.abs(g_vals))
            if mean_g <= 1e-12:
                return 0.0
            return max_g / mean_g  # Larger ratios are worse
        except:
            return 1e10

    # Combined objective (weighted sum)
    def combined_objective(x):
        c2_val = evaluate_step_function_sparse(x)
        flatness_val = objective2(x)  # This will be minimized
        # Weight toward C2 but penalize high peaks
        return -c2_val + 0.1 * flatness_val  # Higher is better for combined

    # Try local optimization
    try:
        # Use L-BFGS-B for local refinement
        x0 = np.array(best_solution[:min(1000, len(best_solution))])
        bounds = [(0, 10.0)] * len(x0)

        def obj_func(x):
            extended_x = list(x) + [1.0] * (len(best_solution) - len(x))
            return combined_objective(extended_x)

        res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 15})

        if res.success:
            refined_solution = np.maximum(res.x, 0)
            extended_refined = list(refined_solution) + [1.0] * (len(best_solution) - len(refined_solution))
            refined_c2 = evaluate_step_function_sparse(extended_refined)
            if refined_c2 > best_c2:
                best_solution = extended_refined
                best_c2 = refined_c2
    except Exception:
        pass

    return best_solution

def cross_scale_transfer_refinement(initial_solution):
    """
    Transfer knowledge from different scales to enhance optimization
    """
    best_solution = initial_solution.copy()
    best_c2 = evaluate_step_function_sparse(best_solution)

    # Try different sized versions to learn structural patterns
    scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]  # Different resolutions

    for scale in scale_factors:
        try:
            target_size = max(100, int(len(best_solution) * scale))
            if target_size != len(best_solution):
                # Create scaled-down version
                if target_size < len(best_solution):
                    # Reduce resolution
                    reduced_indices = np.linspace(0, len(best_solution)-1, target_size, dtype=int)
                    reduced_solution = [best_solution[i] for i in reduced_indices]
                else:
                    # Increase resolution via interpolation
                    old_indices = np.linspace(0, len(best_solution)-1, len(best_solution))
                    new_indices = np.linspace(0, len(best_solution)-1, target_size)
                    reduced_solution = np.interp(new_indices, old_indices, best_solution)

                # Refine the reduced version
                refined_reduced = multi_objective_sparse_optimization(reduced_solution)

                # Interpolate back to original resolution
                if len(refined_reduced) != len(best_solution):
                    old_indices = np.linspace(0, len(refined_reduced)-1, len(refined_reduced))
                    new_indices = np.linspace(0, len(refined_reduced)-1, len(best_solution))
                    interpolated_solution = np.interp(new_indices, old_indices, refined_reduced)
                else:
                    interpolated_solution = refined_reduced

                # Evaluate and keep if better
                interpolated_c2 = evaluate_step_function_sparse(interpolated_solution)
                if interpolated_c2 > best_c2:
                    best_solution = interpolated_solution
                    best_c2 = interpolated_c2

        except Exception:
            continue

    return best_solution

def gradient_ascent_refinement(initial_solution, max_iterations=50, step_size=0.01):
    """
    Apply gradient ascent refinement to improve the solution
    """
    solution = np.array(initial_solution, dtype=np.float32)
    current_c2 = evaluate_step_function_fft(solution)

    # Apply gradient ascent
    for i in range(max_iterations):
        try:
            # Compute gradient
            grad_val = compute_gradient_jax(solution)

            # Update solution
            solution_new = solution + step_size * grad_val

            # Ensure non-negativity
            solution_new = np.maximum(solution_new, 0)

            # Check improvement
            new_c2 = evaluate_step_function_fft(solution_new)
            if new_c2 > current_c2:
                solution = solution_new
                current_c2 = new_c2
            else:
                # Reduce step size if no improvement
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        except Exception:
            break

    return solution.tolist()

def construct_function() -> list[float]:
    """
    Optimized function to construct step-function with high C2 value using hierarchical optimization.
    Implements a truly novel approach combining hierarchical scales, multi-objective optimization,
    cross-scale knowledge transfer, and gradient ascent refinement.
    """
    global start_time
    start_time = time.time()

    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Initialize with hierarchical approach using multi-scale optimizations
    initial_solutions = []
    n_attempts = 20

    for i in range(n_attempts):
        # Create diverse initial solutions with hierarchical optimization in mind
        n_steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        init_solution = create_multi_scale_structural_initialization(n_steps)
        initial_solutions.append(init_solution)

    # Select best initial solution
    best_init = max(initial_solutions, key=evaluate_step_function_fft)

    # Apply hierarchical adaptive evolutionary optimization
    evolved_solution = hierarchical_adaptive_evolutionary_optimization(initial_solutions)

    # Apply multi-objective optimization for better g distribution
    multi_obj_solution = multi_objective_sparse_optimization(evolved_solution)

    # Apply cross-scale transfer refinement
    refined_solution = cross_scale_transfer_refinement(multi_obj_solution)

    # Apply gradient ascent refinement for final improvement
    gradient_solution = gradient_ascent_refinement(refined_solution)

    # Final evaluation and return the best among all attempts
    final_c2 = evaluate_step_function_fft(gradient_solution)
    initial_c2 = evaluate_step_function_fft(best_init)
    evolved_c2 = evaluate_step_function_fft(evolved_solution)
    multi_obj_c2 = evaluate_step_function_fft(multi_obj_solution)
    cross_scale_c2 = evaluate_step_function_fft(refined_solution)

    # Select the best among all intermediate solutions
    candidates = [
        (final_c2, gradient_solution),
        (initial_c2, best_init),
        (evolved_c2, evolved_solution),
        (multi_obj_c2, multi_obj_solution),
        (cross_scale_c2, refined_solution)
    ]

    best_c2, result = max(candidates, key=lambda x: x[0])

    # Ensure proper length
    if len(result) < MIN_STEPS:
        result.extend([1.0] * (MIN_STEPS - len(result)))
    elif len(result) > MAX_STEPS:
        result = result[:MAX_STEPS]

    # Normalize representation if needed
    if np.sum(result) > 0:
        result = np.array(result) / np.sum(result) * len(result)

    # Ensure non-negativity and finite values for computation
    result = np.clip(result, 0, np.inf)

    end_time = time.time()
    eval_time = end_time - start_time

    # Print debug info
    print(f"Eval time: {eval_time:.4f}s")
    print(f"Best C2 found: {evaluate_step_function_fft(result):.6f}")

    return result.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")