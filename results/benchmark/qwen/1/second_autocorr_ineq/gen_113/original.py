# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
import random
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
from scipy.optimize import differential_evolution, minimize
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Global constants
MAX_TIME_SECONDS = 90.0
DEFAULT_STEPS = 1000
MIN_STEPS = 100
MAX_STEPS = 5000
POPULATION_SIZE_BASE = 15
INITIAL_REFINEMENT_ITERATIONS = 50
FINAL_REFINEMENT_ITERATIONS = 30
STOCHASTIC_PERTURBATION_MAGNITUDE = 0.02

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals, dx):
    """
    Efficiently compute autoconvolution using Numba JIT compilation
    """
    n = len(f_vals)
    # Using the mathematical property that f*f = sum_{i,j} f[i]*f[j] * delta_{i+j}
    # For step functions, we can precompute the result more efficiently

    # Autoconvolution result has length 2*n - 1
    g = np.zeros(2*n - 1)

    # More efficient computation: for each pair of indices, accumulate into correct position
    # This is a direct implementation of discrete convolution
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < len(g):
                g[idx] += f_vals[i] * f_vals[j]

    return g

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """
    Compute L1, L2^2, and L-infinity norms efficiently
    """
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

@jit(nopython=True)
def compute_c2_numba(f_vals, dx):
    """
    Compute C2 value using optimized numba functions
    """
    # Compute autoconvolution
    g_vals = compute_autoconvolution_numba(f_vals, dx)

    # Compute norms
    l1, l2_sq, linf = compute_norms_numba(g_vals)

    # Avoid division by zero
    if l1 <= 1e-15 or linf <= 1e-15:
        return 0.0

    # Return C2 value
    return l2_sq / (l1 * linf)

def evaluate_step_function(f_vals):
    """
    Evaluate a step function and return C2 value
    """
    try:
        # Ensure non-negative values
        f_vals = np.array([max(0.0, x) for x in f_vals])

        # Use a fixed dx for consistent spacing
        dx = 0.5 / len(f_vals) if len(f_vals) > 0 else 0.001

        # Compute C2 value
        c2 = compute_c2_numba(f_vals, dx)
        return c2
    except Exception as e:
        return 0.0

def compute_multiobjective_score(f_vals):
    """
    Compute multi-objective score combining C2 and flatness penalty
    """
    try:
        # Ensure non-negative values
        f_vals = np.array([max(0.0, x) for x in f_vals])

        # Use a fixed dx for consistent spacing
        dx = 0.5 / len(f_vals) if len(f_vals) > 0 else 0.001

        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals, dx)

        # Compute norms
        l1, l2_sq, linf = compute_norms_numba(g_vals)

        # Avoid division by zero
        if l1 <= 1e-15 or linf <= 1e-15:
            return 0.0, 0.0

        # Calculate C2
        c2 = l2_sq / (l1 * linf)

        # Simple flatness heuristic: encourage low variance in g_vals
        if len(g_vals) > 1:
            variance = np.var(g_vals)
            mean_val = np.mean(g_vals)
            flatness = variance / (mean_val * mean_val + 1e-10)
        else:
            flatness = 0.0

        return c2, flatness
    except Exception as e:
        return 0.0, 0.0

def create_multi_scale_initialization(n_steps):
    """
    Create diverse initial solutions at multiple scales
    """
    # Different initialization strategies
    strategies = [
        lambda n: create_bell_shaped_pattern(n),
        lambda n: create_alternating_pattern(n),
        lambda n: create_peak_centered_pattern(n),
        lambda n: create_smooth_transition_pattern(n),
        lambda n: create_random_spatial_pattern(n)
    ]

    # Choose a strategy randomly
    strategy = np.random.choice(strategies)
    return strategy(n_steps)

def create_bell_shaped_pattern(n_steps):
    """Create a bell-shaped base pattern"""
    x = np.linspace(0, 1, n_steps)
    # Create Gaussian-like shape with emphasis on edges
    pattern = 1.0 + 0.8 * np.exp(-15 * (x - 0.5)**2) - 0.3 * np.exp(-5 * x**2) - 0.3 * np.exp(-5 * (1-x)**2)
    pattern = np.clip(pattern, 0, np.inf)
    return pattern / np.sum(pattern) * n_steps

def create_alternating_pattern(n_steps):
    """Create alternating high/low pattern"""
    pattern = []
    for i in range(n_steps):
        if i % 2 == 0:
            pattern.append(1.0 + np.random.random() * 0.5)
        else:
            pattern.append(0.2 + np.random.random() * 0.3)
    return np.array(pattern) / np.sum(pattern) * n_steps

def create_peak_centered_pattern(n_steps):
    """Create peak-centered pattern with tapering edges"""
    pattern = np.zeros(n_steps)
    center = n_steps // 2
    width = max(1, n_steps // 6 + np.random.randint(-1, 2))

    # Create a central peak
    pattern[max(0, center-width//2):min(n_steps, center+width//2)] = 1.0

    # Add tapering to edges
    for i in range(center - width//2):
        pattern[i] *= (i / (center - width//2))
    for i in range(center + width//2, n_steps):
        pattern[i] *= ((n_steps - i) / (width//2 + 1))

    # Add some noise
    noise = np.random.normal(0, 0.05, n_steps)
    pattern = pattern + noise
    pattern = np.clip(pattern, 0, np.inf)
    return pattern / np.sum(pattern) * n_steps

def create_smooth_transition_pattern(n_steps):
    """Create smooth transition pattern"""
    pattern = np.zeros(n_steps)
    # Create smooth ramp with some random variation
    for i in range(n_steps):
        x = i / (n_steps - 1) if n_steps > 1 else 0.5
        pattern[i] = 0.5 + 0.5 * np.sin(np.pi * x) + np.random.normal(0, 0.1)

    pattern = np.clip(pattern, 0, np.inf)
    return pattern / np.sum(pattern) * n_steps

def create_random_spatial_pattern(n_steps):
    """Create random spatial pattern"""
    pattern = np.random.random(n_steps)
    pattern = np.clip(pattern, 0, np.inf)
    return pattern / np.sum(pattern) * n_steps

def adaptive_evolutionary_optimization(initial_population):
    """
    Adaptive evolutionary optimization with dynamic population sizing
    """
    # Track convergence
    best_scores = []
    patience_counter = 0
    max_patience = 10
    population_size = POPULATION_SIZE_BASE

    # Start with initial population
    population = initial_population.copy()
    current_best = max(population, key=evaluate_step_function)
    best_scores.append(evaluate_step_function(current_best))

    # Adaptive parameters based on convergence behavior
    for generation in range(200):  # Limited to prevent timeout
        if len(population) < 10:
            population_size = 10
        elif len(population) > 30:
            population_size = 20
        else:
            population_size = len(population) // 2 + 5

        # Evaluate all individuals
        fitnesses = [evaluate_step_function(ind) for ind in population]

        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitnesses = [fitnesses[i] for i in sorted_indices]

        # Update best
        current_best = sorted_population[0]
        best_scores.append(sorted_fitnesses[0])

        # Check for convergence
        if len(best_scores) >= 5:
            recent_improvement = best_scores[-1] - best_scores[-5]
            if recent_improvement < 1e-6:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= max_patience:
                # Increase population size to escape local minimum
                population_size = min(population_size * 2, 50)
                patience_counter = 0

        # Create offspring using tournament selection and crossover
        new_population = []

        # Elitism: keep the best 20%
        elite_count = max(1, int(0.2 * population_size))
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

            # Mutation (add noise)
            mutation_strength = 0.1 * (1 - generation / 200.0)  # Decrease over time
            for i in range(len(parent)):
                if np.random.random() < 0.2:  # 20% chance to mutate each element
                    noise = np.random.normal(0, mutation_strength)
                    parent[i] = max(0, parent[i] + noise)

            new_population.append(parent)

        # Replace population
        population = new_population

        # Early termination based on time
        if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
            break

    return current_best

def advanced_refinement_strategy(initial_solution):
    """
    Advanced refinement using multiple optimization techniques
    """
    # Try different optimization methods
    best_solution = initial_solution.copy()
    best_c2 = evaluate_step_function(best_solution)

    # Method 1: Differential evolution on a subset
    try:
        # Reduce dimensionality for faster optimization
        reduced_solution = best_solution[:min(len(best_solution), 500)]
        bounds = [(0, 10.0) for _ in range(len(reduced_solution))]

        def objective(x):
            # Pad with zeros to original size
            extended_x = list(x) + [1.0] * (len(best_solution) - len(x))
            return -evaluate_step_function(extended_x)

        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=30,
            popsize=10,
            seed=42,
            disp=False
        )

        if de_result.success:
            refined_solution = np.maximum(de_result.x, 0)
            # Extend back to original size
            extended_refined = list(refined_solution) + [1.0] * (len(best_solution) - len(refined_solution))
            refined_c2 = evaluate_step_function(extended_refined)
            if refined_c2 > best_c2:
                best_solution = extended_refined
                best_c2 = refined_c2
    except Exception as e:
        pass

    # Method 2: Local refinement around best solution
    try:
        # Use scipy.optimize.minimize with L-BFGS-B
        x0 = np.array(best_solution[:min(200, len(best_solution))])
        bounds = [(0, 10.0)] * len(x0)

        def objective(x):
            # Extend to full size
            extended_x = list(x) + [1.0] * (len(best_solution) - len(x))
            return -evaluate_step_function(extended_x)

        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 20})

        if res.success:
            refined_solution = np.maximum(res.x, 0)
            # Extend back to original size
            extended_refined = list(refined_solution) + [1.0] * (len(best_solution) - len(refined_solution))
            refined_c2 = evaluate_step_function(extended_refined)
            if refined_c2 > best_c2:
                best_solution = extended_refined
                best_c2 = refined_c2
    except Exception as e:
        pass

    # Method 3: Add stochastic perturbations to escape local minima
    try:
        # Apply small random perturbations
        perturbed = best_solution.copy()
        for i in range(len(perturbed)):
            if np.random.random() < 0.1:  # 10% chance to perturb each element
                perturbation = np.random.normal(0, STOCHASTIC_PERTURBATION_MAGNITUDE)
                perturbed[i] = max(0, perturbed[i] + perturbation)

        # Normalize
        if np.sum(perturbed) > 0:
            perturbed = perturbed / np.sum(perturbed) * len(perturbed)

        perturbed_c2 = evaluate_step_function(perturbed)
        if perturbed_c2 > best_c2:
            best_solution = perturbed
            best_c2 = perturbed_c2
    except Exception as e:
        pass

    return best_solution

def construct_function() -> list[float]:
    """
    Optimized function to construct step-function with high C2 value.
    Uses adaptive evolutionary optimization with multi-scale initialization.
    """
    global start_time
    start_time = time.time()

    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Initialize with multi-scale approach
    initial_solutions = []
    n_attempts = 10
    target_steps = DEFAULT_STEPS

    for i in range(n_attempts):
        # Create diverse initial solutions
        n_steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        init_solution = create_multi_scale_initialization(n_steps)
        initial_solutions.append(init_solution)

    # Select best initial solution
    best_init = max(initial_solutions, key=evaluate_step_function)

    # Apply adaptive evolutionary optimization
    evolved_solution = adaptive_evolutionary_optimization(initial_solutions)

    # Refine the solution
    refined_solution = advanced_refinement_strategy(evolved_solution)

    # Final evaluation and return the best
    final_c2 = evaluate_step_function(refined_solution)
    initial_c2 = evaluate_step_function(best_init)

    if final_c2 > initial_c2:
        result = refined_solution
    else:
        result = best_init

    # Ensure proper length
    if len(result) < MIN_STEPS:
        result.extend([1.0] * (MIN_STEPS - len(result)))
    elif len(result) > MAX_STEPS:
        result = result[:MAX_STEPS]

    # Normalize if needed
    if np.sum(result) > 0:
        result = np.array(result) / np.sum(result) * len(result)

    # Ensure non-negativity and finite values
    result = np.clip(result, 0, np.inf)

    end_time = time.time()
    eval_time = end_time - start_time

    # Print debug info
    print(f"Eval time: {eval_time:.4f}s")
    print(f"Best C2 found: {evaluate_step_function(result):.6f}")

    return result.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")