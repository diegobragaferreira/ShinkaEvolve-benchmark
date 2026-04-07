# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
import random
from scipy.optimize import differential_evolution
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Global constants for optimization
MAX_TIME_SECONDS = 90.0
DEFAULT_STEPS = 1000
MIN_STEPS = 100
MAX_STEPS = 5000
POPULATION_SIZE_BASE = 15

# Numba optimized versions for performance-critical parts
@jit(nopython=True)
def compute_autoconvolution_numba(f_vals, dx):
    """
    Efficiently compute autoconvolution using Numba JIT compilation
    """
    n = len(f_vals)
    # Autoconvolution result has length 2*n - 1
    g = np.zeros(2*n - 1)

    # Direct implementation of discrete convolution
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
def compute_c2_piecewise_numba(f_vals, dx):
    """
    Compute C2 value using optimized numba functions with proper piecewise linear integration
    """
    # Compute autoconvolution
    g_vals = compute_autoconvolution_numba(f_vals, dx)

    # Compute norms using piecewise integration for ||g||₂²
    # For consecutive pairs (y1, y2) with spacing dx:
    # integral of y^2 ≈ dx/3 * (y1^2 + y1*y2 + y2^2)
    l2_sq_norm = 0.0
    for i in range(len(g_vals) - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        l2_sq_norm += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0

    # L1 norm (sum of absolute values)
    l1_norm = 0.0
    for i in range(len(g_vals)):
        l1_norm += abs(g_vals[i])

    # L-infinity norm (maximum absolute value)
    linf_norm = 0.0
    for i in range(len(g_vals)):
        abs_val = abs(g_vals[i])
        if abs_val > linf_norm:
            linf_norm = abs_val

    # Avoid division by zero
    if l1_norm <= 1e-15 or linf_norm <= 1e-15:
        return 0.0

    # Return C2 value
    return l2_sq_norm / (l1_norm * linf_norm)

def evaluate_step_function(f_vals):
    """
    Evaluate a step function and return C2 value using piecewise linear integration
    """
    try:
        # Ensure non-negative values
        f_vals = np.array([max(0.0, x) for x in f_vals])

        # Use a fixed dx for consistent spacing
        dx = 0.5 / len(f_vals) if len(f_vals) > 0 else 0.001

        # Compute C2 value with proper piecewise integration
        c2 = compute_c2_piecewise_numba(f_vals, dx)
        return c2
    except Exception as e:
        return 0.0

def create_multi_scale_initialization(n_steps):
    """
    Create diverse initial solutions at multiple scales
    """
    # Different initialization strategies
    strategies = [
        create_bell_shaped_pattern,
        create_alternating_pattern,
        create_peak_centered_pattern,
        create_smooth_transition_pattern,
        create_random_spatial_pattern
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

def adaptive_evolutionary_optimization(initial_solutions):
    """
    Enhanced adaptive evolutionary optimization using multiple initial solutions
    """
    # Track convergence
    best_scores = []
    patience_counter = 0
    max_patience = 10
    population_size = POPULATION_SIZE_BASE

    # Start with best initial solution
    population = initial_solutions.copy()
    
    # Evaluate all individuals
    fitnesses = [evaluate_step_function(ind) for ind in population]
    
    # Sort by fitness (descending)
    sorted_indices = np.argsort(fitnesses)[::-1]
    sorted_population = [population[i] for i in sorted_indices]
    sorted_fitnesses = [fitnesses[i] for i in sorted_indices]

    current_best = sorted_population[0]
    best_scores.append(sorted_fitnesses[0])

    # Adaptive parameters based on convergence behavior
    for generation in range(100):  # Limited to prevent timeout
        # Check time limit
        if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
            break

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
            mutation_strength = 0.1 * (1 - generation / 100.0)  # Decrease over time
            for i in range(len(parent)):
                if np.random.random() < 0.2:  # 20% chance to mutate each element
                    noise = np.random.normal(0, mutation_strength)
                    parent[i] = max(0, parent[i] + noise)

            new_population.append(parent)

        # Replace population
        population = new_population

        # Evaluate all individuals with multi-objective scoring
        fitnesses = [evaluate_step_function(ind) for ind in population]
        
        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitnesses = [fitnesses[i] for i in sorted_indices]

        # Update best
        current_best = sorted_population[0]
        best_scores.append(sorted_fitnesses[0])

    return current_best

def stochastic_refinement(initial_solution):
    """
    Apply stochastic refinements to escape local minima
    """
    best_solution = initial_solution.copy()
    best_c2 = evaluate_step_function(best_solution)

    # Apply multiple stochastic perturbations
    for _ in range(10):  # Apply multiple rounds of perturbation
        # Apply small random perturbations
        perturbed = best_solution.copy()
        for i in range(len(perturbed)):
            if np.random.random() < 0.1:  # 10% chance to perturb each element
                noise = np.random.normal(0, 0.02)
                perturbed[i] = max(0, perturbed[i] + noise)

        # Normalize
        if np.sum(perturbed) > 0:
            perturbed = perturbed / np.sum(perturbed) * len(perturbed)

        perturbed_c2 = evaluate_step_function(perturbed)
        if perturbed_c2 > best_c2:
            best_solution = perturbed
            best_c2 = perturbed_c2

    # Apply differential evolution on subset for fine-tuning
    try:
        # Use smaller subset for faster optimization
        subset_size = min(len(best_solution), 200)
        subset_indices = np.random.choice(len(best_solution), subset_size, replace=False)
        subset_solution = [best_solution[i] for i in subset_indices]
        
        bounds = [(0, 10.0) for _ in range(len(subset_solution))]

        def objective(x):
            # Reconstruct full solution
            extended_x = best_solution.copy()
            for i, idx in enumerate(subset_indices):
                extended_x[idx] = max(0, x[i])
            return -evaluate_step_function(extended_x)

        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=10,
            popsize=5,
            seed=42,
            disp=False
        )

        if de_result.success:
            refined_solution = best_solution.copy()
            for i, idx in enumerate(subset_indices):
                refined_solution[idx] = max(0, de_result.x[i])
            
            refined_c2 = evaluate_step_function(refined_solution)
            if refined_c2 > best_c2:
                best_solution = refined_solution
                best_c2 = refined_c2
    except Exception:
        pass

    return best_solution

def construct_function() -> list[float]:
    """
    Optimized function to construct step-function with high C2 value.
    Uses adaptive evolutionary optimization with multi-scale initialization
    and piecewise linear integration for improved accuracy.
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

    # Refine the solution with stochastic methods
    refined_solution = stochastic_refinement(evolved_solution)

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