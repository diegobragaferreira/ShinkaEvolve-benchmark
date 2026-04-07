# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import List
from numba import njit
import warnings
import time

# JIT-compiled fast autoconvolution and norm computation
@njit
def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
    """
    Fast computation of autoconvolution norms using Numba JIT compilation
    """
    n = len(f_values)

    # Compute autoconvolution g = f * f using discrete convolution
    # The resulting g will have length 2*n - 1 where n is the length of f
    g_length = 2 * n - 1
    g = np.zeros(g_length)

    # Manual convolution loop for speed (Numba-compatible)
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]

    # Compute the norms
    # ||g||₂² = sum(g[i]²) using proper piecewise integration
    norm_g_2_squared = 0.0

    # For piecewise linear integration, we use trapezoidal-like approach:
    # for consecutive pairs of points (y1, y2) with unit spacing:
    # integral of y^2 ≈ (1/3)(y1^2 + y1*y2 + y2^2)
    for i in range(g_length - 1):
        y1 = g[i]
        y2 = g[i + 1]
        norm_g_2_squared += (y1 * y1 + y1 * y2 + y2 * y2) / 3.0

    # ||g||₁ = sum(|g[i]|)
    norm_g_1 = 0.0
    for i in range(g_length):
        norm_g_1 += abs(g[i])

    # ||g||∞ = max(|g[i]|)
    norm_g_inf = 0.0
    for i in range(g_length):
        abs_g = abs(g[i])
        if abs_g > norm_g_inf:
            norm_g_inf = abs_g

    return norm_g_2_squared, norm_g_1, norm_g_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the norms ||g||₂², ||g||₁, and ||g||∞ for the autoconvolution g = f*f
    Using proper piecewise linear integration for ||g||₂² as specified in requirements
    """
    f = np.array(f_values)
    
    # Use the fast JIT-compiled version
    norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms_fast(f)
    
    return norm_g_2_squared, norm_g_1, norm_g_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function
    """
    try:
        norm_g_2_squared, norm_g_1, norm_g_inf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero with stricter thresholds
        if norm_g_1 <= 1e-15 or norm_g_inf <= 1e-15:
            return 0.0

        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception:
        return 0.0

def generate_initial_population(n_individuals: int, n_steps: int) -> List[List[float]]:
    """
    Generate diverse initial population for evolutionary algorithm with enhanced patterns
    """
    population = []
    
    # Create various types of initial configurations
    for i in range(n_individuals):
        # Type 1: Alternating high/low segments with smooth transitions
        if i % 4 == 0:
            f = np.zeros(n_steps)
            segment_size = max(1, n_steps // 12)
            for j in range(0, n_steps, segment_size):
                end_idx = min(j + segment_size, n_steps)
                if (j // segment_size) % 2 == 0:
                    # High region with variation
                    f[j:end_idx] = 0.7 + np.random.random(end_idx - j) * 0.25
                else:
                    # Low region with variation  
                    f[j:end_idx] = 0.1 + np.random.random(end_idx - j) * 0.15

            # Apply Gaussian smoothing for transitions
            x = np.linspace(-1, 1, n_steps)
            gaussian_width = 0.2 + np.random.random() * 0.15
            gaussian = np.exp(-0.5 * (x / gaussian_width)**2)
            f = f * gaussian * 0.5 + gaussian * 0.5
            
            # Ensure non-negativity and normalize
            f = np.clip(f, 0, None)
            if np.sum(f) > 0:
                f = f / np.sum(f)
            population.append(f.tolist())

        # Type 2: Multi-peak distribution  
        elif i % 4 == 1:
            f = np.ones(n_steps) * 0.1  # Base low values
            # Add multiple peaks at different positions
            n_peaks = 2 + np.random.randint(1, 4)
            for _ in range(n_peaks):
                peak_pos = np.random.randint(0, n_steps)
                peak_width = max(1, n_steps // 15 + np.random.randint(-2, 3))
                start = max(0, peak_pos - peak_width // 2)
                end = min(n_steps, peak_pos + peak_width // 2)
                f[start:end] = np.maximum(f[start:end], 0.7 + np.random.random(end - start) * 0.2)
            
            # Add smoothing
            x = np.linspace(-1, 1, n_steps)
            gaussian = np.exp(-0.5 * (x / 0.2)**2)
            f = f * gaussian * 0.5 + gaussian * 0.5
            
            # Ensure non-negativity and normalize
            f = np.clip(f, 0, None)
            if np.sum(f) > 0:
                f = f / np.sum(f)
            population.append(f.tolist())

        # Type 3: Gaussian-like distribution
        elif i % 4 == 2:
            x = np.linspace(-1, 1, n_steps)
            sigma = 0.15 + np.random.random() * 0.2
            mu = np.random.random() * 0.3 - 0.15  # Centered around -0.15 to 0.15
            f = np.exp(-0.5 * ((x - mu) / sigma)**2)
            if np.sum(f) > 0:
                f = f / np.sum(f)
            population.append(f.tolist())

        # Type 4: Uniform distribution with clustering
        else:
            f = np.random.random(n_steps)
            # Add some structure with clustering
            clusters = 3 + np.random.randint(1, 4)
            for _ in range(clusters):
                center = np.random.randint(0, n_steps)
                width = max(1, n_steps // 10 + np.random.randint(-2, 3))
                start = max(0, center - width // 2)
                end = min(n_steps, center + width // 2)
                f[start:end] = np.maximum(f[start:end], 0.5 + np.random.random(end - start) * 0.3)
            f = np.clip(f, 0, 1)
            if np.sum(f) > 0:
                f = f / np.sum(f)
            population.append(f.tolist())

    return population

def enhanced_evolutionary_optimization(max_generations: int = 30) -> List[float]:
    """
    Enhanced evolutionary algorithm with adaptive parameters and improved diversity
    """
    n_steps = 500
    
    # Initial population size
    pop_size = 15
    max_pop_size = 25
    min_pop_size = 8

    # Track convergence
    previous_best = -np.inf
    no_improvement_count = 0
    max_no_improvement = 5

    # Generate initial population
    population = generate_initial_population(pop_size, n_steps)

    best_solution = None
    best_c2 = -np.inf

    for generation in range(max_generations):
        # Evaluate fitness of current population
        fitness_scores = []
        for individual in population:
            c2 = evaluate_c2(individual)
            fitness_scores.append(c2)

            if c2 > best_c2:
                best_c2 = c2
                best_solution = individual.copy()

        # Check for convergence
        current_best = max(fitness_scores)
        if current_best > previous_best:
            previous_best = current_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Adaptively adjust population size based on convergence
        if no_improvement_count > max_no_improvement:
            if pop_size < max_pop_size:
                pop_size = min(pop_size + 1, max_pop_size)
            elif pop_size > min_pop_size:
                pop_size = max(pop_size - 1, min_pop_size)
            no_improvement_count = 0

        # Selection: Keep top third of individuals for better exploration
        sorted_indices = np.argsort(fitness_scores)[::-1][:pop_size//3]
        selected_population = [population[i] for i in sorted_indices]

        # Elitism: keep the best individual
        if best_solution is not None:
            selected_population.append(best_solution)

        # Generate offspring through crossover and mutation
        new_population = selected_population.copy()

        # Create new individuals through crossover and mutation
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = selected_population[np.random.randint(0, len(selected_population))]
            parent2 = selected_population[np.random.randint(0, len(selected_population))]

            # Crossover (uniform)
            child = []
            for i in range(n_steps):
                if np.random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])

            # Mutation with adaptive rate and controlled intensity
            mutation_rate = 0.18 * np.exp(-generation/max_generations)  # Decreasing rate
            mutation_intensity = 0.06 * (1 - generation/max_generations)  # Decreasing intensity

            for i in range(n_steps):
                if np.random.random() < mutation_rate:
                    # Adjust intensity based on generation progress
                    delta = np.random.normal(0, mutation_intensity * (1 + generation/max_generations))
                    child[i] = max(0, child[i] + delta)

            # Normalize
            child_sum = sum(child)
            if child_sum > 0:
                child = [val / child_sum for val in child]

            new_population.append(child)

        # Trim to population size
        population = new_population[:pop_size]

    return best_solution if best_solution is not None else [1.0/n_steps] * n_steps

def local_refinement_optimization(initial_f: List[float], max_iter: int = 15) -> List[float]:
    """
    Enhanced local refinement using hybrid approach with targeted perturbations
    """
    f = np.array(initial_f)
    n_steps = len(f)

    # Use a combination of random search and local perturbations
    for iteration in range(max_iter):
        current_c2 = evaluate_c2(f.tolist())

        # Try a mixed strategy: some random changes, some gradient-informed changes
        best_f = f.copy()
        best_c2 = current_c2

        # Try several types of perturbations to improve chances of finding better local solutions
        for _ in range(50):  # Increased attempts for better search
            perturbed_f = f.copy()

            # 70% chance of more exploratory perturbation
            if np.random.random() < 0.7:
                # Select index based on importance (random selection)
                idx = np.random.randint(0, n_steps)
                # Use a slightly larger step size for exploration
                delta = np.random.normal(0, 0.02)
                perturbed_f[idx] = max(0, perturbed_f[idx] + delta)
            else:
                # Pure random perturbation with different variance
                idx = np.random.randint(0, n_steps)
                delta = np.random.normal(0, 0.015)
                perturbed_f[idx] = max(0, perturbed_f[idx] + delta)

            # Normalize
            if np.sum(perturbed_f) > 0:
                perturbed_f = perturbed_f / np.sum(perturbed_f)

            new_c2 = evaluate_c2(perturbed_f.tolist())

            if new_c2 > best_c2:
                best_c2 = new_c2
                best_f = perturbed_f

        f = best_f

        # Conservative early stopping
        if abs(best_c2 - current_c2) < 1e-7:
            break

    return f.tolist()

def sophisticated_initialization() -> List[float]:
    """
    Generate a sophisticated initial configuration based on mathematical intuition
    """
    n_steps = 500

    # Create a step function that tries to balance flatness with sufficient mass
    # We want to create a function that when convolved produces a relatively flat profile
    # but with enough energy to achieve high C2

    # Create a base function with carefully designed alternating high/low regions
    f = np.zeros(n_steps)

    # Divide into segments with varied sizes to create interesting convolution behavior
    segment_sizes = [max(1, n_steps // 10 + np.random.randint(-2, 3))] * 8
    cumulative = 0
    for i, size in enumerate(segment_sizes):
        if cumulative + size > n_steps:
            size = n_steps - cumulative
        end_idx = cumulative + size
        if i % 2 == 0:
            # High region
            f[cumulative:end_idx] = 0.65 + np.random.random(end_idx - cumulative) * 0.25
        else:
            # Low region
            f[cumulative:end_idx] = 0.1 + np.random.random(end_idx - cumulative) * 0.15
        cumulative = end_idx
        if cumulative >= n_steps:
            break

    # Add Gaussian envelope for smoothness
    x = np.linspace(-1, 1, n_steps)
    gaussian_width = 0.25 + np.random.random() * 0.15
    gaussian = np.exp(-0.5 * (x / gaussian_width)**2)
    f = f * gaussian * 0.5 + gaussian * 0.5

    # Add some peak structures for extra complexity
    n_peaks = 2 + np.random.randint(1, 4)
    for _ in range(n_peaks):
        peak_pos = np.random.randint(0, n_steps)
        peak_width = max(1, n_steps // 18 + np.random.randint(-2, 3))
        start = max(0, peak_pos - peak_width // 2)
        end = min(n_steps, peak_pos + peak_width // 2)
        peak_height = 0.3 + np.random.random() * 0.4
        f[start:end] = np.maximum(f[start:end], peak_height)

    # Ensure non-negativity and normalize
    f = np.clip(f, 0, None)
    if np.sum(f) > 0:
        f = f / np.sum(f)

    return f.tolist()

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value using enhanced methods
    """
    start_time = time.time()
    
    try:
        # Strategy 1: Try sophisticated initialization
        initial_f = sophisticated_initialization()
        c2_initial = evaluate_c2(initial_f)

        # Strategy 2: Run enhanced evolutionary optimization
        evolved_solution = enhanced_evolutionary_optimization(35)
        evolved_c2 = evaluate_c2(evolved_solution)

        # Strategy 3: Local refinement of the evolved solution
        refined_solution = local_refinement_optimization(evolved_solution, 15)
        refined_c2 = evaluate_c2(refined_solution)

        # Return the best of all three approaches
        best_c2 = max(c2_initial, evolved_c2, refined_c2)
        
        if best_c2 == c2_initial:
            return initial_f
        elif best_c2 == evolved_c2:
            return evolved_solution
        else:
            return refined_solution

    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple initialization
        n_steps = 500
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")