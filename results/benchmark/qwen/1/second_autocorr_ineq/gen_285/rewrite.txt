# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from typing import List
import time
from numba import njit
import warnings

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
    
    # Manual convolution loop for speed
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

        # Avoid division by zero
        if norm_g_1 <= 1e-12 or norm_g_inf <= 1e-12:
            return 0.0

        c2 = norm_g_2_squared / (norm_g_1 * norm_g_inf)
        return c2
    except Exception:
        return 0.0

def generate_multiscale_initialization(n_steps: int) -> List[float]:
    """
    Generate diverse initial configuration using multi-scale patterns for better exploration
    """
    # Create base alternating pattern
    f = np.zeros(n_steps)
    
    # Different segment sizes for multi-scale effect
    segment_sizes = [max(1, n_steps // 15), max(1, n_steps // 10), max(1, n_steps // 8)]
    
    for i, seg_size in enumerate(segment_sizes):
        for j in range(0, n_steps, seg_size):
            end_idx = min(j + seg_size, n_steps)
            if (j // seg_size) % 2 == 0:
                # High amplitude for alternating pattern
                amplitude = 0.7 + 0.2 * (i % 2)
                f[j:end_idx] = amplitude + np.random.random(end_idx - j) * 0.1
            else:
                # Low amplitude 
                amplitude = 0.1 + 0.1 * (i % 2)
                f[j:end_idx] = amplitude + np.random.random(end_idx - j) * 0.1
    
    # Add structured Gaussian peaks
    x = np.linspace(-1, 1, n_steps)
    num_peaks = max(2, n_steps // 50)
    for _ in range(num_peaks):
        center = np.random.uniform(-0.4, 0.4)
        width = np.random.uniform(0.05, 0.15)
        amplitude = 0.3 + np.random.random() * 0.4
        gauss_peak = amplitude * np.exp(-0.5 * ((x - center) / width)**2)
        f += gauss_peak
    
    # Apply smoothing for better transitions
    if n_steps > 20:
        kernel_size = min(9, max(3, n_steps // 20))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size > 1:
            kernel = np.exp(-np.arange(kernel_size)**2 / (2 * (kernel_size/4)**2))
            kernel = kernel / np.sum(kernel)
            f = np.convolve(f, kernel, mode='same')
    
    # Ensure non-negativity and normalize
    f = np.clip(f, 0, None)
    if np.sum(f) > 0:
        f = f / np.sum(f) * n_steps * 0.3
    
    return f.tolist()

def adaptive_evolutionary_search(n_steps: int = 500, max_time: float = 85.0) -> List[float]:
    """
    Custom evolutionary algorithm with adaptive parameters and enhanced operators
    """
    # Calculate time remaining
    start_time = time.time()
    
    # Adaptive population sizing based on problem size
    pop_size = min(max(15, n_steps // 30), 30)
    max_iterations = min(max(20, n_steps // 20), 60)
    
    # Generate diverse initial population
    population = []
    initial_pop_size = min(20, pop_size)
    
    for i in range(initial_pop_size):
        # Mix different initialization strategies
        if i % 4 == 0:
            # Multi-scale initialization
            f = generate_multiscale_initialization(n_steps)
        elif i % 4 == 1:
            # Random with structure
            f = np.random.random(n_steps)
            f = np.clip(f, 0, 1)
            f = f / np.sum(f)
            f = f.tolist()
        elif i % 4 == 2:
            # Peak distribution
            f = np.zeros(n_steps)
            center = n_steps // 2
            width = max(1, n_steps // 12)
            f[max(0, center-width//2):min(n_steps, center+width//2)] = 1.0
            f += np.random.normal(0, 0.05, n_steps)
            f = np.clip(f, 0, None)
            f = f / np.sum(f)
            f = f.tolist()
        else:
            # Gaussian-like
            x = np.linspace(-1, 1, n_steps)
            sigma = 0.2 + np.random.random() * 0.3
            mu = np.random.random() * 0.4 - 0.2
            f = np.exp(-0.5 * ((x - mu) / sigma)**2)
            f = f / np.sum(f)
            f = f.tolist()
        
        population.append(f)
    
    best_solution = None
    best_c2 = -np.inf
    
    # Evolutionary search loop
    for iteration in range(max_iterations):
        if time.time() - start_time > max_time * 0.95:
            break
            
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            c2 = evaluate_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_c2:
                best_c2 = c2
                best_solution = individual.copy()
        
        # Selection - keep top 50% (but at least 2)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        selected_count = max(2, pop_size // 2)
        selected_indices = sorted_indices[:selected_count]
        selected_population = [population[i] for i in selected_indices]
        
        # Create offspring through crossover and mutation
        new_population = selected_population.copy()
        
        # Elitism: keep the best individual
        if best_solution is not None:
            new_population.append(best_solution)
        
        # Generate new individuals through crossover and mutation
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = selected_population[np.random.randint(0, len(selected_population))]
            parent2 = selected_population[np.random.randint(0, len(selected_population))]
            
            # Crossover (uniform with some structure preservation)
            child = []
            for i in range(n_steps):
                if np.random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            
            # Mutation with adaptive rate
            mutation_rate = 0.1 * np.exp(-iteration/max_iterations)  # Decreasing rate
            for i in range(n_steps):
                if np.random.random() < mutation_rate:
                    # Add small random perturbation
                    delta = np.random.normal(0, 0.03)
                    child[i] = max(0, child[i] + delta)
            
            # Normalize
            child_sum = sum(child)
            if child_sum > 0:
                child = [val / child_sum for val in child]
            
            new_population.append(child)
        
        # Trim to population size
        population = new_population[:pop_size]
    
    return best_solution if best_solution is not None else [1.0/n_steps] * n_steps

def multi_scale_optimization(max_time: float = 85.0) -> List[float]:
    """
    Perform multi-scale optimization approach
    """
    start_time = time.time()
    
    # Try different sizes to find best configuration
    configurations = [200, 300, 500, 1000]
    best_solution = None
    best_c2 = -np.inf
    
    for n_steps in configurations:
        if time.time() - start_time > max_time * 0.9:
            break
            
        try:
            # Run evolutionary search with adaptive parameters
            solution = adaptive_evolutionary_search(n_steps, max_time - (time.time() - start_time))
            c2_value = evaluate_c2(solution)
            
            if c2_value > best_c2:
                best_c2 = c2_value
                best_solution = solution
                
        except Exception as e:
            warnings.warn(f"Failed at size {n_steps}: {e}")
            continue
    
    # If we have a solution, do final refinement with simple optimization
    if best_solution is not None and len(best_solution) > 0:
        try:
            # Simple local refinement
            refined_solution = best_solution.copy()
            original_c2 = evaluate_c2(refined_solution)
            
            # Try to improve with basic perturbations
            for _ in range(10):
                # Make small random adjustments
                perturbed = refined_solution.copy()
                for i in range(len(perturbed)):
                    if np.random.random() < 0.1:
                        delta = np.random.normal(0, 0.01)
                        perturbed[i] = max(0, perturbed[i] + delta)
                
                # Normalize
                perturbed_sum = sum(perturbed)
                if perturbed_sum > 0:
                    perturbed = [val / perturbed_sum for val in perturbed]
                
                new_c2 = evaluate_c2(perturbed)
                if new_c2 > original_c2:
                    refined_solution = perturbed
                    original_c2 = new_c2
                    
            return refined_solution
        except:
            pass
    
    # Return best found solution or fallback
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to uniform distribution
        n_steps = 500
        return [1.0/n_steps] * n_steps

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value using advanced optimization
    """
    try:
        # Run multi-scale optimization
        final_solution = multi_scale_optimization()
        return final_solution
            
    except Exception as e:
        warnings.warn(f"Error in optimization: {e}")
        # Fallback to simple initialization
        n_steps = 500
        return [1.0/n_steps] * n_steps

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")