# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
from scipy.optimize import differential_evolution
import random

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals, dx):
    """
    Efficiently compute autoconvolution using Numba JIT compilation
    """
    n = len(f_vals)
    # Autoconvolution using discrete convolution formula
    # For step functions, we can use a more efficient approach
    g = np.zeros(2*n - 1)

    # Compute convolution manually for efficiency
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

        # Handle edge cases
        if len(f_vals) == 0:
            return 0.0
            
        # Use a fixed dx for consistent spacing
        dx = 0.5 / len(f_vals) if len(f_vals) > 0 else 0.001

        # Compute C2 value
        c2 = compute_c2_numba(f_vals, dx)
        return c2
    except Exception as e:
        return 0.0

def sophisticated_initialization(n_steps):
    """
    Create a sophisticated initial step function with mathematical patterns
    """
    # Use a combination of different mathematical patterns to create diversity
    pattern_type = np.random.choice(['gaussian', 'alternating', 'peak_centered'])
    
    if pattern_type == 'gaussian':
        # Create Gaussian-like pattern with emphasis on edges
        x = np.linspace(-1, 1, n_steps)
        sigma = 0.2 + np.random.random() * 0.2
        mu = np.random.random() * 0.4 - 0.2
        pattern = np.exp(-0.5 * ((x - mu) / sigma)**2)
        pattern = pattern / np.sum(pattern)
        
    elif pattern_type == 'alternating':
        # Create alternating high/low pattern
        pattern = []
        for i in range(n_steps):
            if i % 2 == 0:
                pattern.append(max(0.0, 1.0 + np.random.normal(0, 0.1)))
            else:
                pattern.append(max(0.0, 0.2 + np.random.normal(0, 0.05)))
        pattern = np.array(pattern) / np.sum(pattern)
        
    else:  # peak_centered
        # Create peak-centered pattern with tapering edges
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
        pattern = pattern / np.sum(pattern)
    
    return pattern.tolist()

def create_diverse_initial_population(pop_size, min_steps, max_steps):
    """
    Create diverse population with varied patterns and sizes
    """
    population = []
    for i in range(pop_size):
        # Randomly choose number of steps
        n_steps = np.random.randint(min_steps, max_steps)
        
        # Create initial solution
        individual = sophisticated_initialization(n_steps)
        population.append(individual)
        
    return population

def evolutionary_optimization(population_size=20, max_iter=50):
    """
    Use evolutionary algorithm with differential evolution for optimization
    """
    # Fixed parameters for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Use a reasonable range for number of steps
    min_steps = 500
    max_steps = 2000

    # Create diverse initial population
    initial_population = create_diverse_initial_population(population_size, min_steps, max_steps)
    
    # Evaluate initial population
    fitnesses = [evaluate_step_function(ind) for ind in initial_population]
    
    # Find best initial solution
    best_idx = np.argmax(fitnesses)
    best_solution = initial_population[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    
    # Keep the best individual and add it to new population
    new_population = [best_solution]  # Elitism
    
    # Generate remaining population
    for i in range(population_size - 1):
        # Create new individual from existing ones with crossover and mutation
        parent1_idx = np.random.randint(0, len(initial_population))
        parent2_idx = np.random.randint(0, len(initial_population))
        
        parent1 = initial_population[parent1_idx]
        parent2 = initial_population[parent2_idx]
        
        # Simple crossover and mutation
        child = []
        for j in range(len(parent1)):
            if np.random.random() < 0.5:
                child.append(parent1[j])
            else:
                child.append(parent2[j])
                
        # Add mutation
        for j in range(len(child)):
            if np.random.random() < 0.1:  # 10% mutation rate
                mutation_strength = 0.2
                noise = np.random.normal(0, mutation_strength)
                child[j] = max(0.0, child[j] + noise)
        
        new_population.append(child)
    
    # Refine with differential evolution on best solutions
    # Focus on the best few individuals
    top_individuals = sorted(zip(new_population, fitnesses), key=lambda x: x[1], reverse=True)[:5]
    top_solutions = [ind for ind, fit in top_individuals]
    
    # Take the best solution as starting point
    if len(top_solutions) > 0:
        start_solution = top_solutions[0]
    else:
        # Fallback to simple initialization
        n_steps = np.random.randint(min_steps, max_steps)
        start_solution = sophisticated_initialization(n_steps)
    
    # Define bounds for differential evolution (step heights between 0 and 10)
    bounds = [(0.0, 10.0)] * len(start_solution)

    # Run differential evolution with fewer iterations for speed
    result = differential_evolution(
        lambda x: -evaluate_step_function(x),  # Negative because we want to maximize
        bounds,
        maxiter=max_iter,
        popsize=min(population_size, 15),
        seed=42,
        strategy='best1bin',
        tol=1e-6,
        recombination=0.7,
        disp=False
    )

    # Return best solution found
    return result.x.tolist()

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value.
    Uses evolutionary optimization to find better solutions than random initialization.
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Run evolutionary optimization
    start_time = time.time()
    try:
        best_solution = evolutionary_optimization(population_size=20, max_iter=30)
    except Exception as e:
        # Fallback to simple approach if evolution fails
        print(f"Evolution failed with error: {e}")
        best_solution = [1.0] * 100  # Default case

    end_time = time.time()
    eval_time = end_time - start_time

    # Ensure the solution is valid
    if not best_solution:
        best_solution = [1.0] * 100

    print(f"Eval time: {eval_time:.4f}s")
    print(f"Best C2 found: {evaluate_step_function(best_solution):.6f}")

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")