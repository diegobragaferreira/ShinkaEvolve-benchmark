# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from joblib import Parallel, delayed
import random
from typing import List, Tuple
import time

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the autoconvolution g = f*f and its norms efficiently.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    if not f_values:
        return 0.0, 0.0, 0.0
    
    # Create step function on [-1/4, 1/4] with equal spacing
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    # Step size in x domain [-1/4, 1/4]
    dx = 0.5 / (n - 1) if n > 1 else 0.5
    
    # Compute autoconvolution using numpy's convolution
    # This computes discrete convolution which approximates f*f
    g = np.convolve(f_values, f_values, mode='full')
    
    # Trim to proper range [-1/2, 1/2] (corresponding to [-1/4, 1/4] + [-1/4, 1/4])
    # The result should be of length 2*n - 1
    g_len = len(g)
    half_len = g_len // 2
    
    # Take center part to approximate [-1/2, 1/2] range
    g_centered = g[half_len:half_len+1]
    
    # However, for accurate representation, we do it properly:
    # We know that f is defined on [-1/4, 1/4] with n points, which means 
    # spacing is dx = 0.5/(n-1). Autoconvolution spans [-1/2, 1/2].
    # So the result should be 2n-1 points, covering [-1/2, 1/2].
    # But we'll use the more accurate approach by manually computing
    # the convolution integral with appropriate sampling
    
    # Let's redefine the approach: use discrete convolution properly
    # and then integrate properly for the L2 norm using trapezoidal rule
    
    # Compute the actual convolution
    g = np.convolve(f_values, f_values, mode='full')
    
    # Center the convolution (this gives us a symmetric result)
    g_centered = g[len(g)//2:]
    
    # But we want to focus on the main region [-1/2, 1/2], so:
    # g is convolved from two functions each on [-1/4, 1/4]
    # So result support is [-1/2, 1/2] with 2*n-1 points
    g_full = g
    g = g_full[len(g_full)//2 : len(g_full)//2 + n + n - 1] 
    
    # Now we compute the three norms
    # ||g||∞ = max of |g|
    norm_inf = np.max(np.abs(g)) if len(g) > 0 else 0.0
    
    # ||g||₁ = sum of |g| * dx
    if len(g) <= 1:
        norm_1 = 0.0
    else:
        # Trapezoidal approximation for ||g||₁
        norm_1 = np.sum(np.abs(g)) * dx
        
    # ||g||₂² = ∫ g² dx ≈ (dx/3) * Σ (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    if len(g) <= 1:
        norm_2_squared = 0.0
    else:
        # Piecewise linear integration (trapezoidal-like for quadratic form)
        # Use the formula for piecewise integration of g^2
        # For each segment [x_i, x_{i+1}] with values y_i, y_{i+1}:
        # ∫ y^2 dx ≈ (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
        g_squares = g**2
        norm_2_squared = 0.0
        for i in range(len(g)-1):
            # dx is the step size from the original function
            dx_segment = 0.5 / (n - 1) if n > 1 else 0.5
            y1, y2 = g[i], g[i+1]
            norm_2_squared += (dx_segment / 3.0) * (y1**2 + y1*y2 + y2**2)
    
    return norm_2_squared, norm_1, norm_inf

def compute_c2(f_values: List[float]) -> float:
    """Compute the C2 value for given step function."""
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    c2 = norm_2_squared / (norm_1 * norm_inf)
    return c2

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply mutation to individual with adaptive strategy."""
    mutated = individual.copy()
    n = len(mutated)
    
    # Apply Gaussian perturbation to some elements
    for i in range(n):
        if random.random() < mutation_rate:
            # Add small Gaussian noise to element
            mutated[i] += np.random.normal(0, 0.05 * np.mean(mutated) if np.mean(mutated) > 0 else 0.01)
            # Ensure non-negativity
            mutated[i] = max(0.0, mutated[i])
            
    # Occasionally perform a local search mutation
    if random.random() < 0.3:  # 30% chance of local search
        # Take a few adjacent elements and average them to smooth
        start_idx = random.randint(0, max(0, n-5))
        end_idx = min(start_idx + 5, n)
        avg_val = np.mean(mutated[start_idx:end_idx]) if end_idx > start_idx else 0.0
        for i in range(start_idx, end_idx):
            mutated[i] = avg_val
            
    return mutated

def evolve_population(population: List[List[float]], 
                      fitnesses: List[float], 
                      generation: int,
                      population_size: int) -> List[List[float]]:
    """Generate next generation using tournament selection and mutation."""
    # Sort by fitness
    sorted_indices = np.argsort(fitnesses)[::-1]
    
    # Keep top 30%
    elite_count = max(1, population_size // 3)
    elites = [population[i] for i in sorted_indices[:elite_count]]
    
    # Generate offspring through tournament selection and mutation
    offspring = []
    
    while len(offspring) < population_size - elite_count:
        # Tournament selection
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        
        # Mutate the winner
        mutated = mutate_individual(population[winner_index])
        offspring.append(mutated)
    
    # Combine elites and offspring
    new_population = elites + offspring
    
    # Possibly adapt population size based on diversity
    return new_population

def evaluate_population(population: List[List[float]]) -> List[float]:
    """Evaluate fitness for entire population in parallel."""
    def evaluate_single(individual):
        try:
            return compute_c2(individual)
        except Exception:
            return 0.0
    
    # Parallel evaluation
    fitnesses = Parallel(n_jobs=-1, backend='threading')(
        delayed(evaluate_single)(ind) for ind in population
    )
    
    return fitnesses

def construct_function() -> List[float]:
    """Optimized function to construct step-function with high C2 value."""
    # Parameters for evolution
    population_size = 50
    generations = 200
    max_time_seconds = 85  # Leave room for finalization
    
    start_time = time.time()
    
    # Initialize population with diverse step functions
    def create_random_individual():
        length = np.random.randint(100, 1000)
        return np.clip(np.random.exponential(scale=0.5, size=length), 0, 10.0).tolist()
    
    population = [create_random_individual() for _ in range(population_size)]
    
    best_fitness = -float('inf')
    best_individual = None
    
    # Precompute initial fitnesses
    fitnesses = evaluate_population(population)
    
    for gen in range(generations):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Update best solution
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_individual = population[current_best_idx].copy()
        
        # Evolve population
        population = evolve_population(population, fitnesses, gen, population_size)
        
        # Evaluate new population
        fitnesses = evaluate_population(population)
        
        # Early stopping if no improvement
        if gen > 10 and abs(best_fitness - fitnesses[current_best_idx]) < 1e-6:
            break
            
        # Adaptive population size adjustment
        if gen % 50 == 0 and gen > 0:
            population_size = max(20, min(200, population_size + (10 if gen % 100 == 0 else 0)))
            
    # Final evaluation of best individual to get precise value
    if best_individual is not None:
        final_c2 = compute_c2(best_individual)
        # If it looks good, try one more local refinement
        if final_c2 > 0.95:
            refined = mutate_individual(best_individual, mutation_rate=0.05)
            refined_c2 = compute_c2(refined)
            if refined_c2 > final_c2:
                best_individual = refined
    
    return best_individual if best_individual is not None else []

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
