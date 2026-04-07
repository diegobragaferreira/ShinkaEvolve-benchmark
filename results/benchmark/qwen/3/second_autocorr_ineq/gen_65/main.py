# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
import multiprocessing as mp
from joblib import Parallel, delayed
import time
from numba import jit, njit
import copy

# Global constants
POPULATION_SIZE = 200
GENERATIONS = 100
INITIAL_MUTATION_RATE = 0.3
FINAL_MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 10
MAX_STEPS = 50000
MIN_STEPS = 100
SEED = 42

# Set seed for reproducibility
np.random.seed(SEED)
random.seed(SEED)

@njit
def compute_autoconvolution_norms_jit(f_values, x, dx):
    """Optimized JIT version for computing autoconvolution norms"""
    n_steps = len(f_values)
    step_width = 0.5 / n_steps
    step_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps)
    
    # Build piecewise constant function
    f_func = np.zeros_like(x)
    for i in range(n_steps):
        pos = step_positions[i]
        height = f_values[i]
        left = pos - step_width/2
        right = pos + step_width/2
        # Simple approach to find indices within range
        for j in range(len(x)):
            if left <= x[j] <= right:
                f_func[j] = height
    
    # Perform autoconvolution using manual convolution (faster for small arrays)
    g = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            if i+j < len(x):
                g[i+j] += f_func[i] * f_func[j]
    
    g = g[:len(g)//2 + 1]  # Take only first half (since it's symmetric)
    g = g * dx
    
    # Compute the required norms
    g_squared = g**2
    g_abs = np.abs(g)
    
    # ||g||₂² (using trapezoidal rule for integration)
    norm_2_squared = np.trapz(g_squared, dx=dx)
    
    # ||g||₁ (L1 norm)
    norm_1 = np.trapz(g_abs, dx=dx)
    
    # ||g||∞ (infinity norm)
    norm_inf = np.max(g_abs)
    
    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation from step function values.
    Returns ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array and normalize
    f = np.array(f_values)
    
    # Create step function on [-1/4, 1/4]
    # We'll use 1000 points for accurate convolution
    x = np.linspace(-0.25, 0.25, 1000)
    dx = x[1] - x[0]
    
    # Optimize this computation using the JIT version
    try:
        norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms_jit(f_values, x, dx)
    except Exception:
        # Fallback to standard method if JIT fails
        n_steps = len(f)
        step_width = 0.5 / n_steps
        step_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps)
        
        # Build piecewise constant function
        f_func = np.zeros_like(x)
        for i, (pos, height) in enumerate(zip(step_positions, f)):
            left = pos - step_width/2
            right = pos + step_width/2
            mask = (x >= left) & (x <= right)
            f_func[mask] = height
        
        # Perform autoconvolution
        g = signal.convolve(f_func, f_func, mode='full')
        g = g[:len(g)//2 + 1]  # Take only first half (since it's symmetric)
        
        # Adjust for proper scaling due to discretization
        g = g * dx
        
        # Compute the required norms
        g_squared = g**2
        g_abs = np.abs(g)
        
        # ||g||₂² (using trapezoidal rule for integration)
        norm_2_squared = np.trapz(g_squared, dx=dx)
        
        # ||g||₁ (L1 norm)
        norm_1 = np.trapz(g_abs, dx=dx)
        
        # ||g||∞ (infinity norm)
        norm_inf = np.max(g_abs)
    
    return norm_2_squared, norm_1, norm_inf

def calculate_c2(f_values):
    """Calculate C₂ from step function values"""
    try:
        norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2
    except:
        return 0.0

def construct_alternating_pattern(n_steps):
    """Create alternating high/low pattern for better initialization"""
    pattern = []
    for i in range(n_steps):
        if i % 2 == 0:
            pattern.append(1.0)
        else:
            pattern.append(0.1)
    return pattern

def initialize_population(pop_size, min_steps, max_steps):
    """Initialize population with diverse step functions including hybrid patterns"""
    population = []
    for _ in range(pop_size):
        # Random number of steps
        n_steps = np.random.randint(min_steps, max_steps)
        
        # 70% chance of using alternating pattern + noise
        if np.random.random() < 0.7:
            heights = construct_alternating_pattern(n_steps)
            # Add some noise to make it more diverse
            noise = np.random.normal(0, 0.1, n_steps)
            heights = np.array(heights) + noise
            heights = np.maximum(heights, 0)
        else:
            # Standard random initialization
            heights = np.random.exponential(scale=1.0, size=n_steps)
            heights = np.maximum(heights, 0)
        
        population.append(heights.tolist())
    return population

def crossover(parent1, parent2):
    """Perform uniform crossover between two parents"""
    if len(parent1) != len(parent2):
        # Make them same length by truncating or padding
        min_len = min(len(parent1), len(parent2))
        parent1 = parent1[:min_len]
        parent2 = parent2[:min_len]
        
    if np.random.random() < CROSSOVER_RATE:
        # Uniform crossover
        child1, child2 = [], []
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2
    else:
        return parent1, parent2

def mutate(individual, mutation_rate, generation=None):
    """Mutate individual with adaptive scaling"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Adaptive noise based on current value - prevents too large changes
            noise_scale = 0.1 * mutated[i] + 0.01
            noise = np.random.normal(0, noise_scale)
            mutated[i] = max(0, mutated[i] + noise)
    return mutated

def evaluate_fitness_parallel(population):
    """Evaluate fitness of entire population in parallel with error handling"""
    def safe_calculate_c2(ind):
        try:
            return calculate_c2(ind)
        except:
            return 0.0
    
    # Use joblib for parallel evaluation with error handling
    results = Parallel(n_jobs=-1, backend='threading')(delayed(safe_calculate_c2)(ind) for ind in population)
    return results

def evaluate_fitness_sequential(population):
    """Sequential evaluation for debugging purposes"""
    results = []
    for ind in population:
        try:
            results.append(calculate_c2(ind))
        except:
            results.append(0.0)
    return results

def select_parents(population, fitness_scores):
    """Tournament selection with diversity preservation"""
    selected = []
    tournament_size = min(5, max(2, len(population) // 4))
    
    # Prune low performing individuals first to preserve diversity
    sorted_indices = np.argsort(fitness_scores)
    # Keep top 80% to maintain diversity
    keep_indices = sorted_indices[int(len(sorted_indices)*0.2):]
    reduced_population = [population[i] for i in keep_indices]
    reduced_fitness = [fitness_scores[i] for i in keep_indices]
    
    # Select from reduced pool
    for _ in range(len(population)):
        # Tournament selection with different tournament sizes randomly
        actual_tournament_size = np.random.choice([tournament_size, tournament_size+1])
        tournament_indices = np.random.choice(len(reduced_population), 
                                             min(actual_tournament_size, len(reduced_population)), 
                                             replace=False)
        tournament_fitness = [reduced_fitness[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(reduced_population[winner_index].copy())
    
    return selected

def elitism(population, fitness_scores, elite_count):
    """Keep best individuals with diversity consideration"""
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite = [population[i].copy() for i in sorted_indices[:elite_count]]
    return elite

def local_optimization_fallback(top_individuals, iterations=10):
    """Apply local optimization to top individuals"""
    optimized = []
    for individual in top_individuals:
        # Simple hill climbing local search
        current_individual = individual.copy()
        current_c2 = calculate_c2(current_individual)
        
        for _ in range(iterations):
            # Try small perturbations
            mutated = current_individual.copy()
            idx = np.random.randint(len(mutated))
            noise = np.random.normal(0, 0.05 * mutated[idx] + 0.005)
            mutated[idx] = max(0, mutated[idx] + noise)
            
            new_c2 = calculate_c2(mutated)
            if new_c2 > current_c2:
                current_individual = mutated
                current_c2 = new_c2
        
        optimized.append(current_individual)
    return optimized

def adaptive_evolution():
    """Main evolutionary algorithm with adaptive parameters"""
    # Initialize population
    population = initialize_population(POPULATION_SIZE, MIN_STEPS, MAX_STEPS)
    
    best_individual = None
    best_fitness = -np.inf
    
    for generation in range(GENERATIONS):
        # Evaluate fitness with parallel processing
        fitness_scores = evaluate_fitness_parallel(population)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best C2 = {best_fitness:.4f}")
        
        # Elitism
        elite = elitism(population, fitness_scores, ELITISM_COUNT)
        
        # Selection
        parents = select_parents(population, fitness_scores)
        
        # Calculate adaptive mutation rate
        mutation_rate = INITIAL_MUTATION_RATE + (FINAL_MUTATION_RATE - INITIAL_MUTATION_RATE) * (generation / GENERATIONS)
        
        # Crossover and mutation
        new_population = elite.copy()
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = np.random.choice(len(parents), 2, replace=False)
            child1, child2 = crossover(parents[p1], parents[p2])
            
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]
        
        # Apply local optimization to top performers every 20 generations
        if generation % 20 == 0 and generation > 0:
            top_individuals = [population[i] for i in np.argsort(fitness_scores)[-10:]]
            optimized = local_optimization_fallback(top_individuals)
            # Replace bottom 5% with optimized versions
            replacement_count = max(1, len(population) // 20)
            for i in range(replacement_count):
                if i < len(optimized):
                    population[i] = optimized[i]
    
    return best_individual

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Run adaptive evolution
    result = adaptive_evolution()
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    print(f"Evaluated in {eval_time:.2f} seconds")
    print(f"Best C2 found: {calculate_c2(result):.6f}")
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
