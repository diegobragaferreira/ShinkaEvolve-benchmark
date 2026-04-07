# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
import multiprocessing as mp
from joblib import Parallel, delayed
import time
from numba import jit

# Global constants
POPULATION_SIZE = 150
GENERATIONS = 80
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 8
MAX_STEPS = 20000
MIN_STEPS = 500
SEED = 42

# Set seed for reproducibility
np.random.seed(SEED)
random.seed(SEED)

@jit(nopython=True)
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

def initialize_population(pop_size, min_steps, max_steps):
    """Initialize population with diverse step functions"""
    population = []
    for _ in range(pop_size):
        # Random number of steps
        n_steps = np.random.randint(min_steps, max_steps)
        # Generate heights using a more sophisticated pattern
        # Mix of exponential distribution and uniform random to avoid all zeros
        heights = np.random.exponential(scale=1.0, size=n_steps) + \
                  np.random.uniform(0, 0.5, size=n_steps) + \
                  np.random.gamma(shape=2, scale=0.5, size=n_steps)
        # Clip negative values
        heights = np.maximum(heights, 0)
        population.append(heights.tolist())
    return population

def crossover(parent1, parent2):
    """Perform crossover between two parents"""
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
    """Mutate individual with Gaussian noise and adaptive scaling"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Adaptive noise based on current value - prevents too large changes
            noise_scale = 0.1 * mutated[i] + 0.01
            noise = np.random.normal(0, noise_scale)
            mutated[i] = max(0, mutated[i] + noise)
    return mutated

def evaluate_fitness(population):
    """Evaluate fitness of entire population in parallel"""
    # Use joblib for parallel evaluation with error handling
    def safe_calculate_c2(ind):
        try:
            return calculate_c2(ind)
        except:
            return 0.0
    
    results = Parallel(n_jobs=-1, backend='threading')(delayed(safe_calculate_c2)(ind) for ind in population)
    return results

def select_parents(population, fitness_scores):
    """Tournament selection with better randomness"""
    selected = []
    tournament_size = min(5, max(2, len(population) // 4))
    
    for _ in range(len(population)):
        # Tournament selection with different tournament sizes randomly
        actual_tournament_size = np.random.choice([tournament_size, tournament_size+1])
        tournament_indices = np.random.choice(len(population), 
                                             min(actual_tournament_size, len(population)), 
                                             replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_index].copy())
    
    return selected

def elitism(population, fitness_scores, elite_count):
    """Keep best individuals with diversity consideration"""
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite = [population[i].copy() for i in sorted_indices[:elite_count]]
    return elite

def adaptive_evolution():
    """Main evolutionary algorithm with adaptive parameters"""
    # Initialize population
    population = initialize_population(POPULATION_SIZE, MIN_STEPS, MAX_STEPS)
    
    best_individual = None
    best_fitness = -np.inf
    
    for generation in range(GENERATIONS):
        # Evaluate fitness
        fitness_scores = evaluate_fitness(population)
        
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
        
        # Crossover and mutation
        new_population = elite.copy()
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = np.random.choice(len(parents), 2, replace=False)
            child1, child2 = crossover(parents[p1], parents[p2])
            
            child1 = mutate(child1, MUTATION_RATE, generation)
            child2 = mutate(child2, MUTATION_RATE, generation)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]
    
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
