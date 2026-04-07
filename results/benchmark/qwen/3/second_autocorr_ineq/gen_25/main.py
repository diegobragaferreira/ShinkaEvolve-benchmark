# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import time
import random
from scipy.optimize import differential_evolution, minimize
import copy

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@njit
def compute_autoconvolution_norms(f_vals):
    """
    Compute the norms of the autoconvolution g = f*f
    """
    n = len(f_vals)
    # Create convolution result array - size 2*n-1
    g = np.zeros(2 * n - 1)

    # Compute convolution manually (f*f)
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    # Compute norms
    g_squared = g * g
    norm_2_sq = np.sum(g_squared)
    norm_1 = np.sum(np.abs(g))
    norm_inf = np.max(np.abs(g))

    return norm_2_sq, norm_1, norm_inf

@njit
def calculate_c2(f_vals):
    """
    Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)
    """
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_vals)

    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    return norm_2_sq / (norm_1 * norm_inf)

def generate_initial_population(pop_size, min_length, max_length):
    """Generate initial population with hybrid approach"""
    population = []
    for _ in range(pop_size):
        # Use hybrid initialization - alternating pattern with noise
        n = np.random.randint(min_length, max_length)
        f_values = np.zeros(n)
        
        # Create alternating high/low pattern
        for i in range(n):
            if i % 2 == 0:
                f_values[i] = np.random.uniform(0.7, 1.0)
            else:
                f_values[i] = np.random.uniform(0.0, 0.3)
        
        # Add some random noise
        noise_level = 0.1
        f_values += np.random.normal(0, noise_level, n)
        f_values = np.maximum(f_values, 0)
        
        population.append(f_values.tolist())
    
    return population

def mutate_individual(individual, generation, max_generations):
    """Apply mutation with adaptive rate"""
    mutated = individual.copy()
    mutation_rate = 0.3 - (generation / max_generations) * 0.25  # Decreasing rate
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Apply Gaussian mutation
            mutated[i] += np.random.normal(0, 0.1 * mutated[i] if mutated[i] > 0 else 0.1)
            mutated[i] = max(0, mutated[i])  # Ensure non-negative
    
    return mutated

def uniform_crossover(parent1, parent2):
    """Uniform crossover between two parents"""
    child1, child2 = [], []
    for i in range(min(len(parent1), len(parent2))):
        if np.random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    
    # Pad shorter child with values from longer parent
    if len(parent1) > len(parent2):
        for i in range(len(parent2), len(parent1)):
            child1.append(parent1[i])
    elif len(parent2) > len(parent1):
        for i in range(len(parent1), len(parent2)):
            child2.append(parent2[i])
    
    return child1, child2

def local_optimization(individual, max_iter=50):
    """Apply local optimization to improve individual"""
    def objective(x):
        return -calculate_c2(x)  # Negative because we want to maximize
    
    # Wrap individual to match scipy format
    x0 = np.array(individual)
    
    try:
        # Use differential evolution for local search
        result = differential_evolution(objective, 
                                       bounds=[(0, 10) for _ in range(len(x0))],
                                       maxiter=max_iter,
                                       popsize=5,
                                       seed=42)
        
        if result.success:
            optimized = result.x.tolist()
            # Ensure non-negative
            optimized = [max(0, val) for val in optimized]
            return optimized
    except:
        pass
    
    return individual

def construct_function() -> list[float]:
    """Construct a step-function with high C2 value using evolutionary optimization"""
    start_time = time.time()
    
    # Parameters
    pop_size = 50
    max_generations = 100
    min_length = 100
    max_length = 2000
    elite_size = int(pop_size * 0.1)  # Top 10%
    
    # Initialize population
    population = generate_initial_population(pop_size, min_length, max_length)
    
    best_fitness = -float('inf')
    best_individual = None
    
    # Evolution loop
    for generation in range(max_generations):
        if time.time() - start_time > 85:  # Leave buffer for cleanup
            break
            
        # Evaluate fitness of all individuals
        fitness_scores = []
        for individual in population:
            fitness = calculate_c2(individual)
            fitness_scores.append((fitness, individual))
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
        
        # Sort by fitness
        fitness_scores.sort(reverse=True)
        
        # Keep elites
        new_population = [ind for _, ind in fitness_scores[:elite_size]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            tournament_size = 5
            tournament1 = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
            tournament2 = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
            
            parent1 = max(tournament1, key=lambda x: x[0])[1]
            parent2 = max(tournament2, key=lambda x: x[0])[1]
            
            # Crossover
            child1, child2 = uniform_crossover(parent1, parent2)
            
            # Mutate
            child1 = mutate_individual(child1, generation, max_generations)
            child2 = mutate_individual(child2, generation, max_generations)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:pop_size]
        
        # Apply local optimization to top individuals
        top_individuals = [ind for _, ind in fitness_scores[:elite_size]]
        for ind in top_individuals:
            improved = local_optimization(ind)
            # Check if improvement helped
            if calculate_c2(improved) > calculate_c2(ind):
                # Replace in population if it's in there
                try:
                    idx = population.index(ind)
                    population[idx] = improved
                except ValueError:
                    # If not in population, replace worst performer
                    worst_idx = np.argmin([calculate_c2(ind) for ind in population])
                    population[worst_idx] = improved
    
    # Final optimization of best individual
    if best_individual is not None:
        final_best = local_optimization(best_individual)
        final_fitness = calculate_c2(final_best)
        if final_fitness > best_fitness:
            best_individual = final_best
    
    # Return best solution found
    return best_individual if best_individual is not None else [0.0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
