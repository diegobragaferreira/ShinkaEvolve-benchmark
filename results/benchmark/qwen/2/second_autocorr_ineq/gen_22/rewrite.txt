# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from joblib import Parallel, delayed
import random
import time
from collections import deque

def compute_autoconvolution_norms(f_values: list[float]) -> tuple[float, float, float]:
    """
    Compute the autoconvolution g = f*f and its norms efficiently.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    if len(f_values) < 2:
        return 0.0, 0.0, 0.0
    
    # Compute autoconvolution using scipy's convolution
    g = signal.convolve(f_values, f_values, mode='full')
    
    # Compute the three norms using proper integration
    # ||g||∞ = max of |g|
    norm_inf = np.max(np.abs(g)) if len(g) > 0 else 0.0
    
    # ||g||₁ = sum of |g| * dx (using trapezoidal rule)
    if len(g) <= 1:
        norm_1 = 0.0
    else:
        # For the L1 norm, we integrate |g| using trapezoidal rule
        # Since we're dealing with discrete data, we approximate the integral
        # with uniform spacing, so each segment has width 1/n where n is the 
        # number of samples in the original function f
        dx = 0.5 / (len(f_values) - 1) if len(f_values) > 1 else 0.5
        norm_1 = np.sum(np.abs(g)) * dx
    
    # ||g||₂² = ∫ g² dx computed via trapezoidal-like integration
    # Using the formula for piecewise integration of g^2:
    # For each segment [x_i, x_{i+1}] with values y_i, y_{i+1}:
    # ∫ y^2 dx ≈ (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    if len(g) <= 1:
        norm_2_squared = 0.0
    else:
        # Use the correct trapezoidal-like integration for g^2
        dx = 0.5 / (len(f_values) - 1) if len(f_values) > 1 else 0.5
        norm_2_squared = 0.0
        for i in range(len(g)-1):
            y1, y2 = g[i], g[i+1]
            norm_2_squared += (dx / 3.0) * (y1**2 + y1*y2 + y2**2)
    
    return norm_2_squared, norm_1, norm_inf

def evaluate_c2(f_values: list[float]) -> float:
    """Evaluate C2 for a given step function with proper numerical integration."""
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero with numerical stability
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0
    
    c2 = norm_2_squared / (norm_1 * norm_inf)
    return c2

def mutate_individual(individual: list[float], mutation_rate: float = 0.1, sigma: float = 0.1) -> list[float]:
    """Apply mutation to an individual with robust strategies."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Use combination of Gaussian and Cauchy for robust mutation
            if random.random() < 0.7:  # 70% Gaussian
                mutated[i] += np.random.normal(0, sigma)
            else:  # 30% Cauchy for heavy-tailed exploration
                mutated[i] += np.random.standard_cauchy() * sigma
            
            # Clip to ensure non-negative values
            mutated[i] = max(0.0, mutated[i])
            
    return mutated

def crossover(parent1: list[float], parent2: list[float]) -> list[float]:
    """Perform crossover between two parents."""
    if len(parent1) != len(parent2):
        # If lengths differ, create a new one of average length
        new_length = max(len(parent1), len(parent2))
        child = [0.0] * new_length
        for i in range(new_length):
            if i < len(parent1) and i < len(parent2):
                # Blend crossover
                alpha = random.random()
                child[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
            elif i < len(parent1):
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
    else:
        # Same length, do simple uniform crossover
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
                
    return child

def tournament_selection(population: list[list[float]], fitnesses: list[float], tournament_size: int = 3) -> list[float]:
    """Select an individual using tournament selection."""
    if len(population) < tournament_size:
        tournament_size = len(population)
        
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_index = selected_indices[0]
    best_fitness = fitnesses[selected_indices[0]]
    
    for i in range(1, tournament_size):
        if fitnesses[selected_indices[i]] > best_fitness:
            best_fitness = fitnesses[selected_indices[i]]
            best_index = selected_indices[i]
            
    return population[best_index].copy()

def evolve_step_function(max_time_seconds: int = 85) -> list[float]:
    """Main evolutionary algorithm to optimize step function."""
    start_time = time.time()
    
    # Parameters
    pop_size = 50
    generations = 1000
    mutation_rate = 0.1
    elite_size = 5
    min_population = 20
    max_population = 200
    
    # Track best solution
    best_c2 = 0.0
    best_individual = None
    
    # Initialize population with various sizes
    population = []
    for _ in range(pop_size):
        # Randomly choose length between 100 and 5000 (similar to AlphaEvolve)
        n = random.randint(100, 5000)
        individual = [random.random() * 2 for _ in range(n)]
        population.append(individual)
    
    # Evolution loop
    generation = 0
    stagnation_counter = 0
    max_stagnation = 50
    last_best_c2 = 0.0
    
    # Early stopping buffer
    recent_improvements = deque(maxlen=10)
    
    while generation < generations and (time.time() - start_time) < max_time_seconds - 1:
        # Evaluate fitness of entire population
        def evaluate_fitness(indiv):
            return evaluate_c2(indiv)
            
        # Parallel evaluation of fitness
        fitnesses = Parallel(n_jobs=-1)(
            delayed(evaluate_fitness)(indiv) for indiv in population
        )
        
        # Update best solution
        for i, fitness in enumerate(fitnesses):
            if fitness > best_c2:
                best_c2 = fitness
                best_individual = population[i].copy()
                recent_improvements.append(fitness)
                
        # Check for stagnation
        if abs(best_c2 - last_best_c2) < 1e-6:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_best_c2 = best_c2
            
        # Early stop if stagnating too much
        if stagnation_counter >= max_stagnation:
            break
                
        # Sort population by fitness
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitnesses = [fitnesses[i] for i in sorted_indices]
        
        # Keep elites
        new_population = sorted_population[:elite_size]
        
        # Generate offspring using tournament selection and crossover
        while len(new_population) < pop_size:
            if (time.time() - start_time) >= max_time_seconds - 1:
                break
                
            # Tournament selection
            parent1 = tournament_selection(sorted_population, sorted_fitnesses)
            parent2 = tournament_selection(sorted_population, sorted_fitnesses)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child, mutation_rate)
            
            # Add to new population
            new_population.append(child)
            
        # Update population
        population = new_population
        
        # Adjust population size dynamically
        if len(population) < min_population:
            pop_size = min_population
        elif len(population) > max_population:
            pop_size = max_population
        else:
            # Adaptively adjust based on diversity and improvement
            if len(population) > 0:
                population_diversity = np.std([np.mean(indiv) for indiv in population])
                if population_diversity < 0.01 and len(recent_improvements) > 2:
                    # If very homogeneous, increase diversity
                    pop_size = min(max_population, int(pop_size * 1.2))
                elif len(recent_improvements) > 2 and recent_improvements[-1] > np.mean(list(recent_improvements)[:-1]):
                    # If improving, maintain population size
                    pass
                else:
                    # If not improving much, reduce population size slightly
                    pop_size = max(min_population, int(pop_size * 0.9))
                
        # Adjust mutation rate based on progress
        if best_c2 > 0.95:  # If we're close to good solution
            mutation_rate = 0.05
        elif best_c2 > 0.9:
            mutation_rate = 0.08
        else:
            mutation_rate = 0.1
            
        generation += 1
    
    # Return the best individual found
    return best_individual if best_individual is not None else [0.0]

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    try:
        # Run evolutionary algorithm
        f_values = evolve_step_function(max_time_seconds=85)
        return f_values
    except Exception as e:
        # Fallback to random generation if anything fails
        print(f"Error in evolution: {e}")
        f_values = [np.random.random()] * np.random.randint(100, 1000)
        return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")