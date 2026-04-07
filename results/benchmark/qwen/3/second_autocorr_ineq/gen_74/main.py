# EVOLVE-BLOCK-START

import numpy as np
from typing import List
import random
import time
from numba import njit

@njit
def compute_autoconvolution_norms_numba(f_vals):
    """
    Compute the three norms needed for C2 calculation using numba JIT.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    n = len(f_vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    
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
def calculate_c2_numba(f_vals):
    """
    Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞) using numba JIT
    """
    norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms_numba(f_vals)

    # Avoid division by zero
    if norm_1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    return norm_2_sq / (norm_1 * norm_inf)

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    Returns (||g||₂², ||g||₁, ||g||∞)
    """
    # Use numba-compiled version for performance
    try:
        return compute_autoconvolution_norms_numba(np.array(f_values, dtype=np.float64))
    except:
        return 0.0, 0.0, 0.0

def calculate_c2(f_values: List[float]) -> float:
    """
    Calculate the C2 constant from the step function values.
    """
    try:
        # Use numba-compiled version for performance
        return calculate_c2_numba(np.array(f_values, dtype=np.float64))
    except:
        return 0.0

def mutate_individual(individual: List[float], mutation_rate: float = 0.1, 
                     strength: float = 0.3) -> List[float]:
    """
    Mutate an individual by slightly perturbing some values with adaptive strength.
    """
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add Gaussian noise with adaptive strength
            noise = random.gauss(0, strength * max(0.1, mutated[i]))
            mutated[i] = max(0, mutated[i] + noise)
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """
    Perform uniform crossover between two parents with weighted probability.
    """
    child = []
    min_len = min(len(parent1), len(parent2))
    
    # Crossover at positions with 70% probability of picking from parent1
    for i in range(min_len):
        if random.random() < 0.7:  # 70% chance to pick from parent1
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    # Add remaining elements from longer parent (if any)
    if len(parent1) > min_len:
        child.extend(parent1[min_len:])
    elif len(parent2) > min_len:
        child.extend(parent2[min_len:])
        
    return child

def initialize_population(population_size: int, min_steps: int = 100, max_steps: int = 1000) -> List[List[float]]:
    """
    Initialize a diverse population, mixing several strategies.
    """
    population = []
    
    # Strategy 1: Random initialization with peak enhancement
    for _ in range(population_size // 3):
        n = random.randint(min_steps, max_steps)
        individual = [random.uniform(0.1, 1.0) for _ in range(n)]
        # Enhance center peak
        if len(individual) > 5:
            peak_pos = len(individual) // 2
            individual[peak_pos] *= 2.0
        population.append(individual)
    
    # Strategy 2: Uniform initialization 
    for _ in range(population_size // 3):
        n = random.randint(min_steps, max_steps)
        individual = [0.5] * n
        # Add some variation
        for i in range(n):
            if random.random() < 0.3:
                individual[i] += random.uniform(-0.2, 0.2)
                individual[i] = max(0, individual[i])
        population.append(individual)
        
    # Strategy 3: Alternating pattern initialization
    for _ in range(population_size // 3):
        n = random.randint(min_steps, max_steps)
        individual = []
        for i in range(n):
            individual.append(1.0 if i % 2 == 0 else 0.1)
        population.append(individual)
        
    # Ensure we have enough individuals
    while len(population) < population_size:
        n = random.randint(min_steps, max_steps)
        individual = [random.uniform(0.1, 1.0) for _ in range(n)]
        population.append(individual)
        
    return population[:population_size]

def local_optimize(individual: List[float], iterations: int = 10) -> List[float]:
    """
    Simple local optimization around individual to improve fitness.
    """
    best_individual = individual.copy()
    best_fitness = calculate_c2(best_individual)
    
    for _ in range(iterations):
        # Make small perturbations
        mutated = best_individual.copy()
        for i in range(len(mutated)):
            if random.random() < 0.3:  # 30% chance to modify
                # Small random change
                change = random.uniform(-0.1, 0.1) * mutated[i] if mutated[i] > 0 else 0.01
                mutated[i] = max(0, mutated[i] + change)
        
        fitness = calculate_c2(mutated)
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = mutated
            
    return best_individual

def evolve_step_function(max_generations: int = 100, population_size: int = 50, 
                        elite_fraction: float = 0.2, adapt_mutation: bool = True) -> List[float]:
    """
    Evolve a step function to maximize C2 with adaptive parameters.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize population
    population = initialize_population(population_size)
    
    best_fitness = 0.0
    best_individual = None
    stale_generations = 0
    max_stale_generations = 25
    
    for generation in range(max_generations):
        # Adaptive mutation rate: starts high, decreases over time
        if adapt_mutation:
            mutation_rate = 0.3 - (0.25 * (generation / max_generations))
            mutation_rate = max(0.05, mutation_rate)  # Minimum rate
        else:
            mutation_rate = 0.1
            
        # Evaluate fitness of each individual
        fitness_scores = []
        for individual in population:
            fitness = calculate_c2(individual)
            fitness_scores.append((fitness, individual))
            
        # Sort by fitness descending
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Track best
        current_best = fitness_scores[0][0]
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = fitness_scores[0][1].copy()
            stale_generations = 0
        else:
            stale_generations += 1
            
        # Early stopping
        if stale_generations >= max_stale_generations:
            break
            
        # Selection: keep top elite_fraction
        elite_count = max(1, int(elite_fraction * population_size))
        elites = [ind for _, ind in fitness_scores[:elite_count]]
        
        # Apply local optimization to top individuals
        elite_with_local_opt = []
        for elite in elites:
            refined_elite = local_optimize(elite, iterations=5)
            elite_with_local_opt.append(refined_elite)
        elites = elite_with_local_opt
        
        # Create new generation
        new_population = elites[:]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            selected_parents = []
            for _ in range(2):
                tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
                winner = max(tournament, key=lambda x: x[0])
                selected_parents.append(winner[1])
                
            # Crossover
            if len(selected_parents) >= 2:
                child = crossover(selected_parents[0], selected_parents[1])
                # Mutation with adaptive rate
                child = mutate_individual(child, mutation_rate=mutation_rate)
                new_population.append(child)
            else:
                # If no valid crossover, just mutate one parent
                if selected_parents:
                    child = mutate_individual(selected_parents[0], mutation_rate=mutation_rate)
                    new_population.append(child)
                    
        population = new_population[:population_size]
        
    return best_individual if best_individual is not None else []

def construct_function() -> List[float]:
    """
    Function to construct step-function with high C2 value.
    """
    # Run evolution to find optimal step function
    start_time = time.time()
    
    # Use more generations and a larger population for better optimization
    evolved_function = evolve_step_function(
        max_generations=70,      # More generations for better exploration
        population_size=40,      # Larger population for diversity
        elite_fraction=0.15      # Slightly smaller elite fraction for more diversity
    )
    
    end_time = time.time()
    
    # Ensure we always return a valid function (at least one step)
    if not evolved_function:
        evolved_function = [1.0]  # Fallback to single unit step
        
    return evolved_function

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
