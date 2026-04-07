# EVOLVE-BLOCK-START

import numpy as np
from numba import njit
import random
import time
from scipy import signal

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@njit
def compute_autoconvolution_norms(f_values):
    """
    Compute the autoconvolution g = f*f and return its L2, L1, and L-infinity norms.
    Uses piecewise linear integration for L2 norm.
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Create convolution using discrete convolution (equivalent to autoconvolution)
    # The convolution will have length 2*n - 1
    g = np.zeros(2 * n - 1)
    
    # Compute autoconvolution manually for efficiency
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Compute norms
    # L2 norm squared
    l2_norm_squared = 0.0
    if len(g) >= 2:
        # Piecewise linear integration using trapezoidal rule approximation
        # For intervals, we use (h/3)(y1^2 + y1*y2 + y2^2) for each adjacent pair
        h = 1.0  # Since step size is normalized to 1 for simplicity
        for i in range(len(g) - 1):
            y1 = g[i]
            y2 = g[i+1]
            l2_norm_squared += (h/3.0) * (y1*y1 + y1*y2 + y2*y2)
    
    # L1 norm
    l1_norm = np.sum(np.abs(g)) / (len(g) + 1)  # Normalize by number of intervals
    
    # L-infinity norm
    l_inf_norm = np.max(np.abs(g))
    
    return l2_norm_squared, l1_norm, l_inf_norm

@njit
def calculate_c2(l2_norm_squared, l1_norm, l_inf_norm):
    """Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    if l1_norm <= 1e-15 or l_inf_norm <= 1e-15:
        return 0.0
    return l2_norm_squared / (l1_norm * l_inf_norm)

def generate_initial_population(pop_size, min_length=500, max_length=2000):
    """Generate initial population with enhanced hybrid initialization."""
    population = []
    for _ in range(pop_size):
        # Random length within range
        length = np.random.randint(min_length, max_length + 1)
        
        # Enhanced hybrid initialization: start with alternating pattern with more structure
        individual = []
        for i in range(length):
            if i % 4 == 0:
                individual.append(np.random.uniform(0.7, 1.0))  # High peaks
            elif i % 4 == 1:
                individual.append(np.random.uniform(0.3, 0.7))  # Medium
            elif i % 4 == 2:
                individual.append(np.random.uniform(0.0, 0.3))  # Low valleys
            else:  # i % 4 == 3
                individual.append(np.random.uniform(0.0, 0.5))  # Very low
        
        # Add some structured noise to make it more diverse
        for i in range(len(individual)):
            if np.random.random() < 0.3:  # 30% chance to modify
                individual[i] = max(0.0, individual[i] + np.random.normal(0, 0.1))
        
        population.append(individual)
    
    return population

def mutate_individual(individual, generation, max_generations):
    """Mutate an individual with improved adaptive mutation rate."""
    # Improved adaptive mutation rate that decreases more gradually
    base_mutation_rate = 0.4
    decay_factor = 0.02
    mutation_rate = base_mutation_rate - (generation / max_generations) * (base_mutation_rate - decay_factor)
    
    mutated = individual.copy()
    
    # Mutate each element with probability mutation_rate
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Add normally distributed noise with adaptive scale
            noise_scale = 0.15 + (0.05 * (generation / max_generations))
            mutated[i] = max(0.0, mutated[i] + np.random.normal(0, noise_scale))
    
    return mutated

def crossover(parent1, parent2):
    """Perform improved crossover between two individuals."""
    # Ensure both parents have same length
    min_len = min(len(parent1), len(parent2))
    child1, child2 = [], []
    
    # Improved uniform crossover with bias towards preserving good features
    for i in range(min_len):
        if np.random.random() < 0.6:  # 60% chance to take from parent1
            child1.append(parent1[i])
        else:
            child1.append(parent2[i])
            
        if np.random.random() < 0.6:  # 60% chance to take from parent2
            child2.append(parent2[i])
        else:
            child2.append(parent1[i])
    
    # Handle differing lengths by extending with random values
    if len(parent1) > min_len:
        for i in range(min_len, len(parent1)):
            child1.append(np.random.uniform(0, 1))
    elif len(parent2) > min_len:
        for i in range(min_len, len(parent2)):
            child1.append(np.random.uniform(0, 1))
        
    if len(parent2) > min_len:
        for i in range(min_len, len(parent2)):
            child2.append(np.random.uniform(0, 1))
    elif len(parent1) > min_len:
        for i in range(min_len, len(parent1)):
            child2.append(np.random.uniform(0, 1))
    
    return child1, child2

def evaluate_fitness(individual):
    """Evaluate fitness of an individual (C2 value)."""
    try:
        l2, l1, l_inf = compute_autoconvolution_norms(individual)
        c2 = calculate_c2(l2, l1, l_inf)
        return c2
    except Exception:
        return 0.0

def evolutionary_optimization(max_generations=30, pop_size=70):
    """Main evolutionary optimization loop with enhanced parameters."""
    # Initialize population with improved settings
    population = generate_initial_population(pop_size, 500, 2000)
    
    best_c2 = 0.0
    best_individual = None
    
    # Evolutionary process
    for generation in range(max_generations):
        # Evaluate fitness of all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append((individual, fitness))
        
        # Sort by fitness descending
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Track best solution so far
        current_best = fitness_scores[0][1]
        if current_best > best_c2:
            best_c2 = current_best
            best_individual = fitness_scores[0][0].copy()
        
        # Select top individuals (higher selection pressure)
        top_count = int(pop_size * 0.3)  # Top 30%
        selected = [ind for ind, _ in fitness_scores[:top_count]]
        
        # Generate new population through crossover and mutation
        new_population = selected.copy()  # Elitism
        
        while len(new_population) < pop_size:
            # Selection (tournament selection with larger tournament size)
            tournament_size = 5
            parent1 = max(random.sample(selected, tournament_size), key=lambda x: evaluate_fitness(x))
            parent2 = max(random.sample(selected, tournament_size), key=lambda x: evaluate_fitness(x))
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate_individual(child1, generation, max_generations)
            child2 = mutate_individual(child2, generation, max_generations)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:pop_size]
        
        # Early stopping check - if we're stagnating, stop early
        if generation > 10 and abs(current_best - best_c2) < 1e-8:
            break
    
    return best_individual, best_c2

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Run evolutionary optimization with improved parameters
    best_individual, best_c2 = evolutionary_optimization(
        max_generations=30,
        pop_size=70
    )
    
    # Ensure we don't exceed time limits
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for final processing
        # If we're near time limit, return a good heuristic solution
        return [np.random.random() for _ in range(500)]
    
    return best_individual if best_individual is not None else [0.5]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
