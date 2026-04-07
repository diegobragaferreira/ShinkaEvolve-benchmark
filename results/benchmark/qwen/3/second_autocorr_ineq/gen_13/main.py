# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import random
from typing import List
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

@jit(nopython=True)
def compute_autoconvolution_jit(f_vals: np.ndarray, n: int) -> np.ndarray:
    """Compute autoconvolution g = f * f efficiently using numba"""
    # Create output array
    g = np.zeros(2*n - 1)
    
    # Compute convolution using nested loop (optimized for Numba)
    for i in range(len(f_vals)):
        for j in range(len(f_vals)):
            idx = i + j
            if 0 <= idx < len(g):
                g[idx] += f_vals[i] * f_vals[j]
    
    return g

@jit(nopython=True)
def compute_c2_norms_jit(g_vals: np.ndarray) -> tuple:
    """Compute norms needed for C2 efficiently with numba"""
    # Compute L2 norm squared
    l2_sq = 0.0
    for i in range(len(g_vals)):
        l2_sq += g_vals[i] * g_vals[i]
    
    # Compute L1 norm
    l1 = 0.0
    for i in range(len(g_vals)):
        l1 += g_vals[i]
    
    # Compute infinity norm
    linf = 0.0
    for i in range(len(g_vals)):
        if g_vals[i] > linf:
            linf = g_vals[i]
    
    return l2_sq, l1, linf

@jit(nopython=True)
def compute_c2_score_jit(f_vals: np.ndarray) -> float:
    """Compute C2 score for given function values"""
    # Get size parameters
    n = len(f_vals)
    if n == 0:
        return 0.0
    
    # Compute autoconvolution
    g_vals = compute_autoconvolution_jit(f_vals, n)
    
    # Compute norms
    l2_sq, l1, linf = compute_c2_norms_jit(g_vals)
    
    # Avoid division by zero
    if l1 <= 1e-15 or linf <= 1e-15:
        return 0.0
    
    # Return C2 score
    return l2_sq / (l1 * linf)

@jit(nopython=True)
def initialize_population_jit(pop_size: int, individual_size: int) -> np.ndarray:
    """Initialize population with random step functions"""
    population = np.zeros((pop_size, individual_size))
    for i in range(pop_size):
        for j in range(individual_size):
            population[i, j] = np.random.random()
    return population

@jit(nopython=True)
def mutate_individual_jit(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Mutate an individual by randomly changing some values"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] = np.random.random()
    return mutated

@jit(nopython=True)
def crossover_individuals_jit(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float) -> tuple:
    """Perform crossover between two individuals"""
    if np.random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    
    return child1, child2

@jit(nopython=True)
def select_parents_jit(population: np.ndarray, fitness_scores: np.ndarray, tournament_size: int = 3) -> tuple:
    """Select parents using tournament selection"""
    parent1_idx = np.random.randint(0, len(population))
    parent2_idx = np.random.randint(0, len(population))
    
    # Tournament selection for first parent
    for _ in range(tournament_size - 1):
        contestant_idx = np.random.randint(0, len(population))
        if fitness_scores[contestant_idx] > fitness_scores[parent1_idx]:
            parent1_idx = contestant_idx
    
    # Tournament selection for second parent
    for _ in range(tournament_size - 1):
        contestant_idx = np.random.randint(0, len(population))
        if fitness_scores[contestant_idx] > fitness_scores[parent2_idx]:
            parent2_idx = contestant_idx
    
    return population[parent1_idx], population[parent2_idx]

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value - Optimized version"""
    
    # Parameters for evolution
    pop_size = 100
    generations = 50
    individual_size = 500  # Start with smaller size for efficiency
    mutation_rate = 0.1
    crossover_rate = 0.8
    elite_size = 10
    
    # Initialize population
    population = initialize_population_jit(pop_size, individual_size)
    
    best_fitness = -1.0
    best_individual = None
    
    # Evolution loop
    for gen in range(generations):
        # Compute fitness for each individual
        fitness_scores = np.zeros(pop_size)
        for i in range(pop_size):
            fitness_scores[i] = compute_c2_score_jit(population[i])
            
            # Track best individual
            if fitness_scores[i] > best_fitness:
                best_fitness = fitness_scores[i]
                best_individual = population[i].copy()
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = population[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]
        
        # Keep elite
        elite = population[:elite_size]
        
        # Generate new population
        new_population = []
        new_population.extend(elite)
        
        # Fill remaining slots with offspring
        while len(new_population) < pop_size:
            # Select parents
            parent1, parent2 = select_parents_jit(population, fitness_scores)
            
            # Crossover
            child1, child2 = crossover_individuals_jit(parent1, parent2, crossover_rate)
            
            # Mutate
            child1 = mutate_individual_jit(child1, mutation_rate)
            child2 = mutate_individual_jit(child2, mutation_rate)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        # Update population
        population = np.array(new_population[:pop_size])
    
    # Final refinement: Try to improve the best solution
    if best_individual is not None:
        # Perform local search around the best individual
        current_best = best_individual.copy()
        current_fitness = compute_c2_score_jit(current_best)
        
        # Try small perturbations
        for _ in range(100):
            # Create small random perturbation
            perturbed = current_best.copy()
            for i in range(len(perturbed)):
                if np.random.random() < 0.1:  # 10% chance to modify
                    perturbed[i] = max(0.0, min(1.0, perturbed[i] + np.random.normal(0, 0.05)))
            
            new_fitness = compute_c2_score_jit(perturbed)
            if new_fitness > current_fitness:
                current_best = perturbed
                current_fitness = new_fitness
        
        # Return final result with some noise to ensure diversity
        return [float(x) for x in current_best]
    
    # Fallback if no good individual found
    return [np.random.random() for _ in range(100)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
