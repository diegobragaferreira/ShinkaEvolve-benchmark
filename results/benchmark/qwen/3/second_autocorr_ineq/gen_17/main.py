# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
import time
from scipy import signal
from typing import List, Tuple
import multiprocessing as mp
from functools import partial

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class AutoconvolutionEvaluator:
    """Efficient evaluator for autoconvolution norms with numerical stability"""
    
    @staticmethod
    def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
        """
        Compute the three norms needed for C₂ calculation:
        ||g||₂², ||g||₁, ||g||∞ where g = f*f
        """
        if not f_values or len(f_values) == 0:
            return 0.0, 0.0, 0.0
        
        # Create step function on [-1/4, 1/4]
        n_steps = len(f_values)
        dx = 0.5 / n_steps  # Step size
        
        # Create piecewise constant function from step heights
        f = np.array(f_values, dtype=np.float64)
        
        # Ensure non-negative values
        f = np.maximum(0.0, f)
        
        # Compute autoconvolution g = f * f using discrete convolution
        g = signal.convolve(f, f, mode='full')
        
        # Result has length 2*n_steps - 1  
        g_len = len(g)
        
        # Extract the central region corresponding to [-1/4, 1/4]
        # This ensures we focus on the relevant domain
        central_start = (g_len - n_steps) // 2
        central_end = central_start + n_steps
        g_centered = g[central_start:central_end]
        
        # Normalize to match step width
        g_centered = g_centered * dx
        
        # Compute norms
        g_abs = np.abs(g_centered)
        
        # ||g||₂² using trapezoidal-like integration
        if len(g_centered) >= 2:
            # For piecewise linear segments, integrate g^2 using trapezoidal rule
            g_squared = g_centered ** 2
            # Using trapezoidal rule: sum of (y[i] + y[i+1]) * dx / 2
            norm_2_sq = np.sum((g_squared[:-1] + g_squared[1:]) * dx / 2)
        else:
            norm_2_sq = g_centered[0] ** 2 * dx if len(g_centered) > 0 else 0.0
        
        # ||g||₁
        norm_1 = np.sum(g_abs) * dx
        
        # ||g||∞
        norm_inf = np.max(g_abs)
        
        return norm_2_sq, norm_1, norm_inf

def evaluate_c2_parallel(individual, evaluator: AutoconvolutionEvaluator) -> float:
    """Parallel evaluation function with error handling"""
    try:
        # Ensure non-negative values
        f_values = [max(0.0, float(x)) for x in individual]
        
        # Compute the norms
        norm_2_sq, norm_1, norm_inf = evaluator.compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
        
        # Calculate C₂
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2 if not np.isnan(c2) and not np.isinf(c2) else 0.0
        
    except Exception:
        return 0.0

def adaptive_evolution_step(
    toolbox: base.Toolbox, 
    pop: list, 
    evaluator: AutoconvolutionEvaluator,
    generation: int,
    time_limit: float,
    start_time: float
) -> tuple:
    """Perform one generation of evolution with adaptive controls"""
    
    # Adaptive mutation and crossover rates based on generation
    generation_factor = min(1.0, generation / 50.0)
    mutpb = 0.3 - 0.1 * generation_factor  # Decreasing mutation rate
    cxpb = 0.5 - 0.2 * generation_factor  # Decreasing crossover rate
    
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    
    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    
    # Use parallel evaluation for efficiency
    if len(invalid_ind) > 0:
        with mp.Pool(min(mp.cpu_count(), 8)) as pool:
            evaluate_func = partial(evaluate_c2_parallel, evaluator=evaluator)
            fitnesses = pool.map(evaluate_func, invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
    
    # Replace population
    pop[:] = offspring
    
    return pop

def construct_function() -> List[float]:
    """Function to construct step-function with high C₂ value using evolutionary optimization."""
    
    # Algorithm parameters with early termination
    POPSIZE = 40
    MAX_GEN = 150
    TIME_LIMIT = 85  # seconds
    
    start_time = time.time()
    
    # Define problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Initialize individuals with adaptive generation
    def random_height():
        # Use log-normal distribution to generate more balanced values
        return abs(np.random.lognormal(0, 0.5)) * 0.5
    
    def create_individual():
        # Random number of steps between 200 and 800 for better balance
        n_steps = random.randint(200, 800)
        return creator.Individual([random_height() for _ in range(n_steps)])
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.05)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize evaluator
    evaluator = AutoconvolutionEvaluator()
    
    # Initialize population
    pop = toolbox.population(n=POPSIZE)
    
    # Evaluate initial population
    fitnesses = []
    with mp.Pool(min(mp.cpu_count(), 8)) as pool:
        evaluate_func = partial(evaluate_c2_parallel, evaluator=evaluator)
        fitnesses = pool.map(evaluate_func, pop)
    
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
    
    # Track best solution
    best_fitness = max(fitnesses) if fitnesses else 0.0
    best_individual = None
    
    # Main evolution loop with adaptive controls
    for gen in range(MAX_GEN):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Adaptive evolution step
        pop = adaptive_evolution_step(toolbox, pop, evaluator, gen, TIME_LIMIT, start_time)
        
        # Track best solution after each generation
        current_fitnesses = [ind.fitness.values[0] for ind in pop]
        max_fitness = max(current_fitnesses)
        
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            # Find the actual individual with best fitness
            best_idx = current_fitnesses.index(max_fitness)
            best_individual = list(pop[best_idx])
    
    # Final evaluation of best individual
    if best_individual is not None:
        final_fitness = evaluate_c2_parallel(best_individual, evaluator)
        if final_fitness > 0.0:
            return [max(0.0, float(x)) for x in best_individual]
    
    # Fallback: return reasonable solution with better initialization
    fallback_size = random.randint(300, 600)
    fallback_vals = []
    for _ in range(fallback_size):
        fallback_vals.append(abs(np.random.lognormal(0, 0.5)) * 0.5)
    return fallback_vals

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
