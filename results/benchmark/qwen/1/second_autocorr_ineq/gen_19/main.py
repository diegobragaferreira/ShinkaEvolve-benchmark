# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
from typing import List
from numba import jit
import time

@jit(nopython=True)
def compute_autoconvolution_norms_numba(f_values):
    """Optimized computation of autoconvolution norms using numba"""
    n = len(f_values)
    
    # Initialize autoconvolution array
    g = np.zeros(2*n - 1)
    
    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]
    
    # Keep only center portion
    half_len = n - 1
    g_center = g[half_len:-half_len]
    
    # Compute norms
    norm_2_squared = np.sum(g_center**2)
    norm_1 = np.sum(np.abs(g_center))
    norm_inf = np.max(np.abs(g_center))
    
    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms(f_values):
    """Compute the norms needed for C2 calculation with proper handling"""
    try:
        if not f_values:
            return 0.0, 0.0, 0.0, 0.0

        # Convert to numpy array for easier manipulation
        f = np.array(f_values, dtype=np.float64)

        # Ensure non-negative values
        f = np.maximum(f, 0.0)

        # Compute autoconvolution g = f * f
        g = np.convolve(f, f, mode='full')

        # Keep only the valid convolution part (middle)
        half_len = len(f) - 1
        g_valid = g[half_len:-half_len]

        # Compute norms
        norm_2_squared = np.sum(g_valid**2)
        norm_1 = np.sum(np.abs(g_valid))
        norm_inf = np.max(np.abs(g_valid))

        # Avoid division by zero
        if norm_1 == 0 or norm_inf == 0:
            return 0.0, 0.0, 0.0, 0.0

        # C2 = ||g||₂² / (||g||₁ · ||g||∞)
        c2 = norm_2_squared / (norm_1 * norm_inf)

        return c2, norm_2_squared, norm_1, norm_inf
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def evaluate_c2(individual: List[float]) -> tuple:
    """Evaluate fitness of individual (step function)"""
    try:
        # Ensure non-negative values
        individual = [max(0, x) for x in individual]

        # Compute C2 value
        c2, _, _, _ = compute_autoconvolution_norms(individual)

        # Return negative because we want to maximize
        return (-c2,)
    except Exception:
        # Return very poor fitness if error occurs
        return (-1e10,)

def sophisticated_initialization(n_steps):
    """Create a sophisticated initial step function"""
    # Create alternating high/low segments
    high_val = 1.0
    low_val = 0.1

    # Alternate between high and low values
    f_values = []
    for i in range(n_steps):
        if i % 2 == 0:
            f_values.append(high_val)
        else:
            f_values.append(low_val)

    # Add some randomness for diversity
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, n_steps)
    f_values = np.array(f_values) + noise
    f_values = np.maximum(f_values, 0.0)  # Ensure non-negative

    return f_values.tolist()

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value using evolutionary algorithm"""
    # Set up evolutionary algorithm
    random.seed(42)
    np.random.seed(42)

    # Problem parameters
    population_size = 50
    num_generations = 30
    num_steps = 500  # Fixed size for consistency

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define how to create individuals
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=num_steps)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define operators
    toolbox.register("evaluate", evaluate_c2)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    pop = toolbox.population(n=population_size)

    # Run evolution
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                          ngen=num_generations, stats=stats,
                                          halloffame=hof, verbose=False)
    except:
        # Fallback to simple approach if evolution fails
        return sophisticated_initialization(500)

    # Get best individual
    best_individual = hof[0]

    # Ensure non-negative values
    best_individual = [max(0, x) for x in best_individual]

    # Normalize to avoid extreme values that might cause numerical issues
    total = sum(best_individual)
    if total > 0:
        best_individual = [x / total * len(best_individual) for x in best_individual]

    return best_individual

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")