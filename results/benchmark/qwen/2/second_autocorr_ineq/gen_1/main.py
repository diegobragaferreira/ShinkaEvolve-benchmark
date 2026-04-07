# EVOLVE-BLOCK-START

import numpy as np
import random
from deap import base, creator, tools, algorithms
from functools import partial
import time

def compute_autoconvolution_norms(f_values):
    """
    Compute the norms needed for C2 calculation using piecewise linear integration
    """
    # Scale the function to domain [-1/4, 1/4]
    n = len(f_values)
    if n == 0:
        return 0, 0, 0

    # Create step function on [-1/4, 1/4]
    step_width = 0.5 / n
    x = np.linspace(-0.25, 0.25, n+1)

    # Compute autoconvolution g = f * f
    # Using discrete convolution with proper spacing
    g_values = np.zeros(2*n - 1)
    for i in range(n):
        for j in range(n):
            idx = i + j
            g_values[idx] += f_values[i] * f_values[j]

    # Normalize by step width for proper integral approximation
    g_values *= step_width

    # Compute norms using piecewise linear integration
    # ||g||₂² using trapezoidal-like piecewise integration
    g_squared = g_values ** 2
    g_abs = np.abs(g_values)

    # For ||g||₂² using piecewise linear integration formula
    # For adjacent points y1, y2 with distance h: contribution = h/3*(y1² + y1*y2 + y2²)
    if len(g_values) < 2:
        norm_2_sq = 0
    else:
        norm_2_sq = 0
        h = 2 * step_width  # Step size for convolution result
        for i in range(len(g_values) - 1):
            y1 = g_values[i]
            y2 = g_values[i+1]
            norm_2_sq += (h/3) * (y1**2 + y1*y2 + y2**2)

    # ||g||₁: sum of absolute values / number of intervals
    norm_1 = np.sum(g_abs) / (len(g_values) - 1) if len(g_values) > 1 else 0

    # ||g||∞: maximum absolute value
    norm_inf = np.max(g_abs) if len(g_values) > 0 else 0

    return norm_2_sq, norm_1, norm_inf

def calculate_c2(f_values):
    """Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

        # Handle numerical issues
        if norm_1 == 0 or norm_inf == 0:
            return 0.0

        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    except Exception:
        return 0.0

# Evolutionary algorithm setup
def create_individual():
    """Create a random individual (step function)"""
    n = random.randint(100, 1000)  # Random length
    return [random.uniform(0, 1) for _ in range(n)]

def evaluate(individual):
    """Evaluate fitness (C2 value)"""
    c2 = calculate_c2(individual)
    return (c2,)  # Return tuple for DEAP

def mutate_individual(individual):
    """Mutate an individual"""
    for i in range(len(individual)):
        if random.random() < 0.1:  # 10% mutation rate
            individual[i] = max(0, individual[i] + random.gauss(0, 0.1))  # Ensure non-negative
    return individual,

def crossover_individuals(ind1, ind2):
    """Crossover two individuals"""
    min_len = min(len(ind1), len(ind2))
    if min_len > 0:
        cx_point = random.randint(1, min_len - 1)
        ind1[cx_point:], ind2[cx_point:] = ind2[cx_point:], ind1[cx_point:]
    return ind1, ind2

def construct_function() -> list[float]:
    """Construct step function using evolutionary optimization"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialize DEAP structures
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Algorithm parameters
    population_size = 50
    generations = 20

    # Create initial population
    population = toolbox.population(n=population_size)

    # Run evolution
    for gen in range(generations):
        # Select
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        population[:] = offspring

        # Track best individual
        best = tools.selBest(population, 1)[0]
        current_c2 = evaluate(best)[0]

    # Return the best solution found
    return tools.selBest(population, 1)[0]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")