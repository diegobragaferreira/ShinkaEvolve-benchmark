# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from deap import base, creator, tools, algorithms
import random
import time

def compute_autoconvolution_norms(f_values):
    """Compute the three norms needed for C2 calculation"""
    # Create step function
    n = len(f_values)
    if n == 0:
        return 0, 0, 0

    # Create step function on [-1/4, 1/4] with equal spacing
    step_width = 0.5 / n
    x = np.linspace(-0.25, 0.25, n, endpoint=False) + step_width/2

    # Compute autoconvolution g = f * f
    # Using convolution with proper scaling
    f = np.array(f_values)
    g = signal.convolve(f, f, mode='full')

    # The result has 2*n-1 elements, centered around index n-1
    # We want the middle portion that corresponds to the actual convolution
    mid_start = n - 1
    mid_end = 2 * n - 2
    g_middle = g[mid_start:mid_end]

    # Compute norms
    g_squared = g_middle ** 2
    g_abs = np.abs(g_middle)

    # L2 norm squared
    norm_2_sq = np.sum(g_squared)

    # L1 norm
    norm_1 = np.sum(g_abs)

    # L-infinity norm
    norm_inf = np.max(g_abs)

    return norm_2_sq, norm_1, norm_inf

def calculate_c2(f_values):
    """Calculate C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

        # Handle numerical stability
        if norm_1 < 1e-12 or norm_inf < 1e-12:
            return 0

        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    except Exception:
        return 0

def construct_function() -> list[float]:
    """Optimized function to construct step-function with high C2 value using evolutionary algorithm."""

    # Set up the evolutionary algorithm
    random.seed(42)
    np.random.seed(42)

    # Problem parameters
    POPSIZE = 50
    NGEN = 30
    IND_SIZE = 200  # Number of steps
    MUTPB = 0.2
    CXPB = 0.5

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define the gene representation (non-negative real values)
    def create_individual():
        return [random.uniform(0, 2) for _ in range(IND_SIZE)]

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define evaluation function
    def eval_func(individual):
        # Ensure all values are non-negative
        individual = [max(0, x) for x in individual]
        c2 = calculate_c2(individual)
        return (c2,)

    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    pop = toolbox.population(n=POPSIZE)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolution loop
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace old population with new one
        pop[:] = offspring

    # Get best individual
    best_ind = tools.selBest(pop, 1)[0]
    # Ensure non-negative values
    best_ind = [max(0, x) for x in best_ind]

    return best_ind

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")