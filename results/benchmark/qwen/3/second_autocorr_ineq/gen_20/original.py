# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
import time
from scipy import signal
from typing import List

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def compute_autoconvolution_norms(f_values: List[float]):
    """
    Compute the three norms needed for C₂ calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    if not f_values:
        return 0.0, 0.0, 0.0

    # Create step function on [-1/4, 1/4]
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    dx = 0.5 / n_steps  # Step size
    x = np.linspace(-0.25, 0.25, n_steps, endpoint=False) + dx/2

    # Create piecewise constant function from step heights
    f = np.array(f_values)

    # Compute autoconvolution g = f * f
    # Using discrete convolution
    g = signal.convolve(f, f, mode='full')

    # Adjust indices for the correct domain
    # Result has length 2*n_steps - 1
    g_len = len(g)
    g_x = np.linspace(-0.5, 0.5, g_len, endpoint=False) + 0.5/g_len/2

    # Extract the central region corresponding to [-1/4, 1/4]
    # This is a rough approximation - more precise would involve exact integration
    central_start = (g_len - n_steps) // 2
    central_end = central_start + n_steps
    g_centered = g[central_start:central_end]

    # Normalize to match step width
    g_centered = g_centered * dx

    # Compute norms
    g_abs = np.abs(g_centered)

    # ||g||₂² using trapezoidal-like integration
    # For piecewise linear segments, use formula for trapezoidal rule
    if len(g_centered) >= 2:
        # Trapezoidal rule for integral of g^2
        g_squared = g_centered ** 2
        # Approximate integral of g^2 with trapezoidal rule
        norm_2_sq = np.sum((g_squared[:-1] + g_squared[1:]) * dx / 2)
    else:
        norm_2_sq = g_centered[0] ** 2 * dx if len(g_centered) > 0 else 0.0

    # ||g||₁
    norm_1 = np.sum(g_abs) * dx

    # ||g||∞
    norm_inf = np.max(g_abs)

    return norm_2_sq, norm_1, norm_inf

def evaluate_c2(individual):
    """Evaluate fitness for individual (step function heights)"""
    try:
        # Convert individual to list of floats
        f_values = [max(0.0, float(x)) for x in individual]  # Ensure non-negative

        # Compute the norms
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_1 <= 1e-12 or norm_inf <= 1e-12:
            return (0.0,)

        # Calculate C₂
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return (c2,)
    except Exception as e:
        return (0.0,)

def construct_function() -> List[float]:
    """Function to construct step-function with high C₂ value using evolutionary optimization."""

    # Algorithm parameters
    POPSIZE = 50
    NGEN = 200
    MUTPB = 0.3
    CXPB = 0.5
    TOURNAMENT_SIZE = 3
    TIME_LIMIT = 85  # seconds

    start_time = time.time()

    # Define problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Initialize individuals with random positive values (non-negative)
    def random_height():
        return abs(random.gauss(0.5, 0.3))

    def create_individual():
        # Random number of steps between 100 and 1000
        n_steps = random.randint(100, 1000)
        return creator.Individual([random_height() for _ in range(n_steps)])

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_c2)
    toolbox.register("mate", tools.cxUniform, indpb=0.05)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Initialize population
    pop = toolbox.population(n=POPSIZE)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Main evolution loop
    best_fitness = 0.0
    best_individual = None
    generation_counter = 0

    for gen in range(NGEN):
        if time.time() - start_time > TIME_LIMIT:
            break

        generation_counter += 1

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
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Track best solution
        best_in_gen = max(pop, key=lambda x: x.fitness.values[0])
        if best_in_gen.fitness.values[0] > best_fitness:
            best_fitness = best_in_gen.fitness.values[0]
            best_individual = list(best_in_gen)

    # Final evaluation of best individual
    if best_individual is not None:
        final_fitness = evaluate_c2(best_individual)[0]
        if final_fitness > 0.0:
            # Return the best individual found
            return [max(0.0, float(x)) for x in best_individual]

    # Fallback: return reasonable random solution
    fallback_size = random.randint(100, 1000)
    return [abs(random.gauss(0.5, 0.3)) for _ in range(fallback_size)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")