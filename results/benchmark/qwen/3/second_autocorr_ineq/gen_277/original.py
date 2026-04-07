# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
import time
from scipy import signal
from typing import List
from numba import jit
import math

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """
    Compute autoconvolution using numba for speed
    """
    n = len(f_vals)
    # Autoconvolution size is 2*n - 1
    g_size = 2 * n - 1
    g_vals = np.zeros(g_size)

    # Compute convolution directly with numba optimization
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_size:
                g_vals[idx] += f_vals[i] * f_vals[j]

    return g_vals

@jit(nopython=True)
def compute_norms_numba(g_vals, dx):
    """
    Compute norms efficiently with numba
    """
    n = len(g_vals)

    # Compute L1 norm (sum of absolute values)
    l1_norm = 0.0
    for i in range(n):
        l1_norm += abs(g_vals[i])

    # Compute L2 norm squared
    l2_norm_sq = 0.0
    for i in range(n):
        l2_norm_sq += g_vals[i] * g_vals[i]

    # Compute infinity norm
    linf_norm = 0.0
    for i in range(n):
        val = abs(g_vals[i])
        if val > linf_norm:
            linf_norm = val

    return l1_norm, l2_norm_sq, linf_norm

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

    # Create piecewise constant function from step heights
    f = np.array(f_values)

    # Compute autoconvolution g = f * f using numba optimized version
    g = compute_autoconvolution_numba(f)

    # Adjust indices for the correct domain
    # Result has length 2*n_steps - 1
    g_len = len(g)

    # Extract the central region corresponding to [-1/4, 1/4]
    # This takes the middle n_steps elements of the full convolution
    central_start = (g_len - n_steps) // 2
    central_end = central_start + n_steps
    g_centered = g[central_start:central_end]

    # Compute norms using numba optimized version
    g_abs = np.abs(g_centered)

    # Compute norms using numba
    norm_1, norm_2_sq, norm_inf = compute_norms_numba(g_abs, dx)

    # Normalize for the trapezoidal integration that will happen below
    # The actual integral should be scaled properly
    # Since we're using trapezoidal integration, and we have dx spacing,
    # we multiply by dx the norms that represent integrals
    norm_1 *= dx
    norm_2_sq *= dx

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
    """Function to construct step-function with high C₂ value using enhanced evolutionary optimization."""

    # Algorithm parameters
    INITIAL_POPSIZE = 30
    MAX_POPSIZE = 150
    NGEN = 200
    INITIAL_MUTPB = 0.4
    INITIAL_CXPB = 0.6
    TOURNAMENT_SIZE = 3
    TIME_LIMIT = 85  # seconds
    DIVERSITY_THRESHOLD = 0.01
    STAGNATION_LIMIT = 25
    IMPROVEMENT_THRESHOLD = 1e-6

    start_time = time.time()

    # Define problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Multiple initialization strategies to increase diversity
    def create_uniform_individual():
        n_steps = random.randint(150, 800)
        return creator.Individual([random.uniform(0.5, 1.0) for _ in range(n_steps)])

    def create_alternating_individual():
        n_steps = random.randint(150, 800)
        individual = []
        for i in range(n_steps):
            if i % 2 == 0:
                individual.append(random.uniform(0.7, 1.0))
            else:
                individual.append(random.uniform(0.1, 0.3))
        return creator.Individual(individual)

    def create_gaussian_individual():
        n_steps = random.randint(150, 800)
        # Create normally distributed peaks
        individual = [random.gauss(0.5, 0.2) for _ in range(n_steps)]
        return creator.Individual(individual)

    def create_sine_individual():
        n_steps = random.randint(150, 800)
        individual = []
        for i in range(n_steps):
            # Sine-based pattern with some noise
            val = 0.5 + 0.3 * np.sin(i * 2 * np.pi / n_steps) + random.gauss(0, 0.05)
            individual.append(max(0.0, val))
        return creator.Individual(individual)

    def create_structured_individual():
        """Create a structured pattern with high peaks and low valleys."""
        n_steps = random.randint(150, 800)
        individual = []
        for i in range(n_steps):
            # Create structured pattern: high, medium, low, low, high, etc.
            pattern_pos = i % 8
            if pattern_pos < 2:
                individual.append(random.uniform(0.8, 1.0))
            elif pattern_pos < 5:
                individual.append(random.uniform(0.3, 0.7))
            else:
                individual.append(random.uniform(0.0, 0.3))
        # Add structured noise to make it more diverse
        for i in range(len(individual)):
            if random.random() < 0.3:  # 30% chance to modify
                individual[i] = max(0.0, individual[i] + random.gauss(0, 0.1))
        return creator.Individual(individual)

    # Create multiple initialization strategies
    init_strategies = [
        create_uniform_individual,
        create_alternating_individual,
        create_gaussian_individual,
        create_sine_individual,
        create_structured_individual
    ]

    best_overall_fitness = 0.0
    best_overall_individual = None

    # Run multiple parallel optimizations with different initialization strategies
    for strategy_idx, init_func in enumerate(init_strategies):
        if time.time() - start_time > TIME_LIMIT - 5:  # Leave margin for cleanup
            break

        print(f"Running optimization with strategy {strategy_idx + 1}")

        # Use a fresh toolbox for each strategy to avoid interference
        local_toolbox = base.Toolbox()
        local_toolbox.register("individual", init_func)
        local_toolbox.register("population", tools.initRepeat, list, local_toolbox.individual)
        local_toolbox.register("evaluate", evaluate_c2)
        local_toolbox.register("mate", tools.cxUniform, indpb=0.1)
        local_toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.15)
        local_toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

        # Initialize population with dynamic sizing
        current_popsize = INITIAL_POPSIZE
        pop = local_toolbox.population(n=current_popsize)

        # Evaluate initial population
        fitnesses = list(map(local_toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Main evolution loop with enhanced diversity management
        best_fitness = 0.0
        best_individual = None
        generation_counter = 0
        stagnation_count = 0
        prev_best_fitness = 0.0
        improvement_history = []

        for gen in range(NGEN):
            if time.time() - start_time > TIME_LIMIT:
                break

            generation_counter += 1

            # Adaptive population size: grow if early generations and slow down
            if gen < NGEN//3 and current_popsize < MAX_POPSIZE:
                current_popsize = min(MAX_POPSIZE, current_popsize + 3)
                # Grow population by adding random individuals
                extra_individuals = local_toolbox.population(n=current_popsize - len(pop))
                pop.extend(extra_individuals)

            # Adaptive mutation rate: decrease over generations with more aggressive decay
            adaptive_mutpb = INITIAL_MUTPB * (1.0 - gen / NGEN)**1.5
            adaptive_cxpb = INITIAL_CXPB * (1.0 - gen / NGEN)**1.2

            # Select the next generation individuals
            offspring = local_toolbox.select(pop, len(pop))
            offspring = list(map(local_toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < adaptive_cxpb:
                    local_toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < adaptive_mutpb:
                    local_toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(local_toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            pop[:] = offspring

            # Track best solution
            best_in_gen = max(pop, key=lambda x: x.fitness.values[0])
            if best_in_gen.fitness.values[0] > best_fitness:
                best_fitness = best_in_gen.fitness.values[0]
                best_individual = list(best_in_gen)
                stagnation_count = 0
                improvement_history.append(best_fitness)
            else:
                stagnation_count += 1

            # Check for stagnation and introduce diversity
            if stagnation_count >= STAGNATION_LIMIT:
                # Introduce some randomness to escape local optima
                for i in range(len(pop) // 4):  # Perturb 25% of population
                    if random.random() < 0.6:
                        ind_idx = random.randint(0, len(pop)-1)
                        ind = pop[ind_idx]
                        for j in range(len(ind)):
                            if random.random() < 0.15:
                                ind[j] = max(0.0, ind[j] + random.gauss(0, 0.08))
                stagnation_count = 0

            # Check diversity periodically
            if gen % 10 == 0 and len(pop) > 1:
                # Compute diversity metric - variance of fitness values
                fitness_values = [ind.fitness.values[0] for ind in pop if ind.fitness.values[0] > 0]
                if len(fitness_values) >= 2:
                    diversity = np.var(fitness_values)
                    # If diversity is too low, add some new random individuals
                    if diversity < DIVERSITY_THRESHOLD:
                        additional_individuals = local_toolbox.population(n=len(pop)//3)
                        pop.extend(additional_individuals)

            # Early termination check based on improvement rate
            if len(improvement_history) >= 2:
                improvement_rate = improvement_history[-1] - improvement_history[-2]
                if improvement_rate < IMPROVEMENT_THRESHOLD:
                    # Stop if there's no significant improvement
                    if len(improvement_history) > 10:
                        improvement_avg = sum(improvement_history[-10:]) / 10.0
                        if improvement_avg < IMPROVEMENT_THRESHOLD * 0.1:
                            break

        # Final evaluation of best individual from this run
        if best_individual is not None:
            final_fitness = evaluate_c2(best_individual)[0]
            if final_fitness > 0.0 and final_fitness > best_overall_fitness:
                best_overall_fitness = final_fitness
                best_overall_individual = list(best_individual)

    # Return the best individual found across all strategies
    if best_overall_individual is not None:
        return [max(0.0, float(x)) for x in best_overall_individual]

    # Fallback: return reasonable random solution
    fallback_size = random.randint(150, 800)
    return [abs(random.gauss(0.5, 0.3)) for _ in range(fallback_size)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")