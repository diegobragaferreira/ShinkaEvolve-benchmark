# EVOLVE-BLOCK-START

import numpy as np
from numba import jit
import time
from joblib import Parallel, delayed
import random
from deap import base, creator, tools, algorithms
import copy

# Core computation module with JIT compilation
@jit(nopython=True)
def compute_autoconvolution_jit(f_vals, step_width):
    """
    Compute autoconvolution using numba for speed
    """
    n = len(f_vals)
    # Autoconvolution size is 2*n - 1
    g_size = 2 * n - 1
    g_vals = np.zeros(g_size)

    # Compute convolution directly
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_size:
                g_vals[idx] += f_vals[i] * f_vals[j]

    return g_vals

@jit(nopython=True)
def compute_norms_jit(g_vals):
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

@jit(nopython=True)
def compute_convolution_norms(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using direct computation
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Compute autoconvolution using direct convolution sum
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j] * dx

    # Compute norms using piecewise linear integration approach
    # For ||g||₂² using trapezoidal-like formula: (dx/3)(g₀² + g₀g₁ + g₁²)
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

def compute_c2(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    g2_sq, g1, ginf = compute_convolution_norms(f_values)

    if g1 == 0 or ginf == 0:
        return 0.0

    return g2_sq / (g1 * ginf)

# Optimization module
def evaluate_individual(individual):
    """Evaluate fitness of individual - returns negative C2 for maximization"""
    try:
        # Ensure non-negative values
        individual = np.maximum(0.0, individual)
        c2 = compute_c2(individual.tolist())
        return -c2  # Negative because we want to maximize C2
    except Exception as e:
        return 1e10  # Penalty for invalid solutions

def initialize_population(pop_size, n_steps, init_strategy='mixed'):
    """Initialize population with different strategies"""
    population = []

    if init_strategy == 'random':
        for _ in range(pop_size):
            individual = np.random.uniform(0, 1, n_steps)
            population.append(individual)

    elif init_strategy == 'pattern':
        # Create patterned individuals
        for _ in range(pop_size):
            individual = np.zeros(n_steps)
            # Create a symmetric pattern
            half = n_steps // 2
            for i in range(n_steps):
                if i < half:
                    individual[i] = i / half
                else:
                    individual[i] = (n_steps - i) / half
            population.append(individual)

    elif init_strategy == 'geometric':
        # Create geometrically shaped individuals
        for _ in range(pop_size):
            individual = np.zeros(n_steps)
            x = np.linspace(-0.25, 0.25, n_steps)
            # Create a bell curve
            individual = np.exp(-0.5 * (x / 0.1) ** 2)
            population.append(individual)

    else:  # mixed strategy
        # Combine different initialization approaches
        strategies = ['random', 'pattern', 'geometric']
        for i in range(pop_size):
            strategy = strategies[i % len(strategies)]
            if strategy == 'random':
                individual = np.random.uniform(0, 1, n_steps)
            elif strategy == 'pattern':
                individual = np.zeros(n_steps)
                half = n_steps // 2
                for j in range(n_steps):
                    if j < half:
                        individual[j] = j / half
                    else:
                        individual[j] = (n_steps - j) / half
            else:  # geometric
                individual = np.zeros(n_steps)
                x = np.linspace(-0.25, 0.25, n_steps)
                individual = np.exp(-0.5 * (x / 0.1) ** 2)
            population.append(individual)

    return population

def mutate_individual(individual, mut_rate=0.1, gen_num=0):
    """Mutate an individual with adaptive rate"""
    # Decrease mutation rate over generations
    adaptive_mut_rate = mut_rate * (1.0 - 0.05 * gen_num / 100)
    adaptive_mut_rate = max(adaptive_mut_rate, 0.01)

    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < adaptive_mut_rate:
            # Apply small Gaussian perturbation
            change = np.random.normal(0, 0.1 * mutated[i])
            mutated[i] = max(0, mutated[i] + change)
    return mutated

def crossover_individuals(parent1, parent2):
    """Uniform crossover between two individuals"""
    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(len(child1)):
        if np.random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]

    return child1, child2

def evolutionary_optimization(n_steps, pop_size=50, generations=50):
    """Perform evolutionary optimization"""
    # Create toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: np.random.uniform(0, 1), n_steps)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operators
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    pop = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)

    # Evolution loop
    for gen in range(generations):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.1:  # Mutation probability
                toolbox.mutate(mutant, gen_num=gen)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        # Replace the old population with the new one
        pop[:] = offspring

    # Return best individual
    best_ind = tools.selBest(pop, 1)[0]
    return np.array(best_ind)

# Improved initialization function
def generate_multiscale_gaussian_initial_function(n_steps):
    """Generate an initial function using multi-scale Gaussian pattern construction.

    This creates a structured pattern with multiple Gaussian bumps at different scales
    to encourage good convolution behavior across various spatial frequencies.
    """
    # Create base function with multi-scale Gaussian components
    f_values = np.zeros(n_steps)

    # Define multiple scales of Gaussian bumps
    scales = [n_steps // 20, n_steps // 15, n_steps // 10, n_steps // 8]
    centers = [n_steps // 4, n_steps // 2, 3 * n_steps // 4]

    # Create Gaussian bumps at different scales and positions
    for scale in scales:
        for center in centers:
            # Create Gaussian bump
            x = np.arange(n_steps)
            gauss = np.exp(-0.5 * ((x - center) / scale) ** 2)
            f_values += gauss * np.random.uniform(0.5, 1.5)  # Random amplitude

    # Add some additional structured variation
    # Create a base pattern that encourages uniformity in convolution
    base_pattern = np.sin(np.linspace(0, 4*np.pi, n_steps)) * 0.3 + 0.7
    f_values += base_pattern * 0.5

    # Ensure non-negativity and normalize
    f_values = np.maximum(f_values, 0)

    # Normalize to control the overall magnitude
    total = np.sum(f_values)
    if total > 0:
        f_values = f_values / total * 5.0

    return f_values.tolist()

def adaptive_optimization_search():
    """Perform adaptive optimization with multiple resolutions and strategies"""

    # Try different configurations with varying complexity
    resolutions = [200, 300, 400, 500]

    best_solution = None
    best_c2 = -np.inf

    # Multi-start approach with different initial patterns
    for res in resolutions:
        # Try multiple random initializations for each resolution
        for start_iter in range(3):
            np.random.seed(start_iter * 1000 + res)

            # Generate initial function using multi-scale Gaussian construction
            f_values = generate_multiscale_gaussian_initial_function(res)

            # Ensure non-negativity
            f_values = [max(0, x) for x in f_values]

            # Normalize for better numerical behavior
            total = sum(f_values)
            if total > 0:
                f_values = [x / total * 10 for x in f_values]

            # Simple local improvement
            current_f = f_values.copy()
            current_c2 = compute_c2(current_f)

            # Gradient-like local search
            for _ in range(15):
                test_f = current_f.copy()
                # Modify a few points randomly
                indices = np.random.choice(len(test_f), min(8, len(test_f)//5), replace=False)
                for idx in indices:
                    # Small perturbation
                    change = np.random.normal(0, 0.05 * test_f[idx])
                    test_f[idx] = max(0, test_f[idx] + change)

                test_c2 = compute_c2(test_f)
                if test_c2 > current_c2:
                    current_c2 = test_c2
                    current_f = test_f

            # Check if this is our best solution
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_solution = current_f.copy()

    return best_solution

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using hybrid evolutionary approach."""
    # Set parameters
    n_steps = 200
    max_time = 85  # seconds
    start_time = time.time()

    # Initialize with diverse strategies
    population = initialize_population(20, n_steps, 'mixed')

    # Evaluate initial population
    fitnesses = Parallel(n_jobs=-1)(delayed(evaluate_individual)(ind) for ind in population)

    # Find best initial solution
    best_idx = np.argmin(fitnesses)  # Minimizing negative C2
    best_individual = population[best_idx].copy()
    best_c2 = -fitnesses[best_idx]

    # Perform evolutionary optimization
    try:
        # Use evolutionary optimization for the main search
        evolved_population = evolutionary_optimization(n_steps, pop_size=30, generations=30)

        # Evaluate evolved solution
        evolved_c2 = compute_c2(evolved_population.tolist())

        if evolved_c2 > best_c2:
            best_individual = evolved_population
            best_c2 = evolved_c2

    except Exception as e:
        pass

    # Local refinement using coordinate-wise improvements
    refined_individual = best_individual.copy()
    old_c2 = best_c2

    # Refinement loop with enhanced local search
    for coord_iter in range(20):
        if time.time() - start_time > max_time:
            break

        improved = False
        # Sample a subset of indices for more efficient search
        search_indices = np.random.choice(len(refined_individual),
                                        min(20, len(refined_individual)//2),
                                        replace=False)

        for i in search_indices:
            if time.time() - start_time > max_time:
                break

            original_value = refined_individual[i]

            # Try multiple step sizes for adaptive search
            step_sizes = [0.005, 0.01, 0.02, 0.05, 0.1]

            # Try both positive and negative perturbations
            for step in step_sizes:
                for direction in [1, -1]:
                    if time.time() - start_time > max_time:
                        break
                    test_individual = refined_individual.copy()
                    new_val = original_value + direction * step
                    test_individual[i] = max(0, new_val)

                    new_c2 = compute_c2(test_individual.tolist())
                    if new_c2 > old_c2:
                        refined_individual = test_individual
                        old_c2 = new_c2
                        improved = True

                        # If we found improvement, break to move to next index
                        # to avoid getting stuck in local refinements
                        break
                if improved:
                    break

        # Break if no improvement was found in this iteration
        if not improved:
            break

    # Ensure final solution is properly formatted
    final_solution = np.maximum(0.0, refined_individual)

    # Normalize for better numerical behavior
    if np.sum(final_solution) > 0:
        final_solution = final_solution / np.sum(final_solution) * 10

    return final_solution.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")