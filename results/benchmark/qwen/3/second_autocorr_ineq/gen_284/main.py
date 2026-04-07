# EVOLVE-BLOCK-START

import numpy as np
from numba import jit
import time
from joblib import Parallel, delayed
import random
from deap import base, creator, tools, algorithms
import copy
from scipy.stats import qmc

# Core computation module with JIT compilation for maximum performance
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

# Enhanced initialization function with Latin Hypercube Sampling
def generate_lhs_initialized_functions(n_functions, n_steps, seed=42):
    """Generate multiple initial functions using Latin Hypercube Sampling"""
    # Set up LHS sampler
    sampler = qmc.LatinHypercube(d=n_steps, seed=seed)

    # Generate samples in [0,1]^n_steps
    samples = sampler.random(n=n_functions)

    # Transform to appropriate range and create functions
    lhs_functions = []

    for sample in samples:
        # Create function with patterned structure influenced by LHS sample
        f_values = np.zeros(n_steps)

        # Create base pattern with some structure
        half = n_steps // 2
        quarter = n_steps // 4

        # Base pattern influenced by sample values
        for i in range(n_steps):
            if i < quarter:
                # Rising edge - adjust slope based on sample
                base_slope = sample[i % len(sample)] * 0.5 + 0.5  # 0.5 to 1.0
                f_values[i] = i / quarter * base_slope
            elif i < half:
                # Peak region - adjust height based on sample
                peak_height = 0.7 + 0.3 * sample[(i+1) % len(sample)]  # 0.7 to 1.0
                f_values[i] = peak_height
            elif i < 3*quarter:
                # Falling edge - adjust slope based on sample
                base_slope = sample[(i+2) % len(sample)] * 0.5 + 0.3  # 0.3 to 0.8
                f_values[i] = (3*quarter - i) / quarter * base_slope
            else:
                # Low tail - adjust base level
                base_level = sample[(i+3) % len(sample)] * 0.3  # 0 to 0.3
                f_values[i] = (n_steps - i) / quarter * base_level

        # Apply smoothing with LHS influence
        smoothed = []
        for i in range(n_steps):
            if i == 0 or i == n_steps - 1:
                smoothed.append(f_values[i])
            else:
                # Weighted average influenced by sample
                weight1 = 0.2 + 0.3 * sample[i % len(sample)]
                weight2 = 0.6 + 0.2 * sample[(i+1) % len(sample)]
                weight3 = 0.2 + 0.3 * sample[(i+2) % len(sample)]
                smoothed.append(weight1 * f_values[i-1] + weight2 * f_values[i] + weight3 * f_values[i+1])

        # Normalize to ensure reasonable magnitude and non-negativity
        total_area = sum(smoothed) * (0.5 / n_steps)
        if total_area > 0:
            normalized = [x / total_area * 2.0 for x in smoothed]
        else:
            normalized = smoothed

        lhs_functions.append(normalized)

    return lhs_functions

def generate_patterned_initial_function(n_steps):
    """Generate an initial function based on mathematical insight about optimal convolution shapes"""
    # Create a function designed to produce uniform convolution profiles
    # This pattern balances peak and flat regions to encourage good C2 values

    f_values = []

    # Create a pattern that starts low, rises to a peak, then falls back down
    # but with enough variation to be interesting
    half = n_steps // 2
    quarter = n_steps // 4

    # Base pattern with multiple regions
    for i in range(n_steps):
        if i < quarter:
            # Rising edge
            f_values.append(i / quarter)
        elif i < half:
            # Peak region
            f_values.append(1.0)
        elif i < 3*quarter:
            # Falling edge
            f_values.append((3*quarter - i) / quarter)
        else:
            # Low tail
            f_values.append((n_steps - i) / quarter)

    # Apply some smoothing to reduce sharp transitions
    smoothed = []
    for i in range(n_steps):
        if i == 0 or i == n_steps - 1:
            smoothed.append(f_values[i])
        else:
            # Weighted average
            smoothed.append(0.2 * f_values[i-1] + 0.6 * f_values[i] + 0.2 * f_values[i+1])

    # Normalize to ensure reasonable magnitude
    total_area = sum(smoothed) * (0.5 / n_steps)
    if total_area > 0:
        smoothed = [x / total_area * 2.0 for x in smoothed]

    return smoothed

# Optimization module with enhanced strategies
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
    """Initialize population with different strategies including patterned approaches"""
    population = []

    if init_strategy == 'random':
        for _ in range(pop_size):
            individual = np.random.uniform(0, 1, n_steps)
            population.append(individual)

    elif init_strategy == 'pattern':
        # Create patterned individuals using our enhanced pattern generator
        for _ in range(pop_size):
            individual = generate_patterned_initial_function(n_steps)
            population.append(np.array(individual))

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
                individual = generate_patterned_initial_function(n_steps)
            else:  # geometric
                individual = np.zeros(n_steps)
                x = np.linspace(-0.25, 0.25, n_steps)
                individual = np.exp(-0.5 * (x / 0.1) ** 2)
            population.append(np.array(individual))

    return population

def mutate_individual(individual, mut_rate=0.1, gen_num=0):
    """Mutate an individual with adaptive rate"""
    # Decrease mutation rate over generations - start high, decrease to low
    adaptive_mut_rate = mut_rate * (1.0 - 0.05 * gen_num / 100)
    adaptive_mut_rate = max(adaptive_mut_rate, 0.01)

    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < adaptive_mut_rate:
            # Apply small Gaussian perturbation
            change = np.random.normal(0, 0.1 * mutated[i])
            mutated[i] = max(0, mutated[i] + change)
    return mutated

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

# Multi-resolution adaptive optimization approach with LHS
def adaptive_optimization_search(n_steps):
    """Perform adaptive optimization with multiple resolutions and strategies using LHS"""

    # Try different configurations with varying complexity
    resolutions = [150, 200, 250]

    best_solution = None
    best_c2 = -np.inf

    # Generate LHS-based initial functions for diverse exploration
    lhs_functions = generate_lhs_initialized_functions(5, n_steps, seed=42)

    # Multi-start approach with LHS and patterned initial functions
    # First try LHS functions
    for i, lhs_func in enumerate(lhs_functions):
        try:
            # Ensure non-negativity
            lhs_func = [max(0, x) for x in lhs_func]

            # Normalize for better numerical behavior
            total = sum(lhs_func)
            if total > 0:
                lhs_func = [x / total * 10 for x in lhs_func]

            # Simple local improvement
            current_f = lhs_func.copy()
            current_c2 = compute_c2(current_f)

            # Gradient-like local search
            for _ in range(10):
                test_f = current_f.copy()
                # Modify a few points randomly
                indices = np.random.choice(len(test_f), min(5, len(test_f)//4), replace=False)
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
        except Exception as e:
            continue

    # Then try patterned functions as backup
    for res in resolutions:
        # Try multiple random initializations for each resolution
        for start_iter in range(2):
            np.random.seed(start_iter * 1000 + res)

            # Generate initial function using pattern-based approach
            if res <= 200:
                # For smaller problems, use patterned construction
                f_values = generate_patterned_initial_function(res)
            else:
                # For larger problems, use a more structured approach
                f_values = generate_patterned_initial_function(res)
                # Add some noise for exploration
                noise_level = 0.1
                for i in range(len(f_values)):
                    if np.random.random() < 0.3:
                        f_values[i] *= (1 + np.random.normal(0, noise_level))

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
            for _ in range(10):
                test_f = current_f.copy()
                # Modify a few points randomly
                indices = np.random.choice(len(test_f), min(5, len(test_f)//4), replace=False)
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

    return best_solution if best_solution is not None else generate_patterned_initial_function(n_steps)

# Main optimization function with enhanced hybrid approach
def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using hybrid approach."""
    # Set parameters
    n_steps = 200
    max_time = 85  # seconds
    start_time = time.time()

    # First attempt: Multi-resolution adaptive search
    try:
        adaptive_solution = adaptive_optimization_search(n_steps)
        if adaptive_solution is not None:
            best_individual = np.array(adaptive_solution)
            best_c2 = compute_c2(adaptive_solution)
        else:
            raise Exception("Adaptive search failed")
    except:
        # Fallback to standard initialization
        population = initialize_population(20, n_steps, 'pattern')
        fitnesses = Parallel(n_jobs=-1)(delayed(evaluate_individual)(ind) for ind in population)
        best_idx = np.argmin(fitnesses)
        best_individual = population[best_idx].copy()
        best_c2 = -fitnesses[best_idx]

    # Perform evolutionary optimization with reduced scope due to time constraints
    try:
        # Use evolutionary optimization for the main search
        evolved_population = evolutionary_optimization(n_steps, pop_size=20, generations=20)

        # Evaluate evolved solution
        evolved_c2 = compute_c2(evolved_population.tolist())

        if evolved_c2 > best_c2:
            best_individual = evolved_population
            best_c2 = evolved_c2

    except Exception as e:
        pass

    # Local refinement using coordinate-wise improvements with early termination
    refined_individual = best_individual.copy()
    old_c2 = best_c2

    # Refinement loop with time management
    for coord_iter in range(15):  # Reduced iterations due to time limits
        if time.time() - start_time > max_time:
            break

        improved = False
        # Sample only a subset of indices for efficiency
        sample_indices = np.random.choice(len(refined_individual), min(10, len(refined_individual)//3), replace=False)

        for i in sample_indices:
            if time.time() - start_time > max_time:
                break

            # Try small perturbations
            original_value = refined_individual[i]
            step_sizes = [0.01, 0.05, 0.1]

            for step in step_sizes:
                # Try increasing and decreasing
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