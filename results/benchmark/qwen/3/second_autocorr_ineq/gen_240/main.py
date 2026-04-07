# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
import time
from scipy import signal
from typing import List
from numba import jit
from joblib import Parallel, delayed
import copy

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

def create_multiscale_gaussian_individual():
    """Create an individual with multi-scale Gaussian bumps for better initialization"""
    n_steps = random.randint(100, 1000)

    # Create a base function with multiple Gaussian bumps
    individual = np.zeros(n_steps)

    # Add several Gaussian bumps at different positions and scales
    num_bumps = random.randint(3, 8)
    for _ in range(num_bumps):
        # Random center position (normalized to [0, 1])
        center = random.random()
        # Random scale (width) of the bump
        scale = random.uniform(0.05, 0.2)
        # Random height
        height = random.uniform(0.5, 1.5)

        # Create Gaussian bump
        x = np.linspace(0, 1, n_steps)
        gaussian = height * np.exp(-0.5 * ((x - center) / scale) ** 2)
        individual += gaussian

    # Ensure non-negative values and normalize somewhat
    individual = np.maximum(individual, 0.0)

    # Add some random perturbations for diversity
    for i in range(len(individual)):
        if random.random() < 0.05:  # 5% chance to perturb
            individual[i] = max(0.0, individual[i] + random.gauss(0, 0.05))

    # Convert to list and wrap in Individual
    return creator.Individual(individual.tolist())

def create_uniform_individual():
    """Create uniform distribution initialization"""
    n_steps = random.randint(100, 1000)
    return creator.Individual([random.uniform(0.5, 1.0) for _ in range(n_steps)])

def create_alternating_individual():
    """Create alternating high/low pattern"""
    n_steps = random.randint(100, 1000)
    individual = []
    for i in range(n_steps):
        if i % 2 == 0:
            individual.append(random.uniform(0.7, 1.0))
        else:
            individual.append(random.uniform(0.1, 0.3))
    return creator.Individual(individual)

def create_gaussian_individual():
    """Create normally distributed peaks"""
    n_steps = random.randint(100, 1000)
    individual = [random.gauss(0.5, 0.2) for _ in range(n_steps)]
    return creator.Individual(individual)

def create_sine_individual():
    """Create sine-based pattern"""
    n_steps = random.randint(100, 1000)
    individual = []
    for i in range(n_steps):
        # Sine-based pattern with some noise
        val = 0.5 + 0.3 * np.sin(i * 2 * np.pi / n_steps) + random.gauss(0, 0.05)
        individual.append(max(0.0, val))
    return creator.Individual(individual)

def create_structured_individual():
    """Create structured pattern with high peaks and low valleys"""
    n_steps = random.randint(100, 1000)
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
        if random.random() < 0.4:  # 40% chance to modify
            individual[i] = max(0.0, individual[i] + random.gauss(0, 0.1))
    return creator.Individual(individual)

def adaptive_local_search(individual, max_iterations=50):
    """Apply adaptive local search to improve individual"""
    best_individual = individual.copy()
    best_c2 = evaluate_c2(best_individual)[0]
    
    # Try different perturbation strategies
    for _ in range(max_iterations):
        # Create candidate by perturbing randomly selected elements
        candidate = individual.copy()
        
        # Pick random elements to modify
        num_changes = random.randint(1, max(1, len(individual) // 10))
        indices = random.sample(range(len(candidate)), num_changes)
        
        for i in indices:
            # Apply adaptive change based on current value
            current_val = candidate[i]
            # Small perturbation
            perturbation = random.gauss(0, 0.05 * current_val if current_val > 0 else 0.05)
            candidate[i] = max(0.0, current_val + perturbation)
        
        # Evaluate candidate
        candidate_c2 = evaluate_c2(candidate)[0]
        
        if candidate_c2 > best_c2:
            best_c2 = candidate_c2
            best_individual = candidate.copy()
    
    return best_individual, best_c2

def run_single_optimization(init_func, strategy_name, start_time, time_limit):
    """Run a single optimization with given initialization strategy"""
    if time.time() - start_time > time_limit - 5:  # Leave margin for cleanup
        return None, 0.0

    print(f"Running optimization with strategy: {strategy_name}")

    # Define problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", init_func)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_c2)
    toolbox.register("mate", tools.cxUniform, indpb=0.05)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Algorithm parameters
    INITIAL_POPSIZE = 30
    NGEN = 200
    INITIAL_MUTPB = 0.3
    INITIAL_CXPB = 0.5
    STAGNATION_LIMIT = 20

    # Initialize population
    pop = toolbox.population(n=INITIAL_POPSIZE)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Main evolution loop with enhanced diversity management
    best_fitness = 0.0
    best_individual = None
    generation_counter = 0
    stagnation_count = 0
    prev_best_fitness = 0.0

    for gen in range(NGEN):
        if time.time() - start_time > time_limit:
            break

        generation_counter += 1

        # Adaptive population size: grow if early generations and slow down
        if gen < NGEN//3 and len(pop) < 100:
            new_popsize = min(100, len(pop) + 2)
            if new_popsize > len(pop):
                # Grow population by adding random individuals
                extra_individuals = toolbox.population(n=new_popsize - len(pop))
                pop.extend(extra_individuals)

        # Adaptive mutation rate: decrease over generations
        adaptive_mutpb = INITIAL_MUTPB * (1.0 - gen / NGEN)
        adaptive_cxpb = INITIAL_CXPB * (1.0 - gen / NGEN)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < adaptive_cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < adaptive_mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Apply local search to top individuals
        # Sort population by fitness and select top 15%
        sorted_pop = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)
        top_count = max(1, len(pop) // 7)
        top_individuals = sorted_pop[:top_count]

        # Apply local search to top individuals
        for i, ind in enumerate(top_individuals):
            if random.random() < 0.4:  # Apply local search to 40% of top individuals
                improved_ind, improved_fitness = adaptive_local_search(ind)
                if improved_fitness > ind.fitness.values[0]:
                    # Replace with improved version
                    ind[:] = improved_ind
                    ind.fitness.values = (improved_fitness,)

        # Replace population
        pop[:] = offspring

        # Track best solution
        best_in_gen = max(pop, key=lambda x: x.fitness.values[0])
        if best_in_gen.fitness.values[0] > best_fitness:
            best_fitness = best_in_gen.fitness.values[0]
            best_individual = list(best_in_gen)
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Check for stagnation and introduce diversity
        if stagnation_count >= STAGNATION_LIMIT:
            # Introduce some randomness to escape local optima
            for i in range(len(pop) // 5):  # Perturb 20% of population
                if random.random() < 0.5:
                    ind_idx = random.randint(0, len(pop)-1)
                    ind = pop[ind_idx]
                    for j in range(len(ind)):
                        if random.random() < 0.1:
                            ind[j] = max(0.0, ind[j] + random.gauss(0, 0.05))
            stagnation_count = 0

    return best_individual, best_fitness

def construct_function() -> List[float]:
    """Function to construct step-function with high C₂ value using enhanced evolutionary optimization."""

    # Algorithm parameters
    TIME_LIMIT = 85  # seconds

    start_time = time.time()

    # Define multiple initialization strategies
    init_strategies = [
        ("multiscale_gaussian", create_multiscale_gaussian_individual),
        ("uniform", create_uniform_individual),
        ("alternating", create_alternating_individual),
        ("gaussian", create_gaussian_individual),
        ("sine", create_sine_individual),
        ("structured", create_structured_individual)
    ]

    best_overall_fitness = 0.0
    best_overall_individual = None

    # Run multiple parallel optimizations with different initialization strategies
    results = Parallel(n_jobs=-1)(
        delayed(run_single_optimization)(
            init_func, 
            strategy_name, 
            start_time, 
            TIME_LIMIT
        )
        for strategy_name, init_func in init_strategies
    )

    # Process results and find the best overall
    for individual, fitness in results:
        if individual is not None and fitness > best_overall_fitness:
            best_overall_fitness = fitness
            best_overall_individual = individual

    # Return the best individual found across all strategies
    if best_overall_individual is not None:
        return [max(0.0, float(x)) for x in best_overall_individual]

    # Fallback: return reasonable random solution
    fallback_size = random.randint(100, 1000)
    return [abs(random.gauss(0.5, 0.3)) for _ in range(fallback_size)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")