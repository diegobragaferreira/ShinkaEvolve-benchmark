# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
import time
from scipy import signal
from typing import List
from numba import jit, prange

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True, fastmath=True)
def compute_autoconvolution_numba(f_vals):
    """
    Compute autoconvolution using numba for speed
    """
    n = len(f_vals)
    # Autoconvolution size is 2*n - 1
    g_size = 2 * n - 1
    g_vals = np.zeros(g_size, dtype=np.float64)

    # Compute convolution directly with numba optimization
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_size:
                g_vals[idx] += f_vals[i] * f_vals[j]

    return g_vals

@jit(nopython=True, fastmath=True)
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

@jit(nopython=True, fastmath=True)
def compute_trapezoidal_norms_numba(g_vals, dx):
    """
    Compute trapezoidal norms efficiently with numba
    """
    n = len(g_vals)

    # For L2 norm squared using trapezoidal rule
    l2_norm_sq = 0.0
    if n >= 2:
        # Trapezoidal rule: sum of (y[i]^2 + y[i+1]^2)/2 * dx
        for i in range(n-1):
            l2_norm_sq += (g_vals[i] * g_vals[i] + g_vals[i+1] * g_vals[i+1]) * dx / 2.0
    elif n == 1:
        l2_norm_sq = g_vals[0] * g_vals[0] * dx

    # For L1 norm using trapezoidal rule (average of adjacent heights * dx)
    l1_norm = 0.0
    if n >= 2:
        for i in range(n-1):
            l1_norm += (abs(g_vals[i]) + abs(g_vals[i+1])) * dx / 2.0
    elif n == 1:
        l1_norm = abs(g_vals[0]) * dx

    # Infinity norm
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

    # Compute norms using numba optimized version with trapezoidal integration
    g_abs = np.abs(g_centered)

    # Compute norms using numba - using trapezoidal rules for more accurate integration
    norm_1, norm_2_sq, norm_inf = compute_trapezoidal_norms_numba(g_abs, dx)

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

def create_hybrid_individual():
    """Create an individual with hybrid initialization strategies"""
    n_steps = random.randint(100, 1000)
    
    # Choose initialization strategy with preference for promising approaches
    strategy = random.choice(['alternating', 'gaussian', 'geometric'])
    
    if strategy == 'alternating':
        # Create alternating high/low pattern for better initialization
        individual = []
        for i in range(n_steps):
            if i % 2 == 0:
                individual.append(random.uniform(0.7, 1.0))
            else:
                individual.append(random.uniform(0.1, 0.3))
                
    elif strategy == 'gaussian':
        # Create multi-scale Gaussian bumps for better initialization
        individual = np.zeros(n_steps)
        num_bumps = random.randint(3, 8)
        for _ in range(num_bumps):
            center = random.random()
            scale = random.uniform(0.05, 0.2)
            height = random.uniform(0.5, 1.5)
            x = np.linspace(0, 1, n_steps)
            gaussian = height * np.exp(-0.5 * ((x - center) / scale) ** 2)
            individual += gaussian
        individual = np.maximum(individual, 0.0)
        
    else:  # geometric
        # Create geometric pattern with decay
        individual = np.zeros(n_steps)
        for i in range(n_steps):
            pos = i / (n_steps - 1) if n_steps > 1 else 0.5
            # Geometric decay with some oscillation
            amplitude = 0.8 * (0.9 ** i)
            oscillation = 0.1 * np.sin(8 * np.pi * pos)
            value = max(0.0, amplitude + oscillation + 0.05)
            individual[i] = value
    
    # Add some random perturbations for diversity
    for i in range(len(individual)):
        if random.random() < 0.05:  # 5% chance to perturb
            individual[i] = max(0.0, individual[i] + random.gauss(0, 0.05))
    
    # Convert to list and wrap in Individual
    return creator.Individual(individual.tolist())

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

def construct_function() -> List[float]:
    """Function to construct step-function with high C₂ value using enhanced evolutionary optimization."""

    # Algorithm parameters
    INITIAL_POPSIZE = 40
    MAX_POPSIZE = 100
    NGEN = 250
    INITIAL_MUTPB = 0.35
    INITIAL_CXPB = 0.6
    TOURNAMENT_SIZE = 4
    TIME_LIMIT = 85  # seconds
    STAGNATION_LIMIT = 20

    start_time = time.time()

    # Define problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Use hybrid initialization
    toolbox.register("individual", create_hybrid_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_c2)
    toolbox.register("mate", tools.cxUniform, indpb=0.05)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

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
    
    # Track improvement for early stopping
    last_improvement_gen = 0
    improvement_threshold = 0.001

    for gen in range(NGEN):
        if time.time() - start_time > TIME_LIMIT:
            break

        generation_counter += 1

        # Adaptive mutation rate: decrease over generations
        adaptive_mutpb = INITIAL_MUTPB * (1.0 - gen / NGEN)
        adaptive_cxpb = INITIAL_CXPB * (1.0 - gen / NGEN)
        
        # Adaptive population size that increases in later generations
        adaptive_popsize = max(INITIAL_POPSIZE, 
                              int(INITIAL_POPSIZE * (1.0 + gen / NGEN * 0.5)))

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

        # Replace population with adaptive size
        pop[:] = offspring[:adaptive_popsize]

        # Track best solution
        best_in_gen = max(pop, key=lambda x: x.fitness.values[0])
        if best_in_gen.fitness.values[0] > best_fitness:
            best_fitness = best_in_gen.fitness.values[0]
            best_individual = list(best_in_gen)
            last_improvement_gen = gen
        elif gen - last_improvement_gen > STAGNATION_LIMIT:
            # Early stopping if no improvement for too many generations
            break

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