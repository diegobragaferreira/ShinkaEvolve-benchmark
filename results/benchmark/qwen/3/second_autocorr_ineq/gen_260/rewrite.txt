# EVOLVE-BLOCK-START

import numpy as np
from deap import base, creator, tools, algorithms
import random
import time
from scipy import signal
from typing import List
from numba import jit

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
    """Create an individual with multi-scale Gaussian bumps with hierarchical structure"""
    n_steps = random.randint(200, 1000)

    # Create a base function with multiple hierarchical Gaussian bumps
    individual = np.zeros(n_steps)

    # Create multiple scales of Gaussian bumps - from large to small
    # This creates a more structured and potentially more effective pattern
    scales = [0.3, 0.2, 0.1, 0.05, 0.02]  # Different scale factors
    num_bumps_per_scale = [2, 3, 4, 6, 10]  # Number of bumps per scale

    for scale_idx, (scale, num_bumps) in enumerate(zip(scales, num_bumps_per_scale)):
        # Scale-specific heights (larger bumps for coarser scales)
        height_factor = 1.0 / (scale_idx + 1)  # Smaller scales get higher weights
        for _ in range(num_bumps):
            # Position is randomized but constrained to avoid extreme clustering
            center = random.uniform(scale, 1.0 - scale)
            height = random.uniform(0.5, 1.5) * height_factor
            x = np.linspace(0, 1, n_steps)
            gaussian = height * np.exp(-0.5 * ((x - center) / scale) ** 2)
            individual += gaussian

    # Ensure non-negative values and normalize somewhat
    individual = np.maximum(individual, 0.0)
    
    # Add some additional random structure to encourage exploration
    for i in range(len(individual)):
        if random.random() < 0.03:  # 3% chance to add noise
            individual[i] = max(0.0, individual[i] + random.gauss(0, 0.03))

    # Convert to list and wrap in Individual
    return creator.Individual(individual.tolist())

def create_geometric_individual():
    """Create a geometric pattern initialization"""
    n_steps = random.randint(200, 1000)
    
    # Create geometric pattern with oscillations
    individual = np.zeros(n_steps)
    for i in range(n_steps):
        pos = i / (n_steps - 1) if n_steps > 1 else 0.5
        # Geometric decay with multiple oscillation frequencies
        amplitude = 0.7 * (0.85 ** i)
        oscillation1 = 0.15 * np.sin(6 * np.pi * pos)
        oscillation2 = 0.05 * np.sin(12 * np.pi * pos)
        value = max(0.0, amplitude + oscillation1 + oscillation2 + 0.03)
        individual[i] = value
    
    # Add some random noise for diversity
    for i in range(len(individual)):
        if random.random() < 0.02:  # 2% chance to perturb
            individual[i] = max(0.0, individual[i] + random.gauss(0, 0.02))
            
    return creator.Individual(individual.tolist())

def create_uniform_individual():
    """Create a uniform distribution initialization"""
    n_steps = random.randint(200, 1000)
    individual = np.ones(n_steps) * random.uniform(0.7, 1.3)
    
    # Add some variation
    for i in range(len(individual)):
        if random.random() < 0.05:  # 5% chance to perturb
            individual[i] = max(0.0, individual[i] + random.gauss(0, 0.05))
            
    return creator.Individual(individual.tolist())

def adaptive_local_search(individual, max_iterations=30):
    """Apply enhanced adaptive local search to improve individual"""
    best_individual = individual.copy()
    best_c2 = evaluate_c2(best_individual)[0]
    
    # Try different perturbation strategies
    for iter_count in range(max_iterations):
        # Create candidate by perturbing randomly selected elements
        candidate = individual.copy()
        
        # Adjust perturbation strength based on iteration count
        perturbation_strength = max(0.01, 0.1 * (1.0 - iter_count / max_iterations))
        
        # Pick random elements to modify (more aggressive in early iterations)
        num_changes = min(max(1, len(individual) // 20), max(5, int(0.1 * len(individual))))
        indices = random.sample(range(len(candidate)), num_changes)
        
        for i in indices:
            # Apply adaptive change based on current value
            current_val = candidate[i]
            # Adaptive perturbation
            perturbation = random.gauss(0, perturbation_strength * current_val if current_val > 0 else perturbation_strength)
            candidate[i] = max(0.0, current_val + perturbation)
        
        # Evaluate candidate
        candidate_c2 = evaluate_c2(candidate)[0]
        
        if candidate_c2 > best_c2:
            best_c2 = candidate_c2
            best_individual = candidate.copy()
    
    return best_individual, best_c2

def multi_start_optimization(num_starts=10, max_generations=200):
    """Run multiple optimization starts with different strategies"""
    best_overall_fitness = 0.0
    best_overall_individual = None
    
    # Strategy weights for different initialization approaches
    strategies = [
        ("multiscale_gaussian", 0.4),
        ("geometric", 0.3),
        ("uniform", 0.3)
    ]
    
    # Run multiple optimization runs
    for start_idx in range(num_starts):
        # Select initialization strategy based on weights
        strategy_name, _ = random.choices(strategies, weights=[w for _, w in strategies])[0]
        
        # Create toolbox for this run
        local_toolbox = base.Toolbox()
        
        # Register appropriate individual creation function
        if strategy_name == "multiscale_gaussian":
            local_toolbox.register("individual", create_multiscale_gaussian_individual)
        elif strategy_name == "geometric":
            local_toolbox.register("individual", create_geometric_individual)
        else:  # uniform
            local_toolbox.register("individual", create_uniform_individual)
            
        local_toolbox.register("population", tools.initRepeat, list, local_toolbox.individual)
        local_toolbox.register("evaluate", evaluate_c2)
        local_toolbox.register("mate", tools.cxUniform, indpb=0.05)
        local_toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        local_toolbox.register("select", tools.selTournament, tournsize=4)

        # Initialize population with adaptive size
        pop_size = max(20, 30 + start_idx * 2)  # Gradually increase population size
        pop = local_toolbox.population(n=pop_size)

        # Evaluate initial population
        fitnesses = list(map(local_toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Main evolution loop with adaptive parameters
        best_fitness = 0.0
        best_individual = None
        stagnation_count = 0
        prev_best_fitness = 0.0

        for gen in range(max_generations):
            # Adaptive parameters that change over generations
            adaptive_mutpb = 0.35 * (1.0 - gen / max_generations)
            adaptive_cxpb = 0.6 * (1.0 - gen / max_generations)
            
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

            # Apply local search to top individuals
            sorted_pop = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)
            top_count = max(1, len(pop) // 5)
            top_individuals = sorted_pop[:top_count]

            # Apply local search to top individuals
            for i, ind in enumerate(top_individuals):
                if random.random() < 0.5:  # Apply local search to 50% of top individuals
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

            # Early stopping if no improvement for too many generations
            if stagnation_count >= 15:
                break

        # Final evaluation of best individual from this run
        if best_individual is not None:
            final_fitness = evaluate_c2(best_individual)[0]
            if final_fitness > 0.0 and final_fitness > best_overall_fitness:
                best_overall_fitness = final_fitness
                best_overall_individual = list(best_individual)

    return best_overall_individual, best_overall_fitness

def construct_function() -> List[float]:
    """Function to construct step-function with high C₂ value using enhanced evolutionary optimization."""
    
    # Algorithm parameters
    TIME_LIMIT = 85  # seconds
    INITIAL_POPSIZE = 30
    NGEN = 200
    INITIAL_MUTPB = 0.35
    INITIAL_CXPB = 0.6
    TOURNAMENT_SIZE = 4

    start_time = time.time()

    # Define problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Multi-start approach for better exploration
    best_individual, best_fitness = multi_start_optimization(num_starts=12, max_generations=150)

    # If no good solution found, fall back to single optimization run
    if best_individual is None or best_fitness < 0.1:
        # Single optimization run with hybrid initialization
        toolbox = base.Toolbox()

        # Use hybrid initialization with preference for multiscale Gaussian
        def create_hybrid_individual():
            strategy = random.choice(['multiscale_gaussian', 'geometric', 'uniform'])
            if strategy == 'multiscale_gaussian':
                return create_multiscale_gaussian_individual()
            elif strategy == 'geometric':
                return create_geometric_individual()
            else:
                return create_uniform_individual()

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

        # Main evolution loop
        best_fitness = 0.0
        best_individual = None
        generation_counter = 0

        for gen in range(NGEN):
            if time.time() - start_time > TIME_LIMIT:
                break

            generation_counter += 1

            # Adaptive mutation rate: decrease over generations with more controlled decay
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
            sorted_pop = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)
            top_count = max(1, len(pop) // 8)
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

    # Final evaluation and return
    if best_individual is not None:
        final_fitness = evaluate_c2(best_individual)[0]
        if final_fitness > 0.0:
            # Return the best individual found
            return [max(0.0, float(x)) for x in best_individual]

    # Fallback: return reasonable random solution
    fallback_size = random.randint(200, 1000)
    return [abs(random.gauss(0.5, 0.3)) for _ in range(fallback_size)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")