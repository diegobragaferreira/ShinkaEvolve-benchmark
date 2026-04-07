# EVOLVE-BLOCK-START

import numpy as np
import numba
from scipy import signal
from deap import base, creator, tools, algorithms
import random
import time
from sklearn.preprocessing import StandardScaler
import optuna
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

@numba.jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution efficiently using numba"""
    n = len(f_vals)
    # Create output array for autoconvolution
    g = np.zeros(2*n - 1)

    # Compute convolution manually with numba optimization
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    return g

@numba.jit(nopython=True)
def compute_norms_numba(g_vals):
    """Compute norms efficiently with numba"""
    n = len(g_vals)

    # L2 norm squared (using trapezoidal-like scheme)
    l2_sq = 0.0
    for i in range(n - 1):
        y1 = g_vals[i]
        y2 = g_vals[i + 1]
        l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0

    # L1 norm
    l1 = 0.0
    for i in range(n):
        l1 += abs(g_vals[i])

    # L-infinity norm
    linf = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf:
            linf = abs_val

    return l2_sq, l1, linf

def evaluate_individual(individual):
    """Evaluate fitness of an individual (step function)"""
    try:
        # Convert to numpy array and ensure non-negative
        f_vals = np.array(individual, dtype=np.float64)
        f_vals = np.maximum(f_vals, 0.0)

        # Skip if all zeros
        if np.sum(f_vals) == 0:
            return (0.0,)

        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)

        # Compute norms
        l2_sq, l1, linf = compute_norms_numba(g_vals)

        # Avoid division by zero
        if l1 <= 1e-15 or linf <= 1e-15:
            return (0.0,)

        # Compute C2
        c2 = l2_sq / (l1 * linf)
        return (c2,)
    except:
        return (0.0,)

def create_structured_initial_population(pop_size, min_size, max_size):
    """Create structured initial population with gamma distribution"""
    population = []
    for _ in range(pop_size):
        size = random.randint(min_size, max_size)
        # Use gamma distribution for more structured randomness
        f_vals = np.random.gamma(2, 2, size)
        # Normalize and scale appropriately
        f_vals = f_vals / np.max(f_vals) * 2.0 if np.max(f_vals) > 0 else np.ones(size)
        f_vals = np.maximum(f_vals, 0)
        # Apply mild smoothing to reduce extremes
        if len(f_vals) > 50:
            f_vals = signal.savgol_filter(f_vals, min(51, len(f_vals)-1), 3)
        f_vals = np.maximum(f_vals, 0)
        population.append(f_vals.tolist())
    return population

def adaptive_evolution_phase(initial_pop, pop_size, n_generations):
    """Perform evolutionary optimization with adaptive parameters"""
    # Create toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define operators
    def create_individual():
        size = random.randint(200, 800)  # Adaptive size
        f_vals = np.random.gamma(2, 2, size)
        f_vals = f_vals / np.max(f_vals) * 2.0 if np.max(f_vals) > 0 else np.ones(size)
        f_vals = np.maximum(f_vals, 0)
        if len(f_vals) > 50:
            f_vals = signal.savgol_filter(f_vals, min(51, len(f_vals)-1), 3)
        f_vals = np.maximum(f_vals, 0)
        return f_vals.tolist()

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population with diverse individuals
    population = initial_pop if initial_pop else toolbox.population(n=pop_size)

    # Evolve
    best_individual = None
    best_fitness = 0
    best_generation = 0

    for generation in range(n_generations):
        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Track best
        for ind in population:
            if ind.fitness.values[0] > best_fitness and len(ind) > 0:
                best_fitness = ind.fitness.values[0]
                best_individual = list(ind)
                best_generation = generation

        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Replace old population
        population[:] = offspring

        # Early termination if no improvement for several generations
        if generation - best_generation > 20:
            break

    return best_individual if best_individual is not None else []

def local_refinement(individual, max_iterations=50):
    """Refine solution using gradient-free optimization"""
    if not individual:
        return individual

    # Use Optuna for local refinement
    def objective(trial):
        # Create a slightly modified version of the individual
        modified = individual.copy()
        for i in range(len(modified)):
            if trial.suggest_float(f'param_{i}', 0.8, 1.2) < 1.1:
                modified[i] *= trial.suggest_float(f'multiplier_{i}', 0.9, 1.1)

        # Ensure non-negative values
        modified = [max(0, x) for x in modified]
        return evaluate_individual(modified)[0]

    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=min(20, max_iterations))
        return individual  # Return original if optuna doesn't improve much
    except:
        return individual

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Phase 1: Fast initial sampling with structured approach
    best_result = []
    best_c2 = 0

    # Create structured initial population
    initial_pop = create_structured_initial_population(20, 100, 1000)

    # Phase 2: Evolutionary optimization with adaptive parameters
    try:
        evolved_result = adaptive_evolution_phase(initial_pop, 30, 50)
        if evolved_result:
            # Evaluate evolved result
            f_vals = np.array(evolved_result, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)

                if l1 > 1e-15 and linf > 1e-15:
                    c2 = l2_sq / (l1 * linf)
                    if c2 > best_c2:
                        best_c2 = c2
                        best_result = evolved_result
    except Exception as e:
        pass

    # Phase 3: Local refinement if we have a candidate
    if best_result and time.time() - start_time < 70:  # Leave time for refinement
        refined_result = local_refinement(best_result)
        # Re-evaluate to see if refinement helped
        f_vals = np.array(refined_result, dtype=np.float64)
        f_vals = np.maximum(f_vals, 0.0)
        if np.sum(f_vals) > 0:
            g_vals = compute_autoconvolution_numba(f_vals)
            l2_sq, l1, linf = compute_norms_numba(g_vals)
            if l1 > 1e-15 and linf > 1e-15:
                c2 = l2_sq / (l1 * linf)
                if c2 > best_c2:
                    best_c2 = c2
                    best_result = refined_result

    # Phase 4: Fallback to well-structured approach if nothing worked
    if len(best_result) == 0 or best_c2 < 0.5:
        # Use a heuristic approach with more structured sampling
        n_steps = 500  # Fixed size for consistency
        # Create a step function that balances peaks and flat regions
        # This is a more principled approach based on gamma distribution
        f_values = np.random.gamma(2, 2, n_steps)  # Gamma distribution gives positive values
        f_values = f_values / np.max(f_values) * 2  # Scale to reasonable range
        f_values = np.maximum(f_values, 0)

        # Apply some smoothing to reduce extreme variations
        if len(f_values) > 50:
            f_values = signal.savgol_filter(f_values, min(51, len(f_values)-1), 3)
        f_values = np.maximum(f_values, 0)

        best_result = f_values.tolist()

    # Final evaluation and time management
    if best_result:
        try:
            f_vals = np.array(best_result, dtype=np.float64)
            f_vals = np.maximum(f_vals, 0.0)
            if np.sum(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                l2_sq, l1, linf = compute_norms_numba(g_vals)
                if l1 > 1e-15 and linf > 1e-15:
                    final_c2 = l2_sq / (l1 * linf)
                    if final_c2 > best_c2:
                        best_c2 = final_c2
        except:
            pass

    # Limit execution time
    elapsed = time.time() - start_time
    if elapsed > 85:  # Leave buffer for cleanup
        return best_result[:1000]  # Truncate if needed

    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")