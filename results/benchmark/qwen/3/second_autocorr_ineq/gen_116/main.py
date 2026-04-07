# EVOLVE-BLOCK-START

import numpy as np
from numba import jit
import time
from joblib import Parallel, delayed
import random
from deap import base, creator, tools, algorithms
import copy
from scipy import signal

# Core computation module with FFT optimization for better performance
@jit(nopython=True)
def compute_convolution_norms(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    Uses FFT-based convolution for improved performance with large arrays.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # For small arrays, use direct computation; for large arrays, use FFT
    if n_steps < 1000:
        # Direct convolution for smaller arrays
        g_size = 2 * n_steps - 1
        g = np.zeros(g_size)

        # Compute autoconvolution using direct convolution sum
        for i in range(n_steps):
            for j in range(n_steps):
                k = i + j
                if 0 <= k < g_size:
                    g[k] += f_values[i] * f_values[j] * dx

        # Compute norms using piecewise linear integration approach
        g2_sq = 0.0
        for i in range(len(g)-1):
            g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

        # ||g||₁ = sum(|g_i| * dx)
        g1 = np.sum(np.abs(g)) * dx

        # ||g||∞ = max(|g_i|)
        ginf = np.max(np.abs(g))

    else:
        # FFT-based convolution for larger arrays
        # Pad array to power of 2 for efficiency
        padded_length = 2 ** int(np.ceil(np.log2(2 * n_steps - 1)))
        f_padded = np.pad(f_values, (0, padded_length - n_steps), mode='constant')

        # Compute autoconvolution using FFT
        g_fft = np.fft.fft(f_padded) * np.fft.fft(f_padded).conj()
        g = np.fft.ifft(g_fft).real[:2*n_steps-1]

        # Scale by dx for proper normalization
        g = g * dx

        # Compute norms using piecewise linear integration approach
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

def evolutionary_optimization(n_steps, pop_size=80, generations=60):
    """Perform enhanced evolutionary optimization"""
    # Create toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: np.random.uniform(0, 1), n_steps)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operators with improved parameters
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.3)  # Lower crossover rate for more stable inheritance
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=5)  # Higher tournament size for stronger selection

    # Create initial population with diversified initialization
    pop = initialize_population(pop_size, n_steps, 'pattern')

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)

    # Evolution loop with elitism
    elite_size = max(1, pop_size // 10)  # Keep top 10% as elite

    for gen in range(generations):
        # Elitism: Keep best individuals
        elites = tools.selBest(pop, elite_size)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop) - elite_size)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.4:  # Reduced crossover rate
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.15:  # Increased mutation rate for exploration
                toolbox.mutate(mutant, gen_num=gen)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        # Replace the old population with the new one, keeping elites
        pop = elites + offspring

    # Return best individual
    best_ind = tools.selBest(pop, 1)[0]
    return np.array(best_ind)

# Improved initialization function
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

            # Generate initial function using pattern-based approach
            if res <= 300:
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

    # Local refinement using adaptive coordinate-wise improvements
    refined_individual = best_individual.copy()
    old_c2 = best_c2

    # Tabu list to prevent cycling
    tabu_list = []
    tabu_size = min(20, len(refined_individual) // 4)

    # Refinement loop with adaptive strategy
    improvement_count = 0
    stagnation_count = 0
    max_stagnation = 10

    for coord_iter in range(50):  # Increased iterations for better refinement
        if time.time() - start_time > max_time:
            break

        improved = False
        # Determine adaptive step size based on progress
        adaptive_step = max(0.001, 0.1 * (1.0 - improvement_count / 50.0))

        # Sample indices to avoid checking all elements every iteration
        sample_indices = np.random.choice(len(refined_individual),
                                         min(15, len(refined_individual) // 3),
                                         replace=False)

        for i in sample_indices:
            if time.time() - start_time > max_time:
                break

            # Skip if in tabu list
            if (i, refined_individual[i]) in tabu_list:
                continue

            # Try various perturbation sizes
            step_sizes = [adaptive_step, adaptive_step * 2, adaptive_step * 5]

            for step in step_sizes:
                # Try both directions
                for direction in [1, -1]:
                    if time.time() - start_time > max_time:
                        break

                    test_individual = refined_individual.copy()
                    new_val = refined_individual[i] + direction * step

                    # If we're reducing, make sure it doesn't go below 0
                    if direction == -1 and new_val < 0:
                        continue

                    test_individual[i] = new_val if new_val >= 0 else 0

                    new_c2 = compute_c2(test_individual.tolist())
                    if new_c2 > old_c2:
                        refined_individual = test_individual
                        old_c2 = new_c2
                        improved = True
                        improvement_count += 1
                        stagnation_count = 0

                        # Add to tabu list
                        tabu_list.append((i, refined_individual[i]))
                        if len(tabu_list) > tabu_size:
                            tabu_list.pop(0)

                        break  # Break after finding an improvement
                if improved:
                    break  # Break after finding an improvement

        if not improved:
            stagnation_count += 1
            if stagnation_count >= max_stagnation:
                break  # Stop if no improvement for a while

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