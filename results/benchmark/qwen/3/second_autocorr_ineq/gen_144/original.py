# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
import multiprocessing as mp
from joblib import Parallel, delayed
import time
from numba import njit

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 3
MAX_STEPS = 10000
MIN_STEPS = 500
SEED = 42

# Set seed for reproducibility
np.random.seed(SEED)
random.seed(SEED)

@njit
def compute_autoconvolution_norms_numba(f_values, n_steps):
    """
    Compute the three norms needed for C2 calculation from step function values (JIT compiled).
    Returns ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array and normalize
    f = np.array(f_values, dtype=np.float64)

    # Create step function on [-1/4, 1/4]
    # We'll use 1000 points for accurate convolution
    x = np.linspace(-0.25, 0.25, 1000, dtype=np.float64)
    dx = x[1] - x[0]

    # Create step function with proper spacing
    step_width = 0.5 / n_steps
    step_positions = np.linspace(-0.25 + step_width/2, 0.25 - step_width/2, n_steps, dtype=np.float64)

    # Build piecewise constant function
    f_func = np.zeros_like(x, dtype=np.float64)
    for i in range(n_steps):
        pos = step_positions[i]
        height = f[i]
        left = pos - step_width/2
        right = pos + step_width/2
        mask = (x >= left) & (x <= right)
        f_func[mask] = height

    # Perform autoconvolution using manual implementation for numba compatibility
    # Create convolution result array
    g = np.zeros(len(x), dtype=np.float64)

    # Manual convolution (this will be JIT compiled)
    for i in range(len(x)):
        total = 0.0
        for j in range(len(x)):
            if i - j >= 0 and i - j < len(x):
                total += f_func[j] * f_func[i-j]
        g[i] = total

    # Adjust for proper scaling due to discretization
    g = g * dx

    # Compute the required norms
    g_squared = g**2
    g_abs = np.abs(g)

    # ||g||₂² (using trapezoidal rule for integration)
    norm_2_squared = 0.0
    for i in range(len(g)-1):
        norm_2_squared += (dx/3) * (g_squared[i] + g_squared[i+1] + g_squared[i]*g_squared[i+1])

    # ||g||₁ (L1 norm)
    norm_1 = 0.0
    for i in range(len(g)-1):
        norm_1 += (dx/2) * (g_abs[i] + g_abs[i+1])

    # ||g||∞ (infinity norm)
    norm_inf = 0.0
    for i in range(len(g)):
        if g_abs[i] > norm_inf:
            norm_inf = g_abs[i]

    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms(f_values):
    """
    Compute the three norms needed for C2 calculation from step function values.
    Returns ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array and normalize
    f = np.array(f_values)

    # Get the number of steps
    n_steps = len(f)

    # Call JIT compiled function
    norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms_numba(f_values, n_steps)

    return norm_2_squared, norm_1, norm_inf

def calculate_c2(f_values):
    """Calculate C₂ from step function values"""
    try:
        norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)

        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0

        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2
    except:
        return 0.0

def initialize_population(pop_size, min_steps, max_steps):
    """Initialize population with diverse step functions"""
    population = []
    for _ in range(pop_size):
        # Random number of steps
        n_steps = np.random.randint(min_steps, max_steps)
        # Random heights with some structure - using exponential distribution for variety
        heights = np.random.exponential(scale=1.0, size=n_steps)
        # Clip negative values
        heights = np.maximum(heights, 0)
        population.append(heights.tolist())
    return population

def crossover(parent1, parent2):
    """Perform crossover between two parents"""
    if len(parent1) != len(parent2):
        # Make them same length by truncating or padding
        min_len = min(len(parent1), len(parent2))
        parent1 = parent1[:min_len]
        parent2 = parent2[:min_len]

    if np.random.random() < CROSSOVER_RATE:
        # Uniform crossover
        child1, child2 = [], []
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2
    else:
        return parent1, parent2

def mutate(individual, mutation_rate):
    """Mutate individual with Gaussian noise"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1 * mutated[i] + 0.01)
            mutated[i] = max(0, mutated[i] + noise)
    return mutated

def evaluate_fitness(population):
    """Evaluate fitness of entire population"""
    results = Parallel(n_jobs=-1)(delayed(calculate_c2)(ind) for ind in population)
    return results

def select_parents(population, fitness_scores):
    """Tournament selection"""
    selected = []
    tournament_size = min(5, len(population) // 2)

    for _ in range(len(population)):
        # Tournament
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_index].copy())

    return selected

def elitism(population, fitness_scores, elite_count):
    """Keep best individuals"""
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite = [population[i].copy() for i in sorted_indices[:elite_count]]
    return elite

def adaptive_evolution():
    """Main evolutionary algorithm"""
    # Initialize population
    population = initialize_population(POPULATION_SIZE, MIN_STEPS, MAX_STEPS)

    best_individual = None
    best_fitness = -np.inf

    for generation in range(GENERATIONS):
        # Evaluate fitness
        fitness_scores = evaluate_fitness(population)

        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()

        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best C2 = {best_fitness:.4f}")

        # Elitism
        elite = elitism(population, fitness_scores, ELITISM_COUNT)

        # Selection
        parents = select_parents(population, fitness_scores)

        # Crossover and mutation
        new_population = elite.copy()
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = np.random.choice(len(parents), 2, replace=False)
            child1, child2 = crossover(parents[p1], parents[p2])

            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]

    return best_individual

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Run adaptive evolution
    result = adaptive_evolution()

    end_time = time.time()
    eval_time = end_time - start_time

    print(f"Evaluated in {eval_time:.2f} seconds")
    print(f"Best C2 found: {calculate_c2(result):.6f}")

    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")