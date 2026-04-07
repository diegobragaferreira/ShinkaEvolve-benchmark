# EVOLVE-BLOCK-START

import numpy as np
from numba import njit, prange
import random
import time
import math
from joblib import Parallel, delayed

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

@njit
def compute_autoconvolution_norms_jit(f_values):
    """
    Fast JIT-compiled version for computing autoconvolution norms.
    Uses piecewise linear integration for L2 norm.
    """
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Compute autoconvolution manually for efficiency
    g = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            g[i + j] += f_values[i] * f_values[j]

    # Compute norms
    # L2 norm squared using piecewise linear integration: (h/3)(y1² + y1y2 + y2²)
    l2_norm_squared = 0.0
    if len(g) >= 2:
        h = 1.0  # Normalized step size
        for i in range(len(g) - 1):
            y1 = g[i]
            y2 = g[i+1]
            l2_norm_squared += (h/3.0) * (y1*y1 + y1*y2 + y2*y2)

    # L1 norm (normalized by number of intervals)
    l1_norm = np.sum(np.abs(g)) / (len(g) + 1)

    # L-infinity norm
    l_inf_norm = np.max(np.abs(g))

    return l2_norm_squared, l1_norm, l_inf_norm

@njit
def calculate_c2_jit(l2_norm_squared, l1_norm, l_inf_norm):
    """Fast JIT-compiled C2 calculation"""
    if l1_norm <= 1e-15 or l_inf_norm <= 1e-15:
        return 0.0
    return l2_norm_squared / (l1_norm * l_inf_norm)

def compute_autoconvolution_norms(f_values):
    """Compute the three norms needed for C2 calculation"""
    try:
        l2, l1, l_inf = compute_autoconvolution_norms_jit(f_values)
        return l2, l1, l_inf
    except Exception:
        return 0.0, 0.0, 0.0

def calculate_c2(f_values):
    """Calculate C₂ from step function values"""
    try:
        norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        return calculate_c2_jit(norm_2_squared, norm_1, norm_inf)
    except:
        return 0.0

def generate_geometric_initial_function(length):
    """Generate structured initial function with geometric patterns"""
    # Create multi-scale Gaussian bumps for better structure
    individual = np.zeros(length)

    # Add several Gaussian bumps at different positions and scales
    for _ in range(min(10, length//5)):
        center = np.random.randint(0, length)
        width = np.random.randint(1, max(3, length//20))
        height = np.random.uniform(0.5, 1.5)

        # Fill in the Gaussian shape
        for i in range(length):
            distance = abs(i - center)
            if distance < width * 3:  # Only fill within reasonable range
                gaussian_val = height * math.exp(-0.5 * (distance / width)**2)
                individual[i] += gaussian_val

    # Apply some randomization to avoid perfect symmetry
    for i in range(length):
        if np.random.random() < 0.5:
            individual[i] = max(0.0, individual[i] + np.random.normal(0, 0.1))

    return individual.tolist()

def generate_structured_initial_function(length):
    """Generate more sophisticated structured initial function"""
    individual = []

    # Create pattern with alternating high/medium/low regions
    pattern_types = ['high', 'medium', 'low', 'very_low']

    for i in range(length):
        pattern_type = pattern_types[i % len(pattern_types)]
        if pattern_type == 'high':
            individual.append(np.random.uniform(0.8, 1.2))
        elif pattern_type == 'medium':
            individual.append(np.random.uniform(0.4, 0.8))
        elif pattern_type == 'low':
            individual.append(np.random.uniform(0.1, 0.4))
        else:  # very_low
            individual.append(np.random.uniform(0.0, 0.2))

    # Add structured noise
    for i in range(len(individual)):
        if np.random.random() < 0.3:
            individual[i] = max(0.0, individual[i] + np.random.normal(0, 0.1))

    return individual

def initialize_population(pop_size, min_steps, max_steps):
    """Initialize population with diverse and structured functions"""
    population = []
    for _ in range(pop_size):
        # Random number of steps
        n_steps = np.random.randint(min_steps, max_steps)

        # Alternate between different initialization strategies
        if np.random.random() < 0.6:
            # Use geometric initialization
            individual = generate_geometric_initial_function(n_steps)
        else:
            # Use structured initialization
            individual = generate_structured_initial_function(n_steps)

        population.append(individual)

    return population

def crossover(parent1, parent2):
    """Perform uniform crossover between two parents"""
    # Ensure both parents have same length by truncating to shorter
    min_len = min(len(parent1), len(parent2))
    parent1 = parent1[:min_len]
    parent2 = parent2[:min_len]

    # Uniform crossover
    child1, child2 = [], []
    for i in range(min_len):
        if np.random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])

    # Handle extra elements by appending
    if len(parent1) > min_len:
        child1.extend(parent1[min_len:])
    elif len(parent2) > min_len:
        child1.extend(parent2[min_len:])

    if len(parent2) > min_len:
        child2.extend(parent2[min_len:])
    elif len(parent1) > min_len:
        child2.extend(parent1[min_len:])

    return child1, child2

def mutate(individual, generation, max_generations):
    """Mutate individual with adaptive scaling"""
    mutated = individual.copy()

    # Adaptive mutation rate
    base_mutation_rate = 0.3
    decay_factor = 0.1
    mutation_rate = base_mutation_rate - (generation / max_generations) * (base_mutation_rate - decay_factor)

    # Mutate each element
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Adaptive noise scale based on current value
            current_value = mutated[i]
            noise_scale = 0.15 + (0.05 * (generation / max_generations))

            # Adjust scale based on value magnitude
            if current_value > 0:
                noise_scale = max(0.01, noise_scale * current_value)

            mutated[i] = max(0.0, mutated[i] + np.random.normal(0, noise_scale))

    return mutated

def evaluate_fitness_parallel(population):
    """Parallel fitness evaluation"""
    def safe_calculate_c2(ind):
        try:
            return calculate_c2(ind)
        except:
            return 0.0

    # Use joblib for parallel execution
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(safe_calculate_c2)(ind) for ind in population
    )
    return results

def select_parents(population, fitness_scores):
    """Tournament selection with variable tournament sizes"""
    selected = []

    # Use variable tournament sizes to balance exploration/exploitation
    for _ in range(len(population)):
        tournament_size = np.random.choice([3, 4, 5, 6], p=[0.2, 0.3, 0.3, 0.2])
        tournament_indices = np.random.choice(len(population), min(tournament_size, len(population)), replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_index].copy())

    return selected

def elitism(population, fitness_scores, elite_count):
    """Keep best individuals"""
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite = [population[i].copy() for i in sorted_indices[:elite_count]]
    return elite

def single_evolution(max_generations=60, pop_size=100, min_steps=500, max_steps=2000):
    """Run a single evolutionary process"""
    # Initialize population
    population = initialize_population(pop_size, min_steps, max_steps)

    best_individual = None
    best_fitness = -np.inf

    for generation in range(max_generations):
        # Evaluate fitness
        fitness_scores = evaluate_fitness_parallel(population)

        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()

        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best C2 = {best_fitness:.6f}")

        # Elitism - keep top performers
        elite = elitism(population, fitness_scores, pop_size // 10)

        # Selection
        parents = select_parents(population, fitness_scores)

        # Crossover and mutation
        new_population = elite.copy()
        while len(new_population) < pop_size:
            p1, p2 = np.random.choice(len(parents), 2, replace=False)
            child1, child2 = crossover(parents[p1], parents[p2])

            child1 = mutate(child1, generation, max_generations)
            child2 = mutate(child2, generation, max_generations)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:pop_size]

    return best_individual, best_fitness

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()

    # Multi-start approach: run several independent evolutions
    best_result = None
    best_score = -np.inf

    # Run multiple independent evolutions with different random seeds
    num_starts = 5
    for start in range(num_starts):
        # Set different seed for each start
        np.random.seed(42 + start)
        random.seed(42 + start)
        print(f"Starting evolution attempt {start + 1}/{num_starts}")

        try:
            result, score = single_evolution(
                max_generations=60,
                pop_size=100,
                min_steps=500,
                max_steps=2000
            )

            if score > best_score:
                best_score = score
                best_result = result

        except Exception as e:
            print(f"Evolution attempt {start + 1} failed: {e}")
            continue

    end_time = time.time()
    eval_time = end_time - start_time

    print(f"Evaluated in {eval_time:.2f} seconds")
    if best_result is not None:
        final_c2 = calculate_c2(best_result)
        print(f"Best C2 found: {final_c2:.6f}")
    else:
        print("No valid result found")
        best_result = [0.5]  # fallback

    return best_result if best_result is not None else [0.5]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")