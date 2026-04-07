# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist
import math
from numba import njit
from sklearn.cluster import KMeans

@njit
def compute_convolution_norms_numba(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    JIT compiled version for improved performance.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using piecewise linear integration
    # g[k] = integral f(x) * f(y) where x+y = k*dx - domain_length/2
    # We'll compute this as a discrete approximation

    # Create the full convolution grid with 2*n_steps-1 points
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Compute autoconvolution - JIT compiled loop
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                # Trapezoidal-like integration contribution
                # For two adjacent pieces with heights f[i] and f[j],
                # the convolution contributes to g[k]
                g[k] += f_values[i] * f_values[j] * dx

    # Compute norms
    # ||g||₂² = sum(g_i² * dx) - but we're doing piecewise linear integration
    # Using trapezoidal rule: ∫ g(x)² dx ≈ (dx/3)(g₀² + g₀g₁ + g₁²) + ...
    g2_sq = 0.0
    for i in range(len(g)-1):
        g2_sq += (dx/3) * (g[i]**2 + g[i]*g[i+1] + g[i+1]**2)

    # ||g||₁ = sum(|g_i| * dx)
    g1 = np.sum(np.abs(g)) * dx

    # ||g||∞ = max(|g_i|)
    ginf = np.max(np.abs(g))

    return g2_sq, g1, ginf

@njit
def compute_c2_numba(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞) - JIT compiled version"""
    g2_sq, g1, ginf = compute_convolution_norms_numba(f_values)

    if g1 == 0 or ginf == 0:
        return 0.0

    return g2_sq / (g1 * ginf)

def compute_convolution_norms(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    """
    return compute_convolution_norms_numba(f_values, domain_length)

def compute_c2(f_values):
    """Compute C₂ = ||g||₂² / (||g||₁ · ||g||∞)"""
    return compute_c2_numba(f_values)

def construct_geometric_initial_function(n_steps):
    """Construct a good initial function using geometric patterns"""
    # Create a function that encourages flat convolution
    # Use a combination of geometric and trigonometric patterns

    f_values = np.zeros(n_steps)

    # Base pattern: smooth transition with some oscillation
    x = np.linspace(0, 1, n_steps)

    # Create a multi-scale pattern that helps avoid extreme peaks
    scale1 = 0.3 * np.sin(2 * np.pi * x) + 0.7
    scale2 = 0.2 * np.sin(4 * np.pi * x) + 0.8
    scale3 = 0.1 * np.sin(8 * np.pi * x) + 0.9

    # Combine scales with decreasing influence
    combined = 0.5 * scale1 + 0.3 * scale2 + 0.2 * scale3

    # Normalize to reasonable values
    combined = np.clip(combined, 0, 1)

    # Add some structured variation to make it less regular
    # Create some bumps in the middle to help with convolution
    center = n_steps // 2
    width = n_steps // 4
    bump = np.exp(-((np.arange(n_steps) - center)**2) / (2 * (width/3)**2))
    bump = bump / np.max(bump) * 0.3
    combined = np.clip(combined + bump, 0, 1)

    # Add some controlled randomness
    noise = np.random.normal(0, 0.02, n_steps)
    combined = np.clip(combined + noise, 0, 1)

    return combined.tolist()

def adaptive_evolutionary_optimization():
    """Main optimization routine using adaptive evolutionary strategy"""
    n_steps = 1000  # Using larger number of steps for better resolution

    # Generate diverse initial population
    pop_size = 50
    population = []
    for _ in range(pop_size):
        # Mix of geometric and random patterns
        if np.random.random() < 0.7:
            individual = construct_geometric_initial_function(n_steps)
        else:
            individual = np.random.random(n_steps).tolist()

        # Ensure non-negativity and reasonable scaling
        individual = [max(0, x) for x in individual]
        # Normalize to avoid extremely large peaks
        total = sum(individual)
        if total > 0:
            individual = [x / total * 10 for x in individual]

        population.append(individual)

    # Track best solution
    best_individual = None
    best_c2 = -1
    generation = 0
    max_generations = 30  # Limit to keep within time constraints

    # Adaptive parameters
    mutation_rate = 0.3
    crossover_rate = 0.8
    elite_count = 5

    # Evolution loop
    for generation in range(max_generations):
        # Evaluate population
        fitness_scores = []
        for individual in population:
            c2 = compute_c2(individual)
            fitness_scores.append(c2)
            if c2 > best_c2:
                best_c2 = c2
                best_individual = individual.copy()

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Print progress every few generations
        if generation % 5 == 0:
            print(f"Generation {generation}: Best C2 = {best_c2:.6f}")

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        for i in range(elite_count):
            new_population.append(sorted_population[i].copy())

        # Generate offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection for parents
            parent1 = tournament_selection(sorted_population, sorted_fitness, 3)
            parent2 = tournament_selection(sorted_population, sorted_fitness, 3)

            # Crossover
            if np.random.random() < crossover_rate:
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            mutate_individual(child1, mutation_rate)
            mutate_individual(child2, mutation_rate)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:pop_size]

        # Adapt mutation rate: decrease over time to fine-tune
        mutation_rate = max(0.05, mutation_rate * 0.95)

        # Diversity check
        if generation > 5 and generation % 3 == 0:
            # Check if population has converged too much
            diversity = calculate_population_diversity(population)
            if diversity < 0.01:  # Low diversity
                # Introduce some random diversity
                for i in range(min(5, len(population))):
                    population[i] = construct_geometric_initial_function(n_steps)

    return best_individual

def tournament_selection(population, fitness_scores, tournament_size):
    """Select an individual using tournament selection"""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index].copy()

def uniform_crossover(parent1, parent2):
    """Perform uniform crossover between two parents"""
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

def mutate_individual(individual, mutation_rate):
    """Mutate an individual with given probability"""
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            # Gaussian mutation with adaptive strength
            mutation_strength = 0.1 * individual[i] if individual[i] > 0 else 0.1
            individual[i] = max(0, individual[i] + np.random.normal(0, mutation_strength))

def calculate_population_diversity(population):
    """Calculate diversity measure of population"""
    if len(population) < 2:
        return 0
    # Calculate pairwise distances between individuals
    vectors = np.array(population)
    distances = pdist(vectors, metric='euclidean')
    return np.mean(distances) / np.std(distances) if np.std(distances) > 0 else 0

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    np.random.seed(42)  # For reproducibility

    # Try adaptive evolutionary optimization first
    try:
        result = adaptive_evolutionary_optimization()
        if result is not None:
            return result
    except Exception as e:
        print(f"Evolutionary optimization failed: {e}")

    # Fall back to simpler approach if evolutionary optimization fails
    n_steps = 500

    # Use constructed function with geometric pattern
    f_values = construct_geometric_initial_function(n_steps)

    # Normalize
    total = sum(f_values)
    if total > 0:
        f_values = [x / total * 10 for x in f_values]

    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")