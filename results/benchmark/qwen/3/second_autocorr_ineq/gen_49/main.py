# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List
import numba
from numba import jit

@jit(nopython=True)
def compute_autoconvolution_fast(f_vals):
    """
    Fast numba-based autoconvolution computation for step functions
    This correctly computes the convolution assuming piecewise constant
    functions on equal intervals.
    """
    n = len(f_vals)
    # Result has length 2*n-1
    g = np.zeros(2*n - 1)

    # Manual computation of the convolution sum for step functions
    # Each term contributes to the convolution according to the overlap
    for i in range(n):
        for j in range(n):
            # In convolution, the value at index i+j comes from f[i] * f[j]
            g[i + j] += f_vals[i] * f_vals[j]

    return g

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """Compute the three norms needed for C2 calculation with improved accuracy"""
    # Create step function with appropriate spacing
    n = len(f_values)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Generate step positions [-1/4, 1/4] - these are the left edges of steps
    dx = 0.5 / n  # Width of each step

    # Convert to numpy array and ensure non-negative
    f_array = np.array(f_values, dtype=np.float64)
    f_array = np.maximum(f_array, 0.0)  # Clip negative values

    # Compute autoconvolution using fast numba implementation
    g = compute_autoconvolution_fast(f_array)

    # We need to properly scale the result
    # The convolution result represents the integral of the product
    # But we want the actual function values at the resulting grid points
    # For step functions, the result g[k] corresponds to the integral
    # of f(x)*f(k*dx-x) over all overlapping regions

    # Scale by step width for proper normalization
    g = g * dx

    # Compute norms
    g_abs = np.abs(g)

    # L2 norm squared (sum of squares times dx)
    g2_squared = np.sum(g_abs**2) * dx

    # L1 norm (sum of absolute values times dx)
    g1 = np.sum(g_abs) * dx

    # L-infinity norm
    g_inf = np.max(g_abs)

    return g2_squared, g1, g_inf

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function"""
    g2_squared, g1, g_inf = compute_autoconvolution_norms(f_values)

    # Avoid division by zero
    if g1 <= 1e-15 or g_inf <= 1e-15:
        return 0.0

    return g2_squared / (g1 * g_inf)

def gaussian_mutate(parent: List[float], strength: float = 0.3) -> List[float]:
    """Apply Gaussian perturbations to create offspring"""
    child = parent.copy()
    n = len(child)

    if n == 0:
        return child

    # Apply small Gaussian noise to each element
    for i in range(n):
        # Add Gaussian noise scaled by strength and current value
        noise = np.random.normal(0, strength * max(1e-6, child[i]))
        child[i] = max(0, child[i] + noise)  # Keep non-negative

    return child

def adaptive_crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Create offspring through adaptive crossover"""
    n1, n2 = len(parent1), len(parent2)
    n = max(n1, n2)

    # Determine crossover points
    point1 = random.randint(1, min(n1, n)) if n1 > 0 else 0
    point2 = random.randint(1, min(n2, n)) if n2 > 0 else 0

    # Alternate between parents based on some criteria
    # Here we use a simple alternating strategy with probability
    offspring = []

    for i in range(n):
        if i < point1 and i < n1:
            offspring.append(parent1[i])
        elif i < point2 and i < n2:
            offspring.append(parent2[i])
        else:
            # Randomly choose from either parent or average
            if random.random() < 0.7:  # 70% chance to pick from parents
                choices = []
                if i < n1:
                    choices.append(parent1[i])
                if i < n2:
                    choices.append(parent2[i])

                if choices:
                    offspring.append(random.choice(choices))
                else:
                    offspring.append(0.0)
            else:
                # Average the two values if available
                val1 = parent1[i] if i < n1 else 0
                val2 = parent2[i] if i < n2 else 0
                offspring.append((val1 + val2) / 2.0)

    return offspring

def initialize_population(pop_size: int, min_length: int = 100, max_length: int = 1000) -> List[List[float]]:
    """Initialize population with diverse solutions"""
    population = []

    # Start with some well-known good configurations
    # Basic uniform step function
    uniform_length = random.randint(min_length, max_length)
    uniform = [1.0] * uniform_length
    population.append(uniform)

    # Random step functions
    for _ in range(pop_size - 1):
        length = random.randint(min_length, max_length)
        # Start with a few peaks
        individual = [random.random() for _ in range(length)]
        population.append(individual)

    return population

def select_parents(population: List[List[float]], fitnesses: List[float], tournament_size: int = 3) -> List[List[float]]:
    """Tournament selection for parent selection"""
    selected = []
    for _ in range(len(population)):
        # Tournament selection
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_index].copy())

    return selected

def optimize_step_function() -> List[float]:
    """Main optimization routine using genetic algorithm approach"""
    # Parameters
    pop_size = 20
    generations = 50
    elite_size = 2

    # Initialize population
    population = initialize_population(pop_size)

    best_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [calculate_c2(individual) for individual in population]

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Print progress every 10 generations
        if gen % 10 == 0:
            print(f"Generation {gen}: Best C2 = {best_fitness:.6f}")

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]

        # Keep elite
        elite = population[:elite_size]

        # Select parents
        parents = select_parents(population, fitnesses)

        # Create new population through crossover and mutation
        new_population = elite.copy()

        # Adaptive mutation rate - decreases over generations
        # Start with higher mutation rate for exploration, then decrease for exploitation
        mutation_rate = 0.3 * (1 - gen / generations) + 0.05

        while len(new_population) < pop_size:
            # Select two parents
            p1_idx, p2_idx = random.sample(range(len(parents)), 2)
            p1, p2 = parents[p1_idx], parents[p2_idx]

            # Crossover
            offspring = adaptive_crossover(p1, p2)

            # Mutation with adaptive rate
            mutated_offspring = gaussian_mutate(offspring, mutation_rate)

            new_population.append(mutated_offspring)

        population = new_population[:pop_size]

    # Final evaluation
    final_fitnesses = [calculate_c2(individual) for individual in population]
    best_individual = population[np.argmax(final_fitnesses)]

    return best_individual

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Use more sophisticated optimization approach
    try:
        # Set seed for reproducibility
        np.random.seed(42)
        random.seed(42)

        # Run optimization
        best_solution = optimize_step_function()

        return best_solution
    except Exception as e:
        # Fallback to basic approach if optimization fails
        print(f"Optimization failed with error: {e}, using fallback")
        return [1.0] * 500  # Return simple uniform function as fallback

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")