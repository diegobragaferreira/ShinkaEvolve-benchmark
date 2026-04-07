# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List
import numba
from numba import jit
from scipy.stats import qmc

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

def construct_multiscale_gaussian_pattern(n: int) -> List[float]:
    """
    Construct a step function using multi-scale Gaussian patterns.
    Creates a hierarchy of Gaussian bumps with varying scales and amplitudes.
    """
    # Create base pattern with multiple Gaussian components
    pattern = np.zeros(n)

    # Define multiple scales for the Gaussian bumps
    scales = [n//8, n//16, n//32, n//64]  # Different scale levels
    scales = [s for s in scales if s >= 2]  # Filter out too small scales

    # Create Gaussian bumps at different locations and scales
    for i, scale in enumerate(scales):
        # Position the bump in the middle of the function
        center = n // 2 + (i - len(scales)//2) * n // 8
        center = max(scale, min(n - scale, center))  # Bound to valid range

        # Create Gaussian with decreasing amplitude for smaller scales
        amplitude = 1.0 / (i + 1)  # Smaller scales have lower amplitude

        # Generate Gaussian curve
        x = np.arange(n)
        gaussian = amplitude * np.exp(-0.5 * ((x - center) / scale)**2)

        # Add to pattern
        pattern += gaussian

    # Ensure non-negativity and normalize
    pattern = np.maximum(pattern, 0)

    # Add some random variation to avoid being too deterministic
    noise_factor = 0.1
    noise = np.random.normal(0, noise_factor * np.std(pattern), n)
    pattern = np.maximum(pattern + noise, 0)

    # Convert to list and return
    return pattern.tolist()

def initialize_population(pop_size: int, min_length: int = 100, max_length: int = 1000) -> List[List[float]]:
    """Initialize population with diverse solutions using multi-scale patterns"""
    population = []

    # Start with structured multi-scale Gaussian patterns
    for i in range(pop_size):
        # Determine length for this individual
        length = random.randint(min_length, max_length)

        # Create multi-scale Gaussian pattern
        individual = construct_multiscale_gaussian_pattern(length)
        population.append(individual)

    return population

def adaptive_gaussian_mutate(parent: List[float], generation: int, max_generations: int) -> List[float]:
    """Apply adaptive Gaussian perturbations to create offspring"""
    child = parent.copy()
    n = len(child)

    if n == 0:
        return child

    # Adaptive mutation rate that decreases over generations
    initial_mutation_rate = 0.3
    final_mutation_rate = 0.05
    mutation_rate = initial_mutation_rate - (initial_mutation_rate - final_mutation_rate) * (generation / max_generations)
    mutation_rate = max(final_mutation_rate, mutation_rate)

    # Apply Gaussian noise to each element
    for i in range(n):
        # Add Gaussian noise scaled by adaptive rate and current value
        noise = np.random.normal(0, mutation_rate * max(1e-6, child[i]))
        child[i] = max(0, child[i] + noise)  # Keep non-negative

    return child

def local_search_refinement(solution: List[float], max_evals: int = 100,
                          use_differential_evolution: bool = True) -> List[float]:
    """Apply local search refinement to improve a solution"""
    try:
        if not use_differential_evolution:
            # Enhanced coordinate-wise refinement with adaptive step sizes
            solution_array = np.array(solution)
            best_solution = solution_array.copy()
            best_c2 = calculate_c2(best_solution.tolist())

            # Keep track of previous improvements for adaptive step sizing
            improvement_history = []
            max_history = 10

            for iteration in range(max_evals):
                # Adaptive step size calculation based on recent improvements
                current_step_size = 0.05  # Base step size

                if len(improvement_history) > 2:
                    # Calculate average improvement rate
                    avg_improvement = np.mean(improvement_history[-3:])
                    if avg_improvement > 1e-8:  # If there's consistent improvement
                        current_step_size = min(0.2, current_step_size * 1.1)  # Increase step size
                    else:
                        current_step_size = max(1e-4, current_step_size * 0.9)  # Decrease step size

                # Test perturbations for each dimension with adaptive step size
                test_solution = best_solution.copy()
                idx = random.randint(0, len(test_solution) - 1)

                # Compute finite difference gradient estimate
                epsilon = current_step_size * max(1e-6, best_solution[idx])
                test_solution[idx] = max(0, best_solution[idx] + epsilon)

                test_c2 = calculate_c2(test_solution.tolist())

                # If improvement, accept and record
                if test_c2 > best_c2:
                    best_c2 = test_c2
                    best_solution = test_solution.copy()
                    improvement_history.append(test_c2 - best_c2)
                else:
                    # Record negative improvement for step size adjustment
                    improvement_history.append(0.0)

                # Keep history bounded
                if len(improvement_history) > max_history:
                    improvement_history.pop(0)

            return best_solution.tolist()
        else:
            # Use differential evolution for local refinement
            # Convert to numpy array
            solution_array = np.array(solution)

            # Define bounds for each parameter
            bounds = [(0.0, 3.0) for _ in range(len(solution_array))]

            # Objective function for local search
            def objective(x):
                return -calculate_c2(x.tolist())  # Negative because we minimize

            # Use differential evolution for local refinement
            result = differential_evolution(objective, bounds, maxiter=max_evals//10,
                                          popsize=10, seed=42, disp=False)

            # Return refined solution with non-negative values
            refined = np.maximum(result.x, 0)
            return refined.tolist()

    except:
        # If local search fails, return original solution
        return solution

def adaptive_crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Create offspring through crossover with adaptive strategy"""
    n1, n2 = len(parent1), len(parent2)
    n = max(n1, n2)

    # Use a more sophisticated crossover approach
    offspring = []
    crossover_points = [random.randint(0, n) for _ in range(3)]  # Multiple crossover points
    crossover_points.sort()

    # Select segments from parents based on crossover points
    current_parent = random.choice([0, 1])  # 0 for parent1, 1 for parent2
    segment_start = 0

    for i in range(n):
        # Check if we should switch parents based on crossover points
        while segment_start < len(crossover_points) and i >= crossover_points[segment_start]:
            current_parent = 1 - current_parent  # Switch parent
            segment_start += 1

        # Choose value from appropriate parent
        if current_parent == 0 and i < n1:
            offspring.append(parent1[i])
        elif current_parent == 1 and i < n2:
            offspring.append(parent2[i])
        else:
            # Fill remaining with random values or from the other parent
            # Random choice with preference for the existing parent if possible
            if i < n1:
                offspring.append(parent1[i])
            elif i < n2:
                offspring.append(parent2[i])
            else:
                offspring.append(0.0)

    return offspring

def select_parents(population: List[List[float]], fitnesses: List[float],
                  tournament_size: int = 3) -> List[List[float]]:
    """Tournament selection for parent selection with bias towards better performers"""
    selected = []
    fitness_array = np.array(fitnesses)

    # Normalize fitness for selection pressure
    fitness_normalized = (fitness_array - np.min(fitness_array)) / (np.max(fitness_array) - np.min(fitness_array) + 1e-10)

    for _ in range(len(population)):
        # Tournament selection with fitness weighting
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitness_normalized[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_index].copy())

    return selected

def optimize_step_function() -> List[float]:
    """Main optimization routine with adaptive strategies"""
    # Parameters
    pop_size = 25
    generations = 60
    elite_size = 3
    early_stopping_patience = 10

    # Initialize population with advanced initialization
    population = initialize_population(pop_size)

    best_fitness_history = []
    best_fitness = -float('inf')
    best_individual = None
    no_improvement_count = 0

    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [calculate_c2(individual) for individual in population]

        # Track best fitness
        current_best = max(fitnesses)
        best_fitness_history.append(current_best)

        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[fitnesses.index(current_best)].copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Print progress every 10 generations
        if gen % 10 == 0:
            print(f"Generation {gen}: Best C2 = {best_fitness:.6f}")

        # Early stopping condition
        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at generation {gen} due to no improvement")
            break

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]

        # Keep elite
        elite = population[:elite_size]

        # Apply local search to top performers in later generations
        if gen >= generations * 0.7:  # Apply in final 30% of generations
            for i in range(min(elite_size, len(elite))):
                if i < len(elite):
                    refined = local_search_refinement(elite[i], max_evals=50)
                    refined_fitness = calculate_c2(refined)
                    if refined_fitness > calculate_c2(elite[i]):
                        elite[i] = refined

        # Select parents
        parents = select_parents(population, fitnesses)

        # Create new population through crossover and mutation
        new_population = elite.copy()

        while len(new_population) < pop_size:
            # Select two parents
            p1_idx, p2_idx = random.sample(range(len(parents)), 2)
            p1, p2 = parents[p1_idx], parents[p2_idx]

            # Crossover
            offspring = adaptive_crossover(p1, p2)

            # Mutation with adaptive rate
            mutated_offspring = adaptive_gaussian_mutate(offspring, gen, generations)

            new_population.append(mutated_offspring)

        population = new_population[:pop_size]

    # Final evaluation and cleanup
    final_fitnesses = [calculate_c2(individual) for individual in population]
    best_final_individual = population[np.argmax(final_fitnesses)]

    # Ensure we return the best individual found during optimization
    if best_individual is None:
        return best_final_individual
    else:
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