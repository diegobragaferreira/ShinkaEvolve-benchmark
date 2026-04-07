# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution
import random
from typing import List
import numba
from numba import jit
import time

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

@jit(nopython=True)
def compute_autoconvolution_norms_jit(f_vals) -> tuple:
    """JIT-compiled version of norm computation for speed"""
    n = len(f_vals)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Step width
    dx = 0.5 / n

    # Compute autoconvolution
    g = compute_autoconvolution_fast(f_vals)

    # Scale by step width
    g = g * dx

    # Compute norms
    g_abs = np.abs(g)
    
    # L2 norm squared
    g2_squared = np.sum(g_abs**2) * dx
    
    # L1 norm  
    g1 = np.sum(g_abs) * dx
    
    # L-infinity norm
    g_inf = np.max(g_abs)

    return g2_squared, g1, g_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """Compute the three norms needed for C2 calculation"""
    return compute_autoconvolution_norms_jit(np.array(f_values, dtype=np.float64))

def calculate_c2(f_values: List[float]) -> float:
    """Calculate C2 value for given step function"""
    g2_squared, g1, g_inf = compute_autoconvolution_norms(f_values)

    # Avoid division by zero
    if g1 <= 1e-15 or g_inf <= 1e-15:
        return 0.0

    return g2_squared / (g1 * g_inf)

def construct_adaptive_multiscale_pattern(n: int) -> List[float]:
    """
    Construct a step function using adaptive multi-scale patterns optimized for C2.
    Creates a hierarchy of Gaussian bumps with adaptive scaling, amplitude, and positioning.
    """
    pattern = np.zeros(n)
    
    # Use logarithmic scale for scales to ensure variety
    scale_factors = np.logspace(np.log10(n//16), np.log10(n//2), num=5, base=2, dtype=int)
    scale_factors = [s for s in scale_factors if s >= 2 and s <= n//2]

    # Create Gaussian bumps with adaptive characteristics
    for i, scale in enumerate(scale_factors):
        # Position more carefully to avoid edge effects
        center_offset = int((i - len(scale_factors)//2) * n // 12)
        center = n // 2 + center_offset
        center = max(scale, min(n - scale, center))  # Bound to valid range

        # Amplitude decreases with scale but increases with importance
        amplitude = 1.0 / (1 + i * 0.2)  # Prefer larger scales with lower amplitude

        # Generate Gaussian curve
        x = np.arange(n)
        gaussian = amplitude * np.exp(-0.5 * ((x - center) / scale)**2)

        # Add to pattern
        pattern += gaussian

    # Ensure non-negativity
    pattern = np.maximum(pattern, 0)

    # Apply additional smoothing to encourage flat convolution
    if n > 10:
        # Simple moving average smoothing
        window_size = max(3, n // 50)
        if window_size % 2 == 0:
            window_size += 1
        smoothed = np.convolve(pattern, np.ones(window_size)/window_size, mode='same')
        pattern = np.maximum(pattern, smoothed)

    # Add controlled noise to create diversity
    noise_level = 0.05 * np.std(pattern)
    noise = np.random.normal(0, noise_level, n)
    pattern = np.maximum(pattern + noise, 0)

    # Normalize to have reasonable magnitude
    if np.sum(pattern) > 0:
        pattern = pattern / np.sum(pattern) * 5.0

    return pattern.tolist()

def initialize_population(pop_size: int, min_length: int = 100, max_length: int = 1000) -> List[List[float]]:
    """Initialize population with diverse and structured solutions"""
    population = []

    # Start with structured multi-scale patterns
    for i in range(pop_size):
        # Vary length for diversity
        length = random.randint(min_length, max_length)
        
        # Create adaptive multiscale pattern
        individual = construct_adaptive_multiscale_pattern(length)
        population.append(individual)

    return population

def adaptive_gaussian_mutate(parent: List[float], generation: int, max_generations: int, 
                           fitness_history: List[float] = None) -> List[float]:
    """Apply adaptive Gaussian perturbations with enhanced control"""
    child = parent.copy()
    n = len(child)

    if n == 0:
        return child

    # Adaptive mutation rate that decreases over generations but adapts to convergence
    initial_mutation_rate = 0.4
    final_mutation_rate = 0.02
    
    # Adaptive based on convergence rate
    mutation_rate = initial_mutation_rate
    
    if len(fitness_history) >= 3:
        recent_changes = [fitness_history[i] - fitness_history[i-1] 
                         for i in range(1, min(4, len(fitness_history)))]
        avg_change = np.mean(recent_changes)
        if avg_change < 1e-6:  # Slow convergence, increase mutation
            mutation_rate = min(0.5, mutation_rate * 1.3)
        elif avg_change > 1e-3:  # Fast convergence, decrease mutation
            mutation_rate = max(0.01, mutation_rate * 0.8)
    
    mutation_rate = max(final_mutation_rate, mutation_rate - 
                       (initial_mutation_rate - final_mutation_rate) * (generation / max_generations))
    
    # Apply Gaussian noise with variance adapting to current value
    for i in range(n):
        if random.random() < 0.8:  # Only apply to 80% of dimensions
            noise_variance = mutation_rate * max(1e-6, child[i])
            noise = np.random.normal(0, noise_variance)
            child[i] = max(0, child[i] + noise)

    return child

def gradient_based_local_search(solution: List[float], max_evals: int = 100) -> List[float]:
    """Apply gradient-based local search using finite differences"""
    try:
        solution_array = np.array(solution, dtype=np.float64)
        best_solution = solution_array.copy()
        best_c2 = calculate_c2(best_solution.tolist())
        
        # Estimate gradient using finite differences
        epsilon = 1e-4
        step_size = 0.05
        
        for iteration in range(max_evals):
            gradients = np.zeros_like(best_solution)
            
            # Compute finite differences for each dimension
            for i in range(len(best_solution)):
                # Forward difference
                perturbed = best_solution.copy()
                perturbed[i] = max(0, best_solution[i] + epsilon)
                c2_plus = calculate_c2(perturbed.tolist())
                
                # Backward difference
                perturbed = best_solution.copy()
                perturbed[i] = max(0, best_solution[i] - epsilon)
                c2_minus = calculate_c2(perturbed.tolist())
                
                gradients[i] = (c2_plus - c2_minus) / (2 * epsilon)
            
            # Update using gradient ascent
            update = step_size * gradients
            
            # Apply update with adaptive step size
            new_solution = best_solution + update
            
            # Ensure non-negativity
            new_solution = np.maximum(new_solution, 0)
            
            new_c2 = calculate_c2(new_solution.tolist())
            
            if new_c2 > best_c2:
                best_c2 = new_c2
                best_solution = new_solution
            else:
                # Reduce step size if no improvement
                step_size *= 0.9
                
        return best_solution.tolist()
        
    except Exception:
        return solution

def adaptive_crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Create offspring through adaptive crossover with improved recombination"""
    n1, n2 = len(parent1), len(parent2)
    n = max(n1, n2)

    # Use uniform crossover with weighted preference for better parent
    offspring = []
    
    # Determine better parent based on current fitness
    fitness1 = calculate_c2(parent1)
    fitness2 = calculate_c2(parent2)
    better_parent = parent1 if fitness1 > fitness2 else parent2
    
    for i in range(n):
        if i < n1 and i < n2:
            # Choose based on probability biased toward better parent
            if random.random() < 0.7:  # 70% chance from better parent
                offspring.append(better_parent[i])
            else:
                offspring.append(parent1[i] if i < n1 else parent2[i])
        elif i < n1:
            offspring.append(parent1[i])
        elif i < n2:
            offspring.append(parent2[i])
        else:
            offspring.append(0.0)

    return offspring

def select_parents(population: List[List[float]], fitnesses: List[float],
                  tournament_size: int = 3) -> List[List[float]]:
    """Tournament selection with fitness proportionality"""
    selected = []
    fitness_array = np.array(fitnesses)
    
    # Normalize fitness for selection pressure
    fitness_normalized = fitness_array - np.min(fitness_array) + 1e-10
    fitness_probs = fitness_normalized / np.sum(fitness_normalized)
    
    for _ in range(len(population)):
        # Tournament selection with probabilistic weighting
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitness_probs[i] for i in tournament_indices]
        selected_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[selected_index].copy())

    return selected

def optimize_step_function() -> List[float]:
    """Main optimization routine with improvements"""
    # Parameters
    pop_size = 30
    generations = 80
    elite_size = 4
    early_stopping_patience = 8
    
    # Initialize population
    population = initialize_population(pop_size)
    fitness_history = []
    best_fitness = -float('inf')
    best_individual = None
    no_improvement_count = 0

    start_time = time.time()
    timeout_seconds = 80

    for gen in range(generations):
        # Check for timeout
        if time.time() - start_time > timeout_seconds:
            break
            
        # Evaluate fitness
        fitnesses = [calculate_c2(individual) for individual in population]
        current_best = max(fitnesses)
        fitness_history.append(current_best)

        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[fitnesses.index(current_best)].copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Print progress every 10 generations
        if gen % 10 == 0:
            print(f"Generation {gen}: Best C2 = {best_fitness:.6f}")

        # Early stopping based on convergence rate
        if no_improvement_count >= early_stopping_patience:
            # Check if recent improvement rate is very slow
            if len(fitness_history) >= 5:
                recent_change = fitness_history[-1] - fitness_history[-5]
                if recent_change < 1e-8:
                    print(f"Early stopping at generation {gen} due to minimal improvement")
                    break

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]

        # Keep elite
        elite = population[:elite_size]

        # Apply gradient-based local search to top performers in later generations
        if gen >= generations * 0.6:  # Apply in final 40% of generations
            for i in range(min(elite_size, len(elite))):
                if i < len(elite):
                    refined = gradient_based_local_search(elite[i], max_evals=30)
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

            # Mutation with adaptive rate and fitness history
            mutated_offspring = adaptive_gaussian_mutate(offspring, gen, generations, fitness_history)

            new_population.append(mutated_offspring)

        population = new_population[:pop_size]

    # Final evaluation and cleanup
    final_fitnesses = [calculate_c2(individual) for individual in population]
    best_final_individual = population[np.argmax(final_fitnesses)]

    # Return the best individual found during optimization
    if best_individual is None:
        return best_final_individual
    else:
        return best_individual

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run optimization
    best_solution = optimize_step_function()
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")