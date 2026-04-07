# EVOLVE-BLOCK-START
import numpy as np
from numba import jit
import time
from joblib import Parallel, delayed
import random
from deap import base, creator, tools, algorithms
import copy
from scipy.optimize import minimize
from scipy.stats import qmc

# Core computation module with JIT compilation
@jit(nopython=True)
def compute_autoconvolution_jit(f_vals, step_width):
    """
    Compute autoconvolution using numba for speed
    """
    n = len(f_vals)
    # Autoconvolution size is 2*n - 1
    g_size = 2 * n - 1
    g_vals = np.zeros(g_size)

    # Compute convolution directly
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_size:
                g_vals[idx] += f_vals[i] * f_vals[j]

    return g_vals

@jit(nopython=True)
def compute_convolution_norms(f_values, domain_length=0.5):
    """
    Compute the three norms needed for C2 calculation using the provided step function.
    """
    n_steps = len(f_values)
    if n_steps == 0:
        return 0.0, 0.0, 0.0

    # Step size
    dx = domain_length / n_steps

    # Compute autoconvolution g = f * f using direct computation
    g_size = 2 * n_steps - 1
    g = np.zeros(g_size)

    # Compute autoconvolution using direct convolution sum
    for i in range(n_steps):
        for j in range(n_steps):
            k = i + j
            if 0 <= k < g_size:
                g[k] += f_values[i] * f_values[j] * dx

    # Compute norms using piecewise linear integration approach
    # For ||g||₂² using trapezoidal-like formula: (dx/3)(g₀² + g₀g₁ + g₁²)
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

# Advanced optimization module with adaptive techniques
@jit(nopython=True)
def compute_autoconvolution_norms(f_vals):
    """
    Compute the autoconvolution g = f*f and return its norms.
    Uses fast numba-compiled operations.
    """
    n = len(f_vals)
    # Autoconvolution using direct computation
    g = np.zeros(2*n - 1)

    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]

    # Compute norms
    g_squared = g * g
    norm_g2_squared = np.sum(g_squared)
    norm_g1 = np.sum(np.abs(g))
    norm_g_inf = np.max(np.abs(g))

    return norm_g2_squared, norm_g1, norm_g_inf

def calculate_c2(f_vals):
    """
    Calculate C2 value for given step function values.
    """
    norm_g2_squared, norm_g1, norm_g_inf = compute_autoconvolution_norms(f_vals)

    # Avoid division by zero
    if norm_g1 < 1e-15 or norm_g_inf < 1e-15:
        return 0.0

    c2 = norm_g2_squared / (norm_g1 * norm_g_inf)
    return c2

def latin_hypercube_sample(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """Generate Latin Hypercube Sample for diverse initialization."""
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    sample = sampler.random(n=n_samples)
    return sample

def adaptive_initialize_population(pop_size: int, min_length: int = 300, max_length: int = 2000) -> list[list[float]]:
    """Initialize population with diverse configurations using LHS-based approach."""
    population = []
    
    # Generate diverse lengths
    lengths = np.random.randint(min_length, max_length, pop_size)
    
    # Create base patterns using LHS for better diversity
    lhs_samples = latin_hypercube_sample(pop_size, 4, seed=42)
    
    for i in range(pop_size):
        length = lengths[i]
        
        # Different pattern types based on LHS values
        pattern_type = i % 4
        if pattern_type == 0:  # Gaussian-like
            base_vals = np.exp(-np.linspace(0, 3, length//2)**2)
            vals = np.concatenate([base_vals, np.flip(base_vals)])
        elif pattern_type == 1:  # Peaks
            vals = np.zeros(length)
            for peak_pos in [length//4, length//2, 3*length//4]:
                if peak_pos < length:
                    vals[peak_pos] = 1.0
            # Add smoothing
            vals = np.convolve(vals, np.ones(5)/5, mode='same')
        elif pattern_type == 2:  # Uniform with noise
            vals = np.ones(length) + np.random.normal(0, 0.1, length)
        else:  # Exponential decay
            vals = np.exp(-np.linspace(0, 4, length))
            
        # Normalize to reasonable range
        vals = np.maximum(vals, 0.0)
        if np.sum(vals) > 0:
            vals = vals / np.sum(vals) * 20
        
        population.append(vals.tolist())
    
    return population

def adaptive_local_search(initial_vals: list[float], max_iter: int = 500, tolerance: float = 1e-6) -> list[float]:
    """Perform adaptive local optimization with dynamic step sizes."""
    def objective(f_vals):
        return -calculate_c2(f_vals)
    
    # Adaptive step sizes based on initial values
    step_sizes = [max(0.1, val * 0.05) for val in initial_vals]
    
    # Try different optimization methods
    results = []
    
    # Nelder-Mead with adaptive parameters
    try:
        result = minimize(objective, initial_vals, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xtol': tolerance, 'ftol': tolerance})
        if result.success:
            results.append(result.x)
    except:
        pass
    
    # L-BFGS with adaptive parameters
    try:
        result = minimize(objective, initial_vals, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': tolerance})
        if result.success:
            results.append(result.x)
    except:
        pass
    
    # Coordinate-wise refinement if no optimization succeeded
    if not results:
        refined = initial_vals.copy()
        for idx in range(len(refined)):
            old_val = refined[idx]
            best_val = old_val
            best_score = calculate_c2(refined)
            
            # Try small variations
            step_size = max(0.001, old_val * 0.01)
            for variation in [-step_size, 0, step_size]:
                test_vals = refined.copy()
                test_vals[idx] = max(0, old_val + variation)
                new_score = calculate_c2(test_vals)
                if new_score > best_score:
                    best_score = new_score
                    best_val = test_vals[idx]
                    
            refined[idx] = best_val
        return refined
    
    # Return best result
    if results:
        best_result = min(results, key=lambda x: -calculate_c2(x))
        return np.maximum(best_result, 0.0).tolist()
    
    return initial_vals

def adaptive_selection(population: list[list[float]], fitness_scores: list[float], 
                       elite_size: int = 3, tournament_size: int = 4) -> list[list[float]]:
    """Adaptive selection that considers both fitness and diversity."""
    # Sort by fitness
    sorted_indices = sorted(range(len(fitness_scores)), 
                          key=lambda i: fitness_scores[i], reverse=True)
    selected = [population[i] for i in sorted_indices[:elite_size]]
    
    # Add diverse individuals
    remaining_indices = sorted_indices[elite_size:]
    np.random.shuffle(remaining_indices)
    
    # Add tournament-selected individuals
    for i in range(min(len(remaining_indices), len(population) - elite_size)):
        tournament_indices = np.random.choice(remaining_indices[:10], 
                                            min(tournament_size, len(remaining_indices[:10])), 
                                            replace=False)
        tournament_fitness = [fitness_scores[j] for j in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx])
        
    return selected[:len(population)]

def adaptive_crossover(parent1: list[float], parent2: list[float]) -> list[float]:
    """Adaptive crossover with variable recombination."""
    n1, n2 = len(parent1), len(parent2)
    n = max(n1, n2)
    
    # Determine crossover point with adaptive randomness
    crossover_point = random.randint(1, n-1) if n > 1 else 0
    
    # Create offspring with blending and random selection
    offspring = []
    for i in range(n):
        if i < crossover_point and i < n1:
            offspring.append(parent1[i])
        elif i >= crossover_point and i < n2:
            offspring.append(parent2[i])
        else:
            # Choose randomly or average
            choices = []
            if i < n1:
                choices.append(parent1[i])
            if i < n2:
                choices.append(parent2[i])
                
            if choices:
                # Weighted choice with preference for parent1
                if random.random() < 0.7:
                    offspring.append(random.choice(choices))
                else:
                    offspring.append(sum(choices) / len(choices))
            else:
                offspring.append(0.0)
    
    return offspring

def adaptive_mutation(individual: list[float], generation: int, max_gen: int) -> list[float]:
    """Adaptive mutation with decreasing strength over time."""
    mutated = individual.copy()
    mutation_rate = 0.3 * (1 - generation/max_gen) + 0.05  # Decreasing over generations
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Adaptive noise based on current value
            noise_std = max(0.001, mutated[i] * 0.1)
            noise = np.random.normal(0, noise_std)
            mutated[i] = max(0, mutated[i] + noise)
    
    return mutated

def smart_restart_check(current_best_score: float, previous_best_scores: list[float], 
                       patience: int = 10, improvement_threshold: float = 1e-6) -> bool:
    """Check if we should restart based on stagnation."""
    if len(previous_best_scores) < patience:
        return False
    
    recent_scores = previous_best_scores[-patience:]
    if len(recent_scores) < 2:
        return False
    
    # Check for minimal improvement
    if (recent_scores[-1] - recent_scores[0]) < improvement_threshold:
        return True
    
    return False

def adaptive_evolutionary_optimizer() -> list[float]:
    """Main adaptive evolutionary optimization routine."""
    start_time = time.time()
    
    # Parameters
    initial_pop_size = 25
    max_generations = 75
    elite_size = 4
    max_time_seconds = 85  # Leave margin for cleanup
    
    # Initialize population
    population = adaptive_initialize_population(initial_pop_size)
    best_overall = None
    best_c2_score = -float('inf')
    best_history = []
    
    # Evolution loop
    for generation in range(max_generations):
        if time.time() - start_time > max_time_seconds:
            break
            
        # Evaluate current population
        fitness_scores = []
        for individual in population:
            score = calculate_c2(individual)
            fitness_scores.append(score)
            
            # Track best overall
            if score > best_c2_score:
                best_c2_score = score
                best_overall = individual.copy()
        
        # Update history
        best_history.append(best_c2_score)
        
        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Best C2 = {best_c2_score:.6f}")
        
        # Smart restart check
        if smart_restart_check(best_c2_score, best_history, patience=15):
            # Reinitialize with diverse patterns
            population = adaptive_initialize_population(initial_pop_size)
            print(f"Restart at generation {generation} due to stagnation")
            continue
        
        # Selection
        selected_population = adaptive_selection(population, fitness_scores, elite_size)
        
        # Create new population
        new_population = selected_population.copy()
        
        # Generate offspring - keep some elite and add mutated children
        while len(new_population) < initial_pop_size:
            # Select parents
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            # Crossover
            offspring = adaptive_crossover(parent1, parent2)
            
            # Mutation
            mutated_offspring = adaptive_mutation(offspring, generation, max_generations)
            
            new_population.append(mutated_offspring)
        
        # Keep only required population size
        population = new_population[:initial_pop_size]
        
        # Local optimization on top individuals periodically
        if generation % 5 == 0:
            top_individuals = sorted(zip(population, fitness_scores), 
                                   key=lambda x: x[1], reverse=True)[:3]
            for individual, _ in top_individuals:
                optimized = adaptive_local_search(individual)
                score = calculate_c2(optimized)
                if score > best_c2_score:
                    best_c2_score = score
                    best_overall = optimized.copy()
    
    # Final optimization of best individual
    if best_overall is not None:
        final_candidate = adaptive_local_search(best_overall)
        final_score = calculate_c2(final_candidate)
        if final_score > best_c2_score:
            return final_candidate
    
    return best_overall if best_overall is not None else [1.0] * 500

def generate_multiscale_gaussian_initial_function(n_steps):
    """Generate an initial function using multi-scale Gaussian pattern construction.

    This creates a structured pattern with multiple Gaussian bumps at different scales
    to encourage good convolution behavior across various spatial frequencies.
    """
    # Create base function with multi-scale Gaussian components
    f_values = np.zeros(n_steps)

    # Define multiple scales of Gaussian bumps - more systematically spaced
    scales = [n_steps // 25, n_steps // 20, n_steps // 15, n_steps // 10, n_steps // 8]

    # Create more evenly distributed centers
    centers = []
    num_centers = min(5, n_steps // 8)
    for i in range(num_centers):
        centers.append(int((i + 1) * n_steps / (num_centers + 1)))

    # Create Gaussian bumps at different scales and positions
    for scale in scales:
        for center in centers:
            # Create Gaussian bump with deterministic amplitude scaling
            x = np.arange(n_steps)
            gauss = np.exp(-0.5 * ((x - center) / scale) ** 2)
            # Use deterministic amplitude instead of random for consistency
            amplitude = 1.0 + 0.5 * np.sin(center / n_steps * np.pi * 2)
            f_values += gauss * amplitude

    # Add some additional structured variation
    # Create a base pattern that encourages uniformity in convolution
    base_pattern = np.sin(np.linspace(0, 4*np.pi, n_steps)) * 0.3 + 0.7
    f_values += base_pattern * 0.5

    # Ensure non-negativity and normalize
    f_values = np.maximum(f_values, 0)

    # Normalize to control the overall magnitude
    total = np.sum(f_values)
    if total > 0:
        f_values = f_values / total * 5.0

    return f_values.tolist()

def enhanced_local_refinement(initial_f, max_iterations=30):
    """Enhanced local refinement with adaptive search space"""
    refined_individual = initial_f.copy()
    old_c2 = compute_c2(refined_individual)
    
    for coord_iter in range(max_iterations):
        improved = False
        # Sample a subset of indices for more efficient search
        search_indices = np.random.choice(len(refined_individual),
                                        min(30, len(refined_individual)//2),
                                        replace=False)

        for i in search_indices:
            original_value = refined_individual[i]

            # Try multiple step sizes for adaptive search
            step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

            # Try both positive and negative perturbations with enhanced exploration
            for step in step_sizes:
                for direction in [1, -1]:
                    test_individual = refined_individual.copy()
                    new_val = original_value + direction * step
                    test_individual[i] = max(0, new_val)

                    new_c2 = compute_c2(test_individual)
                    if new_c2 > old_c2:
                        refined_individual = test_individual
                        old_c2 = new_c2
                        improved = True

                        # If we found improvement, break to move to next index
                        break
                if improved:
                    break

        # Break if no improvement was found in this iteration
        if not improved:
            break

    return refined_individual

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using hybrid evolutionary approach."""
    # Set parameters
    n_steps = 200
    max_time = 85  # seconds
    start_time = time.time()

    # Try several different initialization strategies and optimization approaches
    best_solution = None
    best_c2 = -np.inf
    
    # Strategy 1: Adaptive evolutionary optimization
    try:
        np.random.seed(42)
        random.seed(42)
        evol_solution = adaptive_evolutionary_optimizer()
        if evol_solution is not None:
            evol_c2 = calculate_c2(evol_solution)
            if evol_c2 > best_c2:
                best_c2 = evol_c2
                best_solution = evol_solution
    except Exception as e:
        pass

    # Strategy 2: Multi-scale Gaussian initialization with local refinement
    try:
        if best_solution is None:
            np.random.seed(42)
            f_values = generate_multiscale_gaussian_initial_function(n_steps)
            # Normalize for better numerical behavior
            total = sum(f_values)
            if total > 0:
                f_values = [x / total * 10 for x in f_values]
            
            # Local refinement
            refined_solution = enhanced_local_refinement(f_values)
            refined_c2 = compute_c2(refined_solution)
            
            if refined_c2 > best_c2:
                best_c2 = refined_c2
                best_solution = refined_solution
    except Exception as e:
        pass

    # Strategy 3: Direct optimization using scipy with diverse starting points
    try:
        if best_solution is None:
            # Multiple initialization attempts
            for attempt in range(3):
                np.random.seed(attempt * 1000 + 42)
                
                # Try different initializations
                if attempt == 0:
                    # Gamma distribution
                    initial_f_vals = np.random.gamma(2.0, 2.0, n_steps)
                elif attempt == 1:
                    # Geometric pattern
                    base_vals = np.geomspace(1, 0.01, num=n_steps // 2)
                    peaks = np.zeros(n_steps)
                    peak_positions = [n_steps // 4, n_steps // 2, 3 * n_steps // 4]
                    for pos in peak_positions:
                        if pos < n_steps:
                            peaks[pos] = 2.0
                    combined = base_vals[:n_steps // 2] + peaks[:n_steps // 2]
                    if len(combined) < n_steps:
                        remaining = n_steps - len(combined)
                        tail_vals = np.geomspace(0.01, 0.001, num=remaining)
                        combined = np.concatenate([combined, tail_vals])
                    if len(combined) > n_steps:
                        combined = combined[:n_steps]
                    initial_f_vals = combined
                else:
                    # Random uniform
                    initial_f_vals = np.random.uniform(0, 1, n_steps)
                
                # Normalize
                if np.sum(initial_f_vals) > 0:
                    initial_f_vals = initial_f_vals / np.sum(initial_f_vals) * 100
                
                initial_f_vals_list = initial_f_vals.tolist()
                
                def objective(f_vals):
                    return -calculate_c2(f_vals)
                
                # Try optimization
                try:
                    result = minimize(objective, initial_f_vals_list, method='Nelder-Mead', 
                                    options={'maxiter': 200, 'xtol': 1e-6})
                    if result.success:
                        optimized_values = np.maximum(result.x, 0.0)
                        final_c2 = calculate_c2(optimized_values.tolist())
                        if final_c2 > best_c2:
                            best_c2 = final_c2
                            best_solution = optimized_values.tolist()
                except:
                    pass
    except Exception as e:
        pass

    # Final fallback to simple uniform distribution
    if best_solution is None:
        best_solution = [1.0] * n_steps

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")