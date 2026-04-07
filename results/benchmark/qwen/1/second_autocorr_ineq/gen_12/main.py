# EVOLVE-BLOCK-START

import numpy as np
from numba import jit
import time
from scipy.optimize import differential_evolution, minimize
from scipy import signal
import random

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals, dx):
    """
    Efficiently compute autoconvolution using Numba JIT compilation
    """
    n = len(f_vals)
    # Autoconvolution using discrete convolution formula
    g = np.zeros(2*n - 1)
    
    # Compute convolution manually for efficiency
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < len(g):
                g[idx] += f_vals[i] * f_vals[j]
    
    return g

@jit(nopython=True)
def compute_norms_numba(g_vals):
    """
    Compute L1, L2^2, and L-infinity norms efficiently
    """
    n = len(g_vals)
    
    # L1 norm approximation (sum of absolute values)
    l1_norm = 0.0
    for i in range(n):
        l1_norm += abs(g_vals[i])
    
    # L2^2 norm (sum of squares)
    l2_sq_norm = 0.0
    for i in range(n):
        l2_sq_norm += g_vals[i] * g_vals[i]
    
    # L-infinity norm (maximum absolute value)
    linf_norm = 0.0
    for i in range(n):
        abs_val = abs(g_vals[i])
        if abs_val > linf_norm:
            linf_norm = abs_val
    
    return l1_norm, l2_sq_norm, linf_norm

def compute_trapezoidal_l2_sq(g_vals):
    """
    Compute L2^2 norm using proper trapezoidal integration for better accuracy
    """
    if len(g_vals) < 2:
        return np.sum(np.square(g_vals)) if len(g_vals) > 0 else 0.0
    
    # Trapezoidal rule for L2^2 norm
    # For each adjacent pair, integrate the square of the linear interpolation
    # We'll compute it as sum of trapezoidal contributions for consecutive pairs
    l2_sq = 0.0
    for i in range(len(g_vals) - 1):
        y1 = g_vals[i]
        y2 = g_vals[i+1]
        # For trapezoidal integration of y^2, we compute (h/3)*(y1^2 + y1*y2 + y2^2)
        # But since h=1 in our discrete case, it's simply (1/3)*(y1^2 + y1*y2 + y2^2)
        l2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0
    
    return l2_sq

@jit(nopython=True)
def compute_c2_numba(f_vals, dx):
    """
    Compute C2 value using optimized numba functions
    """
    # Compute autoconvolution
    g_vals = compute_autoconvolution_numba(f_vals, dx)
    
    # Compute norms
    l1, l2_sq, linf = compute_norms_numba(g_vals)
    
    # Avoid division by zero
    if l1 <= 1e-15 or linf <= 1e-15:
        return 0.0
    
    # Return C2 value
    return l2_sq / (l1 * linf)

def evaluate_step_function(f_vals):
    """
    Evaluate a step function and return C2 value
    """
    try:
        # Ensure non-negative values
        f_vals = np.array([max(0.0, x) for x in f_vals])
        
        # Use a fixed dx for consistent spacing
        dx = 0.5 / len(f_vals) if len(f_vals) > 0 else 0.001
        
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals, dx)
        
        # Compute norms with more accurate integration for L2^2
        l1_norm, l2_sq_norm, linf_norm = compute_norms_numba(g_vals)
        
        # If any norm is too small, return 0
        if l1_norm <= 1e-15 or linf_norm <= 1e-15:
            return 0.0
            
        # Use trapezoidal integration for L2^2 norm
        l2_sq_norm = compute_trapezoidal_l2_sq(g_vals)
        
        # Return C2 value
        return l2_sq_norm / (l1_norm * linf_norm)
        
    except Exception as e:
        return 0.0

def sophisticated_initialization(n_steps):
    """
    Create a sophisticated initial population with alternating high/low regions
    and Gaussian weighting
    """
    # Create alternating high-low pattern with some randomness
    pattern = []
    for i in range(n_steps):
        if i % 4 < 2:
            pattern.append(max(0.0, 1.0 + np.random.normal(0, 0.1)))
        else:
            pattern.append(max(0.0, 0.1 + np.random.normal(0, 0.05)))
    
    # Apply Gaussian smoothing to make transitions smoother
    smoothed = []
    sigma = 0.3
    for i in range(n_steps):
        weighted_sum = 0.0
        weight_sum = 0.0
        for j in range(max(0, i-5), min(n_steps, i+6)):
            weight = np.exp(-0.5 * ((i-j)/sigma)**2)
            weighted_sum += weight * pattern[j]
            weight_sum += weight
        smoothed.append(weighted_sum / weight_sum if weight_sum > 0 else 0.0)
    
    # Normalize to avoid extreme values
    max_val = max(smoothed) if max(smoothed) > 0 else 1.0
    normalized = [x/max_val for x in smoothed]
    
    return normalized

def create_initial_population(pop_size, n_steps):
    """
    Create initial population of step functions with sophisticated initialization
    """
    population = []
    for _ in range(pop_size):
        # Use sophisticated initialization
        step_heights = sophisticated_initialization(n_steps)
        population.append(step_heights)
    return population

def mutate_individual(individual, mutation_rate=0.15, noise_scale=0.1):
    """
    Mutate an individual step function with enhanced strategy
    """
    mutated = individual.copy()
    
    # Apply Gaussian noise to selected elements
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale)
            mutated[i] = max(0.0, mutated[i] + noise)
    
    # Occasionally adjust some values to maintain diversity
    for i in range(len(mutated)):
        if np.random.random() < 0.05:
            mutated[i] = max(0.0, mutated[i] * np.random.uniform(0.7, 1.3))
    
    return mutated

def crossover_individuals(parent1, parent2):
    """
    Perform crossover between two individuals with blending
    """
    # Blend parents with some probability
    child1 = []
    child2 = []
    
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            # Blend with some noise
            alpha = np.random.random()
            child1.append(alpha * parent1[i] + (1-alpha) * parent2[i] + np.random.normal(0, 0.01))
            child2.append((1-alpha) * parent1[i] + alpha * parent2[i] + np.random.normal(0, 0.01))
        else:
            child1.append(parent1[i] + np.random.normal(0, 0.01))
            child2.append(parent2[i] + np.random.normal(0, 0.01))
    
    # Ensure non-negativity
    child1 = [max(0.0, x) for x in child1]
    child2 = [max(0.0, x) for x in child2]
    
    return child1, child2

def local_refinement(x0, bounds, maxiter=50):
    """
    Local refinement using L-BFGS-B optimization
    """
    def objective(x):
        # Minimize negative C2 (equivalent to maximizing C2)
        return -evaluate_step_function(x)
    
    def gradient(x):
        # In practice, we'd compute gradients, but for now just return zeros
        # A full implementation would require analytical or finite difference gradients
        return np.zeros_like(x)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': maxiter, 'ftol': 1e-6})
        return result.x if result.success else x0
    except:
        return x0

def evolve_steps():
    """
    Evolve step functions to maximize C2 with hybrid approach
    """
    # Parameters
    pop_size = 30
    generations = 100
    n_steps = 1000  # Fixed number for consistency
    elite_size = 3
    
    # Create initial population with sophisticated initialization
    population = create_initial_population(pop_size, n_steps)
    
    # Define bounds for optimization (each step height between 0 and 10)
    bounds = [(0.0, 10.0) for _ in range(n_steps)]
    
    best_fitness = 0.0
    best_individual = None
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = evaluate_step_function(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness
        fitness_scores.sort(reverse=True)
        
        # Update best solution
        current_best_fitness, current_best_individual = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()
        
        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Gen {generation}: Best C2 = {best_fitness:.6f}")
        
        # Select elite
        elite = [ind for _, ind in fitness_scores[:elite_size]]
        
        # Create new population with local refinement
        new_population = elite.copy()
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(fitness_scores, 3)
            parent2 = tournament_selection(fitness_scores, 3)
            
            # Crossover
            child1, child2 = crossover_individuals(parent1, parent2)
            
            # Mutate
            child1 = mutate_individual(child1, mutation_rate=0.15, noise_scale=0.05)
            child2 = mutate_individual(child2, mutation_rate=0.15, noise_scale=0.05)
            
            # Apply local refinement
            child1 = local_refinement(child1, bounds, maxiter=20)
            child2 = local_refinement(child2, bounds, maxiter=20)
            
            new_population.extend([child1, child2])
        
        # Trim population to exact size
        population = new_population[:pop_size]
    
    # Final local refinement on best solution
    if best_individual is not None:
        refined_best = local_refinement(best_individual, bounds, maxiter=50)
        final_fitness = evaluate_step_function(refined_best)
        if final_fitness > best_fitness:
            best_fitness = final_fitness
            best_individual = refined_best
    
    return best_individual if best_individual is not None else []

def tournament_selection(fitness_scores, k):
    """
    Select individual using tournament selection
    """
    tournament = random.sample(fitness_scores, min(k, len(fitness_scores)))
    return max(tournament, key=lambda x: x[0])[1]

def construct_function() -> list[float]:
    """
    Function to construct step-function with high C2 value.
    Uses evolutionary optimization with sophisticated initialization and hybrid refinement.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run evolution
    start_time = time.time()
    try:
        best_solution = evolve_steps()
    except Exception as e:
        # Fallback to simple approach if evolution fails
        print(f"Evolution failed with error: {e}")
        best_solution = [1.0] * 100  # Default case
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Ensure the solution is valid
    if not best_solution:
        best_solution = [1.0] * 100
    
    print(f"Eval time: {eval_time:.4f}s")
    print(f"Best C2 found: {evaluate_step_function(best_solution):.6f}")
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
