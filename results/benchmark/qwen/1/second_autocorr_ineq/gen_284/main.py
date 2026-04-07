# EVOLVE-BLOCK-START

import numpy as np
from numba import jit, prange
import time
import random
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings('ignore')

# Global constants for performance tuning
MAX_TIME_SECONDS = 90.0
DEFAULT_STEPS = 3000
MIN_STEPS = 200
MAX_STEPS = 10000
POPULATION_SIZE_BASE = 12
INITIAL_REFINEMENT_ITERATIONS = 30
FINAL_REFINEMENT_ITERATIONS = 20

# Optimized sparse convolution computation using Numba
@jit(nopython=True, parallel=True)
def compute_sparse_autoconvolution(f_vals):
    """
    Efficiently compute autoconvolution using optimized Numba loops with parallel processing.
    Exploits the fact that step functions are piecewise constant.
    """
    n = len(f_vals)
    
    # For dense convolution: g[i] = sum_{j+k=i} f[j] * f[k]
    g = np.zeros(2*n - 1, dtype=np.float64)
    
    # Optimized nested loop for autoconvolution with parallel processing
    for i in prange(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    
    return g

@jit(nopython=True)
def compute_sparse_norms(g_vals):
    """
    Compute L1, L2^2, and L-infinity norms using highly optimized loops
    """
    n = len(g_vals)
    
    # L1 norm (sum of absolute values)
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

@jit(nopython=True)
def compute_c2_sparse(f_vals):
    """
    Fast computation of C2 using sparse convolution and norms
    """
    # Compute autoconvolution
    g_vals = compute_sparse_autoconvolution(f_vals)
    
    # Compute norms
    l1, l2_sq, linf = compute_sparse_norms(g_vals)
    
    # Avoid division by zero
    if l1 <= 1e-15 or linf <= 1e-15:
        return 0.0
    
    # Return C2 value
    return l2_sq / (l1 * linf)

def evaluate_step_function_sparse(f_vals):
    """
    Evaluate step function with robust error handling and sparse computation
    """
    try:
        # Ensure non-negative values with clipping
        f_vals = np.array([max(0.0, x) for x in f_vals])
        
        # Handle edge cases
        if len(f_vals) == 0 or np.isnan(np.sum(f_vals)) or np.isinf(np.sum(f_vals)):
            return 0.0
            
        # If all values are zero, return 0
        if np.sum(f_vals) == 0:
            return 0.0
            
        # Compute C2 value using sparse method
        c2 = compute_c2_sparse(f_vals)
        
        # Validate result
        if np.isnan(c2) or np.isinf(c2) or c2 < 0:
            return 0.0
            
        return c2
    except Exception:
        return 0.0

def create_mathematical_pattern(n_steps):
    """Create a mathematically informed pattern that's likely to produce good C2 values"""
    # Create a pattern designed to generate favorable autoconvolution properties
    # Based on theoretical insights from optimal step functions
    x = np.linspace(0, 1, n_steps)
    
    # Combination of multiple frequency components to create complex convolution profile
    pattern = (
        0.8 * np.exp(-10 * (x - 0.5)**2) +  # Central peak
        0.4 * np.exp(-20 * (x - 0.2)**2) +  # Left peak
        0.4 * np.exp(-20 * (x - 0.8)**2) +  # Right peak
        0.2 * np.sin(10 * np.pi * x) +     # High frequency oscillation
        0.1 * np.sin(5 * np.pi * x)         # Medium frequency oscillation
    )
    
    # Clip negative values and normalize
    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    
    return pattern

def create_balanced_pattern(n_steps):
    """Create a balanced pattern optimized for convolution"""
    # Create a pattern that maintains good distributional properties
    pattern = np.ones(n_steps) * 0.6
    
    # Add structured variation to increase convolution diversity
    for i in range(0, n_steps, 8):
        if i < n_steps:
            pattern[i] = 1.2 + np.random.random() * 0.3
    
    # Add some smoothing
    pattern = np.clip(pattern, 0, np.inf)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    
    return pattern

def create_peak_and_valley(n_steps):
    """Create alternating peak-valley pattern optimized for C2"""
    pattern = []
    for i in range(n_steps):
        if i % 4 == 0:
            pattern.append(1.0 + np.random.random() * 0.4)
        elif i % 4 == 2:
            pattern.append(0.3 + np.random.random() * 0.2)
        else:
            pattern.append(0.7 + np.random.random() * 0.2)
    
    pattern = np.array(pattern)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_multiscale_sparse_initialization(n_steps):
    """
    Create diverse initial solutions with different structural properties
    """
    strategies = [
        lambda n: create_mathematical_pattern(n),
        lambda n: create_balanced_pattern(n),
        lambda n: create_peak_and_valley(n),
        lambda n: create_bell_shaped_pattern(n),
        lambda n: create_alternating_pattern(n)
    ]
    
    # Choose a strategy
    strategy = np.random.choice(strategies)
    pattern = strategy(n_steps)
    
    # Apply minor random perturbations for diversity
    noise = np.random.normal(0, 0.03, n_steps)
    pattern = pattern + noise
    pattern = np.maximum(pattern, 0.0)
    
    # Final normalization
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    
    return pattern

def create_bell_shaped_pattern(n_steps):
    """Create bell-shaped pattern optimized for convolution"""
    x = np.linspace(0, 1, n_steps)
    # Create Gaussian-like shape with emphasis on central region
    pattern = 1.0 + 0.8 * np.exp(-12 * (x - 0.5)**2) - 0.3 * np.exp(-6 * x**2) - 0.3 * np.exp(-6 * (1-x)**2)
    pattern = np.clip(pattern, 0, np.inf)
    
    # Normalize appropriately
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def create_alternating_pattern(n_steps):
    """Create alternating high/low pattern for convolution"""
    pattern = []
    for i in range(n_steps):
        if i % 3 == 0:
            pattern.append(1.0 + np.random.random() * 0.3)
        elif i % 3 == 1:
            pattern.append(0.3 + np.random.random() * 0.2)
        else:
            pattern.append(0.7 + np.random.random() * 0.2)
    
    pattern = np.array(pattern)
    if np.sum(pattern) > 0:
        pattern = pattern * n_steps / np.sum(pattern)
    return pattern

def adaptive_evolutionary_optimization(initial_population):
    """
    Enhanced evolutionary optimization with adaptive parameters
    """
    # Track convergence and adapt parameters dynamically
    best_scores = []
    patience_counter = 0
    max_patience = 8
    population_size = POPULATION_SIZE_BASE
    
    # Start with initial population
    population = [list(ind) for ind in initial_population]
    current_best = max(population, key=evaluate_step_function_sparse)
    best_scores.append(evaluate_step_function_sparse(current_best))
    
    # Adaptive parameters based on convergence behavior
    for generation in range(180):  # Limited to prevent timeout
        # Dynamic population sizing
        if len(population) < 10:
            population_size = 10
        elif len(population) > 25:
            population_size = 20
        else:
            population_size = len(population) // 2 + 5
            
        # Evaluate all individuals with sparse computation
        fitnesses = [evaluate_step_function_sparse(ind) for ind in population]
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitnesses = [fitnesses[i] for i in sorted_indices]
        
        # Update best
        current_best = sorted_population[0]
        best_scores.append(sorted_fitnesses[0])
        
        # Check for convergence with multiple criteria
        if len(best_scores) >= 5:
            # Check for stagnation in improvement
            recent_improvement = best_scores[-1] - best_scores[-5]
            if recent_improvement < 1e-9:
                patience_counter += 1
            else:
                patience_counter = 0
                
            if patience_counter >= max_patience:
                # Increase population size to escape local minimum
                population_size = min(population_size * 2, 40)
                patience_counter = 0
                
        # Create offspring using tournament selection and crossover
        new_population = []
        
        # Elitism: keep the best 25%
        elite_count = max(1, int(0.25 * population_size))
        new_population.extend(sorted_population[:elite_count])
        
        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(sorted_population), tournament_size)
            tournament_fitnesses = [sorted_fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            
            # Clone selected parent
            parent = sorted_population[winner_index].copy()
            
            # Adaptive mutation with generation-dependent strength
            mutation_strength = 0.15 * (1 - generation / 180.0)  # Decrease over time
            
            # Mutate with sparse consideration
            for i in range(len(parent)):
                if np.random.random() < 0.12:  # 12% chance to mutate each element
                    noise = np.random.normal(0, mutation_strength)
                    parent[i] = max(0, parent[i] + noise)
                    
            new_population.append(parent)
            
        # Replace population
        population = new_population
        
        # Early termination based on time
        if time.time() - start_time > MAX_TIME_SECONDS * 0.9:
            break
            
    return current_best

def advanced_refinement_strategy(initial_solution):
    """
    Advanced refinement using multiple techniques for better convergence
    """
    # Try different refinement methods
    best_solution = initial_solution.copy()
    best_c2 = evaluate_step_function_sparse(best_solution)
    
    # Method 1: Differential evolution for global search on reduced dimensions
    try:
        # Focus on dimensions that contribute most to improvement
        reduced_solution = best_solution[:min(len(best_solution), 500)]
        bounds = [(0, 10.0) for _ in range(len(reduced_solution))]
        
        def objective(x):
            # Pad with zeros to original size
            extended_x = list(x) + [1.0] * (len(best_solution) - len(x))
            return -evaluate_step_function_sparse(extended_x)
        
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=25,
            popsize=8,
            seed=42,
            disp=False
        )
        
        if de_result.success:
            refined_solution = np.maximum(de_result.x, 0)
            # Extend back to original size
            extended_refined = list(refined_solution) + [1.0] * (len(best_solution) - len(refined_solution))
            refined_c2 = evaluate_step_function_sparse(extended_refined)
            if refined_c2 > best_c2:
                best_solution = extended_refined
                best_c2 = refined_c2
    except Exception:
        pass
    
    # Method 2: Local refinement with coordinate-wise optimization
    try:
        # Use coordinate descent on smaller subset for efficiency
        x0 = np.array(best_solution[:min(300, len(best_solution))])
        bounds = [(0, 10.0)] * len(x0)
        
        def objective(x):
            # Extend to full size
            extended_x = list(x) + [1.0] * (len(best_solution) - len(x))
            return -evaluate_step_function_sparse(extended_x)
        
        # Use L-BFGS-B for local refinement
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 25})
        
        if res.success:
            refined_solution = np.maximum(res.x, 0)
            # Extend back to original size
            extended_refined = list(refined_solution) + [1.0] * (len(best_solution) - len(refined_solution))
            refined_c2 = evaluate_step_function_sparse(extended_refined)
            if refined_c2 > best_c2:
                best_solution = extended_refined
                best_c2 = refined_c2
    except Exception:
        pass
    
    # Method 3: Sparse-aware perturbations with structure preservation
    try:
        # Apply small random perturbations respecting sparse structure
        perturbed = best_solution.copy()
        for i in range(len(perturbed)):
            if np.random.random() < 0.08:  # 8% chance to perturb each element
                # Use adaptive perturbation strength
                perturbation = np.random.normal(0, 0.015)
                perturbed[i] = max(0, perturbed[i] + perturbation)
        
        # Normalize sparse representation
        if np.sum(perturbed) > 0:
            perturbed = perturbed / np.sum(perturbed) * len(perturbed)
            
        perturbed_c2 = evaluate_step_function_sparse(perturbed)
        if perturbed_c2 > best_c2:
            best_solution = perturbed
            best_c2 = perturbed_c2
    except Exception:
        pass
    
    return best_solution

def hybrid_optimization_pipeline():
    """
    Hybrid optimization pipeline combining multiple strategies
    """
    global start_time
    start_time = time.time()
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize with diverse mathematical patterns
    initial_solutions = []
    n_attempts = 20
    
    for i in range(n_attempts):
        # Create diverse initial solutions with mathematical optimization in mind
        n_steps = np.random.randint(MIN_STEPS, MAX_STEPS)
        init_solution = create_multiscale_sparse_initialization(n_steps)
        initial_solutions.append(init_solution)
    
    # Select best initial solution
    best_init = max(initial_solutions, key=evaluate_step_function_sparse)
    
    # Apply adaptive evolutionary optimization
    evolved_solution = adaptive_evolutionary_optimization(initial_solutions)
    
    # Apply advanced refinement
    refined_solution = advanced_refinement_strategy(evolved_solution)
    
    # Final evaluation and return the best
    final_c2 = evaluate_step_function_sparse(refined_solution)
    initial_c2 = evaluate_step_function_sparse(best_init)
    
    if final_c2 > initial_c2:
        result = refined_solution
    else:
        result = best_init
    
    # Ensure proper length
    if len(result) < MIN_STEPS:
        result.extend([1.0] * (MIN_STEPS - len(result)))
    elif len(result) > MAX_STEPS:
        result = result[:MAX_STEPS]
    
    # Normalize if needed
    if np.sum(result) > 0:
        result = np.array(result) / np.sum(result) * len(result)
    
    # Ensure non-negativity and finite values
    result = np.clip(result, 0, np.inf)
    
    return result.tolist()

def construct_function() -> list[float]:
    """
    Optimized function to construct step-function with high C2 value.
    Uses hybrid approach combining mathematical insights, evolutionary optimization,
    and advanced refinement techniques.
    """
    return hybrid_optimization_pipeline()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
