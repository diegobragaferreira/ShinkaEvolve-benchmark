# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import fft, ifft
from numba import jit, prange
import random
import time
from scipy.signal import convolve
import copy
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit
import warnings

# Global constants for performance and optimization
POPULATION_SIZE_BASE = 20
MAX_GENERATIONS = 100
MUTATION_RATE = 0.8
CROSSOVER_PROB = 0.7
NUM_STARTS = 6
MAX_TIME_SECONDS = 90.0

@jit(nopython=True)
def compute_autoconvolution_sparse(f_vals):
    """
    Sparse convolution computation optimized for step functions
    Uses direct computation for better numerical control and compatibility
    """
    n = len(f_vals)
    if n == 0:
        return np.array([])
    
    # For autoconvolution g = f * f, we compute:
    # g[k] = sum_{i+j=k} f[i] * f[j]
    g = np.zeros(2 * n - 1)
    
    # Direct computation using nested loops - optimized for step functions
    for i in range(n):
        for j in range(n):
            g[i + j] += f_vals[i] * f_vals[j]
    
    return g

@jit(nopython=True)
def compute_autoconvolution_fft_efficient(f_vals):
    """
    FFT-based autoconvolution for computational efficiency
    This provides fast computation but needs careful handling for proper integration
    """
    n = len(f_vals)
    if n == 0:
        return np.array([])
    
    # Pad to appropriate length for linear convolution
    pad_len = 2 * n - 1
    padded = np.zeros(pad_len)
    padded[:n] = f_vals
    
    # FFT method for convolution
    F = fft(padded)
    G_fft = F * F  # Point-wise multiplication in frequency domain
    g = ifft(G_fft).real
    
    return g[:pad_len]

@jit(nopython=True)
def compute_c2_norms_sparse(f_vals):
    """
    Compute C2 norms with proper integration using trapezoidal approach
    This implements the exact method described in problem formulation:
    for interval with heights y1, y2 and width h, contribution is (h/3)(y1² + y1y2 + y2²)
    """
    f = np.array(f_vals, dtype=np.float64)
    f = np.maximum(f, 0)  # Clip negative values to 0

    if len(f) == 0:
        return 0.0, 0.0, 0.0

    # Compute autoconvolution using sparse method
    g_full = compute_autoconvolution_sparse(f)

    # Trim to central portion matching domain [-1/4, 1/4] 
    # Assuming we want the middle portion of the autoconvolution
    n = len(f)
    g_center_start = (len(g_full) - 2*n + 1) // 2
    g_center_end = g_center_start + 2*n - 1
    g_trimmed = g_full[g_center_start:g_center_end]

    if len(g_trimmed) < 2:
        norm_l2_sq = 0.0
    else:
        # Trapezoidal integration for L2^2 using exact formula from problem
        # Each interval contributes (h/3)(y1^2 + y1y2 + y2^2)
        # Width of each interval is 1/n since we map n points over domain [-1/4, 1/4]  
        # So interval width is (0.5)/n = 0.5/n
        h = 0.5 / (len(g_trimmed) - 1) if len(g_trimmed) > 1 else 0.001
        g_abs = np.abs(g_trimmed)
        
        # Use trapezoidal-like formula for piecewise linear integration
        # For n points, we have n-1 intervals
        y1 = g_abs[:-1]
        y2 = g_abs[1:]
        norm_l2_sq = np.sum(h * (y1**2 + y1*y2 + y2**2) / 3.0)

    # L1 norm (normalized by number of points + 1)
    norm_l1 = np.sum(np.abs(g_trimmed)) / (len(g_trimmed) + 1) if len(g_trimmed) > 0 else 1e-15

    # Infinity norm
    norm_inf = np.max(np.abs(g_trimmed)) if len(g_trimmed) > 0 else 1e-15

    return norm_l2_sq, norm_l1, norm_inf

@jit(nopython=True)
def compute_c2_score_sparse(f_vals):
    """
    Compute C2 score using the sparse norms computation
    """
    norm_l2_sq, norm_l1, norm_inf = compute_c2_norms_sparse(f_vals)

    # Avoid division by zero
    if norm_l1 <= 1e-15 or norm_inf <= 1e-15:
        return 0.0

    return norm_l2_sq / (norm_l1 * norm_inf)

# JAX versions for automatic differentiation and gradient computation
@jax_jit
def compute_autoconvolution_jax(f_vals):
    """JAX version of autoconvolution for gradient computation"""
    f = jnp.array(f_vals)
    f = jnp.maximum(f, 0)  # Clip negative values to 0

    if len(f) == 0:
        return jnp.array([])

    # For exact matching with sparse computation, we should replicate 
    # the indexing, but for simplicity we use standard convolution
    # Note: In practice, the indexing is important for the correct C2 computation
    g = jnp.convolve(f, f, mode='full')
    
    # Trim to center portion for comparison with our manual sparse version
    n = len(f)
    center_start = len(g) // 2 - n + 1
    center_end = len(g) // 2 + n - 1
    g_trimmed = g[center_start:center_end]

    return g_trimmed

@jax_jit
def compute_c2_jax(f_vals):
    """JAX version of C2 computation"""
    g = compute_autoconvolution_jax(f_vals)

    # Compute norms
    norm_l2_sq = jnp.sum(g * g)
    norm_l1 = jnp.sum(jnp.abs(g)) / (len(g) + 1) if len(g) > 0 else 1e-15
    norm_inf = jnp.max(jnp.abs(g)) if len(g) > 0 else 1e-15

    # Avoid division by zero
    return norm_l2_sq / (norm_l1 * norm_inf)

# Gradient computation using JAX
def compute_c2_gradient_jax_manual(f_vals):
    """Manual gradient computation from JAX for robustness"""
    try:
        f = jnp.array(f_vals, dtype=jnp.float32)
        grad_func = grad(compute_c2_jax)
        gradients = grad_func(f)
        return np.array(gradients)
    except Exception:
        return np.zeros_like(f_vals)

def generate_mathematical_pattern(n):
    """
    Generate step function patterns based on mathematical formulas
    These are designed to produce desirable autoconvolution properties
    """
    pattern = np.zeros(n)
    
    # Pattern 1: Multi-bump Gaussian pattern
    x = np.linspace(-1, 1, n)
    for i in range(3):
        center = -0.5 + i * 0.5
        width = 0.2 + 0.1 * np.random.random()
        height = 0.8 + 0.4 * np.random.random()
        pattern += height * np.exp(-((x - center)**2) / (2 * width**2))
    
    # Pattern 2: Sine-modulated pattern with periodicity
    period_mod = 0.5 + 0.5 * np.random.random() 
    pattern += 0.5 * np.sin(period_mod * np.pi * x) + 0.7
    
    # Pattern 3: Asymmetric pattern with sharp transitions
    asymmetry = 0.2 + 0.3 * np.random.random()
    for i in range(n):
        if i < n // 2:
            pattern[i] += 0.3 + 0.2 * np.random.random()
        else:
            pattern[i] += 0.7 + 0.3 * np.random.random() * asymmetry
    
    # Ensure non-negativity and normalize to meaningful scale
    pattern = np.maximum(pattern, 0.0)
    if np.sum(pattern) > 0:
        pattern = pattern * n / np.sum(pattern)
    
    return pattern.tolist()

def generate_multi_scale_patterns(n):
    """Generate diverse patterns with different structural properties"""
    patterns = []
    
    # Uniform pattern
    patterns.append([1.0] * n)
    
    # Alternating pattern
    pattern = []
    for i in range(n):
        if i % 2 == 0:
            pattern.append(1.0 + 0.2 * np.random.random())
        else:
            pattern.append(0.3 + 0.2 * np.random.random())
    patterns.append(pattern)
    
    # Multi-scale mathematical pattern
    patterns.append(generate_mathematical_pattern(n))
    
    # Bell-shaped pattern with center peak
    x = np.linspace(-1, 1, n)
    pattern = 1.0 + 0.5 * np.exp(-8 * (x**2))
    pattern = np.maximum(pattern, 0.0)
    if np.sum(pattern) > 0:
        pattern = pattern * n / np.sum(pattern)
    patterns.append(pattern.tolist())
    
    # Randomized version
    pattern = [1.0 + 0.5 * np.random.random() for _ in range(n)]
    patterns.append(pattern)
    
    # Smoothed version with low-pass filter
    pattern = np.array([1.0 + 0.5 * np.random.random() for _ in range(n)])
    # Apply simple smoothing
    kernel = np.array([0.2, 0.6, 0.2])  # Simple moving average kernel
    smoothed = np.convolve(pattern, kernel, mode='same')
    patterns.append(smoothed.tolist())
    
    return patterns

def gradient_guided_evolutionary_optimization(initial_population, max_generations):
    """
    Novel hybrid evolutionary approach combining population-based search with gradient guidance
    """
    # Track best solutions and adaptation
    best_fitness_history = []
    patience_counter = 0
    max_patience = 15
    current_pop_size = POPULATION_SIZE_BASE
    
    # Start with initial population
    population = [ind.copy() for ind in initial_population]
    
    # Store best solution so far
    best_individual = max(population, key=compute_c2_score_sparse)
    best_fitness = compute_c2_score_sparse(best_individual)
    
    # Track convergence
    last_improvement = 0
    generation_counter = 0
    
    for generation in range(max_generations):
        generation_counter += 1
        
        # Evaluate all individuals
        fitnesses = [compute_c2_score_sparse(ind) for ind in population]
        
        # Track best in this generation
        gen_best_idx = np.argmax(fitnesses)
        gen_best_individual = population[gen_best_idx]
        gen_best_fitness = fitnesses[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual.copy()
            last_improvement = generation
        
        # Update history
        best_fitness_history.append(best_fitness)
        
        # Check for convergence
        if len(best_fitness_history) >= 10:
            if best_fitness_history[-1] - best_fitness_history[-10] < 1e-8:
                patience_counter += 1
            else:
                patience_counter = 0
                
            if patience_counter >= max_patience:
                # Increase population size to escape local minima
                current_pop_size = min(current_pop_size * 2, 100)
                patience_counter = 0
        
        # Create new population
        new_population = []
        
        # Elitism: keep top 20% 
        elite_count = max(1, int(0.2 * current_pop_size))
        sorted_indices = np.argsort(fitnesses)[::-1]
        elite_individuals = [population[i] for i in sorted_indices[:elite_count]]
        new_population.extend(elite_individuals)
        
        # Fill rest with offspring
        while len(new_population) < current_pop_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            
            # Clone parent
            parent = population[winner_index].copy()
            
            # Apply crossover and mutation
            if np.random.random() < CROSSOVER_PROB:
                # Simple uniform crossover
                other_parent = population[np.random.randint(len(population))].copy()
                for i in range(len(parent)):
                    if np.random.random() < 0.5:
                        parent[i] = other_parent[i]
            
            # Mutation with gradient guidance
            if np.random.random() < MUTATION_RATE:
                # Hybrid mutation: random perturbation + gradient direction
                for i in range(len(parent)):
                    if np.random.random() < 0.15:
                        # Small random perturbation
                        noise = np.random.normal(0, 0.05)
                        parent[i] = max(0, parent[i] + noise)
                        
                        # Occasionally incorporate gradient direction
                        if np.random.random() < 0.3:
                            try:
                                # Get current gradient
                                grad_val = compute_c2_gradient_jax_manual(parent)
                                parent[i] = max(0, parent[i] + 0.01 * grad_val[i])
                            except:
                                pass
            
            new_population.append(parent)
        
        # Replace population
        population = new_population
        
        # Early exit if time limit exceeded
        if time.time() - start_time > MAX_TIME_SECONDS * 0.95:
            break
    
    return best_individual, best_fitness

def advanced_local_refinement(solution, max_iterations=50):
    """
    Advanced local refinement using gradient ascent with adaptive learning rate
    """
    current_solution = solution.copy()
    current_fitness = compute_c2_score_sparse(current_solution)
    
    # Use simple gradient ascent with adaptive learning rate
    learning_rate = 0.01
    patience = 0
    max_patience = 10
    
    for iteration in range(max_iterations):
        try:
            # Compute gradients
            gradients = compute_c2_gradient_jax_manual(current_solution)
            
            # Update with gradient ascent
            new_solution = [max(0, x + learning_rate * grad) for x, grad in zip(current_solution, gradients)]
            
            # Evaluate new solution
            new_fitness = compute_c2_score_sparse(new_solution)
            
            if new_fitness > current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness
                patience = 0
                # Reduce learning rate slowly for stability
                learning_rate *= 0.95
            else:
                patience += 1
                # If no improvement, reduce learning rate more aggressively
                learning_rate *= 0.8
                if patience > max_patience:
                    break
                    
            # Prevent very small learning rates
            if learning_rate < 1e-8:
                break
                
        except Exception:
            # If gradient computation fails, do simple local search
            try:
                # Simple local perturbation
                new_solution = current_solution.copy()
                for i in range(len(new_solution)):
                    if np.random.random() < 0.1:
                        new_solution[i] = max(0, new_solution[i] + np.random.normal(0, 0.01))
                
                new_fitness = compute_c2_score_sparse(new_solution)
                if new_fitness > current_fitness:
                    current_solution = new_solution
                    current_fitness = new_fitness
            except:
                break
    
    return current_solution, current_fitness

def robust_optimization_pipeline():
    """
    Main optimization pipeline with robust initialization and multi-stage refinement
    """
    global start_time
    start_time = time.time()
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    best_overall_fitness = 0.0
    best_overall_solution = None
    
    # Multi-start approach with varying population configurations
    for start_num in range(NUM_STARTS):
        # Vary population size and problem size for diversity
        pop_size = POPULATION_SIZE_BASE + (start_num % 4) * 10
        if start_num < NUM_STARTS // 2:
            n_steps = random.randint(400, 800)
        else:
            n_steps = random.randint(600, 1200)
        
        # Generate diverse initial population
        initial_patterns = generate_multi_scale_patterns(n_steps)
        initial_population = []
        
        # Add noise to make solutions diverse
        for pattern in initial_patterns:
            noisy_pattern = [max(0.0, x + np.random.normal(0, 0.02)) for x in pattern]
            initial_population.append(noisy_pattern)
        
        # Perform gradient-guided evolutionary optimization
        try:
            final_solution, final_fitness = gradient_guided_evolutionary_optimization(
                initial_population, max_generations=MAX_GENERATIONS
            )
            
            # Apply local refinement
            refined_solution, refined_fitness = advanced_local_refinement(
                final_solution, max_iterations=30
            )
            
            if refined_fitness > best_overall_fitness:
                best_overall_fitness = refined_fitness
                best_overall_solution = refined_solution.copy()
                
        except Exception as e:
            continue
        
        # Check time limit
        if time.time() - start_time > MAX_TIME_SECONDS * 0.95:
            break
    
    # If no good solution found, fallback to mathematical pattern
    if best_overall_solution is None:
        n_steps = 800
        best_overall_solution = generate_mathematical_pattern(n_steps)
        best_overall_fitness = compute_c2_score_sparse(best_overall_solution)
    
    return best_overall_solution, best_overall_fitness

def construct_function() -> list[float]:
    """
    Main function to construct step-function with high C2 value.
    Implements a novel hybrid evolutionary-gradient approach.
    """
    try:
        # Run the robust optimization pipeline
        final_solution, final_fitness = robust_optimization_pipeline()
        
        # Ensure we return a reasonable-sized list
        if len(final_solution) < 50:
            final_solution.extend([1.0] * (50 - len(final_solution)))
        elif len(final_solution) > 10000:
            final_solution = final_solution[:10000]
            
        return final_solution
        
    except Exception as e:
        # Fallback to simple uniform distribution
        warnings.warn(f"Fallback due to error: {str(e)}")
        return [1.0] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")