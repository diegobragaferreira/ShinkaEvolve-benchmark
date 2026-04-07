# EVOLVE-BLOCK-START

import numpy as np
from scipy import fftpack
from scipy.optimize import differential_evolution, minimize
import random
import time
from typing import List, Tuple
from numba import njit, prange
import heapq

@njit(parallel=True)
def compute_sparse_convolution(f_vals):
    """
    Compute autoconvolution using FFT for efficiency
    This uses the mathematical property: f*f = IDFT(DFT(f) * DFT(f))
    """
    n = len(f_vals)
    
    # Pad to power of 2 for efficient FFT
    padded_n = 1
    while padded_n < 2*n - 1:
        padded_n <<= 1
    
    # Zero-padding
    f_padded = np.zeros(padded_n)
    f_padded[:n] = f_vals
    
    # FFT-based convolution
    f_fft = fftpack.fft(f_padded)
    g_fft = f_fft * f_fft.conj()  # Element-wise multiplication
    g_padded = fftpack.ifft(g_fft).real
    
    # Extract the valid convolution part
    # For f*f, we get 2*n-1 elements centered around index n-1
    g = g_padded[n-1:n-1+(2*n-1)]
    
    return g

@njit
def compute_convolution_norms_sparse(f_vals):
    """
    Compute norms using the sparse FFT-based convolution
    """
    n = len(f_vals)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Compute convolution using FFT
    g = compute_sparse_convolution(f_vals)
    
    # Compute norms
    g_abs = np.abs(g)
    
    # L2 norm squared using trapezoidal-like integration
    # For simplicity, we treat it as discrete sum weighted by dx
    dx = 1.0 / n  # Approximate step size
    g2_sq = np.sum(g_abs**2) * dx
    
    # L1 norm
    g1 = np.sum(g_abs) * dx
    
    # L-infinity norm
    g_inf = np.max(g_abs)
    
    return g2_sq, g1, g_inf

@njit
def compute_c2_sparse(f_vals):
    """
    Compute C2 using sparse FFT-based approach
    """
    g2_sq, g1, g_inf = compute_convolution_norms_sparse(f_vals)
    
    if g1 <= 1e-15 or g_inf <= 1e-15:
        return 0.0
    
    return g2_sq / (g1 * g_inf)

def construct_sparse_initial_function(n_steps):
    """
    Construct initial function with better mathematical properties
    Uses multi-scale pattern that naturally promotes flat convolution
    """
    # Create multi-scale pattern designed to generate flatter convolution results
    x = np.linspace(0, 1, n_steps)
    
    # Base multi-scale oscillation that avoids sharp peaks
    base_pattern = (
        0.5 * np.sin(2 * np.pi * x) +
        0.3 * np.sin(4 * np.pi * x) +
        0.2 * np.sin(8 * np.pi * x) +
        0.4
    )
    
    # Add structured variation to promote good convolution properties
    # Create several localized "bumps" that interact well in convolution
    bumps = np.zeros(n_steps)
    bump_positions = [n_steps//4, n_steps//2, 3*n_steps//4]
    bump_width = n_steps // 10
    
    for pos in bump_positions:
        if pos < n_steps:
            # Gaussian bump
            bump = np.exp(-((np.arange(n_steps) - pos)**2) / (2 * (bump_width/3)**2))
            bumps += bump * 0.3
    
    # Combine base pattern with bumps
    combined = base_pattern + bumps
    
    # Normalize and clip to [0, 1]
    combined = np.clip(combined, 0, 1)
    
    # Add some controlled noise for better exploration
    noise_amplitude = 0.05
    noise = np.random.normal(0, noise_amplitude, n_steps)
    combined = np.clip(combined + noise, 0, 1)
    
    # Ensure some minimum variation to prevent trivial solutions
    if np.std(combined) < 0.01:
        combined = np.ones_like(combined) * 0.5
    
    return combined.tolist()

def sparse_adaptive_optimization():
    """
    Main optimization using sparse FFT and hybrid approach
    """
    # Initialize with multi-scale pattern
    n_steps = 2000  # Larger for better resolution
    initial_pop_size = 20
    max_generations = 25
    
    # Generate diverse initial population
    population = []
    for i in range(initial_pop_size):
        # Mix of structured patterns and random variation
        if i % 3 == 0:
            # Structured initial function
            individual = construct_sparse_initial_function(n_steps)
        elif i % 3 == 1:
            # Simple geometric
            x = np.linspace(0, 1, n_steps)
            individual = np.clip(0.5 + 0.3 * np.sin(4 * np.pi * x), 0, 1).tolist()
        else:
            # Random with constraints
            individual = [random.uniform(0, 1) for _ in range(n_steps)]
            individual = np.clip(np.array(individual), 0, 1).tolist()
        
        population.append(individual)
    
    # Track best solution
    best_individual = None
    best_c2 = -1
    generation = 0
    
    # Evolutionary loop
    for generation in range(max_generations):
        # Evaluate population with sparse method
        fitness_scores = []
        for individual in population:
            c2 = compute_c2_sparse(individual)
            fitness_scores.append(c2)
            if c2 > best_c2:
                best_c2 = c2
                best_individual = individual.copy()
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Print progress
        if generation % 5 == 0:
            print(f"Generation {generation}: Best C2 = {best_c2:.6f}")
        
        # Create new population
        new_population = []
        elite_count = 3
        
        # Elitism
        for i in range(elite_count):
            new_population.append(sorted_population[i].copy())
        
        # Generate offspring
        while len(new_population) < initial_pop_size:
            # Tournament selection with adaptive size
            tourn_size = min(5, len(sorted_population)//2)
            parent1 = tournament_selection(sorted_population, sorted_fitness, tourn_size)
            parent2 = tournament_selection(sorted_population, sorted_fitness, tourn_size)
            
            # Crossover with adaptive rate
            if random.random() < 0.7:
                # Uniform crossover with bias toward better parent
                child1, child2 = adaptive_uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutate children
            mutate_individual_sparse(child1, 0.3)
            mutate_individual_sparse(child2, 0.3)
            
            new_population.extend([child1, child2])
        
        # Trim to exact size
        population = new_population[:initial_pop_size]
        
        # Local refinement of best solution occasionally
        if generation % 4 == 0 and generation > 0:
            # Use local optimization on top solutions
            for i in range(min(3, len(population))):
                if random.random() < 0.5:
                    refined = local_refinement(population[i])
                    refined_c2 = compute_c2_sparse(refined)
                    if refined_c2 > compute_c2_sparse(population[i]):
                        population[i] = refined
    
    return best_individual

def tournament_selection(population, fitness_scores, tournament_size):
    """Tournament selection with adaptive size"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index].copy()

def adaptive_uniform_crossover(parent1, parent2):
    """Adaptive uniform crossover that favors better parent traits"""
    child1 = []
    child2 = []
    
    # Determine bias based on fitness difference
    fitness_diff = 1.0  # Simplified - in practice would compare fitness
    
    for i in range(len(parent1)):
        # Bias towards better parent if one is significantly better
        if random.random() < 0.5 + 0.3 * (fitness_diff > 0.1):
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    
    return child1, child2

def mutate_individual_sparse(individual, mutation_rate):
    """Mutate individual with enhanced sparse mutation"""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Adaptive mutation strength
            strength = 0.1 * (individual[i] if individual[i] > 0 else 0.1)
            
            # Add noise
            noise = np.random.normal(0, strength)
            new_value = individual[i] + noise
            
            # Ensure non-negativity
            individual[i] = max(0, new_value)

def local_refinement(individual):
    """Apply local refinement to improve solution quality"""
    # Convert to numpy for easier handling
    x = np.array(individual)
    
    # Simple gradient ascent using finite differences
    step_size = 0.01
    tolerance = 1e-6
    max_iter = 50
    
    for _ in range(max_iter):
        old_c2 = compute_c2_sparse(individual.tolist())
        
        # Estimate gradient using finite differences
        grad = np.zeros_like(x)
        eps = 1e-5
        
        for i in range(len(x)):
            # Perturb dimension i
            x_plus = x.copy()
            x_plus[i] = max(0, x[i] + eps)
            c2_plus = compute_c2_sparse(x_plus.tolist())
            
            x_minus = x.copy()
            x_minus[i] = max(0, x[i] - eps)
            c2_minus = compute_c2_sparse(x_minus.tolist())
            
            grad[i] = (c2_plus - c2_minus) / (2 * eps)
        
        # Update using gradient
        x_new = x + step_size * grad
        
        # Ensure non-negativity
        x_new = np.maximum(x_new, 0)
        
        # Check for improvement
        new_c2 = compute_c2_sparse(x_new.tolist())
        if new_c2 > old_c2:
            x = x_new
        else:
            break
    
    return x.tolist()

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    np.random.seed(42)  # For reproducibility
    
    # Try sparse FFT optimization
    try:
        result = sparse_adaptive_optimization()
        if result is not None:
            return result
    except Exception as e:
        print(f"Sparse optimization failed: {e}")
    
    # Fall back to basic approach
    n_steps = 500
    f_values = construct_sparse_initial_function(n_steps)
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")