# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize
from numba import njit
import random
from typing import List, Tuple
import time

@njit
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

@njit
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

def generate_pattern_family(size: int, pattern_type: str) -> List[float]:
    """Generate different structured patterns based on pattern type."""
    if pattern_type == "geometric":
        # Geometric decay with slight randomization
        base_vals = np.geomspace(1, 0.01, num=size // 2)
        if len(base_vals) < size:
            remaining = size - len(base_vals)
            tail_vals = np.geomspace(0.01, 0.001, num=remaining)
            base_vals = np.concatenate([base_vals, tail_vals])
        base_vals = np.clip(base_vals, 0, 100)
        return base_vals.tolist()
        
    elif pattern_type == "peaks":
        # Concentrated peaks
        vals = np.zeros(size)
        peak_positions = [size // 4, size // 2, 3 * size // 4]
        for pos in peak_positions:
            if pos < size:
                vals[pos] = 5.0
        # Add some geometric decay around peaks
        for i in range(size):
            dist_from_peak = min([abs(i - pos) for pos in peak_positions])
            vals[i] *= max(0, 1 - dist_from_peak / (size // 8))
        vals = np.clip(vals, 0, 100)
        return vals.tolist()
        
    elif pattern_type == "sawtooth":
        # Alternating high-low pattern
        vals = np.zeros(size)
        for i in range(size):
            vals[i] = 1.0 if i % 2 == 0 else 0.1
        return vals.tolist()
        
    elif pattern_type == "bimodal":
        # Two distinct modes
        mid = size // 2
        vals = np.ones(size)
        vals[:mid] = np.linspace(0.5, 3.0, mid)
        vals[mid:] = np.linspace(3.0, 0.5, size - mid)
        return vals.tolist()
        
    elif pattern_type == "uniform":
        # Simple uniform values
        return [1.0] * size
        
    else:
        # Default random pattern
        return [random.uniform(0.1, 2.0) for _ in range(size)]

def adaptive_local_optimize(initial_vals: List[float], max_iter: int = 300) -> List[float]:
    """Perform adaptive local optimization on the initial pattern."""
    def objective(f_vals):
        return -calculate_c2(f_vals)
    
    # Try multiple optimization methods
    results = []
    
    # Nelder-Mead optimization
    try:
        result = minimize(objective, initial_vals, method='Nelder-Mead', 
                         options={'maxiter': max_iter, 'xtol': 1e-6})
        if result.success:
            results.append(result.x)
    except:
        pass
    
    # L-BFGS optimization  
    try:
        result = minimize(objective, initial_vals, method='L-BFGS-B', 
                         options={'maxiter': max_iter})
        if result.success:
            results.append(result.x)
    except:
        pass
    
    # If we got results, return the best one
    if results:
        best_result = min(results, key=lambda x: -calculate_c2(x))
        return np.maximum(best_result, 0.0).tolist()
    
    return initial_vals

def adaptive_evolutionary_search() -> List[float]:
    """Main adaptive evolutionary search algorithm."""
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initial parameters
    max_generations = 60  # Increased generations for better search
    population_size = 15  # Larger population for better diversity
    pattern_types = ["geometric", "peaks", "sawtooth", "bimodal", "uniform"]
    
    # Start with diverse initial population
    population = []
    for i in range(population_size):
        pattern_type = pattern_types[i % len(pattern_types)]
        size = np.random.randint(300, 1800)  # Wider range for better exploration
        individual = generate_pattern_family(size, pattern_type)
        population.append(individual)
    
    best_overall = None
    best_c2_score = -float('inf')
    start_time = time.time()
    
    # Evolutionary cycle
    for generation in range(max_generations):
        # Check time limit
        if time.time() - start_time > 85:  # Leave 5 seconds for cleanup
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
        
        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best C2 = {best_c2_score:.6f}")
        
        # Selection: keep top 50% 
        sorted_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        selected_indices = sorted_indices[:population_size // 2]
        selected_population = [population[i] for i in selected_indices]
        
        # Create new population through crossover and mutation
        new_population = selected_population.copy()
        
        # Generate offspring
        while len(new_population) < population_size:
            # Select parents
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            
            # Uniform crossover (better than single point)
            offspring = []
            for i in range(min(len(parent1), len(parent2))):
                if random.random() < 0.5:
                    offspring.append(parent1[i])
                else:
                    offspring.append(parent2[i])
            
            # Ensure offspring has at least one element
            if len(offspring) == 0:
                offspring = [1.0]
            
            # Extend offspring to match max length if needed
            max_len = max(len(parent1), len(parent2))
            if len(offspring) < max_len:
                # Fill with values from first parent
                for i in range(len(offspring), max_len):
                    if i < len(parent1):
                        offspring.append(parent1[i])
                    else:
                        offspring.append(1.0)
            
            # Mutate offspring with adaptive strength
            mutation_strength = 0.3 * (1 - generation / max_generations) + 0.05  # Decreasing over time
            mutated_offspring = []
            
            for val in offspring:
                # Add noise with adaptive strength
                noise = np.random.normal(0, mutation_strength * max(1e-6, val))
                mutated_val = max(0, val + noise)
                mutated_offspring.append(mutated_val)
            
            # Occasionally replace with new pattern
            if random.random() < 0.15:  # Increased chance for pattern replacement
                new_pattern_type = random.choice(pattern_types)
                new_size = np.random.randint(300, 1800)
                mutated_offspring = generate_pattern_family(new_size, new_pattern_type)
            
            new_population.append(mutated_offspring)
        
        # Keep only the required population size
        population = new_population[:population_size]
        
        # Periodic local optimization of top individuals
        if generation % 4 == 0:  # More frequent local optimization
            for i in range(len(population[:4])):  # Optimize top 4
                population[i] = adaptive_local_optimize(population[i], max_iter=200)
    
    # Final optimization of the best individual found
    if best_overall:
        final_candidate = adaptive_local_optimize(best_overall, max_iter=300)
        final_score = calculate_c2(final_candidate)
        if final_score > best_c2_score:
            return final_candidate
        else:
            return best_overall
    
    # Fallback
    return [1.0] * 500

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value using adaptive pattern evolution."""
    try:
        return adaptive_evolutionary_search()
    except Exception as e:
        # Fallback to simple uniform function
        return [1.0] * 500

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
