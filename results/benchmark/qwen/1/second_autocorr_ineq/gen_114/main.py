# EVOLVE-BLOCK-START
import numpy as np
import time
from numba import jit
import random
from collections import deque

# Global constants
N_BINS = 1000
DOMAIN = [-0.25, 0.25]
STEP_WIDTH = (DOMAIN[1] - DOMAIN[0]) / N_BINS

@jit(nopython=True)
def compute_autoconvolution_numba(f_vals):
    """Compute autoconvolution using fast Numba implementation"""
    n = len(f_vals)
    if n == 0:
        return np.array([])
        
    # Convolution result has length 2*n-1
    g_len = 2 * n - 1
    g = np.zeros(g_len)

    # Compute convolution manually for efficiency
    for i in range(n):
        f_i = f_vals[i]
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_i * f_vals[j]

    # Trim to center portion (length n-1) - this is the actual autoconvolution
    offset = (n - 1) // 2
    g_trimmed = g[offset:(2*n-1)-offset]
    return g_trimmed

@jit(nopython=True)
def compute_c2_numba(g_vals):
    """Compute C2 value using fast Numba implementation"""
    if len(g_vals) == 0:
        return 0.0

    # Compute norms
    g_l2_sq = 0.0
    g_l1 = 0.0
    g_max = 0.0

    # For L2 norm squared (trapezoidal integration)
    for i in range(len(g_vals) - 1):
        val1 = g_vals[i]
        val2 = g_vals[i+1]
        h = STEP_WIDTH
        g_l2_sq += (h/3) * (val1*val1 + val1*val2 + val2*val2)

    # For L1 norm (sum of absolute values)
    for i in range(len(g_vals)):
        g_l1 += abs(g_vals[i])

    # For infinity norm (max absolute value)
    for i in range(len(g_vals)):
        if abs(g_vals[i]) > g_max:
            g_max = abs(g_vals[i])

    # Compute C2 with robust division
    if g_l1 > 1e-15 and g_max > 1e-15:
        c2 = g_l2_sq / (g_l1 * g_max)
    else:
        c2 = 0.0

    return c2

def generate_initial_population(size, dim):
    """Generate initial population with multi-scale patterns"""
    population = []
    for _ in range(size):
        # Create pattern with different characteristics
        pattern_type = random.choice(['gaussian', 'sine', 'mixed', 'spike'])
        
        if pattern_type == 'gaussian':
            # Gaussian-like pattern
            params = np.zeros(dim)
            center = dim // 2
            sigma = dim / 6
            for i in range(dim):
                params[i] = 1.0 * np.exp(-0.5 * ((i - center) / sigma) ** 2)
        elif pattern_type == 'sine':
            # Sine wave pattern
            params = np.zeros(dim)
            for i in range(dim):
                params[i] = 0.5 * np.sin(2 * np.pi * i / (dim / 4)) + 0.5
        elif pattern_type == 'mixed':
            # Mixed pattern
            params = np.zeros(dim)
            # Gaussian component
            center = dim // 2
            sigma = dim / 6
            for i in range(dim):
                params[i] += 1.0 * np.exp(-0.5 * ((i - center) / sigma) ** 2)
            # Sine component
            for i in range(dim):
                params[i] += 0.3 * np.sin(2 * np.pi * i / (dim / 6))
            # Random component
            np.random.seed(random.randint(1, 1000))
            params += np.random.random(dim) * 0.2
        else:  # spike
            # Spike pattern
            params = np.zeros(dim)
            peak_pos = random.randint(dim//4, 3*dim//4)
            params[peak_pos] = 2.0
            # Add some smoothing
            for i in range(max(0, peak_pos-5), min(dim, peak_pos+6)):
                params[i] = max(0, params[i] - abs(i - peak_pos) * 0.1)
        
        # Ensure non-negative
        params = np.maximum(params, 0)
        
        # Normalize to reasonable amplitude
        max_val = np.max(params)
        if max_val > 0:
            params = params / max_val * 1.5
            
        population.append(params.tolist())
    
    return population

def mutate_individual(individual, mutation_rate=0.1, neighborhood_size=3):
    """Mutate individual with neighborhood-based mutations"""
    mutated = individual.copy()
    dim = len(mutated)
    
    # Select positions to mutate
    num_mutations = int(mutation_rate * dim)
    positions = random.sample(range(dim), min(num_mutations, dim))
    
    for pos in positions:
        # Determine mutation type based on neighborhood
        left_bound = max(0, pos - neighborhood_size)
        right_bound = min(dim, pos + neighborhood_size)
        
        # Get neighborhood average
        neighborhood_avg = np.mean(mutated[left_bound:right_bound])
        
        # Mutate with different strategies
        mutation_type = random.choices(['local', 'global', 'smooth'], weights=[0.4, 0.3, 0.3])[0]
        
        if mutation_type == 'local':
            # Local adjustment based on neighboring values
            neighbor_avg = np.mean([mutated[max(0, pos-1)], mutated[min(dim-1, pos+1)]])
            mutated[pos] = max(0, mutated[pos] + random.uniform(-0.2, 0.2) * neighbor_avg)
        elif mutation_type == 'global':
            # Global random adjustment
            mutated[pos] = max(0, mutated[pos] + random.uniform(-0.5, 0.5))
        else:  # smooth
            # Smooth adjustment towards neighborhood average
            mutated[pos] = max(0, neighborhood_avg + random.uniform(-0.3, 0.3) * neighborhood_avg)
    
    return mutated

def crossover(parent1, parent2, crossover_rate=0.8):
    """Single-point crossover with optional recombination"""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    dim = len(parent1)
    point = random.randint(1, dim - 1)
    
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    
    return child1, child2

def evaluate_fitness(individual):
    """Evaluate fitness with primary focus on C2 and secondary on smoothness"""
    try:
        # Clip negative values
        f_vals = np.clip(individual, 0, None)
        
        if len(f_vals) == 0:
            return 0.0
            
        # Compute autoconvolution
        g_vals = compute_autoconvolution_numba(f_vals)
        
        if len(g_vals) == 0:
            return 0.0
            
        # Compute C2
        c2 = compute_c2_numba(g_vals)
        
        # Secondary fitness metric: smoothness (penalize sharp changes)
        smoothness_penalty = 0.0
        if len(f_vals) > 2:
            # Measure variation in adjacent differences
            diffs = np.diff(f_vals)
            smoothness_penalty = -0.01 * np.std(diffs)
        
        # Combined fitness
        fitness = c2 + smoothness_penalty
        
        return fitness
    except Exception as e:
        return -1e10  # Large penalty for invalid results

def adaptive_evolutionary_search(dim, max_time=85):
    """Main evolutionary optimization routine"""
    start_time = time.time()
    
    # Phase 1: Global exploration with large population
    pop_size = max(20, min(100, dim // 5))
    population = generate_initial_population(pop_size, dim)
    
    best_fitness = float('-inf')
    best_individual = None
    
    # Track recent fitness improvements for adaptive control
    fitness_history = deque(maxlen=10)
    
    # Evolutionary loop
    generation = 0
    max_generations = 50
    
    while generation < max_generations and (time.time() - start_time < max_time - 5):
        # Evaluate fitness
        fitness_scores = []
        for ind in population:
            fit = evaluate_fitness(ind)
            fitness_scores.append((fit, ind))
        
        # Sort by fitness
        fitness_scores.sort(reverse=True)
        
        # Update best
        current_best_fit, current_best_ind = fitness_scores[0]
        if current_best_fit > best_fitness:
            best_fitness = current_best_fit
            best_individual = current_best_ind.copy()
        
        # Record fitness history
        fitness_history.append(current_best_fit)
        
        # Selection: keep top 50%
        selected_count = max(10, pop_size // 2)
        selected = [ind for _, ind in fitness_scores[:selected_count]]
        
        # Create new population through reproduction
        new_population = selected.copy()
        
        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual)
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate_individual(child1)
            child2 = mutate_individual(child2)
            
            new_population.extend([child1, child2])
            
        # Trim to population size
        population = new_population[:pop_size]
        
        # Adaptive population size adjustment based on improvement
        if len(fitness_history) >= 5:
            recent_improvement = fitness_history[-1] - fitness_history[0]
            if recent_improvement < 0.001 and pop_size > 10:
                pop_size = max(10, pop_size - 2)
            elif recent_improvement > 0.01 and pop_size < 100:
                pop_size = min(100, pop_size + 2)
        
        generation += 1
    
    # Phase 2: Local refinement with focused search
    if best_individual is not None and (time.time() - start_time < max_time - 5):
        # Apply focused gradient-like refinement
        refined_individual = best_individual.copy()
        
        # Apply multiple rounds of local fine-tuning
        for round_num in range(5):
            if time.time() - start_time > max_time - 5:
                break
                
            # Create small perturbations around best individual
            local_neighbors = []
            for _ in range(10):
                neighbor = refined_individual.copy()
                for i in range(len(neighbor)):
                    # Small random adjustments
                    if random.random() < 0.3:
                        neighbor[i] = max(0, neighbor[i] + random.uniform(-0.1, 0.1))
                local_neighbors.append(neighbor)
            
            # Evaluate neighbors and select best
            neighbor_fitnesses = [(evaluate_fitness(n), n) for n in local_neighbors]
            neighbor_fitnesses.sort(reverse=True)
            
            if neighbor_fitnesses and neighbor_fitnesses[0][0] > evaluate_fitness(refined_individual):
                refined_individual = neighbor_fitnesses[0][1]
        
        return refined_individual
    
    return best_individual if best_individual is not None else [1.0] * dim

def construct_function() -> list[float]:
    """Function to construct step-function with high C2 value."""
    start_time = time.time()
    
    # Multi-start approach with different strategies
    best_c2 = -np.inf
    best_params = None
    
    # Try different dimensions for exploration
    dimensions = [200, 400, 600, 800, 1000]
    
    for dim in dimensions:
        if time.time() - start_time > 85:
            break
            
        try:
            # Run evolutionary optimization
            params = adaptive_evolutionary_search(dim, max_time=85)
            
            # Evaluate final solution
            f_vals = np.clip(params, 0, None)
            if len(f_vals) > 0:
                g_vals = compute_autoconvolution_numba(f_vals)
                c2 = compute_c2_numba(g_vals)
                
                if c2 > best_c2:
                    best_c2 = c2
                    best_params = params.copy()
                    
        except Exception as e:
            continue
    
    # If no valid parameters found, return default
    if best_params is None:
        return [0.5] * 100
    
    # Final check and conversion to list
    final_f_vals = np.clip(best_params, 0, None)
    return final_f_vals.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")