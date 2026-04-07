# EVOLVE-BLOCK-START

import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
from multiprocessing import Pool
import random
import time
from functools import lru_cache
import copy

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

# Global cache for fitness evaluations
fitness_cache = {}

def compute_autocorrelation_constant(sequence):
    """Compute the autocorrelation constant C₁ for a sequence using optimized FFT."""
    n = len(sequence)
    if n == 0:
        return float('inf')
        
    sum_seq = np.sum(sequence)
    if sum_seq < 1e-10:
        return float('inf')
    
    # Pad sequence and perform FFT-based convolution
    padded_seq = np.pad(sequence, (0, n-1), 'constant', constant_values=0)
    conv_result = np.real(ifft(fft(padded_seq) * np.conj(fft(padded_seq))))
    
    max_conv = np.max(conv_result[:2*n-1])
    
    # Compute C₁ = 2n * max(b) / (sum(a))^2
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    
    return c1

@lru_cache(maxsize=10000)
def cached_compute_c1(sequence_tuple):
    """Cached version of C₁ computation."""
    sequence = list(sequence_tuple)
    return compute_autocorrelation_constant(sequence)

def evaluate_fitness(sequence):
    """Evaluate the fitness of a sequence (inverse of C₁)."""
    try:
        # Check cache first
        seq_tuple = tuple(sequence)
        if seq_tuple in fitness_cache:
            return fitness_cache[seq_tuple]
        
        c1 = cached_compute_c1(seq_tuple)
        if c1 == float('inf'):
            fitness = 0.0
        else:
            fitness = 1.0 / c1
            
        fitness_cache[seq_tuple] = fitness
        return fitness
    except Exception:
        return 0.0

def generate_structured_sequence(n):
    """Generate a structured sequence to improve convergence."""
    # Use a combination of sine waves and Gaussian-like decay
    base_seq = []
    for i in range(n):
        # Sine component for periodicity
        sine_component = np.sin(i * np.pi / n)
        # Gaussian-like decay for localization
        gaussian_component = np.exp(-0.5 * ((i - n/2) / (n/4)) ** 2)
        # Combine components
        combined = abs(sine_component) * gaussian_component
        base_seq.append(combined)
    
    # Normalize and scale
    total = sum(base_seq)
    if total > 0:
        base_seq = [x * 100 / total for x in base_seq]
    else:
        base_seq = [1.0] * n
        
    return base_seq

def generate_random_valid_sequence(n):
    """Generate a random valid sequence."""
    # Ensure at least one element is positive
    seq = np.random.rand(n) * 1000  # Scale up for better variation
    # Clip to reasonable range
    seq = np.clip(seq, 0, 1000)
    # Ensure sum is not too small
    if np.sum(seq) < 0.01:
        seq[0] = 0.01
    return seq.tolist()

def get_good_direction_to_move_into(sequence):
    """Returns the direction to move into the sequence using gradient estimates."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    
    # Avoid division by zero
    if sum_sequence < 1e-10:
        return None
        
    # Normalize the sequence
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    
    # Use FFT for faster convolution
    try:
        conv_result = np.real(ifft(fft(normalized_sequence, 2*n-1) *
                                   np.conj(fft(normalized_sequence, 2*n-1))))
        rhs = np.max(conv_result[:2*n-1])
    except Exception:
        return None
    
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
        
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None
        
    # Normalize the result
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]
    
    # Move in the direction of improvement
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    
    # Clip to prevent extreme values
    new_sequence = [max(0, min(x, 1000)) for x in new_sequence]
    return new_sequence

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []

    # Precompute the convolution constraint matrix efficiently
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    try:
        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')
        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except Exception:
        return None

def adaptive_evolution_step(population, fitness_scores, generation, max_generations):
    """Perform an adaptive evolution step."""
    n = len(population)
    
    # Adaptive mutation rate
    mutation_rate = max(0.05, 0.1 - generation / max_generations * 0.05)
    
    # Sort population by fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]
    
    # Keep top performers
    elite_size = max(1, n // 4)
    elite_indices = sorted_indices[:elite_size]
    new_population = [copy.deepcopy(population[i]) for i in elite_indices]
    
    # Generate offspring
    while len(new_population) < n:
        # Tournament selection
        parent1 = population[random.choice(elite_indices)]
        parent2 = population[random.choice(elite_indices)]
        
        # Crossover (uniform)
        child = []
        for a, b in zip(parent1, parent2):
            if random.random() < 0.5:
                child.append(a)
            else:
                child.append(b)
        
        # Mutation
        if random.random() < mutation_rate:
            for i in range(len(child)):
                if random.random() < 0.1:
                    child[i] *= random.uniform(0.9, 1.1)
        
        # Ensure non-negative and reasonable values
        child = [max(0, x) for x in child]
        child = [min(x, 1000) for x in child]
        
        # Make sure it's valid
        if sum(child) > 0.01:
            new_population.append(child)
    
    return new_population[:n]

def evolve_sequence(population_size=50, generations=20, stagnation_threshold=5):
    """Evolve a population of sequences to find optimal ones."""
    start_time = time.time()
    
    # Initial population with mixed strategies
    population = []
    for _ in range(population_size):
        n = random.randint(100, 1000)
        
        # Use structured initialization for some individuals
        if random.random() < 0.4:
            seq = generate_structured_sequence(n)
        else:
            seq = generate_random_valid_sequence(n)
        population.append(seq)
    
    best_fitness = 0
    best_individual = None
    stagnation_count = 0
    
    for gen in range(generations):
        if time.time() - start_time > 160:  # Leave buffer
            break
            
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append(fitness)
            
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = copy.deepcopy(population[max_fitness_idx])
            stagnation_count = 0
        else:
            stagnation_count += 1
            
        # Check for stagnation
        if stagnation_count > stagnation_threshold:
            break
            
        # Adaptive evolution
        population = adaptive_evolution_step(population, fitness_scores, gen, generations)
        
    return best_individual, best_fitness

def search_for_best_sequence():
    """Function to search for the best coefficient sequence."""
    start_time = time.time()
    
    # Try evolutionary approach multiple times
    best_sequence = None
    best_fitness = 0
    
    # Run evolution multiple times with different parameters
    for attempt in range(5):
        try:
            current_sequence, current_fitness = evolve_sequence(
                population_size=30 + attempt*10,
                generations=10 + attempt*3
            )
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_sequence = current_sequence
        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            continue
            
        if time.time() - start_time > 160:  # Leave 10 seconds for cleanup
            break
    
    # If no good sequence found, use a fallback
    if best_sequence is None:
        n = random.randint(100, 1000)
        best_sequence = generate_random_valid_sequence(n)
        
    # Final refinement using gradient-based method
    try:
        refined = get_good_direction_to_move_into(best_sequence)
        if refined is not None:
            refined_fitness = evaluate_fitness(refined)
            if refined_fitness > best_fitness:
                best_sequence = refined
    except Exception:
        pass
        
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")