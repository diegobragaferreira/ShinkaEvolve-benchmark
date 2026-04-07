# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import random
import time
import heapq
from collections import OrderedDict
from functools import lru_cache
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class LRUCache:
    """LRU cache for storing previous evaluations."""
    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value

# Global cache instance
evaluation_cache = LRUCache()

def compute_autocorrelation_constant(sequence):
    """
    Compute C₁ for a given sequence using FFT for efficiency.
    C₁ = 2n * max(convolution) / (sum(sequence))^2
    """
    n = len(sequence)
    if n == 0:
        return float('inf')

    # Compute convolution using FFT for efficiency
    try:
        # Pad to next power of 2 for efficient FFT
        padded_len = 1 << (n - 1).bit_length()
        padded_seq = np.pad(sequence, (0, padded_len - n), 'constant')

        # FFT-based convolution
        fft_seq = fft(padded_seq)
        conv_fft = fft_seq * fft_seq.conj()
        conv_result = ifft(conv_fft).real[:2*n-1]

        # Check for numerical precision issues
        if np.any(np.isnan(conv_result)) or np.any(np.isinf(conv_result)):
            # If there are numerical issues, fall back to direct computation for small sequences
            if n < 100:
                # Direct convolution for small sequences
                conv_result = np.convolve(sequence, sequence, mode='full')[:2*n-1]
            else:
                # For larger sequences, adjust padding to ensure better numerical properties
                padded_len = 2 * n - 1
                padded_seq = np.pad(sequence, (0, padded_len - n), 'constant')
                fft_seq = fft(padded_seq)
                conv_fft = fft_seq * fft_seq.conj()
                conv_result = ifft(conv_fft).real[:2*n-1]

        max_conv = np.max(conv_result)
    except Exception:
        # Fallback to direct convolution if FFT fails
        conv_result = np.convolve(sequence, sequence, mode='full')[:2*n-1]
        max_conv = np.max(conv_result)

    sum_seq = np.sum(sequence)
    if sum_seq < 0.01:
        return float('inf')

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return c1

@lru_cache(maxsize=1000)
def cached_compute_c1(sequence_tuple):
    """Cached version to avoid recomputation."""
    return compute_autocorrelation_constant(list(sequence_tuple))

def evaluate_objective_cached(sequence):
    """
    Evaluate the objective function: -1/C₁ (we minimize this to maximize 1/C₁)
    Uses caching to avoid recomputation.
    """
    key = tuple(sequence)
    cached_result = evaluation_cache.get(key)
    if cached_result is not None:
        return cached_result
        
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        result = float('inf')  # Invalid solution
    else:
        result = -1.0 / c1  # Negative because we want to maximize 1/C₁
    
    evaluation_cache.put(key, result)
    return result

def generate_initial_sequence():
    """
    Generate a good initial random sequence with more structure.
    """
    # Try to make sequences that have some structure in them
    n = random.randint(100, 1000)
    
    # Use a combination of distributions to create structure
    choice = random.random()
    
    if choice < 0.25:
        # Power law distribution - heavy tail
        sequence = [random.expovariate(0.1) for _ in range(n)]
        # Normalize to prevent extreme values
        max_val = max(sequence)
        sequence = [x * 100.0 / max_val if max_val > 0 else 1.0 for x in sequence]
    elif choice < 0.5:
        # Uniform distribution with some peaks
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
        # Add a few peaks
        for i in range(min(10, len(sequence)//20)):
            peak_pos = random.randint(0, len(sequence)-1)
            sequence[peak_pos] = random.uniform(100.0, 1000.0)
    elif choice < 0.75:
        # Mixed distribution
        sequence = []
        for i in range(n):
            if random.random() < 0.7:
                sequence.append(random.uniform(0.1, 10.0))
            else:
                sequence.append(random.uniform(50.0, 100.0))
    else:
        # Create a sequence with specific pattern to avoid convolution maxima
        sequence = []
        for i in range(n):
            # Alternating pattern with small variations
            if i % 3 == 0:
                sequence.append(random.uniform(10.0, 50.0))
            elif i % 3 == 1:
                sequence.append(random.uniform(50.0, 100.0))
            else:
                sequence.append(random.uniform(100.0, 1000.0))

    return sequence

def adaptive_mutation(sequence, mutation_rate=0.1, generation=None):
    """Adaptive mutation operator designed to reduce convolution peaks."""
    mutated = sequence.copy()
    n = len(mutated)
    
    # Adaptive mutation rate: lower for longer sequences
    if n > 500:
        mutation_rate *= 0.5
    elif n < 200:
        mutation_rate *= 1.5

    # Calculate standard deviation for mutation scaling
    std_dev = np.std(mutated) if len(sequence) > 0 else 1.0
    mutation_scale = max(0.1, std_dev * 0.1)  # Scale mutation by sequence variability
    
    # Use a novel mutation type that considers neighbor relationships
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Differential evolution inspired mutation
            neighbor_indices = [max(0, i-1), min(n-1, i+1)]
            neighbors = [mutated[idx] for idx in neighbor_indices if idx != i]
            
            if neighbors:
                # Mutate towards neighbor average to promote smooth transitions
                neighbor_avg = sum(neighbors) / len(neighbors)
                mutated[i] = max(0.0, mutated[i] + random.gauss(0, mutation_scale) + 
                                0.3 * (neighbor_avg - mutated[i]))
            else:
                mutated[i] = max(0.0, mutated[i] + random.gauss(0, mutation_scale))
                
    return mutated

def differential_crossover(parent1, parent2, mutation_rate=0.5, generation=None):
    """Differential evolution inspired crossover."""
    n = len(parent1)
    offspring = []
    
    for i in range(n):
        if random.random() < mutation_rate:
            # Blend from both parents with preference to the first
            if random.random() < 0.7:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        else:
            # Keep original from parent1
            offspring.append(parent1[i])
    
    return offspring

def gradient_guided_local_search(sequence, max_iter=100):
    """
    Enhanced local search using gradient information.
    """
    n = len(sequence)
    
    # Bounds for optimization
    bounds = [(0.0, 1000.0) for _ in range(n)]
    
    # Constraint: sum must be ≥ 0.01
    def sum_constraint(x):
        return np.sum(x) - 0.01

    constraints = [{'type': 'ineq', 'fun': sum_constraint}]

    # Objective function to minimize
    def objective(x):
        return evaluate_objective_cached(tuple(x))

    # Try multiple optimization methods
    methods_to_try = ['SLSQP', 'L-BFGS-B']

    for method in methods_to_try:
        try:
            # Use smaller tolerance for faster convergence
            result = minimize(objective, sequence, method=method, bounds=bounds,
                            constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6})
            if result.success:
                return result.x.tolist()
        except Exception:
            continue

    # Fallback to direct method if optimization fails
    return sequence

def adaptive_sequence_length(sequence, target_ratio=0.8):
    """Dynamically adjust sequence length based on performance characteristics."""
    n = len(sequence)

    # If sequence is too long, consider truncation
    if n > 1000:
        # Keep most significant elements based on magnitude
        indices = np.argsort(sequence)[::-1][:n//2]
        new_sequence = [sequence[i] for i in sorted(indices)]
        return new_sequence

    # If sequence is too short, consider expansion
    if n < 50:
        # Expand with copies and slight mutations
        expanded = sequence.copy()
        for i in range(10):
            idx = random.randint(0, len(sequence)-1)
            expanded.append(max(0.0, sequence[idx] * (1 + random.uniform(-0.2, 0.2))))
        return expanded

    return sequence

def hierarchical_evolutionary_optimization(max_time_seconds=180):
    """
    Hierarchical evolutionary optimization using gradient guidance.
    """
    start_time = time.time()
    
    # Initial population generation
    population_size = 50
    population = [generate_initial_sequence() for _ in range(population_size)]
    
    # Evaluate initial population
    fitness_scores = []
    for seq in population:
        fitness = evaluate_objective_cached(tuple(seq))
        fitness_scores.append((seq, fitness))
    
    # Sort by fitness (lower is better)
    fitness_scores.sort(key=lambda x: x[1])
    
    # Track best solution globally
    global_best = fitness_scores[0][0]
    global_best_fitness = fitness_scores[0][1]
    
    # Main evolution loop
    for generation in range(100):  # Max generations
        if time.time() - start_time > max_time_seconds - 10:  # Leave buffer
            break

        # Keep top performers (elite)
        top_performers = [seq for seq, _ in fitness_scores[:10]]
        
        # Create new population
        new_population = top_performers[:]
        
        # Generate offspring using differential evolution-inspired operators
        while len(new_population) < population_size:
            # Select two parents
            parent1 = random.choice(top_performers)
            parent2 = random.choice(top_performers)
            
            # Apply differential crossover
            child = differential_crossover(parent1, parent2, mutation_rate=0.6)
            
            # Apply adaptive mutation
            child = adaptive_mutation(child, mutation_rate=0.15, generation=generation)
            
            # Apply sequence length adaptation
            child = adaptive_sequence_length(child)
            
            # Apply gradient-guided local search
            if random.random() < 0.3:  # 30% chance of local search
                child = gradient_guided_local_search(child, max_iter=50)
            
            new_population.append(child)
        
        # Evaluate new population
        fitness_scores = []
        for seq in new_population:
            fitness = evaluate_objective_cached(tuple(seq))
            fitness_scores.append((seq, fitness))
        
        # Sort population by fitness
        fitness_scores.sort(key=lambda x: x[1])
        
        # Update global best
        if fitness_scores[0][1] < global_best_fitness:
            global_best = fitness_scores[0][0]
            global_best_fitness = fitness_scores[0][1]
    
    # Final optimization of the best sequence
    final_best = gradient_guided_local_search(global_best, max_iter=200)
    
    return final_best

def search_for_best_sequence():
    """
    Main function to search for the best coefficient sequence.
    """
    random.seed(42)
    np.random.seed(42)
    
    # Use hierarchical evolutionary approach
    best_sequence = hierarchical_evolutionary_optimization(max_time_seconds=170)
    
    return best_sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")