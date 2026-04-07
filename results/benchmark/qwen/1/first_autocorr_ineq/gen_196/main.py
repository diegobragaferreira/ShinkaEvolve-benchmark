# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
from scipy import optimize
import random
from typing import List, Tuple
import time
import numba
from numba import jit
from joblib import Parallel, delayed
import operator
from collections import OrderedDict
import heapq

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def fast_convolve_jit(a, b):
    """Fast convolution using Numba JIT compilation."""
    n = len(a)
    m = len(b)
    result = np.zeros(n + m - 1)
    for i in range(n):
        for j in range(m):
            result[i + j] += a[i] * b[j]
    return result

class LRUCache:
    """LRU Cache implementation for efficient evaluation caching."""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

_evaluator_cache = LRUCache(capacity=5000)

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Computes the autocorrelation constant C₁ and its reciprocal 1/C₁.
    Uses cached evaluation for efficiency.
    """
    seq_tuple = tuple(sequence)
    cached = _evaluator_cache.get(seq_tuple)
    if cached is not None:
        return cached

    if not sequence or sum(sequence) < 0.01:
        result = (float('inf'), 0.0)
        _evaluator_cache.put(seq_tuple, result)
        return result

    n = len(sequence)

    # Use FFT-based convolution for efficiency O(n log n)
    if n > 500:
        try:
            conv = fftconvolve(sequence, sequence, mode='full')
        except Exception:
            # Fallback to JIT for large sequences if FFT fails
            conv = fast_convolve_jit(sequence, sequence)
    else:
        conv = fast_convolve_jit(sequence, sequence)
    
    # Take the maximum of the convolution (excluding the zero-padding)
    max_conv = np.max(conv[n-1:])
    sum_seq = sum(sequence)

    if sum_seq == 0:
        result = (float('inf'), 0.0)
        _evaluator_cache.put(seq_tuple, result)
        return result

    c1 = 2 * n * max_conv / (sum_seq ** 2)
    inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

    result = (c1, inv_c1)
    _evaluator_cache.put(seq_tuple, result)
    return result

def generate_step_function(n: int, num_steps: int = None) -> List[float]:
    """Generate a step function with randomly placed steps."""
    if num_steps is None:
        num_steps = max(2, min(20, n // 10))
    step_positions = sorted(random.sample(range(n), num_steps))
    step_heights = [random.uniform(0.1, 100.0) for _ in range(num_steps)]

    sequence = [0.0] * n
    for i, (pos, height) in enumerate(zip(step_positions, step_heights)):
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = n
        sequence[pos:end_pos] = [height] * (end_pos - pos)
    return sequence

def generate_gaussian_distribution(n: int) -> List[float]:
    """Generate a Gaussian-like distribution."""
    sequence = [random.gauss(50.0, 20.0) for _ in range(n)]
    return [max(0.01, x) for x in sequence]

def generate_uniform_distribution(n: int) -> List[float]:
    """Generate a uniform distribution."""
    return [random.uniform(0.1, 100.0) for _ in range(n)]

def generate_pattern_based_sequence(n: int) -> List[float]:
    """Generate a sequence with a custom pattern to encourage structure."""
    # Create a combination of peaks and valleys
    sequence = [0.0] * n
    num_peaks = max(2, min(10, n // 50))
    peak_positions = sorted(random.sample(range(n), num_peaks))
    
    # Assign increasing heights to peaks
    peak_heights = [random.uniform(10, 100) for _ in range(num_peaks)]
    for i, (pos, height) in enumerate(zip(peak_positions, peak_heights)):
        # Set the area around the peak to the height
        radius = max(1, n // 20)
        start = max(0, pos - radius)
        end = min(n, pos + radius)
        for j in range(start, end):
            sequence[j] = height
    return sequence

def generate_from_known_patterns(n: int) -> List[float]:
    """Generate sequences based on known effective patterns for autocorrelation."""
    patterns = [
        lambda x: [1000 * np.exp(-i/10) for i in range(x)],
        lambda x: [1.0] * x,
        lambda x: generate_step_function(x, 5),
        lambda x: [1000 * np.exp(-i/20) for i in range(x)]  # Exponential decay
    ]
    pattern_func = random.choice(patterns)
    sequence = pattern_func(n)
    # Normalize to reasonable mass
    total_mass = sum(sequence)
    if total_mass > 0:
        sequence = [x / total_mass * 100 for x in sequence]
    return sequence

def generate_diverse_population(population_size: int, length_range=(50, 500)) -> List[List[float]]:
    """Generate a diverse initial population."""
    population = []
    
    for _ in range(population_size):
        n = random.randint(*length_range)
        method = random.choice(['step', 'gaussian', 'uniform', 'pattern', 'known_pattern'])
        if method == 'step':
            sequence = generate_step_function(n)
        elif method == 'gaussian':
            sequence = generate_gaussian_distribution(n)
        elif method == 'uniform':
            sequence = generate_uniform_distribution(n)
        elif method == 'pattern':
            sequence = generate_pattern_based_sequence(n)
        else:  # known_pattern
            sequence = generate_from_known_patterns(n)
        population.append(sequence)
    return population

def mutate_sequence(sequence: List[float], mutation_rate=0.1, mutation_strength=0.3) -> List[float]:
    """Apply mutation to a sequence with specified rate and strength."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply Gaussian noise scaled by mutation strength
            noise = random.gauss(0, mutation_strength * mutated[i])
            mutated[i] = max(0.01, mutated[i] + noise)
    return mutated

def adaptive_mutate_sequence(sequence: List[float], generation: int, max_generations: int,
                           base_mutation_rate: float = 0.15, mutation_strength: float = 0.2) -> List[float]:
    """Apply adaptive mutation with decreasing rate over generations."""
    # Decrease mutation rate as generations progress
    adaptive_rate = base_mutation_rate * (1 - generation / max_generations)
    return mutate_sequence(sequence, adaptive_rate, mutation_strength)

def crossover_sequences(seq1: List[float], seq2: List[float]) -> List[float]:
    """Perform crossover between two sequences with pattern preservation."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if seq1 else seq2

    # Single-point crossover with a probabilistic bias towards better parts
    crossover_point = random.randint(1, min_len - 1)
    # Bias towards choosing the better parent's segments
    if random.random() < 0.7:
        # 70% chance to preserve better parts
        if sum(seq1) > sum(seq2):
            child = seq1[:crossover_point] + seq2[crossover_point:]
        else:
            child = seq2[:crossover_point] + seq1[crossover_point:]
    else:
        child = seq1[:crossover_point] + seq2[crossover_point:]

    # Ensure minimum positive value for all elements
    child = [max(0.01, x) for x in child]
    return child

def compute_gradient(sequence: List[float], epsilon: float = 1e-6) -> List[float]:
    """Compute gradient for a sequence."""
    n = len(sequence)
    base_inv_c1 = compute_autocorrelation_constant(sequence)[1]
    grad = np.zeros(n)
    
    for i in range(n):
        perturbed = sequence[:]
        perturbed[i] += epsilon
        perturbed = np.maximum(perturbed, 0.01)  # Keep non-negative
        perturbed_inv_c1 = compute_autocorrelation_constant(perturbed)[1]
        grad[i] = (perturbed_inv_c1 - base_inv_c1) / epsilon
    
    return grad

def simulated_annealing_perturbation(sequence: List[float], temperature: float) -> List[float]:
    """Apply simulated annealing-style perturbations."""
    perturbed = sequence.copy()
    n = len(perturbed)
    # Apply random perturbations with probability proportional to temperature
    for i in range(n):
        if random.random() < temperature:
            factor = random.uniform(0.9, 1.1)
            perturbed[i] = max(0.01, perturbed[i] * factor)
    return perturbed

def multi_stage_local_refinement(sequence: List[float], iterations: int = 10) -> List[float]:
    """Multi-stage refinement combining gradient, adaptive steps, and simulated annealing."""
    refined = np.array(sequence, dtype=float)
    
    # Stage 1: Gradient ascent
    for _ in range(iterations // 3):
        grad = compute_gradient(refined.tolist())
        refined += 0.01 * grad
        refined = np.maximum(refined, 0.01)
    
    # Stage 2: Simulated Annealing
    temp = 1.0
    for _ in range(iterations // 3):
        perturbed = simulated_annealing_perturbation(refined.tolist(), temp)
        _, curr_fitness = compute_autocorrelation_constant(refined.tolist())
        _, perturbed_fitness = compute_autocorrelation_constant(perturbed)
        if perturbed_fitness > curr_fitness:
            refined = np.array(perturbed)
        temp *= 0.95  # Cool down
    
    # Stage 3: Fine-tuning with gradient
    for _ in range(iterations // 3):
        grad = compute_gradient(refined.tolist())
        refined += 0.005 * grad
        refined = np.maximum(refined, 0.01)
        
    return refined.tolist()

def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    try:
        n = len(f_sequence)
        c = -np.ones(n)
        a_ub = []
        b_ub = []
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

        result = optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, method='highs')

        if result.success:
            g_sequence = result.x
            return g_sequence
        else:
            return None
    except Exception:
        return None

def get_good_direction_to_move_into(sequence: List[float]) -> List[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    if sum_sequence < 1e-10:
        return None
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    try:
        conv_result = np.convolve(normalized_sequence, normalized_sequence)
        rhs = np.max(conv_result)
    except Exception:
        return None
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_g_fun = np.sum(g_fun)
    if sum_g_fun < 1e-10:
        return None
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g_fun for x in g_fun]
    t = 0.01
    new_sequence = [
        (1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)
    ]
    return new_sequence

def evaluate_fitness_single(sequence: List[float]) -> float:
    """Evaluate fitness for a single sequence."""
    _, inv_c1 = compute_autocorrelation_constant(sequence)
    return inv_c1

def evaluate_fitness_batch_parallel(sequences: List[List[float]]) -> List[float]:
    """Evaluate multiple sequences in parallel for better performance."""
    fitness_scores = Parallel(n_jobs=-1)(delayed(evaluate_fitness_single)(seq) for seq in sequences)
    return fitness_scores

def adaptive_selection(population: List[List[float]], fitness_scores: List[float], 
                      generation: int, max_generations: int) -> List[List[float]]:
    """Adaptively select parents with varying tournament sizes based on diversity and generation."""
    if len(fitness_scores) < 2:
        return population[:1]
    
    diversity = np.std(fitness_scores)
    fitness_mean = np.mean(fitness_scores)
    
    # Adjust tournament size based on diversity and generation
    max_tournsize = 10
    min_tournsize = 3
    
    # Dynamically scale based on generation progress
    progress_factor = generation / max_generations
    tournsize_factor = 1.0 - 0.5 * progress_factor  # Reduce as generation progresses
    
    if diversity < 0.05:  # Low diversity - more selective
        tournsize = max_tournsize * tournsize_factor
    elif diversity > 0.2:  # High diversity - less selective
        tournsize = min_tournsize * tournsize_factor
    else:  # Medium diversity
        tournsize = min_tournsize + (max_tournsize - min_tournsize) * (0.5 - diversity) * tournsize_factor

    tournsize = max(min_tournsize, min(max_tournsize, int(tournsize)))
    
    selected = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), min(tournsize, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx].copy())
    return selected

def adaptive_convergence_check(fitness_history: List[float], patience: int = 20) -> bool:
    """Check for convergence based on recent fitness history."""
    if len(fitness_history) < patience:
        return False
    
    recent = fitness_history[-patience:]
    if np.std(recent) < 1e-6:
        return True
    return False

def optimize_step_function_evolutionary(max_time_seconds=170) -> List[float]:
    """
    Evolutionary optimization to find optimal step function that maximizes 1/C1.
    """
    start_time = time.time()
    _evaluator_cache.clear()

    # Initialize population with diverse strategies
    population_size = 60  # Slightly increased for better exploration
    population = generate_diverse_population(population_size, (100, 1000))

    best_sequence = None
    best_inv_c1 = 0.0
    fitness_history = []
    
    generation = 0
    stagnation_count = 0
    max_stagnation = 50

    while time.time() - start_time < max_time_seconds and stagnation_count < max_stagnation:
        generation += 1

        # Evaluate fitness (1/C1) of each individual in batches
        fitness_scores = evaluate_fitness_batch_parallel(population)

        # Track best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_inv_c1 = fitness_scores[current_best_idx]

        if current_best_inv_c1 > best_inv_c1:
            best_inv_c1 = current_best_inv_c1
            best_sequence = population[current_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Record history for convergence check
        fitness_history.append(current_best_inv_c1)
        
        # Check for convergence
        if adaptive_convergence_check(fitness_history):
            break

        # Try to improve best sequence using LP-based direction
        if best_sequence is not None:
            improved = get_good_direction_to_move_into(best_sequence)
            if improved is not None:
                _, improved_inv_c1 = compute_autocorrelation_constant(improved)
                if improved_inv_c1 > best_inv_c1:
                    best_sequence = improved
                    best_inv_c1 = improved_inv_c1

        # Apply multi-stage local refinement to best sequence
        if best_sequence is not None:
            refined = multi_stage_local_refinement(best_sequence, iterations=8)
            _, refined_inv_c1 = compute_autocorrelation_constant(refined)
            if refined_inv_c1 > best_inv_c1:
                best_sequence = refined
                best_inv_c1 = refined_inv_c1

        # Selection with adaptive tournament selection and elitism
        selected_parents = adaptive_selection(population, fitness_scores, generation, 100)

        # Create new population through crossover and mutation
        new_population = [best_sequence.copy()]  # Elitism: keep best individual

        while len(new_population) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Crossover
            child = crossover_sequences(parent1, parent2)

            # Mutation with adaptive rate based on generation
            child = adaptive_mutate_sequence(child, generation, 100, 0.15, 0.2)

            new_population.append(child)

        population = new_population[:population_size]

    # Final cleanup and validation
    if best_sequence is not None:
        # Normalize sequence to make sure it's valid
        sum_seq = sum(best_sequence)
        if sum_seq > 0.01:
            best_sequence = [x / sum_seq * 100 for x in best_sequence]

    return best_sequence if best_sequence else generate_step_function(100)

def search_for_best_sequence() -> List[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Use evolutionary optimization approach with diversity management
        best_sequence = optimize_step_function_evolutionary()
        return best_sequence
    except Exception as e:
        print(f"Optimization failed with error: {e}")
        # Fallback to simple random approach
        return generate_step_function(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")