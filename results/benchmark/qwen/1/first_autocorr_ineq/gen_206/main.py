# EVOLVE-BLOCK-START

import numpy as np
from scipy.fft import fft, ifft
from scipy import optimize
from typing import List, Tuple
import random
import time
from joblib import Parallel, delayed

# Set fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class FastAutocorrelationEvaluator:
    """Efficient evaluator for autocorrelation constants using FFT with caching."""
    
    def __init__(self):
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(self, sequence: List[float]) -> Tuple[float, float]:
        """
        Computes the autocorrelation constant C1 and its reciprocal 1/C1.
        Uses caching and FFT convolution for efficiency.
        """
        # Create a hashable representation for caching
        seq_tuple = tuple(sequence)
        if seq_tuple in self._cache:
            self._cache_hits += 1
            return self._cache[seq_tuple]

        self._cache_misses += 1

        if not sequence or sum(sequence) < 0.01:
            result = (float('inf'), 0.0)
            self._cache[seq_tuple] = result
            return result

        n = len(sequence)
        
        # Use FFT-based convolution for efficiency O(n log n)
        if n > 500:
            try:
                # Pad to length 2*n - 1 for full convolution
                padded_length = 2 * n - 1
                fa = fft(sequence, padded_length)
                fb = fft(sequence, padded_length)
                conv = ifft(fa * fb.conj()).real[:n]
            except Exception:
                # Fallback to manual convolution if FFT fails
                conv = np.convolve(sequence, sequence, mode='full')[:n]
        else:
            conv = np.convolve(sequence, sequence, mode='full')[:n]
            
        max_conv = np.max(conv)
        sum_seq = sum(sequence)

        if sum_seq == 0:
            result = (float('inf'), 0.0)
            self._cache[seq_tuple] = result
            return result

        c1 = 2 * n * max_conv / (sum_seq ** 2)
        inv_c1 = 1.0 / c1 if c1 > 0 else 0.0

        result = (c1, inv_c1)
        self._cache[seq_tuple] = result
        return result

# Global evaluator instance
_evaluator = FastAutocorrelationEvaluator()

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """Compute C1 and 1/C1 using global cached evaluator."""
    return _evaluator.evaluate(sequence)

def convolve_fft(a: List[float], b: List[float]) -> List[float]:
    """Compute convolution using FFT for better performance."""
    n = len(a)
    if n == 0:
        return []

    # Pad to length 2*n - 1 for full convolution
    padded_length = 2 * n - 1
    fa = fft(a, padded_length)
    fb = fft(b, padded_length)
    result = ifft(fa * fb.conj()).real
    # Return only the valid convolution part
    return result[:n].tolist()

def evaluate_fitness(sequence: List[float]) -> float:
    """Evaluate fitness as inverse of C1 (higher is better)."""
    c1, inv_c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf') or c1 <= 0:
        return 0.0
    return inv_c1

def generate_structured_sequence(n: int) -> List[float]:
    """Generate a structured sequence with golden-ratio step positions."""
    # Use golden ratio based spacing for step positions to avoid periodicity
    golden_ratio = (1 + np.sqrt(5)) / 2
    # Number of steps based on sequence length
    num_steps = max(3, min(30, n // 10))

    # Golden ratio based step positions
    step_positions = []
    for i in range(num_steps):
        pos = int((i * golden_ratio) % n)
        step_positions.append(pos)

    # Ensure unique positions and sort them
    step_positions = sorted(set(step_positions))
    while len(step_positions) < num_steps:
        # Add random positions if we missed any
        new_pos = random.randint(0, n-1)
        if new_pos not in step_positions:
            step_positions.append(new_pos)
    step_positions = sorted(step_positions[:num_steps])

    # Generate step heights with adaptive geometric decay
    heights = []
    base_height = 100.0
    decay_factor = 0.85

    for i in range(len(step_positions)):
        # Apply geometric decay with some variance
        height = base_height * (decay_factor ** i)
        # Add controlled randomness to avoid perfect symmetry
        variance = random.uniform(0.8, 1.2)
        heights.append(height * variance)

    # Create final sequence
    sequence = [0.0] * n
    for i, (pos, height) in enumerate(zip(step_positions, heights)):
        # Determine end position
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = n

        # Ensure proper bounds
        pos = max(0, min(n-1, pos))
        end_pos = max(pos+1, min(n, end_pos))

        # Set step values
        if end_pos > pos:
            sequence[pos:end_pos] = [height] * (end_pos - pos)

    return sequence

def generate_step_based_sequence(length: int) -> List[float]:
    """Generate a step-based sequence pattern."""
    # Create a few prominent steps
    sequence = [0.0] * length
    num_steps = max(2, min(20, length // 10))
    step_positions = sorted(random.sample(range(length), num_steps))

    for i, pos in enumerate(step_positions):
        if i < len(step_positions) - 1:
            end_pos = step_positions[i+1]
        else:
            end_pos = length
        height = random.uniform(10.0, 100.0)
        sequence[pos:end_pos] = [height] * (end_pos - pos)

    return sequence

def generate_diverse_population(population_size: int, length_range=(100, 1000)) -> List[List[float]]:
    """Generate a diverse initial population."""
    population = []

    # Generate sequences using different methods
    for _ in range(population_size // 3):
        n = random.randint(*length_range)
        population.append(generate_structured_sequence(n))

    for _ in range(population_size // 3):
        n = random.randint(*length_range)
        population.append(generate_step_based_sequence(n))

    # Fill remaining with standard random sequences
    while len(population) < population_size:
        n = random.randint(*length_range)
        sequence = [random.uniform(0.1, 100.0) for _ in range(n)]
        population.append(sequence)

    return population

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Apply mutation to sequence with multiplicative Gaussian perturbation."""
    mutated = sequence.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply multiplicative Gaussian perturbation
            perturbation = random.gauss(1, 0.1)
            mutated[i] *= abs(perturbation)  # Ensure non-negative
            mutated[i] = max(0.01, mutated[i])
    return mutated

def crossover_sequences(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform crossover between two sequences."""
    min_len = min(len(parent1), len(parent2))
    crossover_point = random.randint(1, min_len - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

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

def local_refinement(sequence: List[float], max_iter: int = 100) -> List[float]:
    """
    Apply local refinement to improve the sequence using gradient-based optimization.
    """
    # Convert to numpy array for easier manipulation
    x0 = np.array(sequence)
    n = len(x0)

    # Define objective function to minimize (negative of 1/C1)
    def objective(x):
        # Ensure non-negativity and avoid near-zero values
        x = np.maximum(x, 1e-6)
        c1, _ = compute_autocorrelation_constant(x.tolist())
        return -1.0 / c1 if c1 > 0 else 1e6

    # Define bounds
    bounds = [(1e-6, 1000.0) for _ in range(n)]

    # Use L-BFGS-B for local optimization
    try:
        result = optimize.minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': max_iter})
        if result.success:
            refined = np.maximum(result.x, 1e-6).tolist()
            return refined
    except Exception as e:
        print(f"Local refinement error: {e}")

    return sequence

def adaptive_tournament_selection(population: List[List[float]],
                                fitness_scores: List[float],
                                generation: int,
                                population_size: int) -> List[float]:
    """Perform adaptive tournament selection based on diversity and generation."""

    # Determine tournament size based on generation and population diversity
    if generation <= 20:  # Early generations
        tournament_size = min(9, max(5, population_size // 4))
    elif generation >= 50:  # Later generations
        tournament_size = min(4, max(3, population_size // 8))
    else:  # Middle generations
        tournament_size = 5

    # Perform tournament selection
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]

    return population[winner_idx].copy()

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence.
    Uses evolutionary optimization with adaptive selection and local search.
    """
    start_time = time.time()
    max_time = 175  # Leave some time for cleanup

    # Configuration
    population_size = 50
    generations = 100
    max_stagnation = 20
    elite_size = 5

    # Initialize population with diverse sequences
    population = generate_diverse_population(population_size)

    best_solution = None
    best_fitness = 0.0
    stagnation_counter = 0
    fitness_history = []

    for generation in range(generations):
        # Check time limit
        if time.time() - start_time > max_time:
            break

        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual.copy()

        fitness_history.append(best_fitness)

        # Check for stagnation using multi-generational trend analysis
        if len(fitness_history) > 10:
            recent_improvement = np.mean(fitness_history[-10:]) - np.mean(fitness_history[:-10])
            if recent_improvement < 0.0001:
                stagnation_counter += 1
                if stagnation_counter >= max_stagnation:
                    # Reset with new diverse population
                    population = generate_diverse_population(population_size)
                    stagnation_counter = 0
        else:
            stagnation_counter = 0

        # Local refinement for top performers
        top_performers = sorted(range(len(population)),
                               key=lambda i: fitness_scores[i], reverse=True)[:5]
        for idx in top_performers:
            refined = local_refinement(population[idx])
            c1, inv_c1 = compute_autocorrelation_constant(refined)
            if inv_c1 > fitness_scores[idx]:
                population[idx] = refined
                if inv_c1 > best_fitness:
                    best_fitness = inv_c1
                    best_solution = refined

        # Selection and reproduction
        selected_parents = []

        # Fitness proportionate selection with some randomness
        total_fitness = sum(fitness_scores)
        if total_fitness > 0:
            probabilities = [f / total_fitness for f in fitness_scores]
            selected_parents.extend(random.choices(population, probabilities, k=population_size // 2))
        else:
            # Fallback to tournament selection
            for _ in range(population_size // 2):
                selected_parents.append(adaptive_tournament_selection(population, fitness_scores, generation, population_size))

        # Elitism: keep the best individual
        selected_parents.insert(0, best_solution.copy() if best_solution is not None else generate_structured_sequence(100))

        # Create new population
        new_population = [best_solution.copy() if best_solution is not None else generate_structured_sequence(100)]

        # Crossover and mutation
        while len(new_population) < population_size:
            parent1 = random.choice(selected_parents)
            parent2 = random.choice(selected_parents)

            # Simple uniform crossover
            child = []
            for i in range(min(len(parent1), len(parent2))):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])

            # Mutation with adaptive rate
            mutation_rate = max(0.05, 0.3 * (1 - generation / 100))
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    child[i] = max(0.01, child[i] + random.gauss(0, 1.0))

            new_population.append(child)

        population = new_population[:population_size]

        # Debug output every few generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best inv_C1 = {best_fitness:.4f}")

    # Final refinement of the best sequence
    if best_solution is not None:
        refined = local_refinement(best_solution)
        c1, inv_c1 = compute_autocorrelation_constant(refined)
        if inv_c1 > best_fitness:
            best_solution = refined

    return best_solution if best_solution is not None else generate_structured_sequence(100)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")