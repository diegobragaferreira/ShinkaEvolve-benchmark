# EVOLVE-BLOCK-START
import numpy as np
from scipy import optimize
from scipy.optimize import minimize
import random
import time
from collections import deque

def compute_convolution_fft(seq):
    """Compute the autoconvolution using FFT for efficiency."""
    n = len(seq)
    padded_seq = np.pad(seq, (0, n-1), mode='constant')
    conv_result = np.fft.ifft(np.fft.fft(padded_seq) * np.conj(np.fft.fft(padded_seq)))
    return np.real(conv_result[:n])

def calculate_c1(sequence):
    """Calculate the C1 constant for a given sequence."""
    if len(sequence) == 0:
        return float('inf')

    sequence = np.array(sequence)
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return float('inf')

    conv = compute_convolution_fft(sequence)
    max_b = np.max(conv)
    n = len(sequence)

    # Avoid division by zero or very small numbers
    if max_b <= 1e-12:
        return float('inf')

    c1 = (2 * n * max_b) / (sum_a ** 2)
    return c1

def evaluate_fitness(sequence):
    """Evaluate the inverse of C1 as fitness (we want to maximize 1/C1)."""
    c1 = calculate_c1(sequence)
    if c1 == float('inf') or c1 > 10000:
        return 0.0  # Penalty for invalid sequences
    return 1.0 / c1

def build_convolution_constraints(sequence):
    """Build the linear inequality constraints for the convolution."""
    n = len(sequence)
    if n == 0:
        return [], []

    # Build constraint matrix A_ub such that A_ub * x <= b_ub
    # Each row represents a convolution constraint
    a_ub = []
    b_ub = []

    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = sequence[i]
        a_ub.append(row)
        b_ub.append(np.max(compute_convolution_fft(sequence)))  # Upper bound from max convolution

    # Add non-negativity constraints
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    return a_ub, b_ub

def quadratic_optimization_approach(initial_sequence):
    """Attempt to solve using a direct quadratic programming approach."""
    n = len(initial_sequence)
    if n == 0:
        return initial_sequence

    # Objective: minimize (sum(a))^2 / (2 * n * max(conv)) => maximize 1/C1
    # This is not strictly a quadratic problem, but we can approximate and use a QP solver for direction finding.
    
    # We reformulate to be a quadratic program where we minimize a function related to our objective
    # Let’s define a surrogate objective: minimize a quadratic approximation of the inverse C1
    # But since exact solution is complex, we'll do a simpler direct approach
    # Using scipy.optimize.minimize with trust-constr method for constrained problems
    try:
        # Define the objective function to minimize
        def obj_func(x):
            # x is the sequence
            if np.sum(x) < 0.01:
                return 1e10  # Penalize invalid sequences
            c1 = calculate_c1(x)
            if c1 <= 0:
                return 1e10
            return 1.0 / c1  # We want to maximize 1/C1, so minimize -1/C1

        # Define bounds (non-negative)
        bounds = [(0, 1000) for _ in range(n)]

        # Constraints
        # For simplicity in this approach, we focus on basic feasibility rather than detailed constraints
        # This approach uses trust-constr method which handles constraints well
        
        result = minimize(obj_func, initial_sequence, method='trust-constr', bounds=bounds, 
                          options={'maxiter': 1000, 'verbose': 0})
        
        if result.success:
            return result.x.tolist()
        else:
            return initial_sequence
    except Exception as e:
        return initial_sequence

def get_good_direction_to_move_into(
    sequence: list[float],
    iteration: int = 0,
) -> list[float] | None:
    """Returns the direction to move into the sequence using quadratic optimization."""
    return quadratic_optimization_approach(sequence)

def initialize_good_sequence():
    """Initialize sequence with known good patterns."""
    patterns = [
        # Simple uniform pattern
        [1.0] * 100,
        # Alternating pattern
        [1.0, 0.0] * 50,
        # Exponential decay
        [1.0 / (i + 1) for i in range(100)],
        # Gaussian-like decay
        [np.exp(-i**2 / 200.0) for i in range(100)]
    ]

    pattern = random.choice(patterns)
    noise_level = 0.1
    noisy_pattern = [max(0.0, x + random.uniform(-noise_level, noise_level) * x)
                     for x in pattern]
    return noisy_pattern

def evolutionary_mutation(sequence, mutation_rate=0.05):
    """Apply evolutionary mutation to a sequence."""
    new_sequence = sequence.copy()

    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            factor = random.uniform(0.9, 1.1)
            new_sequence[i] = max(0.0, new_sequence[i] * factor)

    if random.random() < 0.1 and len(new_sequence) > 10:
        idx = random.randint(0, len(new_sequence) - 1)
        new_sequence[idx] = max(0.0, new_sequence[idx] - random.uniform(0, 10))

    return new_sequence

def adaptive_evolutionary_search():
    """Adaptive evolutionary search with dynamic parameters and diversity management."""
    population_size = 20
    max_iterations = 2000
    base_mutation_rate = 0.05
    elite_size = 10
    
    initial_population = [initialize_good_sequence() for _ in range(population_size)]
    fitness_scores = [evaluate_fitness(seq) for seq in initial_population]

    best_sequence = initial_population[np.argmax(fitness_scores)]
    best_fitness = max(fitness_scores)

    elite_sequences = [best_sequence]
    elite_fitnesses = [best_fitness]

    diversity_queue = deque(maxlen=5)
    diversity_queue.append(best_sequence)

    start_time = time.time()
    timeout_seconds = 180

    for iteration in range(max_iterations):
        if time.time() - start_time > timeout_seconds - 2:
            break

        current_mutation_rate = base_mutation_rate * (1 - iteration / max_iterations)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_half = [initial_population[i] for i in sorted_indices[:population_size // 2]]

        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(top_half)
            parent2 = random.choice(top_half)

            crossover_point = min(len(parent1), len(parent2)) // 2
            child = parent1[:crossover_point] + parent2[crossover_point:]

            mutated_child = evolutionary_mutation(child, current_mutation_rate)
            new_population.append(mutated_child)

        fitness_scores = [evaluate_fitness(seq) for seq in new_population]

        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness:
            best_fitness = fitness_scores[current_best_idx]
            best_sequence = new_population[current_best_idx]

            elite_sequences.append(best_sequence)
            elite_fitnesses.append(best_fitness)

            if len(elite_sequences) > elite_size:
                top_elite_indices = np.argsort(elite_fitnesses)[-elite_size:]
                elite_sequences = [elite_sequences[i] for i in top_elite_indices]
                elite_fitnesses = [elite_fitnesses[i] for i in top_elite_indices]

            diversity_queue.append(best_sequence)

        if len(diversity_queue) == 5:
            recent_solutions = list(diversity_queue)
            if all(np.allclose(s, recent_solutions[0], rtol=1e-3) for s in recent_solutions):
                new_pop = []
                for _ in range(population_size):
                    new_pop.append(initialize_good_sequence())
                new_population = new_pop
                fitness_scores = [evaluate_fitness(seq) for seq in new_population]
                current_best_idx = np.argmax(fitness_scores)
                if fitness_scores[current_best_idx] > best_fitness:
                    best_fitness = fitness_scores[current_best_idx]
                    best_sequence = new_population[current_best_idx]

        if iteration % 5 == 0:
            refined_population = []
            for seq in new_population:
                h_function = get_good_direction_to_move_into(seq, iteration)
                if h_function is not None and evaluate_fitness(h_function) > evaluate_fitness(seq):
                    refined_population.append(h_function)
                else:
                    refined_population.append(seq)
            new_population = refined_population
            fitness_scores = [evaluate_fitness(seq) for seq in new_population]

    return best_sequence

def search_for_best_sequence() -> list[float]:
    """Function to search for the best coefficient sequence."""
    return adaptive_evolutionary_search()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")